"""Train-only segments and policy-role selection on a known historical cohort.

This extension was designed after the original results were public. Retaining
the original data roles prevents direct fitting on test labels, but does not
create an untouched experiment or erase researcher knowledge of that cohort.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from consumer_cards.constants import FEATURES, TARGET
from consumer_cards.contracts import aligned, binary_labels, feature_frame, fraction, source_ids
from consumer_cards.data import feature_groups
from consumer_cards.evaluation import decision_queue
from consumer_cards.models import fit_calibration, logits
from consumer_cards.runtime import bounded_native_threads

COST_RATIOS = (1, 2, 5, 10, 20)
THRESHOLDS = tuple(np.arange(1, 100) / 100)
CAPACITIES = (0.05, 0.10, 0.20)
ROLES = ("train", "calibration", "policy_validation", "test")
TREE_PARAMETERS = {"criterion": "gini", "max_depth": 3, "min_samples_leaf": 200, "random_state": 20261030}


def cost_ratio(value) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError("Missed-default cost ratio must be a positive real scalar")
    result = float(raw)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Missed-default cost ratio must be finite and positive")
    return result


def confusion(y, p, threshold: float, ratio: float) -> dict:
    labels, probability = aligned(y, p)
    threshold, ratio = fraction(threshold, "Threshold"), cost_ratio(ratio)
    flagged = probability >= threshold
    tp = int(((labels == 1) & flagged).sum())
    fp = int(((labels == 0) & flagged).sum())
    fn = int(((labels == 1) & ~flagged).sum())
    tn = int(((labels == 0) & ~flagged).sum())
    defaults, reviewed = int(labels.sum()), int(flagged.sum())
    return {"n": len(labels), "defaults": defaults, "reviewed": reviewed,
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "review_rate": reviewed / len(labels), "default_capture": tp / defaults if defaults else None,
            "queue_default_rate": tp / reviewed if reviewed else None, "cost_units": ratio * fn + fp}


def select_cost_threshold(y, p, ratio: float) -> tuple[float, float]:
    """Only the supplied policy-role labels determine the chosen threshold.

    Select the lowest threshold among equal minimum costs over 0.01..0.99.
    Review-all/review-none are reported as separate benchmarks, not silently
    substituted for a threshold policy after examining its evaluation results.
    """
    labels, probability = aligned(y, p)
    ratio = cost_ratio(ratio)
    totals = [confusion(labels, probability, threshold, ratio)["cost_units"] for threshold in THRESHOLDS]
    selected = int(np.argmin(totals))
    return float(THRESHOLDS[selected]), float(totals[selected])


def policy_sensitivity(policy_y, policy_p, test_y, test_p, ratios=COST_RATIOS) -> pd.DataFrame:
    """Choose each scenario on policy validation; report known-test outcomes."""
    selected_y, selected_p = aligned(policy_y, policy_p)
    evaluation_y, evaluation_p = aligned(test_y, test_p)
    ratios = tuple(cost_ratio(value) for value in ratios)
    if not ratios or len(set(ratios)) != len(ratios):
        raise ValueError("Cost-ratio scenarios must be nonempty and unique")
    records = []
    for ratio in ratios:
        threshold, selection_cost = select_cost_threshold(selected_y, selected_p, ratio)
        row = confusion(evaluation_y, evaluation_p, threshold, ratio)
        review_none = ratio * int(evaluation_y.sum())
        review_all = int((1 - evaluation_y).sum())
        records.append({"missed_default_cost": ratio, "false_alarm_cost": 1,
                        "selected_threshold": threshold, "policy_validation_cost_units": selection_cost,
                        **row, "review_none_cost_units": review_none, "review_all_cost_units": review_all,
                        "minus_best_constant_cost_units": row["cost_units"] - min(review_none, review_all)})
    return pd.DataFrame(records)


def validate_parts(frame: pd.DataFrame, parts: dict) -> np.ndarray:
    if set(parts) != set(ROLES):
        raise ValueError("Exactly the original four data roles are required")
    X = feature_frame(frame[list(FEATURES)])
    binary_labels(frame[TARGET].to_numpy())
    rows = []
    groups = feature_groups(X)
    seen: set = set()
    for name in ROLES:
        index = np.asarray(parts[name])
        if index.ndim != 1 or not len(index) or index.dtype.kind not in "iu" or (index < 0).any() or (index >= len(frame)).any():
            raise ValueError("Data-role row positions must be nonempty valid integer vectors")
        current = set(groups[index])
        if seen & current:
            raise ValueError("Duplicate model-input group crosses data roles")
        seen.update(current)
        rows.extend(index.tolist())
    if len(rows) != len(frame) or len(set(rows)) != len(frame):
        raise ValueError("Data roles must conserve each row exactly once")
    return groups


@dataclass
class FittedTree:
    model: DecisionTreeClassifier
    calibrator: object
    parts: dict

    @bounded_native_threads
    def raw_predictions(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(feature_frame(frame))[:, 1]

    @bounded_native_threads
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict_proba(logits(self.raw_predictions(frame)))[:, 1]


@bounded_native_threads
def fit_tree(frame: pd.DataFrame, parts: dict) -> FittedTree:
    validate_parts(frame, parts)
    X, y = feature_frame(frame[list(FEATURES)]), binary_labels(frame[TARGET].to_numpy())
    train, calibration = parts["train"], parts["calibration"]
    binary_labels(y[train], both_classes=True)
    binary_labels(y[calibration], both_classes=True)
    model = DecisionTreeClassifier(**TREE_PARAMETERS).fit(X.iloc[train], y[train])
    calibrator = fit_calibration(model.predict_proba(X.iloc[calibration])[:, 1], y[calibration])
    return FittedTree(model, calibrator, {name: np.asarray(parts[name]).copy() for name in ROLES})


def leaf_paths(tree: DecisionTreeClassifier) -> dict[int, str]:
    """Exact numeric CART decision paths; PAY codes are ordered numerically."""
    paths = {}
    def visit(node: int, clauses: list[str]):
        left, right = tree.tree_.children_left[node], tree.tree_.children_right[node]
        if left == right:
            paths[node] = " AND ".join(clauses) if clauses else "all training clients"
            return
        name = FEATURES[tree.tree_.feature[node]]
        threshold = tree.tree_.threshold[node]
        visit(int(left), clauses + [f"{name} <= {threshold:.17g}"])
        visit(int(right), clauses + [f"{name} > {threshold:.17g}"])
    visit(0, [])
    return paths


@bounded_native_threads
def leaf_segments(frame: pd.DataFrame, fitted: FittedTree) -> pd.DataFrame:
    X = feature_frame(frame[list(FEATURES)])
    y = binary_labels(frame[TARGET].to_numpy())
    leaves = fitted.model.apply(X)
    probability, raw = fitted.predict(X), fitted.raw_predictions(X)
    paths, records = leaf_paths(fitted.model), []
    for role in ROLES:
        index = fitted.parts[role]
        for leaf, rule in sorted(paths.items()):
            chosen = index[leaves[index] == leaf]
            count = len(chosen)
            records.append({"partition": role, "leaf_id": leaf, "rule": rule, "n": count,
                            "defaults": int(y[chosen].sum()), "default_rate": float(y[chosen].mean()) if count else None,
                            "raw_probability": float(raw[chosen[0]]) if count else None,
                            "calibrated_probability": float(probability[chosen[0]]) if count else None})
    return pd.DataFrame(records)


def predictive_metrics(y, p) -> dict:
    labels, probability = aligned(y, p, both_classes=True)
    return {"auc": float(roc_auc_score(labels, probability)),
            "average_precision": float(average_precision_score(labels, probability)),
            "brier": float(brier_score_loss(labels, probability)), "log_loss": float(log_loss(labels, probability))}


def capacity_scenarios(y, p, ids) -> pd.DataFrame:
    # Shared queue implementation fixes outcome-blind tie handling by source ID.
    labels, probability = aligned(y, p)
    identifiers = source_ids(ids, len(labels))
    records = []
    for capacity in CAPACITIES:
        row, _ = decision_queue(labels, probability, identifiers, capacity)
        records.append(row)
    return pd.DataFrame(records)
