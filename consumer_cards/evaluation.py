"""Frozen cost accounting, paired uncertainty and retrospective review queues."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .constants import SEED
from .contracts import aligned, fraction, probabilities, source_ids
from .runtime import bounded_native_threads


def cost(y, p, threshold):
    labels, probability = aligned(y, p)
    threshold = fraction(threshold, "Threshold")
    flagged = probability >= threshold
    return 5 * ((labels == 1) & ~flagged).astype(int) + ((labels == 0) & flagged).astype(int)


def select_threshold(y, p):
    labels, probability = aligned(y, p)
    grid = np.arange(1, 100) / 100
    totals = [int(cost(labels, probability, t).sum()) for t in grid]
    index = int(np.argmin(totals))
    return float(grid[index]), int(totals[index])


def metrics(y, p, threshold):
    labels, probability = aligned(y, p, both_classes=True)
    threshold = fraction(threshold, "Threshold")
    flagged = probability >= threshold
    return {
        "auc": float(roc_auc_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability)),
        "threshold": threshold,
        "false_negative": int(((labels == 1) & ~flagged).sum()),
        "false_positive": int(((labels == 0) & flagged).sum()),
        "accepted": int((~flagged).sum()),
        "cost_units": int(cost(labels, probability, threshold).sum()),
    }


@bounded_native_threads
def paired_group_bootstrap(y, a, b, ta, tb, groups, n=2000):
    labels, first = aligned(y, a, both_classes=True)
    _, second = aligned(labels, b, both_classes=True)
    ta, tb = fraction(ta, "Scorecard threshold"), fraction(tb, "Challenger threshold")
    groups = np.asarray(groups)
    if groups.shape != labels.shape or not all(isinstance(v, (str, int, np.integer)) for v in groups):
        raise ValueError("Bootstrap group labels must be an aligned integer/string vector")
    groups = groups.astype(str)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("Bootstrap repetitions must be an integer >= 2")
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(n):
        index = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        yi = labels[index]
        if len(np.unique(yi)) != 2:
            raise ValueError("One-class bootstrap draw; uncertainty audit required")
        scores.append([
            roc_auc_score(yi, second[index]) - roc_auc_score(yi, first[index]),
            np.mean((second[index] - yi) ** 2 - (first[index] - yi) ** 2),
            float(np.mean(cost(yi, second[index], tb) - cost(yi, first[index], ta))),
        ])
    point = [
        float(roc_auc_score(labels, second) - roc_auc_score(labels, first)),
        float(np.mean((second - labels) ** 2 - (first - labels) ** 2)),
        float(np.mean(cost(labels, second, tb) - cost(labels, first, ta))),
    ]
    interval = np.quantile(np.asarray(scores), [0.025, 0.975], axis=0)
    return {
        name: {"challenger_minus_scorecard": point[i], "ci95": interval[:, i].tolist()}
        for i, name in enumerate(["auc", "brier", "cost_per_client"])
    }


def decision_queue(y, p, ids, capacity_fraction):
    """Flag exactly floor(capacity*n); tied scores use ascending original ID.

    This is a retrospective ranking scenario, not a causal intervention estimate.
    Labels are used for accounting only and never enter rank or capacity choice.
    """
    labels, probability = aligned(y, p)
    ids = source_ids(ids, len(labels))
    capacity = fraction(capacity_fraction, "Review capacity")
    count = int(np.floor(capacity * len(labels)))
    order = np.lexsort((ids, -probability))
    flagged = np.zeros(len(labels), dtype=bool)
    flagged[order[:count]] = True
    cutoff = float(probability[order[count - 1]]) if count else None
    defaults_flagged = int(((labels == 1) & flagged).sum())
    nondefaults_flagged = int(((labels == 0) & flagged).sum())
    defaults_unflagged = int(((labels == 1) & ~flagged).sum())
    defaults = int(labels.sum())
    result = {
        "capacity_fraction": capacity, "capacity_count": count, "flagged": int(flagged.sum()),
        "cutoff_probability": cutoff,
        "tied_at_cutoff": int((probability == cutoff).sum()) if cutoff is not None else 0,
        "tie_break": "probability_desc_source_id_asc",
        "defaults_flagged": defaults_flagged, "nondefaults_flagged": nondefaults_flagged,
        "defaults_unflagged": defaults_unflagged,
        "capture_rate": defaults_flagged / defaults if defaults else None,
        "precision": defaults_flagged / count if count else None,
        "flag_rate": count / len(labels),
        "cost_units": 5 * defaults_unflagged + nondefaults_flagged,
    }
    return result, flagged
