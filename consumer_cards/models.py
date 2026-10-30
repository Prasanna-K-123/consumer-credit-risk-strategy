"""Original fitted model choices with separate calibration and policy rows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .constants import CATEGORICAL, CONTINUOUS, SEED, TARGET
from .contracts import aligned, binary_labels, feature_frame, probabilities
from .data import feature_groups, partition
from .evaluation import select_threshold
from .runtime import bounded_native_threads
from .woe import WOETransformer


def logits(p):
    probability = probabilities(p)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


@bounded_native_threads
def fit_calibration(p, y):
    labels, probability = aligned(y, p, both_classes=True)
    return LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(logits(probability), labels)


@dataclass
class FittedCardModels:
    transform: WOETransformer
    scorecard: LogisticRegression
    challenger: HistGradientBoostingClassifier
    calibrators: dict
    thresholds: dict
    calibration_info: dict
    parts: dict
    groups: np.ndarray

    @bounded_native_threads
    def raw_predictions(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        X = feature_frame(frame)
        return {
            "scorecard": self.scorecard.predict_proba(self.transform.transform(X))[:, 1],
            "challenger": self.challenger.predict_proba(X)[:, 1],
        }

    @bounded_native_threads
    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        raw = self.raw_predictions(frame)
        return {name: calibrator.predict_proba(logits(raw[name]))[:, 1]
                for name, calibrator in self.calibrators.items()}

    @bounded_native_threads
    def scorecard_contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Exact calibrated log-odds decomposition, not legal adverse-action reasons."""
        X = feature_frame(frame)
        transformed = self.transform.transform(X)
        calibrator = self.calibrators["scorecard"]
        slope = float(calibrator.coef_[0, 0])
        terms = transformed * self.scorecard.coef_[0] * slope
        intercept = float(calibrator.intercept_[0] + slope * self.scorecard.intercept_[0])
        reconstructed = terms.sum(axis=1).to_numpy() + intercept
        # The calibration logit uses clipping at boundaries; disclose if active.
        raw = self.raw_predictions(X)["scorecard"]
        clipping = (raw < 1e-6) | (raw > 1 - 1e-6)
        reconstructed[clipping] = np.nan
        terms.columns = ["log_odds_" + name for name in terms.columns]
        terms["calibrated_intercept"] = intercept
        terms["calibrated_log_odds"] = reconstructed
        terms["raw_probability_clipped"] = clipping
        return terms


@bounded_native_threads
def fit_models(frame: pd.DataFrame) -> FittedCardModels:
    X = feature_frame(frame[[*CONTINUOUS, *CATEGORICAL]])
    y = binary_labels(frame[TARGET].to_numpy(), both_classes=True)
    groups = feature_groups(X)
    parts = partition(groups)
    train, cal, val = [parts[name] for name in ("train", "calibration", "policy_validation")]
    transform = WOETransformer(list(CONTINUOUS), list(CATEGORICAL), n_bins=5)
    encoded = transform.fit_transform(X.iloc[train], pd.Series(y[train], index=X.iloc[train].index))
    scorecard = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED).fit(encoded, y[train])
    challenger = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.05,
                                               l2_regularization=10.0, early_stopping=False, random_state=SEED).fit(X.iloc[train], y[train])
    fitted = FittedCardModels(transform, scorecard, challenger, {}, {}, {}, parts, groups)
    raw_cal = fitted.raw_predictions(X.iloc[cal])
    raw_val = fitted.raw_predictions(X.iloc[val])
    for name in ("scorecard", "challenger"):
        calibrator = fit_calibration(raw_cal[name], y[cal])
        pval = calibrator.predict_proba(logits(raw_val[name]))[:, 1]
        threshold, val_cost = select_threshold(y[val], pval)
        fitted.calibrators[name] = calibrator
        fitted.thresholds[name] = threshold
        fitted.calibration_info[name] = {"intercept": float(calibrator.intercept_[0]), "slope": float(calibrator.coef_[0, 0]), "validation_cost": val_cost}
    return fitted
