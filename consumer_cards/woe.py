"""Train-only five-quantile WoE, preserving the recorded scientific choices."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .contracts import binary_labels


@dataclass
class WOETransformer:
    continuous_features: list[str]
    categorical_features: list[str]
    n_bins: int = 5
    bin_edges_: dict[str, np.ndarray] = field(default_factory=dict)
    mappings_: dict[str, dict[object, float]] = field(default_factory=dict)
    iv_: dict[str, float] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        labels = binary_labels(np.asarray(y), both_classes=True)
        if len(labels) != len(X) or X.index.has_duplicates:
            raise ValueError("WoE requires aligned rows with unique indices")
        if isinstance(self.n_bins, bool) or not isinstance(self.n_bins, int) or self.n_bins < 2:
            raise ValueError("At least two quantile bins are required")
        y = pd.Series(labels, index=X.index)
        total_events = float(y.sum())
        total_non_events = float((1 - y).sum())
        self.bin_edges_, self.mappings_, self.iv_ = {}, {}, {}
        for col in self.continuous_features:
            values = pd.to_numeric(X[col], errors="coerce")
            quantiles = np.unique(np.nanquantile(values, np.linspace(0, 1, self.n_bins + 1)))
            if len(quantiles) < 3:
                quantiles = np.array([-np.inf, np.inf], dtype=float)
            else:
                quantiles[0], quantiles[-1] = -np.inf, np.inf
            self.bin_edges_[col] = quantiles
            groups = pd.cut(values, bins=quantiles, include_lowest=True, duplicates="drop")
            self._fit_mapping(col, groups.astype(str), y, total_events, total_non_events)
        for col in self.categorical_features:
            groups = X[col].astype("object").where(X[col].notna(), "__MISSING__").astype(str)
            self._fit_mapping(col, groups, y, total_events, total_non_events)
        return self

    def _fit_mapping(self, col, groups, y, total_events, total_non_events):
        grouped = pd.DataFrame({"group": groups, "y": y}).groupby("group", dropna=False)["y"].agg(["sum", "count"])
        grouped["non_event"] = grouped["count"] - grouped["sum"]
        mapping, iv, smoothing = {}, 0.0, 0.5
        count = max(len(grouped), 1)
        event_denom = total_events + smoothing * count
        non_event_denom = total_non_events + smoothing * count
        for group, row in grouped.iterrows():
            event_dist = (float(row["sum"]) + smoothing) / event_denom
            non_event_dist = (float(row["non_event"]) + smoothing) / non_event_denom
            value = float(np.log(event_dist / non_event_dist))
            mapping[group] = value
            iv += (event_dist - non_event_dist) * value
        self.mappings_[col], self.iv_[col] = mapping, float(iv)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.mappings_:
            raise ValueError("WOETransformer is not fitted")
        out = pd.DataFrame(index=X.index)
        for col in self.continuous_features:
            values = pd.to_numeric(X[col], errors="coerce")
            groups = pd.cut(values, bins=self.bin_edges_[col], include_lowest=True, duplicates="drop").astype(str)
            out[col] = groups.map(self.mappings_[col]).fillna(0.0).astype(float)
        for col in self.categorical_features:
            groups = X[col].astype("object").where(X[col].notna(), "__MISSING__").astype(str)
            out[col] = groups.map(self.mappings_[col]).fillna(0.0).astype(float)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
