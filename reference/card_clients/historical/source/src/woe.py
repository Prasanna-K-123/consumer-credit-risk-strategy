from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class WOETransformer:
    continuous_features: list[str]
    categorical_features: list[str]
    n_bins: int = 5
    bin_edges_: dict[str, np.ndarray] = field(default_factory=dict)
    mappings_: dict[str, dict[object, float]] = field(default_factory=dict)
    iv_: dict[str, float] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        y = pd.Series(y, index=X.index).astype(int)
        total_events = float(y.sum())
        total_non_events = float((1 - y).sum())

        for col in self.continuous_features:
            values = pd.to_numeric(X[col], errors="coerce")
            quantiles = np.unique(
                np.nanquantile(values, np.linspace(0, 1, self.n_bins + 1))
            )
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

    def _fit_mapping(
        self,
        col: str,
        groups: pd.Series,
        y: pd.Series,
        total_events: float,
        total_non_events: float,
    ) -> None:
        tmp = pd.DataFrame({"group": groups, "y": y})
        agg = tmp.groupby("group", dropna=False)["y"].agg(["sum", "count"])
        agg["non_event"] = agg["count"] - agg["sum"]

        mapping: dict[object, float] = {}
        iv = 0.0
        smoothing = 0.5
        n_groups = max(len(agg), 1)
        event_denom = total_events + smoothing * n_groups
        non_event_denom = total_non_events + smoothing * n_groups

        for group, row in agg.iterrows():
            events = float(row["sum"])
            non_events = float(row["non_event"])
            event_dist = (events + smoothing) / event_denom
            non_event_dist = (non_events + smoothing) / non_event_denom
            woe = float(np.log(event_dist / non_event_dist))
            mapping[group] = woe
            iv += (event_dist - non_event_dist) * woe

        self.mappings_[col] = mapping
        self.iv_[col] = float(iv)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.mappings_:
            raise RuntimeError("WOETransformer is not fitted.")

        out = pd.DataFrame(index=X.index)
        for col in self.continuous_features:
            values = pd.to_numeric(X[col], errors="coerce")
            groups = pd.cut(
                values,
                bins=self.bin_edges_[col],
                include_lowest=True,
                duplicates="drop",
            ).astype(str)
            out[col] = groups.map(self.mappings_[col]).fillna(0.0).astype(float)

        for col in self.categorical_features:
            groups = X[col].astype("object").where(X[col].notna(), "__MISSING__").astype(str)
            out[col] = groups.map(self.mappings_[col]).fillna(0.0).astype(float)

        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
