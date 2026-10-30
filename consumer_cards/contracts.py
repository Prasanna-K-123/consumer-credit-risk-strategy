"""Validation precedes conversion, clipping and NumPy arithmetic."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import FEATURES


def real_vector(value, name: str, *, nonempty: bool = True) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or (nonempty and not len(raw)):
        raise ValueError(f"{name} must be a nonempty one-dimensional real numeric vector")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite")
    return raw.astype(float, copy=False)


def binary_labels(value, *, both_classes: bool = False) -> np.ndarray:
    raw = real_vector(value, "Labels")
    if not np.isin(raw, [0, 1]).all():
        raise ValueError("Labels must be exactly 0 or 1 before integer conversion")
    if both_classes and len(np.unique(raw)) != 2:
        raise ValueError("Both binary outcome classes are required")
    return raw.astype(np.int64)


def probabilities(value) -> np.ndarray:
    raw = real_vector(value, "Probabilities")
    if ((raw < 0) | (raw > 1)).any():
        raise ValueError("Probabilities must lie in [0, 1]")
    return raw


def aligned(y, p, *, both_classes: bool = False) -> tuple[np.ndarray, np.ndarray]:
    labels, probability = binary_labels(y, both_classes=both_classes), probabilities(p)
    if labels.shape != probability.shape:
        raise ValueError("Labels and probabilities must have identical one-dimensional shapes")
    return labels, probability


def fraction(value, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real scalar")
    result = float(raw)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def source_ids(value, n: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (n,) or raw.dtype.kind not in "iu" or (raw < 0).any():
        raise ValueError("Source IDs must be aligned nonnegative integers")
    if len(np.unique(raw)) != n:
        raise ValueError("Source IDs must be unique for deterministic tie handling")
    return raw


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not len(frame) or frame.columns.duplicated().any():
        raise ValueError("A nonempty feature table with unique columns is required")
    if set(frame.columns) != set(FEATURES):
        missing, extra = set(FEATURES) - set(frame.columns), set(frame.columns) - set(FEATURES)
        raise ValueError(f"Feature schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for name in FEATURES:
        real_vector(frame[name].to_numpy(), f"Feature {name}")
    # Preserve original dtypes and values: duplicate-vector hashes depend on them.
    return frame.loc[:, list(FEATURES)].copy()
