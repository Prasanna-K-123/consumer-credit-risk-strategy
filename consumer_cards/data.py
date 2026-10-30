"""Official-source acquisition, source audits and historical group partitions."""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .constants import EXCLUDED, FEATURES, HISTORICAL, PROTOCOL_SHA256, SEED, SOURCE_SHA256, SOURCE_URL, TARGET
from .contracts import binary_labels, feature_frame


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(folder: Path | None = None) -> dict:
    path = (Path(folder) if folder is not None else Path(__file__).parent / "resources") / "PROTOCOL.json"
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("Historical protocol hash mismatch")
    protocol = json.loads(path.read_text())
    bindings = {"seed": SEED, "xls_sha256": SOURCE_SHA256, "n": 30000, "excluded_features": ["ID", "SEX", "EDUCATION", "MARRIAGE"]}
    for key, expected in bindings.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Protocol/implementation configuration mismatch: {key}")
    return protocol


def acquire(path: Path) -> dict:
    """Verify before publication; never replace an existing mismatched source."""
    path = Path(path)
    if path.exists():
        if sha256(path) != SOURCE_SHA256:
            raise ValueError("Existing source does not match pinned XLS; retain it separately and use a fresh path")
        return {"path": str(path), "sha256": SOURCE_SHA256, "downloaded": False}
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "consumer-card-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = response.read(30_000_001)
    if len(archive) > 30_000_000:
        raise ValueError("Source archive exceeds the declared size limit")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        candidates = [item for item in zipped.infolist() if item.filename.lower().endswith(".xls")]
        if len(candidates) != 1 or candidates[0].file_size > 20_000_000:
            raise ValueError("Expected exactly one bounded XLS in official source archive")
        raw = zipped.read(candidates[0])
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Downloaded XLS differs from pinned historical source")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download")
    try:
        temporary.write_bytes(raw)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": SOURCE_SHA256, "archive_sha256": hashlib.sha256(archive).hexdigest(), "downloaded": True}


def load_source(path: Path) -> pd.DataFrame:
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("Source XLS hash mismatch")
    frame = pd.read_excel(path, header=1)
    expected = set(FEATURES) | set(EXCLUDED) | {TARGET}
    if frame.shape != (30000, 25) or set(frame.columns) != expected or frame.columns.duplicated().any():
        raise ValueError("Pinned card source shape/schema mismatch")
    if frame.isna().any().any() or not frame.ID.is_unique:
        raise ValueError("Card source contains missing values or repeated source IDs")
    if not np.array_equal(frame.ID.to_numpy(), np.arange(1, 30001)):
        raise ValueError("Pinned card source IDs must be the original ordered 1..30000 keys")
    binary_labels(frame[TARGET].to_numpy(), both_classes=True)
    feature_frame(frame[list(FEATURES)])
    return frame


def feature_groups(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(feature_frame(frame), index=False).astype(str).to_numpy()


def partition(groups) -> dict[str, np.ndarray]:
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups) or not all(isinstance(v, (str, int, np.integer)) for v in groups):
        raise ValueError("Partition groups must be a nonempty integer/string vector")
    if len(np.unique(groups)) < 10:
        raise ValueError("At least ten distinct feature groups are required")
    ids = np.arange(len(groups))
    fit, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED).split(ids, groups=groups))
    rest_dev, rest_test = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED + 1).split(rest, groups=groups[rest]))
    dev, test = rest[rest_dev], rest[rest_test]
    cal_local, policy_local = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED + 2).split(dev, groups=groups[dev]))
    parts = {"train": fit, "calibration": dev[cal_local], "policy_validation": dev[policy_local], "test": test}
    joined = np.concatenate(list(parts.values()))
    if len(joined) != len(groups) or len(np.unique(joined)) != len(groups):
        raise ValueError("Partition rows do not conserve source population")
    seen: set = set()
    for index in parts.values():
        current = set(groups[index])
        if seen & current:
            raise ValueError("Duplicate model-input group crosses partitions")
        seen.update(current)
    return parts


def source_audit(frame: pd.DataFrame, groups: np.ndarray, parts: dict) -> dict:
    labels = binary_labels(frame[TARGET].to_numpy(), both_classes=True)
    grouped = pd.DataFrame({"group": groups, "label": labels}).groupby("group").label.agg(["size", "min", "max"])
    duplicates = grouped[grouped["size"] > 1]
    categories = {name: {str(k): int(v) for k, v in frame[name].value_counts().sort_index().items()}
                  for name in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "SEX", "EDUCATION", "MARRIAGE"]}
    return {
        "rows": len(frame), "source_sha256": SOURCE_SHA256,
        "feature_dtypes": {name: str(frame[name].dtype) for name in FEATURES},
        "feature_ranges": {name: {"min": float(frame[name].min()), "max": float(frame[name].max())} for name in FEATURES},
        "category_counts": categories, "unique_model_input_groups": int(len(grouped)),
        "duplicate_input_rows": int(len(frame) - len(grouped)),
        "duplicate_groups": int(len(duplicates)),
        "label_discordant_duplicate_groups": int((duplicates["min"] != duplicates["max"]).sum()),
        "partitions": {name: {"rows": len(index), "groups": len(np.unique(groups[index])), "defaults": int(labels[index].sum())} for name, index in parts.items()},
        "population": "Existing Taiwan card clients in a single 2005 cohort",
        "group_identity_limit": "Model-input hashes are grouping devices, not verified unique people",
        "category_handling": "PAY status codes retained as categorical WoE inputs and numeric HGB inputs; sex, education, marital status and ID excluded from fitting; age retained; no fairness certification",
        "out_of_time_validation": False,
    }
