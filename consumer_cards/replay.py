"""Execute the original fits; verify all scientific outputs and their identities."""
from __future__ import annotations

import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import brier_score_loss

from .constants import CAPACITIES, EXCLUDED, FEATURES, HISTORICAL, HISTORICAL_HASHES, MODEL_NAMES, PROTOCOL_SHA256, SOURCE_SHA256, TARGET
from .contracts import aligned, source_ids
from .data import load_protocol, load_source, sha256, source_audit
from .evaluation import decision_queue, metrics, paired_group_bootstrap
from .models import fit_models
from .runtime import bounded_native_threads

FLOAT_TOLERANCE = 1e-7
CSV_NAMES = ("partition_manifest.csv", "holdout_predictions.csv", "calibration_bins.csv", "group_audit.csv")


def dump(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def historical_integrity(folder: Path = HISTORICAL) -> dict:
    folder = Path(folder)
    checked = {}
    for name, expected in HISTORICAL_HASHES.items():
        actual = sha256(folder / name)
        if actual != expected:
            raise ValueError(f"Historical reference hash mismatch: {name}")
        checked[name] = actual
    load_protocol(folder)
    return checked


def _new_output(folder: Path) -> Path:
    folder = Path(folder).resolve()
    historical = HISTORICAL.resolve()
    if folder == historical or historical in folder.parents or folder in historical.parents:
        raise ValueError("Output must not overlap the immutable historical reference")
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Use a fresh output directory; existing evidence is not overwritten")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_queue_outputs(prediction_file: Path, output: Path, *, require_fresh: bool = False) -> dict:
    destination, protected = Path(output).resolve(), HISTORICAL.resolve()
    if destination == protected or protected in destination.parents or destination in protected.parents:
        raise ValueError("Queue output must not overlap immutable historical evidence")
    output = _new_output(output) if require_fresh else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(prediction_file, dtype={"feature_group": str})
    required = {"source_id", "default", *MODEL_NAMES}
    if not required <= set(frame.columns):
        raise ValueError("Prediction file lacks required card model columns")
    ids = source_ids(frame.source_id.to_numpy(), len(frame))
    rows, assignments = [], []
    for name in MODEL_NAMES:
        for capacity in CAPACITIES:
            result, flags = decision_queue(frame.default.to_numpy(), frame[name].to_numpy(), ids, capacity)
            rows.append({"model": name, **result})
            assignments.extend({"source_id": int(source_id), "model": name, "capacity_fraction": capacity, "flagged": int(flag)}
                               for source_id, flag in zip(ids, flags))
    pd.DataFrame(rows).to_csv(output / "decision_queue.csv", index=False)
    pd.DataFrame(assignments).to_csv(output / "queue_assignments.csv", index=False)
    receipt = {
        "analysis": "Postpublication descriptive review-capacity scenarios on historical held-out predictions",
        "source_prediction_sha256": sha256(prediction_file),
        "capacities": list(CAPACITIES),
        "ranking": "probability_desc_source_id_asc", "queue_size": "floor(capacity_fraction * n)",
        "tied_boundary": "Break tied probabilities by ascending original source ID; never use outcomes",
        "cost": "Assumed error weights FN=5, FP=1; not money or causal intervention utility",
        "scope": "No capacity is selected using these evaluation outcomes; reported values are descriptive, without new confirmatory intervals",
        "file_sha256": {name: sha256(output / name) for name in ("decision_queue.csv", "queue_assignments.csv")},
    }
    dump(output / "queue_receipt.json", receipt)
    return receipt


@bounded_native_threads
def run_replay(data: Path, output: Path, historical: Path = HISTORICAL) -> dict:
    destination, protected = Path(output).resolve(), Path(historical).resolve()
    if destination == protected or protected in destination.parents or destination in protected.parents:
        raise ValueError("Replay output must not overlap immutable historical evidence")
    historical_integrity(historical)
    protocol = load_protocol(historical)
    frame = load_source(data)
    output = _new_output(output)
    fitted = fit_models(frame)
    y = frame[TARGET].to_numpy(dtype=int)
    test = fitted.parts["test"]
    X_test = frame.iloc[test][list(FEATURES)]
    raw = fitted.raw_predictions(X_test)
    predictions = fitted.predict(X_test)
    results, bins, group_audit = {}, [], []
    for name, probability in predictions.items():
        threshold = fitted.thresholds[name]
        results[name] = metrics(y[test], probability, threshold)
        results[name]["raw_test_brier"] = float(brier_score_loss(y[test], raw[name]))
        for i in range(10):
            mask = (probability >= i / 10) & ((probability < (i + 1) / 10) if i < 9 else (probability <= 1))
            count = int(mask.sum())
            bins.append({"model": name, "bin_left": i / 10, "bin_right": (i + 1) / 10, "n": count,
                         "mean_probability": float(probability[mask].mean()) if count else None,
                         "default_rate": float(y[test][mask].mean()) if count else None})
    sex, age = frame.SEX.iloc[test].to_numpy(), frame.AGE.iloc[test].to_numpy()
    masks = [("sex_1", sex == 1), ("sex_2", sex == 2), ("age_under30", age < 30),
             ("age_30to49", (age >= 30) & (age < 50)), ("age_50plus", age >= 50)]
    for label, mask in masks:
        for name, probability in predictions.items():
            row = metrics(y[test][mask], probability[mask], fitted.thresholds[name])
            row.update(model=name, group=label, n=int(mask.sum()), default_rate=float(y[test][mask].mean()),
                       false_negative_rate=row["false_negative"] / int(y[test][mask].sum()),
                       false_positive_rate=row["false_positive"] / int((1 - y[test][mask]).sum()))
            group_audit.append(row)
    holdout = pd.DataFrame({"source_id": frame.ID.iloc[test].to_numpy(), "feature_group": fitted.groups[test],
                            "default": y[test], "SEX_for_audit_only": sex, "AGE": age, **predictions})
    split = pd.DataFrame({"source_id": frame.ID, "feature_group": fitted.groups, "partition": ""})
    for name, index in fitted.parts.items():
        split.loc[index, "partition"] = name
    csvs = {"partition_manifest.csv": split.to_csv(index=False),
            "holdout_predictions.csv": holdout.to_csv(index=False, float_format="%.12g"),
            "calibration_bins.csv": pd.DataFrame(bins).to_csv(index=False),
            "group_audit.csv": pd.DataFrame(group_audit).to_csv(index=False)}
    for name, content in csvs.items():
        (output / name).write_text(content, encoding="utf-8", newline="")
    summary = {
        "source": protocol["source"], "data_sha256": SOURCE_SHA256,
        "license": "CC BY 4.0; I-Cheng Yeh, UCI DOI10.24432/C55S3H",
        "rows": len(frame), "features": list(FEATURES), "excluded": list(EXCLUDED),
        "unique_model_input_groups": len(np.unique(fitted.groups)),
        "duplicate_input_rows": len(frame) - len(np.unique(fitted.groups)),
        "partitions": {name: {"rows": len(index), "groups": len(np.unique(fitted.groups[index])), "defaults": int(y[index].sum()),
                              "default_rate": float(y[index].mean())} for name, index in fitted.parts.items()},
        "cross_partition_duplicate_groups": 0, "models": results, "calibration": fitted.calibration_info,
        "paired_group_bootstrap_replicates": 2000,
        "paired_uncertainty": paired_group_bootstrap(y[test], predictions["scorecard"], predictions["challenger"],
                                                      fitted.thresholds["scorecard"], fitted.thresholds["challenger"], fitted.groups[test]),
        "constant_policies": {"accept_all_cost_units": int(y[test].sum() * 5), "reject_all_cost_units": int((1 - y[test]).sum())},
        "protocol_sha256": PROTOCOL_SHA256,
        "file_sha256": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in csvs.items()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "scope": protocol["scope"],
        "uncertainty_scope": "Fixed fitted models and calibration/thresholds; group resampling does not include training/refit uncertainty.",
        "fair_comparison": "Both models use identical 20 predictors and exactly the same training/calibration/policy-validation/test clients. No test-driven tuning. Separate preprocessing is part of each prespecified model.",
    }
    dump(output / "summary.json", summary)
    dump(output / "source_audit.json", source_audit(frame, fitted.groups, fitted.parts))
    write_queue_outputs(output / "holdout_predictions.csv", output)
    code_hashes = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    dump(output / "replay_receipt.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": SOURCE_SHA256,
        "protocol_sha256": PROTOCOL_SHA256, "implementation_sha256": code_hashes,
        "native_thread_limit": 2,
        "execution": "New full source read, original fixed model refits, calibration, validation threshold selection, predictions and 2000 group-bootstrap draws",
        "evidence_status": "Numerical replay of previously published historical evidence; not a new untouched holdout or preregistered experiment",
        "file_sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    })
    verification = verify_replay(output, historical)
    dump(output / "verification.json", verification)
    return {"output": str(output), "verification": verification}


def _compare(expected, actual, path="summary", *, atol=FLOAT_TOLERANCE) -> int:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError(f"Scientific dictionary keys differ at {path}")
        return sum(_compare(expected[key], actual[key], f"{path}.{key}", atol=atol) for key in expected)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Scientific list differs at {path}")
        return sum(_compare(left, right, f"{path}[{i}]", atol=atol) for i, (left, right) in enumerate(zip(expected, actual)))
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(expected) is not type(actual) or expected != actual:
            raise ValueError(f"Scientific identity differs at {path}")
    elif isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise ValueError(f"Exact integer differs at {path}")
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not np.isfinite(actual) or not np.isclose(expected, actual, rtol=0, atol=atol):
            raise ValueError(f"Numerical comparison failed at {path}: {actual} versus {expected}")
    else:
        raise ValueError(f"Unsupported comparison value at {path}")
    return 1


def _compare_csv(reference: Path, current: Path) -> int:
    expected = pd.read_csv(reference, dtype={"feature_group": str})
    actual = pd.read_csv(current, dtype={"feature_group": str})
    if list(expected.columns) != list(actual.columns) or expected.shape != actual.shape:
        raise ValueError(f"CSV schema/shape mismatch: {reference.name}")
    fields = 0
    for name in expected.columns:
        first, second = expected[name], actual[name]
        if pd.api.types.is_float_dtype(first):
            try:
                values = second.to_numpy(float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Nonnumeric CSV column {reference.name}:{name}") from error
            if not np.allclose(first.to_numpy(float), values, rtol=0, atol=FLOAT_TOLERANCE, equal_nan=True):
                raise ValueError(f"Numerical CSV difference: {reference.name}:{name}")
        elif not first.equals(second):
            raise ValueError(f"Exact CSV identity/count difference: {reference.name}:{name}")
        fields += len(first)
    return fields


def verify_replay(output: Path, historical: Path = HISTORICAL) -> dict:
    """Artifact comparison only; run_replay separately performs the actual refits."""
    output, historical = Path(output), Path(historical)
    historical_integrity(historical)
    expected = json.loads((historical / "summary.json").read_text())
    actual = json.loads((output / "summary.json").read_text())
    for name in CSV_NAMES:
        if sha256(output / name) != actual.get("file_sha256", {}).get(name):
            raise ValueError(f"Replay artifact hash mismatch: {name}")
    if actual.get("protocol_sha256") != PROTOCOL_SHA256 or actual.get("data_sha256") != SOURCE_SHA256:
        raise ValueError("Replay source/protocol identity mismatch")
    for summary in (expected, actual):
        summary.pop("environment", None)
        summary.pop("file_sha256", None)
    field_count = _compare(expected, actual)
    csv_count = sum(_compare_csv(historical / name, output / name) for name in CSV_NAMES)
    receipt = json.loads((output / "replay_receipt.json").read_text())
    if receipt.get("source_sha256") != SOURCE_SHA256 or receipt.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Replay receipt source/protocol identity mismatch")
    implementation = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    if receipt.get("implementation_sha256") != implementation:
        raise ValueError("Replay implementation identity differs from current source; execute a new replay")
    required_files = {"summary.json", *CSV_NAMES, "source_audit.json", "decision_queue.csv", "queue_assignments.csv", "queue_receipt.json"}
    if set(receipt.get("file_sha256", {})) != required_files:
        raise ValueError("Replay receipt does not bind the complete expected artifact set")
    for name, expected_hash in receipt["file_sha256"].items():
        if sha256(output / name) != expected_hash:
            raise ValueError(f"Replay receipt artifact hash mismatch: {name}")
    return {"passed": True, "scientific_summary_fields_compared": field_count,
            "csv_cells_compared": csv_count, "absolute_tolerance": FLOAT_TOLERANCE,
            "relative_tolerance": 0, "exact_integer_and_identity_checks": True,
            "historical_file_hashes_verified": len(HISTORICAL_HASHES),
            "scope": "All historical scientific summary fields, calibration parameters, paired confidence intervals and prediction/group/calibration CSV cells; exact immutable references; receipt binds source/category audit and new queue artifacts. This comparison itself does not refit models."}
