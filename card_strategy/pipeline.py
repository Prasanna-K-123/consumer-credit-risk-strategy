"""Full-source strategy extension with immutable-reference and evidence guards."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import export_text

from consumer_cards.constants import FEATURES, HISTORICAL, PROTOCOL_SHA256, ROOT, SOURCE_SHA256, TARGET
from consumer_cards.contracts import binary_labels, source_ids
from consumer_cards.data import feature_groups, load_source, partition, sha256
from consumer_cards.models import fit_models
from consumer_cards.replay import historical_integrity
from consumer_cards.runtime import bounded_native_threads

from .analysis import CAPACITIES, COST_RATIOS, ROLES, TREE_PARAMETERS, capacity_scenarios, fit_tree, leaf_segments, policy_sensitivity, predictive_metrics, validate_parts

PACKAGE = Path(__file__).parent
STRATEGY_PROTOCOL_SHA256 = "a58409d59e8af06c798ea284ca29c8e72ded239622fb06a5765ef76626d60014"
MODELS = ("scorecard", "challenger", "cart")
EVIDENCE_FILES = ("summary.json", "predictions.csv", "policy_sensitivity.csv", "capacity.csv", "leaf_segments.csv", "tree.txt", "protocol.json")
SCOPE = ("Retrospective extension of a known Taiwan 2005 cohort. Training, calibration and policy-validation roles remain separate, "
         "but neither this CART benchmark nor its sensitivity results form a new untouched evaluation. Cost ratios are illustrative "
         "error units, not lending economics or measured intervention benefit. No cost ratio, capacity or winning model is selected "
         "using the reported evaluation outcomes. Python CART is not SAS or CHAID execution.")


def _dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_strategy_protocol() -> dict:
    protocol_path = PACKAGE / "PROTOCOL.json"
    if sha256(protocol_path) != STRATEGY_PROTOCOL_SHA256:
        raise ValueError("Fixed strategy protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text())
    bindings = {"version": 1, "source_sha256": SOURCE_SHA256, "historical_protocol_sha256": PROTOCOL_SHA256,
                "features": list(FEATURES), "tree_parameters": TREE_PARAMETERS, "data_roles": list(ROLES),
                "baseline_models": ["scorecard", "challenger"], "false_negative_cost_ratios": list(COST_RATIOS),
                "false_positive_cost": 1, "capacities": list(CAPACITIES),
                "threshold_grid": {"first": .01, "last": .99, "step": .01}}
    for name, expected in bindings.items():
        if protocol.get(name) != expected:
            raise ValueError(f"Strategy protocol and implementation disagree: {name}")
    return protocol


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def fresh_output(output: Path, source: Path, historical: Path = HISTORICAL) -> Path:
    output, source = Path(output).resolve(), Path(source).resolve()
    protected = [ROOT / name for name in ("reference", "consumer_cards", "card_audit", "us_card_market", "card_strategy", "tools", "docs", "tests", "reports")]
    protected.append(Path(historical).resolve())
    if any(_overlap(output, path) for path in protected):
        raise ValueError("Strategy output must not overlap published references, source code, documentation or tests")
    if _overlap(output, source):
        raise ValueError("Strategy output must not overlap the source file")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a fresh strategy output directory; existing evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _implementation_hashes() -> dict:
    paths = [*PACKAGE.glob("*.py"), *(ROOT / "consumer_cards").glob("*.py")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def _partition_manifest(frame: pd.DataFrame, groups: np.ndarray, parts: dict) -> pd.DataFrame:
    roles = np.full(len(frame), "", dtype=object)
    for name in ROLES:
        roles[parts[name]] = name
    return pd.DataFrame({"source_id": source_ids(frame.ID.to_numpy(), len(frame)), "feature_group": groups, "partition": roles})


def _verify_original_manifest(frame: pd.DataFrame, groups: np.ndarray, parts: dict, historical: Path) -> pd.DataFrame:
    current = _partition_manifest(frame, groups, parts)
    reference = pd.read_csv(historical / "partition_manifest.csv", dtype={"feature_group": str})
    if not current.equals(reference):
        raise ValueError("Strategy data roles do not exactly match the original partition manifest")
    return current


@bounded_native_threads
def run_strategy(source: Path, output: Path, historical: Path | None = None) -> dict:
    """Read pinned source, refit fixed models and write the full retrospective pack.

    Acquisition is intentionally separate. Pass an acquired XLS from
    consumer_cards.data.acquire or the root reproduction workflow.
    """
    historical = Path(historical) if historical is not None else HISTORICAL
    protocol = load_strategy_protocol()
    historical_integrity(historical)
    frame = load_source(source)
    groups = feature_groups(frame[list(FEATURES)])
    original_parts = partition(groups)
    validate_parts(frame, original_parts)
    manifest = _verify_original_manifest(frame, groups, original_parts, historical)
    output = fresh_output(output, source, historical)
    baseline = fit_models(frame)
    if any(not np.array_equal(original_parts[name], baseline.parts[name]) for name in ROLES):
        raise ValueError("Baseline refit changed the original data-role positions")
    tree = fit_tree(frame, original_parts)
    X, y = frame[list(FEATURES)], binary_labels(frame[TARGET].to_numpy())
    raw = {**baseline.raw_predictions(X), "cart": tree.raw_predictions(X)}
    probability = {**baseline.predict(X), "cart": tree.predict(X)}
    test, policy = original_parts["test"], original_parts["policy_validation"]
    known = pd.read_csv(historical / "holdout_predictions.csv")
    if not np.array_equal(frame.ID.iloc[test].to_numpy(), known.source_id.to_numpy()):
        raise ValueError("Refitted historical evaluation order changed")
    for model in ("scorecard", "challenger"):
        if not np.allclose(probability[model][test], known[model].to_numpy(), rtol=0, atol=1e-7):
            raise ValueError(f"Baseline historical probabilities changed: {model}")
    predictions = manifest.assign(default=y, cart_leaf_id=tree.model.apply(X))
    for model in MODELS:
        predictions[model] = probability[model]
        predictions[model + "_raw"] = raw[model]
    sensitivities, capacities, metrics = [], [], {}
    for model in MODELS:
        sensitivity = policy_sensitivity(y[policy], probability[model][policy], y[test], probability[model][test])
        sensitivity.insert(0, "model", model)
        sensitivities.append(sensitivity)
        capacity = capacity_scenarios(y[test], probability[model][test], frame.ID.iloc[test].to_numpy())
        capacity.insert(0, "model", model)
        capacities.append(capacity)
        metrics[model] = {"calibrated": predictive_metrics(y[test], probability[model][test]),
                          "raw": predictive_metrics(y[test], raw[model][test])}
    leaves = leaf_segments(frame, tree)
    for name, table in (("predictions.csv", predictions), ("policy_sensitivity.csv", pd.concat(sensitivities, ignore_index=True)),
                        ("capacity.csv", pd.concat(capacities, ignore_index=True)), ("leaf_segments.csv", leaves)):
        table.to_csv(output / name, index=False, float_format="%.12g", lineterminator="\n")
    (output / "tree.txt").write_text(export_text(tree.model, feature_names=list(FEATURES), decimals=8).rstrip() + "\n", encoding="utf-8")
    _dump(output / "protocol.json", protocol)
    summary = {"analysis": protocol["analysis"], "source_sha256": SOURCE_SHA256,
               "historical_protocol_sha256": PROTOCOL_SHA256, "extension_protocol_sha256": sha256(PACKAGE / "PROTOCOL.json"),
               "population": "Existing Taiwan credit-card clients, single 2005 cohort", "n": len(frame),
               "partitions": {name: {"n": len(index), "defaults": int(y[index].sum())} for name, index in original_parts.items()},
               "models": metrics, "tree_parameters": TREE_PARAMETERS,
               "tree": {"leaves": int(tree.model.get_n_leaves()), "depth": int(tree.model.get_depth()),
                        "training_root_samples": int(tree.model.tree_.n_node_samples[0]),
                        "smallest_training_leaf": int(leaves.loc[leaves.partition == "train", "n"].min()),
                        "calibration_intercept": float(tree.calibrator.intercept_[0]),
                        "calibration_slope": float(tree.calibrator.coef_[0, 0])},
               "policy_cost_ratios": list(COST_RATIOS), "capacity_fractions": list(CAPACITIES),
               "threshold_ties": "Lowest threshold among equal minimum policy-validation cost on 0.01..0.99",
               "policy_units": "FN_ratio * missed_defaults + 1 * nondefault_clients_flagged; not currency or treatment effect",
               "baseline_test_probabilities_reconciled": True, "baseline_absolute_tolerance": 1e-7,
               "scope": SCOPE}
    _dump(output / "summary.json", summary)
    _dump(output / "receipt.json", {"executed_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": SOURCE_SHA256,
          "protocol_sha256": sha256(PACKAGE / "PROTOCOL.json"), "native_thread_limit": 2,
          "execution": "Read the pinned XLS; refit fixed scorecard/boosting and fixed shallow CART on training rows only; fit separate calibration; select each cost threshold on policy rows; report every scenario on known historical evaluation rows",
          "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
          "implementation_sha256": _implementation_hashes(), "file_sha256": {name: sha256(output / name) for name in EVIDENCE_FILES}, "scope": SCOPE})
    verification = verify_strategy(output, source, historical)
    _dump(output / "verification.json", verification)
    return {"output": str(output), "verification": verification, "summary": summary}


def verify_strategy(output: Path, source: Path | None = None, historical: Path | None = None) -> dict:
    """Verify identities and accounting; an independent review remains separate."""
    output = Path(output)
    historical = Path(historical) if historical is not None else HISTORICAL
    load_strategy_protocol()
    historical_integrity(historical)
    receipt = json.loads((output / "receipt.json").read_text())
    if receipt.get("source_sha256") != SOURCE_SHA256 or receipt.get("protocol_sha256") != sha256(PACKAGE / "PROTOCOL.json"):
        raise ValueError("Strategy source/protocol receipt identity mismatch")
    if receipt.get("implementation_sha256") != _implementation_hashes():
        raise ValueError("Strategy implementation changed; execute the extension again")
    if set(receipt.get("file_sha256", {})) != set(EVIDENCE_FILES):
        raise ValueError("Strategy receipt must bind the complete evidence set")
    for name, digest in receipt["file_sha256"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Strategy artifact identity mismatch: {name}")
    if json.loads((output / "protocol.json").read_text()) != load_strategy_protocol():
        raise ValueError("Recorded strategy protocol differs from checked implementation")
    predictions = pd.read_csv(output / "predictions.csv", dtype={"feature_group": str})
    expected_columns = ["source_id", "feature_group", "partition", "default", "cart_leaf_id", *[value for model in MODELS for value in (model, model + "_raw")]]
    if list(predictions.columns) != expected_columns or len(predictions) != 30000:
        raise ValueError("Strategy predictions schema/population differs")
    source_ids(predictions.source_id.to_numpy(), len(predictions))
    labels = binary_labels(predictions.default.to_numpy(), both_classes=True)
    manifest = pd.read_csv(historical / "partition_manifest.csv", dtype={"feature_group": str})
    if not predictions[["source_id", "feature_group", "partition"]].equals(manifest):
        raise ValueError("Strategy predictions do not conserve original row identities and data roles")
    if source is not None:
        frame = load_source(source)
        if not np.array_equal(labels, frame[TARGET].to_numpy()):
            raise ValueError("Strategy outcome labels differ from the pinned source")
    test = predictions.partition == "test"
    policy = predictions.partition == "policy_validation"
    sensitivity, capacity = [], []
    summary = json.loads((output / "summary.json").read_text())
    for model in MODELS:
        metrics = {"calibrated": predictive_metrics(labels[test], predictions.loc[test, model]),
                   "raw": predictive_metrics(labels[test], predictions.loc[test, model + "_raw"])}
        for kind in metrics:
            for key, value in metrics[kind].items():
                if not np.isclose(value, summary["models"][model][kind][key], rtol=0, atol=1e-9):
                    raise ValueError(f"Strategy metric accounting differs: {model}.{kind}.{key}")
        current = policy_sensitivity(labels[policy], predictions.loc[policy, model], labels[test], predictions.loc[test, model])
        current.insert(0, "model", model)
        sensitivity.append(current)
        current_capacity = capacity_scenarios(labels[test], predictions.loc[test, model], predictions.loc[test, "source_id"].to_numpy())
        current_capacity.insert(0, "model", model)
        capacity.append(current_capacity)
    for name, current in (("policy_sensitivity.csv", pd.concat(sensitivity, ignore_index=True)), ("capacity.csv", pd.concat(capacity, ignore_index=True))):
        recorded = pd.read_csv(output / name)
        if list(recorded.columns) != list(current.columns) or recorded.shape != current.shape:
            raise ValueError(f"Strategy accounting shape/schema differs: {name}")
        for column in current:
            if pd.api.types.is_numeric_dtype(current[column]):
                if not np.allclose(recorded[column], current[column], rtol=0, atol=1e-8, equal_nan=True):
                    raise ValueError(f"Strategy scenario accounting differs: {name}.{column}")
            elif not recorded[column].equals(current[column]):
                raise ValueError(f"Strategy scenario identity differs: {name}.{column}")
    leaves = pd.read_csv(output / "leaf_segments.csv")
    leaf_columns = ["partition", "leaf_id", "rule", "n", "defaults", "default_rate", "raw_probability", "calibrated_probability"]
    if list(leaves.columns) != leaf_columns or set(leaves.partition) != set(ROLES) or leaves.duplicated(["partition", "leaf_id"]).any():
        raise ValueError("Strategy leaf summaries must identify every role and leaf uniquely")
    for column in ("leaf_id", "n", "defaults"):
        values = leaves[column].to_numpy()
        if values.dtype.kind not in "iu" or (values < 0).any():
            raise ValueError("Strategy leaf counts and identities must be nonnegative integers")
    for role in ROLES:
        for leaf in predictions.cart_leaf_id.unique():
            selected = predictions[(predictions.partition == role) & (predictions.cart_leaf_id == leaf)]
            recorded = leaves[(leaves.partition == role) & (leaves.leaf_id == leaf)]
            if len(recorded) != 1 or int(recorded.iloc[0].n) != len(selected) or int(recorded.iloc[0].defaults) != int(selected.default.sum()):
                raise ValueError("Strategy leaf support/default counts differ from predictions")
            expected_values = {"default_rate": selected.default.mean(), "raw_probability": selected.cart_raw.mean(), "calibrated_probability": selected.cart.mean()}
            for name, value in expected_values.items():
                if not np.isclose(recorded.iloc[0][name], value, rtol=0, atol=1e-10, equal_nan=True):
                    raise ValueError(f"Strategy leaf probability/default-rate differs: {name}")
    expected_parts = {role: {"n": int((predictions.partition == role).sum()), "defaults": int(predictions.loc[predictions.partition == role, "default"].sum())} for role in ROLES}
    if summary.get("partitions") != expected_parts or summary.get("n") != len(predictions) or summary.get("tree_parameters") != TREE_PARAMETERS:
        raise ValueError("Strategy summary population/configuration differs")
    return {"status": "PASS", "source_labels_reconciled": source is not None,
            "population_records": len(predictions), "evaluation_records": int(test.sum()),
            "policy_scenarios": len(MODELS) * len(COST_RATIOS), "capacity_scenarios": len(MODELS) * len(CAPACITIES),
            "leaf_role_rows": len(leaves), "artifact_hashes": len(EVIDENCE_FILES),
            "scope": "Artifact identities and same-implementation accounting checks; separate independent review is required for a second implementation."}
