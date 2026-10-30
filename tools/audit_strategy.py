"""Independent SQLite and direct-arithmetic review of the strategy extension.

This module imports no fitting, threshold-selection or queue code from
``card_strategy``. It checks recorded evidence, not applicant proficiency or a
fresh experiment. Source-label and rule checks require the pinned source XLS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from card_audit.review import independent_metrics

MODELS = ("scorecard", "challenger", "cart")
ROLES = ("train", "calibration", "policy_validation", "test")
RATIOS = (1, 2, 5, 10, 20)
CAPACITIES = (.05, .10, .20)
FILES = ("summary.json", "predictions.csv", "policy_sensitivity.csv", "capacity.csv", "leaf_segments.csv", "tree.txt", "protocol.json", "receipt.json", "verification.json")
PREDICTION_COLUMNS = ("source_id", "feature_group", "partition", "default", "cart_leaf_id", *[v for m in MODELS for v in (m, m + "_raw")])
INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")
CLAUSE = re.compile(r"([A-Z][A-Z_0-9]*) (<=|>) (-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?)\Z", re.I)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON evidence")))
    if not isinstance(value, dict):
        raise ValueError("JSON evidence must be an object")
    return value


def read_csv(path: Path, columns=None, *, allow_empty=False) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = tuple(reader.fieldnames or ())
        if not names or len(names) != len(set(names)) or (columns is not None and names != tuple(columns)):
            raise ValueError("Exact nonduplicated CSV schema required: " + path.name)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV evidence must be nonempty: " + path.name)
    for row in rows:
        if set(row) != set(names) or any(v is None or (v == "" and not allow_empty) for v in row.values()):
            raise ValueError("Missing or extra CSV values: " + path.name)
    return rows


def integer(value, label, *, minimum=0, maximum=2**64-1) -> int:
    if not isinstance(value, str) or not INTEGER.fullmatch(value):
        raise ValueError("Canonical integer required: " + label)
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError("Integer outside contract: " + label)
    return parsed


def real(value, label, *, probability=False) -> float:
    if isinstance(value, bool):
        raise ValueError("Numeric evidence required: " + label)
    try:
        parsed = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Numeric evidence required: " + label) from exc
    if not math.isfinite(parsed) or (probability and not 0 <= parsed <= 1):
        raise ValueError("Finite evidence within contract required: " + label)
    return parsed


def close(actual, expected, label, *, tolerance=1e-8) -> None:
    if expected is None:
        if actual not in (None, ""):
            raise ValueError("Undefined evidence disagreement: " + label)
        return
    found = real(actual, label)
    if not math.isclose(found, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError("Independent evidence disagreement: " + label)


def checked_keys(rows: list[dict], fields: tuple[str, ...], expected: set[tuple], label: str) -> dict:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError("Duplicate scenario key: " + label)
        result[key] = row
    if set(result) != expected:
        raise ValueError("Incomplete or extra scenario keys: " + label)
    return result


def validate_predictions(rows: list[dict], manifest: list[dict]) -> list[dict]:
    if len(rows) != 30000 or len(manifest) != 30000:
        raise ValueError("Exactly 30000 original customer records required")
    group_roles = {}
    for i, (row, known) in enumerate(zip(rows, manifest), 1):
        if any(row[name] != known[name] for name in ("source_id", "feature_group", "partition")):
            raise ValueError("Customer/group/role differs from immutable manifest")
        if integer(row["source_id"], "source_id", minimum=1) != i:
            raise ValueError("Source identities must be ordered original keys 1..30000")
        integer(row["feature_group"], "feature_group")
        if row["partition"] not in ROLES:
            raise ValueError("Unknown data role")
        previous = group_roles.setdefault(row["feature_group"], row["partition"])
        if previous != row["partition"]:
            raise ValueError("Feature group crosses data roles")
        row["source_id"] = i
        row["default"] = integer(row["default"], "default", maximum=1)
        row["cart_leaf_id"] = integer(row["cart_leaf_id"], "cart_leaf_id")
        for model in MODELS:
            for column in (model, model + "_raw"):
                row[column] = real(row[column], column, probability=True)
    if {r["partition"] for r in rows} != set(ROLES):
        raise ValueError("Every data role is required")
    return rows


def database(rows: list[dict]) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
      CREATE TABLE scores (source_id INTEGER, role TEXT, outcome INTEGER,
        leaf INTEGER, model TEXT, probability REAL, raw REAL,
        PRIMARY KEY(source_id,model)) STRICT;
      CREATE TABLE thresholds (threshold REAL PRIMARY KEY) STRICT;
    """)
    con.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?)", ((r["source_id"], r["partition"], r["default"], r["cart_leaf_id"], m, r[m], r[m + "_raw"]) for r in rows for m in MODELS))
    con.executemany("INSERT INTO thresholds VALUES (?)", ((v / 100,) for v in range(1, 100)))
    return con


def check_policy(con: sqlite3.Connection, rows: list[dict]) -> int:
    expected = {(m, float(r)) for m in MODELS for r in RATIOS}
    for row in rows:
        row["missed_default_cost"] = real(row["missed_default_cost"], "cost ratio")
    keyed = checked_keys(rows, ("model", "missed_default_cost"), expected, "policies")
    grid = list(con.execute("""SELECT model,threshold,
      SUM(outcome=1 AND probability<threshold) AS fn,
      SUM(outcome=0 AND probability>=threshold) AS fp
      FROM scores CROSS JOIN thresholds WHERE role='policy_validation'
      GROUP BY model,threshold ORDER BY model,threshold"""))
    for model, ratio in sorted(expected):
        choice = min((r for r in grid if r["model"] == model), key=lambda r: (ratio * r["fn"] + r["fp"], r["threshold"]))
        row = keyed[model, ratio]
        threshold = choice["threshold"]
        values = dict(con.execute("""SELECT COUNT(*) n,SUM(outcome) defaults,
          SUM(probability>=?) reviewed,
          SUM(outcome=1 AND probability>=?) true_positive,
          SUM(outcome=0 AND probability>=?) false_positive,
          SUM(outcome=1 AND probability<?) false_negative,
          SUM(outcome=0 AND probability<?) true_negative
          FROM scores WHERE role='test' AND model=?""", (threshold,) * 5 + (model,)).fetchone())
        values.update(selected_threshold=threshold, policy_validation_cost_units=ratio * choice["fn"] + choice["fp"], false_alarm_cost=1)
        values.update(review_rate=values["reviewed"] / values["n"], default_capture=values["true_positive"] / values["defaults"], queue_default_rate=values["true_positive"] / values["reviewed"] if values["reviewed"] else None)
        values.update(cost_units=ratio * values["false_negative"] + values["false_positive"], review_none_cost_units=ratio * values["defaults"], review_all_cost_units=values["n"] - values["defaults"])
        values["minus_best_constant_cost_units"] = values["cost_units"] - min(values["review_none_cost_units"], values["review_all_cost_units"])
        if set(row) != {"model", "missed_default_cost", *values}:
            raise ValueError("Policy evidence schema differs")
        for name, expected_value in values.items():
            close(row[name], expected_value, "policy " + model + "." + str(ratio) + "." + name)
    return len(expected)


def check_capacity(con: sqlite3.Connection, rows: list[dict]) -> int:
    for row in rows:
        row["capacity_fraction"] = real(row["capacity_fraction"], "capacity", probability=True)
    expected = {(m, c) for m in MODELS for c in CAPACITIES}
    keyed = checked_keys(rows, ("model", "capacity_fraction"), expected, "capacity")
    for model, capacity in sorted(expected):
        ordered = list(con.execute("""SELECT source_id,outcome,probability FROM scores
          WHERE role='test' AND model=? ORDER BY probability DESC,source_id ASC""", (model,)))
        n, positives = len(ordered), sum(r["outcome"] for r in ordered)
        k = math.floor(capacity * n)
        selected = ordered[:k]
        tp = sum(r["outcome"] for r in selected)
        fp, fn = k - tp, positives - tp
        cutoff = selected[-1]["probability"] if selected else None
        values = dict(capacity_count=k, flagged=k, tied_at_cutoff=sum(r["probability"] == cutoff for r in ordered), defaults_flagged=tp, nondefaults_flagged=fp, defaults_unflagged=fn, cost_units=5 * fn + fp, cutoff_probability=cutoff, capture_rate=tp / positives, precision=tp / k if k else None, flag_rate=k / n)
        row = keyed[model, capacity]
        if set(row) != {"model", "capacity_fraction", "tie_break", *values} or row["tie_break"] != "probability_desc_source_id_asc":
            raise ValueError("Capacity schema/tie rule differs")
        for name, value in values.items():
            close(row[name], value, "capacity " + model + "." + str(capacity) + "." + name)
    return len(expected)


def check_leaves(con, rows, predictions, summary, frame=None) -> int:
    for row in rows:
        row["leaf_id"] = integer(row["leaf_id"], "leaf_id")
    leaves = {r["cart_leaf_id"] for r in predictions}
    keyed = checked_keys(rows, ("partition", "leaf_id"), {(role, leaf) for role in ROLES for leaf in leaves}, "leaves")
    training = {}
    rules = {}
    for leaf in sorted(leaves):
        rule_set = {keyed[role, leaf]["rule"] for role in ROLES}
        if len(rule_set) != 1:
            raise ValueError("Leaf rules disagree between roles")
        rule = next(iter(rule_set))
        clauses = rule.split(" AND ")
        if len(clauses) > 3 or any(CLAUSE.fullmatch(c) is None for c in clauses):
            raise ValueError("Malformed or over-depth CART rule")
        rules[leaf] = clauses
        tr = con.execute("SELECT COUNT(*) n,SUM(outcome) d FROM scores WHERE model='cart' AND role='train' AND leaf=?", (leaf,)).fetchone()
        training[leaf] = (tr["n"], tr["d"] / tr["n"])
        for role in ROLES:
            row = keyed[role, leaf]
            result = dict(con.execute("""SELECT COUNT(*) n,COALESCE(SUM(outcome),0) defaults,
              AVG(outcome) default_rate,MIN(raw) lo,MAX(raw) hi,
              MIN(probability) cal_lo,MAX(probability) cal_hi
              FROM scores WHERE model='cart' AND role=? AND leaf=?""", (role, leaf)).fetchone())
            for name in ("n", "defaults", "default_rate"):
                close(row[name], result[name], "leaf " + role + "." + str(leaf) + "." + name)
            for name, low, high in (("raw_probability", "lo", "hi"), ("calibrated_probability", "cal_lo", "cal_hi")):
                close(row[name], result[low], "leaf " + name)
                if result["n"]:
                    close(result[low], result[high], "constant probability within leaf", tolerance=1e-12)
            if result["n"]:
                close(result["lo"], training[leaf][1], "CART raw probability must equal training leaf outcome rate")
                clipped = min(1 - 1e-6, max(1e-6, result["lo"]))
                logit = math.log(clipped / (1 - clipped))
                z = real(summary["tree"]["calibration_intercept"], "tree intercept") + real(summary["tree"]["calibration_slope"], "tree slope") * logit
                expected_p = 1 / (1 + math.exp(-z))
                close(result["cal_lo"], expected_p, "tree calibration mapping")
    close(summary["tree"]["leaves"], len(leaves), "leaf count", tolerance=0)
    close(summary["tree"]["depth"], max(len(v) for v in rules.values()), "tree depth", tolerance=0)
    close(summary["tree"]["training_root_samples"], sum(v[0] for v in training.values()), "training root count", tolerance=0)
    close(summary["tree"]["smallest_training_leaf"], min(v[0] for v in training.values()), "smallest training leaf", tolerance=0)
    if min(v[0] for v in training.values()) < 200:
        raise ValueError("Declared minimum training leaf support violated")
    if frame is not None:
        import numpy as np
        from consumer_cards.constants import FEATURES
        memberships = {}
        for leaf, clauses in rules.items():
            mask = np.ones(len(frame), dtype=bool)
            for clause in clauses:
                name, op, threshold = CLAUSE.fullmatch(clause).groups()
                if name not in FEATURES:
                    raise ValueError("Rule uses an excluded or unknown predictor")
                values = frame[name].to_numpy(dtype=np.float32)
                mask &= values <= float(threshold) if op == "<=" else values > float(threshold)
            memberships[leaf] = mask
        if not np.all(sum(memberships.values()) == 1):
            raise ValueError("Tree rules must assign every source row exactly once")
        for row in predictions:
            if not memberships[row["cart_leaf_id"]][row["source_id"] - 1]:
                raise ValueError("Recorded leaf differs from exact source rule")
    return len(keyed)


def audit(evidence: Path, source: Path | None = None, historical: Path | None = None) -> dict:
    evidence = Path(evidence)
    historical = Path(historical) if historical is not None else ROOT / "reference/card_clients/historical"
    hashes = {name: digest(evidence / name) for name in FILES}
    summary, receipt, verification = (load_json(evidence / name) for name in ("summary.json", "receipt.json", "verification.json"))
    protocol = load_json(evidence / "protocol.json")
    if protocol != load_json(ROOT / "card_strategy/PROTOCOL.json"):
        raise ValueError("Strategy protocol differs from checked source")
    if (summary.get("tree_parameters") != protocol["tree_parameters"]
            or summary.get("policy_cost_ratios") != list(RATIOS)
            or summary.get("capacity_fractions") != list(CAPACITIES)
            or summary.get("historical_protocol_sha256") != protocol["historical_protocol_sha256"]):
        raise ValueError("Summary configuration differs from declared strategy protocol")
    protocol_hash = digest(ROOT / "card_strategy/PROTOCOL.json")
    if receipt.get("protocol_sha256") != protocol_hash or summary.get("extension_protocol_sha256") != protocol_hash:
        raise ValueError("Protocol identities differ")
    code_paths = [* (ROOT / "card_strategy").glob("*.py"), * (ROOT / "consumer_cards").glob("*.py")]
    current_code = {str(path.relative_to(ROOT)): digest(path) for path in sorted(code_paths)}
    if receipt.get("implementation_sha256") != current_code:
        raise ValueError("Execution receipt is not bound to the current fitting implementation")
    expected_files = set(FILES) - {"receipt.json", "verification.json"}
    if set(receipt.get("file_sha256", {})) != expected_files or any(receipt["file_sha256"][name] != hashes[name] for name in expected_files):
        raise ValueError("Executed receipt does not bind complete unchanged evidence")
    if verification.get("status") != "PASS" or receipt.get("source_sha256") != protocol["source_sha256"] or summary.get("source_sha256") != protocol["source_sha256"]:
        raise ValueError("Execution status or source identity differs")
    manifest = read_csv(historical / "partition_manifest.csv", ("source_id", "feature_group", "partition"))
    predictions = validate_predictions(read_csv(evidence / "predictions.csv", PREDICTION_COLUMNS), manifest)
    test = [row for row in predictions if row["partition"] == "test"]
    known = read_csv(historical / "holdout_predictions.csv")
    known_by_id = {integer(r["source_id"], "historical source_id"): r for r in known}
    if len(known_by_id) != len(known) or set(known_by_id) != {r["source_id"] for r in test}:
        raise ValueError("Historical evaluation keys differ")
    for row in test:
        other = known_by_id[row["source_id"]]
        if integer(other["default"], "historical default", maximum=1) != row["default"]:
            raise ValueError("Evaluation labels differ from historical source evidence")
        for model in MODELS[:2]:
            close(row[model], real(other[model], "historical probability"), "historical baseline " + model, tolerance=1e-7)
    frame = None
    if source is not None:
        from consumer_cards.constants import TARGET
        from consumer_cards.data import load_source
        frame = load_source(Path(source))
        if frame[TARGET].tolist() != [r["default"] for r in predictions]:
            raise ValueError("Outcome labels differ from the pinned source XLS")
    close(summary.get("n"), len(predictions), "summary population", tolerance=0)
    if set(summary.get("partitions", {})) != set(ROLES) or set(summary.get("models", {})) != set(MODELS):
        raise ValueError("Summary must contain exact models and roles")
    for role in ROLES:
        selected = [r for r in predictions if r["partition"] == role]
        close(summary["partitions"][role]["n"], len(selected), role + " count", tolerance=0)
        close(summary["partitions"][role]["defaults"], sum(r["default"] for r in selected), role + " defaults", tolerance=0)
    metrics = {}
    for model in MODELS:
        metrics[model] = {}
        for kind, column in (("calibrated", model), ("raw", model + "_raw")):
            values = independent_metrics([r["default"] for r in test], [r[column] for r in test])
            metrics[model][kind] = values
            for name in ("auc", "average_precision", "brier", "log_loss"):
                close(summary["models"][model][kind][name], values[name], model + "." + kind + "." + name, tolerance=1e-9)
    con = database(predictions)
    try:
        policy_count = check_policy(con, read_csv(evidence / "policy_sensitivity.csv", allow_empty=True))
        capacity_count = check_capacity(con, read_csv(evidence / "capacity.csv", allow_empty=True))
        leaf_count = check_leaves(con, read_csv(evidence / "leaf_segments.csv", allow_empty=True), predictions, summary, frame)
    finally:
        con.close()
    for name, value in {"population_records": len(predictions), "evaluation_records": len(test), "policy_scenarios": policy_count, "capacity_scenarios": capacity_count, "leaf_role_rows": leaf_count, "artifact_hashes": len(expected_files)}.items():
        close(verification.get(name), value, "verification " + name, tolerance=0)
    return {"status": "PASS", "review_implementation": "SQLite policy-threshold counts and capacity order; separate pair-count metrics and direct probability arithmetic; no imports from card_strategy", "source_labels_and_rules_checked": source is not None, "source_labels_reconciled": source is not None, "population_records": len(predictions), "evaluation_records": len(test), "policy_scenarios": policy_count, "capacity_scenarios": capacity_count, "leaf_role_rows": leaf_count, "independent_metrics": metrics, "input_sha256": hashes, "review_code_sha256": digest(Path(__file__)), "historical_manifest_sha256": digest(historical / "partition_manifest.csv"), "scope": "Independent software accounting of a retrospective known-cohort extension; not external institutional approval, a new unseen test, monetary benefit or applicant skill certification."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New JSON review path; existing outputs are preserved")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new review output path")
    try:
        result = audit(args.evidence, args.source, args.historical)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as exc:
        parser.exit(2, f"STRATEGY_REVIEW_FAILED: {exc}\n")
    print(json.dumps({name: result[name] for name in ("status", "population_records", "policy_scenarios", "capacity_scenarios", "leaf_role_rows")}))


if __name__ == "__main__":
    main()
