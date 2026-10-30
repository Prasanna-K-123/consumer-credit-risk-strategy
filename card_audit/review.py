"""Independent evidence controls; this module never fits or selects a model.

SQLite window/group queries and a separate Python pair-count formulation check
the published calibrated scores. New queue views are descriptive extensions of
an already inspected historical cohort, not a new confirmatory experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
from typing import Iterable


MODELS = ("scorecard", "challenger")
ROLES = ("train", "calibration", "policy_validation", "test")
PREDICTION_COLUMNS = (
    "source_id", "feature_group", "default", "SEX_for_audit_only", "AGE",
    *MODELS,
)
MANIFEST_COLUMNS = ("source_id", "feature_group", "partition")
CAPACITIES = (0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 1.0)
SQL = Path(__file__).with_name("sql")
INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")
EPSILON = sys.float_info.epsilon


def numeric(value, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} outside [{minimum}, {maximum}]")
    return value


def binary(value) -> int:
    number = numeric(value, "outcome", 0, 1)
    if number not in (0, 1):
        raise ValueError("Outcome must be exactly zero or one")
    return int(number)


def validated_vectors(y: Iterable, p: Iterable) -> tuple[list[int], list[float]]:
    try:
        outcomes = [binary(v) for v in y]
        probabilities = [numeric(v, "probability", 0, 1) for v in p]
    except TypeError as exc:
        raise ValueError("One-dimensional outcome/probability sequences required") from exc
    if not outcomes or len(outcomes) != len(probabilities):
        raise ValueError("Aligned nonempty outcomes and probabilities required")
    return outcomes, probabilities


def independent_metrics(y: Iterable, p: Iterable) -> dict:
    """Pair-count ROC and score-block average precision, without sklearn.

    Score ties receive half credit in ROC. AP integrates recall only after the
    complete tied score block, so arbitrary input order cannot change the value.
    AUC is undefined for a one-class sample; AP is undefined without positives.
    """
    outcomes, probabilities = validated_vectors(y, p)
    blocks: dict[float, list[int]] = {}
    for outcome, probability in zip(outcomes, probabilities):
        block = blocks.setdefault(probability, [0, 0])
        block[outcome] += 1
    positives, negatives = sum(outcomes), len(outcomes) - sum(outcomes)
    below = 0
    pairs = 0.0
    for probability in sorted(blocks):
        negative, positive = blocks[probability]
        pairs += positive * (below + negative / 2)
        below += negative
    seen = true_positives = 0
    ap_numerator = 0.0
    for probability in sorted(blocks, reverse=True):
        negative, positive = blocks[probability]
        true_positives += positive
        seen += negative + positive
        ap_numerator += positive * true_positives / seen
    clipped = [min(1 - EPSILON, max(EPSILON, v)) for v in probabilities]
    return {
        "rows": len(outcomes), "defaults": positives,
        "auc": pairs / (positives * negatives) if positives and negatives else None,
        "average_precision": ap_numerator / positives if positives else None,
        "brier": math.fsum((p - y) ** 2 for y, p in zip(outcomes, probabilities)) / len(outcomes),
        "log_loss": -math.fsum(math.log(p) if y else math.log1p(-p)
                               for y, p in zip(outcomes, clipped)) / len(outcomes),
    }


def read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"Exact CSV header required in {path.name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Nonempty CSV required in {path.name}")
    for i, row in enumerate(rows, 2):
        if set(row) != set(columns) or any(v is None or v == "" for v in row.values()):
            raise ValueError(f"Incomplete/extra CSV fields at {path.name}:{i}")
    return rows


def integer_text(value: str, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, str) or not INTEGER.fullmatch(value):
        raise ValueError(f"Canonical integer required for {name}")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"Invalid {name}")
    return result


def probability_text(value: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Numeric probability required") from exc
    return numeric(result, "probability", 0, 1)


def validate_inputs(predictions: Path, manifest: Path) -> tuple[list[dict], list[dict]]:
    manifest_rows = read_csv(manifest, MANIFEST_COLUMNS)
    prediction_rows = read_csv(predictions, PREDICTION_COLUMNS)
    keys: set[int] = set()
    feature_roles: dict[str, str] = {}
    for row in manifest_rows:
        row["source_id"] = integer_text(row["source_id"], "source_id", 1, 2**63 - 1)
        integer_text(row["feature_group"], "feature_group", 0, 2**64 - 1)
        if row["source_id"] in keys:
            raise ValueError("Duplicate source_id in partition manifest")
        keys.add(row["source_id"])
        if row["partition"] not in ROLES:
            raise ValueError("Unrecognized partition role")
        previous = feature_roles.setdefault(row["feature_group"], row["partition"])
        if previous != row["partition"]:
            raise ValueError("Feature group crosses partition boundary")
    if {row["partition"] for row in manifest_rows} != set(ROLES):
        raise ValueError("Manifest must contain all four study roles")
    keys = set()
    for row in prediction_rows:
        row["source_id"] = integer_text(row["source_id"], "source_id", 1, 2**63 - 1)
        integer_text(row["feature_group"], "feature_group", 0, 2**64 - 1)
        if row["source_id"] in keys:
            raise ValueError("Duplicate source_id in holdout predictions")
        keys.add(row["source_id"])
        row["default"] = integer_text(row["default"], "default", 0, 1)
        row["AGE"] = integer_text(row["AGE"], "AGE", 1, 120)
        row["SEX_for_audit_only"] = integer_text(row["SEX_for_audit_only"], "SEX_for_audit_only", 1, 2)
        for model in MODELS:
            row[model] = probability_text(row[model])
    expected = {row["source_id"]: row["feature_group"] for row in manifest_rows
                if row["partition"] == "test"}
    observed = {row["source_id"]: row["feature_group"] for row in prediction_rows}
    if observed != expected:
        raise ValueError("Prediction customer/group keys must equal the manifest test population")
    return prediction_rows, manifest_rows


def create_database(path: Path, predictions: list[dict], manifest: list[dict], thresholds: dict[str, float]) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.create_function("LN", 1, math.log, deterministic=True)
    connection.create_function("CLIP", 1, lambda p: min(1 - EPSILON, max(EPSILON, p)), deterministic=True)
    connection.executescript("""
      PRAGMA foreign_keys = ON;
      CREATE TABLE manifest (
        source_id INTEGER PRIMARY KEY CHECK(source_id > 0),
        feature_group TEXT NOT NULL,
        partition TEXT NOT NULL CHECK(partition IN ('train','calibration','policy_validation','test'))
      ) STRICT;
      CREATE TABLE customers (
        source_id INTEGER PRIMARY KEY REFERENCES manifest(source_id),
        feature_group TEXT NOT NULL,
        outcome INTEGER NOT NULL CHECK(outcome IN (0, 1)),
        sex INTEGER NOT NULL CHECK(sex IN (1, 2)),
        age INTEGER NOT NULL CHECK(age BETWEEN 1 AND 120)
      ) STRICT;
      CREATE TABLE policies (
        model TEXT PRIMARY KEY, threshold REAL NOT NULL CHECK(threshold BETWEEN 0 AND 1)
      ) STRICT;
      CREATE TABLE predictions (
        source_id INTEGER NOT NULL REFERENCES customers(source_id),
        model TEXT NOT NULL REFERENCES policies(model),
        probability REAL NOT NULL CHECK(probability BETWEEN 0 AND 1),
        PRIMARY KEY(source_id, model)
      ) STRICT;
      CREATE TABLE capacities (fraction REAL PRIMARY KEY CHECK(fraction BETWEEN 0 AND 1)) STRICT;
    """)
    connection.executemany("INSERT INTO manifest VALUES (?, ?, ?)",
        ((r["source_id"], r["feature_group"], r["partition"]) for r in sorted(manifest, key=lambda r: r["source_id"])))
    connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        ((r["source_id"], r["feature_group"], r["default"], r["SEX_for_audit_only"], r["AGE"])
         for r in sorted(predictions, key=lambda r: r["source_id"])))
    connection.executemany("INSERT INTO policies VALUES (?, ?)", sorted(thresholds.items()))
    connection.executemany("INSERT INTO predictions VALUES (?, ?, ?)",
        ((r["source_id"], model, r[model]) for r in sorted(predictions, key=lambda r: r["source_id"])
         for model in MODELS))
    connection.executemany("INSERT INTO capacities VALUES (?)", ((v,) for v in CAPACITIES))
    connection.commit()
    return connection


def query(connection: sqlite3.Connection, filename: str) -> list[dict]:
    return [dict(row) for row in connection.execute((SQL / filename).read_text())]


def require_close(actual, expected, label: str, tolerance: float = 1e-10) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"Expected undefined {label}")
    elif isinstance(expected, bool) or not isinstance(expected, numbers.Real) or not math.isfinite(expected):
        raise ValueError(f"Malformed reference value: {label}")
    elif isinstance(expected, numbers.Integral):
        if not isinstance(actual, numbers.Integral) or actual != expected:
            raise ValueError(f"Count disagreement: {label}")
    elif actual is None or not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance):
        raise ValueError(f"Numeric disagreement: {label}")


def compare_summary(summary: dict, population: list[dict], confusion: list[dict], metrics: dict) -> None:
    for row in population:
        frozen = summary["partitions"][row["partition"]]
        require_close(row["rows"], frozen["rows"], f"{row['partition']} rows")
        require_close(row["feature_groups"], frozen["groups"], f"{row['partition']} groups")
    for row in confusion:
        model = row["model"]
        frozen = summary["models"][model]
        for key in ("auc", "brier", "log_loss"):
            require_close(metrics[model][key], frozen[key], model + " " + key, tolerance=1e-7)
        for key in ("false_negative", "false_positive"):
            require_close(row[key], frozen[key], model + " " + key)
        require_close(row["rows"] - row["flagged"], frozen["accepted"], model + " unflagged")
        require_close(5 * row["false_negative"] + row["false_positive"], frozen["cost_units"], model + " scenario cost")
        require_close(row["defaults"], summary["partitions"]["test"]["defaults"], "test defaults")


def validate_queue_reference(actual: list[dict], path: Path) -> None:
    """Compare independent queue aggregates with the model pipeline's CSV."""
    with path.open(newline="", encoding="utf-8") as stream:
        expected = list(csv.DictReader(stream))
    keys = [(row.get("model"), probability_text(row.get("capacity_fraction", ""))) for row in expected]
    wanted = {(row["model"], row["capacity_fraction"]): row for row in actual}
    if len(keys) != len(set(keys)) or set(keys) != set(wanted):
        raise ValueError("Queue reference model/capacity keys disagree")
    counts = {"capacity_count", "flagged", "tied_at_cutoff", "defaults_flagged", "nondefaults_flagged", "defaults_unflagged", "cost_units"}
    for key, row in zip(keys, expected):
        calculated = wanted[key]
        for column in counts | {"cutoff_probability", "capture_rate", "precision", "flag_rate"}:
            value = row.get(column)
            if value in (None, ""):
                parsed = None
            elif column in counts:
                parsed = integer_text(value, column, 0, 2**63 - 1)
            else:
                parsed = probability_text(value)
            require_close(calculated[column], parsed, f"queue {key} {column}", tolerance=1e-9)
        if row.get("tie_break") != calculated["tie_break"]:
            raise ValueError("Queue reference tie policy disagrees")


def validate_assignment_reference(connection: sqlite3.Connection, path: Path) -> int:
    """Require exact customer membership, including every tied boundary."""
    actual = {
        (row["source_id"], row["model"], row["capacity_fraction"]): row["flagged"]
        for row in query(connection, "queue_assignments.sql")
    }
    columns = ("source_id", "model", "capacity_fraction", "flagged")
    rows = read_csv(path, columns)
    seen = set()
    for row in rows:
        source_id = integer_text(row["source_id"], "queue source_id", 1, 2**63 - 1)
        capacity = probability_text(row["capacity_fraction"])
        flag = integer_text(row["flagged"], "queue flagged", 0, 1)
        key = (source_id, row["model"], capacity)
        if key in seen or key not in actual or actual[key] != flag:
            raise ValueError("Queue assignment identity/flag disagreement")
        seen.add(key)
    if seen != set(actual):
        raise ValueError("Queue assignments must cover the complete customer/model/capacity population")
    return len(seen)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Empty report output")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(predictions: Path, manifest: Path, summary_path: Path, output: Path,
          queue_reference: Path | None = None,
          queue_assignments: Path | None = None) -> dict:
    predictions, manifest, summary_path, output = map(Path, (predictions, manifest, summary_path, output))
    prediction_rows, manifest_rows = validate_inputs(predictions, manifest)
    try:
        summary = json.loads(summary_path.read_text())
        thresholds = {m: numeric(summary["models"][m]["threshold"], "threshold", 0, 1) for m in MODELS}
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Study summary schema is incomplete") from exc
    if output.resolve() in {p.resolve() for p in (predictions, manifest, summary_path)}:
        raise ValueError("Output must be a report directory distinct from inputs")
    with tempfile.TemporaryDirectory(prefix="card-sql-review-") as temporary:
        database = Path(temporary) / "card_review.sqlite"
        connection = create_database(database, prediction_rows, manifest_rows, thresholds)
        try:
            controls = query(connection, "integrity.sql")
            if any(row["violations"] for row in controls):
                raise ValueError("SQL integrity control failed")
            population = query(connection, "population.sql")
            confusion = query(connection, "confusion.sql")
            ranking = {row["model"]: row for row in query(connection, "ranking.sql")}
            calibration = query(connection, "calibration.sql")
            segments = query(connection, "segments.sql")
            queues = query(connection, "queues.sql")
            for row in queues:
                row["tied_at_cutoff"] = connection.execute(
                    "SELECT COUNT(*) FROM predictions WHERE model = ? AND probability = ?",
                    (row["model"], row["cutoff_probability"])).fetchone()[0]
                row["tie_break"] = "probability_desc_source_id_asc"
            independent = {}
            for row in confusion:
                model = row["model"]
                independent[model] = independent_metrics(
                    [r["default"] for r in prediction_rows], [r[model] for r in prediction_rows])
                for key in ("rows", "defaults", "brier", "log_loss"):
                    require_close(row[key], independent[model][key], f"SQL/Python {model} {key}")
                for key in ("auc", "average_precision"):
                    require_close(ranking[model][key], independent[model][key], f"SQL/Python {model} {key}")
                for key in ("rows", "defaults"):
                    if sum(r[key] for r in calibration if r["model"] == model) != row[key]:
                        raise ValueError("Calibration bands fail population reconciliation")
            try:
                compare_summary(summary, population, confusion, independent)
            except (KeyError, TypeError) as exc:
                raise ValueError("Study summary metrics/populations are incomplete") from exc
            if queue_reference is not None:
                validate_queue_reference(queues, Path(queue_reference))
            assignments_compared = (validate_assignment_reference(connection, Path(queue_assignments))
                                    if queue_assignments is not None else 0)
            result = {
                "schema_version": 1, "status": "PASS",
                "analysis_scope": "Postpublication diagnostics of frozen historical card scores; no fitting or new untouched test.",
                "source_sha256": {"holdout_predictions.csv": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                                  "partition_manifest.csv": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                  "summary.json": hashlib.sha256(summary_path.read_bytes()).hexdigest()},
                "sql_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(SQL.glob("*.sql"))},
                "controls": controls, "population": population, "independent_metrics": independent,
                "sql_confusion": confusion,
                "scenario_cost": "5 units per missed default plus 1 per nondefault flagged; not observed currency, profit or intervention benefit.",
                "queue_capacities": list(CAPACITIES),
                "queue_reference_checked": queue_reference is not None,
                "queue_assignment_reference_checked": queue_assignments is not None,
                "queue_assignments_compared": assignments_compared,
                "numeric_tolerance": {"SQL_vs_Python": 1e-10, "published_metrics": 1e-7, "queue_reference": 1e-9},
            }
            if queue_reference is not None:
                result["source_sha256"]["decision_queue.csv"] = hashlib.sha256(Path(queue_reference).read_bytes()).hexdigest()
            if queue_assignments is not None:
                result["source_sha256"]["queue_assignments.csv"] = hashlib.sha256(Path(queue_assignments).read_bytes()).hexdigest()
            output.mkdir(parents=True, exist_ok=True)
            for name, rows in (("population.csv", population), ("confusion.csv", confusion),
                               ("calibration.csv", calibration), ("segments.csv", segments), ("decision_queues.csv", queues)):
                write_csv(output / name, rows)
            result["report_sha256"] = {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in ("population.csv", "confusion.csv", "calibration.csv", "segments.csv", "decision_queues.csv")
            }
            (output / "review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            connection.commit()
        finally:
            connection.close()
        (output / "card_review.sqlite").write_bytes(database.read_bytes())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Defaults to summary.json beside predictions")
    parser.add_argument("--queue-reference", type=Path, help="Optional scientific pipeline decision_queue.csv for independent parity")
    parser.add_argument("--queue-assignments", type=Path, help="Optional pipeline queue_assignments.csv for exact customer membership parity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.predictions, args.partition_manifest,
                       args.summary or args.predictions.with_name("summary.json"),
                       args.output, args.queue_reference, args.queue_assignments)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"CARD_SQL_REVIEW_FAILED: {exc}\n")
    print("CARD_SQL_REVIEW_PASS " + json.dumps({m: result["independent_metrics"][m] for m in MODELS}, allow_nan=False))


if __name__ == "__main__":
    main()
