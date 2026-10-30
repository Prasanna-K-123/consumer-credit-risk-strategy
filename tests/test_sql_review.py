"""Independent closed-form examples and corrupted-evidence controls."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from card_audit.review import (
    CAPACITIES, MANIFEST_COLUMNS, MODELS, PREDICTION_COLUMNS,
    audit, independent_metrics,
)


def write_rows(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class AnalyticMetricTests(unittest.TestCase):
    def test_tied_scores_use_half_roc_credit_and_complete_ap_blocks(self):
        result = independent_metrics([0, 1, 1, 0], [.5, .5, .2, .8])
        self.assertAlmostEqual(result["auc"], .125)
        self.assertAlmostEqual(result["average_precision"], 5 / 12)
        self.assertAlmostEqual(result["brier"], .445)
        self.assertAlmostEqual(result["log_loss"], -(math.log(.5) + math.log(.2)) / 2)
        reordered = independent_metrics([1, 0, 0, 1], [.5, .8, .5, .2])
        self.assertEqual(result, reordered)

    def test_one_class_scores_report_defined_and_undefined_separately(self):
        negative = independent_metrics([0, 0], [.1, .7])
        self.assertIsNone(negative["auc"])
        self.assertIsNone(negative["average_precision"])
        positive = independent_metrics([1, 1], [.1, .7])
        self.assertIsNone(positive["auc"])
        self.assertEqual(positive["average_precision"], 1.0)

    def test_boundaries_have_finite_loss_and_perfect_rank(self):
        result = independent_metrics([0, 1], [0, 1])
        self.assertEqual(result["auc"], 1)
        self.assertEqual(result["average_precision"], 1)
        self.assertEqual(result["brier"], 0)
        self.assertTrue(math.isfinite(result["log_loss"]))

    def test_contract_rejects_bad_values_before_clipping(self):
        cases = [([0, .5], [.1, .5]), ([0, 2], [.1, .5]),
                 ([False, 1], [.1, .5]), ([0, 1], [.1, math.inf]),
                 ([0, 1], [.1, math.nan]), ([0, 1], [.1, 1.01]),
                 ([0, 1], [.1, -.01]), ([0, 1], [.1]),
                 ([], []), ([[0], [1]], [.1, .5]),
                 ([0, 1], [[.1], [.5]]), ([0, 1], [True, .5])]
        for y, p in cases:
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                independent_metrics(y, p)


class SQLIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.predictions = self.root / "holdout_predictions.csv"
        self.manifest = self.root / "partition_manifest.csv"
        self.summary = self.root / "summary.json"
        self.manifest_rows = [
            {"source_id": 1, "feature_group": "101", "partition": "train"},
            {"source_id": 2, "feature_group": "102", "partition": "calibration"},
            {"source_id": 3, "feature_group": "103", "partition": "policy_validation"},
        ]
        self.prediction_rows = []
        for i in range(20):
            source_id = i + 4
            # Two different customer IDs share a model-input group in test.
            group = str(200 + (0 if i == 1 else i))
            self.manifest_rows.append({"source_id": source_id, "feature_group": group, "partition": "test"})
            self.prediction_rows.append({"source_id": source_id, "feature_group": group, "default": i % 2,
                "SEX_for_audit_only": 1 + i % 2, "AGE": 25 + i,
                "scorecard": .5, "challenger": i / 20})
        challenger_brier = math.fsum((i / 20 - i % 2) ** 2 for i in range(20)) / 20
        challenger_loss = -math.fsum(math.log(i / 20) if i % 2 else math.log1p(-i / 20)
                                    for i in range(20)) / 20
        self.summary_data = {
            "partitions": {"train": {"rows": 1, "groups": 1}, "calibration": {"rows": 1, "groups": 1},
                "policy_validation": {"rows": 1, "groups": 1}, "test": {"rows": 20, "groups": 19, "defaults": 10}},
            "models": {
                "scorecard": {"threshold": .5, "auc": .5, "brier": .25, "log_loss": math.log(2),
                    "false_negative": 0, "false_positive": 10, "accepted": 0, "cost_units": 10},
                "challenger": {"threshold": .5, "auc": .55, "brier": challenger_brier, "log_loss": challenger_loss,
                    "false_negative": 5, "false_positive": 5, "accepted": 10, "cost_units": 30},
            },
        }
        self.write_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def write_fixture(self):
        write_rows(self.predictions, PREDICTION_COLUMNS, self.prediction_rows)
        write_rows(self.manifest, MANIFEST_COLUMNS, self.manifest_rows)
        self.summary.write_text(json.dumps(self.summary_data))

    def run_review(self, name="review"):
        return audit(self.predictions, self.manifest, self.summary, self.root / name)

    def test_full_sql_population_metric_and_tie_queue_reconciliation(self):
        result = self.run_review()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(row["violations"] == 0 for row in result["controls"]))
        metrics = result["independent_metrics"]
        self.assertAlmostEqual(metrics["scorecard"]["average_precision"], .5)
        self.assertAlmostEqual(metrics["challenger"]["average_precision"],
                               sum(k / (2 * k - 1) for k in range(1, 11)) / 10)
        with (self.root / "review" / "decision_queues.csv").open() as stream:
            queues = list(csv.DictReader(stream))
        self.assertEqual(len(queues), len(MODELS) * len(CAPACITIES))
        scorecard = {float(row["capacity_fraction"]): row for row in queues if row["model"] == "scorecard"}
        self.assertEqual(scorecard[0.0]["cutoff_probability"], "")
        self.assertEqual(scorecard[0.0]["cost_units"], "50")
        self.assertEqual(scorecard[.2]["flagged"], "4")
        self.assertEqual(scorecard[.2]["defaults_flagged"], "2")
        self.assertEqual(scorecard[.2]["tied_at_cutoff"], "20")
        self.assertEqual(scorecard[1.0]["cost_units"], "10")
        # Database is an actual usable report artifact, not a rendered SQL claim.
        with sqlite3.connect(self.root / "review" / "card_review.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 40)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM manifest").fetchone()[0], 23)
        # Repeated evidence generation is byte stable for report tables and JSON.
        self.run_review("again")
        for path in (self.root / "review").iterdir():
            if path.suffix in (".json", ".csv"):
                self.assertEqual(path.read_bytes(), (self.root / "again" / path.name).read_bytes())

    def test_exact_customer_and_group_identity_is_required(self):
        mutations = [lambda: self.prediction_rows.pop(),
                     lambda: self.prediction_rows.append(dict(self.prediction_rows[0])),
                     lambda: self.prediction_rows[0].update(source_id=1),
                     lambda: self.prediction_rows[0].update(feature_group="999"),
                     lambda: self.manifest_rows[0].update(feature_group="200"),
                     lambda: self.manifest_rows[0].update(partition="development"),
                     lambda: self.manifest_rows.append(dict(self.manifest_rows[0]))]
        original_predictions = json.dumps(self.prediction_rows)
        original_manifest = json.dumps(self.manifest_rows)
        for mutate in mutations:
            self.prediction_rows = json.loads(original_predictions)
            self.manifest_rows = json.loads(original_manifest)
            mutate()
            self.write_fixture()
            with self.assertRaises(ValueError):
                self.run_review()

    def test_csv_value_guards_reject_invalid_not_coerced_fields(self):
        mutations = [("default", "0.5"), ("default", "2"), ("source_id", "4.0"),
                     ("source_id", "-4"), ("AGE", "nan"), ("SEX_for_audit_only", "3"),
                     ("feature_group", "1.4"), ("scorecard", "inf"), ("scorecard", "nan"),
                     ("scorecard", "1.01"), ("challenger", ""), ("scorecard", "oops")]
        original = dict(self.prediction_rows[0])
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                self.prediction_rows[0] = {**original, key: value}
                self.write_fixture()
                with self.assertRaises(ValueError):
                    self.run_review()

    def test_known_reference_counts_and_metrics_cannot_be_silently_changed(self):
        for model, key, value in [("scorecard", "false_positive", 9),
                                  ("scorecard", "auc", .7),
                                  ("challenger", "threshold", math.nan),
                                  ("challenger", "cost_units", 29)]:
            original = self.summary_data["models"][model][key]
            self.summary_data["models"][model][key] = value
            self.write_fixture()
            with self.assertRaises(ValueError):
                self.run_review()
            self.summary_data["models"][model][key] = original

    def test_queue_parity_rejects_different_capacity_or_tie_result(self):
        self.run_review()
        queue = self.root / "review" / "decision_queues.csv"
        # A correct separately supplied queue file passes all count/rate checks.
        result = audit(self.predictions, self.manifest, self.summary, self.root / "parity", queue)
        self.assertTrue(result["queue_reference_checked"])
        with queue.open() as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        rows[-1]["defaults_flagged"] = "9"
        write_rows(queue, columns, rows)
        with self.assertRaises(ValueError):
            audit(self.predictions, self.manifest, self.summary, self.root / "bad", queue)

    def test_exact_assignment_parity_detects_same_count_wrong_customer(self):
        assignments = []
        for model in MODELS:
            order = sorted(self.prediction_rows, key=lambda row: (-row[model], row["source_id"]))
            for capacity in CAPACITIES:
                selected = {row["source_id"] for row in order[:math.floor(capacity * len(order))]}
                for row in self.prediction_rows:
                    assignments.append({"source_id": row["source_id"], "model": model,
                        "capacity_fraction": capacity, "flagged": int(row["source_id"] in selected)})
        path = self.root / "queue_assignments.csv"
        columns = ("source_id", "model", "capacity_fraction", "flagged")
        write_rows(path, columns, assignments)
        result = audit(self.predictions, self.manifest, self.summary, self.root / "assignments",
                       queue_assignments=path)
        self.assertEqual(result["queue_assignments_compared"], 20 * 2 * len(CAPACITIES))
        # Move one tied review flag between customers with the same outcome.
        # Aggregate captured defaults and scenario cost remain unchanged.
        for row in assignments:
            if row["model"] == "scorecard" and row["capacity_fraction"] == .1:
                if row["source_id"] == 4:
                    row["flagged"] = 0
                if row["source_id"] == 6:
                    row["flagged"] = 1
        write_rows(path, columns, assignments)
        with self.assertRaises(ValueError):
            audit(self.predictions, self.manifest, self.summary, self.root / "bad_assignments",
                  queue_assignments=path)

    def test_optimized_python_keeps_corruption_guards(self):
        self.prediction_rows[0]["scorecard"] = "inf"
        self.write_fixture()
        completed = subprocess.run([sys.executable, "-O", "-m", "card_audit",
            "--predictions", str(self.predictions), "--partition-manifest", str(self.manifest),
            "--summary", str(self.summary), "--output", str(self.root / "optimized")],
            cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("CARD_SQL_REVIEW_FAILED", completed.stderr)
        self.assertNotIn("CARD_SQL_REVIEW_PASS", completed.stdout)
        self.assertFalse((self.root / "optimized").exists())


if __name__ == "__main__":
    unittest.main()
