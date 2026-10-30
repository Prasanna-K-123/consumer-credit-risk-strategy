"""Independent checks must detect contradictory evidence even after rehashing."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from tools import audit_strategy as review

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence(tmp_path):
    # Published source checkouts carry these executed references. The fallback
    # only supports the first local development run before reference promotion.
    source = ROOT / "reference/card_strategy"
    if not source.is_dir():
        source = ROOT / "work/strategy-current"
    target = tmp_path / "strategy"
    shutil.copytree(source, target)
    return target


def change_csv(folder, name, mutate, *, rehash=True):
    path = folder / name
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    mutate(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if rehash:
        receipt_path = folder / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["file_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt))


def test_independent_audit_reconciles_complete_recorded_evidence(evidence):
    result = review.audit(evidence)
    assert result["status"] == "PASS"
    assert result["policy_scenarios"] == 15
    assert result["capacity_scenarios"] == 9
    assert result["population_records"] == 30000
    assert result["source_labels_reconciled"] is False


def test_unrecorded_byte_change_fails_receipt(evidence):
    with (evidence / "tree.txt").open("a") as stream:
        stream.write("unrecorded change\n")
    with pytest.raises(ValueError, match="receipt"):
        review.audit(evidence)


@pytest.mark.parametrize("column,value,match", [
    ("source_id", "2", "manifest"),
    ("default", "0.5", "integer"),
    ("cart", "nan", "Finite"),
    ("cart_raw", "1.01", "Finite"),
])
def test_rehashed_invalid_customer_evidence_is_rejected(evidence, column, value, match):
    change_csv(evidence, "predictions.csv", lambda rows: rows[0].__setitem__(column, value))
    with pytest.raises(ValueError, match=match):
        review.audit(evidence)


def test_missing_policy_scenario_cannot_hide_behind_pass_status(evidence):
    change_csv(evidence, "policy_sensitivity.csv", lambda rows: rows.pop())
    with pytest.raises(ValueError, match="scenario keys"):
        review.audit(evidence)


def test_rehashed_policy_selection_error_is_found_by_sql(evidence):
    change_csv(evidence, "policy_sensitivity.csv", lambda rows: rows[0].__setitem__("selected_threshold", "0.99"))
    with pytest.raises(ValueError, match="selected_threshold"):
        review.audit(evidence)


def test_rehashed_capacity_result_error_is_found_by_ordered_scores(evidence):
    change_csv(evidence, "capacity.csv", lambda rows: rows[0].__setitem__("defaults_flagged", "0"))
    with pytest.raises(ValueError, match="defaults_flagged"):
        review.audit(evidence)


def test_rehashed_leaf_rate_error_is_found_from_source_rows(evidence):
    change_csv(evidence, "leaf_segments.csv", lambda rows: rows[0].__setitem__("default_rate", "0.999"))
    with pytest.raises(ValueError, match="default_rate"):
        review.audit(evidence)


def test_verification_population_claim_is_recomputed(evidence):
    path = evidence / "verification.json"
    recorded = json.loads(path.read_text())
    recorded["evaluation_records"] = 1
    path.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="verification evaluation_records"):
        review.audit(evidence)


def test_equally_good_policy_threshold_still_must_obey_tie_rule(evidence):
    # Enumerate CART policy costs directly in Python, independently of the
    # audit's SQL cross-join. An equal-cost alternate is not the recorded rule.
    with (evidence / "predictions.csv").open(newline="") as stream:
        selected = [(int(r["default"]), float(r["cart"])) for r in csv.DictReader(stream)
                    if r["partition"] == "policy_validation"]
    alternative = None
    for ratio in review.RATIOS:
        costs = [(sum(ratio * (y == 1 and p < t / 100) + (y == 0 and p >= t / 100)
                      for y, p in selected), t / 100) for t in range(1, 100)]
        minimum = min(c for c, _ in costs)
        tied = [t for c, t in costs if c == minimum]
        if len(tied) > 1:
            alternative = ratio, tied[-1]
            break
    assert alternative is not None
    ratio, alternate_threshold = alternative
    def change(rows):
        row = next(r for r in rows if r["model"] == "cart" and float(r["missed_default_cost"]) == ratio)
        assert float(row["selected_threshold"]) < alternate_threshold
        row["selected_threshold"] = str(alternate_threshold)
    change_csv(evidence, "policy_sensitivity.csv", change)
    with pytest.raises(ValueError, match="selected_threshold"):
        review.audit(evidence)
