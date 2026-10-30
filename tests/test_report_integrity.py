"""Presentation must refuse unverified or contradictory quantitative evidence."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("research_presentation", ROOT / "tools/build_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


@pytest.mark.parametrize("value", [True, np.bool_(False), float("nan"), float("inf"), "0.1", None])
def test_invalid_numeric_evidence_cannot_become_a_reported_result(value):
    with pytest.raises(ValueError, match="Finite numeric evidence"):
        report.number(value, "example")


def market_evidence():
    return (pd.read_csv(ROOT / "reference/us_market/quarterly.csv"),
            json.loads((ROOT / "reference/us_market/report.json").read_text()))


def test_market_report_accepts_numpy_numeric_cells_and_verified_sources():
    frame, evidence = market_evidence()
    report.validate_market(frame, evidence)


def test_market_report_rejects_stale_latest_rate():
    frame, evidence = market_evidence()
    evidence["latest"]["dpd90_balances_pct"] += .01
    with pytest.raises(ValueError, match="latest metric mismatch"):
        report.validate_market(frame, evidence)


def test_market_report_rejects_percentage_point_change_with_wrong_sign():
    frame, evidence = market_evidence()
    evidence["year_on_year_indicators"][0]["yoy_change_pp"] *= -1
    with pytest.raises(ValueError, match="year-on-year indicator mismatch"):
        report.validate_market(frame, evidence)


def test_market_report_rejects_duplicate_quarters_instead_of_drawing_them():
    frame, evidence = market_evidence()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Unique YYYYQn"):
        report.validate_market(frame, evidence)


def test_failed_sql_audit_cannot_be_presented_as_reconciled():
    with pytest.raises(ValueError, match="audit must pass"):
        report.validate_evidence({}, pd.DataFrame(), pd.DataFrame(), {"status": "FAIL"})


def test_uncompared_queue_cannot_be_presented_as_sql_reconciled():
    with pytest.raises(ValueError, match="queue reconciliation"):
        report.validate_evidence({}, pd.DataFrame(), pd.DataFrame(), {
            "status": "PASS", "controls": [{"control": "population", "violations": 0}],
            "queue_reference_checked": False,
        })
