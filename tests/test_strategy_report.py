"""A report may only use source-reconciled evidence bound to independent review."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("strategy_presentation", ROOT / "tools/build_strategy_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def evidence(tmp_path):
    target = tmp_path / "evidence"
    shutil.copytree(ROOT / "reference/card_strategy", target)
    return target


def review_change(folder, key, value):
    path = folder / "independent_review.json"
    content = json.loads(path.read_text())
    content[key] = value
    path.write_text(json.dumps(content))


@pytest.mark.parametrize("key,value", [("status", "FAIL"), ("source_labels_reconciled", False)])
def test_report_refuses_unreconciled_or_failed_review(tmp_path, key, value):
    folder = evidence(tmp_path)
    review_change(folder, key, value)
    with pytest.raises(ValueError, match="independent strategy review"):
        report.load(folder)


def test_report_refuses_post_review_changed_scenarios(tmp_path):
    folder = evidence(tmp_path)
    path = folder / "policy_sensitivity.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed since independent review"):
        report.load(folder)


def test_report_requires_complete_input_identity_binding(tmp_path):
    folder = evidence(tmp_path)
    review_change(folder, "input_sha256", {})
    with pytest.raises(ValueError, match="complete strategy evidence"):
        report.load(folder)
