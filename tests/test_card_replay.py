"""Adversarial checks of the published replay contract, without claiming refits."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from consumer_cards.constants import HISTORICAL, HISTORICAL_HASHES, PROTOCOL_SHA256, SOURCE_SHA256
from consumer_cards.data import load_protocol, load_source, sha256
from consumer_cards.replay import CSV_NAMES, dump, historical_integrity, verify_replay, write_queue_outputs


@pytest.fixture(scope="module")
def replay_fixture(tmp_path_factory):
    folder = tmp_path_factory.mktemp("artifact-comparison-fixture")
    for name in (*CSV_NAMES, "summary.json"):
        shutil.copyfile(HISTORICAL / name, folder / name)
    dump(folder / "source_audit.json", {"test_fixture": "Artifact verification tests; no actual model fitting claimed"})
    write_queue_outputs(folder / "holdout_predictions.csv", folder)
    import consumer_cards.replay
    implementation = {path.name: sha256(path) for path in sorted(Path(consumer_cards.replay.__file__).parent.glob("*.py"))}
    dump(folder / "replay_receipt.json", {
        "source_sha256": SOURCE_SHA256, "protocol_sha256": PROTOCOL_SHA256,
        "implementation_sha256": implementation,
        "file_sha256": {path.name: sha256(path) for path in sorted(folder.iterdir()) if path.is_file()},
    })
    return folder


@pytest.fixture
def current(replay_fixture, tmp_path):
    folder = tmp_path / "current"
    shutil.copytree(replay_fixture, folder)
    return folder


def rebind_current_file_hashes(folder):
    """Defeat simple hash-only guards so tests exercise scientific comparison."""
    summary = json.loads((folder / "summary.json").read_text())
    summary["file_sha256"] = {name: sha256(folder / name) for name in CSV_NAMES}
    dump(folder / "summary.json", summary)
    receipt = json.loads((folder / "replay_receipt.json").read_text())
    receipt["file_sha256"] = {name: sha256(folder / name) for name in receipt["file_sha256"]}
    dump(folder / "replay_receipt.json", receipt)


def test_immutable_historical_sources_match_recorded_bytes():
    assert historical_integrity() == HISTORICAL_HASHES
    assert load_protocol()["xls_sha256"] == SOURCE_SHA256


def test_packaged_protocol_is_the_exact_original():
    import consumer_cards.data
    packaged = Path(consumer_cards.data.__file__).parent / "resources" / "PROTOCOL.json"
    assert packaged.read_bytes() == (HISTORICAL / "PROTOCOL.json").read_bytes()


def test_source_manifest_package_and_checkout_mirror_agree():
    import consumer_cards.data
    root = Path(consumer_cards.data.__file__).parents[1]
    packaged = json.loads((root / "consumer_cards/resources/source_manifest.json").read_text())
    assert packaged == json.loads((root / "data/card_source_manifest.json").read_text())
    assert packaged["source_xls_sha256"] == SOURCE_SHA256
    assert packaged["historical_study"]["protocol_sha256"] == PROTOCOL_SHA256


def test_complete_reference_artifact_fixture_passes(current):
    result = verify_replay(current)
    assert result["passed"] and result["csv_cells_compared"] == 132267
    assert result["scientific_summary_fields_compared"] == 87


@pytest.mark.parametrize("section,key", [("calibration", "intercept"), ("calibration", "slope"), ("models", "brier")])
def test_scientific_comparison_rejects_tampered_metrics_even_after_rehash(current, section, key):
    summary = json.loads((current / "summary.json").read_text())
    summary[section]["scorecard"][key] += 0.01
    dump(current / "summary.json", summary)
    rebind_current_file_hashes(current)
    with pytest.raises(ValueError, match="Numerical comparison"):
        verify_replay(current)


@pytest.mark.parametrize("measure", ["auc", "brier", "cost_per_client"])
def test_every_paired_interval_is_verified(current, measure):
    summary = json.loads((current / "summary.json").read_text())
    summary["paired_uncertainty"][measure]["ci95"][1] += 0.05
    dump(current / "summary.json", summary)
    rebind_current_file_hashes(current)
    with pytest.raises(ValueError, match="Numerical comparison"):
        verify_replay(current)


@pytest.mark.parametrize("filename,column", [
    ("holdout_predictions.csv", "challenger"),
    ("calibration_bins.csv", "default_rate"),
    ("group_audit.csv", "false_negative_rate"),
])
def test_prediction_calibration_and_group_cells_checked_after_rehash(current, filename, column):
    frame = pd.read_csv(current / filename, dtype={"feature_group": str})
    frame.loc[0, column] += 0.05
    frame.to_csv(current / filename, index=False)
    rebind_current_file_hashes(current)
    with pytest.raises(ValueError, match="CSV difference"):
        verify_replay(current)


def test_partition_identity_cannot_be_rounded_or_tolerated(current):
    frame = pd.read_csv(current / "partition_manifest.csv", dtype={"feature_group": str})
    frame.loc[0, "source_id"] += 1
    frame.to_csv(current / "partition_manifest.csv", index=False)
    rebind_current_file_hashes(current)
    with pytest.raises(ValueError, match="Exact CSV"):
        verify_replay(current)


def test_missing_new_queue_artifact_is_not_silently_ignored(current):
    (current / "decision_queue.csv").unlink()
    with pytest.raises(FileNotFoundError):
        verify_replay(current)


def test_recorded_implementation_mapping_is_enforced(current):
    receipt = json.loads((current / "replay_receipt.json").read_text())
    receipt["implementation_sha256"]["models.py"] = "0" * 64
    dump(current / "replay_receipt.json", receipt)
    with pytest.raises(ValueError, match="implementation identity"):
        verify_replay(current)


def test_replay_artifact_hash_is_checked_before_scientific_comparison(current):
    with (current / "holdout_predictions.csv").open("a") as stream:
        stream.write("altered\n")
    with pytest.raises(ValueError, match="artifact hash"):
        verify_replay(current)


def test_portable_float_tolerance_is_explicit_and_narrow(current):
    summary = json.loads((current / "summary.json").read_text())
    summary["models"]["scorecard"]["brier"] += 1e-8
    dump(current / "summary.json", summary)
    rebind_current_file_hashes(current)
    assert verify_replay(current)["absolute_tolerance"] == 1e-7


def test_historical_reference_tampering_fails_even_if_its_summary_is_consistent(tmp_path):
    historical = tmp_path / "historical"
    shutil.copytree(HISTORICAL, historical)
    protocol = json.loads((historical / "PROTOCOL.json").read_text())
    protocol["seed"] += 1
    dump(historical / "PROTOCOL.json", protocol)
    with pytest.raises(ValueError, match="Historical reference hash"):
        historical_integrity(historical)


def test_queue_writer_cannot_overwrite_historical_references():
    with pytest.raises(ValueError, match="immutable historical"):
        write_queue_outputs(HISTORICAL / "holdout_predictions.csv", HISTORICAL)


def test_source_checksum_guard_survives_optimized_python(tmp_path):
    source = tmp_path / "wrong-source.xls"
    source.write_bytes(b"not the official historical source")
    command = [sys.executable, "-O", "-c", "from pathlib import Path; from consumer_cards.data import load_source; load_source(Path(__import__('sys').argv[1]))", str(source)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Source XLS hash mismatch" in result.stderr


def test_interval_verification_survives_optimized_python(current):
    summary = json.loads((current / "summary.json").read_text())
    summary["paired_uncertainty"]["cost_per_client"]["ci95"][0] = -99.0
    dump(current / "summary.json", summary)
    rebind_current_file_hashes(current)
    result = subprocess.run([sys.executable, "-O", "-m", "consumer_cards", "verify", "--output", str(current)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Numerical comparison failed" in result.stderr
    assert '"passed": true' not in result.stdout
