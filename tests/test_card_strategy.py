"""Independent decision arithmetic and information-boundary checks."""
import json

import numpy as np
import pandas as pd
import pytest

from consumer_cards.constants import FEATURES, ROOT, TARGET
from consumer_cards.data import feature_groups, partition
from card_strategy.analysis import confusion, fit_tree, leaf_paths, leaf_segments, policy_sensitivity, select_cost_threshold, validate_parts
from card_strategy.pipeline import fresh_output, load_strategy_protocol


def test_cost_selection_matches_hand_computed_policy_scenarios():
    # At 1:1, flagging just the .8 case costs one missed default.
    # At 5:1, flagging everyone costs two false alarms, below any miss.
    y, p = [1, 0, 0, 1], [.8, .7, .6, .1]
    assert select_cost_threshold(y, p, 1) == (.71, 1.0)
    assert select_cost_threshold(y, p, 5) == (.01, 2.0)
    # Tied minimum costs use the lowest grid threshold explicitly.
    assert select_cost_threshold([0, 1], [.2, .8], 1)[0] == .21


def test_inclusive_threshold_and_confusion_conservation():
    result = confusion([1, 0, 1, 0], [.5, .5, .49, .01], .5, 10)
    assert [result[name] for name in ("true_positive", "false_positive", "false_negative", "true_negative")] == [1, 1, 1, 1]
    assert result["cost_units"] == 11
    assert result["reviewed"] == 2
    assert result["default_capture"] == .5
    assert result["queue_default_rate"] == .5


def test_evaluation_outcomes_cannot_choose_cost_thresholds():
    policy_y, policy_p = [1, 0, 0, 1], [.8, .7, .6, .1]
    first = policy_sensitivity(policy_y, policy_p, [1, 1, 0, 0], [.9, .8, .2, .1])
    changed = policy_sensitivity(policy_y, policy_p, [0, 0, 1, 1], [.9, .8, .2, .1])
    np.testing.assert_array_equal(first.selected_threshold, changed.selected_threshold)
    np.testing.assert_array_equal(first.policy_validation_cost_units, changed.policy_validation_cost_units)
    assert first.loc[0, "cost_units"] != changed.loc[0, "cost_units"]
    assert (first.review_all_cost_units == 2).all()
    np.testing.assert_array_equal(first.review_none_cost_units, [2, 4, 10, 20, 40])
    assert list(first.missed_default_cost) == [1, 2, 5, 10, 20]


@pytest.mark.parametrize("ratio", [0, -1, np.inf, np.nan, True, "5", [5]])
def test_invalid_economic_assumptions_fail(ratio):
    with pytest.raises(ValueError):
        select_cost_threshold([0, 1], [.2, .8], ratio)


@pytest.mark.parametrize("labels,probability", [([0, .5], [.2, .8]), ([0, 1], [.2, 1.1]), ([0, 1], [.2, np.nan]), ([0, 1], [.2]), ([[0, 1]], [.2, .8])])
def test_invalid_outcomes_and_probabilities_fail_before_cost_selection(labels, probability):
    with pytest.raises(ValueError):
        select_cost_threshold(labels, probability, 5)


def test_sensitivity_rejects_empty_or_duplicate_scenarios():
    for ratios in ([], [1, 1]):
        with pytest.raises(ValueError):
            policy_sensitivity([0, 1], [.1, .9], [0, 1], [.1, .9], ratios)


@pytest.fixture(scope="module")
def synthetic_fit():
    rng = np.random.default_rng(94721)
    count = 1600
    frame = pd.DataFrame({name: rng.integers(0, 10, count) for name in FEATURES})
    frame["LIMIT_BAL"] = np.arange(count) * 1000
    frame["AGE"] = rng.integers(21, 75, count)
    labels = (frame.PAY_0 >= 5).astype(int).to_numpy()
    noise = rng.random(count) < .1
    labels[noise] = 1 - labels[noise]
    frame[TARGET] = labels
    groups = feature_groups(frame[list(FEATURES)])
    parts = partition(groups)
    return frame, parts, fit_tree(frame, parts)


def test_test_outcomes_cannot_change_tree_or_calibration(synthetic_fit):
    frame, parts, original = synthetic_fit
    changed = frame.copy()
    changed.loc[parts["test"], TARGET] = 1 - changed.loc[parts["test"], TARGET]
    refitted = fit_tree(changed, parts)
    X = frame[list(FEATURES)]
    np.testing.assert_array_equal(original.model.tree_.threshold, refitted.model.tree_.threshold)
    np.testing.assert_array_equal(original.raw_predictions(X), refitted.raw_predictions(X))
    np.testing.assert_array_equal(original.predict(X), refitted.predict(X))
    assert original.model.tree_.n_node_samples[0] == len(parts["train"])
    assert original.model.get_depth() <= 3
    leaf_counts = original.model.tree_.n_node_samples[original.model.tree_.children_left == -1]
    assert leaf_counts.min() >= 200


def test_policy_outcomes_cannot_change_tree_or_calibration(synthetic_fit):
    frame, parts, original = synthetic_fit
    changed = frame.copy()
    changed.loc[parts["policy_validation"], TARGET] = 1 - changed.loc[parts["policy_validation"], TARGET]
    refitted = fit_tree(changed, parts)
    np.testing.assert_array_equal(original.predict(frame[list(FEATURES)]), refitted.predict(frame[list(FEATURES)]))


def test_leaf_rules_and_counts_agree_with_actual_tree(synthetic_fit):
    frame, parts, fitted = synthetic_fit
    leaves = leaf_segments(frame, fitted)
    for role, index in parts.items():
        subset = leaves[leaves.partition == role]
        assert subset.n.sum() == len(index)
        assert subset.defaults.sum() == frame.iloc[index][TARGET].sum()
        for row in subset.itertuples():
            if row.n:
                assert row.default_rate == row.defaults / row.n
    paths = leaf_paths(fitted.model)
    predicted_leaf = fitted.model.apply(frame[list(FEATURES)])
    for leaf, rule in paths.items():
        rule_mask = np.ones(len(frame), dtype=bool)
        if rule != "all training clients":
            for clause in rule.split(" AND "):
                field, operator, value = clause.split()
                current = frame[field].to_numpy()
                rule_mask &= current <= float(value) if operator == "<=" else current > float(value)
        np.testing.assert_array_equal(rule_mask, predicted_leaf == leaf)


def test_partition_overlap_and_missing_rows_fail(synthetic_fit):
    frame, parts, _ = synthetic_fit
    duplicated = {name: index.copy() for name, index in parts.items()}
    duplicated["test"][0] = duplicated["train"][0]
    with pytest.raises(ValueError):
        validate_parts(frame, duplicated)
    missing = {name: index.copy() for name, index in parts.items()}
    missing["test"] = missing["test"][1:]
    with pytest.raises(ValueError):
        validate_parts(frame, missing)


def test_output_preserves_sources_references_and_existing_evidence(tmp_path):
    source = tmp_path / "input" / "data.xls"
    source.parent.mkdir()
    source.write_bytes(b"source")
    for destination in (source, source.parent, tmp_path, ROOT / "reference" / "card_strategy", ROOT):
        with pytest.raises(ValueError):
            fresh_output(destination, source)
    output = tmp_path / "results"
    assert fresh_output(output, source) == output
    (output / "receipt.json").write_text("previous")
    with pytest.raises(FileExistsError):
        fresh_output(output, source)
    assert source.read_bytes() == b"source"
    assert (output / "receipt.json").read_text() == "previous"


@pytest.mark.parametrize("directory", ["consumer_cards", "card_audit", "us_card_market", "card_strategy", "tools", "docs", "tests", "reports"])
def test_output_cannot_be_nested_inside_existing_project_sources(tmp_path, directory):
    source = tmp_path / "source.xls"
    source.write_bytes(b"source")
    with pytest.raises(ValueError):
        fresh_output(ROOT / directory / "new-output", source)


def test_protocol_states_retrospective_and_python_boundaries():
    protocol = load_strategy_protocol()
    assert protocol["tree_parameters"]["min_samples_leaf"] == 200
    assert "after the original historical outcomes were public" in protocol["scope"]
    assert "not SAS or CHAID" in protocol["scope"]


def test_protocol_corruption_requires_explicit_review(tmp_path, monkeypatch):
    import card_strategy.pipeline as pipeline
    protocol = load_strategy_protocol()
    protocol["scope"] = "Unjustified fresh evaluation claim"
    (tmp_path / "PROTOCOL.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(pipeline, "PACKAGE", tmp_path)
    with pytest.raises(ValueError, match="protocol hash"):
        load_strategy_protocol()
