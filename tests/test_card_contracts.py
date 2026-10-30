"""Adversarial input contracts and independent credit-cost accounting."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from consumer_cards.constants import FEATURES
from consumer_cards.contracts import aligned, binary_labels, feature_frame, probabilities, source_ids
from consumer_cards.evaluation import cost, metrics, paired_group_bootstrap, select_threshold


@pytest.mark.parametrize("values", [
    [np.nan], [np.inf], [-np.inf], [-0.001], [1.001], [],
    [[0.2], [0.8]], 0.5, ["0.2", "0.8"], [0.2 + 0.1j],
])
def test_probabilities_reject_invalid_values_before_clipping_or_conversion(values):
    with pytest.raises(ValueError):
        probabilities(values)


@pytest.mark.parametrize("values", [[0.5, 1], [0, 2], [-1, 1], [np.nan, 1], [np.inf, 0], ["0", "1"], [[0], [1]], []])
def test_labels_reject_fractional_nonbinary_and_malformed_inputs(values):
    with pytest.raises(ValueError):
        binary_labels(values)


def test_valid_probability_endpoints_and_binary_labels_are_preserved():
    np.testing.assert_array_equal(probabilities([0, 0.5, 1]), [0, 0.5, 1])
    np.testing.assert_array_equal(binary_labels([0.0, 1.0, 0.0]), [0, 1, 0])


@pytest.mark.parametrize("labels, scores", [([0, 1], [[0.1], [0.9]]), ([[0], [1]], [0.1, 0.9]), ([0, 1], [0.1]), ([], [])])
def test_cost_refuses_shapes_that_numpy_could_broadcast(labels, scores):
    with pytest.raises(ValueError):
        aligned(labels, scores)
    with pytest.raises(ValueError):
        cost(labels, scores, 0.5)
    with pytest.raises(ValueError):
        select_threshold(labels, scores)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, np.nan, np.inf, -np.inf, [0.5], "0.5"])
def test_cost_and_metrics_refuse_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        cost([0, 1], [0.2, 0.8], threshold)
    with pytest.raises(ValueError):
        metrics([0, 1], [0.2, 0.8], threshold)


def test_confusion_cost_matches_independently_enumerated_cases():
    # TN, FP at equality, FN, TP: equality belongs to the flagged side.
    np.testing.assert_array_equal(cost([0, 0, 1, 1], [0.1, 0.5, 0.1, 0.9], 0.5), [0, 1, 5, 0])
    np.testing.assert_array_equal(cost([0, 1], [0, 1], 0), [1, 0])
    np.testing.assert_array_equal(cost([0, 1], [0, 1], 1), [0, 0])
    # One-class batches have well-defined cost even though AUC is undefined.
    np.testing.assert_array_equal(cost([1, 1], [0.1, 0.9], 0.5), [5, 0])
    np.testing.assert_array_equal(cost([0, 0], [0.1, 0.9], 0.5), [0, 1])


def test_threshold_selection_uses_lowest_grid_threshold_on_a_cost_tie():
    # At 0.01 through 0.20 the one default is flagged and the nondefault is not.
    threshold, total = select_threshold([0, 1], [0, 0.2])
    assert threshold == 0.01
    assert total == 0


def test_reported_metrics_have_direct_cost_and_probability_oracles():
    result = metrics([0, 0, 1, 1], [0.1, 0.5, 0.1, 0.9], 0.5)
    assert result["false_negative"] == 1
    assert result["false_positive"] == 1
    assert result["accepted"] == 2
    assert result["cost_units"] == 6
    assert result["brier"] == pytest.approx(0.27)


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_auc_metrics_require_both_classes(labels):
    with pytest.raises(ValueError):
        metrics(labels, [0.1, 0.9], 0.5)


@pytest.mark.parametrize("values", [[np.inf, 0.5], [-np.inf, 0.5], [-0.1, 0.5], [1.1, 0.5], [np.nan, 0.5], [[0.1], [0.9]]])
def test_logit_conversion_does_not_hide_invalid_inputs(values):
    from consumer_cards.models import logits

    with pytest.raises(ValueError):
        logits(values)


def test_logit_endpoint_handling_is_finite_and_retains_order():
    from consumer_cards.models import logits

    converted = logits([0, 0.2, 0.5, 0.8, 1])
    assert converted.shape == (5, 1)
    assert np.isfinite(converted).all()
    assert np.all(np.diff(converted[:, 0]) > 0)
    assert converted[2, 0] == pytest.approx(0)


@pytest.mark.parametrize("labels, scores", [
    ([0, 0], [0.1, 0.2]), ([1, 1], [0.8, 0.9]),
    ([0, 2], [0.1, 0.9]), ([0.2, 1], [0.1, 0.9]),
    ([0, 1], [0.1]), ([0, 1], [[0.1], [0.9]]),
])
def test_calibration_requires_two_binary_classes_and_aligned_vectors(labels, scores):
    from consumer_cards.models import fit_calibration

    with pytest.raises(ValueError):
        fit_calibration(scores, labels)


def test_calibrator_can_fit_valid_binary_endpoint_observations():
    from consumer_cards.models import fit_calibration, logits

    model = fit_calibration([0, 0.2, 0.8, 1], [0, 0, 1, 1])
    prediction = model.predict_proba(logits([0.1, 0.9]))[:, 1]
    assert np.isfinite(prediction).all()
    assert 0 < prediction[0] < prediction[1] < 1


@pytest.mark.parametrize("identifiers, count", [([1, 1], 2), ([1.0, 2.0], 2), ([-1, 2], 2), ([1], 2), ([[1], [2]], 2)])
def test_source_ids_reject_ambiguous_ties_and_misalignment(identifiers, count):
    with pytest.raises(ValueError):
        source_ids(identifiers, count)


def test_feature_contract_reorders_without_mutating_input_or_adding_targets():
    original = pd.DataFrame({name: [1, 2] for name in reversed(FEATURES)})
    untouched = original.copy(deep=True)
    checked = feature_frame(original)
    assert list(checked.columns) == list(FEATURES)
    pd.testing.assert_frame_equal(original, untouched)
    checked.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(original, untouched)
    with pytest.raises(ValueError):
        feature_frame(original.assign(default=[0, 1]))
    with pytest.raises(ValueError):
        feature_frame(original.drop(columns=[FEATURES[0]]))
    invalid = original.astype(float)
    invalid.loc[0, FEATURES[0]] = np.inf
    with pytest.raises(ValueError):
        feature_frame(invalid)


def test_paired_bootstrap_of_identical_models_has_zero_contrasts():
    # Each group contains both classes, so every group-resampled draw does too.
    result = paired_group_bootstrap(
        np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.3, 0.7]),
        np.array([0.2, 0.8, 0.3, 0.7]), 0.5, 0.5,
        np.array([10, 10, 20, 20]), n=12,
    )
    assert set(result) == {"auc", "brier", "cost_per_client"}
    for comparison in result.values():
        assert comparison["challenger_minus_scorecard"] == 0
        assert comparison["ci95"] == [0, 0]


@pytest.mark.parametrize("groups, repetitions", [([1], 10), ([[1], [2]], 10), ([1, 2], 1), ([1, 2], 2.5), ([1, 2], True)])
def test_paired_bootstrap_rejects_bad_group_or_repetition_contract(groups, repetitions):
    with pytest.raises(ValueError):
        paired_group_bootstrap([0, 1], [0.1, 0.9], [0.2, 0.8], 0.5, 0.5, groups, n=repetitions)


def test_contract_guards_survive_optimized_python():
    program = """
import numpy as np
from consumer_cards.evaluation import cost, decision_queue
from consumer_cards.models import logits
cases = [
    lambda: logits([np.inf]),
    lambda: cost([0.25, 1], [0.1, 0.9], 0.5),
    lambda: cost([0, 1], [[0.1], [0.9]], 0.5),
    lambda: decision_queue([0, 1], [0.1, 0.9], [1, 2], np.nan),
]
for index, operation in enumerate(cases):
    try:
        operation()
    except ValueError:
        continue
    raise SystemExit('Contract accepted invalid case %s under -O' % index)
"""
    completed = subprocess.run(
        [sys.executable, "-O", "-c", program],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
