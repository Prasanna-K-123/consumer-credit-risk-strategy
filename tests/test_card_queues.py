"""Capacity, ordering and accounting properties of the review queue."""
import numpy as np
import pytest

from consumer_cards.evaluation import decision_queue


def sample():
    # Two tied scores at each of the first two nontrivial cutoffs.
    return np.array([1, 0, 1, 0, 1]), np.array([0.7, 0.9, 0.9, 0.2, 0.7]), np.array([50, 20, 10, 40, 30])


@pytest.mark.parametrize("capacity, expected", [
    (0, 0), (0.19, 0), (0.2, 1), (0.39, 1), (0.4, 2),
    (0.59, 2), (0.6, 3), (0.99, 4), (1, 5),
])
def test_queue_selects_exact_floor_capacity_without_rounding_up(capacity, expected):
    labels, scores, identifiers = sample()
    report, flagged = decision_queue(labels, scores, identifiers, capacity)
    assert flagged.dtype == np.bool_
    assert flagged.shape == (5,)
    assert int(flagged.sum()) == expected
    assert report["capacity_count"] == report["flagged"] == expected
    assert report["flag_rate"] == expected / 5


def test_ties_use_original_identifier_and_accounting_matches_manual_ledger():
    labels, scores, identifiers = sample()
    report, flagged = decision_queue(labels, scores, identifiers, 0.6)
    np.testing.assert_array_equal(flagged, [False, True, True, False, True])
    assert set(identifiers[flagged]) == {10, 20, 30}
    assert report["cutoff_probability"] == 0.7
    assert report["tied_at_cutoff"] == 2
    assert report["defaults_flagged"] == 2
    assert report["nondefaults_flagged"] == 1
    assert report["defaults_unflagged"] == 1
    assert report["capture_rate"] == pytest.approx(2 / 3)
    assert report["precision"] == pytest.approx(2 / 3)
    # The ledger is one missed default (5 units) and one flagged nondefault (1).
    assert report["cost_units"] == 6


def test_tie_break_is_not_input_row_order():
    labels = [0, 1, 0, 1]
    report, flagged = decision_queue(labels, [0.5] * 4, [40, 10, 30, 20], 0.5)
    np.testing.assert_array_equal(flagged, [False, True, False, True])
    assert report["tied_at_cutoff"] == 4
    assert report["cutoff_probability"] == 0.5


@pytest.mark.parametrize("order", [[4, 3, 2, 1, 0], [2, 0, 4, 1, 3], [1, 4, 0, 3, 2]])
def test_permuting_batch_preserves_selected_clients_and_report(order):
    labels, scores, identifiers = sample()
    expected_report, expected_flags = decision_queue(labels, scores, identifiers, 0.6)
    report, flagged = decision_queue(labels[order], scores[order], identifiers[order], 0.6)
    assert set(identifiers[order][flagged]) == set(identifiers[expected_flags])
    assert report == expected_report


def test_ranking_does_not_use_outcomes_or_mutate_inputs():
    labels, scores, identifiers = sample()
    originals = [value.copy() for value in (labels, scores, identifiers)]
    original_report, original_flags = decision_queue(labels, scores, identifiers, 0.6)
    changed_report, changed_flags = decision_queue(1 - labels, scores, identifiers, 0.6)
    np.testing.assert_array_equal(changed_flags, original_flags)
    assert original_report["cost_units"] == 6
    assert changed_report["cost_units"] == 7
    for supplied, original in zip((labels, scores, identifiers), originals):
        np.testing.assert_array_equal(supplied, original)


def test_increasing_capacity_produces_nested_queues():
    labels, scores, identifiers = sample()
    previous = set()
    for capacity in (0, 0.2, 0.4, 0.6, 0.8, 1):
        _, flagged = decision_queue(labels, scores, identifiers, capacity)
        current = set(identifiers[flagged])
        assert previous.issubset(current)
        previous = current
    assert previous == set(identifiers)


@pytest.mark.parametrize("label, capacity, cost_units, capture, precision", [
    (0, 0, 0, None, None), (0, 0.5, 2, None, 0), (0, 1, 4, None, 0),
    (1, 0, 20, 0, None), (1, 0.5, 10, 0.5, 1), (1, 1, 0, 1, 1),
])
def test_zero_full_and_one_class_queues_have_defined_accounting(label, capacity, cost_units, capture, precision):
    report, flagged = decision_queue([label] * 4, [0.1, 0.9, 0.7, 0.4], [3, 4, 2, 1], capacity)
    assert report["cost_units"] == cost_units
    assert report["capture_rate"] == capture
    assert report["precision"] == precision
    if capacity == 0:
        assert not flagged.any()
        assert report["cutoff_probability"] is None
        assert report["tied_at_cutoff"] == 0
    if capacity == 1:
        assert flagged.all()


@pytest.mark.parametrize("capacity", [-0.001, 1.001, np.inf, -np.inf, np.nan, [0.5], "0.5"])
def test_queue_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        decision_queue([0, 1], [0.2, 0.8], [1, 2], capacity)


@pytest.mark.parametrize("labels, scores, identifiers", [
    ([0, 1], [0.2, 0.8], [1, 1]),
    ([0, 1], [0.2, 0.8], [1.0, 2.0]),
    ([0, 1], [0.2, 0.8], [-1, 2]),
    ([0, 1], [0.2, 0.8], [1]),
    ([0, 1], [[0.2], [0.8]], [1, 2]),
    ([0.1, 1], [0.2, 0.8], [1, 2]),
    ([0, 1], [0.2, np.inf], [1, 2]),
    ([], [], []),
])
def test_queue_rejects_invalid_batch_before_ranking(labels, scores, identifiers):
    with pytest.raises(ValueError):
        decision_queue(labels, scores, identifiers, 0.5)
