"""Small real fits check information boundaries and exact explanations."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from consumer_cards.constants import FEATURES, TARGET
from consumer_cards.models import fit_models


@pytest.fixture(scope="module")
def fitted_synthetic():
    rng = np.random.default_rng(82341)
    count = 500
    frame = pd.DataFrame({name: rng.integers(0, 10, count) for name in FEATURES})
    frame["LIMIT_BAL"] = np.arange(1, count + 1) * 1000
    frame["AGE"] = rng.integers(21, 75, count)
    for name in FEATURES:
        if name.startswith("BILL_AMT"):
            frame[name] = rng.integers(-1000, 100000, count)
        elif name.startswith("PAY_AMT"):
            frame[name] = rng.integers(0, 10000, count)
    frame[TARGET] = rng.integers(0, 2, count)
    return frame, fit_models(frame)


def test_changing_test_outcomes_cannot_change_fits_calibration_or_thresholds(fitted_synthetic):
    frame, original = fitted_synthetic
    changed = frame.copy()
    index = original.parts["test"]
    changed.loc[index, TARGET] = 1 - changed.loc[index, TARGET]
    refitted = fit_models(changed)
    for name in original.parts:
        np.testing.assert_array_equal(original.parts[name], refitted.parts[name])
    X = frame.iloc[index][list(FEATURES)]
    for name in ("scorecard", "challenger"):
        np.testing.assert_array_equal(original.raw_predictions(X)[name], refitted.raw_predictions(X)[name])
        np.testing.assert_array_equal(original.predict(X)[name], refitted.predict(X)[name])
    assert original.thresholds == refitted.thresholds
    assert original.calibration_info == refitted.calibration_info


def test_calibrated_scorecard_contributions_reconstruct_actual_probability(fitted_synthetic):
    frame, fitted = fitted_synthetic
    X = frame.iloc[:30][list(FEATURES)]
    explanation = fitted.scorecard_contributions(X)
    contribution_columns = [name for name in explanation if name.startswith("log_odds_")]
    assert len(contribution_columns) == 20
    reconstructed = explanation[contribution_columns].sum(axis=1) + explanation.calibrated_intercept
    np.testing.assert_allclose(reconstructed, explanation.calibrated_log_odds, rtol=0, atol=1e-12)
    np.testing.assert_allclose(expit(reconstructed), fitted.predict(X)["scorecard"], rtol=0, atol=1e-12)


def test_model_identity_changes_when_the_fit_environment_changes(monkeypatch):
    from consumer_cards import cli
    original = cli.model_identity()
    environment = {**cli.scientific_environment(), "sklearn": "different-tested-environment"}
    monkeypatch.setattr(cli, "scientific_environment", lambda: environment)
    assert cli.model_identity() != original
