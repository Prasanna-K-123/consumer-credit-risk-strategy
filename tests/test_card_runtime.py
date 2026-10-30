"""The library bounds native workers and returns the caller's configuration."""
import os

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from consumer_cards.runtime import NATIVE_THREAD_LIMIT, bounded_native_threads


def pool_settings():
    return {item["filepath"]: item["num_threads"] for item in threadpool_info()}


def test_native_thread_budget_is_two_and_restores_callers_limits():
    @bounded_native_threads
    def calculate():
        np.dot(np.ones((10, 10)), np.ones((10, 10)))
        return pool_settings()
    with threadpool_limits(limits=1):
        before = pool_settings()
        active = calculate()
        assert active and all(count == NATIVE_THREAD_LIMIT == 2 for count in active.values())
        assert pool_settings() == before


def test_native_thread_budget_restores_after_exception_and_does_not_mutate_environment():
    @bounded_native_threads
    def fails():
        raise RuntimeError("deliberate numerical-work failure")
    environment = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    with threadpool_limits(limits=1):
        before = pool_settings()
        with pytest.raises(RuntimeError, match="deliberate"):
            fails()
        assert pool_settings() == before
    assert {name: os.environ.get(name) for name in environment} == environment


def test_wrapped_fit_and_prediction_preserve_public_names():
    from consumer_cards.models import FittedCardModels, fit_calibration, fit_models
    from consumer_cards.evaluation import paired_group_bootstrap
    from consumer_cards.replay import run_replay
    for function in (fit_models, fit_calibration, FittedCardModels.predict, FittedCardModels.raw_predictions, FittedCardModels.scorecard_contributions, paired_group_bootstrap, run_replay):
        assert function.__name__ == function.__wrapped__.__name__
