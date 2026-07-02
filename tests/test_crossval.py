from __future__ import annotations

import numpy as np
import pytest

from gsbench.crossval import BenchmarkResult, CVResult, benchmark, cross_validate
from gsbench.models.bayesian import BayesianRidge
from gsbench.models.gblup import GBLUP


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(1)
    n_samples, n_markers = 30, 15
    X = rng.integers(0, 3, size=(n_samples, n_markers)).astype(float)
    true_beta = rng.normal(size=n_markers)
    y = X @ true_beta + rng.normal(scale=0.1, size=n_samples)
    return X, y


class TestCrossValidate:
    def test_correct_number_of_folds(self, regression_data):
        X, y = regression_data
        result = cross_validate(X, y, BayesianRidge(), n_folds=5, n_repeats=1, verbose=False)

        assert isinstance(result, CVResult)
        assert len(result.per_fold_metrics) == 5
        assert result.model_name == BayesianRidge.name
        assert result.model_abbreviation == BayesianRidge.abbreviation

    def test_repeated_folds_multiply_fold_count(self, regression_data):
        X, y = regression_data
        result = cross_validate(X, y, BayesianRidge(), n_folds=5, n_repeats=2, verbose=False)
        assert len(result.per_fold_metrics) == 10

    def test_predictions_cover_all_samples_exactly_once(self, regression_data):
        X, y = regression_data
        result = cross_validate(X, y, BayesianRidge(), n_folds=5, n_repeats=1, verbose=False)
        assert set(result.predictions.keys()) == set(range(len(y)))

    def test_mean_metrics_has_all_expected_keys(self, regression_data):
        X, y = regression_data
        result = cross_validate(X, y, GBLUP(), n_folds=3, n_repeats=1, verbose=False)
        expected_keys = {"r", "r2", "rmse", "mae", "bias", "slope", "spearman", "nrmse"}
        assert expected_keys.issubset(result.mean_metrics.keys())
        assert result.total_time_seconds >= 0.0


class TestBenchmark:
    def test_runs_multiple_models_and_returns_results(self, regression_data):
        X, y = regression_data
        models = [BayesianRidge(), GBLUP()]
        result = benchmark(X, y, models, n_folds=3, n_repeats=1, verbose=False)

        assert isinstance(result, BenchmarkResult)
        assert len(result.results) == 2

        abbrs = {r.model_abbreviation for r in result.results}
        assert abbrs == {"BRR", "GBLUP"}
        for cv_result in result.results:
            assert isinstance(cv_result, CVResult)
            assert len(cv_result.per_fold_metrics) == 3
