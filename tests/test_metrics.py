from __future__ import annotations

import numpy as np
import pytest

from gsbench.metrics import compute_metrics


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_metrics(y_true, y_true.copy())

        assert metrics["r"] == pytest.approx(1.0)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["bias"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["slope"] == pytest.approx(1.0)
        assert metrics["spearman"] == pytest.approx(1.0)
        assert metrics["nrmse"] == pytest.approx(0.0, abs=1e-10)

    def test_constant_offset_bias(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true + 2.0
        metrics = compute_metrics(y_true, y_pred)

        # Perfectly correlated but systematically off by +2.
        assert metrics["r"] == pytest.approx(1.0)
        assert metrics["spearman"] == pytest.approx(1.0)
        assert metrics["bias"] == pytest.approx(2.0)
        assert metrics["rmse"] == pytest.approx(2.0)
        assert metrics["mae"] == pytest.approx(2.0)
        assert metrics["slope"] == pytest.approx(1.0)
        # R2 is penalized by the offset even though correlation is perfect.
        assert metrics["r2"] < 1.0

    def test_monotonic_nonlinear_relationship_favors_spearman(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true**2  # Monotonic but nonlinear.
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["spearman"] == pytest.approx(1.0)
        assert metrics["r"] < 1.0

    def test_known_rmse_and_mae(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([1.0, -1.0, 1.0, -1.0])
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["rmse"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(1.0)
        assert metrics["bias"] == pytest.approx(0.0)

    def test_constant_y_true_gives_nan_r2_and_nrmse(self):
        y_true = np.array([3.0, 3.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y_true, y_pred)

        assert np.isnan(metrics["r2"])
        assert np.isnan(metrics["nrmse"])
        assert np.isnan(metrics["r"])
