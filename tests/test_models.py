from __future__ import annotations

import numpy as np
import pytest

from gsbench.models import AVAILABLE_MODELS, get_models
from gsbench.models.base import GSModel
from gsbench.models.bayesian import BayesianRidge
from gsbench.models.gblup import GBLUP
from gsbench.models.rkhs import RKHS, _median_heuristic_gamma


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(20, 50)).astype(float)
    y = rng.normal(size=20)
    return X, y


@pytest.fixture
def offset_trait_data():
    """Marker/phenotype data shaped like a real trait: many markers, a
    large non-zero mean (e.g. a yield trait around 190). This is the shape
    of data that hid the GBLUP/RKHS mean-centering bug: toy_data's y is
    already ~N(0, 1), so a model with no intercept can't help but look
    fine on it.
    """
    rng = np.random.default_rng(1)
    n_samples, n_markers = 60, 300
    X = rng.integers(0, 3, size=(n_samples, n_markers)).astype(float)
    true_effects = rng.normal(scale=0.5, size=n_markers)
    y = 190.0 + X @ true_effects + rng.normal(scale=5.0, size=n_samples)
    return X, y


class TestGSModelBase:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            GSModel()


class TestAllModels:
    @pytest.mark.parametrize("abbreviation", list(AVAILABLE_MODELS.keys()))
    def test_fit_predict_shape(self, abbreviation, toy_data):
        model_list = get_models([abbreviation])
        if not model_list:
            pytest.skip(f"{abbreviation} not installed")
        model = model_list[0]
        X, y = toy_data

        fitted = model.fit(X, y)
        assert fitted is model

        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (20,)
        assert np.all(np.isfinite(preds))
        assert isinstance(model.get_params(), dict)


class TestGBLUPScale:
    """Regression tests for the GBLUP mean-centering bug: KernelRidge has no
    fit_intercept, so a non-zero-mean y produced predictions clustered near
    0 (RMSE ~190 on a trait with mean ~190) even though the correlation
    looked fine. GBLUP predictions should land on the same scale as BRR's.
    """

    def test_gblup_and_brr_predictions_have_similar_scale(self, offset_trait_data):
        X, y = offset_trait_data

        gblup = GBLUP().fit(X, y)
        brr = BayesianRidge().fit(X, y)

        gblup_preds = gblup.predict(X)
        brr_preds = brr.predict(X)

        # Both models see the same data; their prediction means should be
        # in the same ballpark as each other and as y itself (the bug
        # produced GBLUP predictions clustered near 0 for a trait with
        # mean ~190). Note: we don't compare standard deviations directly
        # against BRR, since Bayesian shrinkage can legitimately produce
        # near-flat predictions in a small-n/large-p regime -- that's a
        # regularization choice, not a scale bug.
        assert gblup_preds.mean() == pytest.approx(brr_preds.mean(), abs=3 * y.std())
        assert gblup_preds.mean() == pytest.approx(y.mean(), abs=3 * y.std())
        # GBLUP's own spread should be a reasonable fraction of y's spread,
        # not collapsed to ~0 (a different failure mode) or blown up.
        assert 0.01 * y.std() < gblup_preds.std() < 10 * y.std()

    def test_gblup_rmse_is_not_orders_of_magnitude_off(self, offset_trait_data):
        X, y = offset_trait_data
        gblup = GBLUP().fit(X, y)
        preds = gblup.predict(X)
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        # A correctly-scaled fit should have in-sample RMSE well below the
        # trait's own standard deviation; the old bug gave RMSE ~= y.mean().
        assert rmse < y.std()


class TestRKHSGamma:
    """Regression tests for the RKHS default gamma grid: a fixed absolute
    grid like [0.001, 0.01, 0.1, 1.0] only makes sense for low-dimensional
    data. With thousands of markers, squared distances are large enough
    that all but the smallest of those values collapse the RBF kernel to
    (near) the identity matrix, leaving nothing to generalize from.
    """

    def test_median_heuristic_gamma_is_positive_and_finite(self, offset_trait_data):
        X, _ = offset_trait_data
        gamma = _median_heuristic_gamma(X)
        assert gamma > 0
        assert np.isfinite(gamma)

    def test_median_heuristic_scales_down_with_more_markers(self):
        rng = np.random.default_rng(2)
        X_small = rng.integers(0, 3, size=(40, 20)).astype(float)
        X_large = rng.integers(0, 3, size=(40, 4000)).astype(float)

        gamma_small = _median_heuristic_gamma(X_small)
        gamma_large = _median_heuristic_gamma(X_large)

        # More markers -> larger squared distances -> smaller gamma needed
        # to avoid kernel collapse. The old fixed grid's smallest value
        # (0.001) is already too large for ~4000 markers.
        assert gamma_large < gamma_small
        assert gamma_large < 0.001

    def test_rkhs_fitted_gamma_is_data_adaptive_not_fixed_grid(self, offset_trait_data):
        X, y = offset_trait_data
        rkhs = RKHS().fit(X, y)
        fitted_gamma = rkhs.get_params()["gamma"]
        # The fitted gamma should be of the same order of magnitude as the
        # median heuristic for this data, not one of the old fixed values
        # (0.001, 0.01, 0.1, 1.0), which are far too large here.
        base_gamma = _median_heuristic_gamma(X)
        assert 0.01 * base_gamma <= fitted_gamma <= 10 * base_gamma

    def test_rkhs_predictions_have_similar_scale_to_brr(self, offset_trait_data):
        X, y = offset_trait_data
        rkhs = RKHS().fit(X, y)
        brr = BayesianRidge().fit(X, y)

        rkhs_preds = rkhs.predict(X)
        brr_preds = brr.predict(X)

        assert rkhs_preds.mean() == pytest.approx(brr_preds.mean(), abs=3 * y.std())


class TestGetModels:
    def test_all_returns_installed_models(self):
        models = get_models("all")
        assert len(models) >= 5
        for model in models:
            assert isinstance(model, GSModel)

    def test_explicit_list(self):
        models = get_models(["GBLUP", "RF"])
        abbrs = {m.abbreviation for m in models}
        assert abbrs == {"GBLUP", "RF"}

    def test_unknown_abbreviation_raises(self):
        with pytest.raises(ValueError):
            get_models(["NOT_A_MODEL"])
