from __future__ import annotations

import numpy as np
import pytest

from gsbench.models import AVAILABLE_MODELS, get_models
from gsbench.models.base import GSModel


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(20, 50)).astype(float)
    y = rng.normal(size=20)
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
