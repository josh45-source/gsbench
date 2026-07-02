"""Tree-ensemble models for genomic selection."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from gsbench.models.base import GSModel


class RFModel(GSModel):
    """Random forest regression on raw marker dosages."""

    name = "Random Forest"
    abbreviation = "RF"

    def __init__(self, n_estimators: int = 500, random_state: int = 42, **kwargs):
        self._model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, **kwargs
        )

    def fit(self, X_train, y_train):
        self._model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float))
        return self

    def predict(self, X_test) -> np.ndarray:
        return self._model.predict(np.asarray(X_test, dtype=float))

    def get_params(self) -> dict:
        return self._model.get_params()


try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class XGBoostModel(GSModel):
    """Gradient-boosted trees via XGBoost. Requires the optional 'full' extra."""

    name = "XGBoost"
    abbreviation = "XGB"

    def __init__(self, **kwargs):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed. Install with 'pip install gsbench[full]'.")
        self._model = XGBRegressor(**kwargs)

    def fit(self, X_train, y_train):
        self._model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float))
        return self

    def predict(self, X_test) -> np.ndarray:
        return self._model.predict(np.asarray(X_test, dtype=float))

    def get_params(self) -> dict:
        return self._model.get_params()


try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class LightGBMModel(GSModel):
    """Gradient-boosted trees via LightGBM. Requires the optional 'full' extra."""

    name = "LightGBM"
    abbreviation = "LGBM"

    def __init__(self, **kwargs):
        if not HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm is not installed. Install with 'pip install gsbench[full]'."
            )
        self._model = LGBMRegressor(**kwargs)

    def fit(self, X_train, y_train):
        self._model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float))
        return self

    def predict(self, X_test) -> np.ndarray:
        return self._model.predict(np.asarray(X_test, dtype=float))

    def get_params(self) -> dict:
        return self._model.get_params()
