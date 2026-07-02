"""Bayesian regression models for genomic selection."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge as _SkBayesianRidge

from gsbench.models.base import GSModel


class BayesianRidge(GSModel):
    """Bayesian ridge regression directly on marker dosages."""

    name = "Bayesian Ridge Regression"
    abbreviation = "BRR"

    def __init__(self, **kwargs):
        self._model = _SkBayesianRidge(**kwargs)

    def fit(self, X_train, y_train):
        self._model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float))
        return self

    def predict(self, X_test) -> np.ndarray:
        return self._model.predict(np.asarray(X_test, dtype=float))

    def get_params(self) -> dict:
        return self._model.get_params()


class BayesLasso(GSModel):
    """ARD regression, used as a sparse approximation of BayesB/BayesC."""

    name = "Bayesian LASSO"
    abbreviation = "BL"

    def __init__(self, **kwargs):
        self._model = ARDRegression(**kwargs)

    def fit(self, X_train, y_train):
        self._model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float))
        return self

    def predict(self, X_test) -> np.ndarray:
        return self._model.predict(np.asarray(X_test, dtype=float))

    def get_params(self) -> dict:
        return self._model.get_params()
