"""GBLUP: genomic best linear unbiased prediction via a genomic-relationship kernel."""

from __future__ import annotations

import numpy as np
from sklearn.kernel_ridge import KernelRidge

from gsbench.models.base import GSModel


class GBLUP(GSModel):
    """Ridge regression on a genomic relationship matrix G = ZZ'/p."""

    name = "GBLUP"
    abbreviation = "GBLUP"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = KernelRidge(kernel="precomputed", alpha=alpha)
        self._X_train = None
        self._marker_means = None
        self._n_markers = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        self._marker_means = X_train.mean(axis=0)
        self._n_markers = X_train.shape[1]
        self._X_train = X_train

        Z_train = X_train - self._marker_means
        G_train = (Z_train @ Z_train.T) / self._n_markers

        self._model.fit(G_train, y_train)
        return self

    def predict(self, X_test) -> np.ndarray:
        X_test = np.asarray(X_test, dtype=float)
        Z_train = self._X_train - self._marker_means
        Z_test = X_test - self._marker_means
        G_test_train = (Z_test @ Z_train.T) / self._n_markers
        return self._model.predict(G_test_train)

    def get_params(self) -> dict:
        return {"alpha": self.alpha}
