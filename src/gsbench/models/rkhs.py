"""RKHS regression using a Gaussian (RBF) kernel with CV-selected bandwidth."""

from __future__ import annotations

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

from gsbench.models.base import GSModel

_DEFAULT_GAMMA_GRID = [0.001, 0.01, 0.1, 1.0]


class RKHS(GSModel):
    """Kernel ridge regression with an RBF kernel; gamma chosen by internal CV."""

    name = "RKHS (Gaussian Kernel)"
    abbreviation = "RKHS"

    def __init__(self, alpha: float = 1.0, gamma_values=None, cv: int = 3):
        self.alpha = alpha
        self.gamma_values = (
            list(gamma_values) if gamma_values is not None else list(_DEFAULT_GAMMA_GRID)
        )
        self.cv = cv
        self._model = None
        self._best_gamma = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        cv_folds = min(self.cv, X_train.shape[0])
        if cv_folds < 2:
            self._best_gamma = self.gamma_values[0]
        else:
            search = GridSearchCV(
                KernelRidge(kernel="rbf", alpha=self.alpha),
                param_grid={"gamma": self.gamma_values},
                cv=cv_folds,
            )
            search.fit(X_train, y_train)
            self._best_gamma = search.best_params_["gamma"]

        self._model = KernelRidge(kernel="rbf", alpha=self.alpha, gamma=self._best_gamma)
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X_test) -> np.ndarray:
        return self._model.predict(np.asarray(X_test, dtype=float))

    def get_params(self) -> dict:
        return {"alpha": self.alpha, "gamma": self._best_gamma}
