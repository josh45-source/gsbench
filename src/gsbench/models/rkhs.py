"""RKHS regression using a Gaussian (RBF) kernel with CV-selected bandwidth."""

from __future__ import annotations

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV

from gsbench.models.base import GSModel

# Multipliers applied to the median-heuristic gamma (see _median_heuristic_gamma)
# to build the CV search grid. A fixed absolute grid like [0.001, 0.01, 0.1, 1.0]
# only works for low-dimensional data: with thousands of markers, squared
# distances are large enough that all but the smallest of those values collapse
# the RBF kernel to (near) the identity matrix, leaving no similarity structure
# to generalize from.
_GAMMA_MULTIPLIERS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


def _median_heuristic_gamma(X: np.ndarray) -> float:
    """Data-adaptive RBF bandwidth: gamma = 1 / median(pairwise squared distance)."""
    sq_dists = pairwise_distances(X, metric="sqeuclidean")
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    upper = upper[upper > 0]
    median_sq_dist = np.median(upper) if upper.size else 1.0
    return 1.0 / median_sq_dist


class RKHS(GSModel):
    """Kernel ridge regression with an RBF kernel; gamma chosen by internal CV."""

    name = "RKHS (Gaussian Kernel)"
    abbreviation = "RKHS"

    def __init__(self, alpha: float = 1.0, gamma_values=None, cv: int = 3):
        self.alpha = alpha
        # None means "derive a data-adaptive grid at fit time"; an explicit
        # list is used verbatim, unchanged from before.
        self.gamma_values = list(gamma_values) if gamma_values is not None else None
        self.cv = cv
        self._model = None
        self._best_gamma = None
        self._y_mean = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if self.gamma_values is not None:
            gamma_grid = self.gamma_values
        else:
            base_gamma = _median_heuristic_gamma(X_train)
            gamma_grid = [base_gamma * m for m in _GAMMA_MULTIPLIERS]

        # KernelRidge has no fit_intercept option, so a non-zero-mean y (e.g.
        # yield ~ 190) can't be reproduced from the kernel alone. Center y
        # ourselves and add the mean back at prediction time, same fix as
        # GBLUP (see gblup.py).
        self._y_mean = y_train.mean()
        y_centered = y_train - self._y_mean

        cv_folds = min(self.cv, X_train.shape[0])
        if cv_folds < 2:
            self._best_gamma = gamma_grid[0]
        else:
            search = GridSearchCV(
                KernelRidge(kernel="rbf", alpha=self.alpha),
                param_grid={"gamma": gamma_grid},
                cv=cv_folds,
            )
            search.fit(X_train, y_centered)
            self._best_gamma = search.best_params_["gamma"]

        self._model = KernelRidge(kernel="rbf", alpha=self.alpha, gamma=self._best_gamma)
        self._model.fit(X_train, y_centered)
        return self

    def predict(self, X_test) -> np.ndarray:
        return self._model.predict(np.asarray(X_test, dtype=float)) + self._y_mean

    def get_params(self) -> dict:
        return {"alpha": self.alpha, "gamma": self._best_gamma}
