"""Abstract base class for genomic selection models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GSModel(ABC):
    """Common interface for genomic selection models: fit() and predict()."""

    name: str = "GSModel"
    abbreviation: str = "GS"

    @abstractmethod
    def fit(self, X_train, y_train): ...

    @abstractmethod
    def predict(self, X_test) -> np.ndarray: ...

    def get_params(self) -> dict:
        return {}

    def __repr__(self):
        return f"GSModel({self.name})"
