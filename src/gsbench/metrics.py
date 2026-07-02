"""Prediction accuracy metrics for genomic selection benchmarking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a = pd.Series(a).rank().to_numpy()
    ranks_b = pd.Series(b).rank().to_numpy()
    return _pearson(ranks_a, ranks_b)


def compute_metrics(y_true, y_pred) -> dict:
    """Compute prediction accuracy metrics comparing predicted vs. observed phenotypes.

    Breeders care most about "r" (prediction accuracy) and "spearman"
    (whether the model ranks genotypes correctly for selection).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(y_pred) - np.mean(y_true))

    y_true_var = float(np.var(y_true))
    y_true_std = float(np.std(y_true))

    r2 = (
        float(1.0 - np.sum(residuals**2) / np.sum((y_true - y_true.mean()) ** 2))
        if y_true_var > 0
        else float("nan")
    )
    nrmse = float(rmse / y_true_std) if y_true_std > 0 else float("nan")

    slope = float(np.polyfit(y_pred, y_true, 1)[0]) if np.std(y_pred) > 0 else float("nan")

    return {
        "r": _pearson(y_true, y_pred),
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "slope": slope,
        "spearman": _spearman(y_true, y_pred),
        "nrmse": nrmse,
    }
