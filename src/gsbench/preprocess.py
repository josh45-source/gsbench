"""Marker filtering, missing-value imputation, and genotype scaling."""

from __future__ import annotations

import warnings

import numpy as np


def filter_markers(X, marker_names, maf: float = 0.05, max_missing: float = 0.2):
    """Drop markers below a minor allele frequency or above a missingness threshold.

    Returns (X_filtered, marker_names_filtered).
    """
    X = np.asarray(X, dtype=float)

    missing_frac = np.isnan(X).mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        allele_freq = np.nanmean(X, axis=0) / 2.0
    marker_maf = np.minimum(allele_freq, 1.0 - allele_freq)

    keep_mask = (marker_maf >= maf) & (missing_frac <= max_missing) & ~np.isnan(marker_maf)

    X_filtered = X[:, keep_mask]
    marker_names_filtered = [m for m, keep in zip(marker_names, keep_mask) if keep]
    return X_filtered, marker_names_filtered


def impute_missing(X, method: str = "mean"):
    """Fill missing genotype values per marker column with its mean or median."""
    X = np.array(X, dtype=float, copy=True)

    if method == "mean":
        agg = np.nanmean
    elif method == "median":
        agg = np.nanmedian
    else:
        raise ValueError(f"Unknown imputation method: {method!r}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fill_values = agg(X, axis=0)
    fill_values = np.where(np.isnan(fill_values), 0.0, fill_values)

    nan_mask = np.isnan(X)
    if nan_mask.any():
        _, col_idx = np.where(nan_mask)
        X[nan_mask] = fill_values[col_idx]
    return X


def scale_genotypes(X, method: str = "center"):
    """Center and/or standardize a genotype matrix column-wise (per marker)."""
    X = np.array(X, dtype=float, copy=True)

    if method == "none":
        return X

    means = X.mean(axis=0)
    centered = X - means

    if method == "center":
        return centered

    if method == "standardize":
        stds = X.std(axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        return centered / stds

    raise ValueError(f"Unknown scaling method: {method!r}")


def preprocess_pipeline(
    X,
    marker_names,
    maf: float = 0.05,
    max_missing: float = 0.2,
    impute: str = "mean",
    scale: str = "center",
) -> dict:
    """Run marker filtering, imputation, and scaling in sequence.

    Returns a dict with keys "X_processed", "marker_names_filtered",
    "n_removed", and "preprocessing_summary".
    """
    X = np.asarray(X, dtype=float)
    n_markers_input = X.shape[1]

    X_filtered, marker_names_filtered = filter_markers(
        X, marker_names, maf=maf, max_missing=max_missing
    )
    n_markers_output = X_filtered.shape[1]
    n_removed = n_markers_input - n_markers_output

    X_imputed = impute_missing(X_filtered, method=impute)
    X_scaled = scale_genotypes(X_imputed, method=scale)

    preprocessing_summary = {
        "n_markers_input": n_markers_input,
        "n_markers_output": n_markers_output,
        "n_removed": n_removed,
        "maf_threshold": maf,
        "max_missing_threshold": max_missing,
        "impute_method": impute,
        "scale_method": scale,
    }

    return {
        "X_processed": X_scaled,
        "marker_names_filtered": marker_names_filtered,
        "n_removed": n_removed,
        "preprocessing_summary": preprocessing_summary,
    }
