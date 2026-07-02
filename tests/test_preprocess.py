from __future__ import annotations

import numpy as np
import pytest

from gsbench.preprocess import (
    filter_markers,
    impute_missing,
    preprocess_pipeline,
    scale_genotypes,
)


class TestFilterMarkers:
    def test_removes_monomorphic_markers(self):
        marker_names = ["mono0", "mono2", "polymorphic", "half_missing"]
        X = np.array(
            [
                [0.0, 2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0, np.nan],
                [0.0, 2.0, 2.0, 1.0],
                [0.0, 2.0, 1.0, np.nan],
                [0.0, 2.0, 0.0, 1.0],
            ]
        )
        X_filtered, names_filtered = filter_markers(X, marker_names, maf=0.05, max_missing=0.2)

        # mono0 (all 0) and mono2 (all 2) both have MAF 0 and should be dropped.
        # half_missing has 40% missingness (> 0.2 default) and should be dropped.
        assert names_filtered == ["polymorphic"]
        assert X_filtered.shape == (5, 1)
        np.testing.assert_array_equal(X_filtered[:, 0], X[:, 2])

    def test_missingness_threshold(self):
        marker_names = ["ok", "too_missing"]
        X = np.array(
            [
                [0.0, np.nan],
                [1.0, np.nan],
                [2.0, np.nan],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        X_filtered, names_filtered = filter_markers(X, marker_names, maf=0.0, max_missing=0.2)
        assert names_filtered == ["ok"]
        assert X_filtered.shape == (5, 1)


class TestImputeMissing:
    def test_mean_imputation_fills_nan(self):
        X = np.array(
            [
                [0.0, 2.0],
                [np.nan, 2.0],
                [2.0, np.nan],
            ]
        )
        X_imputed = impute_missing(X, method="mean")
        assert not np.isnan(X_imputed).any()
        # Column 0 mean of [0, 2] = 1.0; column 1 mean of [2, 2] = 2.0.
        np.testing.assert_allclose(X_imputed[:, 0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(X_imputed[:, 1], [2.0, 2.0, 2.0])

    def test_median_imputation_fills_nan(self):
        X = np.array(
            [
                [0.0],
                [4.0],
                [np.nan],
                [10.0],
            ]
        )
        X_imputed = impute_missing(X, method="median")
        assert not np.isnan(X_imputed).any()
        # Median of [0, 4, 10] = 4.0.
        np.testing.assert_allclose(X_imputed[:, 0], [0.0, 4.0, 4.0, 10.0])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            impute_missing(np.zeros((2, 2)), method="bogus")


class TestScaleGenotypes:
    def test_center_gives_zero_column_means(self):
        X = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        X_scaled = scale_genotypes(X, method="center")
        np.testing.assert_allclose(X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(X_scaled[:, 0], [-1.0, 0.0, 1.0])

    def test_standardize_gives_unit_variance(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        X_scaled = scale_genotypes(X, method="standardize")
        np.testing.assert_allclose(X_scaled.mean(axis=0), [0.0], atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), [1.0], atol=1e-10)

    def test_standardize_constant_column_does_not_divide_by_zero(self):
        X = np.array([[5.0], [5.0], [5.0]])
        X_scaled = scale_genotypes(X, method="standardize")
        assert not np.isnan(X_scaled).any()
        np.testing.assert_allclose(X_scaled, [[0.0], [0.0], [0.0]])

    def test_none_returns_unchanged(self):
        X = np.array([[0.0, 1.0], [2.0, 3.0]])
        X_scaled = scale_genotypes(X, method="none")
        np.testing.assert_array_equal(X_scaled, X)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            scale_genotypes(np.zeros((2, 2)), method="bogus")


class TestPreprocessPipeline:
    def test_runs_filter_impute_scale_in_sequence(self):
        marker_names = ["mono", "good1", "good2"]
        X = np.array(
            [
                [0.0, 0.0, np.nan],
                [0.0, 1.0, 1.0],
                [0.0, 2.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        result = preprocess_pipeline(X, marker_names, maf=0.05, max_missing=0.5)

        assert result["marker_names_filtered"] == ["good1", "good2"]
        assert result["n_removed"] == 1
        assert result["X_processed"].shape == (4, 2)
        assert not np.isnan(result["X_processed"]).any()

        summary = result["preprocessing_summary"]
        assert summary["n_markers_input"] == 3
        assert summary["n_markers_output"] == 2
        assert summary["n_removed"] == 1
        assert summary["impute_method"] == "mean"
        assert summary["scale_method"] == "center"
