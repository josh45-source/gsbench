from __future__ import annotations

import numpy as np
import pytest

from gsbench.io import align_data, read_genotype, read_phenotype

SAMPLES = [f"S{i}" for i in range(1, 6)]
MARKERS = [f"m{i}" for i in range(1, 11)]


class TestReadGenotype:
    def test_csv_auto(self, genotype_csv, genotype_array):
        result = read_genotype(genotype_csv)
        assert result["samples"] == SAMPLES
        assert result["markers"] == MARKERS
        assert result["X"].shape == (5, 10)
        np.testing.assert_array_equal(result["X"], genotype_array)

    def test_csv_explicit_format(self, genotype_csv, genotype_array):
        result = read_genotype(genotype_csv, format="csv")
        np.testing.assert_array_equal(result["X"], genotype_array)

    def test_tsv_auto(self, genotype_tsv, genotype_array):
        result = read_genotype(genotype_tsv)
        assert result["samples"] == SAMPLES
        assert result["markers"] == MARKERS
        np.testing.assert_array_equal(result["X"], genotype_array)

    def test_numeric_matrix_orientation_autodetect(self, genotype_numeric_matrix):
        path, arr = genotype_numeric_matrix
        result = read_genotype(path)
        # arr is 10 markers x 5 samples; expect transposition to samples x markers.
        assert result["X"].shape == (5, 10)
        np.testing.assert_array_equal(result["X"], arr.T)
        assert result["samples"] == [f"sample_{i}" for i in range(5)]
        assert result["markers"] == [f"marker_{i}" for i in range(10)]

    def test_hapmap_auto(self, genotype_hapmap):
        result = read_genotype(genotype_hapmap)
        assert result["samples"] == SAMPLES
        assert result["markers"] == ["m1", "m2"]
        assert result["X"].shape == (5, 2)

        expected = np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [0, 0],
                [np.nan, 1],
            ]
        )
        np.testing.assert_array_equal(result["X"], expected)

    def test_unknown_format_raises(self, genotype_csv):
        with pytest.raises(ValueError):
            read_genotype(genotype_csv, format="bogus")


class TestReadPhenotype:
    def test_all_numeric_columns(self, phenotype_csv, phenotype_values):
        df = read_phenotype(phenotype_csv)
        assert list(df.index) == SAMPLES
        assert set(df.columns) == {"yield", "height"}
        np.testing.assert_allclose(df["yield"].to_numpy(), phenotype_values)

    def test_specific_trait(self, phenotype_csv, phenotype_values):
        df = read_phenotype(phenotype_csv, trait="yield")
        assert list(df.columns) == ["yield"]
        np.testing.assert_allclose(df["yield"].to_numpy(), phenotype_values)

    def test_missing_trait_raises(self, phenotype_csv):
        with pytest.raises(KeyError):
            read_phenotype(phenotype_csv, trait="nonexistent")


class TestAlignData:
    def test_drops_missing_phenotype_samples(
        self, genotype_csv, genotype_array, phenotype_csv_with_missing
    ):
        geno_dict = read_genotype(genotype_csv)
        pheno_df = read_phenotype(phenotype_csv_with_missing)

        X, y, sample_ids = align_data(geno_dict, pheno_df, "yield")

        # S2 and S5 have missing "yield" values and should be dropped.
        assert sample_ids == ["S1", "S3", "S4"]
        np.testing.assert_allclose(y, [10.5, 9.8, 11.1])
        np.testing.assert_array_equal(X, genotype_array[[0, 2, 3], :])

    def test_missing_trait_raises(self, genotype_csv, phenotype_csv):
        geno_dict = read_genotype(genotype_csv)
        pheno_df = read_phenotype(phenotype_csv)
        with pytest.raises(KeyError):
            align_data(geno_dict, pheno_df, "nonexistent")

    def test_no_overlap_raises(self, genotype_csv, tmp_path):
        import pandas as pd

        geno_dict = read_genotype(genotype_csv)
        other_df = pd.DataFrame({"yield": [1.0, 2.0]}, index=["X1", "X2"])
        with pytest.raises(ValueError):
            align_data(geno_dict, other_df, "yield")
