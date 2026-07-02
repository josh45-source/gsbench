from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SAMPLES = [f"S{i}" for i in range(1, 6)]
MARKERS = [f"m{i}" for i in range(1, 11)]


@pytest.fixture
def genotype_array():
    rng = np.random.default_rng(42)
    return rng.integers(0, 3, size=(5, 10)).astype(float)


@pytest.fixture
def genotype_csv(tmp_path, genotype_array):
    df = pd.DataFrame(genotype_array, index=SAMPLES, columns=MARKERS)
    df.index.name = "sample"
    path = tmp_path / "genotypes.csv"
    df.to_csv(path)
    return path


@pytest.fixture
def genotype_tsv(tmp_path, genotype_array):
    df = pd.DataFrame(genotype_array, index=SAMPLES, columns=MARKERS)
    df.index.name = "sample"
    path = tmp_path / "genotypes.tsv"
    df.to_csv(path, sep="\t")
    return path


@pytest.fixture
def genotype_numeric_matrix(tmp_path):
    # 10 rows x 5 columns: more rows than columns, so rows are markers and
    # read_genotype should transpose to 5 samples x 10 markers.
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 3, size=(10, 5)).astype(float)
    path = tmp_path / "genotypes_numeric.txt"
    np.savetxt(path, arr, delimiter=",", fmt="%d")
    return path, arr


@pytest.fixture
def genotype_hapmap(tmp_path):
    header = [
        "rs#",
        "alleles",
        "chrom",
        "pos",
        "strand",
        "assembly#",
        "center",
        "protLSID",
        "assayLSID",
        "panelLSID",
        "QCcode",
    ] + SAMPLES
    rows = [
        [
            "m1",
            "A/G",
            "1",
            "100",
            "+",
            "NA",
            "NA",
            "NA",
            "NA",
            "NA",
            "NA",
            "AA",
            "AG",
            "GG",
            "AA",
            "NN",
        ],
        [
            "m2",
            "C/T",
            "1",
            "200",
            "+",
            "NA",
            "NA",
            "NA",
            "NA",
            "NA",
            "NA",
            "CC",
            "CT",
            "TT",
            "CC",
            "CT",
        ],
    ]
    lines = ["\t".join(header)] + ["\t".join(row) for row in rows]
    path = tmp_path / "genotypes.hmp.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def phenotype_values():
    return [10.5, 12.3, 9.8, 11.1, 10.0]


@pytest.fixture
def phenotype_csv(tmp_path, phenotype_values):
    df = pd.DataFrame(
        {"sample": SAMPLES, "yield": phenotype_values, "height": [1.1, 1.2, 1.0, 1.3, 1.05]}
    )
    df = df.set_index("sample")
    path = tmp_path / "phenotypes.csv"
    df.to_csv(path)
    return path


@pytest.fixture
def phenotype_csv_with_missing(tmp_path):
    values = [10.5, None, 9.8, 11.1, None]
    df = pd.DataFrame({"sample": SAMPLES, "yield": values})
    df = df.set_index("sample")
    path = tmp_path / "phenotypes_missing.csv"
    df.to_csv(path)
    return path
