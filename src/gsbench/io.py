"""Readers for genotype and phenotype data, and sample alignment utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

_HAPMAP_META_COLS = {
    "rs#",
    "rs",
    "alleles",
    "chrom",
    "pos",
    "strand",
    "assembly#",
    "center",
    "protlsid",
    "assaylsid",
    "panellsid",
    "qccode",
}

_IUPAC_MAP = {
    "A": ("A", "A"),
    "C": ("C", "C"),
    "G": ("G", "G"),
    "T": ("T", "T"),
    "R": ("A", "G"),
    "Y": ("C", "T"),
    "S": ("G", "C"),
    "W": ("A", "T"),
    "K": ("G", "T"),
    "M": ("A", "C"),
}

_MISSING_CODES = {"N", "NN", "NA", "-", "--", "."}


def _sniff_delimiter(file: PathLike) -> str:
    """Guess whether a text file is tab- or comma-delimited from its first line."""
    with open(file, "r", newline="") as fh:
        first_line = fh.readline()
    return "\t" if "\t" in first_line else ","


def _split_line(line: str, delimiter: str) -> list:
    return line.rstrip("\r\n").split(delimiter)


def _all_numeric(tokens: list) -> bool:
    if not tokens or tokens == [""]:
        return False
    try:
        for token in tokens:
            float(token)
        return True
    except ValueError:
        return False


def _is_hapmap(header: list) -> bool:
    normalized = [c.strip().lower() for c in header]
    return bool(normalized) and normalized[0] in {"rs#", "rs"} and "alleles" in normalized


def _decode_call(call: str):
    """Decode a HapMap genotype call into a pair of alleles, or (None, None) if missing."""
    call = str(call).strip().upper()
    if call in _MISSING_CODES or call == "":
        return (None, None)
    if len(call) == 2:
        a, b = call[0], call[1]
        if a in _MISSING_CODES or b in _MISSING_CODES:
            return (None, None)
        return (a, b)
    if len(call) == 1 and call in _IUPAC_MAP:
        return _IUPAC_MAP[call]
    return (None, None)


def _row_to_dosage(calls: list) -> np.ndarray:
    """Convert one marker's genotype calls across samples into minor-allele dosage (0/1/2)."""
    decoded = [_decode_call(c) for c in calls]
    alleles = [a for pair in decoded for a in pair if a is not None]
    dosage = np.full(len(calls), np.nan, dtype=float)
    if not alleles:
        return dosage
    counts = Counter(alleles)
    minor_allele = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    for i, (a, b) in enumerate(decoded):
        if a is None or b is None:
            continue
        dosage[i] = (a == minor_allele) + (b == minor_allele)
    return dosage


def _read_hapmap(file: PathLike, delimiter: str) -> dict:
    df = pd.read_csv(file, sep=delimiter, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    id_col = df.columns[0]
    meta_cols = [c for c in df.columns if c.strip().lower() in _HAPMAP_META_COLS]
    sample_cols = [c for c in df.columns if c not in meta_cols]

    markers = df[id_col].astype(str).tolist()
    samples = list(sample_cols)

    n_markers = len(df)
    n_samples = len(sample_cols)
    X = np.full((n_markers, n_samples), np.nan, dtype=float)
    sample_frame = df[sample_cols]
    for i in range(n_markers):
        X[i, :] = _row_to_dosage(sample_frame.iloc[i].tolist())

    return {"X": X.T, "samples": samples, "markers": markers}


def _read_csv_genotype(file: PathLike, delimiter: str) -> dict:
    df = pd.read_csv(file, sep=delimiter, index_col=0)
    samples = [str(s) for s in df.index.tolist()]
    markers = [str(c) for c in df.columns.tolist()]
    X = df.to_numpy(dtype=float)
    return {"X": X, "samples": samples, "markers": markers}


def _read_numeric_matrix(file: PathLike, delimiter: str) -> dict:
    arr = np.loadtxt(file, delimiter=delimiter, ndmin=2)
    n_rows, n_cols = arr.shape
    if n_rows > n_cols:
        # More rows than columns: rows are most likely markers, so transpose
        # to the samples x markers convention.
        arr = arr.T
    n_samples, n_markers = arr.shape
    samples = [f"sample_{i}" for i in range(n_samples)]
    markers = [f"marker_{i}" for i in range(n_markers)]
    return {"X": arr, "samples": samples, "markers": markers}


def read_genotype(file: PathLike, format: str = "auto") -> dict:
    """Read a genotype dosage matrix from CSV/TSV, HapMap, or a plain numeric matrix.

    Returns a dict with keys "X" (numpy array, samples x markers), "samples"
    (list of sample names), and "markers" (list of marker names).
    """
    file = Path(file)
    delimiter = _sniff_delimiter(file)

    with open(file, "r", newline="") as fh:
        header = _split_line(fh.readline(), delimiter)

    fmt = format
    if fmt == "auto":
        if _is_hapmap(header):
            fmt = "hapmap"
        elif _all_numeric(header):
            fmt = "numeric"
        else:
            fmt = "csv"

    if fmt == "hapmap":
        return _read_hapmap(file, delimiter)
    if fmt in ("csv", "tsv"):
        return _read_csv_genotype(file, delimiter)
    if fmt == "numeric":
        return _read_numeric_matrix(file, delimiter)
    raise ValueError(f"Unknown genotype format: {fmt!r}")


def read_phenotype(file: PathLike, trait: Optional[str] = None) -> pd.DataFrame:
    """Read a phenotype file with sample IDs in the first column.

    If `trait` is given, only that column is returned; otherwise all numeric
    columns are returned. The result is indexed by sample ID.
    """
    file = Path(file)
    delimiter = _sniff_delimiter(file)
    df = pd.read_csv(file, sep=delimiter, index_col=0)
    df.index = df.index.astype(str)

    if trait is not None:
        if trait not in df.columns:
            raise KeyError(f"Trait {trait!r} not found in phenotype columns: {list(df.columns)}")
        return df[[trait]]

    return df.select_dtypes(include=[np.number])


def align_data(geno_dict: dict, pheno_df: pd.DataFrame, trait: str):
    """Match samples between genotype and phenotype data on sample ID.

    Drops samples with a missing phenotype value for `trait`. Returns
    (X, y, sample_ids) with X and y aligned in the same sample order.
    """
    if trait not in pheno_df.columns:
        raise KeyError(f"Trait {trait!r} not found in phenotype columns: {list(pheno_df.columns)}")

    samples = list(geno_dict["samples"])
    X_full = np.asarray(geno_dict["X"])
    pheno_by_sample = {str(idx): val for idx, val in pheno_df[trait].items()}

    kept_indices = []
    kept_samples = []
    y_values = []
    for i, sample in enumerate(samples):
        if sample not in pheno_by_sample:
            continue
        value = pheno_by_sample[sample]
        if pd.isna(value):
            continue
        kept_indices.append(i)
        kept_samples.append(sample)
        y_values.append(value)

    if not kept_samples:
        raise ValueError(
            "No overlapping samples with non-missing phenotype values "
            "between genotype and phenotype data."
        )

    X = X_full[kept_indices, :]
    y = np.asarray(y_values, dtype=float)
    return X, y, kept_samples
