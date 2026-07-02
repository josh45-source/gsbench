"""Cross-validated benchmarking of genomic selection models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.model_selection import KFold, RepeatedKFold

from gsbench.metrics import compute_metrics

console = Console()


@dataclass
class CVResult:
    model_name: str
    model_abbreviation: str
    per_fold_metrics: list = field(default_factory=list)
    mean_metrics: dict = field(default_factory=dict)
    total_time_seconds: float = 0.0
    predictions: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    results: list = field(default_factory=list)
    trait_name: str = ""
    n_samples: int = 0
    n_markers: int = 0
    preprocessing_summary: dict = field(default_factory=dict)


def cross_validate(
    X,
    y,
    model,
    n_folds: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    verbose: bool = True,
) -> CVResult:
    """Run (repeated) k-fold cross-validation for a single model.

    Returns a CVResult with per-fold and mean metrics, total wall-clock
    time, and out-of-fold predictions keyed by original sample index.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if n_repeats > 1:
        splitter = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        n_total_folds = n_folds * n_repeats
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        n_total_folds = n_folds

    model_name = getattr(model, "name", model.__class__.__name__)
    model_abbreviation = getattr(model, "abbreviation", model.__class__.__name__)

    per_fold_metrics = []
    predictions: dict = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} folds"),
        TimeElapsedColumn(),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task(f"CV: {model_abbreviation}", total=n_total_folds)

        start_time = time.perf_counter()
        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test), dtype=float)

            per_fold_metrics.append(compute_metrics(y_test, y_pred))
            for idx, true_val, pred_val in zip(test_idx, y_test, y_pred):
                predictions[int(idx)] = (float(true_val), float(pred_val))

            progress.advance(task)
        total_time_seconds = time.perf_counter() - start_time

    metric_keys = per_fold_metrics[0].keys()
    mean_metrics = {
        key: float(np.nanmean([fold[key] for fold in per_fold_metrics])) for key in metric_keys
    }

    return CVResult(
        model_name=model_name,
        model_abbreviation=model_abbreviation,
        per_fold_metrics=per_fold_metrics,
        mean_metrics=mean_metrics,
        total_time_seconds=total_time_seconds,
        predictions=predictions,
    )


def benchmark(
    X,
    y,
    models,
    n_folds: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    verbose: bool = True,
    trait_name: str = "",
    preprocessing_summary: dict | None = None,
) -> BenchmarkResult:
    """Cross-validate each model in `models` and print a comparison table."""
    X = np.asarray(X, dtype=float)
    results = [
        cross_validate(
            X,
            y,
            model,
            n_folds=n_folds,
            n_repeats=n_repeats,
            random_state=random_state,
            verbose=verbose,
        )
        for model in models
    ]

    if verbose:
        _print_summary_table(results)

    return BenchmarkResult(
        results=results,
        trait_name=trait_name,
        n_samples=X.shape[0],
        n_markers=X.shape[1],
        preprocessing_summary=preprocessing_summary or {},
    )


def _print_summary_table(results: list) -> None:
    table = Table(title="GS-Bench Model Comparison")
    table.add_column("Model", style="bold")
    table.add_column("r", justify="right")
    table.add_column("r2", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("Spearman", justify="right")
    table.add_column("Time (s)", justify="right")

    for result in results:
        m = result.mean_metrics
        table.add_row(
            result.model_abbreviation,
            f"{m['r']:.3f}",
            f"{m['r2']:.3f}",
            f"{m['rmse']:.3f}",
            f"{m['mae']:.3f}",
            f"{m['spearman']:.3f}",
            f"{result.total_time_seconds:.2f}",
        )

    console.print(table)
