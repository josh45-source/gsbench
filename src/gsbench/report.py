"""HTML/CSV/plot report generation for genomic selection benchmarking."""

from __future__ import annotations

import base64
import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from gsbench import __version__  # noqa: E402
from gsbench.crossval import BenchmarkResult  # noqa: E402

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_METRIC_KEYS = ("r", "r2", "rmse", "mae", "bias", "slope", "spearman", "nrmse")


def _save_fig(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _model_comparison_barplot(results: list, path: Path) -> None:
    labels = [r.model_abbreviation for r in results]
    means = [r.mean_metrics.get("r", np.nan) for r in results]
    errs = [
        float(np.std([f["r"] for f in r.per_fold_metrics])) if r.per_fold_metrics else 0.0
        for r in results
    ]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
    ax.bar(labels, means, yerr=errs, capsize=4, color="#4C72B0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Prediction accuracy (r)")
    ax.set_title("Model comparison: prediction accuracy")
    fig.tight_layout()
    _save_fig(fig, path)


def _model_comparison_boxplot(results: list, path: Path) -> None:
    labels = [r.model_abbreviation for r in results]
    data = [[f["r"] for f in r.per_fold_metrics] for r in results]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Prediction accuracy (r)")
    ax.set_title("Prediction accuracy across folds")
    fig.tight_layout()
    _save_fig(fig, path)


def _predicted_vs_observed(results: list, path: Path) -> None:
    n_models = max(1, len(results))
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, result in zip(axes_flat, results):
        y_true = [p[0] for p in result.predictions.values()]
        y_pred = [p[1] for p in result.predictions.values()]
        ax.scatter(y_true, y_pred, alpha=0.6, s=18, color="#55A868")
        if y_true:
            lo, hi = min(y_true + y_pred), max(y_true + y_pred)
            ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(result.model_abbreviation)

    for ax in axes_flat[len(results) :]:
        ax.axis("off")

    fig.tight_layout()
    _save_fig(fig, path)


def _bias_plot(results: list, path: Path) -> None:
    labels = [r.model_abbreviation for r in results]
    slopes = [r.mean_metrics.get("slope", np.nan) for r in results]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
    ax.bar(labels, slopes, color="#C44E52")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Ideal slope = 1")
    ax.set_ylabel("Regression slope (observed ~ predicted)")
    ax.set_title("Bias diagnostic")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, path)


def _runtime_comparison(results: list, path: Path) -> None:
    labels = [r.model_abbreviation for r in results]
    times = [r.total_time_seconds for r in results]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.2), 4))
    ax.bar(labels, times, color="#8172B2")
    ax.set_ylabel("Total CV time (seconds)")
    ax.set_title("Runtime comparison")
    fig.tight_layout()
    _save_fig(fig, path)


_PLOT_BUILDERS = {
    "model_comparison_barplot": _model_comparison_barplot,
    "model_comparison_boxplot": _model_comparison_boxplot,
    "predicted_vs_observed": _predicted_vs_observed,
    "bias_plot": _bias_plot,
    "runtime_comparison": _runtime_comparison,
}


def _write_summary_csv(results: list, path: Path) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "abbreviation", *_METRIC_KEYS, "total_time_seconds"])
        for result in results:
            m = result.mean_metrics
            writer.writerow(
                [result.model_name, result.model_abbreviation]
                + [m.get(key, "") for key in _METRIC_KEYS]
                + [result.total_time_seconds]
            )


def generate_report(
    result: BenchmarkResult,
    output_dir: str,
    title: str = "GS-Bench Report",
) -> Path:
    """Write an HTML report, CSV summary, and diagnostic plots into `output_dir`."""
    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = result.results

    plot_b64: dict = {}
    if results:
        for plot_name, builder in _PLOT_BUILDERS.items():
            plot_path = plots_dir / f"{plot_name}.png"
            builder(results, plot_path)
            plot_b64[plot_name] = base64.b64encode(plot_path.read_bytes()).decode("ascii")

    _write_summary_csv(results, output_path / "summary.csv")

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    html = template.render(
        title=title,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        gsbench_version=__version__,
        trait_name=result.trait_name,
        n_samples=result.n_samples,
        n_markers=result.n_markers,
        preprocessing_summary=result.preprocessing_summary,
        results=results,
        metric_keys=_METRIC_KEYS,
        plots=plot_b64,
    )

    (output_path / "report.html").write_text(html, encoding="utf-8")
    return output_path
