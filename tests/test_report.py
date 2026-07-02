from __future__ import annotations

import csv

import numpy as np
import pytest

from gsbench.crossval import BenchmarkResult, CVResult
from gsbench.report import generate_report


def _make_cv_result(abbreviation: str, name: str, seed: int) -> CVResult:
    rng = np.random.default_rng(seed)
    per_fold_metrics = []
    predictions = {}
    idx = 0
    for _ in range(3):
        y_true = rng.normal(size=5)
        y_pred = y_true + rng.normal(scale=0.2, size=5)
        residuals = y_pred - y_true
        rmse = float(np.sqrt(np.mean(residuals**2)))
        per_fold_metrics.append(
            {
                "r": 0.9,
                "r2": 0.8,
                "rmse": rmse,
                "mae": rmse * 0.8,
                "bias": 0.01,
                "slope": 1.02,
                "spearman": 0.85,
                "nrmse": 0.3,
            }
        )
        for true_val, pred_val in zip(y_true, y_pred):
            predictions[idx] = (float(true_val), float(pred_val))
            idx += 1

    return CVResult(
        model_name=name,
        model_abbreviation=abbreviation,
        per_fold_metrics=per_fold_metrics,
        mean_metrics={
            "r": 0.9,
            "r2": 0.8,
            "rmse": 0.5,
            "mae": 0.4,
            "bias": 0.01,
            "slope": 1.02,
            "spearman": 0.85,
            "nrmse": 0.3,
        },
        total_time_seconds=1.23,
        predictions=predictions,
    )


@pytest.fixture
def mock_benchmark_result() -> BenchmarkResult:
    results = [
        _make_cv_result("GBLUP", "GBLUP", seed=1),
        _make_cv_result("BRR", "Bayesian Ridge Regression", seed=2),
    ]
    return BenchmarkResult(
        results=results,
        trait_name="yield",
        n_samples=15,
        n_markers=50,
        preprocessing_summary={"n_markers_input": 60, "n_markers_output": 50, "n_removed": 10},
    )


class TestGenerateReport:
    def test_creates_expected_files(self, tmp_path, mock_benchmark_result):
        output_dir = tmp_path / "report_out"
        result_path = generate_report(mock_benchmark_result, str(output_dir), title="Test Report")

        assert result_path == output_dir
        assert (output_dir / "report.html").exists()
        assert (output_dir / "summary.csv").exists()

        plots_dir = output_dir / "plots"
        expected_plots = [
            "model_comparison_barplot.png",
            "model_comparison_boxplot.png",
            "predicted_vs_observed.png",
            "bias_plot.png",
            "runtime_comparison.png",
        ]
        for plot_name in expected_plots:
            plot_path = plots_dir / plot_name
            assert plot_path.exists()
            assert plot_path.stat().st_size > 0

    def test_html_report_contains_key_content(self, tmp_path, mock_benchmark_result):
        output_dir = tmp_path / "report_out"
        generate_report(mock_benchmark_result, str(output_dir), title="Test Report")

        html = (output_dir / "report.html").read_text(encoding="utf-8")
        assert "Test Report" in html
        assert "GBLUP" in html
        assert "BRR" in html
        assert "yield" in html
        assert "data:image/png;base64," in html

    def test_summary_csv_has_one_row_per_model(self, tmp_path, mock_benchmark_result):
        output_dir = tmp_path / "report_out"
        generate_report(mock_benchmark_result, str(output_dir))

        with open(output_dir / "summary.csv", newline="") as fh:
            rows = list(csv.reader(fh))

        assert rows[0][0] == "model"
        assert len(rows) == 1 + len(mock_benchmark_result.results)
        abbreviations = {row[1] for row in rows[1:]}
        assert abbreviations == {"GBLUP", "BRR"}

    def test_handles_empty_results(self, tmp_path):
        empty_result = BenchmarkResult(results=[], trait_name="yield", n_samples=0, n_markers=0)
        output_dir = tmp_path / "empty_out"
        generate_report(empty_result, str(output_dir))

        assert (output_dir / "report.html").exists()
        assert (output_dir / "summary.csv").exists()
