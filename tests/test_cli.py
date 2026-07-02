from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from gsbench.cli import _DATA_DIR, app

runner = CliRunner()


class TestBundledExampleData:
    def test_example_geno_shape(self):
        df = pd.read_csv(_DATA_DIR / "example_geno.csv", index_col=0)
        assert df.shape == (100, 500)

    def test_example_pheno_shape_and_columns(self):
        df = pd.read_csv(_DATA_DIR / "example_pheno.csv", index_col=0)
        assert df.shape == (100, 2)
        assert set(df.columns) == {"yield", "height"}


class TestExampleCommand:
    def test_copies_files_to_output_dir(self, tmp_path):
        result = runner.invoke(app, ["example", "--output", str(tmp_path)])

        assert result.exit_code == 0
        assert (tmp_path / "example_geno.csv").exists()
        assert (tmp_path / "example_pheno.csv").exists()

    def test_copied_files_match_bundled_data(self, tmp_path):
        runner.invoke(app, ["example", "--output", str(tmp_path)])

        original = (_DATA_DIR / "example_geno.csv").read_bytes()
        copied = (tmp_path / "example_geno.csv").read_bytes()
        assert original == copied

    def test_prints_benchmark_command(self, tmp_path):
        result = runner.invoke(app, ["example", "--output", str(tmp_path)])

        assert "gsbench run" in result.stdout
        assert "--trait yield" in result.stdout


class TestListModelsCommand:
    def test_lists_all_registered_models(self):
        result = runner.invoke(app, ["list-models"])

        assert result.exit_code == 0
        for abbreviation in ("GBLUP", "BRR", "BL", "RKHS", "RF", "XGB", "LGBM"):
            assert abbreviation in result.stdout


class TestRunCommandEndToEnd:
    def test_run_on_example_data_produces_report(self, tmp_path):
        output_dir = tmp_path / "report_out"
        result = runner.invoke(
            app,
            [
                "run",
                str(_DATA_DIR / "example_geno.csv"),
                str(_DATA_DIR / "example_pheno.csv"),
                "--trait",
                "yield",
                "--models",
                "GBLUP",
                "--folds",
                "3",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (output_dir / "report.html").exists()
        assert (output_dir / "summary.csv").exists()
        assert (output_dir / "plots").is_dir()
