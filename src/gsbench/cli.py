"""Command-line interface for gsbench."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from gsbench import __version__
from gsbench.crossval import BenchmarkResult, benchmark
from gsbench.io import align_data, read_genotype, read_phenotype
from gsbench.models import AVAILABLE_MODELS, get_available_models, get_models
from gsbench.preprocess import preprocess_pipeline
from gsbench.report import generate_report

app = typer.Typer(help="GS-Bench: Genomic Selection Model Benchmarking CLI for Plant Breeding")
console = Console()

_DATA_DIR = Path(__file__).parent / "data"


@app.command()
def run(
    geno: Path = typer.Argument(..., exists=True, readable=True, help="Genotype file"),
    pheno: Path = typer.Argument(..., exists=True, readable=True, help="Phenotype file"),
    trait: str = typer.Option(..., help="Trait column name"),
    models: str = typer.Option("all", help="Models: all or comma-separated"),
    folds: int = typer.Option(5, help="Number of CV folds"),
    repeats: int = typer.Option(1, help="Number of CV repeats"),
    maf: float = typer.Option(0.05, help="MAF filter threshold"),
    max_missing: float = typer.Option(0.2, help="Max missingness threshold"),
    impute: str = typer.Option("mean", help="Imputation method: mean or median"),
    scale: str = typer.Option("center", help="Scaling: center, standardize, or none"),
    output: Path = typer.Option("gsbench_output", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
    format: str = typer.Option(
        "auto", help="Genotype file format: auto, csv, tsv, hapmap, numeric"
    ),
) -> None:
    """Benchmark genomic selection models on your genotype/phenotype data."""
    console.print(f"[bold cyan]GS-Bench v{__version__}[/bold cyan]")

    # 1. Read data
    console.rule("[bold]Step 1: Loading Data")
    geno_dict = read_genotype(geno, format=format)
    pheno_df = read_phenotype(pheno, trait=trait)

    # 2. Align samples
    console.rule("[bold]Step 2: Aligning Samples")
    X, y, sample_ids = align_data(geno_dict, pheno_df, trait)
    console.print(f"Aligned dataset: {X.shape[0]} samples x {X.shape[1]} markers")

    # 3. Preprocess
    console.rule("[bold]Step 3: Preprocessing")
    prep = preprocess_pipeline(
        X, geno_dict["markers"], maf=maf, max_missing=max_missing, impute=impute, scale=scale
    )
    X_processed = prep["X_processed"]
    console.print(
        f"Removed {prep['n_removed']} marker(s); "
        f"{prep['preprocessing_summary']['n_markers_output']} remaining"
    )

    # 4. Get models
    console.rule("[bold]Step 4: Selecting Models")
    model_names = models if models == "all" else [m.strip() for m in models.split(",")]
    model_list = get_models(model_names)
    if not model_list:
        console.print("[red]No models available. Check installation or --models value.[/red]")
        raise typer.Exit(code=1)
    console.print(f"Models: {', '.join(m.abbreviation for m in model_list)}")

    # 5. Run benchmark
    console.rule("[bold]Step 5: Cross-Validating")
    result = benchmark(
        X_processed,
        y,
        model_list,
        n_folds=folds,
        n_repeats=repeats,
        random_state=seed,
        verbose=True,
        trait_name=trait,
        preprocessing_summary=prep["preprocessing_summary"],
    )

    # 6. Generate report
    console.rule("[bold]Step 6: Generating Report")
    report_dir = generate_report(result, str(output), title=f"GS-Bench Report: {trait}")
    console.print(f"[green]Report written to {report_dir / 'report.html'}[/green]")

    # 7. Print summary to console
    console.rule("[bold]Step 7: Summary")
    _print_final_summary(result)


def _print_final_summary(result: BenchmarkResult) -> None:
    table = Table(title="Final Summary")
    table.add_column("Model", style="bold")
    table.add_column("r", justify="right")
    table.add_column("Spearman", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("Time (s)", justify="right")

    for cv_result in result.results:
        m = cv_result.mean_metrics
        table.add_row(
            cv_result.model_abbreviation,
            f"{m.get('r', float('nan')):.3f}",
            f"{m.get('spearman', float('nan')):.3f}",
            f"{m.get('rmse', float('nan')):.3f}",
            f"{cv_result.total_time_seconds:.2f}",
        )

    console.print(table)


@app.command()
def list_models() -> None:
    """Print available genomic selection models in a table."""
    installed = set(get_available_models())

    table = Table(title="Available Models")
    table.add_column("Abbreviation", style="bold")
    table.add_column("Name")
    table.add_column("Status")

    for abbr, cls in AVAILABLE_MODELS.items():
        status = (
            "[green]installed[/green]" if abbr in installed else "[yellow]requires extra[/yellow]"
        )
        table.add_row(abbr, cls.name, status)

    console.print(table)
    console.print("Install optional models: [bold]pip install gsbench\\[full][/bold]")


@app.command()
def example(
    output: Path = typer.Option(Path("."), help="Directory to copy the example dataset into"),
) -> None:
    """Copy the bundled example genotype/phenotype dataset to a local directory."""
    output.mkdir(parents=True, exist_ok=True)

    geno_dest = output / "example_geno.csv"
    pheno_dest = output / "example_pheno.csv"
    shutil.copyfile(_DATA_DIR / "example_geno.csv", geno_dest)
    shutil.copyfile(_DATA_DIR / "example_pheno.csv", pheno_dest)

    console.print(f"[green]Example dataset copied to {output}[/green]")
    console.print(f"  {geno_dest}")
    console.print(f"  {pheno_dest}")
    console.print("\nRun the benchmark with:\n")
    console.print(
        f"  gsbench run {geno_dest} {pheno_dest} --trait yield --models GBLUP,BRR,RF --folds 5"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
