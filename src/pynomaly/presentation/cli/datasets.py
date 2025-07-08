"""Dataset management CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pynomaly.domain.entities import Dataset
from pynomaly.presentation.cli.container import get_cli_container

import pandas as pd
import numpy as np

app = typer.Typer()
console = Console()


@app.command("list")
def list_datasets(
    has_target: bool | None = typer.Option(
        None, "--has-target", help="Filter by target presence"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to show"),
):
    """List all datasets."""
    container = get_cli_container()
    dataset_repo = container.dataset_repository()

    # Get all datasets
    datasets = dataset_repo.find_all()

    # Apply filters
    if has_target is not None:
        datasets = [d for d in datasets if d.has_target == has_target]

    # Limit results
    datasets = datasets[:limit]

    if not datasets:
        console.print("No datasets found.")
        return

    # Create table
    table = Table(title="Datasets")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Shape", style="yellow")
    table.add_column("Target", style="blue")
    table.add_column("Size (MB)", style="magenta")
    table.add_column("Created", style="dim")

    for dataset in datasets:
        table.add_row(
            str(dataset.id)[:8],
            dataset.name,
            f"{dataset.n_samples:,} x {dataset.n_features}",
            "✓" if dataset.has_target else "✗",
            f"{dataset.memory_usage / 1024 / 1024:.1f}",
            dataset.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@app.command("load")
def load_dataset(
    file_path: Path = typer.Argument(..., help="Path to dataset file"),
    name: str | None = typer.Option(None, "--name", "-n", help="Dataset name"),
    target_column: str | None = typer.Option(
        None, "--target", "-t", help="Target column name"
    ),
    description: str | None = typer.Option(
        None, "--description", "-d", help="Dataset description"
    ),
    sample_size: int | None = typer.Option(
        None, "--sample", "-s", help="Load only N rows"
    ),
):
    """Load a dataset from file."""
    container = get_cli_container()
    dataset_repo = container.dataset_repository()
    settings = container.config()

    # Check file exists
    if not file_path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(1)

    # Check file size
    file_size_mb = file_path.stat().st_size / 1024 / 1024
    if file_size_mb > settings.max_dataset_size_mb:
        console.print(f"[red]Error:[/red] File too large ({file_size_mb:.1f}MB)")
        console.print(f"Maximum allowed: {settings.max_dataset_size_mb}MB")
        raise typer.Exit(1)

    # Determine loader based on extension
    extension = file_path.suffix.lower()

    with console.status(f"Loading {file_path.name}..."):
        try:
            if extension in [".csv", ".tsv", ".txt"]:
                csv_loader = container.csv_loader()
                df = csv_loader.load(file_path=file_path, nrows=sample_size)
            elif extension in [".parquet", ".pq"]:
                parquet_loader = container.parquet_loader()
                df = parquet_loader.load(file_path=file_path)
                if sample_size:
                    df = df.head(sample_size)
            else:
                console.print(f"[red]Error:[/red] Unsupported file format: {extension}")
                raise typer.Exit(1)

            # Create dataset
            dataset_name = name or file_path.stem
            dataset = Dataset(
                name=dataset_name,
                data=df,
                description=description,
                target_column=target_column,
                metadata={
                    "source": "cli",
                    "file_path": str(file_path),
                    "file_size_mb": file_size_mb,
                },
            )

            # Save dataset
            dataset_repo.save(dataset)

        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to load dataset: {str(e)}")
            raise typer.Exit(1)

    # Display summary
    console.print(f"[green]✓[/green] Loaded dataset '{dataset_name}'")
    console.print(f"  ID: {dataset.id}")
    console.print(f"  Shape: {dataset.n_samples:,} rows x {dataset.n_features} columns")
    console.print(f"  Memory: {dataset.memory_usage / 1024 / 1024:.1f} MB")
    if dataset.has_target:
        console.print(f"  Target: {dataset.target_column}")


@app.command("generate")
def generate_dataset(
    size: int = typer.Option(1000, help="Number of samples in the dataset"),
    feature_count: int = typer.Option(10, help="Number of features"),
    anomaly_rate: float = typer.Option(0.01, help="Rate of anomalies"),
    name: str = typer.Option("generated_dataset", help="Name of the dataset"),
    output: Optional[str] = typer.Option(None, help="Output file name"),
    format: str = typer.Option("csv", help="Output file format"),
):
    """Generate a synthetic dataset for testing and benchmarking."""
    
    # Generate normal data
    data = pd.DataFrame(
        np.random.normal(0, 1, size=(size, feature_count)),
        columns=[f"feature_{i}" for i in range(feature_count)]
    )
    
    # Add anomalies
    n_anomalies = int(size * anomaly_rate)
    data.iloc[-n_anomalies:] = np.random.normal(10, 1, size=(n_anomalies, feature_count))
    
    # Save dataset
    file_name = output or f"{name}.{format}"
    if format == "csv":
        data.to_csv(file_name, index=False)
    elif format == "json":
        data.to_json(file_name, orient="records")
    elif format == "parquet":
        data.to_parquet(file_name, index=False)
    else:
        console.print(f"[red]Error:[/red] Unsupported format: {format}")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] Generated dataset '{name}' with {size} samples and {feature_count} features")
    console.print(f"Anomalies: {n_anomalies} ({anomaly_rate*100:.2f}%)")
    console.print(f"Saved to: {file_name}")


@app.command("list-samples")
def list_sample_datasets():
    """List available sample datasets for testing and benchmarking."""
    
    console.print("[bold blue]Available Sample Datasets:[/bold blue]\n")
    
    # Check if examples directory exists
    examples_dir = Path("examples/sample_datasets")
    if not examples_dir.exists():
        console.print(f"[yellow]Sample datasets directory not found: {examples_dir}[/yellow]")
        console.print("\nTo create sample datasets, use: pynomaly dataset generate")
        return
    
    # List synthetic datasets
    synthetic_dir = examples_dir / "synthetic"
    if synthetic_dir.exists():
        console.print("[green]Synthetic Datasets:[/green]")
        for csv_file in synthetic_dir.glob("*.csv"):
            file_size = csv_file.stat().st_size / 1024 / 1024  # MB
            console.print(f"  • {csv_file.name} ({file_size:.2f} MB)")
    
    # List real-world datasets
    real_world_dir = examples_dir / "real_world"
    if real_world_dir.exists():
        console.print("\n[green]Real-world Datasets:[/green]")
        for csv_file in real_world_dir.glob("*.csv"):
            file_size = csv_file.stat().st_size / 1024 / 1024  # MB
            console.print(f"  • {csv_file.name} ({file_size:.2f} MB)")
    
    console.print("\nTo load a sample dataset:")
    console.print("  pynomaly dataset load examples/sample_datasets/synthetic/financial_fraud.csv --name \"Sample Dataset\"")
    console.print("\nTo create a custom test dataset:")
    console.print("  pynomaly dataset generate --size 10000 --feature-count 15 --anomaly-rate 0.02 --name test_1mb")


@app.command("show")
def show_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID (can be partial)"),
    sample: int = typer.Option(5, "--sample", "-s", help="Show N sample rows"),
    info: bool = typer.Option(False, "--info", "-i", help="Show detailed info"),
):
    """Show dataset details."""
    container = get_cli_container()
    dataset_repo = container.dataset_repository()

    # Find dataset by partial ID
    datasets = dataset_repo.find_all()
    matching = [d for d in datasets if str(d.id).startswith(dataset_id)]

    if not matching:
        console.print(
            f"[red]Error:[/red] No dataset found with ID starting with '{dataset_id}'"
        )
        raise typer.Exit(1)

    if len(matching) > 1:
        console.print(f"[red]Error:[/red] Multiple datasets match '{dataset_id}':")
        for d in matching:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    dataset = matching[0]

    # Display basic info
    console.print(f"\n[bold]Dataset: {dataset.name}[/bold]")
    console.print(f"ID: {dataset.id}")
    console.print(f"Shape: {dataset.n_samples:,} rows x {dataset.n_features} columns")
    console.print(f"Memory: {dataset.memory_usage / 1024 / 1024:.1f} MB")
    console.print(f"Created: {dataset.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    if dataset.description:
        console.print(f"Description: {dataset.description}")

    if dataset.has_target:
        console.print(f"Target column: {dataset.target_column}")

    # Show column info
    console.print("\n[bold]Columns:[/bold]")
    numeric_cols = dataset.get_numeric_features()
    categorical_cols = dataset.get_categorical_features()

    console.print(f"  Numeric: {len(numeric_cols)} columns")
    if numeric_cols:
        console.print(f"    {', '.join(numeric_cols[:5])}")
        if len(numeric_cols) > 5:
            console.print(f"    ... and {len(numeric_cols) - 5} more")

    console.print(f"  Categorical: {len(categorical_cols)} columns")
    if categorical_cols:
        console.print(f"    {', '.join(categorical_cols[:5])}")
        if len(categorical_cols) > 5:
            console.print(f"    ... and {len(categorical_cols) - 5} more")

    # Show sample data
    if sample > 0:
        console.print(f"\n[bold]Sample ({sample} rows):[/bold]")
        sample_df = dataset.data.head(sample)

        # Create table
        table = Table()
        for col in sample_df.columns:
            table.add_column(
                col, style="cyan" if col == dataset.target_column else None
            )

        for _, row in sample_df.iterrows():
            table.add_row(*[str(v) for v in row.values])

        console.print(table)

    # Show detailed info if requested
    if info:
        console.print("\n[bold]Detailed Statistics:[/bold]")
        desc = dataset.data.describe()

        stats_table = Table()
        stats_table.add_column("Statistic", style="yellow")
        for col in desc.columns[:5]:  # Show first 5 columns
            stats_table.add_column(col, style="cyan")

        for stat in desc.index:
            row = [stat] + [f"{desc.loc[stat, col]:.4f}" for col in desc.columns[:5]]
            stats_table.add_row(*row)

        console.print(stats_table)

        if len(desc.columns) > 5:
            console.print(f"... and {len(desc.columns) - 5} more columns")


@app.command("quality")
def check_quality(
    dataset_id: str = typer.Argument(..., help="Dataset ID (can be partial)"),
):
    """Check dataset quality."""
    container = get_cli_container()
    dataset_repo = container.dataset_repository()
    feature_validator = container.feature_validator()

    # Find dataset
    datasets = dataset_repo.find_all()
    matching = [d for d in datasets if str(d.id).startswith(dataset_id)]

    if not matching:
        console.print(
            f"[red]Error:[/red] No dataset found with ID starting with '{dataset_id}'"
        )
        raise typer.Exit(1)

    if len(matching) > 1:
        console.print(f"[red]Error:[/red] Multiple datasets match '{dataset_id}':")
        for d in matching:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    dataset = matching[0]

    with console.status("Checking data quality..."):
        # Run quality check
        quality_report = feature_validator.check_data_quality(dataset)
        suggestions = feature_validator.suggest_preprocessing(quality_report)

    # Display report
    console.print(f"\n[bold]Data Quality Report: {dataset.name}[/bold]")

    # Quality score
    score = quality_report["quality_score"]
    score_color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
    console.print(f"Overall Quality Score: [{score_color}]{score:.2f}[/{score_color}]")

    # Issues
    console.print("\n[bold]Issues Found:[/bold]")

    if quality_report["missing_values"]:
        console.print(
            f"  [yellow]Missing values:[/yellow] {len(quality_report['missing_values'])} columns"
        )
        for col, pct in list(quality_report["missing_values"].items())[:3]:
            console.print(f"    - {col}: {pct:.1%}")

    if quality_report["constant_features"]:
        console.print(
            f"  [red]Constant features:[/red] {len(quality_report['constant_features'])} columns"
        )
        for col in quality_report["constant_features"][:3]:
            console.print(f"    - {col}")

    if quality_report["low_variance_features"]:
        console.print(
            f"  [yellow]Low variance:[/yellow] {len(quality_report['low_variance_features'])} columns"
        )
        for col in quality_report["low_variance_features"][:3]:
            console.print(f"    - {col}")

    if quality_report["infinite_values"]:
        console.print(
            f"  [red]Infinite values:[/red] {len(quality_report['infinite_values'])} columns"
        )

    if quality_report["duplicate_rows"] > 0:
        console.print(
            f"  [yellow]Duplicate rows:[/yellow] {quality_report['duplicate_rows']:,}"
        )

    # Suggestions
    if suggestions:
        console.print("\n[bold]Recommendations:[/bold]")
        for i, suggestion in enumerate(suggestions, 1):
            console.print(f"  {i}. {suggestion}")

    # Preprocessing command suggestions
    console.print("\n[bold]Preprocessing Commands:[/bold]")
    dataset_id_short = str(dataset.id)[:8]

    # Generate specific command suggestions based on issues found
    commands = []

    if quality_report["missing_values"] or quality_report["infinite_values"]:
        missing_strategy = (
            "drop_rows"
            if len(quality_report.get("missing_values", {})) < 3
            else "fill_median"
        )
        commands.append(
            f"pynomaly data clean {dataset_id_short} --missing {missing_strategy} --infinite remove"
        )

    if quality_report["duplicate_rows"] > 0:
        commands.append(f"pynomaly data clean {dataset_id_short} --duplicates")

    # Always suggest a transformation pipeline for better anomaly detection
    commands.append(
        f"pynomaly data transform {dataset_id_short} --scaling standard --encoding onehot"
    )

    # Suggest creating a complete preprocessing pipeline
    commands.append(
        f"pynomaly data pipeline create --name {dataset.name.lower().replace(' ', '_')}_pipeline"
    )

    if commands:
        console.print("  Suggested commands to improve data quality:")
        for cmd in commands:
            console.print(f"    [cyan]{cmd}[/cyan]")

        console.print(
            "\n  To preview changes without applying them, add [yellow]--dry-run[/yellow] to any command."
        )
        console.print(
            "  To save cleaned data as a new dataset, add [yellow]--save-as new_name[/yellow]."
        )


@app.command("split")
def split_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID (can be partial)"),
    test_size: float = typer.Option(
        0.2, "--test-size", "-t", help="Test set proportion"
    ),
    random_state: int | None = typer.Option(42, "--seed", "-s", help="Random seed"),
):
    """Split dataset into train and test sets."""
    container = get_cli_container()
    dataset_repo = container.dataset_repository()

    # Find dataset
    datasets = dataset_repo.find_all()
    matching = [d for d in datasets if str(d.id).startswith(dataset_id)]

    if not matching:
        console.print(
            f"[red]Error:[/red] No dataset found with ID starting with '{dataset_id}'"
        )
        raise typer.Exit(1)

    if len(matching) > 1:
        console.print(f"[red]Error:[/red] Multiple datasets match '{dataset_id}':")
        for d in matching:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    dataset = matching[0]

    try:
        # Split dataset
        train_dataset, test_dataset = dataset.split(
            test_size=test_size, random_state=random_state
        )

        # Save both datasets
        dataset_repo.save(train_dataset)
        dataset_repo.save(test_dataset)

        console.print(f"[green]✓[/green] Split dataset '{dataset.name}'")
        console.print(f"\nTrain set: {train_dataset.name}")
        console.print(f"  ID: {train_dataset.id}")
        console.print(f"  Samples: {train_dataset.n_samples:,}")
        console.print(f"\nTest set: {test_dataset.name}")
        console.print(f"  ID: {test_dataset.id}")
        console.print(f"  Samples: {test_dataset.n_samples:,}")

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to split dataset: {str(e)}")
        raise typer.Exit(1)


@app.command("delete")
def delete_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID (can be partial)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a dataset."""
    container = get_cli_container()
    dataset_repo = container.dataset_repository()

    # Find dataset
    datasets = dataset_repo.find_all()
    matching = [d for d in datasets if str(d.id).startswith(dataset_id)]

    if not matching:
        console.print(
            f"[red]Error:[/red] No dataset found with ID starting with '{dataset_id}'"
        )
        raise typer.Exit(1)

    if len(matching) > 1:
        console.print(f"[red]Error:[/red] Multiple datasets match '{dataset_id}':")
        for d in matching:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    dataset = matching[0]

    # Confirm deletion
    if not force:
        confirm = typer.confirm(f"Delete dataset '{dataset.name}' ({dataset.id})?")
        if not confirm:
            console.print("Deletion cancelled.")
            return

    # Delete dataset
    success = dataset_repo.delete(dataset.id)

    if success:
        console.print(f"[green]✓[/green] Deleted dataset '{dataset.name}'")
    else:
        console.print("[red]Error:[/red] Failed to delete dataset")
        raise typer.Exit(1)


@app.command("export")
def export_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID (can be partial)"),
    output_path: Path = typer.Argument(..., help="Output file path"),
    format: str = typer.Option(
        "csv", "--format", "-f", help="Export format (csv, parquet)"
    ),
):
    """Export dataset to file."""
    container = get_cli_container()
    dataset_repo = container.dataset_repository()

    # Find dataset
    datasets = dataset_repo.find_all()
    matching = [d for d in datasets if str(d.id).startswith(dataset_id)]

    if not matching:
        console.print(
            f"[red]Error:[/red] No dataset found with ID starting with '{dataset_id}'"
        )
        raise typer.Exit(1)

    if len(matching) > 1:
        console.print(f"[red]Error:[/red] Multiple datasets match '{dataset_id}':")
        for d in matching:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    dataset = matching[0]

    # Export based on format
    try:
        if format.lower() == "csv":
            dataset.data.to_csv(output_path, index=False)
        elif format.lower() == "parquet":
            dataset.data.to_parquet(output_path, index=False)
        else:
            console.print(f"[red]Error:[/red] Unsupported format: {format}")
            raise typer.Exit(1)

        file_size_mb = output_path.stat().st_size / 1024 / 1024
        console.print(
            f"[green]✓[/green] Exported dataset '{dataset.name}' to {output_path}"
        )
        console.print(f"  Format: {format}")
        console.print(f"  Size: {file_size_mb:.1f} MB")

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to export dataset: {str(e)}")
        raise typer.Exit(1)
