"""CLI application using Typer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.cli import datasets, detectors, detection, server, performance
from pynomaly.presentation.cli.export import export_app


# Create Typer app
app = typer.Typer(
    name="pynomaly",
    help="Pynomaly - State-of-the-art anomaly detection CLI",
    add_completion=True,
    rich_markup_mode="rich"
)

# Create console for rich output
console = Console()

# Add subcommands
app.add_typer(detectors.app, name="detector", help="Manage anomaly detectors")
app.add_typer(datasets.app, name="dataset", help="Manage datasets")
app.add_typer(detection.app, name="detect", help="Run anomaly detection")
app.add_typer(export_app, name="export", help="Export results to business intelligence platforms")
app.add_typer(server.app, name="server", help="Manage API server")
app.add_typer(performance.app, name="perf", help="Performance monitoring and optimization")

# Store container globally for CLI
_container = None


def get_cli_container():
    """Get or create container for CLI."""
    global _container
    if _container is None:
        _container = create_container()
    return _container


@app.command()
def version():
    """Show version information."""
    container = get_cli_container()
    settings = container.config()
    
    console.print(f"[bold blue]Pynomaly[/bold blue] v{settings.version}")
    console.print(f"Python {sys.version.split()[0]}")
    console.print(f"Storage: {settings.storage_path}")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    set_key: Optional[str] = typer.Option(None, "--set", help="Set config key=value"),
):
    """Manage configuration."""
    container = get_cli_container()
    settings = container.config()
    
    if show:
        # Show configuration
        table = Table(title="Pynomaly Configuration")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        # Show key settings
        config_items = [
            ("App Name", settings.app_name),
            ("Version", settings.version),
            ("Debug Mode", str(settings.debug)),
            ("Storage Path", str(settings.storage_path)),
            ("API Host", settings.api_host),
            ("API Port", str(settings.api_port)),
            ("Max Dataset Size (MB)", str(settings.max_dataset_size_mb)),
            ("Default Contamination Rate", str(settings.default_contamination_rate)),
            ("GPU Enabled", str(settings.gpu_enabled)),
        ]
        
        for key, value in config_items:
            table.add_row(key, value)
        
        console.print(table)
    
    elif set_key:
        # Parse key=value
        if "=" not in set_key:
            console.print("[red]Error:[/red] Use format: --set key=value")
            raise typer.Exit(1)
        
        key, value = set_key.split("=", 1)
        
        # Update config (in real app, would persist this)
        console.print(f"[yellow]Note:[/yellow] Configuration update not yet implemented")
        console.print(f"Would set: {key} = {value}")
    
    else:
        console.print("Use --show to display config or --set key=value to update")


@app.command()
def status():
    """Show system status."""
    container = get_cli_container()
    
    # Get repository counts
    detector_count = container.detector_repository().count()
    dataset_count = container.dataset_repository().count()
    result_count = container.result_repository().count()
    
    # Create status table
    table = Table(title="Pynomaly System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Count", style="yellow")
    
    table.add_row("Detectors", "✓ Active", str(detector_count))
    table.add_row("Datasets", "✓ Active", str(dataset_count))
    table.add_row("Results", "✓ Active", str(result_count))
    table.add_row("API Server", "○ Not running", "-")
    
    console.print(table)
    
    # Show recent activity
    results = container.result_repository().find_recent(5)
    if results:
        console.print("\n[bold]Recent Detection Results:[/bold]")
        recent_table = Table()
        recent_table.add_column("Time", style="dim")
        recent_table.add_column("Detector", style="cyan")
        recent_table.add_column("Dataset", style="green")
        recent_table.add_column("Anomalies", style="red")
        
        for result in results:
            # Get detector and dataset names
            detector = container.detector_repository().find_by_id(result.detector_id)
            dataset = container.dataset_repository().find_by_id(result.dataset_id)
            
            recent_table.add_row(
                result.timestamp.strftime("%Y-%m-%d %H:%M"),
                detector.name if detector else "Unknown",
                dataset.name if dataset else "Unknown",
                f"{result.n_anomalies} ({result.anomaly_rate:.1%})"
            )
        
        console.print(recent_table)


@app.command()
def quickstart():
    """Run interactive quickstart guide."""
    console.print("[bold blue]Welcome to Pynomaly![/bold blue]\n")
    console.print("This quickstart will help you get started with anomaly detection.\n")
    
    # Check if user wants to continue
    if not typer.confirm("Would you like to continue?"):
        console.print("Quickstart cancelled.")
        raise typer.Exit()
    
    console.print("\n[bold]Step 1: Load a dataset[/bold]")
    console.print("You can load data from CSV or Parquet files.")
    console.print("Example: [cyan]pynomaly dataset load data.csv --name my_data[/cyan]")
    
    console.print("\n[bold]Step 2: Create a detector[/bold]")
    console.print("Choose from various algorithms like IsolationForest, LOF, etc.")
    console.print("Example: [cyan]pynomaly detector create --name my_detector --algorithm IsolationForest[/cyan]")
    
    console.print("\n[bold]Step 3: Train the detector[/bold]")
    console.print("Train your detector on the loaded dataset.")
    console.print("Example: [cyan]pynomaly detect train --detector my_detector --dataset my_data[/cyan]")
    
    console.print("\n[bold]Step 4: Detect anomalies[/bold]")
    console.print("Run detection on new data.")
    console.print("Example: [cyan]pynomaly detect run --detector my_detector --dataset test_data[/cyan]")
    
    console.print("\n[bold]Step 5: View and export results[/bold]")
    console.print("Analyze detection results and export them to various platforms.")
    console.print("Examples:")
    console.print("  [cyan]pynomaly detect results --latest[/cyan]")
    console.print("  [cyan]pynomaly export excel results.json output.xlsx[/cyan]")
    console.print("  [cyan]pynomaly export list-formats[/cyan]")
    
    console.print("\n[green]Ready to start![/green] Use [cyan]pynomaly --help[/cyan] to see all commands.")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
):
    """Pynomaly - State-of-the-art anomaly detection platform."""
    if verbose and quiet:
        console.print("[red]Error:[/red] Cannot use --verbose and --quiet together")
        raise typer.Exit(1)
    
    # Set output level
    if quiet:
        console.quiet = True
    elif verbose:
        # In verbose mode, could set logging level
        pass


if __name__ == "__main__":
    app()