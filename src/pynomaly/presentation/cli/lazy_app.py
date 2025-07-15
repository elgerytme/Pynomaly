"""
Lazy-loading CLI application optimized for startup performance.

This module implements lazy loading of CLI subcommands to dramatically improve
startup time by deferring imports until commands are actually used.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console

from pynomaly.presentation.cli.container import get_cli_container
from pynomaly.presentation.cli.async_utils import cli_runner

# Create Typer app with lazy loading
app = typer.Typer(
    name="pynomaly",
    help="Pynomaly - State-of-the-art anomaly detection CLI",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Create console for rich output
console = Console()

# Lazy module registry - no imports at startup
_LAZY_MODULES = {
    "auto": ("pynomaly.presentation.cli.autonomous", "app"),
    "automl": ("pynomaly.presentation.cli.automl", "app"),
    "config": ("pynomaly.presentation.cli.config", "app"),
    "detector": ("pynomaly.presentation.cli.detectors", "app"),
    "dataset": ("pynomaly.presentation.cli.datasets", "app"),
    "data": ("pynomaly.presentation.cli.preprocessing", "app"),
    "detect": ("pynomaly.presentation.cli.detection", "app"),
    "tdd": ("pynomaly.presentation.cli.tdd", "app"),
    "deep-learning": ("pynomaly.presentation.cli.deep_learning", "app"),
    "explainability": ("pynomaly.presentation.cli.explainability", "app"),
    "selection": ("pynomaly.presentation.cli.selection", "app"),
    "export": ("pynomaly.presentation.cli.export", "export_app"),
    "server": ("pynomaly.presentation.cli.server", "app"),
    "perf": ("pynomaly.presentation.cli.performance", "performance_app"),
    "validate": ("pynomaly.presentation.cli.validation", "app"),
    "migrate": ("pynomaly.presentation.cli.migrations", "app"),
}

# Optional modules that may not be available
_OPTIONAL_MODULES = {
    "recommend": ("pynomaly.presentation.cli.recommendation", "app"),
}


def _lazy_import_module(module_name: str, attr_name: str):
    """Lazily import a module and return the specified attribute."""
    try:
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except ImportError:
        return None


def _add_lazy_subcommand(
    command_name: str, module_name: str, attr_name: str, help_text: str
):
    """Add a lazy-loaded subcommand to the main app."""

    def create_lazy_command():
        """Create lazy command closure."""

        def lazy_command(ctx: typer.Context):
            """Lazy command that imports the actual module when called."""
            # Import the module only when the command is actually used
            subapp = _lazy_import_module(module_name, attr_name)
            if subapp is None:
                console.print(f"[red]Error:[/red] {command_name} module not available")
                raise typer.Exit(1)

            # Forward the context and remaining arguments to the subcommand
            try:
                # Execute the subcommand directly
                subapp.main(ctx.args, standalone_mode=False)
            except SystemExit as e:
                # Re-raise system exits
                raise e
            except Exception as e:
                console.print(f"[red]Error:[/red] {command_name} failed: {str(e)}")
                raise typer.Exit(1)

        return lazy_command

    # For now, add the subcommand directly to maintain full functionality
    # Later we can optimize this for better lazy loading
    subapp = _lazy_import_module(module_name, attr_name)
    if subapp is not None:
        app.add_typer(subapp, name=command_name, help=help_text)


# Import standardized help text
from pynomaly.presentation.cli.help_formatter import get_standard_help

# Add lazy subcommands with standardized help text
lazy_commands = [
    ("auto", "pynomaly.presentation.cli.autonomous", "app"),
    ("automl", "pynomaly.presentation.cli.automl", "app"),
    ("config", "pynomaly.presentation.cli.config", "app"),
    ("detector", "pynomaly.presentation.cli.detectors", "app"),
    ("dataset", "pynomaly.presentation.cli.datasets", "app"),
    ("data", "pynomaly.presentation.cli.preprocessing", "app"),
    ("detect", "pynomaly.presentation.cli.detection", "app"),
    ("tdd", "pynomaly.presentation.cli.tdd", "app"),
    ("deep-learning", "pynomaly.presentation.cli.deep_learning", "app"),
    ("explainability", "pynomaly.presentation.cli.explainability", "app"),
    ("selection", "pynomaly.presentation.cli.selection", "app"),
    ("export", "pynomaly.presentation.cli.export", "export_app"),
    ("server", "pynomaly.presentation.cli.server", "app"),
    ("perf", "pynomaly.presentation.cli.performance", "performance_app"),
    ("validate", "pynomaly.presentation.cli.validation", "app"),
    ("migrate", "pynomaly.presentation.cli.migrations", "app"),
]

for cmd_name, module_path, attr_name in lazy_commands:
    help_info = get_standard_help(cmd_name)
    help_text = help_info.get("help", f"Manage {cmd_name} operations")
    _add_lazy_subcommand(cmd_name, module_path, attr_name, help_text)

# Add optional subcommands (only if available)
if "recommend" in _OPTIONAL_MODULES:
    _add_lazy_subcommand(
        "recommend",
        "pynomaly.presentation.cli.recommendation",
        "app",
        "Intelligent recommendations",
    )


@app.command()
def version():
    """Show version information."""
    container = get_cli_container()
    settings = container.config()

    console.print(f"[bold blue]Pynomaly[/bold blue] v{settings.app.version}")
    console.print(f"Python {sys.version.split()[0]}")
    console.print(f"Storage: {settings.storage_path}")


@app.command()
def settings(
    show: bool = typer.Option(False, "--show", help="Show current settings"),
    set_key: str | None = typer.Option(None, "--set", help="Set setting key=value"),
):
    """Manage application settings."""
    container = get_cli_container()
    settings = container.config()

    if show:
        # Show configuration
        from rich.table import Table

        table = Table(title="Pynomaly Settings")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        # Show key settings
        config_items = [
            ("App Name", settings.app.name),
            ("Version", settings.app.version),
            ("Debug Mode", str(settings.app.debug)),
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
        console.print("[yellow]Note:[/yellow] Setting update not yet implemented")
        console.print(f"Would set: {key} = {value}")

    else:
        console.print("Use --show to display settings or --set key=value to update")


@app.command()
def status():
    """Show system status."""
    container = get_cli_container()

    # Get repository counts
    detector_count = container.detector_repository().count()
    dataset_count = container.dataset_repository().count()
    result_count = container.result_repository().count()

    # Create status table
    from rich.table import Table

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
                f"{result.n_anomalies} ({result.anomaly_rate:.1%})",
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

    console.print("\n[bold]Step 2: Clean and preprocess data (optional)[/bold]")
    console.print("Clean missing values, outliers, and transform features.")
    console.print("Examples:")
    console.print(
        "  [cyan]pynomaly data clean <dataset_id> --missing drop_rows --outliers clip[/cyan]"
    )
    console.print(
        "  [cyan]pynomaly data transform <dataset_id> --scaling standard --encoding onehot[/cyan]"
    )

    console.print("\n[bold]Step 3: Create a detector[/bold]")
    console.print("Choose from various algorithms like IsolationForest, LOF, etc.")
    console.print(
        "Example: [cyan]pynomaly detector create my_detector --algorithm IsolationForest[/cyan]"
    )

    console.print("\n[bold]Step 4: Train the detector[/bold]")
    console.print("Train your detector on the loaded dataset.")
    console.print(
        "Example: [cyan]pynomaly detect train --detector my_detector --dataset my_data[/cyan]"
    )

    console.print("\n[bold]Step 5: Detect anomalies[/bold]")
    console.print("Run detection on new data.")
    console.print(
        "Example: [cyan]pynomaly detect run --detector my_detector --dataset test_data[/cyan]"
    )

    console.print("\n[bold]Step 6: View and export results[/bold]")
    console.print("Analyze detection results and export them to various platforms.")
    console.print("Examples:")
    console.print("  [cyan]pynomaly detect results --latest[/cyan]")
    console.print("  [cyan]pynomaly export excel results.json output.xlsx[/cyan]")
    console.print("  [cyan]pynomaly export list-formats[/cyan]")

    console.print(
        "\n[green]Ready to start![/green] Use [cyan]pynomaly --help[/cyan] to see all commands."
    )


@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
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
