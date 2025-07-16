"""Detector management CLI commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pynomaly_detection.domain.entities import Detector
from pynomaly_detection.presentation.cli.container import get_cli_container
from pynomaly_detection.presentation.cli.help_formatter import get_option_help, get_standard_help
from pynomaly_detection.presentation.cli.ux_improvements import CLIErrorHandler, CLIHelpers

# Get standardized help for this command group
_help_info = get_standard_help("detector")

app = typer.Typer(
    name="detector",
    help=_help_info["help"],
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


@app.command("list")
def list_detectors(
    algorithm: str | None = typer.Option(
        None,
        "--algorithm",
        "-a",
        help=get_option_help("algorithm"),
        rich_help_panel="Filters",
    ),
    fitted: bool | None = typer.Option(
        None,
        "--fitted",
        help="Show only trained/fitted detectors",
        rich_help_panel="Filters",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help=get_option_help("limit"),
        rich_help_panel="Display Options",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help=get_option_help("format"),
        rich_help_panel="Display Options",
    ),
):
    """List all detectors with filtering and display options.

    Examples:
      pynomaly detector list
      pynomaly detector list --algorithm IsolationForest
      pynomaly detector list --fitted --format json
      pynomaly detector list --limit 20 --format csv

    [bold]Examples:[/bold]
    • [cyan]pynomaly detector list --algorithm LOF --fitted --limit 5[/cyan]
    • [cyan]pynomaly detector list --format json > detectors.json[/cyan]
    """
    container = get_cli_container()
    detector_repo = container.detector_repository()

    # Get all detectors
    detectors = detector_repo.find_all()

    # Apply filters
    if algorithm:
        detectors = [d for d in detectors if d.algorithm == algorithm]
    if fitted is not None:
        detectors = [d for d in detectors if d.is_fitted == fitted]

    # Limit results
    detectors = detectors[:limit]

    if not detectors:
        console.print("[yellow]No detectors found.[/yellow]")
        console.print("\n[cyan]Create your first detector:[/cyan]")
        console.print(
            "  [white]pynomaly detector create my-detector --algorithm IsolationForest[/white]"
        )
        return

    # Convert to display format
    detector_data = []
    for detector in detectors:
        detector_data.append(
            {
                "id": str(detector.id)[:8],
                "name": detector.name,
                "algorithm": detector.algorithm_name,
                "fitted": "✓" if detector.is_fitted else "✗",
                "created": detector.created_at.strftime("%Y-%m-%d %H:%M"),
            }
        )

    # Display based on format
    if format == "table":
        CLIHelpers.display_enhanced_table(
            detector_data,
            ["id", "name", "algorithm", "fitted", "created"],
            title="Detectors",
            show_lines=True,
        )
    elif format == "json":
        console.print_json(detector_data)
    elif format == "csv":
        import csv
        import sys

        writer = csv.DictWriter(
            sys.stdout, fieldnames=["id", "name", "algorithm", "fitted", "created"]
        )
        writer.writeheader()
        writer.writerows(detector_data)
    else:
        CLIErrorHandler.invalid_format(format, ["table", "json", "csv"])


@app.command("create")
def create_detector(
    name: str = typer.Argument(..., help="Detector name"),
    algorithm: str = typer.Option(
        "IsolationForest", "--algorithm", "-a", help="Algorithm to use"
    ),
    description: str | None = typer.Option(
        None, "--description", "-d", help="Detector description"
    ),
    contamination: float = typer.Option(
        0.1, "--contamination", "-c", help="Expected contamination rate"
    ),
):
    """Create a new detector."""
    container = get_cli_container()
    detector_repo = container.detector_repository()

    # Get available algorithms
    try:
        from pynomaly_detection.infrastructure.adapters import PyODAdapter

        available_algorithms = PyODAdapter.list_algorithms()

        if algorithm not in available_algorithms:
            console.print(f"[red]Error:[/red] Algorithm '{algorithm}' not available")
            console.print(f"Available algorithms: {', '.join(available_algorithms)}")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to access PyOD adapter: {str(e)}")
        console.print("Creating detector without algorithm validation...")
        # Continue without validation for now

    # Create detector
    try:
        detector = Detector(
            name=name,
            algorithm_name=algorithm,
            parameters={"contamination": contamination},
        )

        # Set description in metadata if provided
        if description:
            detector.metadata["description"] = description

        detector_repo.save(detector)

        console.print(
            f"[green]✓[/green] Created detector '{name}' with ID: {detector.id}"
        )
        console.print(f"  Algorithm: {algorithm}")
        console.print(f"  Contamination: {contamination}")

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to create detector: {str(e)}")
        raise typer.Exit(1)


@app.command("show")
def show_detector(
    detector_id: str = typer.Argument(..., help="Detector ID, name, or partial match"),
    format: str = typer.Option(
        "panel", "--format", "-f", help="Output format: panel, json, table"
    ),
):
    """Show detailed information about a detector.

    [bold]Common Usage:[/bold]
    • Show by ID: [cyan]pynomaly detector show 123e4567[/cyan]
    • Show by name: [cyan]pynomaly detector show my-detector[/cyan]
    • Show partial match: [cyan]pynomaly detector show 123e[/cyan]
    • Export to JSON: [cyan]pynomaly detector show my-detector --format json[/cyan]
    """
    container = get_cli_container()
    detector_repo = container.detector_repository()

    # Find detector by ID, name, or partial match
    detectors = detector_repo.find_all()
    matching = []

    for d in detectors:
        if (
            str(d.id).startswith(detector_id)
            or detector_id.lower() in d.name.lower()
            or str(d.id) == detector_id
        ):
            matching.append(d)

    if not matching:
        detector_data = [
            {"id": str(d.id)[:8], "name": d.name, "algorithm": d.algorithm_name}
            for d in detectors
        ]
        CLIErrorHandler.detector_not_found(detector_id, detector_data)

    if len(matching) > 1:
        console.print(f"[yellow]Multiple detectors match '{detector_id}':[/yellow]")
        detector_data = [
            {"id": str(d.id)[:8], "name": d.name, "algorithm": d.algorithm_name}
            for d in matching
        ]
        CLIHelpers.display_enhanced_table(
            detector_data, ["id", "name", "algorithm"], title="Matching Detectors"
        )
        console.print("\n[cyan]Please use a more specific ID or name.[/cyan]")
        raise typer.Exit(1)

    detector = matching[0]

    # Prepare detector info
    detector_info = {
        "id": str(detector.id),
        "name": detector.name,
        "algorithm": detector.algorithm_name,
        "created": detector.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "fitted": detector.is_fitted,
        "description": getattr(detector, "description", None),
        "parameters": detector.parameters,
        "metadata": detector.metadata if detector.is_fitted else None,
    }

    # Display based on format
    if format == "json":
        console.print_json(detector_info)
    elif format == "table":
        # Display as key-value table
        table = Table(title=f"Detector: {detector.name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("ID", str(detector.id)[:8])
        table.add_row("Name", detector.name)
        table.add_row("Algorithm", detector.algorithm_name)
        table.add_row("Created", detector.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        table.add_row("Fitted", "✓ Yes" if detector.is_fitted else "✗ No")

        if detector.description:
            table.add_row("Description", detector.description)

        console.print(table)

        # Show parameters
        if detector.parameters:
            params_table = Table(title="Parameters")
            params_table.add_column("Parameter", style="cyan")
            params_table.add_column("Value", style="yellow")

            for key, value in detector.parameters.items():
                params_table.add_row(key, str(value))

            console.print(params_table)
    else:
        # Panel format (default)
        info_content = f"[bold]ID:[/bold] {detector.id}\n"
        info_content += f"[bold]Algorithm:[/bold] {detector.algorithm_name}\n"
        info_content += f"[bold]Created:[/bold] {detector.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        info_content += f"[bold]Status:[/bold] {'✓ Fitted' if detector.is_fitted else '✗ Not fitted'}\n"

        if detector.description:
            info_content += f"[bold]Description:[/bold] {detector.description}\n"

        if detector.parameters:
            info_content += "\n[bold]Parameters:[/bold]\n"
            for key, value in detector.parameters.items():
                info_content += f"  • {key}: {value}\n"

        if detector.is_fitted and detector.metadata:
            info_content += "\n[bold]Training Info:[/bold]\n"
            if "training_samples" in detector.metadata:
                info_content += (
                    f"  • Training samples: {detector.metadata['training_samples']}\n"
                )
            if "training_time_ms" in detector.metadata:
                info_content += (
                    f"  • Training time: {detector.metadata['training_time_ms']}ms\n"
                )
            if "feature_names" in detector.metadata:
                info_content += (
                    f"  • Features: {len(detector.metadata['feature_names'])}\n"
                )

        console.print(
            Panel(info_content, title=f"Detector: {detector.name}", border_style="blue")
        )


@app.command("delete")
def delete_detector(
    detector_id: str = typer.Argument(..., help="Detector ID, name, or partial match"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a detector with enhanced safety checks.

    [bold]Common Usage:[/bold]
    • Delete by ID: [cyan]pynomaly detector delete 123e4567[/cyan]
    • Delete by name: [cyan]pynomaly detector delete my-detector[/cyan]
    • Force delete: [cyan]pynomaly detector delete my-detector --force[/cyan]

    [bold]Safety:[/bold]
    • Always shows confirmation prompt unless --force is used
    • Displays detector details before deletion
    • Provides helpful error messages for non-existent detectors
    """
    container = get_cli_container()
    detector_repo = container.detector_repository()

    # Find detector by ID, name, or partial match
    detectors = detector_repo.find_all()
    matching = []

    for d in detectors:
        if (
            str(d.id).startswith(detector_id)
            or detector_id.lower() in d.name.lower()
            or str(d.id) == detector_id
        ):
            matching.append(d)

    if not matching:
        detector_data = [
            {"id": str(d.id)[:8], "name": d.name, "algorithm": d.algorithm_name}
            for d in detectors
        ]
        CLIErrorHandler.detector_not_found(detector_id, detector_data)

    if len(matching) > 1:
        console.print(f"[yellow]Multiple detectors match '{detector_id}':[/yellow]")
        detector_data = [
            {"id": str(d.id)[:8], "name": d.name, "algorithm": d.algorithm_name}
            for d in matching
        ]
        CLIHelpers.display_enhanced_table(
            detector_data, ["id", "name", "algorithm"], title="Matching Detectors"
        )
        console.print("\n[cyan]Please use a more specific ID or name.[/cyan]")
        raise typer.Exit(1)

    detector = matching[0]

    # Enhanced confirmation with details
    if not CLIHelpers.confirm_destructive_action(
        action="delete",
        resource_name=f"{detector.name} ({detector.algorithm_name})",
        resource_type="detector",
        force=force,
    ):
        console.print("[yellow]Deletion cancelled.[/yellow]")
        return

    # Delete detector
    try:
        success = detector_repo.delete(detector.id)

        if success:
            console.print(
                f"[green]✓[/green] Successfully deleted detector '{detector.name}'"
            )
        else:
            console.print("[red]Error:[/red] Failed to delete detector")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to delete detector: {str(e)}")
        raise typer.Exit(1)


@app.command("examples")
def show_examples():
    """Show common detector usage examples with rich formatting."""
    from pynomaly_detection.presentation.cli.help_formatter import format_rich_help

    examples = [
        "pynomaly detector create --name fraud_detector --algorithm IsolationForest",
        "pynomaly detector list --algorithm LOF --fitted",
        "pynomaly detector show my_detector --format json",
        "pynomaly detector clone detector_123 new_detector_name",
        "pynomaly detector delete old_detector --force",
    ]

    tips = [
        "Use partial IDs for faster commands (e.g., '123' instead of full UUID)",
        "Filter commands support multiple criteria for precise results",
        "Use --format json for programmatic integration",
        "Always backup important detectors before deletion",
    ]

    format_rich_help(
        title="Detector Management Examples",
        description=_help_info["description"],
        examples=examples,
        tips=tips,
    )


@app.command("algorithms")
def list_algorithms(
    category: str | None = typer.Option(
        None, "--category", "-c", help="Filter by algorithm category"
    ),
):
    """List available anomaly detection algorithms."""
    get_cli_container()

    try:
        from pynomaly_detection.infrastructure.adapters import PyODAdapter

        # Get algorithm info
        algorithms = PyODAdapter.list_algorithms()
        algorithm_info = PyODAdapter.get_algorithm_info()
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to access PyOD adapter: {str(e)}")
        console.print("No algorithms available - check PyOD installation")
        raise typer.Exit(1)

    # Create table
    table = Table(title="Available Anomaly Detection Algorithms")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Type", style="green")
    table.add_column("Description", style="dim")

    for algo in sorted(algorithms):
        info = algorithm_info.get(algo, {})
        algo_category = info.get("category", "Unknown")

        # Apply filter
        if category and category.lower() not in algo_category.lower():
            continue

        table.add_row(
            algo,
            algo_category,
            info.get("type", "Unknown"),
            info.get("description", "")[:50] + "...",
        )

    console.print(table)
    console.print(f"\nTotal algorithms: {len(algorithms)}")


@app.command("clone")
def clone_detector(
    detector_id: str = typer.Argument(..., help="Source detector ID (can be partial)"),
    new_name: str = typer.Argument(..., help="Name for cloned detector"),
):
    """Clone an existing detector."""
    container = get_cli_container()
    detector_repo = container.detector_repository()

    # Find source detector
    detectors = detector_repo.find_all()
    matching = [d for d in detectors if str(d.id).startswith(detector_id)]

    if not matching:
        console.print(
            f"[red]Error:[/red] No detector found with ID starting with '{detector_id}'"
        )
        raise typer.Exit(1)

    if len(matching) > 1:
        console.print(f"[red]Error:[/red] Multiple detectors match '{detector_id}':")
        for d in matching:
            console.print(f"  - {d.id}: {d.name}")
        raise typer.Exit(1)

    source = matching[0]

    # Create clone
    try:
        cloned = Detector(
            name=new_name,
            algorithm=source.algorithm,
            description=f"Cloned from {source.name}",
            parameters=source.parameters.copy() if source.parameters else {},
        )

        detector_repo.save(cloned)

        console.print(
            f"[green]✓[/green] Cloned detector '{source.name}' to '{new_name}'"
        )
        console.print(f"  New ID: {cloned.id}")

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to clone detector: {str(e)}")
        raise typer.Exit(1)


@app.command("interactive")
def interactive_detector_creation():
    """Interactive detector creation wizard.

    This command guides you through creating a detector step by step,
    with helpful prompts and validation.
    """
    console.print(
        Panel.fit(
            "[bold blue]Interactive Detector Creation[/bold blue]\n"
            "This wizard will help you create a new detector with the right settings.",
            title="Detector Wizard",
        )
    )

    # Step 1: Basic information
    console.print("\n[bold]Step 1: Basic Information[/bold]")

    from rich.prompt import Prompt

    name = Prompt.ask("Enter detector name", default="my-detector")

    # Step 2: Algorithm selection
    console.print("\n[bold]Step 2: Algorithm Selection[/bold]")

    try:
        from pynomaly_detection.infrastructure.adapters import PyODAdapter

        algorithms = PyODAdapter.list_algorithms()

        console.print("\n[cyan]Available algorithms:[/cyan]")
        for i, algo in enumerate(algorithms[:10], 1):  # Show first 10
            console.print(f"  {i}. {algo}")

        if len(algorithms) > 10:
            console.print(f"  ... and {len(algorithms) - 10} more")
            console.print(
                "\n[dim]Use 'pynomaly detector algorithms' to see all algorithms[/dim]"
            )

        algorithm = Prompt.ask(
            "Select algorithm", choices=algorithms, default="IsolationForest"
        )
    except Exception:
        console.print("[yellow]Using default algorithm (PyOD not available)[/yellow]")
        algorithm = "IsolationForest"

    # Step 3: Parameters
    console.print("\n[bold]Step 3: Parameters[/bold]")

    contamination = Prompt.ask(
        "Expected contamination rate (fraction of anomalies)", default="0.1"
    )

    try:
        contamination = float(contamination)
        if not 0.001 <= contamination <= 0.5:
            console.print(
                "[yellow]Warning: Contamination rate outside typical range (0.001-0.5)[/yellow]"
            )
    except ValueError:
        console.print("[red]Invalid contamination rate, using default 0.1[/red]")
        contamination = 0.1

    # Step 4: Optional description
    console.print("\n[bold]Step 4: Description (Optional)[/bold]")
    description = Prompt.ask("Enter description", default="")

    # Step 5: Confirmation
    console.print("\n[bold]Step 5: Confirmation[/bold]")
    console.print(f"[cyan]Name:[/cyan] {name}")
    console.print(f"[cyan]Algorithm:[/cyan] {algorithm}")
    console.print(f"[cyan]Contamination:[/cyan] {contamination}")
    if description:
        console.print(f"[cyan]Description:[/cyan] {description}")

    from rich.prompt import Confirm

    if not Confirm.ask("\nCreate this detector?"):
        console.print("[yellow]Detector creation cancelled.[/yellow]")
        return

    # Create detector
    try:
        container = get_cli_container()
        detector_repo = container.detector_repository()

        detector = Detector(
            name=name,
            algorithm_name=algorithm,
            parameters={"contamination": contamination},
        )

        if description:
            detector.metadata["description"] = description

        detector_repo.save(detector)

        console.print(
            Panel.fit(
                f"[bold green]✓ Successfully created detector![/bold green]\n"
                f"Name: {name}\n"
                f"ID: {detector.id}\n"
                f"Algorithm: {algorithm}",
                title="Success",
                border_style="green",
            )
        )

        console.print("\n[cyan]Next steps:[/cyan]")
        console.print(
            f"  • Train detector: [white]pynomaly detect train {name} <dataset>[/white]"
        )
        console.print(f"  • Show details: [white]pynomaly detector show {name}[/white]")

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to create detector: {str(e)}")
        raise typer.Exit(1)


@app.command("examples")
def show_examples():
    """Show common detector usage examples.

    This command provides practical examples of how to use detector commands
    in real-world scenarios.
    """
    examples = [
        {
            "title": "Basic Detector Creation",
            "description": "Create a simple detector with default settings",
            "command": "pynomaly detector create fraud-detector --algorithm IsolationForest",
        },
        {
            "title": "Custom Parameters",
            "description": "Create detector with custom contamination rate",
            "command": "pynomaly detector create anomaly-detector --algorithm LOF --contamination 0.05",
        },
        {
            "title": "List Filtered Detectors",
            "description": "Show only fitted detectors using IsolationForest",
            "command": "pynomaly detector list --algorithm IsolationForest --fitted",
        },
        {
            "title": "Export Detector List",
            "description": "Export detector information to CSV",
            "command": "pynomaly detector list --format csv > detectors.csv",
        },
        {
            "title": "Interactive Creation",
            "description": "Use the interactive wizard for guided setup",
            "command": "pynomaly detector interactive",
        },
        {
            "title": "Clone Existing Detector",
            "description": "Clone a detector with similar settings",
            "command": "pynomaly detector clone fraud-detector new-fraud-detector",
        },
    ]

    CLIHelpers.show_command_examples("detector", examples)
