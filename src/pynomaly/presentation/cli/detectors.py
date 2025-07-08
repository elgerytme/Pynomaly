"""Detector management CLI commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from pynomaly.domain.entities import Detector
from pynomaly.presentation.cli.container import get_cli_container

app = typer.Typer()
console = Console()


@app.command("list")
def list_detectors(
    algorithm: str
    | None = typer.Option(None, "--algorithm", "-a", help="Filter by algorithm"),
    fitted: bool
    | None = typer.Option(None, "--fitted", help="Filter by fitted status"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to show"),
):
    """List all detectors."""
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
        console.print("No detectors found.")
        return

    # Create table
    table = Table(title="Detectors")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Algorithm", style="yellow")
    table.add_column("Fitted", style="blue")
    table.add_column("Created", style="dim")

    for detector in detectors:
        table.add_row(
            str(detector.id)[:8],
            detector.name,
            detector.algorithm_name,
            "✓" if detector.is_fitted else "✗",
            detector.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@app.command("create")
def create_detector(
    name: str = typer.Argument(..., help="Detector name"),
    algorithm: str = typer.Option(
        "IsolationForest", "--algorithm", "-a", help="Algorithm to use"
    ),
    description: str
    | None = typer.Option(None, "--description", "-d", help="Detector description"),
    contamination: float = typer.Option(
        0.1, "--contamination", "-c", help="Expected contamination rate"
    ),
):
    """Create a new detector."""
    container = get_cli_container()
    detector_repo = container.detector_repository()

    # Get available algorithms
    try:
        from pynomaly.infrastructure.adapters import PyODAdapter

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
    detector_id: str = typer.Argument(..., help="Detector ID (can be partial)"),
):
    """Show detector details."""
    container = get_cli_container()
    detector_repo = container.detector_repository()

    # Find detector by partial ID
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

    detector = matching[0]

    # Display detector info
    console.print(f"\n[bold]Detector: {detector.name}[/bold]")
    console.print(f"ID: {detector.id}")
    console.print(f"Algorithm: {detector.algorithm}")
    console.print(f"Created: {detector.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"Fitted: {'Yes' if detector.is_fitted else 'No'}")

    if detector.description:
        console.print(f"Description: {detector.description}")

    if detector.parameters:
        console.print("\n[bold]Parameters:[/bold]")
        for key, value in detector.parameters.items():
            console.print(f"  {key}: {value}")

    if detector.is_fitted and detector.metadata:
        console.print("\n[bold]Training Info:[/bold]")
        if "training_samples" in detector.metadata:
            console.print(
                f"  Training samples: {detector.metadata['training_samples']}"
            )
        if "training_time_ms" in detector.metadata:
            console.print(f"  Training time: {detector.metadata['training_time_ms']}ms")
        if "feature_names" in detector.metadata:
            console.print(f"  Features: {len(detector.metadata['feature_names'])}")


@app.command("delete")
def delete_detector(
    detector_id: str = typer.Argument(..., help="Detector ID (can be partial)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a detector."""
    container = get_cli_container()
    detector_repo = container.detector_repository()

    # Find detector by partial ID
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

    detector = matching[0]

    # Confirm deletion
    if not force:
        confirm = typer.confirm(f"Delete detector '{detector.name}' ({detector.id})?")
        if not confirm:
            console.print("Deletion cancelled.")
            return

    # Delete detector
    success = detector_repo.delete(detector.id)

    if success:
        console.print(f"[green]✓[/green] Deleted detector '{detector.name}'")
    else:
        console.print("[red]Error:[/red] Failed to delete detector")
        raise typer.Exit(1)


@app.command("algorithms")
def list_algorithms(
    category: str
    | None = typer.Option(None, "--category", "-c", help="Filter by category"),
):
    """List available algorithms."""
    get_cli_container()

    try:
        from pynomaly.infrastructure.adapters import PyODAdapter

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
