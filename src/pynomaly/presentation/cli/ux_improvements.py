"""
CLI User Experience Improvements

This module provides utilities and enhancements to improve the overall
user experience of the Pynomaly CLI.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

console = Console()


class CLIErrorHandler:
    """Centralized error handling with helpful suggestions."""

    @staticmethod
    def detector_not_found(
        detector_query: str, available_detectors: list[dict[str, Any]]
    ) -> None:
        """Handle detector not found error with suggestions."""
        console.print(
            f"[red]Error:[/red] No detector found matching '{detector_query}'"
        )

        # Suggest similar detectors using fuzzy matching
        detector_names = [
            d.get("name", "") for d in available_detectors if d.get("name")
        ]
        similar = difflib.get_close_matches(
            detector_query, detector_names, n=3, cutoff=0.6
        )

        if similar:
            console.print("\n[yellow]Did you mean:[/yellow]")
            for suggestion in similar:
                matching_detector = next(
                    (d for d in available_detectors if d.get("name") == suggestion),
                    None,
                )
                if matching_detector:
                    detector_id = matching_detector.get("id", "unknown")[:8]
                    console.print(f"  • {suggestion} ({detector_id})")

        # Show available detectors
        if available_detectors:
            console.print("\n[cyan]Available detectors:[/cyan]")
            for detector in available_detectors[:5]:  # Show first 5
                name = detector.get("name", "Unknown")
                detector_id = detector.get("id", "unknown")[:8]
                algorithm = detector.get("algorithm", "Unknown")
                console.print(f"  • {name} ({detector_id}) - {algorithm}")

            if len(available_detectors) > 5:
                console.print(f"  ... and {len(available_detectors) - 5} more")

        # Provide creation option
        console.print("\n[cyan]Create new detector:[/cyan]")
        console.print("Run [white]pynomaly detector create --help[/white] for options")

        raise typer.Exit(1)

    @staticmethod
    def dataset_not_found(
        dataset_query: str, available_datasets: list[dict[str, Any]]
    ) -> None:
        """Handle dataset not found error with suggestions."""
        console.print(f"[red]Error:[/red] No dataset found matching '{dataset_query}'")

        # Suggest similar datasets
        dataset_names = [d.get("name", "") for d in available_datasets if d.get("name")]
        similar = difflib.get_close_matches(
            dataset_query, dataset_names, n=3, cutoff=0.6
        )

        if similar:
            console.print("\n[yellow]Did you mean:[/yellow]")
            for suggestion in similar:
                matching_dataset = next(
                    (d for d in available_datasets if d.get("name") == suggestion), None
                )
                if matching_dataset:
                    dataset_id = matching_dataset.get("id", "unknown")[:8]
                    console.print(f"  • {suggestion} ({dataset_id})")

        # Show available datasets
        if available_datasets:
            console.print("\n[cyan]Available datasets:[/cyan]")
            for dataset in available_datasets[:5]:  # Show first 5
                name = dataset.get("name", "Unknown")
                dataset_id = dataset.get("id", "unknown")[:8]
                size = dataset.get("size", "Unknown")
                console.print(f"  • {name} ({dataset_id}) - {size} rows")

            if len(available_datasets) > 5:
                console.print(f"  ... and {len(available_datasets) - 5} more")

        # Provide creation option
        console.print("\n[cyan]Load new dataset:[/cyan]")
        console.print("Run [white]pynomaly dataset load --help[/white] for options")

        raise typer.Exit(1)

    @staticmethod
    def file_not_found(file_path: str) -> None:
        """Handle file not found error with suggestions."""
        console.print(f"[red]Error:[/red] File not found: {file_path}")

        # Check if similar files exist
        path = Path(file_path)
        if path.parent.exists():
            similar_files = []
            for file in path.parent.iterdir():
                if file.is_file() and file.suffix == path.suffix:
                    similar_files.append(file.name)

            if similar_files:
                similar = difflib.get_close_matches(
                    path.name, similar_files, n=3, cutoff=0.6
                )
                if similar:
                    console.print("\n[yellow]Similar files in directory:[/yellow]")
                    for suggestion in similar:
                        console.print(f"  • {path.parent / suggestion}")

        console.print("\n[cyan]Tips:[/cyan]")
        console.print("  • Check the file path is correct")
        console.print("  • Ensure the file exists and is accessible")
        console.print("  • Use absolute paths to avoid confusion")

        raise typer.Exit(1)

    @staticmethod
    def invalid_format(format_value: str, supported_formats: list[str]) -> None:
        """Handle invalid format error with suggestions."""
        console.print(f"[red]Error:[/red] Invalid format '{format_value}'")

        # Suggest similar formats
        similar = difflib.get_close_matches(
            format_value, supported_formats, n=3, cutoff=0.6
        )
        if similar:
            console.print("\n[yellow]Did you mean:[/yellow]")
            for suggestion in similar:
                console.print(f"  • {suggestion}")

        console.print("\n[cyan]Supported formats:[/cyan]")
        for fmt in supported_formats:
            console.print(f"  • {fmt}")

        raise typer.Exit(1)


class CLIHelpers:
    """Helper utilities for CLI user experience."""

    @staticmethod
    def confirm_destructive_action(
        action: str,
        resource_name: str,
        resource_type: str = "resource",
        force: bool = False,
    ) -> bool:
        """Confirm destructive actions with detailed information."""
        if force:
            return True

        console.print(f"[yellow]⚠️  About to {action} {resource_type}:[/yellow]")
        console.print(f"  Name: {resource_name}")

        return Confirm.ask(f"Are you sure you want to {action} this {resource_type}?")

    @staticmethod
    def show_progress_with_steps(steps: list[str], step_function) -> None:
        """Show progress through multiple steps with rich formatting."""
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(steps))

            for i, step in enumerate(steps):
                progress.update(task, description=f"[cyan]{step}...", completed=i)
                result = step_function(i, step)
                if not result:
                    console.print(f"[red]Failed at step: {step}[/red]")
                    return False
                progress.update(task, completed=i + 1)

            progress.update(task, description="[green]Complete!", completed=len(steps))
        return True

    @staticmethod
    def display_enhanced_table(
        data: list[dict[str, Any]],
        columns: list[str],
        title: str = None,
        show_lines: bool = True,
    ) -> None:
        """Display data in an enhanced table format."""
        if not data:
            console.print(
                f"[yellow]No {title.lower() if title else 'data'} found.[/yellow]"
            )
            return

        table = Table(title=title, show_lines=show_lines)

        # Add columns
        for col in columns:
            table.add_column(col.title(), style="cyan")

        # Add rows
        for row in data:
            table.add_row(*[str(row.get(col, "-")) for col in columns])

        console.print(table)

    @staticmethod
    def show_command_examples(command: str, examples: list[dict[str, str]]) -> None:
        """Display command examples with syntax highlighting."""
        console.print(f"\n[bold blue]Examples for {command}:[/bold blue]")

        for example in examples:
            console.print(f"\n[bold]{example['title']}:[/bold]")
            console.print(f"[dim]{example['description']}[/dim]")

            # Show command with syntax highlighting
            syntax = Syntax(
                example["command"], "bash", theme="monokai", line_numbers=False
            )
            console.print(syntax)

    @staticmethod
    def interactive_selection(
        items: list[dict[str, Any]],
        item_type: str,
        display_key: str = "name",
        id_key: str = "id",
    ) -> str | None:
        """Interactive selection from a list of items."""
        if not items:
            console.print(f"[yellow]No {item_type}s available.[/yellow]")
            return None

        # Display options
        console.print(f"\n[cyan]Available {item_type}s:[/cyan]")
        for i, item in enumerate(items, 1):
            display_name = item.get(display_key, "Unknown")
            item_id = item.get(id_key, "unknown")[:8]
            console.print(f"  {i}. {display_name} ({item_id})")

        # Get user selection
        while True:
            try:
                choice = Prompt.ask(
                    f"Select {item_type} (1-{len(items)} or name/ID)", default="1"
                )

                # Try to parse as number
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(items):
                        return items[index][id_key]
                except ValueError:
                    pass

                # Try to match by name or ID
                for item in items:
                    if (
                        choice.lower() in item.get(display_key, "").lower()
                        or choice.lower() in item.get(id_key, "").lower()
                    ):
                        return item[id_key]

                console.print(f"[red]Invalid selection: {choice}[/red]")

            except KeyboardInterrupt:
                console.print("\n[yellow]Selection cancelled.[/yellow]")
                return None


class WorkflowHelper:
    """Helper for managing multi-step workflows."""

    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.steps = []
        self.current_step = 0
        self.context = {}

    def add_step(self, name: str, description: str, function, **kwargs):
        """Add a step to the workflow."""
        self.steps.append(
            {
                "name": name,
                "description": description,
                "function": function,
                "kwargs": kwargs,
            }
        )

    def execute(self) -> bool:
        """Execute the workflow with progress tracking."""
        console.print(
            Panel.fit(
                f"[bold blue]Starting workflow: {self.workflow_name}[/bold blue]",
                title="Workflow",
            )
        )

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Executing workflow...", total=len(self.steps)
            )

            for i, step in enumerate(self.steps):
                progress.update(
                    task, description=f"[cyan]{step['description']}...", completed=i
                )

                try:
                    result = step["function"](self.context, **step["kwargs"])
                    if not result:
                        console.print(f"[red]Failed at step: {step['name']}[/red]")
                        return False

                    # Update context with result if it's a dict
                    if isinstance(result, dict):
                        self.context.update(result)

                    progress.update(task, completed=i + 1)

                except Exception as e:
                    console.print(f"[red]Error in step {step['name']}: {str(e)}[/red]")
                    return False

            progress.update(
                task,
                description="[green]Workflow completed!",
                completed=len(self.steps),
            )

        console.print(
            Panel.fit(
                f"[bold green]Workflow completed successfully: {self.workflow_name}[/bold green]",
                title="Success",
            )
        )

        return True


def create_setup_wizard():
    """Create an interactive setup wizard for new users."""
    console.print(
        Panel.fit(
            "[bold blue]Welcome to Pynomaly![/bold blue]\n"
            "This wizard will help you set up your first anomaly detection workflow.",
            title="Setup Wizard",
        )
    )

    # Step 1: Data source
    console.print("\n[bold]Step 1: Data Source[/bold]")
    data_path = Prompt.ask(
        "Enter the path to your data file (CSV, Parquet, etc.)", default="data.csv"
    )

    if not Path(data_path).exists():
        console.print(f"[yellow]Warning:[/yellow] File {data_path} does not exist.")
        if not Confirm.ask("Continue with setup anyway?"):
            return None

    # Step 2: Algorithm selection
    console.print("\n[bold]Step 2: Algorithm Selection[/bold]")
    console.print("Choose an anomaly detection algorithm:")
    console.print("  1. IsolationForest - Fast, good for general use")
    console.print("  2. LOF - Good for local outliers")
    console.print("  3. OneClassSVM - Robust, slower")
    console.print("  4. DBSCAN - Clustering-based")

    algorithm = Prompt.ask(
        "Select algorithm",
        choices=["IsolationForest", "LOF", "OneClassSVM", "DBSCAN"],
        default="IsolationForest",
    )

    # Step 3: Configuration
    console.print("\n[bold]Step 3: Configuration[/bold]")
    contamination = Prompt.ask("Expected contamination rate (0.01-0.5)", default="0.1")

    # Step 4: Output options
    console.print("\n[bold]Step 4: Output Options[/bold]")
    output_format = Prompt.ask(
        "Output format", choices=["csv", "json", "excel"], default="csv"
    )

    # Generate configuration
    config = {
        "data_path": data_path,
        "algorithm": algorithm,
        "contamination": float(contamination),
        "output_format": output_format,
    }

    console.print("\n[green]Setup complete![/green]")
    console.print("Generated configuration:")
    console.print_json(config)

    return config
