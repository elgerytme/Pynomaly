"""
CLI User Experience Improvements

This module provides utilities and enhancements to improve the overall
user experience of the Pynomaly CLI.
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table

console = Console()


class CLIErrorHandler:
    """Centralized error handling with helpful suggestions and logging."""

    _session_id: str = "unknown"

    @staticmethod
    def _log_error(error_type: str, details: dict[str, Any]) -> None:
        """Log error details for debugging and monitoring."""
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "details": details,
            "session_id": getattr(CLIErrorHandler, "_session_id", "unknown")
        }

        # In production, this would log to a proper logging system
        # For now, we'll append to a local error log file
        try:
            log_path = Path.home() / ".pynomaly" / "error.log"
            log_path.parent.mkdir(exist_ok=True)

            with open(log_path, "a") as f:
                f.write(json.dumps(error_log) + "\n")
        except Exception:
            # Silently fail if logging fails
            pass

    @staticmethod
    def _show_support_info() -> None:
        """Show support and troubleshooting information."""
        console.print("\n[bold blue]Need Help?[/bold blue]")
        console.print("• Documentation: [cyan]https://docs.example.com[/cyan]")
        console.print("• GitHub Issues: [cyan]https://github.com/pynomaly/pynomaly/issues[/cyan]")
        console.print("• Community: [cyan]https://discord.gg/pynomaly[/cyan]")
        console.print("• Check logs: [cyan]~/.pynomaly/error.log[/cyan]")

    @classmethod
    def set_session_id(cls, session_id: str) -> None:
        """Set session ID for error tracking."""
        cls._session_id = session_id

    @classmethod
    def get_error_summary(cls) -> dict[str, Any]:
        """Get summary of recent errors for diagnostics."""
        try:
            log_path = Path.home() / ".pynomaly" / "error.log"
            if not log_path.exists():
                return {"total_errors": 0, "recent_errors": []}

            recent_errors = []
            with open(log_path) as f:
                for line in f:
                    try:
                        error_data = json.loads(line.strip())
                        recent_errors.append(error_data)
                    except json.JSONDecodeError:
                        continue

            # Get last 10 errors
            recent_errors = recent_errors[-10:]

            # Count error types
            error_counts = {}
            for error in recent_errors:
                error_type = error.get("error_type", "unknown")
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

            return {
                "total_errors": len(recent_errors),
                "recent_errors": recent_errors,
                "error_counts": error_counts
            }
        except Exception:
            return {"total_errors": 0, "recent_errors": [], "error": "Failed to read error log"}

    @staticmethod
    def detector_not_found(
        detector_query: str, available_detectors: list[dict[str, Any]]
    ) -> None:
        """Handle detector not found error with suggestions."""
        console.print(
            f"[red]❌ Error:[/red] No detector found matching '{detector_query}'"
        )

        # Log the error
        CLIErrorHandler._log_error("detector_not_found", {
            "query": detector_query,
            "available_count": len(available_detectors)
        })

        # Suggest similar detectors using fuzzy matching
        detector_names = [
            d.get("name", "") for d in available_detectors if d.get("name")
        ]
        similar = difflib.get_close_matches(
            detector_query, detector_names, n=3, cutoff=0.6
        )

        if similar:
            console.print("\n[yellow]🤔 Did you mean:[/yellow]")
            for suggestion in similar:
                matching_detector = next(
                    (d for d in available_detectors if d.get("name") == suggestion),
                    None,
                )
                if matching_detector:
                    detector_id = matching_detector.get("id", "unknown")[:8]
                    algorithm = matching_detector.get("algorithm", "Unknown")
                    console.print(f"  • {suggestion} ({detector_id}) - {algorithm}")

        # Show available detectors
        if available_detectors:
            console.print("\n[cyan]📋 Available detectors:[/cyan]")
            for detector in available_detectors[:5]:  # Show first 5
                name = detector.get("name", "Unknown")
                detector_id = detector.get("id", "unknown")[:8]
                algorithm = detector.get("algorithm", "Unknown")
                status = detector.get("status", "Unknown")
                status_icon = "✅" if status == "active" else "⏸️" if status == "paused" else "❓"
                console.print(f"  • {name} ({detector_id}) - {algorithm} {status_icon}")

            if len(available_detectors) > 5:
                console.print(f"  ... and {len(available_detectors) - 5} more")
                console.print("  💡 Use [white]pynomaly detector list --all[/white] to see all")

        # Provide creation option
        console.print("\n[cyan]🔧 Create new detector:[/cyan]")
        console.print("Run [white]pynomaly detector create --help[/white] for options")
        console.print("Quick start: [white]pynomaly detector create my-detector --algorithm IsolationForest[/white]")

        # Show support info
        CLIErrorHandler._show_support_info()

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

    # Create workflow helper
    workflow = WorkflowHelper("Pynomaly Setup Wizard")

    # Step 1: Environment validation
    def validate_environment(context):
        """Validate environment and dependencies."""
        console.print("\n[bold cyan]🔍 Validating Environment[/bold cyan]")

        # Check Python version
        import sys
        python_version = sys.version_info
        if python_version < (3, 8):
            console.print(f"[red]⚠️ Python {python_version.major}.{python_version.minor} detected. Python 3.8+ required.[/red]")
            return False

        console.print(f"[green]✓[/green] Python {python_version.major}.{python_version.minor} detected")

        # Check for required packages
        required_packages = ['pandas', 'numpy', 'scikit-learn', 'rich']
        for package in required_packages:
            try:
                __import__(package)
                console.print(f"[green]✓[/green] {package} available")
            except ImportError:
                console.print(f"[yellow]⚠️[/yellow] {package} not found (will be installed)")

        return True

    # Step 2: Data source configuration
    def configure_data_source(context):
        """Configure data source settings."""
        console.print("\n[bold cyan]📊 Data Source Configuration[/bold cyan]")

        # Get data file path
        data_path = Prompt.ask(
            "Enter the path to your data file (CSV, Parquet, JSON, etc.)",
            default="data.csv"
        )

        # Validate file exists
        if not Path(data_path).exists():
            console.print(f"[yellow]⚠️ Warning:[/yellow] File {data_path} does not exist.")

            # Show file suggestions if in same directory
            current_dir = Path(".")
            data_files = list(current_dir.glob("*.csv")) + list(current_dir.glob("*.parquet")) + list(current_dir.glob("*.json"))

            if data_files:
                console.print("\n[cyan]Found these data files in current directory:[/cyan]")
                for i, file in enumerate(data_files[:5], 1):
                    console.print(f"  {i}. {file.name}")

                use_existing = Confirm.ask("Use one of these files instead?")
                if use_existing:
                    selected = CLIHelpers.interactive_selection(
                        [{"name": f.name, "id": str(f)} for f in data_files],
                        "data file",
                        display_key="name",
                        id_key="id"
                    )
                    if selected:
                        data_path = selected

            if not Path(data_path).exists():
                if not Confirm.ask("Continue with setup anyway?"):
                    return False

        # Data format detection
        data_format = "csv"
        if data_path.endswith(".parquet"):
            data_format = "parquet"
        elif data_path.endswith(".json"):
            data_format = "json"
        elif data_path.endswith(".xlsx"):
            data_format = "excel"

        console.print(f"[green]✓[/green] Data format detected: {data_format}")

        # Data preprocessing options
        console.print("\n[bold]Data Preprocessing Options:[/bold]")
        handle_missing = Prompt.ask(
            "How to handle missing values?",
            choices=["drop", "fill_mean", "fill_median", "fill_mode", "skip"],
            default="fill_mean"
        )

        normalize_data = Confirm.ask("Normalize/standardize data?", default=True)

        return {
            "data_path": data_path,
            "data_format": data_format,
            "handle_missing": handle_missing,
            "normalize_data": normalize_data
        }

    # Step 3: Algorithm selection with intelligent recommendations
    def select_algorithm(context):
        """Select anomaly detection algorithm with recommendations."""
        console.print("\n[bold cyan]🤖 Algorithm Selection[/bold cyan]")

        # Show algorithm options with detailed information
        algorithms = [
            {
                "name": "IsolationForest",
                "description": "Fast, memory-efficient, good for general use",
                "best_for": "Large datasets, mixed data types",
                "performance": "⚡ Fast",
                "complexity": "🟢 Easy"
            },
            {
                "name": "LocalOutlierFactor",
                "description": "Density-based, good for local outliers",
                "best_for": "Small-medium datasets, local patterns",
                "performance": "🐌 Moderate",
                "complexity": "🟡 Medium"
            },
            {
                "name": "OneClassSVM",
                "description": "Robust, kernel-based approach",
                "best_for": "Non-linear patterns, robustness",
                "performance": "🐌 Slow",
                "complexity": "🔴 Complex"
            },
            {
                "name": "DBSCAN",
                "description": "Clustering-based, density-dependent",
                "best_for": "Clusters, varying densities",
                "performance": "⚡ Fast",
                "complexity": "🟡 Medium"
            }
        ]

        # Display algorithm comparison table
        table = Table(title="Algorithm Comparison", show_lines=True)
        table.add_column("Algorithm", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Best For", style="green")
        table.add_column("Performance", style="yellow")
        table.add_column("Complexity", style="magenta")

        for algo in algorithms:
            table.add_row(
                algo["name"],
                algo["description"],
                algo["best_for"],
                algo["performance"],
                algo["complexity"]
            )

        console.print(table)

        # Get user selection
        algorithm = Prompt.ask(
            "Select algorithm",
            choices=[algo["name"] for algo in algorithms],
            default="IsolationForest"
        )

        # Algorithm-specific configuration
        algo_config = {"algorithm": algorithm}

        if algorithm == "IsolationForest":
            n_estimators = int(Prompt.ask("Number of trees", default="100"))
            algo_config["n_estimators"] = n_estimators
        elif algorithm == "LocalOutlierFactor":
            n_neighbors = int(Prompt.ask("Number of neighbors", default="20"))
            algo_config["n_neighbors"] = n_neighbors
        elif algorithm == "OneClassSVM":
            kernel = Prompt.ask("Kernel type", choices=["rbf", "linear", "poly"], default="rbf")
            algo_config["kernel"] = kernel
        elif algorithm == "DBSCAN":
            eps = float(Prompt.ask("Epsilon (neighborhood radius)", default="0.5"))
            min_samples = int(Prompt.ask("Minimum samples", default="5"))
            algo_config["eps"] = eps
            algo_config["min_samples"] = min_samples

        return algo_config

    # Step 4: Advanced configuration
    def configure_advanced_settings(context):
        """Configure advanced settings."""
        console.print("\n[bold cyan]⚙️ Advanced Configuration[/bold cyan]")

        # Contamination rate
        contamination = float(Prompt.ask(
            "Expected contamination rate (0.01-0.5)",
            default="0.1"
        ))

        # Validation settings
        use_validation = Confirm.ask("Enable cross-validation?", default=True)
        cv_folds = 5
        if use_validation:
            cv_folds = int(Prompt.ask("Number of CV folds", default="5"))

        # Performance settings
        enable_parallel = Confirm.ask("Enable parallel processing?", default=True)

        # Monitoring settings
        enable_monitoring = Confirm.ask("Enable performance monitoring?", default=True)

        return {
            "contamination": contamination,
            "use_validation": use_validation,
            "cv_folds": cv_folds,
            "enable_parallel": enable_parallel,
            "enable_monitoring": enable_monitoring
        }

    # Step 5: Output configuration
    def configure_output(context):
        """Configure output settings."""
        console.print("\n[bold cyan]📤 Output Configuration[/bold cyan]")

        # Output format
        output_format = Prompt.ask(
            "Output format",
            choices=["csv", "json", "excel", "parquet"],
            default="csv"
        )

        # Output path
        output_path = Prompt.ask(
            "Output file path",
            default=f"pynomaly_results.{output_format}"
        )

        # Visualization options
        generate_plots = Confirm.ask("Generate visualization plots?", default=True)

        # Report generation
        generate_report = Confirm.ask("Generate detailed report?", default=True)

        return {
            "output_format": output_format,
            "output_path": output_path,
            "generate_plots": generate_plots,
            "generate_report": generate_report
        }

    # Step 6: Generate configuration and commands
    def generate_final_config(context):
        """Generate final configuration and commands."""
        console.print("\n[bold cyan]🔧 Generating Configuration[/bold cyan]")

        # Compile all settings
        config = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "wizard_version": "2.0",
                "type": "setup_wizard"
            },
            "data_source": context.get("data_source", {}),
            "algorithm": context.get("algorithm", {}),
            "advanced": context.get("advanced", {}),
            "output": context.get("output", {})
        }

        # Save configuration to file
        config_path = Path("pynomaly_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        console.print(f"[green]✓[/green] Configuration saved to {config_path}")

        # Generate CLI commands
        data_path = config["data_source"].get("data_path", "data.csv")
        algorithm = config["algorithm"].get("algorithm", "IsolationForest")
        contamination = config["advanced"].get("contamination", 0.1)
        output_path = config["output"].get("output_path", "results.csv")

        commands = [
            f"pynomaly dataset load {data_path} --name my-dataset",
            f"pynomaly detector create my-detector --algorithm {algorithm} --contamination {contamination}",
            "pynomaly detect train my-detector my-dataset",
            f"pynomaly detect run my-detector my-dataset --output {output_path}"
        ]

        context["commands"] = commands
        context["config"] = config

        return True

    # Add workflow steps
    workflow.add_step("validate_env", "Validating environment", validate_environment)
    workflow.add_step("data_source", "Configuring data source", configure_data_source)
    workflow.add_step("algorithm", "Selecting algorithm", select_algorithm)
    workflow.add_step("advanced", "Configuring advanced settings", configure_advanced_settings)
    workflow.add_step("output", "Configuring output", configure_output)
    workflow.add_step("generate", "Generating configuration", generate_final_config)

    # Execute workflow
    success = workflow.execute()

    if success:
        config = workflow.context.get("config")
        commands = workflow.context.get("commands", [])

        # Display final results
        console.print("\n[green]🎉 Setup Complete![/green]")
        console.print("\n[bold cyan]Next Steps:[/bold cyan]")

        for i, cmd in enumerate(commands, 1):
            console.print(f"  {i}. [white]{cmd}[/white]")

        console.print("\n[bold blue]Additional Resources:[/bold blue]")
        console.print("  • Configuration file: [white]pynomaly_config.json[/white]")
        console.print("  • Documentation: [white]pynomaly --help[/white]")
        console.print("  • Examples: [white]pynomaly quickstart[/white]")

        return config
    else:
        console.print("\n[red]Setup failed. Please try again.[/red]")
        return None


class ProgressIndicator:
    """Enhanced progress indicators for various operations."""

    def __init__(self, console: Console = None):
        """Initialize progress indicator with optional console."""
        self.console = console or Console()
        self._current_progress = None
        self._current_status = None

    def create_progress_bar(self, description: str = "Processing...") -> Progress:
        """Create a rich progress bar with multiple columns."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True
        )

    def create_spinner(self, message: str = "Processing...") -> Status:
        """Create a spinner for indefinite operations."""
        return Status(
            message,
            console=self.console,
            spinner="dots"
        )

    def track_file_upload(self, file_path: str, chunk_size: int = 8192):
        """Track file upload progress."""
        from pathlib import Path

        file_size = Path(file_path).stat().st_size

        with self.create_progress_bar("Uploading file...") as progress:
            task = progress.add_task(f"[cyan]Uploading {Path(file_path).name}...", total=file_size)

            uploaded = 0
            # Simulate file upload (in real implementation, this would be actual upload)
            import time
            while uploaded < file_size:
                chunk = min(chunk_size, file_size - uploaded)
                time.sleep(0.01)  # Simulate network delay
                uploaded += chunk
                progress.update(task, completed=uploaded)

    def track_training_progress(self, total_epochs: int, current_metrics: dict = None):
        """Track model training progress with live metrics."""
        with self.create_progress_bar("Training model...") as progress:
            task = progress.add_task("[cyan]Training...", total=total_epochs)

            # In real implementation, this would track actual training
            import time
            for epoch in range(total_epochs):
                time.sleep(0.1)  # Simulate training time

                # Update progress
                progress.update(
                    task,
                    completed=epoch + 1,
                    description=f"[cyan]Epoch {epoch + 1}/{total_epochs}"
                )

                # Show live metrics if provided
                if current_metrics:
                    metrics_str = " | ".join([f"{k}: {v:.3f}" for k, v in current_metrics.items()])
                    progress.update(task, description=f"[cyan]Epoch {epoch + 1}/{total_epochs} | {metrics_str}")

    def track_data_processing(self, steps: list[str], step_function=None):
        """Track multi-step data processing with detailed progress."""
        with self.create_progress_bar("Processing data...") as progress:
            main_task = progress.add_task("[cyan]Overall Progress", total=len(steps))

            for i, step in enumerate(steps):
                step_task = progress.add_task(f"[yellow]{step}...", total=100)

                # Execute step function if provided
                if step_function:
                    result = step_function(i, step, lambda p: progress.update(step_task, completed=p))
                    if not result:
                        progress.update(step_task, description=f"[red]Failed: {step}")
                        return False
                else:
                    # Simulate step execution
                    import time
                    for j in range(100):
                        time.sleep(0.001)
                        progress.update(step_task, completed=j + 1)

                progress.update(step_task, description=f"[green]✓ {step}")
                progress.update(main_task, completed=i + 1)

        return True

    def track_batch_operation(self, items: list, operation_name: str, batch_function=None):
        """Track batch operations with item-by-item progress."""
        with self.create_progress_bar(f"{operation_name}...") as progress:
            task = progress.add_task(f"[cyan]{operation_name}", total=len(items))

            results = []
            for i, item in enumerate(items):
                # Update description with current item
                item_name = str(item)[:30] + "..." if len(str(item)) > 30 else str(item)
                progress.update(task, description=f"[cyan]{operation_name}: {item_name}")

                # Execute batch function if provided
                if batch_function:
                    result = batch_function(item, i)
                    results.append(result)
                else:
                    # Simulate operation
                    import time
                    time.sleep(0.05)
                    results.append(f"processed_{item}")

                progress.update(task, completed=i + 1)

            progress.update(task, description=f"[green]✓ {operation_name} completed")
            return results

    def show_live_metrics(self, metrics_function, duration: int = 10):
        """Show live updating metrics."""
        from rich.layout import Layout
        from rich.text import Text

        layout = Layout()

        with Live(layout, refresh_per_second=2, console=self.console) as live:
            import time
            start_time = time.time()

            while time.time() - start_time < duration:
                # Get current metrics
                metrics = metrics_function() if metrics_function else {}

                # Create metrics display
                metrics_text = Text()
                metrics_text.append("Live Metrics\n\n", style="bold cyan")

                for key, value in metrics.items():
                    metrics_text.append(f"{key}: ", style="white")
                    metrics_text.append(f"{value}\n", style="green")

                layout.update(Panel(metrics_text, title="System Metrics", border_style="cyan"))
                time.sleep(0.5)

    def show_comparison_progress(self, algorithms: list[str], test_function=None):
        """Show progress for comparing multiple algorithms."""
        with self.create_progress_bar("Comparing algorithms...") as progress:
            main_task = progress.add_task("[cyan]Overall comparison", total=len(algorithms))

            results = {}
            for i, algorithm in enumerate(algorithms):
                # Create task for this algorithm
                algo_task = progress.add_task(f"[yellow]Testing {algorithm}...", total=100)

                # Execute test function if provided
                if test_function:
                    result = test_function(algorithm, lambda p: progress.update(algo_task, completed=p))
                    results[algorithm] = result
                else:
                    # Simulate algorithm testing
                    import time
                    for j in range(100):
                        time.sleep(0.01)
                        progress.update(algo_task, completed=j + 1)
                    results[algorithm] = {"accuracy": 0.85 + (i * 0.03), "time": 10 + (i * 2)}

                progress.update(algo_task, description=f"[green]✓ {algorithm} completed")
                progress.update(main_task, completed=i + 1)

            return results

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._current_progress:
            self._current_progress.stop()
        if self._current_status:
            self._current_status.stop()


# Global progress indicator instance
progress_indicator = ProgressIndicator()
