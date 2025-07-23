"""CLI commands for training automation.

This module provides comprehensive CLI commands for:
- Training job management and monitoring
- Quick optimization workflows
- Hyperparameter tuning configuration
- Experiment tracking and analysis
"""

"""
TODO: This file needs dependency injection refactoring.
Replace direct monorepo imports with dependency injection.
Use interfaces/shared/base_entity.py for abstractions.
"""



from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from interfaces.application.services.training_automation_service import (
    OptimizationStrategy,
    PruningStrategy,
    TrainingAutomationService,
    TrainingConfiguration,
    TrainingStatus,
)
from monorepo.infrastructure.adapters.model_trainer_adapter import (
    create_model_trainer_adapter,
)
from monorepo.infrastructure.persistence.training_job_repository import (
    create_training_job_repository,
)

# Create CLI app
app = typer.Typer(name="training", help="Training automation and optimization commands")
console = Console()
logger = logging.getLogger(__name__)


def get_training_service() -> TrainingAutomationService:
    """Get training automation service."""
    repository = create_training_job_repository()
    trainer = create_model_trainer_adapter()
    return TrainingAutomationService(repository, trainer)


@app.command("create")
def create_training_job(
    name: str = typer.Argument(..., help="Training job name"),
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    algorithms: list[str] | None = typer.Option(
        None, "--algorithm", "-a", help="Target algorithms"
    ),
    max_trials: int = typer.Option(
        100, "--max-trials", "-t", help="Maximum optimization trials"
    ),
    timeout: int = typer.Option(60, "--timeout", "-T", help="Timeout in minutes"),
    strategy: OptimizationStrategy = typer.Option(
        OptimizationStrategy.TPE, "--strategy", "-s", help="Optimization strategy"
    ),
    pruning: PruningStrategy = typer.Option(
        PruningStrategy.MEDIAN, "--pruning", "-p", help="Pruning strategy"
    ),
    metric: str = typer.Option(
        "roc_auc", "--metric", "-m", help="Primary optimization metric"
    ),
    cv_folds: int = typer.Option(5, "--cv-folds", "-k", help="Cross-validation folds"),
    experiment: str | None = typer.Option(
        None, "--experiment", "-e", help="Experiment name"
    ),
    auto_start: bool = typer.Option(False, "--start", help="Start job immediately"),
    save_config: str | None = typer.Option(
        None, "--save-config", help="Save configuration to file"
    ),
):
    """Create a new training job with specified configuration."""

    async def _create_job():
        try:
            service = get_training_service()

            # Create configuration
            config = TrainingConfiguration(
                max_trials=max_trials,
                timeout_minutes=timeout,
                optimization_strategy=strategy,
                pruning_strategy=pruning,
                primary_metric=metric,
                cross_validation_folds=cv_folds,
                experiment_name=experiment,
                track_artifacts=True,
                save_models=True,
            )

            # Save configuration if requested
            if save_config:
                config_dict = {
                    "max_trials": max_trials,
                    "timeout_minutes": timeout,
                    "optimization_strategy": strategy.value,
                    "pruning_strategy": pruning.value,
                    "primary_metric": metric,
                    "cross_validation_folds": cv_folds,
                    "experiment_name": experiment,
                }

                with open(save_config, "w") as f:
                    json.dump(config_dict, f, indent=2)

                rprint(f"[green]Configuration saved to {save_config}[/green]")

            # Create job
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Creating training job...", total=None)

                job = await service.create_training_job(
                    name=name,
                    dataset_id=dataset_id,
                    configuration=config,
                    target_algorithms=algorithms,
                )

                progress.update(task, description="Training job created successfully")

            # Display job information
            table = Table(title=f"Training Job Created: {job.job_id}")
            table.add_column("Property", style="bold blue")
            table.add_column("Value", style="green")

            table.add_row("Job ID", job.job_id)
            table.add_row("Name", job.name)
            table.add_row("Status", job.status.value)
            table.add_row("Dataset ID", job.dataset_id)
            table.add_row("Target Algorithms", ", ".join(job.target_algorithms))
            table.add_row("Max Trials", str(max_trials))
            table.add_row("Timeout", f"{timeout} minutes")
            table.add_row("Strategy", strategy.value)
            table.add_row("Primary Metric", metric)

            console.print(table)

            # Start job if requested
            if auto_start:
                rprint("[yellow]Starting training job...[/yellow]")
                await service.start_training_job(job.job_id)
                rprint(f"[green]Training job {job.job_id} started successfully[/green]")
            else:
                rprint(
                    f"[blue]To start the job, run: mlops training start {job.job_id}[/blue]"
                )

        except Exception as e:
            console.print(f"[red]Error creating training job: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_create_job())


@app.command("start")
def start_training_job(
    job_id: str = typer.Argument(..., help="Training job ID"),
    monitor: bool = typer.Option(False, "--monitor", "-m", help="Monitor job progress"),
):
    """Start a training job."""

    async def _start_job():
        try:
            service = get_training_service()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Starting training job...", total=None)

                await service.start_training_job(job_id)

                progress.update(task, description="Training job started successfully")

            rprint(f"[green]Training job {job_id} started successfully[/green]")

            if monitor:
                rprint(
                    "[blue]Monitoring job progress (Ctrl+C to stop monitoring)...[/blue]"
                )
                await _monitor_job(service, job_id)

        except Exception as e:
            console.print(f"[red]Error starting training job: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_start_job())


@app.command("status")
def get_job_status(
    job_id: str = typer.Argument(..., help="Training job ID"),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed information"
    ),
    metrics: bool = typer.Option(
        False, "--metrics", "-m", help="Show training metrics"
    ),
):
    """Get training job status and information."""

    async def _get_status():
        try:
            service = get_training_service()
            job = await service.get_job_status(job_id)

            if not job:
                console.print(f"[red]Training job {job_id} not found[/red]")
                raise typer.Exit(1)

            # Basic status table
            table = Table(title=f"Training Job Status: {job_id}")
            table.add_column("Property", style="bold blue")
            table.add_column("Value", style="green")

            table.add_row("Name", job.name)
            table.add_row("Status", _format_status(job.status))
            table.add_row("Dataset ID", job.dataset_id)
            table.add_row("Created", job.created_at.strftime("%Y-%m-%d %H:%M:%S"))

            if job.started_at:
                table.add_row("Started", job.started_at.strftime("%Y-%m-%d %H:%M:%S"))

            if job.completed_at:
                table.add_row(
                    "Completed", job.completed_at.strftime("%Y-%m-%d %H:%M:%S")
                )
                table.add_row("Duration", f"{job.execution_time_seconds:.1f} seconds")

            if job.error_message:
                table.add_row("Error", job.error_message)

            console.print(table)

            # Detailed information
            if detailed:
                detail_table = Table(title="Detailed Information")
                detail_table.add_column("Property", style="bold blue")
                detail_table.add_column("Value", style="green")

                detail_table.add_row(
                    "Target Algorithms", ", ".join(job.target_algorithms)
                )
                detail_table.add_row("Total Trials", str(job.total_trials))
                detail_table.add_row("Successful Trials", str(job.successful_trials))
                detail_table.add_row("Failed Trials", str(job.failed_trials))

                if job.total_trials > 0:
                    success_rate = (job.successful_trials / job.total_trials) * 100
                    detail_table.add_row("Success Rate", f"{success_rate:.1f}%")

                if job.best_score is not None:
                    detail_table.add_row("Best Score", f"{job.best_score:.4f}")

                if job.best_parameters:
                    params_str = json.dumps(job.best_parameters, indent=2)
                    detail_table.add_row("Best Parameters", params_str)

                if job.model_path:
                    detail_table.add_row("Model Path", job.model_path)

                if job.experiment_id:
                    detail_table.add_row("Experiment ID", job.experiment_id)

                console.print(detail_table)

            # Training metrics
            if metrics:
                try:
                    job_metrics = await service.get_training_metrics(job_id)

                    metrics_table = Table(title="Training Metrics")
                    metrics_table.add_column("Metric", style="bold blue")
                    metrics_table.add_column("Value", style="green")

                    metrics_table.add_row(
                        "Success Rate", f"{job_metrics['success_rate']:.1%}"
                    )
                    metrics_table.add_row(
                        "Best Score",
                        (
                            f"{job_metrics['best_score']:.4f}"
                            if job_metrics["best_score"]
                            else "N/A"
                        ),
                    )
                    metrics_table.add_row(
                        "Best Algorithm", job_metrics["best_algorithm"] or "N/A"
                    )

                    console.print(metrics_table)

                    # Trial history summary
                    if job_metrics["trial_history"]:
                        rprint(
                            f"[blue]Trial History: {len(job_metrics['trial_history'])} trials[/blue]"
                        )

                except Exception as e:
                    console.print(f"[yellow]Could not retrieve metrics: {e}[/yellow]")

        except Exception as e:
            console.print(f"[red]Error getting job status: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_get_status())


@app.command("list")
def list_training_jobs(
    status_filter: TrainingStatus | None = typer.Option(
        None, "--status", "-s", help="Filter by status"
    ),
    limit: int = typer.Option(
        20, "--limit", "-l", help="Maximum number of jobs to show"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed information"
    ),
):
    """List training jobs."""

    async def _list_jobs():
        try:
            service = get_training_service()
            jobs = await service.list_training_jobs(status_filter, limit)

            if not jobs:
                console.print("[yellow]No training jobs found[/yellow]")
                return

            if detailed:
                # Detailed table
                table = Table(title=f"Training Jobs ({len(jobs)} found)")
                table.add_column("Job ID", style="bold blue")
                table.add_column("Name", style="green")
                table.add_column("Status", style="yellow")
                table.add_column("Dataset", style="cyan")
                table.add_column("Algorithms", style="magenta")
                table.add_column("Trials", style="white")
                table.add_column("Best Score", style="green")
                table.add_column("Created", style="dim")

                for job in jobs:
                    algorithms_str = ", ".join(job.target_algorithms[:2])
                    if len(job.target_algorithms) > 2:
                        algorithms_str += f" (+{len(job.target_algorithms) - 2})"

                    best_score = f"{job.best_score:.3f}" if job.best_score else "-"
                    trials = f"{job.successful_trials}/{job.total_trials}"

                    table.add_row(
                        job.job_id[:8] + "...",
                        job.name[:20],
                        _format_status(job.status),
                        job.dataset_id[:12] + "...",
                        algorithms_str,
                        trials,
                        best_score,
                        job.created_at.strftime("%m/%d %H:%M"),
                    )
            else:
                # Compact table
                table = Table(title=f"Training Jobs ({len(jobs)} found)")
                table.add_column("Job ID", style="bold blue")
                table.add_column("Name", style="green")
                table.add_column("Status", style="yellow")
                table.add_column("Best Score", style="green")
                table.add_column("Created", style="dim")

                for job in jobs:
                    best_score = f"{job.best_score:.3f}" if job.best_score else "-"

                    table.add_row(
                        job.job_id[:12] + "...",
                        job.name[:30],
                        _format_status(job.status),
                        best_score,
                        job.created_at.strftime("%m/%d %H:%M"),
                    )

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error listing training jobs: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_list_jobs())


@app.command("cancel")
def cancel_training_job(
    job_id: str = typer.Argument(..., help="Training job ID"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Cancel a running training job."""

    async def _cancel_job():
        try:
            if not confirm:
                confirmed = typer.confirm(
                    f"Are you sure you want to cancel training job {job_id}?"
                )
                if not confirmed:
                    rprint("[yellow]Cancelled[/yellow]")
                    return

            service = get_training_service()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Cancelling training job...", total=None)

                await service.cancel_training_job(job_id)

                progress.update(task, description="Training job cancelled successfully")

            rprint(f"[green]Training job {job_id} cancelled successfully[/green]")

        except Exception as e:
            console.print(f"[red]Error cancelling training job: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_cancel_job())


@app.command("quick")
def quick_optimize(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    algorithms: list[str] | None = typer.Option(
        None, "--algorithm", "-a", help="Target algorithms"
    ),
    trials: int = typer.Option(50, "--trials", "-t", help="Maximum trials"),
    timeout: int = typer.Option(30, "--timeout", "-T", help="Timeout in minutes"),
    monitor: bool = typer.Option(False, "--monitor", "-m", help="Monitor progress"),
):
    """Quick optimization with sensible defaults."""

    async def _quick_optimize():
        try:
            service = get_training_service()

            rprint("[blue]Starting quick optimization...[/blue]")

            # Create quick configuration
            config = TrainingConfiguration(
                max_trials=trials,
                timeout_minutes=timeout,
                optimization_strategy=OptimizationStrategy.TPE,
                pruning_strategy=PruningStrategy.MEDIAN,
                experiment_name=f"Quick optimization {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            # Create and start job
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Creating and starting optimization...", total=None
                )

                job = await service.create_training_job(
                    name=f"Quick optimization {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    dataset_id=dataset_id,
                    configuration=config,
                    target_algorithms=algorithms,
                )

                await service.start_training_job(job.job_id)

                progress.update(task, description="Optimization started successfully")

            rprint(f"[green]Quick optimization started: {job.job_id}[/green]")

            if monitor:
                rprint(
                    "[blue]Monitoring optimization progress (Ctrl+C to stop monitoring)...[/blue]"
                )
                await _monitor_job(service, job.job_id)
            else:
                rprint(
                    f"[blue]To monitor progress: mlops training status {job.job_id} --detailed[/blue]"
                )

        except Exception as e:
            console.print(f"[red]Error starting quick optimization: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_quick_optimize())


@app.command("algorithms")
def list_algorithms():
    """List supported algorithms."""

    try:
        trainer = create_model_trainer_adapter()
        algorithms = trainer.get_supported_algorithms()

        table = Table(title="Supported Algorithms")
        table.add_column("Algorithm", style="bold blue")
        table.add_column("Type", style="green")
        table.add_column("Best For", style="yellow")

        for algorithm in algorithms:
            info = trainer.get_algorithm_info(algorithm)
            algorithm_type = info.get("type", "unknown")
            best_for = ", ".join(info.get("best_for", []))

            table.add_row(algorithm, algorithm_type, best_for)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing algorithms: {e}[/red]")
        raise typer.Exit(1)


@app.command("cleanup")
def cleanup_old_jobs(
    days: int = typer.Option(30, "--days", "-d", help="Age threshold in days"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clean up old training jobs and artifacts."""

    async def _cleanup():
        try:
            if not confirm:
                confirmed = typer.confirm(
                    f"Are you sure you want to clean up training jobs older than {days} days?"
                )
                if not confirmed:
                    rprint("[yellow]Cancelled[/yellow]")
                    return

            service = get_training_service()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Cleaning up old jobs...", total=None)

                cleaned_count = await service.cleanup_old_jobs(days)

                progress.update(task, description="Cleanup completed")

            rprint(f"[green]Cleaned up {cleaned_count} old training jobs[/green]")

        except Exception as e:
            console.print(f"[red]Error during cleanup: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_cleanup())


async def _monitor_job(service: TrainingAutomationService, job_id: str):
    """Monitor job progress in real-time."""
    import time

    try:
        while True:
            job = await service.get_job_status(job_id)

            if not job:
                console.print(f"[red]Job {job_id} not found[/red]")
                break

            # Clear screen and show status
            console.clear()

            # Status panel
            status_text = f"Job: {job.name}\nStatus: {_format_status(job.status)}\n"
            status_text += f"Trials: {job.successful_trials}/{job.total_trials}\n"

            if job.execution_time_seconds > 0:
                status_text += f"Runtime: {job.execution_time_seconds:.1f}s\n"

            if job.best_score is not None:
                status_text += f"Best Score: {job.best_score:.4f}\n"

            console.print(Panel(status_text, title="Training Progress"))

            # Check if job is complete
            if job.status in [
                TrainingStatus.COMPLETED,
                TrainingStatus.FAILED,
                TrainingStatus.CANCELLED,
            ]:
                rprint(f"[green]Job {job.status.value}![/green]")
                break

            # Wait before next update
            time.sleep(5)

    except KeyboardInterrupt:
        rprint("\n[yellow]Monitoring stopped[/yellow]")


def _format_status(status: TrainingStatus) -> str:
    """Format status with colors."""
    status_colors = {
        TrainingStatus.QUEUED: "blue",
        TrainingStatus.RUNNING: "yellow",
        TrainingStatus.COMPLETED: "green",
        TrainingStatus.FAILED: "red",
        TrainingStatus.CANCELLED: "dim",
        TrainingStatus.PAUSED: "orange",
    }

    color = status_colors.get(status, "white")
    return f"[{color}]{status.value.title()}[/{color}]"


if __name__ == "__main__":
    app()
