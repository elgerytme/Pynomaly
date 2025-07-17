"""Pipeline Management CLI Commands for data processing workflows and pipeline orchestration."""

from __future__ import annotations

import json
import yaml
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm
from rich.live import Live

console = Console()

# Create the pipeline CLI app
pipeline_app = typer.Typer(
    name="pipeline",
    help="Pipeline management operations for data processing workflows and orchestration",
    rich_markup_mode="rich"
)


@pipeline_app.command("create")
def create_pipeline(
    name: str = typer.Argument(..., help="Pipeline name"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Pipeline configuration file"
    ),
    template: str = typer.Option(
        "basic", "--template", "-t", 
        help="Pipeline template: [basic|etl|ml|quality|streaming|batch]"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive pipeline creation"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for pipeline files"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Create a new data processing pipeline."""
    
    if output_dir is None:
        output_dir = Path.cwd() / "pipelines" / name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import pipeline packages
        from packages.data_pipelines.application.services.pipeline_builder import PipelineBuilder
        from packages.data_pipelines.application.services.pipeline_template_service import PipelineTemplateService
        
        # Initialize services
        builder = PipelineBuilder()
        template_service = PipelineTemplateService()
        
        if interactive:
            console.print("[blue]Interactive Pipeline Creation[/blue]")
            
            # Get pipeline details interactively
            description = Prompt.ask("Pipeline description", default="Data processing pipeline")
            schedule = Prompt.ask("Pipeline schedule (cron format)", default="0 0 * * *")
            
            # Create pipeline interactively
            pipeline_config = builder.create_interactive_pipeline(
                name, description, template, schedule
            )
        else:
            # Load configuration from file or template
            if config_file and config_file.exists():
                pipeline_config = builder.load_pipeline_config(config_file)
                pipeline_config["name"] = name
            else:
                pipeline_config = template_service.generate_pipeline_template(name, template)
        
        # Save pipeline configuration
        config_output = output_dir / "pipeline.yaml"
        with open(config_output, 'w') as f:
            yaml.dump(pipeline_config, f, default_flow_style=False)
        
        # Generate pipeline code
        pipeline_code = builder.generate_pipeline_code(pipeline_config)
        code_output = output_dir / f"{name}_pipeline.py"
        with open(code_output, 'w') as f:
            f.write(pipeline_code)
        
        # Generate documentation
        docs_output = output_dir / "README.md"
        documentation = builder.generate_pipeline_documentation(pipeline_config)
        with open(docs_output, 'w') as f:
            f.write(documentation)
        
        console.print(f"[green]✓ Pipeline '{name}' created successfully[/green]")
        console.print(f"Pipeline directory: {output_dir}")
        console.print(f"Configuration: {config_output}")
        console.print(f"Code: {code_output}")
        console.print(f"Documentation: {docs_output}")
    
    except ImportError as e:
        console.print(f"[red]Error: Required pipeline packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error creating pipeline: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@pipeline_app.command("run")
def run_pipeline(
    pipeline_name: str = typer.Argument(..., help="Pipeline name or path to pipeline file"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Pipeline configuration file"
    ),
    environment: str = typer.Option(
        "development", "--env", "-e", help="Environment: [development|staging|production]"
    ),
    parameters: Optional[str] = typer.Option(
        None, "--params", "-p", help="Pipeline parameters (JSON format)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate pipeline without execution"
    ),
    async_mode: bool = typer.Option(
        False, "--async", help="Run pipeline asynchronously"
    ),
    monitor: bool = typer.Option(
        True, "--monitor/--no-monitor", help="Monitor pipeline execution"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Run a data processing pipeline."""
    
    try:
        # Import pipeline execution packages
        from packages.data_pipelines.application.services.pipeline_executor import PipelineExecutor
        from packages.data_pipelines.application.services.pipeline_monitor import PipelineMonitor
        
        # Initialize services
        executor = PipelineExecutor()
        pipeline_monitor = PipelineMonitor() if monitor else None
        
        # Parse parameters
        pipeline_params = {}
        if parameters:
            try:
                pipeline_params = json.loads(parameters)
            except json.JSONDecodeError:
                console.print("[red]Error: Invalid JSON format for parameters[/red]")
                raise typer.Exit(1)
        
        # Load pipeline configuration
        if config_file:
            pipeline_config = executor.load_pipeline_config(config_file)
        else:
            pipeline_config = executor.discover_pipeline_config(pipeline_name)
        
        if dry_run:
            console.print("[blue]Dry run mode - validating pipeline...[/blue]")
            validation_result = executor.validate_pipeline(pipeline_config, environment)
            
            if validation_result["valid"]:
                console.print("[green]✓ Pipeline validation successful[/green]")
                console.print(f"Pipeline: {pipeline_config['name']}")
                console.print(f"Steps: {len(pipeline_config.get('steps', []))}")
                console.print(f"Estimated duration: {validation_result.get('estimated_duration', 'Unknown')}")
            else:
                console.print("[red]✗ Pipeline validation failed:[/red]")
                for error in validation_result["errors"]:
                    console.print(f"  • {error}")
                raise typer.Exit(1)
            return
        
        # Execute pipeline
        if async_mode:
            console.print("[blue]Starting pipeline in async mode...[/blue]")
            execution_id = executor.run_pipeline_async(
                pipeline_config, environment, pipeline_params
            )
            console.print(f"[green]Pipeline started with execution ID: {execution_id}[/green]")
            console.print(f"Monitor with: software pipeline status {execution_id}")
        else:
            console.print("[blue]Starting pipeline execution...[/blue]")
            
            if monitor:
                # Run with real-time monitoring
                with Live(console=console, refresh_per_second=2) as live:
                    execution_result = executor.run_pipeline_with_monitoring(
                        pipeline_config, environment, pipeline_params, live
                    )
            else:
                # Run without monitoring
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Executing pipeline...", total=100)
                    execution_result = executor.run_pipeline(
                        pipeline_config, environment, pipeline_params, progress, task
                    )
            
            # Display execution results
            if execution_result["success"]:
                console.print("\n[green]✓ Pipeline execution completed successfully[/green]")
                console.print(f"Execution time: {execution_result.get('duration', 'Unknown')}")
                console.print(f"Records processed: {execution_result.get('records_processed', 'Unknown')}")
            else:
                console.print("\n[red]✗ Pipeline execution failed[/red]")
                console.print(f"Error: {execution_result.get('error', 'Unknown error')}")
                raise typer.Exit(1)
    
    except ImportError as e:
        console.print(f"[red]Error: Required pipeline packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error running pipeline: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@pipeline_app.command("list")
def list_pipelines(
    status: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by status: [running|completed|failed|scheduled]"
    ),
    environment: Optional[str] = typer.Option(
        None, "--env", "-e", help="Filter by environment"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: [table|json|yaml]"
    ),
    show_details: bool = typer.Option(
        False, "--details", "-d", help="Show detailed pipeline information"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """List all available pipelines."""
    
    try:
        # Import pipeline registry packages
        from packages.data_pipelines.application.services.pipeline_registry import PipelineRegistry
        
        # Initialize registry
        registry = PipelineRegistry()
        
        # Get all pipelines
        pipelines = registry.get_all_pipelines()
        
        # Apply filters
        if status:
            pipelines = [p for p in pipelines if p.get("status") == status]
        
        if environment:
            pipelines = [p for p in pipelines if p.get("environment") == environment]
        
        if not pipelines:
            console.print("[yellow]No pipelines found[/yellow]")
            return
        
        # Display pipelines
        if format == "table":
            table = Table(title="Data Processing Pipelines")
            table.add_column("Name", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Environment", style="blue")
            table.add_column("Last Run", style="yellow")
            
            if show_details:
                table.add_column("Steps", style="magenta")
                table.add_column("Duration", style="red")
            
            for pipeline in pipelines:
                row = [
                    pipeline.get("name", "Unknown"),
                    pipeline.get("status", "Unknown"),
                    pipeline.get("environment", "Unknown"),
                    pipeline.get("last_run", "Never")
                ]
                
                if show_details:
                    row.extend([
                        str(len(pipeline.get("steps", []))),
                        pipeline.get("last_duration", "Unknown")
                    ])
                
                table.add_row(*row)
            
            console.print(table)
        
        elif format == "json":
            console.print(json.dumps(pipelines, indent=2, default=str))
        
        elif format == "yaml":
            console.print(yaml.dump(pipelines, default_flow_style=False))
    
    except ImportError as e:
        console.print(f"[red]Error: Required registry packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error listing pipelines: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@pipeline_app.command("status")
def pipeline_status(
    execution_id: str = typer.Argument(..., help="Pipeline execution ID"),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch pipeline status in real-time"
    ),
    refresh_interval: int = typer.Option(
        5, "--interval", "-i", help="Refresh interval in seconds (for watch mode)"
    ),
    show_logs: bool = typer.Option(
        False, "--logs", "-l", help="Show execution logs"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Check pipeline execution status."""
    
    try:
        # Import status tracking packages
        from packages.data_pipelines.application.services.pipeline_status_tracker import PipelineStatusTracker
        
        # Initialize status tracker
        status_tracker = PipelineStatusTracker()
        
        if watch:
            console.print(f"[blue]Watching pipeline {execution_id} (refresh every {refresh_interval}s)[/blue]")
            console.print("[yellow]Press Ctrl+C to stop watching[/yellow]")
            
            try:
                import time
                while True:
                    # Get current status
                    status_info = status_tracker.get_execution_status(execution_id)
                    
                    # Clear screen and display status
                    console.clear()
                    _display_pipeline_status(status_info, show_logs)
                    
                    # Check if pipeline finished
                    if status_info.get("status") in ["completed", "failed", "cancelled"]:
                        console.print(f"\n[green]Pipeline execution {status_info.get('status')}[/green]")
                        break
                    
                    time.sleep(refresh_interval)
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped watching pipeline[/yellow]")
        else:
            # Get status once
            status_info = status_tracker.get_execution_status(execution_id)
            _display_pipeline_status(status_info, show_logs)
    
    except ImportError as e:
        console.print(f"[red]Error: Required status packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error checking pipeline status: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@pipeline_app.command("stop")
def stop_pipeline(
    execution_id: str = typer.Argument(..., help="Pipeline execution ID"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force stop pipeline"
    ),
    confirm: bool = typer.Option(
        True, "--confirm/--no-confirm", help="Confirm pipeline stop"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Stop a running pipeline."""
    
    try:
        # Import pipeline control packages
        from packages.data_pipelines.application.services.pipeline_controller import PipelineController
        
        # Initialize controller
        controller = PipelineController()
        
        # Get pipeline status first
        pipeline_info = controller.get_pipeline_info(execution_id)
        
        if pipeline_info.get("status") not in ["running", "scheduled"]:
            console.print(f"[yellow]Pipeline {execution_id} is not running (status: {pipeline_info.get('status')})[/yellow]")
            return
        
        # Confirm stop operation
        if confirm and not Confirm.ask(f"Stop pipeline {execution_id}?"):
            console.print("[yellow]Operation cancelled[/yellow]")
            return
        
        # Stop pipeline
        console.print(f"[blue]Stopping pipeline {execution_id}...[/blue]")
        
        if force:
            result = controller.force_stop_pipeline(execution_id)
        else:
            result = controller.graceful_stop_pipeline(execution_id)
        
        if result["success"]:
            console.print(f"[green]✓ Pipeline {execution_id} stopped successfully[/green]")
        else:
            console.print(f"[red]✗ Failed to stop pipeline: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)
    
    except ImportError as e:
        console.print(f"[red]Error: Required control packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error stopping pipeline: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@pipeline_app.command("schedule")
def schedule_pipeline(
    pipeline_name: str = typer.Argument(..., help="Pipeline name"),
    cron_expression: str = typer.Argument(..., help="Cron schedule expression"),
    environment: str = typer.Option(
        "production", "--env", "-e", help="Environment: [development|staging|production]"
    ),
    timezone: str = typer.Option(
        "UTC", "--timezone", "-tz", help="Timezone for schedule"
    ),
    parameters: Optional[str] = typer.Option(
        None, "--params", "-p", help="Pipeline parameters (JSON format)"
    ),
    active: bool = typer.Option(
        True, "--active/--inactive", help="Schedule is active"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Schedule a pipeline to run automatically."""
    
    try:
        # Import scheduling packages
        from packages.data_pipelines.application.services.pipeline_scheduler import PipelineScheduler
        
        # Initialize scheduler
        scheduler = PipelineScheduler()
        
        # Parse parameters
        pipeline_params = {}
        if parameters:
            try:
                pipeline_params = json.loads(parameters)
            except json.JSONDecodeError:
                console.print("[red]Error: Invalid JSON format for parameters[/red]")
                raise typer.Exit(1)
        
        # Create schedule
        schedule_config = {
            "pipeline_name": pipeline_name,
            "cron_expression": cron_expression,
            "environment": environment,
            "timezone": timezone,
            "parameters": pipeline_params,
            "active": active
        }
        
        # Validate cron expression
        if not scheduler.validate_cron_expression(cron_expression):
            console.print(f"[red]Error: Invalid cron expression '{cron_expression}'[/red]")
            raise typer.Exit(1)
        
        # Schedule pipeline
        schedule_id = scheduler.schedule_pipeline(schedule_config)
        
        console.print(f"[green]✓ Pipeline '{pipeline_name}' scheduled successfully[/green]")
        console.print(f"Schedule ID: {schedule_id}")
        console.print(f"Cron expression: {cron_expression}")
        console.print(f"Environment: {environment}")
        console.print(f"Timezone: {timezone}")
        console.print(f"Active: {active}")
        
        # Show next run times
        next_runs = scheduler.get_next_run_times(schedule_id, count=5)
        if next_runs:
            console.print("\n[blue]Next 5 scheduled runs:[/blue]")
            for i, run_time in enumerate(next_runs, 1):
                console.print(f"  {i}. {run_time}")
    
    except ImportError as e:
        console.print(f"[red]Error: Required scheduling packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error scheduling pipeline: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@pipeline_app.command("validate")
def validate_pipeline(
    pipeline_file: Path = typer.Argument(..., help="Pipeline configuration file"),
    environment: str = typer.Option(
        "development", "--env", "-e", help="Target environment for validation"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Enable strict validation mode"
    ),
    check_dependencies: bool = typer.Option(
        True, "--deps/--no-deps", help="Check pipeline dependencies"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Validate pipeline configuration and dependencies."""
    
    if not pipeline_file.exists():
        console.print(f"[red]Pipeline file {pipeline_file} does not exist[/red]")
        raise typer.Exit(1)
    
    try:
        # Import validation packages
        from packages.data_pipelines.application.services.pipeline_validator import PipelineValidator
        
        # Initialize validator
        validator = PipelineValidator()
        
        # Validate pipeline
        validation_result = validator.validate_pipeline_file(
            pipeline_file,
            environment=environment,
            strict=strict,
            check_dependencies=check_dependencies
        )
        
        if validation_result["valid"]:
            console.print("[green]✓ Pipeline configuration is valid[/green]")
            
            # Display validation summary
            summary = validation_result.get("summary", {})
            
            panel_content = f"""
[bold]Validation Summary[/bold]
• Pipeline Name: {summary.get('name', 'Unknown')}
• Steps: {summary.get('step_count', 0)}
• Dependencies: {summary.get('dependency_count', 0)}
• Estimated Resources: {summary.get('estimated_resources', 'Unknown')}
• Validation Mode: {'Strict' if strict else 'Standard'}
            """
            
            console.print(Panel(panel_content, title="Pipeline Validation", border_style="green"))
        else:
            console.print("[red]✗ Pipeline configuration has validation errors:[/red]")
            
            for error in validation_result["errors"]:
                console.print(f"  • {error}")
            
            if validation_result.get("warnings"):
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in validation_result["warnings"]:
                    console.print(f"  • {warning}")
            
            raise typer.Exit(1)
    
    except ImportError as e:
        console.print(f"[red]Error: Required validation packages not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error validating pipeline: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _display_pipeline_status(status_info: Dict[str, Any], show_logs: bool = False) -> None:
    """Display pipeline status information."""
    
    status = status_info.get("status", "Unknown")
    pipeline_name = status_info.get("pipeline_name", "Unknown")
    execution_id = status_info.get("execution_id", "Unknown")
    
    # Status color mapping
    status_colors = {
        "running": "blue",
        "completed": "green",
        "failed": "red",
        "scheduled": "yellow",
        "cancelled": "orange"
    }
    
    status_color = status_colors.get(status, "white")
    
    panel_content = f"""
[bold]Pipeline Status[/bold]
• Name: {pipeline_name}
• Execution ID: {execution_id}
• Status: [{status_color}]{status.upper()}[/{status_color}]
• Progress: {status_info.get('progress', 0)}%
• Started: {status_info.get('start_time', 'Unknown')}
• Duration: {status_info.get('duration', 'Unknown')}
• Records Processed: {status_info.get('records_processed', 0):,}
    """
    
    console.print(Panel(panel_content, title="Pipeline Execution", border_style=status_color))
    
    # Show current step
    if current_step := status_info.get("current_step"):
        console.print(f"\n[blue]Current Step:[/blue] {current_step.get('name', 'Unknown')}")
        console.print(f"Step Progress: {current_step.get('progress', 0)}%")
    
    # Show logs if requested
    if show_logs and status_info.get("logs"):
        console.print("\n[blue]Recent Logs:[/blue]")
        for log_entry in status_info["logs"][-10:]:  # Show last 10 log entries
            timestamp = log_entry.get("timestamp", "")
            level = log_entry.get("level", "INFO")
            message = log_entry.get("message", "")
            console.print(f"[dim]{timestamp}[/dim] [{level}] {message}")


if __name__ == "__main__":
    pipeline_app()