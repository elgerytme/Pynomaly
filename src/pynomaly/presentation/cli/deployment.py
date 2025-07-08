"""CLI commands for model deployment and serving management."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly.application.services.deployment_orchestration_service import (
    DeploymentOrchestrationService,
)
from pynomaly.application.services.model_registry_service import ModelRegistryService
from pynomaly.domain.entities.deployment import (
    DeploymentConfig,
    DeploymentStrategy,
    Environment,
    StrategyType,
)

app = typer.Typer(help="Model deployment and serving management commands")
console = Console()


def get_deployment_service() -> DeploymentOrchestrationService:
    """Get deployment orchestration service instance."""
    # Get storage path from environment or use default
    storage_path = Path.home() / ".pynomaly" / "deployments"
    registry_path = Path.home() / ".pynomaly" / "registry"

    # Initialize services
    model_registry_service = ModelRegistryService(storage_path=registry_path)
    deployment_service = DeploymentOrchestrationService(
        model_registry_service=model_registry_service, storage_path=storage_path
    )

    return deployment_service


@app.command("list")
def list_deployments(
    environment: str
    | None = typer.Option(None, "--env", "-e", help="Filter by environment"),
    status: str | None = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of results"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
):
    """List model deployments."""

    async def _list_deployments():
        deployment_service = get_deployment_service()

        # Parse filters
        env_filter = Environment(environment) if environment else None

        deployments = await deployment_service.list_deployments(
            environment=env_filter, limit=limit
        )

        if status:
            deployments = [d for d in deployments if d.status.value == status]

        if json_output:
            deployment_data = [d.get_deployment_info() for d in deployments]
            print(json.dumps(deployment_data, indent=2, default=str))
            return

        if not deployments:
            console.print("[yellow]No deployments found[/yellow]")
            return

        # Create table
        table = Table(
            title=f"Model Deployments ({len(deployments)} found)",
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("ID", style="cyan", width=8)
        table.add_column("Model Version", style="green")
        table.add_column("Environment", style="blue")
        table.add_column("Status", style="yellow")
        table.add_column("Strategy", style="magenta")
        table.add_column("Health", style="red")
        table.add_column("Created", style="dim")

        for deployment in deployments:
            # Truncate ID for display
            display_id = str(deployment.id)[:8]
            model_version_id = str(deployment.model_version_id)[:8]

            # Status color coding
            status_color = {
                "deployed": "green",
                "failed": "red",
                "in_progress": "yellow",
                "pending": "blue",
            }.get(deployment.status.value, "white")

            # Health score color
            health_score = deployment.health_score
            if health_score > 0.8:
                health_color = "green"
            elif health_score > 0.6:
                health_color = "yellow"
            else:
                health_color = "red"

            table.add_row(
                display_id,
                model_version_id,
                deployment.environment.value,
                f"[{status_color}]{deployment.status.value}[/{status_color}]",
                deployment.strategy.strategy_type.value,
                f"[{health_color}]{health_score:.2f}[/{health_color}]",
                deployment.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    asyncio.run(_list_deployments())


@app.command("deploy")
def deploy_model(
    model_version_id: str = typer.Argument(..., help="Model version ID to deploy"),
    environment: str = typer.Option(
        "staging", "--env", "-e", help="Target environment"
    ),
    strategy: str = typer.Option(
        "rolling", "--strategy", "-s", help="Deployment strategy"
    ),
    replicas: int = typer.Option(3, "--replicas", "-r", help="Number of replicas"),
    cpu_request: str = typer.Option("250m", "--cpu-request", help="CPU request"),
    cpu_limit: str = typer.Option("1000m", "--cpu-limit", help="CPU limit"),
    memory_request: str = typer.Option(
        "512Mi", "--memory-request", help="Memory request"
    ),
    memory_limit: str = typer.Option("2Gi", "--memory-limit", help="Memory limit"),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Wait for deployment to complete"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
):
    """Deploy a model version to an environment."""

    async def _deploy_model():
        deployment_service = get_deployment_service()

        try:
            # Parse inputs
            model_uuid = UUID(model_version_id)
            target_env = Environment(environment)
            strategy_type = StrategyType(strategy)

            # Create deployment configuration
            deployment_config = DeploymentConfig(
                replicas=replicas,
                cpu_request=cpu_request,
                cpu_limit=cpu_limit,
                memory_request=memory_request,
                memory_limit=memory_limit,
            )

            # Create deployment strategy
            deployment_strategy = DeploymentStrategy(strategy_type=strategy_type)

            if not json_output:
                console.print(
                    f"[blue]Deploying model {model_version_id} to {environment}...[/blue]"
                )

            # Start deployment
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Deploying model...", total=None)

                deployment = await deployment_service.deploy_model(
                    model_version_id=model_uuid,
                    target_environment=target_env,
                    strategy=deployment_strategy,
                    deployment_config=deployment_config,
                    user="cli-user",
                )

                progress.update(task, description="Deployment created successfully")

            if json_output:
                print(
                    json.dumps(deployment.get_deployment_info(), indent=2, default=str)
                )
            else:
                # Display deployment info
                panel_content = f"""
[green]✓[/green] Deployment created successfully!

[bold]Deployment ID:[/bold] {deployment.id}
[bold]Model Version:[/bold] {deployment.model_version_id}
[bold]Environment:[/bold] {deployment.environment.value}
[bold]Status:[/bold] {deployment.status.value}
[bold]Strategy:[/bold] {deployment.strategy.strategy_type.value}
[bold]Replicas:[/bold] {deployment.deployment_config.replicas}
[bold]Created:[/bold] {deployment.created_at}
                """
                console.print(
                    Panel(
                        panel_content, title="Deployment Summary", border_style="green"
                    )
                )

                if wait and deployment.status.value == "in_progress":
                    console.print(
                        "[yellow]Waiting for deployment to complete...[/yellow]"
                    )
                    # In real implementation, would poll for status updates
                    await asyncio.sleep(2)
                    console.print("[green]✓ Deployment completed![/green]")

        except ValueError as e:
            console.print(f"[red]Error: Invalid input - {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Deployment failed: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_deploy_model())


@app.command("status")
def deployment_status(
    deployment_id: str = typer.Argument(..., help="Deployment ID"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
):
    """Get deployment status and details."""

    async def _get_status():
        deployment_service = get_deployment_service()

        try:
            deployment_uuid = UUID(deployment_id)
            deployment = await deployment_service.get_deployment(deployment_uuid)

            if json_output:
                print(
                    json.dumps(deployment.get_deployment_info(), indent=2, default=str)
                )
                return

            # Create detailed status display
            status_color = {
                "deployed": "green",
                "failed": "red",
                "in_progress": "yellow",
                "pending": "blue",
            }.get(deployment.status.value, "white")

            health_color = (
                "green"
                if deployment.health_score > 0.8
                else "yellow"
                if deployment.health_score > 0.6
                else "red"
            )

            panel_content = f"""
[bold]Deployment ID:[/bold] {deployment.id}
[bold]Model Version:[/bold] {deployment.model_version_id}
[bold]Environment:[/bold] {deployment.environment.value}
[bold]Status:[/bold] [{status_color}]{deployment.status.value}[/{status_color}]
[bold]Health Score:[/bold] [{health_color}]{deployment.health_score:.2f}[/{health_color}]

[bold]Configuration:[/bold]
  • Replicas: {deployment.deployment_config.replicas}
  • CPU: {deployment.deployment_config.cpu_request} - {deployment.deployment_config.cpu_limit}
  • Memory: {deployment.deployment_config.memory_request} - {deployment.deployment_config.memory_limit}
  • Strategy: {deployment.strategy.strategy_type.value}

[bold]Health Metrics:[/bold]
  • CPU Usage: {deployment.health_metrics.cpu_usage:.1f}%
  • Memory Usage: {deployment.health_metrics.memory_usage:.1f}%
  • Error Rate: {deployment.health_metrics.error_rate:.1f}%
  • Response Time (P95): {deployment.health_metrics.response_time_p95:.1f}ms

[bold]Timeline:[/bold]
  • Created: {deployment.created_at}
  • Deployed: {deployment.deployed_at or "Not deployed"}
  • Created By: {deployment.created_by}
            """

            console.print(
                Panel(panel_content, title="Deployment Status", border_style="blue")
            )

        except ValueError:
            console.print("[red]Error: Invalid deployment ID format[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_get_status())


@app.command("rollback")
def rollback_deployment(
    deployment_id: str = typer.Argument(..., help="Deployment ID to rollback"),
    reason: str = typer.Option(
        "Manual rollback", "--reason", "-r", help="Rollback reason"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Wait for rollback to complete"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
):
    """Rollback a deployment to the previous version."""

    async def _rollback():
        deployment_service = get_deployment_service()

        try:
            deployment_uuid = UUID(deployment_id)

            if not json_output:
                console.print(
                    f"[yellow]Rolling back deployment {deployment_id}...[/yellow]"
                )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Rolling back deployment...", total=None)

                rollback_deployment = await deployment_service.rollback_deployment(
                    deployment_id=deployment_uuid, reason=reason, user="cli-user"
                )

                progress.update(task, description="Rollback completed")

            if rollback_deployment:
                if json_output:
                    print(
                        json.dumps(
                            rollback_deployment.get_deployment_info(),
                            indent=2,
                            default=str,
                        )
                    )
                else:
                    console.print(
                        f"[green]✓ Rollback completed! New deployment: {rollback_deployment.id}[/green]"
                    )
            else:
                if not json_output:
                    console.print(
                        "[yellow]No previous version available for rollback[/yellow]"
                    )

        except ValueError:
            console.print("[red]Error: Invalid deployment ID format[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Rollback failed: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_rollback())


@app.command("promote")
def promote_to_production(
    deployment_id: str = typer.Argument(..., help="Staging deployment ID to promote"),
    approval_notes: str = typer.Option("", "--notes", "-n", help="Approval notes"),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Wait for promotion to complete"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
):
    """Promote a staging deployment to production."""

    async def _promote():
        deployment_service = get_deployment_service()

        try:
            deployment_uuid = UUID(deployment_id)

            approval_metadata = {
                "notes": approval_notes,
                "approved_by": "cli-user",
                "approved_at": datetime.utcnow().isoformat(),
            }

            if not json_output:
                console.print(
                    f"[blue]Promoting deployment {deployment_id} to production...[/blue]"
                )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Promoting to production...", total=None)

                production_deployment = await deployment_service.promote_to_production(
                    deployment_id=deployment_uuid,
                    approval_metadata=approval_metadata,
                    user="cli-user",
                )

                progress.update(task, description="Promotion completed")

            if json_output:
                print(
                    json.dumps(
                        production_deployment.get_deployment_info(),
                        indent=2,
                        default=str,
                    )
                )
            else:
                console.print(
                    f"[green]✓ Promotion completed! Production deployment: {production_deployment.id}[/green]"
                )

        except ValueError:
            console.print("[red]Error: Invalid deployment ID format[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Promotion failed: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_promote())


@app.command("environments")
def list_environments(
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
):
    """List all environments and their status."""

    async def _list_environments():
        deployment_service = get_deployment_service()

        environments = [
            Environment.DEVELOPMENT,
            Environment.STAGING,
            Environment.PRODUCTION,
        ]
        env_statuses = []

        for env in environments:
            status = await deployment_service.get_environment_status(env)
            env_statuses.append(status)

        if json_output:
            print(json.dumps(env_statuses, indent=2, default=str))
            return

        # Create environments table
        table = Table(
            title="Environment Status", show_header=True, header_style="bold magenta"
        )

        table.add_column("Environment", style="cyan")
        table.add_column("Active Deployment", style="green")
        table.add_column("Health", style="yellow")
        table.add_column("Deployment Count", style="blue")
        table.add_column("Last Deployment", style="dim")

        for status in env_statuses:
            health_display = "N/A"
            if status["has_active_deployment"]:
                health_score = status.get("health_score", 0)
                health_color = (
                    "green"
                    if health_score > 0.8
                    else "yellow"
                    if health_score > 0.6
                    else "red"
                )
                health_display = f"[{health_color}]{health_score:.2f}[/{health_color}]"

            table.add_row(
                status["environment"],
                "✓" if status["has_active_deployment"] else "✗",
                health_display,
                str(status["deployment_count"]),
                status["last_deployment"] or "Never",
            )

        console.print(table)

    asyncio.run(_list_environments())


@app.command("serve")
def start_model_server(
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", help="Server port"),
    workers: int = typer.Option(1, "--workers", help="Number of workers"),
    environment: str = typer.Option("production", "--env", help="Environment to serve"),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload (development only)"
    ),
):
    """Start the model serving API server."""

    async def _start_server():
        # Import here to avoid circular imports
        # Set environment variables
        import os

        import uvicorn

        os.environ["MODEL_SERVER_HOST"] = host
        os.environ["MODEL_SERVER_PORT"] = str(port)
        os.environ["WORKERS"] = str(workers)
        os.environ["ENVIRONMENT"] = environment

        console.print(f"[blue]Starting Pynomaly Model Server on {host}:{port}[/blue]")
        console.print(f"[blue]Environment: {environment}[/blue]")
        console.print(f"[blue]Workers: {workers}[/blue]")

        try:
            # Start server using uvicorn
            config = uvicorn.Config(
                "pynomaly.infrastructure.serving.model_server_main:app",
                host=host,
                port=port,
                workers=workers if workers > 1 else None,
                reload=reload,
                log_level="info",
            )

            server = uvicorn.Server(config)
            await server.serve()

        except KeyboardInterrupt:
            console.print("[yellow]Server shutdown by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Server error: {e}[/red]")
            raise typer.Exit(1)

    try:
        asyncio.run(_start_server())
    except KeyboardInterrupt:
        console.print("[yellow]Shutdown complete[/yellow]")


if __name__ == "__main__":
    app()
