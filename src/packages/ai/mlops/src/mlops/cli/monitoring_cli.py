"""
MLOps Monitoring CLI

Command-line interface for managing the advanced monitoring and observability
platform, providing easy access to monitoring operations, configuration, and insights.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import click
import yaml
import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.tree import Tree

from mlops.infrastructure.monitoring.monitoring_orchestrator import MonitoringOrchestrator
from mlops.infrastructure.monitoring.advanced_observability_platform import AdvancedObservabilityPlatform
from mlops.infrastructure.monitoring.model_drift_detector import ModelDriftDetector
from mlops.infrastructure.monitoring.real_time_analytics import RealTimeAnalyticsService


console = Console()
logger = structlog.get_logger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def monitoring_cli(ctx, config, verbose):
    """MLOps Advanced Monitoring and Observability Platform CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        structlog.configure(
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


@monitoring_cli.group()
@click.pass_context
def stack(ctx):
    """Manage monitoring stack lifecycle"""
    pass


@stack.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Stack configuration file')
@click.option('--environment', '-e', default='development', help='Deployment environment')
@click.pass_context
def start(ctx, config, environment):
    """Start the monitoring stack"""
    async def _start_stack():
        try:
            console.print("[bold green]Starting MLOps monitoring stack...[/bold green]")
            
            config_file = config or ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            with Progress() as progress:
                task = progress.add_task("[green]Initializing services...", total=100)
                
                progress.update(task, advance=20)
                await orchestrator.start_stack()
                progress.update(task, advance=80)
                
                progress.update(task, completed=100)
            
            # Display stack status
            status = await orchestrator.get_stack_status()
            display_stack_status(status)
            
            console.print("\n[bold green]✓ Monitoring stack started successfully![/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to start monitoring stack: {e}[/bold red]")
            sys.exit(1)
    
    asyncio.run(_start_stack())


@stack.command()
@click.pass_context
def stop(ctx):
    """Stop the monitoring stack"""
    async def _stop_stack():
        try:
            console.print("[bold yellow]Stopping MLOps monitoring stack...[/bold yellow]")
            
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            await orchestrator.stop_stack()
            
            console.print("[bold green]✓ Monitoring stack stopped successfully![/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to stop monitoring stack: {e}[/bold red]")
            sys.exit(1)
    
    asyncio.run(_stop_stack())


@stack.command()
@click.pass_context
def status(ctx):
    """Show monitoring stack status"""
    async def _show_status():
        try:
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            status = await orchestrator.get_stack_status()
            display_stack_status(status)
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to get stack status: {e}[/bold red]")
            sys.exit(1)
    
    asyncio.run(_show_status())


@stack.command()
@click.argument('service_name')
@click.pass_context
def restart(ctx, service_name):
    """Restart a specific service"""
    async def _restart_service():
        try:
            console.print(f"[bold yellow]Restarting service: {service_name}[/bold yellow]")
            
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            success = await orchestrator.restart_service(service_name)
            
            if success:
                console.print(f"[bold green]✓ Service {service_name} restarted successfully![/bold green]")
            else:
                console.print(f"[bold red]✗ Failed to restart service {service_name}[/bold red]")
                sys.exit(1)
                
        except Exception as e:
            console.print(f"[bold red]✗ Error restarting service: {e}[/bold red]")
            sys.exit(1)
    
    asyncio.run(_restart_service())


@monitoring_cli.group()
@click.pass_context
def models(ctx):
    """Manage model monitoring"""
    pass


@models.command()
@click.argument('model_id')
@click.argument('model_name')
@click.option('--baseline-data', type=click.Path(exists=True), help='Baseline data file (CSV)')
@click.option('--environment', default='production', help='Model environment')
@click.option('--drift-threshold', type=float, default=0.05, help='Drift detection threshold')
@click.pass_context
def register(ctx, model_id, model_name, baseline_data, environment, drift_threshold):
    """Register a model for monitoring"""
    async def _register_model():
        try:
            console.print(f"[bold blue]Registering model {model_name} for monitoring...[/bold blue]")
            
            # Load baseline data if provided
            monitoring_config = {}
            if baseline_data:
                import pandas as pd
                df = pd.read_csv(baseline_data)
                monitoring_config['baseline_data'] = df
                monitoring_config['enable_drift_monitoring'] = True
                monitoring_config['drift_threshold'] = drift_threshold
            
            # Get observability platform instance
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            platform = orchestrator.get_service_instance('observability_platform')
            if not platform:
                console.print("[bold red]✗ Observability platform not running[/bold red]")
                sys.exit(1)
            
            await platform.register_model_monitoring(
                model_id=model_id,
                model_name=model_name,
                environment=environment,
                monitoring_config=monitoring_config
            )
            
            console.print(f"[bold green]✓ Model {model_name} registered successfully![/bold green]")
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to register model: {e}[/bold red]")
            sys.exit(1)
    
    asyncio.run(_register_model())


@models.command()
@click.pass_context
def list(ctx):
    """List monitored models"""
    async def _list_models():
        try:
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            drift_detector = orchestrator.get_service_instance('drift_detector')
            if not drift_detector:
                console.print("[bold red]✗ Drift detector not running[/bold red]")
                return
            
            status = await drift_detector.get_monitoring_status()
            
            table = Table(title="Monitored Models")
            table.add_column("Model ID")
            table.add_column("Features")
            table.add_column("Baseline Samples")
            table.add_column("Drift Detections")
            table.add_column("Status")
            
            for model_summary in status.get("model_summaries", []):
                table.add_row(
                    model_summary["model_id"],
                    str(model_summary["feature_count"]),
                    str(model_summary["baseline_samples"]),
                    str(model_summary["drift_detections"]),
                    "Active" if model_summary["is_actively_monitored"] else "Registered"
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to list models: {e}[/bold red]")
    
    asyncio.run(_list_models())


@models.command()
@click.argument('model_id')
@click.option('--days', type=int, default=7, help='Number of days to show')
@click.pass_context
def drift_history(ctx, model_id, days):
    """Show drift detection history for a model"""
    async def _show_drift_history():
        try:
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            drift_detector = orchestrator.get_service_instance('drift_detector')
            if not drift_detector:
                console.print("[bold red]✗ Drift detector not running[/bold red]")
                return
            
            history = await drift_detector.get_drift_history(model_id, days)
            
            if not history:
                console.print(f"[yellow]No drift detections found for model {model_id} in the last {days} days[/yellow]")
                return
            
            table = Table(title=f"Drift History - {model_id} (Last {days} days)")
            table.add_column("Detected At")
            table.add_column("Drift Type")
            table.add_column("Severity")
            table.add_column("Score")
            table.add_column("Features Affected")
            
            for detection in history:
                table.add_row(
                    detection["detected_at"][:19],  # Truncate timestamp
                    detection["drift_type"],
                    detection["severity"],
                    f"{detection['drift_score']:.4f}",
                    str(len(detection["features_affected"]))
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to get drift history: {e}[/bold red]")
    
    asyncio.run(_show_drift_history())


@monitoring_cli.group()
@click.pass_context
def insights(ctx):
    """Manage AI-generated insights"""
    pass


@insights.command()
@click.option('--types', multiple=True, help='Filter by insight types')
@click.option('--limit', type=int, default=10, help='Maximum number of insights to show')
@click.pass_context
def list(ctx, types, limit):
    """List AI-generated monitoring insights"""
    async def _list_insights():
        try:
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            platform = orchestrator.get_service_instance('observability_platform')
            if not platform:
                console.print("[bold red]✗ Observability platform not running[/bold red]")
                return
            
            insights = await platform.get_monitoring_insights(
                insight_types=list(types) if types else None,
                limit=limit
            )
            
            if not insights:
                console.print("[yellow]No insights available[/yellow]")
                return
            
            for insight in insights:
                panel_title = f"{insight['title']} ({insight['type']})"
                panel_content = f"""
[bold]Impact Level:[/bold] {insight['impact_level']}
[bold]Confidence:[/bold] {insight['confidence_score']:.2f}
[bold]Description:[/bold] {insight['description']}

[bold]Affected Components:[/bold]
{', '.join(insight['affected_components'])}

[bold]Recommendations:[/bold]
{chr(10).join(f"• {rec}" for rec in insight['recommendations'])}

[dim]Generated at: {insight['created_at']}[/dim]
                """
                
                console.print(Panel(panel_content.strip(), title=panel_title))
                console.print()
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to get insights: {e}[/bold red]")
    
    asyncio.run(_list_insights())


@monitoring_cli.group()
@click.pass_context
def alerts(ctx):
    """Manage monitoring alerts"""
    pass


@alerts.command()
@click.option('--pipeline-id', help='Filter by pipeline ID')
@click.pass_context
def list(ctx, pipeline_id):
    """List active alerts"""
    async def _list_alerts():
        try:
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            pipeline_monitor = orchestrator.get_service_instance('pipeline_monitor')
            if not pipeline_monitor:
                console.print("[bold red]✗ Pipeline monitor not running[/bold red]")
                return
            
            alerts = await pipeline_monitor.get_active_alerts(pipeline_id)
            
            if not alerts:
                console.print("[green]No active alerts[/green]")
                return
            
            table = Table(title="Active Alerts")
            table.add_column("Alert ID")
            table.add_column("Pipeline")
            table.add_column("Type")
            table.add_column("Severity")
            table.add_column("Title")
            table.add_column("Triggered At")
            
            for alert in alerts:
                table.add_row(
                    alert["id"][:8] + "...",  # Truncate ID
                    alert["pipeline_id"],
                    alert["alert_type"],
                    alert["severity"],
                    alert["title"],
                    alert["triggered_at"][:19]  # Truncate timestamp
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to get alerts: {e}[/bold red]")
    
    asyncio.run(_list_alerts())


@alerts.command()
@click.argument('alert_id')
@click.pass_context
def resolve(ctx, alert_id):
    """Resolve an alert"""
    async def _resolve_alert():
        try:
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            pipeline_monitor = orchestrator.get_service_instance('pipeline_monitor')
            if not pipeline_monitor:
                console.print("[bold red]✗ Pipeline monitor not running[/bold red]")
                return
            
            success = await pipeline_monitor.resolve_alert(alert_id)
            
            if success:
                console.print(f"[bold green]✓ Alert {alert_id} resolved successfully![/bold green]")
            else:
                console.print(f"[bold red]✗ Failed to resolve alert {alert_id}[/bold red]")
            
        except Exception as e:
            console.print(f"[bold red]✗ Error resolving alert: {e}[/bold red]")
    
    asyncio.run(_resolve_alert())


@monitoring_cli.group()
@click.pass_context
def metrics(ctx):
    """Access monitoring metrics"""
    pass


@metrics.command()
@click.pass_context
def export(ctx):
    """Export metrics in Prometheus format"""
    async def _export_metrics():
        try:
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            platform = orchestrator.get_service_instance('observability_platform')
            if not platform:
                console.print("[bold red]✗ Observability platform not running[/bold red]")
                return
            
            metrics = await platform.get_metrics_export()
            console.print(metrics)
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to export metrics: {e}[/bold red]")
    
    asyncio.run(_export_metrics())


@monitoring_cli.group()
@click.pass_context
def health(ctx):
    """Check platform health"""
    pass


@health.command()
@click.pass_context
def check(ctx):
    """Perform comprehensive health check"""
    async def _health_check():
        try:
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            platform = orchestrator.get_service_instance('observability_platform')
            if not platform:
                console.print("[bold red]✗ Observability platform not running[/bold red]")
                return
            
            health_status = await platform.get_platform_health_status()
            display_health_status(health_status)
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to perform health check: {e}[/bold red]")
    
    asyncio.run(_health_check())


@monitoring_cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.pass_context
def export_config(ctx, output):
    """Export current monitoring stack configuration"""
    async def _export_config():
        try:
            config_file = ctx.obj.get('config_file')
            orchestrator = MonitoringOrchestrator(config_file) if config_file else MonitoringOrchestrator()
            
            config = await orchestrator.export_stack_configuration()
            
            if output:
                with open(output, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                console.print(f"[bold green]✓ Configuration exported to {output}[/bold green]")
            else:
                console.print(yaml.dump(config, default_flow_style=False))
            
        except Exception as e:
            console.print(f"[bold red]✗ Failed to export configuration: {e}[/bold red]")
    
    asyncio.run(_export_config())


def display_stack_status(status: Dict[str, Any]) -> None:
    """Display monitoring stack status in a formatted way"""
    
    # Overall status panel
    overall_status = "🟢 Healthy" if status["health_ratio"] > 0.8 else "🟡 Degraded" if status["health_ratio"] > 0.5 else "🔴 Unhealthy"
    uptime_hours = status["uptime_seconds"] / 3600
    
    overview_content = f"""
[bold]Stack:[/bold] {status['stack_name']}
[bold]Environment:[/bold] {status['environment']}
[bold]Status:[/bold] {overall_status}
[bold]Uptime:[/bold] {uptime_hours:.1f} hours
[bold]Health Ratio:[/bold] {status['health_ratio']:.2%} ({status['healthy_services']}/{status['total_enabled_services']})
    """
    
    console.print(Panel(overview_content.strip(), title="Monitoring Stack Overview"))
    
    # Services status table
    table = Table(title="Services Status")
    table.add_column("Service")
    table.add_column("Enabled")
    table.add_column("Running")
    table.add_column("Health")
    table.add_column("Restarts")
    table.add_column("Errors")
    table.add_column("Uptime")
    
    for service_name, service_status in status["services"].items():
        enabled_icon = "✅" if service_status["enabled"] else "❌"
        running_icon = "🟢" if service_status["running"] else "🔴"
        health_status = service_status.get("health_status", "unknown")
        restart_count = service_status.get("restart_count", 0)
        error_count = service_status.get("error_count", 0)
        uptime_seconds = service_status.get("uptime_seconds", 0)
        uptime_str = f"{uptime_seconds/3600:.1f}h" if uptime_seconds > 0 else "N/A"
        
        table.add_row(
            service_name,
            enabled_icon,
            running_icon,
            health_status,
            str(restart_count),
            str(error_count),
            uptime_str
        )
    
    console.print(table)


def display_health_status(health_status: Dict[str, Any]) -> None:
    """Display platform health status in a formatted way"""
    
    overall_score = health_status["overall_health_score"]
    status_text = health_status["health_status"]
    
    if overall_score > 0.8:
        status_color = "green"
        status_icon = "🟢"
    elif overall_score > 0.6:
        status_color = "yellow"
        status_icon = "🟡"
    else:
        status_color = "red"
        status_icon = "🔴"
    
    health_content = f"""
[bold]Overall Health Score:[/bold] [{status_color}]{overall_score:.2%}[/{status_color}] {status_icon}
[bold]Status:[/bold] [{status_color}]{status_text.title()}[/{status_color}]
[bold]Active Alerts:[/bold] {health_status['active_alerts']}
[bold]Monitoring Insights:[/bold] {health_status['monitoring_insights']}
[bold]Monitored Models:[/bold] {health_status['monitored_models']}
    """
    
    console.print(Panel(health_content.strip(), title="Platform Health Status"))
    
    # Component health breakdown
    table = Table(title="Component Health Breakdown")
    table.add_column("Component")
    table.add_column("Health Score")
    table.add_column("Status")
    
    for component, score in health_status["component_health"].items():
        score_str = f"{score:.1%}"
        if score > 0.8:
            status_text = "[green]Healthy[/green]"
        elif score > 0.6:
            status_text = "[yellow]Degraded[/yellow]"
        else:
            status_text = "[red]Unhealthy[/red]"
        
        table.add_row(component.title(), score_str, status_text)
    
    console.print(table)


if __name__ == '__main__':
    monitoring_cli()