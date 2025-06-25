"""CLI commands for visualization dashboard functionality."""

import asyncio
import json
from pathlib import Path

import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly.application.services.visualization_dashboard_service import (
    DashboardType,
    RealTimeMetrics,
    VisualizationDashboardService,
)
from pynomaly.infrastructure.config.container import Container

console = Console()


@click.group(name="dashboard")
def dashboard_commands():
    """Visualization dashboard management commands."""
    pass


@dashboard_commands.command()
@click.option(
    "--type",
    "dashboard_type",
    type=click.Choice(
        [
            "executive",
            "operational",
            "analytical",
            "performance",
            "real_time",
            "compliance",
        ]
    ),
    default="analytical",
    help="Type of dashboard to generate",
)
@click.option("--output-path", help="Path to save dashboard files")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["html", "png", "pdf", "svg", "json"]),
    default="html",
    help="Export format for dashboard",
)
@click.option(
    "--theme",
    type=click.Choice(["default", "dark", "light", "corporate"]),
    default="default",
    help="Dashboard theme",
)
@click.option("--real-time", is_flag=True, help="Enable real-time updates")
@click.option("--websocket-endpoint", help="WebSocket endpoint for real-time data")
def generate(
    dashboard_type: str,
    output_path: str | None,
    export_format: str,
    theme: str,
    real_time: bool,
    websocket_endpoint: str | None,
):
    """Generate comprehensive visualization dashboard."""

    async def run_generation():
        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize dashboard service
            task1 = progress.add_task("Initializing dashboard service...", total=None)
            storage_path = Path("./dashboards")

            # Get repositories from container
            detector_repo = container.detector_repository()
            result_repo = container.result_repository()
            dataset_repo = container.dataset_repository()

            dashboard_service = VisualizationDashboardService(
                storage_path=storage_path,
                detector_repository=detector_repo,
                result_repository=result_repo,
                dataset_repository=dataset_repo,
            )
            progress.update(task1, completed=True)

            # Generate dashboard
            task2 = progress.add_task(
                f"Generating {dashboard_type} dashboard...", total=None
            )

            try:
                dashboard_type_enum = DashboardType(dashboard_type)

                if dashboard_type_enum == DashboardType.EXECUTIVE:
                    dashboard_data = (
                        await dashboard_service.generate_executive_dashboard()
                    )
                elif dashboard_type_enum == DashboardType.OPERATIONAL:
                    dashboard_data = (
                        await dashboard_service.generate_operational_dashboard(
                            real_time
                        )
                    )
                elif dashboard_type_enum == DashboardType.ANALYTICAL:
                    dashboard_data = (
                        await dashboard_service.generate_analytical_dashboard()
                    )
                elif dashboard_type_enum == DashboardType.PERFORMANCE:
                    dashboard_data = (
                        await dashboard_service.generate_performance_dashboard()
                    )
                elif dashboard_type_enum == DashboardType.REAL_TIME:
                    if not websocket_endpoint:
                        websocket_endpoint = "ws://localhost:8000/ws"
                    dashboard_data = (
                        await dashboard_service.generate_real_time_dashboard(
                            websocket_endpoint
                        )
                    )
                else:
                    raise ValueError(f"Unsupported dashboard type: {dashboard_type}")

                progress.update(task2, completed=True)

                # Display dashboard summary
                _display_dashboard_summary(dashboard_data)

                # Export dashboard if requested
                if output_path:
                    task3 = progress.add_task(
                        f"Exporting dashboard to {export_format}...", total=None
                    )

                    exported_data = await dashboard_service.export_dashboard(
                        dashboard_data.dashboard_id, export_format
                    )

                    # Save exported data
                    output_file = Path(output_path)
                    if export_format == "json":
                        with open(output_file, "w") as f:
                            json.dump(dashboard_data.to_dict(), f, indent=2)
                    else:
                        with open(output_file, "wb") as f:
                            f.write(exported_data)

                    progress.update(task3, completed=True)
                    console.print(f"[green]Dashboard exported to {output_path}[/green]")

            except Exception as e:
                console.print(f"[red]Error generating dashboard: {e}[/red]")
                return

    asyncio.run(run_generation())


@dashboard_commands.command()
@click.option("--dashboard-id", help="Specific dashboard ID to show status for")
@click.option("--detailed", is_flag=True, help="Show detailed status information")
def status(dashboard_id: str | None, detailed: bool):
    """Show dashboard service status and active dashboards."""

    async def run_status():
        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize dashboard service
            task1 = progress.add_task(
                "Checking dashboard service status...", total=None
            )
            storage_path = Path("./dashboards")

            dashboard_service = VisualizationDashboardService(storage_path)
            progress.update(task1, completed=True)

            # Get service status
            status_info = {
                "service_status": "running",
                "storage_path": str(storage_path),
                "cached_dashboards": len(dashboard_service.dashboard_cache),
                "real_time_subscribers": len(dashboard_service.real_time_subscribers),
                "metrics_history_size": len(dashboard_service.metrics_history),
            }

            # Display status
            _display_service_status(status_info, detailed)

            # Show cached dashboards
            if dashboard_service.dashboard_cache:
                _display_cached_dashboards(dashboard_service.dashboard_cache, detailed)

    asyncio.run(run_status())


@dashboard_commands.command()
@click.option("--interval", type=int, default=5, help="Update interval in seconds")
@click.option(
    "--websocket-endpoint", default="ws://localhost:8000/ws", help="WebSocket endpoint"
)
@click.option("--duration", type=int, help="Duration to monitor in seconds")
def monitor(interval: int, websocket_endpoint: str, duration: int | None):
    """Start real-time dashboard monitoring."""

    async def run_monitoring():
        container = Container()

        console.print(
            Panel(
                f"[bold blue]Real-Time Dashboard Monitoring[/bold blue]\n"
                f"WebSocket: {websocket_endpoint}\n"
                f"Update interval: {interval}s\n"
                f"Duration: {duration}s"
                if duration
                else "Duration: Unlimited",
                title="Dashboard Monitor",
            )
        )

        # Initialize dashboard service
        storage_path = Path("./dashboards")
        dashboard_service = VisualizationDashboardService(storage_path)

        # Generate real-time dashboard
        dashboard_data = await dashboard_service.generate_real_time_dashboard(
            websocket_endpoint
        )

        # Create live layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="metrics", size=10),
            Layout(name="charts", size=15),
            Layout(name="footer", size=3),
        )

        start_time = asyncio.get_event_loop().time()

        with Live(layout, refresh_per_second=1 / interval, console=console) as live:
            try:
                iteration = 0
                while True:
                    # Check duration limit
                    if (
                        duration
                        and (asyncio.get_event_loop().time() - start_time) > duration
                    ):
                        break

                    # Generate mock real-time metrics
                    metrics = _generate_mock_real_time_metrics(iteration)

                    # Update dashboard service
                    await dashboard_service.update_real_time_metrics(metrics)

                    # Update live display
                    _update_live_layout(layout, metrics, iteration)

                    # Wait for next update
                    await asyncio.sleep(interval)
                    iteration += 1

            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped by user[/yellow]")

    asyncio.run(run_monitoring())


@dashboard_commands.command()
@click.option(
    "--dashboard-type",
    type=click.Choice(["executive", "operational", "analytical", "performance"]),
    default="analytical",
    help="Dashboard type to compare",
)
@click.option("--metrics", multiple=True, help="Specific metrics to compare")
@click.option("--time-period", type=int, default=30, help="Time period in days")
def compare(dashboard_type: str, metrics: list[str], time_period: int):
    """Compare dashboard metrics across different time periods."""

    async def run_comparison():
        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize dashboard service
            task1 = progress.add_task("Generating comparison analysis...", total=None)
            storage_path = Path("./dashboards")
            dashboard_service = VisualizationDashboardService(storage_path)

            # Generate comparison data (mock for demonstration)
            comparison_data = {
                "current_period": {
                    "anomaly_detection_rate": 97.5,
                    "false_positive_rate": 2.1,
                    "processing_time": 245.7,
                    "accuracy": 94.8,
                },
                "previous_period": {
                    "anomaly_detection_rate": 95.2,
                    "false_positive_rate": 3.1,
                    "processing_time": 267.3,
                    "accuracy": 92.1,
                },
            }

            progress.update(task1, completed=True)

            # Display comparison
            _display_metrics_comparison(
                comparison_data,
                dashboard_type,
                metrics or list(comparison_data["current_period"].keys()),
            )

    asyncio.run(run_comparison())


@dashboard_commands.command()
@click.option("--dashboard-id", required=True, help="Dashboard ID to export")
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["html", "png", "pdf", "svg", "json"]),
    default="html",
    help="Export format",
)
@click.option("--output", required=True, help="Output file path")
@click.option("--config-file", help="Export configuration file")
def export(
    dashboard_id: str, export_format: str, output: str, config_file: str | None
):
    """Export dashboard to various formats."""

    async def run_export():
        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize dashboard service
            task1 = progress.add_task("Loading dashboard for export...", total=None)
            storage_path = Path("./dashboards")
            dashboard_service = VisualizationDashboardService(storage_path)
            progress.update(task1, completed=True)

            # Load export configuration
            export_config = {}
            if config_file:
                with open(config_file) as f:
                    export_config = json.load(f)

            # Export dashboard
            task2 = progress.add_task(
                f"Exporting dashboard to {export_format}...", total=None
            )

            try:
                exported_data = await dashboard_service.export_dashboard(
                    dashboard_id, export_format, export_config
                )

                # Save exported data
                output_path = Path(output)
                if export_format == "json":
                    # Handle JSON export specially
                    with open(output_path, "w") as f:
                        json.dump(
                            {"exported_data": exported_data.decode()}, f, indent=2
                        )
                else:
                    with open(output_path, "wb") as f:
                        f.write(exported_data)

                progress.update(task2, completed=True)

                console.print(
                    Panel(
                        f"[green]Dashboard exported successfully[/green]\n"
                        f"Dashboard ID: {dashboard_id}\n"
                        f"Format: {export_format}\n"
                        f"Output: {output}\n"
                        f"Size: {len(exported_data)} bytes",
                        title="Export Complete",
                    )
                )

            except Exception as e:
                console.print(f"[red]Export failed: {e}[/red]")

    asyncio.run(run_export())


@dashboard_commands.command()
@click.option("--clear-cache", is_flag=True, help="Clear dashboard cache")
@click.option("--reset-metrics", is_flag=True, help="Reset metrics history")
@click.option("--force", is_flag=True, help="Force cleanup without confirmation")
def cleanup(clear_cache: bool, reset_metrics: bool, force: bool):
    """Clean up dashboard service resources."""

    async def run_cleanup():
        if not force:
            if clear_cache:
                confirm = click.confirm("Clear all cached dashboards?")
                if not confirm:
                    console.print("[yellow]Cache cleanup cancelled[/yellow]")
                    return

            if reset_metrics:
                confirm = click.confirm("Reset all metrics history?")
                if not confirm:
                    console.print("[yellow]Metrics reset cancelled[/yellow]")
                    return

        container = Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize dashboard service
            task1 = progress.add_task("Initializing cleanup...", total=None)
            storage_path = Path("./dashboards")
            dashboard_service = VisualizationDashboardService(storage_path)
            progress.update(task1, completed=True)

            cleanup_stats = {"cleared_cache": 0, "reset_metrics": 0}

            if clear_cache:
                task2 = progress.add_task("Clearing dashboard cache...", total=None)
                cleanup_stats["cleared_cache"] = len(dashboard_service.dashboard_cache)
                dashboard_service.dashboard_cache.clear()
                progress.update(task2, completed=True)

            if reset_metrics:
                task3 = progress.add_task("Resetting metrics history...", total=None)
                cleanup_stats["reset_metrics"] = len(dashboard_service.metrics_history)
                dashboard_service.metrics_history.clear()
                progress.update(task3, completed=True)

            console.print(
                Panel(
                    f"[green]Cleanup completed[/green]\n"
                    f"Cached dashboards cleared: {cleanup_stats['cleared_cache']}\n"
                    f"Metrics history entries reset: {cleanup_stats['reset_metrics']}",
                    title="Cleanup Results",
                )
            )

    asyncio.run(run_cleanup())


# Helper functions for display


def _display_dashboard_summary(dashboard_data):
    """Display dashboard generation summary."""
    console.print(
        Panel(
            f"[bold blue]Dashboard Generated Successfully[/bold blue]\n"
            f"ID: {dashboard_data.dashboard_id}\n"
            f"Type: {dashboard_data.dashboard_type.value.title()}\n"
            f"Title: {dashboard_data.title}\n"
            f"Charts: {len(dashboard_data.charts)}\n"
            f"KPIs: {len(dashboard_data.kpis)}\n"
            f"Generated: {dashboard_data.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            title="Dashboard Summary",
        )
    )

    # Display KPIs if available
    if dashboard_data.kpis:
        kpi_table = Table(title="Key Performance Indicators")
        kpi_table.add_column("Metric", style="cyan")
        kpi_table.add_column("Value", style="yellow")
        kpi_table.add_column("Unit", style="green")

        for name, value in dashboard_data.kpis.items():
            unit = "%" if "rate" in name or "percentage" in name else ""
            if "cost" in name or "savings" in name:
                unit = "$"

            kpi_table.add_row(
                name.replace("_", " ").title(),
                f"{value:.1f}" if isinstance(value, float) else str(value),
                unit,
            )

        console.print(kpi_table)

    # Display charts summary
    if dashboard_data.charts:
        chart_table = Table(title="Charts Generated")
        chart_table.add_column("Chart ID", style="cyan")
        chart_table.add_column("Type", style="yellow")
        chart_table.add_column("Engine", style="green")
        chart_table.add_column("Real-time", style="blue")

        for chart in dashboard_data.charts:
            chart_table.add_row(
                chart.get("id", "Unknown"),
                chart.get("type", "Unknown"),
                chart.get("engine", "Unknown"),
                "✓" if chart.get("realtime", False) else "✗",
            )

        console.print(chart_table)


def _display_service_status(status_info: dict, detailed: bool):
    """Display dashboard service status."""
    status_color = "green" if status_info["service_status"] == "running" else "red"

    console.print(
        Panel(
            f"[bold {status_color}]Service Status: {status_info['service_status'].upper()}[/bold {status_color}]\n"
            f"Storage Path: {status_info['storage_path']}\n"
            f"Cached Dashboards: {status_info['cached_dashboards']}\n"
            f"Real-time Subscribers: {status_info['real_time_subscribers']}\n"
            f"Metrics History: {status_info['metrics_history_size']} entries",
            title="Dashboard Service Status",
        )
    )


def _display_cached_dashboards(dashboard_cache: dict, detailed: bool):
    """Display cached dashboards information."""
    if not dashboard_cache:
        console.print("[yellow]No cached dashboards found[/yellow]")
        return

    table = Table(title="Cached Dashboards")
    table.add_column("Dashboard ID", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Title", style="green")
    table.add_column("Generated", style="blue")
    table.add_column("Charts", style="magenta")

    for dashboard_id, dashboard_data in dashboard_cache.items():
        table.add_row(
            dashboard_id,
            dashboard_data.dashboard_type.value,
            dashboard_data.title,
            dashboard_data.generated_at.strftime("%Y-%m-%d %H:%M"),
            str(len(dashboard_data.charts)),
        )

    console.print(table)


def _generate_mock_real_time_metrics(iteration: int) -> RealTimeMetrics:
    """Generate mock real-time metrics for demonstration."""
    import math
    import random

    # Create realistic variations
    base_anomalies = 10 + int(5 * math.sin(iteration * 0.1))
    base_cpu = 45 + 15 * math.sin(iteration * 0.05)
    base_memory = 65 + 10 * math.sin(iteration * 0.03)

    return RealTimeMetrics(
        anomalies_detected=max(0, base_anomalies + random.randint(-3, 3)),
        detection_rate=max(0, min(100, 95 + 5 * math.sin(iteration * 0.1))),
        system_cpu_usage=max(0, min(100, base_cpu + random.uniform(-5, 5))),
        system_memory_usage=max(0, min(100, base_memory + random.uniform(-3, 3))),
        active_detectors=random.randint(8, 15),
        processed_samples=random.randint(1000, 2000),
        processing_latency_ms=max(10, 150 + random.uniform(-30, 30)),
        throughput_per_second=max(100, 800 + random.uniform(-100, 100)),
        error_rate=max(0, min(10, 2 + random.uniform(-1, 1))),
        alert_count=random.randint(0, 5),
        business_kpis={
            "cost_savings": 125000 + random.uniform(-5000, 5000),
            "accuracy": 94 + random.uniform(-2, 2),
        },
    )


def _update_live_layout(layout: Layout, metrics: RealTimeMetrics, iteration: int):
    """Update live layout with real-time metrics."""
    # Header
    layout["header"].update(
        Panel(
            f"[bold blue]Real-Time Dashboard Monitor[/bold blue] - Update #{iteration + 1}",
            style="blue",
        )
    )

    # Metrics
    metrics_table = Table.grid()
    metrics_table.add_column(style="cyan")
    metrics_table.add_column(style="yellow")
    metrics_table.add_column(style="cyan")
    metrics_table.add_column(style="yellow")

    metrics_table.add_row(
        "Anomalies Detected:",
        str(metrics.anomalies_detected),
        "Detection Rate:",
        f"{metrics.detection_rate:.1f}%",
    )
    metrics_table.add_row(
        "CPU Usage:",
        f"{metrics.system_cpu_usage:.1f}%",
        "Memory Usage:",
        f"{metrics.system_memory_usage:.1f}%",
    )
    metrics_table.add_row(
        "Active Detectors:",
        str(metrics.active_detectors),
        "Processed Samples:",
        str(metrics.processed_samples),
    )
    metrics_table.add_row(
        "Latency:",
        f"{metrics.processing_latency_ms:.1f}ms",
        "Throughput:",
        f"{metrics.throughput_per_second:.1f}/s",
    )

    layout["metrics"].update(Panel(metrics_table, title="Live Metrics"))

    # Charts placeholder
    chart_info = """
[cyan]Live Detection Chart[/cyan]: Streaming anomaly detection results
[yellow]System Resources[/yellow]: CPU and memory utilization trends  
[green]Throughput Monitor[/green]: Processing rate and latency
[blue]Alert Stream[/blue]: Real-time alert notifications
    """
    layout["charts"].update(Panel(chart_info.strip(), title="Active Charts"))

    # Footer
    layout["footer"].update(
        Panel(
            f"Last Update: {metrics.timestamp.strftime('%H:%M:%S')} | "
            f"Alerts: {metrics.alert_count} | "
            f"Error Rate: {metrics.error_rate:.1f}%",
            style="dim",
        )
    )


def _display_metrics_comparison(
    comparison_data: dict, dashboard_type: str, metrics: list[str]
):
    """Display metrics comparison between time periods."""
    console.print(
        Panel(
            f"[bold blue]Dashboard Metrics Comparison[/bold blue]\n"
            f"Dashboard Type: {dashboard_type.title()}\n"
            f"Metrics Analyzed: {len(metrics)}",
            title="Comparison Analysis",
        )
    )

    table = Table(title="Performance Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Current Period", style="green")
    table.add_column("Previous Period", style="yellow")
    table.add_column("Change", style="blue")
    table.add_column("Trend", style="magenta")

    for metric in metrics:
        current = comparison_data["current_period"].get(metric, 0)
        previous = comparison_data["previous_period"].get(metric, 0)

        if previous != 0:
            change_pct = ((current - previous) / previous) * 100
            trend = "↗️" if change_pct > 0 else "↘️" if change_pct < 0 else "→"
        else:
            change_pct = 0
            trend = "→"

        table.add_row(
            metric.replace("_", " ").title(),
            f"{current:.1f}",
            f"{previous:.1f}",
            f"{change_pct:+.1f}%",
            trend,
        )

    console.print(table)


if __name__ == "__main__":
    dashboard_commands()
