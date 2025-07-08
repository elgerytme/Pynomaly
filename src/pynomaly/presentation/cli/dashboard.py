"""CLI commands for visualization dashboard functionality."""

import asyncio
import json
from pathlib import Path

import typer
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

# Create Typer app
app = typer.Typer(help="Dashboard related commands")

console = Console()

# Valid dashboard types
VALID_DASHBOARD_TYPES = {
    "executive", "operational", "analytical", "performance", "real_time", "compliance"
}

# Valid export formats
VALID_EXPORT_FORMATS = {"html", "png", "pdf", "svg", "json"}

# Valid themes
VALID_THEMES = {"default", "dark", "light", "corporate"}

def validate_dashboard_type(value: str) -> str:
    """Validate dashboard type."""
    if value not in VALID_DASHBOARD_TYPES:
        raise typer.BadParameter(
            f"Invalid dashboard type '{value}'. Must be one of: {', '.join(VALID_DASHBOARD_TYPES)}"
        )
    return value

def validate_export_format(value: str) -> str:
    """Validate export format."""
    if value not in VALID_EXPORT_FORMATS:
        raise typer.BadParameter(
            f"Invalid export format '{value}'. Must be one of: {', '.join(VALID_EXPORT_FORMATS)}"
        )
    return value

def validate_theme(value: str) -> str:
    """Validate theme."""
    if value not in VALID_THEMES:
        raise typer.BadParameter(
            f"Invalid theme '{value}'. Must be one of: {', '.join(VALID_THEMES)}"
        )
    return value


@app.command()
def generate(
    dashboard_type: str = typer.Option(
        "analytical",
        "--type",
        help="Type of dashboard to generate (executive, operational, analytical, performance, real_time, compliance)",
        callback=validate_dashboard_type,
    ),
    output_path: str | None = typer.Option(
        None, "--output-path", help="Path to save dashboard files"
    ),
    export_format: str = typer.Option(
        "html",
        "--format",
        help="Export format for dashboard (html, png, pdf, svg, json)",
        callback=validate_export_format,
    ),
    theme: str = typer.Option(
        "default",
        "--theme",
        help="Dashboard theme (default, dark, light, corporate)",
        callback=validate_theme,
    ),
    real_time: bool = typer.Option(
        False, "--real-time", help="Enable real-time updates"
    ),
    websocket_endpoint: str | None = typer.Option(
        None, "--websocket-endpoint", help="WebSocket endpoint for real-time data"
    ),
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


@app.command()
def status(
    dashboard_id: str | None = typer.Option(
        None, "--dashboard-id", help="Specific dashboard ID to show status for"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", help="Show detailed status information"
    ),
):
    """Show dashboard service status and active dashboards."""

    async def run_status():
        Container()

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


@app.command()
def monitor(
    interval: int = typer.Option(
        5, "--interval", help="Update interval in seconds"
    ),
    websocket_endpoint: str = typer.Option(
        "ws://localhost:8000/ws",
        "--websocket-endpoint",
        help="WebSocket endpoint",
    ),
    duration: int | None = typer.Option(
        None, "--duration", help="Duration to monitor in seconds"
    ),
):
    """Start real-time dashboard monitoring."""

    async def run_monitoring():
        Container()

        console.print(
            Panel(
                (
                    f"[bold blue]Real-Time Dashboard Monitoring[/bold blue]\n"
                    f"WebSocket: {websocket_endpoint}\n"
                    f"Update interval: {interval}s\n"
                    f"Duration: {duration}s"
                    if duration
                    else "Duration: Unlimited"
                ),
                title="Dashboard Monitor",
            )
        )

        # Initialize dashboard service
        storage_path = Path("./dashboards")
        dashboard_service = VisualizationDashboardService(storage_path)

        # Generate real-time dashboard
        await dashboard_service.generate_real_time_dashboard(websocket_endpoint)

        # Create live layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="metrics", size=10),
            Layout(name="charts", size=15),
            Layout(name="footer", size=3),
        )

        start_time = asyncio.get_event_loop().time()

        with Live(layout, refresh_per_second=1 / interval, console=console):
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


@app.command()
def compare(
    dashboard_type: str = typer.Option(
        "analytical",
        "--dashboard-type",
        help="Dashboard type to compare (executive, operational, analytical, performance)",
    ),
    metrics: list[str] = typer.Option(
        [], "--metrics", help="Specific metrics to compare"
    ),
    time_period: int = typer.Option(
        30, "--time-period", help="Time period in days"
    ),
):
    """Compare dashboard metrics across different time periods."""

    async def run_comparison():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize dashboard service
            task1 = progress.add_task("Generating comparison analysis...", total=None)
            storage_path = Path("./dashboards")
            VisualizationDashboardService(storage_path)

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


@app.command()
def export(
    dashboard_id: str = typer.Option(
        ..., "--dashboard-id", help="Dashboard ID to export"
    ),
    export_format: str = typer.Option(
        "html",
        "--format",
        help="Export format (html, png, pdf, svg, json)",
    ),
    output: str = typer.Option(
        ..., "--output", help="Output file path"
    ),
    config_file: str | None = typer.Option(
        None, "--config-file", help="Export configuration file"
    ),
):
    """Export dashboard to various formats."""

    async def run_export():
        Container()

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


@app.command()
def cleanup(
    clear_cache: bool = typer.Option(
        False, "--clear-cache", help="Clear dashboard cache"
    ),
    reset_metrics: bool = typer.Option(
        False, "--reset-metrics", help="Reset metrics history"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force cleanup without confirmation"
    ),
):
    """Clean up dashboard service resources."""

    async def run_cleanup():
        if not force:
            if clear_cache:
                confirm = typer.confirm("Clear all cached dashboards?")
                if not confirm:
                    console.print("[yellow]Cache cleanup cancelled[/yellow]")
                    return

            if reset_metrics:
                confirm = typer.confirm("Reset all metrics history?")
                if not confirm:
                    console.print("[yellow]Metrics reset cancelled[/yellow]")
                    return

        Container()

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


# Allow stand-alone execution
def main():
    """Main entry point for stand-alone execution."""
    app()

if __name__ == "__main__":
    main()
