"""Typer-compatible wrapper for dashboard CLI commands."""

import typer
from pathlib import Path
from typing import List, Optional

from pynomaly.presentation.cli.dashboard import (
    generate,
    status,
    monitor,
    compare,
    export,
    cleanup,
)

# Create Typer app
app = typer.Typer(
    name="dashboard",
    help="📊 Visualization dashboard management commands",
    add_completion=True,
    rich_markup_mode="rich",
)


@app.command()
def generate_cmd(
    dashboard_type: str = typer.Option(
        "analytical",
        "--type",
        help="Type of dashboard to generate",
    ),
    output_path: Optional[str] = typer.Option(
        None, "--output-path", help="Path to save dashboard files"
    ),
    export_format: str = typer.Option(
        "html", "--format", help="Export format for dashboard"
    ),
    theme: str = typer.Option("default", "--theme", help="Dashboard theme"),
    real_time: bool = typer.Option(False, "--real-time", help="Enable real-time updates"),
    websocket_endpoint: Optional[str] = typer.Option(
        None, "--websocket-endpoint", help="WebSocket endpoint for real-time data"
    ),
):
    """Generate comprehensive visualization dashboard."""
    generate.callback(dashboard_type, output_path, export_format, theme, real_time, websocket_endpoint)


@app.command()
def status_cmd(
    dashboard_id: Optional[str] = typer.Option(
        None, "--dashboard-id", help="Specific dashboard ID to show status for"
    ),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed status information"),
):
    """Show dashboard service status and active dashboards."""
    status.callback(dashboard_id, detailed)


@app.command()
def monitor_cmd(
    interval: int = typer.Option(5, "--interval", help="Update interval in seconds"),
    websocket_endpoint: str = typer.Option(
        "ws://localhost:8000/ws", "--websocket-endpoint", help="WebSocket endpoint"
    ),
    duration: Optional[int] = typer.Option(None, "--duration", help="Duration to monitor in seconds"),
):
    """Start real-time dashboard monitoring."""
    monitor.callback(interval, websocket_endpoint, duration)


@app.command()
def compare_cmd(
    dashboard_type: str = typer.Option(
        "analytical", "--dashboard-type", help="Dashboard type to compare"
    ),
    metrics: List[str] = typer.Option([], "--metrics", help="Specific metrics to compare"),
    time_period: int = typer.Option(30, "--time-period", help="Time period in days"),
):
    """Compare dashboard metrics across different time periods."""
    compare.callback(dashboard_type, metrics, time_period)


@app.command()
def export_cmd(
    dashboard_id: str = typer.Option(..., "--dashboard-id", help="Dashboard ID to export"),
    export_format: str = typer.Option("html", "--format", help="Export format"),
    output: str = typer.Option(..., "--output", help="Output file path"),
    config_file: Optional[str] = typer.Option(None, "--config-file", help="Export configuration file"),
):
    """Export dashboard to various formats."""
    export.callback(dashboard_id, export_format, output, config_file)


@app.command()
def cleanup_cmd(
    clear_cache: bool = typer.Option(False, "--clear-cache", help="Clear dashboard cache"),
    reset_metrics: bool = typer.Option(False, "--reset-metrics", help="Reset metrics history"),
    force: bool = typer.Option(False, "--force", help="Force cleanup without confirmation"),
):
    """Clean up dashboard service resources."""
    cleanup.callback(clear_cache, reset_metrics, force)


if __name__ == "__main__":
    app()