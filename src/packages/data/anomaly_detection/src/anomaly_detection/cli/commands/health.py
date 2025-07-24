"""Health check commands for CLI."""

import typer
import asyncio
from rich import print
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime
from typing import Optional

from ...domain.services.health_monitoring_service import (
    get_health_monitoring_service,
    HealthStatus
)

console = Console()
app = typer.Typer(help="Health monitoring commands")


@app.command()
def status() -> None:
    """Get current system health status."""
    
    async def _check_health():
        health_service = get_health_monitoring_service()
        return await health_service.get_health_status()
    
    try:
        # Run async health check
        health = asyncio.run(_check_health())
        
        # Overall status
        status_color = {
            HealthStatus.HEALTHY: "green",
            HealthStatus.WARNING: "yellow", 
            HealthStatus.CRITICAL: "red",
            HealthStatus.UNKNOWN: "dim"
        }.get(health.overall_status, "dim")
        
        print(f"[bold {status_color}]System Status: {health.overall_status.value.upper()}[/bold {status_color}]")
        print(f"[dim]Uptime: {health.uptime_seconds:.1f} seconds[/dim]")
        print(f"[dim]Last checked: {health.timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        print()
        
        # Health checks table
        table = Table(title="[bold blue]Health Checks[/bold blue]")
        table.add_column("Check", style="cyan", no_wrap=True)
        table.add_column("Status", style="blue")
        table.add_column("Message", style="dim")
        table.add_column("Response Time", justify="right", style="green")
        
        for check in health.checks:
            status_style = {
                HealthStatus.HEALTHY: "green",
                HealthStatus.WARNING: "yellow",
                HealthStatus.CRITICAL: "red",
                HealthStatus.UNKNOWN: "dim"
            }.get(check.status, "dim")
            
            response_time = f"{check.response_time_ms:.1f}ms" if check.response_time_ms else "N/A"
            
            table.add_row(
                check.name,
                f"[{status_style}]{check.status.value}[/{status_style}]",
                check.message,
                response_time
            )
        
        console.print(table)
        
        # Summary
        healthy_count = len([c for c in health.checks if c.status == HealthStatus.HEALTHY])
        warning_count = len([c for c in health.checks if c.status == HealthStatus.WARNING])
        critical_count = len([c for c in health.checks if c.status == HealthStatus.CRITICAL])
        
        print(f"\n[dim]Summary: {healthy_count} healthy, {warning_count} warnings, {critical_count} critical[/dim]")
        
        # Exit with appropriate code
        if health.overall_status == HealthStatus.CRITICAL:
            raise typer.Exit(2)
        elif health.overall_status == HealthStatus.WARNING:
            raise typer.Exit(1)
        
    except Exception as e:
        print(f"[red]✗[/red] Failed to get health status: {e}")
        raise typer.Exit(1)


@app.command()
def check(
    name: str = typer.Argument(..., help="Name of health check to run"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch check continuously")
) -> None:
    """Run a specific health check."""
    
    async def _run_check():
        health_service = get_health_monitoring_service()
        
        if name not in health_service.health_checks:
            available = list(health_service.health_checks.keys())
            print(f"[red]✗[/red] Health check '{name}' not found.")
            print(f"[dim]Available checks: {', '.join(available)}[/dim]")
            raise typer.Exit(1)
        
        check_func = health_service.health_checks[name]
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            return result
            
        except Exception as e:
            print(f"[red]✗[/red] Health check '{name}' failed: {e}")
            raise typer.Exit(1)
    
    def display_result(result):
        status_style = {
            HealthStatus.HEALTHY: "green",
            HealthStatus.WARNING: "yellow",
            HealthStatus.CRITICAL: "red",
            HealthStatus.UNKNOWN: "dim"
        }.get(result.status, "dim")
        
        print(f"[bold {status_style}]{result.name}: {result.status.value.upper()}[/bold {status_style}]")
        print(f"[dim]{result.message}[/dim]")
        
        if result.response_time_ms:
            print(f"[dim]Response time: {result.response_time_ms:.1f}ms[/dim]")
        
        if result.metadata:
            print(f"[dim]Metadata: {result.metadata}[/dim]")
        
        print(f"[dim]Checked at: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    
    if watch:
        try:
            while True:
                result = asyncio.run(_run_check())
                console.clear()
                print(f"[bold blue]Watching health check: {name}[/bold blue]")
                print(f"[dim]Press Ctrl+C to stop[/dim]\n")
                display_result(result)
                
                import time
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\n[yellow]✓[/yellow] Stopped watching health check")
            
    else:
        result = asyncio.run(_run_check())
        display_result(result)
        
        # Exit with appropriate code
        if result.status == HealthStatus.CRITICAL:
            raise typer.Exit(2)
        elif result.status == HealthStatus.WARNING:
            raise typer.Exit(1)


@app.command()
def list() -> None:
    """List all available health checks."""
    
    health_service = get_health_monitoring_service()
    checks = list(health_service.health_checks.keys())
    
    if not checks:
        print("[yellow]ℹ[/yellow] No health checks registered")
        return
    
    table = Table(title=f"[bold blue]Available Health Checks ({len(checks)})[/bold blue]")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="dim")
    
    # Add descriptions for known checks
    descriptions = {
        "cpu_usage": "Check CPU utilization",
        "memory_usage": "Check memory utilization", 
        "disk_usage": "Check disk space usage",
        "database_connection": "Check database connectivity",
        "service_dependencies": "Check external service dependencies"
    }
    
    for check_name in sorted(checks):
        description = descriptions.get(check_name, "Custom health check")
        table.add_row(check_name, description)
    
    console.print(table)


@app.command()
def monitor(
    interval: int = typer.Option(10, "--interval", "-i", help="Check interval in seconds"),
    checks: Optional[str] = typer.Option(None, "--checks", "-c", help="Comma-separated list of checks to monitor")
) -> None:
    """Monitor system health continuously."""
    
    async def _monitor():
        health_service = get_health_monitoring_service()
        
        # Parse specific checks if provided
        check_names = None
        if checks:
            check_names = [c.strip() for c in checks.split(",")]
            # Validate check names
            available_checks = set(health_service.health_checks.keys())
            invalid_checks = set(check_names) - available_checks
            if invalid_checks:
                print(f"[red]✗[/red] Invalid health checks: {', '.join(invalid_checks)}")
                print(f"[dim]Available: {', '.join(available_checks)}[/dim]")
                raise typer.Exit(1)
        
        try:
            while True:
                health = await health_service.get_health_status()
                
                # Filter checks if specified
                filtered_checks = health.checks
                if check_names:
                    filtered_checks = [c for c in health.checks if c.name in check_names]
                
                # Clear screen and display
                console.clear()
                print("[bold blue]Health Monitor[/bold blue]")
                print(f"[dim]Monitoring every {interval} seconds - Press Ctrl+C to stop[/dim]")
                print(f"[dim]Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")
                
                # Overall status
                status_color = {
                    HealthStatus.HEALTHY: "green",
                    HealthStatus.WARNING: "yellow",
                    HealthStatus.CRITICAL: "red",
                    HealthStatus.UNKNOWN: "dim"
                }.get(health.overall_status, "dim")
                
                print(f"[bold {status_color}]Overall Status: {health.overall_status.value.upper()}[/bold {status_color}]")
                print(f"[dim]Uptime: {health.uptime_seconds:.1f}s[/dim]\n")
                
                # Create table
                table = Table()
                table.add_column("Check", style="cyan")
                table.add_column("Status", style="blue")
                table.add_column("Message", style="dim")
                table.add_column("Time", justify="right", style="green")
                
                for check in filtered_checks:
                    status_style = {
                        HealthStatus.HEALTHY: "green",
                        HealthStatus.WARNING: "yellow", 
                        HealthStatus.CRITICAL: "red",
                        HealthStatus.UNKNOWN: "dim"
                    }.get(check.status, "dim")
                    
                    response_time = f"{check.response_time_ms:.1f}ms" if check.response_time_ms else "N/A"
                    
                    table.add_row(
                        check.name,
                        f"[{status_style}]{check.status.value}[/{status_style}]",
                        check.message[:50] + "..." if len(check.message) > 50 else check.message,
                        response_time
                    )
                
                console.print(table)
                
                import time
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n[yellow]✓[/yellow] Stopped monitoring")
    
    asyncio.run(_monitor())