"""Health monitoring commands for Typer CLI."""

import typer
import asyncio
import json
from typing import Optional
from pathlib import Path
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from datetime import datetime

console = Console()

app = typer.Typer(help="System health monitoring commands")


@app.command()
def status(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed health information"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """Check overall system health status."""
    
    try:
        from ...domain.services.health_monitoring_service import HealthMonitoringService
        
        health_service = HealthMonitoringService()
        
        async def check_health():
            report = await health_service.get_health_report()
            
            if json_output:
                print(json.dumps(report.to_dict(), indent=2, default=str))
                return
            
            # Overall status
            status_color = {
                'healthy': 'green',
                'warning': 'yellow',
                'critical': 'red',
                'unknown': 'blue'
            }.get(report.overall_status.value, 'blue')
            
            print(f"[bold {status_color}]System Health: {report.overall_status.value.upper()}[/bold {status_color}]")
            print(f"[blue]Health Score: {report.overall_score:.1f}/100[/blue]")
            print(f"[dim]Uptime: {report.uptime_seconds/3600:.1f} hours[/dim]")
            print(f"[dim]Last Check: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/dim]\\n")
            
            if detailed:
                # Metrics table
                metrics_table = Table(title="[bold blue]Health Metrics[/bold blue]")
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="white")
                metrics_table.add_column("Status", justify="center")
                metrics_table.add_column("Warning", style="yellow")
                metrics_table.add_column("Critical", style="red")
                
                for metric in report.metrics:
                    status_color = {
                        'healthy': '[green]✓ Healthy[/green]',
                        'warning': '[yellow]⚠ Warning[/yellow]',
                        'critical': '[red]✗ Critical[/red]',
                        'unknown': '[blue]? Unknown[/blue]'
                    }.get(metric.status.value, '[blue]? Unknown[/blue]')
                    
                    metrics_table.add_row(
                        metric.name.replace('_', ' ').title(),
                        f"{metric.value:.1f}{metric.unit}",
                        status_color,
                        f"{metric.threshold_warning}{metric.unit}",
                        f"{metric.threshold_critical}{metric.unit}"
                    )
                
                console.print(metrics_table)
                print()
                
                # Active alerts
                if report.active_alerts:
                    alerts_table = Table(title="[bold red]Active Alerts[/bold red]")
                    alerts_table.add_column("Severity", style="red")
                    alerts_table.add_column("Title", style="white")
                    alerts_table.add_column("Message", style="dim")
                    alerts_table.add_column("Time", style="cyan")
                    
                    for alert in report.active_alerts:
                        severity_color = {
                            'info': '[blue]INFO[/blue]',
                            'warning': '[yellow]WARNING[/yellow]',
                            'error': '[red]ERROR[/red]',
                            'critical': '[bold red]CRITICAL[/bold red]'
                        }.get(alert.severity.value, '[blue]INFO[/blue]')
                        
                        alerts_table.add_row(
                            severity_color,
                            alert.title,
                            alert.message[:50] + "..." if len(alert.message) > 50 else alert.message,
                            alert.timestamp.strftime('%H:%M:%S')
                        )
                    
                    console.print(alerts_table)
                    print()
                
                # Performance summary
                if report.performance_summary:
                    perf_panel = Panel(
                        f"[bold]Performance Summary[/bold]\\n{json.dumps(report.performance_summary, indent=2)}",
                        title="Performance Metrics",
                        border_style="blue"
                    )
                    console.print(perf_panel)
                    print()
            
            # Recommendations
            if report.recommendations:
                print("[bold blue]Recommendations:[/bold blue]")
                for i, rec in enumerate(report.recommendations, 1):
                    print(f"  {i}. {rec}")
        
        asyncio.run(check_health())
        
    except ImportError:
        print("[red]✗[/red] Health monitoring components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Health check failed: {e}")
        raise typer.Exit(1)


@app.command()
def monitor(
    interval: int = typer.Option(10, "--interval", "-i", help="Update interval in seconds"),
    duration: int = typer.Option(0, "--duration", "-d", help="Monitoring duration in seconds (0 = infinite)"),
) -> None:
    """Start real-time health monitoring dashboard."""
    
    try:
        from ...domain.services.health_monitoring_service import HealthMonitoringService
        
        health_service = HealthMonitoringService()
        
        async def monitor_health():
            start_time = datetime.utcnow()
            
            def create_dashboard():
                try:
                    # This would normally be async, but for demo we'll create a simple layout
                    return Panel(
                        f"[bold green]Health Monitoring Dashboard[/bold green]\\n"
                        f"Started: {start_time.strftime('%H:%M:%S')}\\n"
                        f"Interval: {interval}s\\n"
                        f"Status: [green]Monitoring...[/green]\\n\\n"
                        f"[dim]Press Ctrl+C to stop[/dim]",
                        title="System Health Monitor",
                        border_style="green"
                    )
                except Exception:
                    return Panel("Monitoring...", title="Health Monitor")
            
            try:
                with Live(create_dashboard(), refresh_per_second=1) as live:
                    elapsed = 0
                    while duration == 0 or elapsed < duration:
                        try:
                            report = await health_service.get_health_report()
                            
                            # Create updated dashboard
                            status_color = {
                                'healthy': 'green',
                                'warning': 'yellow', 
                                'critical': 'red',
                                'unknown': 'blue'
                            }.get(report.overall_status.value, 'blue')
                            
                            current_time = datetime.utcnow()
                            
                            dashboard_content = (
                                f"[bold {status_color}]System Health: {report.overall_status.value.upper()}[/bold {status_color}]\\n"
                                f"[blue]Health Score: {report.overall_score:.1f}/100[/blue]\\n"
                                f"[cyan]Uptime: {report.uptime_seconds/3600:.1f} hours[/cyan]\\n"
                                f"[dim]Current Time: {current_time.strftime('%H:%M:%S')}[/dim]\\n"
                                f"[dim]Monitoring: {elapsed}s[/dim]\\n\\n"
                            )
                            
                            # Add key metrics
                            for metric in report.metrics[:5]:  # Show top 5 metrics
                                status_indicator = {
                                    'healthy': '🟢',
                                    'warning': '🟡',
                                    'critical': '🔴',
                                    'unknown': '⚪'
                                }.get(metric.status.value, '⚪')
                                
                                dashboard_content += f"{status_indicator} {metric.name.replace('_', ' ').title()}: {metric.value:.1f}{metric.unit}\\n"
                            
                            if report.active_alerts:
                                dashboard_content += f"\\n[bold red]Active Alerts: {len(report.active_alerts)}[/bold red]\\n"
                                for alert in report.active_alerts[:3]:  # Show top 3 alerts
                                    dashboard_content += f"  • [red]{alert.title}[/red]\\n"
                            
                            dashboard_content += "\\n[dim]Press Ctrl+C to stop[/dim]"
                            
                            updated_panel = Panel(
                                dashboard_content,
                                title=f"System Health Monitor - {report.overall_status.value.title()}",
                                border_style=status_color
                            )
                            
                            live.update(updated_panel)
                            
                        except Exception as e:
                            error_panel = Panel(
                                f"[red]Error updating dashboard: {str(e)}[/red]\\n\\n[dim]Press Ctrl+C to stop[/dim]",
                                title="Health Monitor - Error",
                                border_style="red"
                            )
                            live.update(error_panel)
                        
                        await asyncio.sleep(interval)
                        elapsed += interval
                        
            except KeyboardInterrupt:
                print("\\n[yellow]Health monitoring stopped by user[/yellow]")
        
        asyncio.run(monitor_health())
        
    except KeyboardInterrupt:
        print("\\n[green]✅ Health monitoring stopped[/green]")
    except ImportError:
        print("[red]✗[/red] Health monitoring components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Health monitoring failed: {e}")
        raise typer.Exit(1)


@app.command()
def alerts(
    severity: Optional[str] = typer.Option(None, "--severity", "-s", help="Filter by severity (info, warning, error, critical)"),
    history: bool = typer.Option(False, "--history", "-h", help="Show alert history"),
    hours: int = typer.Option(24, "--hours", help="Hours of history to show"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """View system alerts."""
    
    try:
        from ...domain.services.health_monitoring_service import HealthMonitoringService, AlertSeverity
        
        health_service = HealthMonitoringService()
        
        async def show_alerts():
            if history:
                alerts = await health_service.get_alert_history(hours)
                title = f"Alert History ({hours}h)"
            else:
                severity_filter = None
                if severity:
                    try:
                        severity_filter = AlertSeverity(severity.lower())
                    except ValueError:
                        print(f"[red]✗[/red] Invalid severity: {severity}")
                        print("Valid severities: info, warning, error, critical")
                        raise typer.Exit(1)
                
                alerts = await health_service.alert_manager.get_active_alerts(severity_filter)
                title = "Active Alerts"
            
            if json_output:
                alert_data = [alert.to_dict() for alert in alerts]
                print(json.dumps(alert_data, indent=2, default=str))
                return
            
            if not alerts:
                print(f"[green]✅ No {title.lower()} found[/green]")
                return
            
            # Create alerts table
            alerts_table = Table(title=f"[bold blue]{title}[/bold blue]")
            alerts_table.add_column("Time", style="cyan")
            alerts_table.add_column("Severity", justify="center")
            alerts_table.add_column("Title", style="white")
            alerts_table.add_column("Metric", style="yellow")
            alerts_table.add_column("Value", justify="right")
            alerts_table.add_column("Threshold", justify="right")
            if history:
                alerts_table.add_column("Status", justify="center")
            
            for alert in alerts:
                severity_color = {
                    'info': '[blue]INFO[/blue]',
                    'warning': '[yellow]WARNING[/yellow]',
                    'error': '[red]ERROR[/red]',
                    'critical': '[bold red]CRITICAL[/bold red]'
                }.get(alert.severity.value, '[blue]INFO[/blue]')
                
                row_data = [
                    alert.timestamp.strftime('%H:%M:%S'),
                    severity_color,
                    alert.title,
                    alert.metric_name.replace('_', ' ').title(),
                    f"{alert.current_value:.1f}",
                    f"{alert.threshold_value:.1f}"
                ]
                
                if history:
                    status = "[green]Resolved[/green]" if alert.resolved else "[red]Active[/red]"
                    row_data.append(status)
                
                alerts_table.add_row(*row_data)
            
            console.print(alerts_table)
            
            if not history:
                print(f"\\n[dim]Total active alerts: {len(alerts)}[/dim]")
                print(f"[dim]Use --history to see resolved alerts[/dim]")
        
        asyncio.run(show_alerts())
        
    except ImportError:
        print("[red]✗[/red] Health monitoring components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Failed to get alerts: {e}")
        raise typer.Exit(1)


@app.command()
def performance(
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """Show performance metrics summary."""
    
    try:
        from ...domain.services.health_monitoring_service import HealthMonitoringService
        
        health_service = HealthMonitoringService()
        
        async def show_performance():
            summary = await health_service.performance_tracker.get_performance_summary()
            
            if json_output:
                print(json.dumps(summary, indent=2, default=str))
                return
            
            if not summary.get('response_time_stats') and not summary.get('error_stats'):
                print("[yellow]⚠[/yellow] No performance data available")
                print("[dim]Performance metrics are collected as the system processes requests[/dim]")
                return
            
            print("[bold blue]Performance Metrics Summary[/bold blue]\\n")
            
            # Response time metrics
            if summary.get('response_time_stats'):
                rt_stats = summary['response_time_stats']
                rt_table = Table(title="Response Time Statistics")
                rt_table.add_column("Metric", style="cyan")
                rt_table.add_column("Value", style="white", justify="right")
                
                rt_table.add_row("Average", f"{rt_stats['avg_ms']:.1f} ms")
                rt_table.add_row("Median", f"{rt_stats['median_ms']:.1f} ms")
                rt_table.add_row("Min", f"{rt_stats['min_ms']:.1f} ms")
                rt_table.add_row("Max", f"{rt_stats['max_ms']:.1f} ms")
                rt_table.add_row("95th Percentile", f"{rt_stats['p95_ms']:.1f} ms")
                rt_table.add_row("99th Percentile", f"{rt_stats['p99_ms']:.1f} ms")
                
                console.print(rt_table)
                print()
            
            # Error metrics
            if summary.get('error_stats'):
                error_stats = summary['error_stats']
                error_table = Table(title="Error Statistics")
                error_table.add_column("Metric", style="cyan")
                error_table.add_column("Value", style="white", justify="right")
                
                error_table.add_row("Total Errors", str(error_stats['total_errors']))
                error_table.add_row("Error Rate", f"{error_stats['error_rate_percent']:.2f}%")
                error_table.add_row("Recent Errors", str(error_stats['recent_errors']))
                
                console.print(error_table)
                print()
            
            # Throughput metrics
            if summary.get('throughput_stats'):
                tp_stats = summary['throughput_stats']
                tp_table = Table(title="Throughput Statistics")
                tp_table.add_column("Metric", style="cyan")
                tp_table.add_column("Value", style="white", justify="right")
                
                tp_table.add_row("Average RPS", f"{tp_stats['avg_rps']:.1f}")
                tp_table.add_row("Max RPS", f"{tp_stats['max_rps']:.1f}")
                tp_table.add_row("Min RPS", f"{tp_stats['min_rps']:.1f}")
                
                console.print(tp_table)
                print()
            
            print(f"[dim]Data points collected: {summary.get('data_points', 'N/A')}[/dim]")
        
        asyncio.run(show_performance())
        
    except ImportError:
        print("[red]✗[/red] Health monitoring components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Failed to get performance metrics: {e}")
        raise typer.Exit(1)


@app.command()
def start_monitoring(
    interval: int = typer.Option(30, "--interval", "-i", help="Check interval in seconds"),
) -> None:
    """Start background health monitoring service."""
    
    try:
        from ...domain.services.health_monitoring_service import HealthMonitoringService
        
        health_service = HealthMonitoringService(check_interval=interval)
        
        async def start_service():
            print(f"[blue]🏥 Starting Health Monitoring Service[/blue]")
            print(f"   Check interval: [cyan]{interval}s[/cyan]")
            print("   Press Ctrl+C to stop\\n")
            
            await health_service.start_monitoring()
            
            try:
                # Keep running until interrupted
                while health_service.monitoring_enabled:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\\n[yellow]🛑 Stopping health monitoring...[/yellow]")
                await health_service.stop_monitoring()
                print("[green]✅ Health monitoring stopped[/green]")
        
        asyncio.run(start_service())
        
    except KeyboardInterrupt:
        print("\\n[green]✅ Health monitoring stopped[/green]")
    except ImportError:
        print("[red]✗[/red] Health monitoring components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Failed to start health monitoring: {e}")
        raise typer.Exit(1)


@app.command()
def thresholds(
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Show thresholds for specific metric"),
    set_warning: Optional[float] = typer.Option(None, "--set-warning", help="Set warning threshold"),
    set_critical: Optional[float] = typer.Option(None, "--set-critical", help="Set critical threshold"),
) -> None:
    """View or update health monitoring thresholds."""
    
    try:
        from ...domain.services.health_monitoring_service import HealthMonitoringService
        
        health_service = HealthMonitoringService()
        
        # Update thresholds if requested
        if metric and set_warning is not None and set_critical is not None:
            if set_warning >= set_critical:
                print("[red]✗[/red] Warning threshold must be less than critical threshold")
                raise typer.Exit(1)
            
            health_service.set_threshold(metric, set_warning, set_critical)
            print(f"[green]✅ Updated thresholds for {metric}[/green]")
            print(f"   Warning: {set_warning}")
            print(f"   Critical: {set_critical}")
            return
        
        # Show thresholds
        thresholds = health_service.thresholds
        
        if metric:
            if metric not in thresholds:
                print(f"[red]✗[/red] Metric '{metric}' not found")
                print(f"Available metrics: {', '.join(thresholds.keys())}")
                raise typer.Exit(1)
            
            threshold_data = thresholds[metric]
            print(f"[bold blue]Thresholds for {metric}:[/bold blue]")
            print(f"   Warning: [yellow]{threshold_data['warning']}[/yellow]")
            print(f"   Critical: [red]{threshold_data['critical']}[/red]")
        else:
            # Show all thresholds
            thresholds_table = Table(title="[bold blue]Health Monitoring Thresholds[/bold blue]")
            thresholds_table.add_column("Metric", style="cyan")
            thresholds_table.add_column("Warning", style="yellow", justify="right")
            thresholds_table.add_column("Critical", style="red", justify="right")
            
            for metric_name, values in thresholds.items():
                thresholds_table.add_row(
                    metric_name.replace('_', ' ').title(),
                    str(values['warning']),
                    str(values['critical'])
                )
            
            console.print(thresholds_table)
            print(f"\\n[dim]Use --metric to view specific thresholds[/dim]")
            print(f"[dim]Use --metric --set-warning --set-critical to update[/dim]")
        
    except ImportError:
        print("[red]✗[/red] Health monitoring components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Failed to manage thresholds: {e}")
        raise typer.Exit(1)