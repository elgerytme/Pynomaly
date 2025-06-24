"""CLI commands for performance management and optimization."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from dependency_injector.wiring import inject, Provide

from pynomaly.infrastructure.config.container import Container
from pynomaly.infrastructure.performance import (
    ConnectionPoolManager,
    QueryOptimizer,
    PerformanceService
)

app = typer.Typer(name="performance", help="Performance management and optimization commands")
console = Console()


@app.command("pools")
@inject
async def list_pools(
    pool_manager: ConnectionPoolManager = Provide[Container.connection_pool_manager]
):
    """List all connection pools and their statistics."""
    try:
        if pool_manager is None:
            console.print("[red]❌ Connection pool manager not available[/red]")
            raise typer.Exit(1)
        
        pool_names = pool_manager.list_pools()
        
        if not pool_names:
            console.print("[yellow]No connection pools found[/yellow]")
            return
        
        table = Table(title="Connection Pools")
        table.add_column("Pool Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Active", style="green")
        table.add_column("Total Requests", style="blue")
        table.add_column("Success Rate", style="green")
        table.add_column("Avg Response Time", style="yellow")
        table.add_column("Errors", style="red")
        
        for pool_name in pool_names:
            try:
                pool_info = pool_manager.get_pool_info(pool_name)
                stats = pool_info["stats"]
                
                success_rate = (
                    stats.successful_requests / max(1, stats.total_requests) * 100
                )
                
                table.add_row(
                    pool_name,
                    pool_info["type"],
                    str(stats.active_connections),
                    str(stats.total_requests),
                    f"{success_rate:.1f}%",
                    f"{stats.avg_response_time:.3f}s",
                    str(stats.connection_errors)
                )
            except Exception as e:
                console.print(f"[red]Error getting info for pool {pool_name}: {e}[/red]")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Error listing pools: {e}[/red]")
        raise typer.Exit(1)


@app.command("pool")
@inject
async def show_pool(
    pool_name: str = typer.Argument(..., help="Pool name to inspect"),
    pool_manager: ConnectionPoolManager = Provide[Container.connection_pool_manager]
):
    """Show detailed information about a specific connection pool."""
    try:
        if pool_manager is None:
            console.print("[red]❌ Connection pool manager not available[/red]")
            raise typer.Exit(1)
        
        pool_info = pool_manager.get_pool_info(pool_name)
        stats = pool_info["stats"]
        
        # Create detailed panel
        info_text = f"""
[bold cyan]Pool Type:[/bold cyan] {pool_info['type']}
[bold cyan]Active Connections:[/bold cyan] {stats.active_connections}
[bold cyan]Idle Connections:[/bold cyan] {stats.idle_connections}
[bold cyan]Overflow Connections:[/bold cyan] {stats.overflow_connections}

[bold green]Performance Metrics:[/bold green]
Total Requests: {stats.total_requests}
Successful Requests: {stats.successful_requests}
Failed Requests: {stats.failed_requests}
Success Rate: {stats.successful_requests / max(1, stats.total_requests) * 100:.2f}%
Average Response Time: {stats.avg_response_time:.3f}s

[bold red]Error Metrics:[/bold red]
Connection Errors: {stats.connection_errors}
Error Rate: {stats.failed_requests / max(1, stats.total_requests) * 100:.2f}%

[bold blue]Connection Lifecycle:[/bold blue]
Connections Created: {stats.connections_created}
Connections Closed: {stats.connections_closed}
Connections Recycled: {stats.connections_recycled}
"""
        
        console.print(Panel(info_text, title=f"Pool: {pool_name}", border_style="blue"))
        
        # Show pool-specific info if available
        if pool_info.get("pool_info"):
            pool_specific = pool_info["pool_info"]
            console.print(f"\n[bold]Pool-Specific Information:[/bold]")
            for key, value in pool_specific.items():
                console.print(f"  {key}: {value}")
        
    except KeyError:
        console.print(f"[red]❌ Pool '{pool_name}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error showing pool: {e}[/red]")
        raise typer.Exit(1)


@app.command("reset-pools")
@inject
async def reset_pool_stats(
    pool_name: Optional[str] = typer.Option(None, "--pool", "-p", help="Reset specific pool (default: all)"),
    pool_manager: ConnectionPoolManager = Provide[Container.connection_pool_manager]
):
    """Reset connection pool statistics."""
    try:
        if pool_manager is None:
            console.print("[red]❌ Connection pool manager not available[/red]")
            raise typer.Exit(1)
        
        if pool_name:
            pool_manager.reset_stats(pool_name)
            console.print(f"[green]✅ Statistics reset for pool '{pool_name}'[/green]")
        else:
            pool_manager.reset_stats()
            console.print("[green]✅ Statistics reset for all pools[/green]")
        
    except KeyError:
        console.print(f"[red]❌ Pool '{pool_name}' not found[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error resetting pool statistics: {e}[/red]")
        raise typer.Exit(1)


@app.command("queries")
@inject
async def query_performance(
    slow_threshold: float = typer.Option(1.0, "--threshold", "-t", help="Slow query threshold in seconds"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of queries to show"),
    optimizer: QueryOptimizer = Provide[Container.query_optimizer]
):
    """Show query performance statistics."""
    try:
        if optimizer is None:
            console.print("[red]❌ Query optimizer not available[/red]")
            raise typer.Exit(1)
        
        # Get performance summary
        summary = optimizer.performance_tracker.get_performance_summary()
        
        # Display summary
        summary_text = f"""
[bold green]Query Performance Summary:[/bold green]
Total Queries: {summary['total_queries']}
Unique Queries: {summary['unique_queries']}
Average Time: {summary['avg_time']:.3f}s
Total Time: {summary['total_time']:.2f}s
Slow Queries: {summary['slow_queries']} (>{slow_threshold}s)
Slowest Query: {summary['slowest_query']:.3f}s

[bold blue]Query Types:[/bold blue]
"""
        
        for query_type, count in summary['query_types'].items():
            summary_text += f"  {query_type}: {count}\n"
        
        console.print(Panel(summary_text, title="Query Performance", border_style="green"))
        
        # Show slow queries
        slow_queries = optimizer.performance_tracker.get_slow_queries(slow_threshold)
        if slow_queries:
            console.print(f"\n[bold red]Slow Queries (>{slow_threshold}s):[/bold red]")
            
            table = Table()
            table.add_column("Query Hash", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Count", style="blue")
            table.add_column("Avg Time", style="red")
            table.add_column("Max Time", style="red")
            
            for query in slow_queries[:limit]:
                table.add_row(
                    query.query_hash[:12] + "...",
                    query.query_type.value,
                    str(query.execution_count),
                    f"{query.avg_time:.3f}s",
                    f"{query.max_time:.3f}s"
                )
            
            console.print(table)
        else:
            console.print("[green]No slow queries found! 🎉[/green]")
        
        # Show most frequent queries
        frequent_queries = optimizer.performance_tracker.get_most_frequent_queries(limit)
        if frequent_queries:
            console.print(f"\n[bold blue]Most Frequent Queries:[/bold blue]")
            
            table = Table()
            table.add_column("Query Hash", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Count", style="blue")
            table.add_column("Avg Time", style="yellow")
            table.add_column("Total Time", style="yellow")
            
            for query in frequent_queries:
                table.add_row(
                    query.query_hash[:12] + "...",
                    query.query_type.value,
                    str(query.execution_count),
                    f"{query.avg_time:.3f}s",
                    f"{query.total_time:.2f}s"
                )
            
            console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Error showing query performance: {e}[/red]")
        raise typer.Exit(1)


@app.command("cache")
@inject
async def cache_stats(
    optimizer: QueryOptimizer = Provide[Container.query_optimizer]
):
    """Show query cache statistics."""
    try:
        if optimizer is None:
            console.print("[red]❌ Query optimizer not available[/red]")
            raise typer.Exit(1)
        
        stats = optimizer.cache.get_stats()
        
        cache_text = f"""
[bold green]Cache Statistics:[/bold green]
Total Entries: {stats['total_entries']}
Max Size: {stats['max_size']}
Total Hits: {stats['total_hits']}
Hit Rate: {stats['hit_rate']:.2%}
Memory Usage: {stats['memory_usage_mb']:.2f} MB
"""
        
        console.print(Panel(cache_text, title="Query Cache", border_style="blue"))
        
    except Exception as e:
        console.print(f"[red]❌ Error showing cache statistics: {e}[/red]")
        raise typer.Exit(1)


@app.command("clear-cache")
@inject
async def clear_cache(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    optimizer: QueryOptimizer = Provide[Container.query_optimizer]
):
    """Clear query cache."""
    try:
        if optimizer is None:
            console.print("[red]❌ Query optimizer not available[/red]")
            raise typer.Exit(1)
        
        if not confirm:
            confirm = typer.confirm("Are you sure you want to clear the query cache?")
            if not confirm:
                console.print("Cache clear cancelled")
                return
        
        await optimizer.clear_cache()
        console.print("[green]✅ Query cache cleared[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error clearing cache: {e}[/red]")
        raise typer.Exit(1)


@app.command("optimize")
@inject
async def optimize_database(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show recommendations without applying changes"),
    optimizer: QueryOptimizer = Provide[Container.query_optimizer]
):
    """Optimize database performance."""
    try:
        if optimizer is None:
            console.print("[red]❌ Query optimizer not available[/red]")
            raise typer.Exit(1)
        
        console.print("[blue]🔍 Analyzing database performance...[/blue]")
        
        if dry_run:
            # Get recommendations without applying
            report = await optimizer.get_optimization_report()
            
            console.print("[green]✅ Analysis complete[/green]")
            
            # Show recommendations
            recommendations = report.get("index_recommendations", [])
            if recommendations:
                console.print(f"\n[bold blue]Index Recommendations:[/bold blue]")
                
                table = Table()
                table.add_column("Table", style="cyan")
                table.add_column("Columns", style="magenta")
                table.add_column("Reason", style="yellow")
                table.add_column("Benefit", style="green")
                
                for rec in recommendations:
                    table.add_row(
                        rec["table"],
                        ", ".join(rec["columns"]),
                        rec["reason"],
                        f"{rec['estimated_benefit']:.1%}"
                    )
                
                console.print(table)
            else:
                console.print("[green]No index recommendations found - database is already optimized! 🎉[/green]")
        
        else:
            # Perform actual optimization
            with console.status("[bold blue]Optimizing database..."):
                result = await optimizer.optimize_database()
            
            console.print("[green]✅ Database optimization complete[/green]")
            
            # Show results
            results_text = f"""
[bold green]Optimization Results:[/bold green]
Indexes Created: {result['indexes_created']}
Recommendations: {len(result['recommendations'])}
"""
            
            if result.get('error'):
                results_text += f"\n[bold red]Error:[/bold red] {result['error']}"
            
            console.print(Panel(results_text, title="Optimization Results", border_style="green"))
            
            # Show created indexes
            if result['recommendations']:
                console.print(f"\n[bold blue]Applied Recommendations:[/bold blue]")
                
                for rec in result['recommendations']:
                    console.print(f"  • {rec['table']}: {', '.join(rec['columns'])} - {rec['reason']}")
        
    except Exception as e:
        console.print(f"[red]❌ Error optimizing database: {e}[/red]")
        raise typer.Exit(1)


@app.command("monitor")
@inject
async def monitor_performance(
    duration: int = typer.Option(60, "--duration", "-d", help="Monitoring duration in seconds"),
    interval: int = typer.Option(5, "--interval", "-i", help="Update interval in seconds"),
    performance_service: PerformanceService = Provide[Container.performance_service]
):
    """Monitor performance in real-time."""
    try:
        if performance_service is None:
            console.print("[red]❌ Performance service not available[/red]")
            raise typer.Exit(1)
        
        console.print(f"[blue]📊 Monitoring performance for {duration} seconds (updating every {interval}s)...[/blue]")
        
        iterations = duration // interval
        
        for i in track(range(iterations), description="Monitoring..."):
            # Get current performance summary
            summary = performance_service.get_performance_summary()
            
            console.clear()
            console.print(f"[bold]Performance Monitor - Update {i+1}/{iterations}[/bold]")
            
            # Show connection pools
            if summary.get("connection_pools"):
                console.print("\n[bold blue]Connection Pools:[/bold blue]")
                for pool_name, pool_data in summary["connection_pools"].items():
                    console.print(f"  {pool_name}: {pool_data['active_connections']} active, "
                                f"{pool_data['avg_response_time']:.3f}s avg time, "
                                f"{pool_data['success_rate']:.1%} success")
            
            # Show query performance
            if summary.get("query_performance"):
                qp = summary["query_performance"]
                console.print(f"\n[bold green]Query Performance:[/bold green]")
                console.print(f"  Total: {qp.get('total_queries', 0)} queries, "
                            f"{qp.get('avg_time', 0):.3f}s avg time")
                console.print(f"  Slow: {qp.get('slow_queries', 0)} queries")
            
            # Show cache performance
            if summary.get("cache_performance"):
                cp = summary["cache_performance"]
                console.print(f"\n[bold yellow]Cache Performance:[/bold yellow]")
                console.print(f"  Entries: {cp.get('total_entries', 0)}, "
                            f"Hit rate: {cp.get('hit_rate', 0):.1%}")
            
            # Show alerts
            alerts = performance_service.get_alerts(unresolved_only=True, limit=3)
            if alerts:
                console.print(f"\n[bold red]Active Alerts ({len(alerts)}):[/bold red]")
                for alert in alerts:
                    console.print(f"  ⚠️  {alert['message']}")
            
            if i < iterations - 1:  # Don't sleep on last iteration
                await asyncio.sleep(interval)
        
        console.print("\n[green]✅ Monitoring complete[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error monitoring performance: {e}[/red]")
        raise typer.Exit(1)


@app.command("report")
@inject
async def performance_report(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file (default: console)"),
    optimizer: QueryOptimizer = Provide[Container.query_optimizer],
    performance_service: PerformanceService = Provide[Container.performance_service]
):
    """Generate comprehensive performance report."""
    try:
        console.print("[blue]📋 Generating performance report...[/blue]")
        
        report_data = {}
        
        # Get query optimization report
        if optimizer:
            report_data["query_optimization"] = await optimizer.get_optimization_report()
        
        # Get performance service summary
        if performance_service:
            report_data["performance_summary"] = performance_service.get_performance_summary()
            report_data["alerts"] = performance_service.get_alerts(unresolved_only=False, limit=20)
        
        # Format report
        report_text = "# Pynomaly Performance Report\n\n"
        
        if "performance_summary" in report_data:
            ps = report_data["performance_summary"]
            report_text += f"## Summary\n"
            report_text += f"- Monitoring Duration: {ps.get('monitoring_duration', 0):.1f} hours\n"
            report_text += f"- Active Alerts: {ps.get('active_alerts', 0)}\n\n"
            
            # Connection pools
            if ps.get("connection_pools"):
                report_text += "## Connection Pools\n"
                for pool_name, pool_data in ps["connection_pools"].items():
                    report_text += f"### {pool_name}\n"
                    report_text += f"- Active Connections: {pool_data['active_connections']}\n"
                    report_text += f"- Success Rate: {pool_data['success_rate']:.1%}\n"
                    report_text += f"- Average Response Time: {pool_data['avg_response_time']:.3f}s\n\n"
        
        if "query_optimization" in report_data:
            qo = report_data["query_optimization"]
            
            # Query performance
            if qo.get("performance_summary"):
                qps = qo["performance_summary"]
                report_text += "## Query Performance\n"
                report_text += f"- Total Queries: {qps['total_queries']}\n"
                report_text += f"- Average Time: {qps['avg_time']:.3f}s\n"
                report_text += f"- Slow Queries: {qps['slow_queries']}\n\n"
            
            # Index recommendations
            if qo.get("index_recommendations"):
                report_text += "## Index Recommendations\n"
                for rec in qo["index_recommendations"]:
                    report_text += f"- {rec['table']}.{', '.join(rec['columns'])}: {rec['reason']}\n"
                report_text += "\n"
        
        # Alerts
        if "alerts" in report_data and report_data["alerts"]:
            report_text += "## Recent Alerts\n"
            for alert in report_data["alerts"][-10:]:  # Last 10 alerts
                timestamp = alert.get('timestamp', 0)
                status = "✅ Resolved" if alert.get('resolved') else "⚠️  Active"
                report_text += f"- {status}: {alert['message']} (Type: {alert['type']})\n"
            report_text += "\n"
        
        # Output report
        if output:
            with open(output, 'w') as f:
                f.write(report_text)
            console.print(f"[green]✅ Report saved to {output}[/green]")
        else:
            console.print(report_text)
        
    except Exception as e:
        console.print(f"[red]❌ Error generating report: {e}[/red]")
        raise typer.Exit(1)


def run_async_command(func):
    """Wrapper to run async CLI commands."""
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


# Note: Async commands are handled individually in their decorators