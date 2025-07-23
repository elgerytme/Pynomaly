"""Main Typer CLI application for Anomaly Detection."""

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from .. import __version__
from .commands import detection, models, data, worker, streaming, explain
from ..infrastructure.config.settings import get_settings
from ..infrastructure.logging import setup_logging

console = Console()
settings = get_settings()

app = typer.Typer(
    name="anomaly-detection",
    help="ML-powered anomaly detection with ensemble methods and model management",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)

# Add command groups
app.add_typer(detection.app, name="detect", help="Anomaly detection commands")
app.add_typer(models.app, name="models", help="Model management commands") 
app.add_typer(data.app, name="data", help="Data generation and management commands")
app.add_typer(worker.app, name="worker", help="Background worker management commands")
app.add_typer(streaming.app, name="streaming", help="Real-time streaming detection commands")
app.add_typer(explain.app, name="explain", help="Model explainability and interpretability commands")


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        print(f"[bold blue]Anomaly Detection CLI[/bold blue] version: [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", 
        help="Enable debug mode",
    ),
) -> None:
    """ML-powered anomaly detection with ensemble methods and model management."""
    setup_logging(verbose=verbose, debug=debug)


@app.command()
def info() -> None:
    """Show application information."""
    table = Table(title="[bold blue]Anomaly Detection Information[/bold blue]")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    table.add_row("Version", __version__)
    table.add_row("Service", "anomaly-detection-api")
    table.add_row("Log Level", settings.logging.level)
    table.add_row("Debug Mode", str(settings.debug))
    table.add_row("API Host", settings.api.host)
    table.add_row("API Port", str(settings.api.port))
    
    console.print(table)


@app.command()
def health() -> None:
    """Check system health status."""
    import httpx
    import asyncio
    
    async def check_health():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{settings.api.host}:{settings.api.port}/health")
                if response.status_code == 200:
                    data = response.json()
                    print(f"[green]✓[/green] Service is [bold green]healthy[/bold green]")
                    print(f"  Service: {data.get('service', 'unknown')}")
                    print(f"  Version: {data.get('version', 'unknown')}")
                    print(f"  Status: {data.get('status', 'unknown')}")
                    if data.get('uptime_seconds'):
                        uptime_hours = data['uptime_seconds'] / 3600
                        print(f"  Uptime: {uptime_hours:.1f} hours")
                else:
                    print(f"[red]✗[/red] Service returned status code: {response.status_code}")
        except Exception as e:
            print(f"[red]✗[/red] Cannot connect to service: {e}")
            print(f"  Make sure the API server is running on {settings.api.host}:{settings.api.port}")
    
    try:
        asyncio.run(check_health())
    except Exception as e:
        print(f"[red]✗[/red] Health check failed: {e}")


@app.command()
def server() -> None:
    """Start the API server."""
    print("[blue]Starting Anomaly Detection API server...[/blue]")
    
    try:
        from ..main import app as fastapi_app
        import uvicorn
        
        uvicorn.run(
            fastapi_app,
            host=settings.api.host,
            port=settings.api.port,
            log_level=settings.logging.level.lower(),
            reload=settings.debug
        )
    except ImportError:
        print("[red]✗[/red] FastAPI server components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Failed to start server: {e}")
        raise typer.Exit(1)


@app.command()
def web() -> None:
    """Start the web dashboard."""
    print("[blue]Starting Anomaly Detection Web Dashboard...[/blue]")
    
    try:
        from ..web.main import web_app
        import uvicorn
        
        uvicorn.run(
            web_app,
            host=settings.api.host,
            port=settings.api.port + 1,  # Use different port
            log_level=settings.logging.level.lower(),
            reload=settings.debug
        )
    except ImportError:
        print("[red]✗[/red] Web dashboard components not available")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Failed to start web dashboard: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()