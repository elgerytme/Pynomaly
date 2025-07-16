"""Server management CLI commands."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import typer
from rich.console import Console

from pynomaly_detection.presentation.cli.container import get_cli_container

app = typer.Typer()
console = Console()


@app.command("start")
def start_server(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run as daemon"),
    log_level: str = typer.Option("info", "--log-level", help="Log level"),
):
    """Start the API server."""
    container = get_cli_container()
    settings = container.config()

    # Override settings if provided
    if host != "0.0.0.0":
        settings.api_host = host
    if port != 8000:
        settings.api_port = port

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "pynomaly.presentation.api.app:app",
        f"--host={settings.api_host}",
        f"--port={settings.api_port}",
        f"--log-level={log_level}",
    ]

    if reload:
        cmd.append("--reload")
    else:
        cmd.extend([f"--workers={workers}"])

    # Check if port is already in use
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((settings.api_host, settings.api_port))
    sock.close()

    if result == 0:
        console.print(f"[red]Error:[/red] Port {settings.api_port} is already in use")
        raise typer.Exit(1)

    # Start server
    console.print("[green]Starting Pynomaly API server...[/green]")
    console.print(f"Host: {settings.api_host}")
    console.print(f"Port: {settings.api_port}")
    console.print(f"Workers: {workers}")
    console.print(f"Reload: {reload}")
    console.print(f"\nAPI docs: http://{settings.api_host}:{settings.api_port}/docs")
    console.print(
        f"Health check: http://{settings.api_host}:{settings.api_port}/health"
    )
    console.print("\nPress CTRL+C to stop the server")

    if daemon:
        # Run as daemon (basic implementation)
        pid_file = settings.storage_path / "pynomaly.pid"

        # Fork process
        try:
            pid = os.fork()
            if pid > 0:
                # Parent process
                console.print(
                    f"\n[green]✓[/green] Server started as daemon (PID: {pid})"
                )
                # Save PID
                pid_file.write_text(str(pid))
                return
        except OSError as e:
            console.print(f"[red]Error:[/red] Failed to fork process: {e}")
            raise typer.Exit(1)

        # Child process continues
        os.setsid()

        # Redirect stdout/stderr to log file
        log_file = settings.log_path / "server.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "a") as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
            os.dup2(f.fileno(), sys.stderr.fileno())

    try:
        # Run server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error:[/red] Server failed to start: {e}")
        raise typer.Exit(1)


@app.command("stop")
def stop_server(force: bool = typer.Option(False, "--force", "-f", help="Force stop")):
    """Stop the API server."""
    container = get_cli_container()
    settings = container.config()

    pid_file = settings.storage_path / "pynomaly.pid"

    if not pid_file.exists():
        console.print("[yellow]No server PID file found[/yellow]")
        console.print("The server may not be running as a daemon")
        return

    try:
        pid = int(pid_file.read_text().strip())

        # Check if process exists
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            console.print("[yellow]Server process not found[/yellow]")
            pid_file.unlink()
            return

        # Send signal
        if force:
            os.kill(pid, signal.SIGKILL)
            console.print(f"[red]Forcefully killed server (PID: {pid})[/red]")
        else:
            os.kill(pid, signal.SIGTERM)
            console.print(f"[green]✓[/green] Sent stop signal to server (PID: {pid})")

        # Remove PID file
        pid_file.unlink()

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to stop server: {e}")
        raise typer.Exit(1)


@app.command("status")
def server_status():
    """Check server status."""
    container = get_cli_container()
    settings = container.config()

    # Check PID file
    pid_file = settings.storage_path / "pynomaly.pid"

    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())

            # Check if process exists
            try:
                os.kill(pid, 0)
                console.print(f"[green]✓[/green] Server is running (PID: {pid})")
            except ProcessLookupError:
                console.print(
                    "[yellow]⚠[/yellow] PID file exists but process not found"
                )
                pid_file.unlink()

        except Exception as e:
            console.print(f"[red]Error reading PID file:[/red] {e}")
    else:
        console.print("[yellow]○[/yellow] Server is not running as daemon")

    # Try to connect to API
    import requests

    api_url = f"http://{settings.api_host}:{settings.api_port}/health"

    try:
        response = requests.get(api_url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            console.print(f"\n[green]✓[/green] API is accessible at {api_url}")
            console.print(f"  Status: {data.get('status', 'unknown')}")
            console.print(f"  Version: {data.get('version', 'unknown')}")
            console.print(f"  Uptime: {data.get('uptime_seconds', 0):.0f} seconds")
        else:
            console.print(f"\n[red]✗[/red] API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        console.print(f"\n[yellow]○[/yellow] Cannot connect to API at {api_url}")
    except Exception as e:
        console.print(f"\n[red]Error checking API:[/red] {e}")


@app.command("logs")
def show_logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    error: bool = typer.Option(False, "--error", "-e", help="Show only errors"),
):
    """Show server logs."""
    container = get_cli_container()
    settings = container.config()

    log_file = settings.log_path / "server.log"

    if not log_file.exists():
        console.print("[yellow]No log file found[/yellow]")
        console.print(f"Expected location: {log_file}")
        return

    if follow:
        # Follow log file
        console.print(f"Following {log_file} (press CTRL+C to stop)...")

        try:
            cmd = ["tail", "-f", str(log_file)]
            if lines > 0:
                cmd.extend(["-n", str(lines)])

            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped following logs[/yellow]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error:[/red] Failed to read logs: {e}")
    else:
        # Show last N lines
        try:
            with open(log_file) as f:
                all_lines = f.readlines()

            if error:
                # Filter error lines
                error_lines = [
                    line
                    for line in all_lines
                    if "ERROR" in line or "CRITICAL" in line or "Exception" in line
                ]
                display_lines = error_lines[-lines:] if lines > 0 else error_lines
            else:
                display_lines = all_lines[-lines:] if lines > 0 else all_lines

            if display_lines:
                console.print(
                    f"[dim]Showing last {len(display_lines)} lines from {log_file}:[/dim]\n"
                )
                for line in display_lines:
                    # Color code by level
                    if "ERROR" in line or "CRITICAL" in line:
                        console.print(f"[red]{line.rstrip()}[/red]")
                    elif "WARNING" in line:
                        console.print(f"[yellow]{line.rstrip()}[/yellow]")
                    elif "INFO" in line:
                        console.print(f"[green]{line.rstrip()}[/green]")
                    else:
                        console.print(line.rstrip())
            else:
                console.print("[yellow]No matching log entries found[/yellow]")

        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to read logs: {e}")


@app.command("config")
def show_server_config():
    """Show server configuration."""
    container = get_cli_container()
    settings = container.config()

    console.print("[bold]Server Configuration:[/bold]\n")

    # API settings
    console.print("[cyan]API Settings:[/cyan]")
    console.print(f"  Host: {settings.api_host}")
    console.print(f"  Port: {settings.api_port}")
    console.print(f"  CORS Origins: {settings.cors_origins}")
    console.print(f"  Rate Limit: {settings.rate_limit_requests}/min")

    # Storage settings
    console.print("\n[cyan]Storage Settings:[/cyan]")
    console.print(f"  Storage Path: {settings.storage_path}")
    console.print(f"  Log Path: {settings.log_path}")
    console.print(f"  Temp Path: {settings.temp_path}")
    console.print(f"  Model Path: {settings.model_path}")

    # Performance settings
    console.print("\n[cyan]Performance Settings:[/cyan]")
    console.print(f"  Max Workers: {settings.max_workers}")
    console.print(f"  Batch Size: {settings.batch_size}")
    console.print(f"  Cache TTL: {settings.cache_ttl_seconds}s")
    console.print(f"  GPU Enabled: {settings.gpu_enabled}")

    # Data settings
    console.print("\n[cyan]Data Settings:[/cyan]")
    console.print(f"  Max Dataset Size: {settings.max_dataset_size_mb}MB")
    console.print(f"  Default Contamination: {settings.default_contamination_rate}")

    # Environment
    console.print("\n[cyan]Environment:[/cyan]")
    console.print(f"  Debug Mode: {settings.debug}")
    console.print(f"  Environment: {settings.environment}")


@app.command("health")
def check_health():
    """Check API health."""
    container = get_cli_container()
    settings = container.config()

    import requests

    base_url = f"http://{settings.api_host}:{settings.api_port}"

    # Check endpoints
    endpoints = [
        ("/health", "Health Check"),
        ("/api/v1/detectors", "Detectors API"),
        ("/api/v1/datasets", "Datasets API"),
        ("/docs", "API Documentation"),
        ("/metrics", "Metrics"),
    ]

    console.print("[bold]API Health Check:[/bold]\n")

    all_healthy = True

    for endpoint, name in endpoints:
        url = base_url + endpoint
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                console.print(f"[green]✓[/green] {name}: OK")
            else:
                console.print(
                    f"[yellow]⚠[/yellow] {name}: Status {response.status_code}"
                )
                all_healthy = False
        except requests.exceptions.ConnectionError:
            console.print(f"[red]✗[/red] {name}: Connection failed")
            all_healthy = False
        except Exception as e:
            console.print(f"[red]✗[/red] {name}: {type(e).__name__}")
            all_healthy = False

    if all_healthy:
        console.print("\n[green]All systems operational![/green]")
    else:
        console.print("\n[yellow]Some services are not available[/yellow]")
        raise typer.Exit(1)
