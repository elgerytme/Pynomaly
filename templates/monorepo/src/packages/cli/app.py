"""CLI application using Typer."""

import typer
from rich.console import Console

from packages.infrastructure.config.settings import settings
from .commands import users

# Create console for rich output
console = Console()

# Create main CLI app
app = typer.Typer(
    name=settings.app_name.lower().replace(" ", "-"),
    help=f"{settings.app_name} CLI - Command-line interface for the application",
    add_completion=False,
)

# Add sub-commands
app.add_typer(users.app, name="users", help="User management commands")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold green]{settings.app_name}[/bold green] v{settings.app_version}")
    console.print(f"Environment: [yellow]{settings.environment}[/yellow]")


@app.command()
def config() -> None:
    """Show configuration information."""
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  App Name: {settings.app_name}")
    console.print(f"  Version: {settings.app_version}")
    console.print(f"  Environment: {settings.environment}")
    console.print(f"  Debug: {settings.debug}")
    console.print(f"  Host: {settings.host}")
    console.print(f"  Port: {settings.port}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()