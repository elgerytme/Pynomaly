"""Database migration CLI commands."""

from __future__ import annotations

from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from monorepo.infrastructure.persistence.migration_manager import (
    create_migration_manager,
    init_and_migrate,
    quick_migrate,
)

app = typer.Typer(
    name="migrate", help="Database migration management", rich_markup_mode="rich"
)
console = Console()


@app.command("status")
def migration_status(
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
):
    """Show current migration status."""
    console.print("[bold blue]Database Migration Status[/bold blue]")

    try:
        manager = create_migration_manager(database_url)
        status = manager.get_migration_status()

        if "error" in status:
            console.print(f"[red]Error:[/red] {status['error']}")
            return

        # Display status information
        info_content = f"[bold]Database URL:[/bold] {status['database_url']}\n"
        info_content += (
            f"[bold]Current Revision:[/bold] {status['current_revision'] or 'None'}\n"
        )
        info_content += (
            f"[bold]Head Revision:[/bold] {status['head_revision'] or 'None'}\n"
        )
        info_content += f"[bold]Migration Needed:[/bold] {'Yes' if status['migration_needed'] else 'No'}\n"

        console.print(
            Panel(info_content, title="Migration Status", border_style="blue")
        )

        # Display revisions table
        if status["revisions"]:
            table = Table(title="Migration Revisions")
            table.add_column("Revision", style="cyan")
            table.add_column("Message", style="green")
            table.add_column("Created", style="yellow")
            table.add_column("Status", style="magenta")

            for revision in status["revisions"]:
                status_str = ""
                if revision["is_current"]:
                    status_str = "✓ Current"
                elif revision["is_head"]:
                    status_str = "→ Head"
                else:
                    status_str = "  -"

                table.add_row(
                    revision["revision"][:8],
                    revision["doc"] or "No message",
                    revision["create_date"] or "Unknown",
                    status_str,
                )

            console.print(table)
        else:
            console.print("[yellow]No migrations found[/yellow]")

    except Exception as e:
        console.print(f"[red]Error checking migration status:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("create")
def create_migration(
    message: str = typer.Argument(..., help="Migration message"),
    autogenerate: bool = typer.Option(
        True, "--autogenerate/--no-autogenerate", help="Auto-generate migration"
    ),
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
):
    """Create a new migration."""
    console.print(f"[bold blue]Creating Migration:[/bold blue] {message}")

    try:
        manager = create_migration_manager(database_url)
        revision = manager.create_migration(message, autogenerate)

        console.print(f"[green]✓[/green] Created migration: {revision}")
        console.print(f"  Message: {message}")
        console.print(f"  Autogenerate: {autogenerate}")

    except Exception as e:
        console.print(f"[red]Error creating migration:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("run")
def run_migrations(
    target: str = typer.Option("head", "--target", "-t", help="Target revision"),
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup before migration"
    ),
):
    """Run database migrations."""
    console.print(f"[bold blue]Running Migrations to:[/bold blue] {target}")

    try:
        manager = create_migration_manager(database_url)

        # Check if migration is needed
        if not manager.check_migration_needed():
            console.print("[green]Database is already up to date[/green]")
            return

        # Create backup if requested
        if backup:
            backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            console.print(f"[yellow]Creating backup:[/yellow] {backup_path}")
            if not manager.backup_database(backup_path):
                console.print(
                    "[yellow]Warning: Backup failed, continuing anyway[/yellow]"
                )

        # Run migrations
        if manager.run_migrations(target):
            console.print("[green]✓[/green] Migrations completed successfully")
        else:
            console.print("[red]✗[/red] Migration failed")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error running migrations:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("rollback")
def rollback_migration(
    target: str = typer.Option(
        "-1", "--target", "-t", help="Target revision to rollback to"
    ),
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Rollback database migration."""
    console.print(f"[bold yellow]Rolling Back to:[/bold yellow] {target}")

    if not force:
        if not Confirm.ask(
            f"Are you sure you want to rollback to revision {target}? This may result in data loss."
        ):
            console.print("[yellow]Rollback cancelled[/yellow]")
            return

    try:
        manager = create_migration_manager(database_url)

        if manager.rollback_migration(target):
            console.print("[green]✓[/green] Rollback completed successfully")
        else:
            console.print("[red]✗[/red] Rollback failed")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error rolling back migration:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("init")
def initialize_database(
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
):
    """Initialize database with migrations."""
    console.print("[bold blue]Initializing Database[/bold blue]")

    try:
        if init_and_migrate(database_url):
            console.print("[green]✓[/green] Database initialized successfully")
        else:
            console.print("[red]✗[/red] Database initialization failed")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error initializing database:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("reset")
def reset_database(
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Reset database to initial state."""
    console.print("[bold red]Resetting Database[/bold red]")

    if not force:
        if not Confirm.ask(
            "Are you sure you want to reset the database? This will delete all data."
        ):
            console.print("[yellow]Reset cancelled[/yellow]")
            return

    try:
        manager = create_migration_manager(database_url)

        if manager.reset_database():
            console.print("[green]✓[/green] Database reset successfully")
        else:
            console.print("[red]✗[/red] Database reset failed")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error resetting database:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("history")
def migration_history(
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
):
    """Show migration history."""
    console.print("[bold blue]Migration History[/bold blue]")

    try:
        manager = create_migration_manager(database_url)
        history = manager.get_migration_history()

        if not history:
            console.print("[yellow]No migration history found[/yellow]")
            return

        table = Table(title="Migration History")
        table.add_column("Revision", style="cyan")
        table.add_column("Message", style="green")
        table.add_column("Created", style="yellow")
        table.add_column("Applied", style="magenta")

        for entry in history:
            table.add_row(
                entry["revision"][:8],
                entry["message"] or "No message",
                entry["create_date"] or "Unknown",
                "✓ Yes" if entry["applied"] else "✗ No",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error getting migration history:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("validate")
def validate_migration(
    revision: str = typer.Argument(..., help="Revision to validate"),
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
):
    """Validate a specific migration."""
    console.print(f"[bold blue]Validating Migration:[/bold blue] {revision}")

    try:
        manager = create_migration_manager(database_url)

        if manager.validate_migration(revision):
            console.print(f"[green]✓[/green] Migration {revision} is valid")
        else:
            console.print(f"[red]✗[/red] Migration {revision} is invalid")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error validating migration:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("quick")
def quick_migration(
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
):
    """Quick migration - initialize and run all migrations."""
    console.print("[bold blue]Quick Migration[/bold blue]")

    try:
        if quick_migrate(database_url):
            console.print("[green]✓[/green] Quick migration completed successfully")
        else:
            console.print("[red]✗[/red] Quick migration failed")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error in quick migration:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("check")
def check_migration_needed(
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
):
    """Check if database migration is needed."""
    console.print("[bold blue]Checking Migration Status[/bold blue]")

    try:
        manager = create_migration_manager(database_url)

        if manager.check_migration_needed():
            console.print("[yellow]Migration is needed[/yellow]")
            raise typer.Exit(1)  # Exit with code 1 to indicate migration needed
        else:
            console.print("[green]Database is up to date[/green]")

    except Exception as e:
        console.print(f"[red]Error checking migration status:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("backup")
def backup_database(
    output_path: str = typer.Argument(..., help="Backup file path"),
    database_url: str = typer.Option(None, "--database-url", "-d", help="Database URL"),
):
    """Create a database backup."""
    console.print(f"[bold blue]Creating Backup:[/bold blue] {output_path}")

    try:
        manager = create_migration_manager(database_url)

        if manager.backup_database(output_path):
            console.print(f"[green]✓[/green] Backup created: {output_path}")
        else:
            console.print("[red]✗[/red] Backup failed")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error creating backup:[/red] {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
