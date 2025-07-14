"""User management CLI commands."""

from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from packages.core.application.dto.user_dto import CreateUserDTO
from packages.core.application.use_cases.create_user import CreateUserUseCase
from packages.core.domain.entities.user import UserRole
from packages.infrastructure.repositories.in_memory_user_repository import InMemoryUserRepository
from packages.infrastructure.security.password_hasher import BCryptPasswordHasher

# Create console for rich output
console = Console()

# Create user sub-command app
app = typer.Typer()

# Global repository instance (in production, use dependency injection)
user_repository = InMemoryUserRepository()
password_hasher = BCryptPasswordHasher()


@app.command()
def create(
    email: str = typer.Option(..., "--email", "-e", help="User email"),
    username: str = typer.Option(..., "--username", "-u", help="Username"),
    full_name: str = typer.Option(..., "--name", "-n", help="Full name"),
    password: str = typer.Option(..., "--password", "-p", help="Password"),
    admin: bool = typer.Option(False, "--admin", help="Make user admin"),
) -> None:
    """Create a new user."""
    
    # Determine roles
    roles = set()
    if admin:
        roles.add(UserRole.ADMIN)
    else:
        roles.add(UserRole.USER)
    
    # Create user DTO
    user_dto = CreateUserDTO(
        email=email,
        username=username,
        full_name=full_name,
        password=password,
        roles=roles,
    )
    
    # Create user
    use_case = CreateUserUseCase(user_repository, password_hasher)
    
    try:
        import asyncio
        user = asyncio.run(use_case.execute(user_dto))
        console.print(f"[green]✓[/green] User created successfully!")
        console.print(f"  ID: {user.id}")
        console.print(f"  Email: {user.email}")
        console.print(f"  Username: {user.username}")
        console.print(f"  Full Name: {user.full_name}")
        console.print(f"  Roles: {', '.join(user.roles)}")
    except ValueError as e:
        console.print(f"[red]✗[/red] Error creating user: {e}")
        raise typer.Exit(1)


@app.command()
def list() -> None:
    """List all users."""
    import asyncio
    
    users = asyncio.run(user_repository.find_all())
    
    if not users:
        console.print("[yellow]No users found.[/yellow]")
        return
    
    table = Table(title="Users")
    table.add_column("ID", style="cyan")
    table.add_column("Email", style="magenta")
    table.add_column("Username", style="green")
    table.add_column("Full Name", style="blue")
    table.add_column("Roles", style="yellow")
    table.add_column("Status", style="red")
    
    for user in users:
        table.add_row(
            str(user.id)[:8] + "...",
            user.email,
            user.username,
            user.full_name,
            ", ".join(user.roles),
            user.status.value,
        )
    
    console.print(table)


@app.command()
def get(user_id: str = typer.Argument(..., help="User ID")) -> None:
    """Get user by ID."""
    import asyncio
    
    try:
        uuid_id = UUID(user_id)
    except ValueError:
        console.print(f"[red]✗[/red] Invalid user ID format: {user_id}")
        raise typer.Exit(1)
    
    user = asyncio.run(user_repository.find_by_id(uuid_id))
    
    if not user:
        console.print(f"[red]✗[/red] User not found: {user_id}")
        raise typer.Exit(1)
    
    console.print(f"[bold]User Details:[/bold]")
    console.print(f"  ID: {user.id}")
    console.print(f"  Email: {user.email}")
    console.print(f"  Username: {user.username}")
    console.print(f"  Full Name: {user.full_name}")
    console.print(f"  Roles: {', '.join(user.roles)}")
    console.print(f"  Status: {user.status}")
    console.print(f"  Created: {user.created_at}")
    console.print(f"  Updated: {user.updated_at}")
    console.print(f"  Last Login: {user.last_login}")
    console.print(f"  Email Verified: {user.email_verified}")
    console.print(f"  Failed Attempts: {user.failed_login_attempts}")
    console.print(f"  Is Locked: {user.is_locked()}")


@app.command()
def delete(user_id: str = typer.Argument(..., help="User ID")) -> None:
    """Delete user by ID."""
    import asyncio
    
    try:
        uuid_id = UUID(user_id)
    except ValueError:
        console.print(f"[red]✗[/red] Invalid user ID format: {user_id}")
        raise typer.Exit(1)
    
    # Confirm deletion
    if not typer.confirm(f"Are you sure you want to delete user {user_id}?"):
        console.print("[yellow]Operation cancelled.[/yellow]")
        return
    
    success = asyncio.run(user_repository.delete(uuid_id))
    
    if success:
        console.print(f"[green]✓[/green] User deleted successfully: {user_id}")
    else:
        console.print(f"[red]✗[/red] User not found: {user_id}")
        raise typer.Exit(1)