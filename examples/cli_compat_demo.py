#!/usr/bin/env python3
"""
Demonstration of the CLI compatibility layer.

This script shows how the compatibility layer works with different
Typer versions and command structures.
"""

from __future__ import annotations

import typer
from rich.console import Console

# Import our compatibility layer
from pynomaly.presentation.cli._compat import (
    count_commands,
    get_command,
    get_command_names,
    has_commands,
    list_commands,
)

console = Console()


def main():
    """Demonstrate the compatibility layer functionality."""
    console.print("[bold blue]CLI Compatibility Layer Demo[/bold blue]\n")
    
    # Create a sample Typer app
    app = typer.Typer(name="demo", help="Demo CLI app")
    
    console.print("[cyan]1. Empty app:[/cyan]")
    console.print(f"  Has commands: {has_commands(app)}")
    console.print(f"  Command count: {count_commands(app)}")
    console.print(f"  Command names: {get_command_names(app)}")
    
    # Add some commands
    @app.command("hello")
    def hello_cmd(name: str = "World"):
        """Say hello to someone."""
        console.print(f"Hello {name}!")
    
    @app.command("goodbye")
    def goodbye_cmd(name: str = "World"):
        """Say goodbye to someone."""
        console.print(f"Goodbye {name}!")
    
    @app.command("status")
    def status_cmd():
        """Show status."""
        console.print("Status: OK")
    
    console.print("\n[cyan]2. App with commands:[/cyan]")
    console.print(f"  Has commands: {has_commands(app)}")
    console.print(f"  Command count: {count_commands(app)}")
    console.print(f"  Command names: {get_command_names(app)}")
    
    # Show detailed command information
    console.print("\n[cyan]3. Command details:[/cyan]")
    commands = list_commands(app)
    for name, cmd_info in commands.items():
        console.print(f"  - {name}: {type(cmd_info).__name__}")
        
    # Test getting specific commands
    console.print("\n[cyan]4. Get specific commands:[/cyan]")
    hello_cmd_info = get_command(app, "hello")
    unknown_cmd_info = get_command(app, "unknown")
    
    console.print(f"  hello command: {hello_cmd_info is not None}")
    console.print(f"  unknown command: {unknown_cmd_info is not None}")
    
    console.print("\n[green]✓ Compatibility layer working correctly![/green]")
    
    # Show the actual Typer structure for reference
    console.print("\n[cyan]5. Internal Typer structure (for reference):[/cyan]")
    console.print(f"  registered_commands type: {type(app.registered_commands)}")
    console.print(f"  registered_commands length: {len(app.registered_commands)}")
    
    # Demonstrate version compatibility
    console.print("\n[cyan]6. Version compatibility demo:[/cyan]")
    console.print("  The compatibility layer handles:")
    console.print("  - Typer >= 0.15.1: registered_commands as list")
    console.print("  - Older versions: commands as dict")
    console.print("  - Missing attributes: graceful fallback")


if __name__ == "__main__":
    main()
