"""
Generic Software CLI Application

Provides basic CLI functionality for software applications.
"""

import click
from typing import Optional

@click.group()
@click.version_option()
def cli():
    """Generic Software CLI Tool"""
    pass

@cli.command()
def version():
    """Show version information"""
    click.echo("Software CLI v0.1.0")

@cli.command()
def config():
    """Manage application configuration"""
    click.echo("Configuration management")

@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
def server(host: str, port: int):
    """Start the application server"""
    click.echo(f"Starting server on {host}:{port}")

@cli.command()
def health():
    """Check application health"""
    click.echo("Application is healthy")

if __name__ == "__main__":
    cli()