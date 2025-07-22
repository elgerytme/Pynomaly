"""CLI entry point for observability package."""

import click


@click.group()
def main():
    """Observability CLI."""
    pass


if __name__ == "__main__":
    main()