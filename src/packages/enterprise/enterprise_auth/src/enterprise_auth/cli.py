"""CLI entry point for enterprise_auth package."""

import click


@click.group()
def main():
    """Enterprise Auth CLI."""
    pass


if __name__ == "__main__":
    main()