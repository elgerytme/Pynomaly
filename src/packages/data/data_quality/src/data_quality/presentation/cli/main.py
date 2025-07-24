
import click


from .profile import profile
from .check import check
from .rule import rule


@click.group()
def cli():
    """A CLI for the data quality package."""
    pass


cli.add_command(profile)
cli.add_command(check)
cli.add_command(rule)


if __name__ == "__main__":
    cli()
