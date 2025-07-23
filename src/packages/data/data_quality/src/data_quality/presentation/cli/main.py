
import click


from src.packages.data.data_quality.src.data_quality.presentation.cli.profile import profile
from src.packages.data.data_quality.src.data_quality.presentation.cli.check import check
from src.packages.data.data_quality.src.data_quality.presentation.cli.rule import rule


@click.group()
def cli():
    """A CLI for the data quality package."""
    pass


cli.add_command(profile)
cli.add_command(check)
cli.add_command(rule)


if __name__ == "__main__":
    cli()
