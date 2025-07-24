
import click
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.application.use_cases.run_data_quality_check import RunDataQualityCheckUseCase
from src.packages.data.data_quality.src.data_quality.di import get_data_quality_check_service


@click.group()
def check():
    """Manage data quality checks."""
    pass


@check.command()
@click.argument("check_id")
@click.option("--file-path", required=True, help="Path to the CSV file to check.")
def run(check_id: str, file_path: str):
    """Run a data quality check."""
    use_case = RunDataQualityCheckUseCase(get_data_quality_check_service())
    check = use_case.execute(UUID(check_id), {"file_path": file_path})
    click.echo(f"Successfully ran data quality check {check.name}")


@check.command()
def list():
    """List all data quality checks."""
    # This will require a new use case to list all checks
    click.echo("Listing all data quality checks (not yet implemented).")
