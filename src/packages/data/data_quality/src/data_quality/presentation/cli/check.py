
import click
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.application.use_cases.run_data_quality_check import RunDataQualityCheckUseCase
from src.packages.data.data_quality.src.data_quality.di import data_quality_check_service


@click.command()
@click.argument("check_id")
def check(check_id: str):
    """Run a data quality check."""
    use_case = RunDataQualityCheckUseCase(data_quality_check_service)
    check = use_case.execute(UUID(check_id))
    click.echo(f"Successfully ran data quality check {check.name}")
