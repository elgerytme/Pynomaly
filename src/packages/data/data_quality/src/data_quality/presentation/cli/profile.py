
import click

from src.packages.data.data_quality.src.data_quality.application.use_cases.create_data_profile import CreateDataProfileUseCase
from src.packages.data.data_quality.src.data_quality.di import data_profiling_service


@click.command()
@click.argument("dataset_name")
def profile(dataset_name: str):
    """Create a data profile for a dataset."""
    use_case = CreateDataProfileUseCase(data_profiling_service)
    profile = use_case.execute(dataset_name)
    click.echo(f"Successfully created data profile for {profile.dataset_name}")
