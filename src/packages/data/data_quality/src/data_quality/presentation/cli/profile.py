
import click

from ...application.use_cases.create_data_profile import CreateDataProfileUseCase
from ...di import get_data_profiling_service
from ...infrastructure.adapters.pandas_csv_adapter import PandasCSVAdapter


@click.group()
def profile():
    """Manage data profiles."""
    pass


@profile.command()
@click.argument("dataset_name")
@click.option("--file-path", required=True, help="Path to the CSV file to profile.")
def create(dataset_name: str, file_path: str):
    """Create a data profile for a dataset."""
    use_case = CreateDataProfileUseCase(get_data_profiling_service())
    adapter = PandasCSVAdapter()
    profile = use_case.execute(dataset_name, adapter, {"file_path": file_path})
    click.echo(f"Successfully created data profile for {profile.dataset_name}")


@profile.command()
def list():
    """List all data profiles."""
    # This will require a new use case to list all profiles
    click.echo("Listing all data profiles (not yet implemented).")
