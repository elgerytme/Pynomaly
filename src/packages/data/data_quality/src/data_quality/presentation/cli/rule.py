
import click
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.application.use_cases.manage_data_quality_rules import ManageDataQualityRulesUseCase
from src.packages.data.data_quality.src.data_quality.di import data_quality_rule_service
from src.packages.data.data_quality.src.data_quality.domain.entities.data_quality_rule import DataQualityRule


@click.group()
def rule():
    """Manage data quality rules."""
    pass


@rule.command()
@click.argument("name")
@click.argument("dataset_name")
def create(name: str, dataset_name: str):
    """Create a new data quality rule."""
    use_case = ManageDataQualityRulesUseCase(data_quality_rule_service)
    new_rule = DataQualityRule(name=name, dataset_name=dataset_name)
    created_rule = use_case.create_rule(new_rule)
    click.echo(f"Successfully created data quality rule {created_rule.name}")


@rule.command()
@click.argument("rule_id")
def get(rule_id: str):
    """Get a data quality rule by its ID."""
    use_case = ManageDataQualityRulesUseCase(data_quality_rule_service)
    retrieved_rule = use_case.get_rule(UUID(rule_id))
    click.echo(retrieved_rule)
