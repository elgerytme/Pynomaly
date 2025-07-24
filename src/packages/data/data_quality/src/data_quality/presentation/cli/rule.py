import click
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.application.use_cases.manage_data_quality_rules import ManageDataQualityRulesUseCase
from src.packages.data.data_quality.src.data_quality.di import get_data_quality_rule_service
from src.packages.data.data_quality.src.data_quality.domain.entities.data_quality_rule import DataQualityRule, RuleType, RuleSeverity, RuleCondition, RuleOperator


@click.group()
def rule():
    """Manage data quality rules."""
    pass


@rule.command()
@click.argument("name")
@click.argument("dataset_name")
@click.option("--description", default="", help="Description of the rule.")
@click.option("--rule-type", type=click.Choice([rt.value for rt in RuleType]), default=RuleType.NOT_NULL.value, help="Type of the rule.")
@click.option("--severity", type=click.Choice([rs.value for rs in RuleSeverity]), default=RuleSeverity.ERROR.value, help="Severity of the rule.")
@click.option("--column-name", help="Column name for the rule condition.")
@click.option("--operator", type=click.Choice([ro.value for ro in RuleOperator]), help="Operator for the rule condition.")
@click.option("--value", help="Value for the rule condition.")
def create(
    name: str,
    dataset_name: str,
    description: str,
    rule_type: str,
    severity: str,
    column_name: str,
    operator: str,
    value: str
):
    """Create a new data quality rule."""
    use_case = ManageDataQualityRulesUseCase(get_data_quality_rule_service())
    
    conditions = []
    if column_name and operator:
        condition = RuleCondition(
            column_name=column_name,
            operator=RuleOperator(operator),
            value=value
        )
        conditions.append(condition)

    new_rule = DataQualityRule(
        name=name,
        description=description,
        rule_type=RuleType(rule_type),
        severity=RuleSeverity(severity),
        dataset_name=dataset_name,
        conditions=conditions
    )
    created_rule = use_case.create_rule(new_rule)
    click.echo(f"Successfully created data quality rule {created_rule.name} with ID {created_rule.id}")


@rule.command()
@click.argument("rule_id")
def get(rule_id: str):
    """Get a data quality rule by its ID."""
    use_case = ManageDataQualityRulesUseCase(get_data_quality_rule_service())
    retrieved_rule = use_case.get_rule(UUID(rule_id))
    if retrieved_rule:
        click.echo(f"Rule Name: {retrieved_rule.name}")
        click.echo(f"Description: {retrieved_rule.description}")
        click.echo(f"Rule Type: {retrieved_rule.rule_type.value}")
        click.echo(f"Severity: {retrieved_rule.severity.value}")
        click.echo(f"Dataset: {retrieved_rule.dataset_name}")
        click.echo("Conditions:")
        for condition in retrieved_rule.conditions:
            click.echo(f"  - Column: {condition.column_name}, Operator: {condition.operator.value}, Value: {condition.value}")
    else:
        click.echo(f"Rule with ID {rule_id} not found.")


@rule.command()
def list():
    """List all data quality rules."""
    # This will require a new use case to list all rules
    click.echo("Listing all data quality rules (not yet implemented).")