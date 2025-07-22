"""Data Science CLI interface."""

import click
import structlog
from typing import Optional

from .infrastructure.di.container import Container

logger = structlog.get_logger()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def main(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """Data Science CLI - Experiment management and feature validation."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(20)  # INFO level
        )


@main.group()
def experiments() -> None:
    """Experiment management commands."""
    pass


@experiments.command()
@click.option('--name', '-n', required=True, help='Experiment name')
@click.option('--description', '-d', help='Experiment description')
def create(name: str, description: Optional[str]) -> None:
    """Create a new experiment."""
    logger.info("Creating new experiment", name=name, description=description)
    # Implementation would use application services
    click.echo(f"Created experiment: {name}")


@experiments.command()
@click.argument('experiment_id')
def run(experiment_id: str) -> None:
    """Run an experiment."""
    logger.info("Running experiment", experiment_id=experiment_id)
    # Implementation would use workflow orchestration service
    click.echo(f"Running experiment: {experiment_id}")


@main.group()
def features() -> None:
    """Feature validation commands."""
    pass


@features.command()
@click.option('--dataset', '-d', required=True, help='Dataset path')
@click.option('--config', '-c', help='Validation configuration')
def validate(dataset: str, config: Optional[str]) -> None:
    """Validate dataset features."""
    logger.info("Validating features", dataset=dataset, config=config)
    # Implementation would use feature validator service
    click.echo(f"Validating features in: {dataset}")


@main.group()  
def metrics() -> None:
    """Metrics calculation commands."""
    pass


@metrics.command()
@click.option('--experiment', '-e', required=True, help='Experiment ID')
@click.option('--output', '-o', help='Output file path')
def calculate(experiment: str, output: Optional[str]) -> None:
    """Calculate experiment metrics."""
    logger.info("Calculating metrics", experiment=experiment, output=output)
    # Implementation would use metrics calculator service
    click.echo(f"Calculating metrics for experiment: {experiment}")


if __name__ == '__main__':
    main()