"""Command-line interface for neuro-symbolic AI package."""

import click
import json
from typing import Optional

from ...application.services.neuro_symbolic_service import NeuroSymbolicService


@click.group()
@click.pass_context
def cli(ctx):
    """Neuro-Symbolic AI CLI"""
    ctx.ensure_object(dict)
    ctx.obj['service'] = NeuroSymbolicService()


@cli.command()
@click.argument('name')
@click.option('--neural-backbone', default='transformer', help='Neural network backbone')
@click.option('--symbolic-reasoner', default='first_order_logic', help='Symbolic reasoning engine')
@click.pass_context
def create_model(ctx, name: str, neural_backbone: str, symbolic_reasoner: str):
    """Create a new neuro-symbolic model."""
    service = ctx.obj['service']
    
    model = service.create_model(
        name=name,
        neural_backbone=neural_backbone,
        symbolic_reasoner=symbolic_reasoner
    )
    
    click.echo(f"Created model: {model.id}")
    click.echo(f"Name: {model.name}")
    click.echo(f"Neural backbone: {model.neural_backbone}")
    click.echo(f"Symbolic reasoner: {model.symbolic_reasoner}")


# Knowledge graph commands have been moved to the knowledge_graph package


@cli.command()
@click.pass_context
def list_models(ctx):
    """List all registered models."""
    service = ctx.obj['service']
    
    models = service.list_models()
    
    if not models:
        click.echo("No models found.")
        return
    
    click.echo("Registered models:")
    for model in models:
        click.echo(f"  ID: {model['id']}")
        click.echo(f"  Name: {model['name']}")
        click.echo(f"  Neural backbone: {model['neural_backbone']}")
        click.echo(f"  Symbolic reasoner: {model['symbolic_reasoner']}")
        click.echo(f"  Trained: {model['is_trained']}")
        click.echo()


@cli.command()
@click.argument('model_id')
@click.argument('input_data')
@click.pass_context
def predict(ctx, model_id: str, input_data: str):
    """Make prediction with explanation."""
    service = ctx.obj['service']
    
    try:
        # Parse input data as JSON
        data = json.loads(input_data)
        
        result = service.predict_with_explanation(model_id, data)
        
        click.echo("Prediction Result:")
        click.echo(f"  Prediction: {result.prediction}")
        click.echo(f"  Confidence: {result.confidence:.3f}")
        click.echo(f"  Explanation: {result.get_explanation_summary()}")
        
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
    except json.JSONDecodeError:
        click.echo("Error: Invalid JSON input", err=True)


if __name__ == '__main__':
    cli()