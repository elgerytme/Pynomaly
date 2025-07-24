"""Command-line interface for the Anomaly Detection SDK."""

import json
import sys
import asyncio
from typing import List, Optional
from pathlib import Path

import click
import pandas as pd

from .client import AnomalyDetectionClient
from .async_client import AsyncAnomalyDetectionClient
from .streaming_client import StreamingClient
from .models import AlgorithmType, BatchProcessingRequest, TrainingRequest
from .exceptions import AnomalyDetectionSDKError


@click.group()
@click.option('--base-url', default='http://localhost:8000', help='Base URL of the service')
@click.option('--api-key', help='API key for authentication')
@click.option('--timeout', default=30.0, help='Request timeout in seconds')
@click.pass_context
def cli(ctx, base_url: str, api_key: Optional[str], timeout: float):
    """Anomaly Detection SDK Command Line Interface."""
    ctx.ensure_object(dict)
    ctx.obj['base_url'] = base_url
    ctx.obj['api_key'] = api_key
    ctx.obj['timeout'] = timeout


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--algorithm', type=click.Choice([e.value for e in AlgorithmType]), 
              default='isolation_forest', help='Algorithm to use')
@click.option('--output', '-o', help='Output file for results')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']), 
              default='json', help='Output format')
@click.pass_context
def detect(ctx, data_file: str, algorithm: str, output: Optional[str], output_format: str):
    """Detect anomalies in data from a file."""
    try:
        # Load data
        data_path = Path(data_file)
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            data = df.values.tolist()
        elif data_path.suffix == '.json':
            with open(data_path) as f:
                data = json.load(f)
        else:
            raise click.ClickException(f"Unsupported file format: {data_path.suffix}")
        
        # Detect anomalies
        with AnomalyDetectionClient(
            base_url=ctx.obj['base_url'],
            api_key=ctx.obj['api_key'],
            timeout=ctx.obj['timeout']
        ) as client:
            result = client.detect_anomalies(data, algorithm=algorithm)
        
        # Output results
        if output:
            output_path = Path(output)
            if output_format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(result.model_dump(), f, indent=2, default=str)
            elif output_format == 'csv':
                df = pd.DataFrame([anomaly.model_dump() for anomaly in result.anomalies])
                df.to_csv(output_path, index=False)
            click.echo(f"Results saved to {output_path}")
        else:
            click.echo(json.dumps(result.model_dump(), indent=2, default=str))
            
    except AnomalyDetectionSDKError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--algorithm', type=click.Choice([e.value for e in AlgorithmType]), 
              default='isolation_forest', help='Algorithm to train')
@click.option('--model-name', help='Name for the trained model')
@click.option('--validation-split', default=0.2, help='Validation data split ratio')
@click.pass_context
def train(ctx, data_file: str, algorithm: str, model_name: Optional[str], validation_split: float):
    """Train a new anomaly detection model."""
    try:
        # Load training data
        data_path = Path(data_file)
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            data = df.values.tolist()
        elif data_path.suffix == '.json':
            with open(data_path) as f:
                data = json.load(f)
        else:
            raise click.ClickException(f"Unsupported file format: {data_path.suffix}")
        
        # Train model
        request = TrainingRequest(
            data=data,
            algorithm=AlgorithmType(algorithm),
            model_name=model_name,
            validation_split=validation_split
        )
        
        with AnomalyDetectionClient(
            base_url=ctx.obj['base_url'],
            api_key=ctx.obj['api_key'],
            timeout=ctx.obj['timeout']
        ) as client:
            result = client.train_model(request)
        
        click.echo(f"Model trained successfully!")
        click.echo(f"Model ID: {result.model_id}")
        click.echo(f"Training time: {result.training_time:.2f}s")
        click.echo(f"Performance metrics: {result.performance_metrics}")
        
    except AnomalyDetectionSDKError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def models(ctx):
    """List all available models."""
    try:
        with AnomalyDetectionClient(
            base_url=ctx.obj['base_url'],
            api_key=ctx.obj['api_key'],
            timeout=ctx.obj['timeout']
        ) as client:
            models = client.list_models()
        
        if not models:
            click.echo("No models found.")
            return
        
        click.echo(f"Found {len(models)} models:")
        for model in models:
            click.echo(f"  - {model.model_id} ({model.algorithm}) - {model.status}")
            click.echo(f"    Created: {model.created_at}")
            click.echo(f"    Version: {model.version}")
            click.echo()
            
    except AnomalyDetectionSDKError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model_id')
@click.pass_context
def model_info(ctx, model_id: str):
    """Get detailed information about a model."""
    try:
        with AnomalyDetectionClient(
            base_url=ctx.obj['base_url'],
            api_key=ctx.obj['api_key'],
            timeout=ctx.obj['timeout']
        ) as client:
            model = client.get_model(model_id)
        
        click.echo(json.dumps(model.model_dump(), indent=2, default=str))
        
    except AnomalyDetectionSDKError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--ws-url', default='ws://localhost:8000/ws/stream', help='WebSocket URL')
@click.option('--algorithm', type=click.Choice([e.value for e in AlgorithmType]), 
              default='isolation_forest', help='Algorithm to use')
@click.option('--threshold', default=0.5, help='Detection threshold')
@click.pass_context
def stream(ctx, ws_url: str, algorithm: str, threshold: float):
    """Start streaming anomaly detection."""
    try:
        from .models import StreamingConfig
        
        config = StreamingConfig(
            algorithm=AlgorithmType(algorithm),
            detection_threshold=threshold
        )
        
        client = StreamingClient(
            ws_url=ws_url, 
            config=config,
            api_key=ctx.obj['api_key']
        )
        
        @client.on_connect
        def on_connect():
            click.echo("Connected to streaming service")
        
        @client.on_anomaly
        def on_anomaly(anomaly):
            click.echo(f"ANOMALY DETECTED: {anomaly.model_dump()}")
        
        @client.on_error
        def on_error(error):
            click.echo(f"Error: {error}", err=True)
        
        client.start()
        
        click.echo("Streaming client started. Send data points (JSON arrays) or 'quit' to exit:")
        
        try:
            while True:
                line = input("> ").strip()
                if line.lower() in ['quit', 'exit', 'q']:
                    break
                
                try:
                    data_point = json.loads(line)
                    if isinstance(data_point, list):
                        client.send_data(data_point)
                    else:
                        click.echo("Data point must be a JSON array")
                except json.JSONDecodeError:
                    click.echo("Invalid JSON format")
        except KeyboardInterrupt:
            pass
        finally:
            client.stop()
            click.echo("Streaming client stopped")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def health(ctx):
    """Check service health status."""
    try:
        with AnomalyDetectionClient(
            base_url=ctx.obj['base_url'],
            api_key=ctx.obj['api_key'],
            timeout=ctx.obj['timeout']
        ) as client:
            health_status = client.get_health()
        
        click.echo(json.dumps(health_status.model_dump(), indent=2, default=str))
        
    except AnomalyDetectionSDKError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()