"""Model management commands for Typer CLI."""

import typer
from pathlib import Path
from typing import Optional
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...infrastructure.repositories.model_repository import ModelRepository
from ...domain.entities.model import ModelStatus
from ...infrastructure.logging import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(help="Model management commands")


@app.command()
def list(
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
    algorithm: Optional[str] = typer.Option(None, "--algorithm", "-a", help="Filter by algorithm"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
) -> None:
    """List saved models."""
    
    try:
        repo = ModelRepository(str(models_dir))
        
        status_filter = ModelStatus(status) if status else None
        models = repo.list_models(status=status_filter, algorithm=algorithm)
        
        if not models:
            print("[yellow]ℹ[/yellow] No models found matching the criteria.")
            return
        
        table = Table(title=f"[bold blue]Found {len(models)} model(s)[/bold blue]")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Algorithm", style="green")
        table.add_column("Status", style="blue")
        table.add_column("Created", style="dim")
        table.add_column("Accuracy", style="green", justify="right")
        
        for model in models:
            model_id_short = model['model_id'][:8] + "..." if len(model['model_id']) > 8 else model['model_id']
            accuracy = f"{model['accuracy']:.3f}" if model.get('accuracy') else "N/A"
            
            table.add_row(
                model['name'],
                model_id_short,
                model['algorithm'],
                model['status'],
                model['created_at'],
                accuracy
            )
        
        console.print(table)
        
    except Exception as e:
        print(f"[red]✗[/red] Error listing models: {e}")
        raise typer.Exit(1)


@app.command()
def info(
    model_id: str = typer.Argument(..., help="Model ID to show information for"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
) -> None:
    """Show detailed information about a model."""
    
    try:
        repo = ModelRepository(str(models_dir))
        metadata = repo.get_model_metadata(model_id)
        
        # Basic info table
        info_table = Table(title=f"[bold blue]Model Information: {metadata['name']}[/bold blue]")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("ID", metadata['model_id'])
        info_table.add_row("Name", metadata['name'])
        info_table.add_row("Algorithm", metadata['algorithm'])
        info_table.add_row("Version", str(metadata['version']))
        info_table.add_row("Status", metadata['status'])
        info_table.add_row("Created", metadata['created_at'])
        info_table.add_row("Updated", metadata['updated_at'])
        
        if metadata.get('description'):
            info_table.add_row("Description", metadata['description'])
        
        console.print(info_table)
        
        # Training info table
        if any(metadata.get(key) for key in ['training_samples', 'training_features', 'contamination_rate', 'training_duration_seconds']):
            training_table = Table(title="[bold blue]Training Information[/bold blue]")
            training_table.add_column("Property", style="cyan")
            training_table.add_column("Value", style="green")
            
            if metadata.get('training_samples'):
                training_table.add_row("Training Samples", str(metadata['training_samples']))
            if metadata.get('training_features'):
                training_table.add_row("Features", str(metadata['training_features']))
            if metadata.get('contamination_rate'):
                training_table.add_row("Contamination Rate", f"{metadata['contamination_rate']:.1%}")
            if metadata.get('training_duration_seconds'):
                training_table.add_row("Training Time", f"{metadata['training_duration_seconds']:.2f} seconds")
            
            console.print(training_table)
        
        # Performance metrics table
        if any(metadata.get(metric) for metric in ['accuracy', 'precision', 'recall', 'f1_score']):
            metrics_table = Table(title="[bold blue]Performance Metrics[/bold blue]")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green", justify="right")
            
            if metadata.get('accuracy'):
                metrics_table.add_row("Accuracy", f"{metadata['accuracy']:.3f}")
            if metadata.get('precision'):
                metrics_table.add_row("Precision", f"{metadata['precision']:.3f}")
            if metadata.get('recall'):
                metrics_table.add_row("Recall", f"{metadata['recall']:.3f}")
            if metadata.get('f1_score'):
                metrics_table.add_row("F1-Score", f"{metadata['f1_score']:.3f}")
            
            console.print(metrics_table)
        
        # Hyperparameters
        if metadata.get('hyperparameters'):
            params_table = Table(title="[bold blue]Hyperparameters[/bold blue]")
            params_table.add_column("Parameter", style="cyan")
            params_table.add_column("Value", style="green")
            
            for param, value in metadata['hyperparameters'].items():
                params_table.add_row(param, str(value))
            
            console.print(params_table)
        
        # Tags
        if metadata.get('tags'):
            print(f"[blue]🏷️ Tags:[/blue] {', '.join(metadata['tags'])}")
        
    except FileNotFoundError:
        print(f"[red]✗[/red] Model with ID '{model_id}' not found")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Error getting model info: {e}")
        raise typer.Exit(1)


@app.command()
def delete(
    model_id: str = typer.Argument(..., help="Model ID to delete"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a saved model."""
    
    try:
        repo = ModelRepository(str(models_dir))
        
        # Get model info first
        try:
            metadata = repo.get_model_metadata(model_id)
            model_name = metadata['name']
        except FileNotFoundError:
            print(f"[red]✗[/red] Model with ID '{model_id}' not found")
            raise typer.Exit(1)
        
        # Confirmation
        if not force:
            confirm = typer.confirm(f"Are you sure you want to delete model '{model_name}' ({model_id})?")
            if not confirm:
                print("[yellow]ℹ[/yellow] Deletion cancelled")
                return
        
        # Delete model
        if repo.delete(model_id):
            print(f"[green]✅[/green] Model '{model_name}' ({model_id}) deleted successfully")
        else:
            print(f"[red]✗[/red] Failed to delete model '{model_id}'")
            raise typer.Exit(1)
        
    except Exception as e:
        print(f"[red]✗[/red] Error deleting model: {e}")
        raise typer.Exit(1)


@app.command()
def stats(
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
) -> None:
    """Show repository statistics."""
    
    try:
        repo = ModelRepository(str(models_dir))
        stats = repo.get_repository_stats()
        
        # Main stats table
        stats_table = Table(title="[bold blue]Model Repository Statistics[/bold blue]")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Models", str(stats['total_models']))
        stats_table.add_row("Storage Size", f"{stats['storage_size_mb']} MB")
        stats_table.add_row("Storage Path", stats['storage_path'])
        
        console.print(stats_table)
        
        # By status table
        if stats.get('by_status'):
            status_table = Table(title="[bold blue]Models by Status[/bold blue]")
            status_table.add_column("Status", style="cyan")
            status_table.add_column("Count", style="green", justify="right")
            
            for status, count in stats['by_status'].items():
                status_table.add_row(status, str(count))
            
            console.print(status_table)
        
        # By algorithm table
        if stats.get('by_algorithm'):
            algo_table = Table(title="[bold blue]Models by Algorithm[/bold blue]")
            algo_table.add_column("Algorithm", style="cyan")
            algo_table.add_column("Count", style="green", justify="right")
            
            for algorithm, count in stats['by_algorithm'].items():
                algo_table.add_row(algorithm, str(count))
            
            console.print(algo_table)
        
    except Exception as e:
        print(f"[red]✗[/red] Error getting repository stats: {e}")
        raise typer.Exit(1)


@app.command()
def export(
    model_id: str = typer.Argument(..., help="Model ID to export"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file path"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json, yaml)"),
) -> None:
    """Export model metadata to file."""
    
    try:
        repo = ModelRepository(str(models_dir))
        metadata = repo.get_model_metadata(model_id)
        
        if format.lower() == "json":
            import json
            with open(output, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        elif format.lower() == "yaml":
            import yaml
            with open(output, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
        else:
            print(f"[red]✗[/red] Unsupported format: {format}")
            raise typer.Exit(1)
        
        print(f"[green]✅[/green] Model metadata exported to: {output}")
        
    except FileNotFoundError:
        print(f"[red]✗[/red] Model with ID '{model_id}' not found")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]✗[/red] Error exporting model: {e}")
        raise typer.Exit(1)