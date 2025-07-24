"""Model management commands for Typer CLI."""

import typer
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import uuid
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...infrastructure.repositories.model_repository import ModelRepository
from ...domain.entities.model import ModelStatus, Model, ModelMetadata, SerializationFormat
from ...domain.entities.dataset import Dataset, DatasetType, DatasetMetadata
from ...domain.services.detection_service import DetectionService
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


@app.command()
def train(
    input_file: Path = typer.Option(..., "--input", "-i", help="Training dataset file path"),
    model_name: str = typer.Option(..., "--model-name", "-n", help="Name for the trained model"),
    algorithm: str = typer.Option("isolation_forest", "--algorithm", "-a", 
                                 help="Algorithm to train"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", 
                                       help="Contamination rate"),
    output_dir: Path = typer.Option(Path("models"), "--output-dir", "-o", 
                                   help="Output directory for saved model"),
    format: str = typer.Option("pickle", "--format", "-f", 
                              help="Model serialization format"),
    has_labels: bool = typer.Option(False, "--has-labels", help="Dataset includes ground truth labels"),
    label_column: str = typer.Option("label", "--label-column", help="Name of label column"),
) -> None:
    """Train and save an anomaly detection model."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Load dataset
        load_task = progress.add_task("Loading training dataset...", total=None)
        
        try:
            if not input_file.exists():
                print(f"[red]✗[/red] Input file '{input_file}' not found")
                raise typer.Exit(1)
            
            # Load data based on file extension
            if input_file.suffix.lower() == '.csv':
                df = pd.read_csv(input_file)
            elif input_file.suffix.lower() == '.json':
                df = pd.read_json(input_file)
            elif input_file.suffix.lower() == '.parquet':
                df = pd.read_parquet(input_file)
            else:
                print(f"[red]✗[/red] Unsupported file format '{input_file.suffix}'")
                raise typer.Exit(1)
            
            progress.update(load_task, completed=True)
            print(f"[green]✓[/green] Loaded training dataset with {len(df)} samples, {len(df.columns)} features")
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to load dataset: {e}")
            raise typer.Exit(1)
        
        # Process labels
        labels = None
        if has_labels and label_column in df.columns:
            labels = df[label_column].values
            df = df.drop(columns=[label_column])
            print(f"[blue]ℹ[/blue] Using '{label_column}' as labels column")
        
        # Create dataset
        dataset = Dataset(
            data=df,
            dataset_type=DatasetType.TRAINING,
            labels=labels,
            metadata=DatasetMetadata(
                name=input_file.stem,
                source=str(input_file),
                description=f"Training dataset loaded from {input_file}"
            )
        )
        
        # Validate dataset
        validation_issues = dataset.validate()
        if validation_issues:
            print("[yellow]⚠[/yellow] Dataset validation issues found:")
            for issue in validation_issues:
                print(f"  • {issue}")
        
        # Train model
        train_task = progress.add_task(f"Training {algorithm} model '{model_name}'...", total=None)
        
        try:
            start_time = datetime.utcnow()
            
            # Initialize services
            service = DetectionService()
            data_array = dataset.to_numpy()
            
            # Algorithm mapping
            algorithm_map = {
                'isolation_forest': 'iforest',
                'one_class_svm': 'ocsvm',
                'lof': 'lof'
            }
            
            # Fit the model
            service.fit(data_array, algorithm_map.get(algorithm, algorithm), contamination=contamination)
            
            # Get predictions for evaluation
            detection_result = service.detect_anomalies(
                data=data_array,
                algorithm=algorithm_map.get(algorithm, algorithm),
                contamination=contamination
            )
            
            end_time = datetime.utcnow()
            training_duration = (end_time - start_time).total_seconds()
            
            progress.update(train_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Model training failed: {e}")
            raise typer.Exit(1)
        
        # Calculate metrics if labels available
        accuracy, precision, recall, f1_score_val = None, None, None, None
        if has_labels and labels is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            pred_labels = detection_result.predictions
            accuracy = float(accuracy_score(labels, pred_labels))
            precision = float(precision_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary'))
            recall = float(recall_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary'))
            f1_score_val = float(f1_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary'))
        
        # Save model
        save_task = progress.add_task("Saving trained model...", total=None)
        
        try:
            # Create model entity
            model_id = str(uuid.uuid4())
            metadata = ModelMetadata(
                model_id=model_id,
                name=model_name,
                algorithm=algorithm,
                status=ModelStatus.TRAINED,
                training_samples=dataset.n_samples,
                training_features=dataset.n_features,
                contamination_rate=contamination,
                training_duration_seconds=training_duration,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score_val,
                feature_names=dataset.feature_names,
                hyperparameters={'contamination': contamination},
                description=f"Trained {algorithm} model on {dataset.n_samples} samples",
            )
            
            # Get the trained model object from the service
            trained_model_obj = service._fitted_models.get(algorithm_map.get(algorithm, algorithm))
            
            model = Model(
                metadata=metadata,
                model_object=trained_model_obj
            )
            
            # Save model using repository
            repo = ModelRepository(str(output_dir))
            format_map = {
                'pickle': SerializationFormat.PICKLE,
                'joblib': SerializationFormat.JOBLIB,
                'json': SerializationFormat.JSON
            }
            format_enum = format_map.get(format, SerializationFormat.PICKLE)
            saved_model_id = repo.save(model, format_enum)
            
            progress.update(save_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to save model: {e}")
            raise typer.Exit(1)
    
    # Display results
    table = Table(title=f"[bold blue]Model Training Results[/bold blue]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model ID", saved_model_id)
    table.add_row("Name", model_name)
    table.add_row("Algorithm", algorithm)
    table.add_row("Training Time", f"{training_duration:.2f} seconds")
    table.add_row("Training Samples", str(dataset.n_samples))
    table.add_row("Features", str(dataset.n_features))
    table.add_row("Contamination", str(contamination))
    table.add_row("Output Directory", str(output_dir))
    
    console.print(table)
    
    if has_labels and labels is not None:
        metrics_table = Table(title="[bold blue]Performance Metrics[/bold blue]")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green", justify="right")
        
        metrics_table.add_row("Accuracy", f"{accuracy:.3f}")
        metrics_table.add_row("Precision", f"{precision:.3f}")
        metrics_table.add_row("Recall", f"{recall:.3f}")
        metrics_table.add_row("F1-Score", f"{f1_score_val:.3f}")
        
        console.print(metrics_table)
    
    print(f"[green]✅[/green] Model training completed successfully!")


@app.command()
def predict(
    model_id: str = typer.Argument(..., help="Model ID to use for prediction"),
    input_file: Path = typer.Option(..., "--input", "-i", help="Input dataset file path"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path for predictions"),
    models_dir: Path = typer.Option(Path("models"), "--models-dir", "-d", help="Models directory"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Detection threshold"),
) -> None:
    """Make predictions using a trained model."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Load model
        load_model_task = progress.add_task("Loading trained model...", total=None)
        
        try:
            repo = ModelRepository(str(models_dir))
            model = repo.load(model_id)
            
            progress.update(load_model_task, completed=True)
            print(f"[green]✓[/green] Loaded model '{model.metadata.name}' ({model.metadata.algorithm})")
            
        except FileNotFoundError:
            print(f"[red]✗[/red] Model with ID '{model_id}' not found")
            raise typer.Exit(1)
        except Exception as e:
            print(f"[red]✗[/red] Failed to load model: {e}")
            raise typer.Exit(1)
        
        # Load dataset
        load_data_task = progress.add_task("Loading input dataset...", total=None)
        
        try:
            if not input_file.exists():
                print(f"[red]✗[/red] Input file '{input_file}' not found")
                raise typer.Exit(1)
            
            # Load data based on file extension
            if input_file.suffix.lower() == '.csv':
                df = pd.read_csv(input_file)
            elif input_file.suffix.lower() == '.json':
                df = pd.read_json(input_file)
            elif input_file.suffix.lower() == '.parquet':
                df = pd.read_parquet(input_file)
            else:
                print(f"[red]✗[/red] Unsupported file format '{input_file.suffix}'")
                raise typer.Exit(1)
            
            progress.update(load_data_task, completed=True)
            print(f"[green]✓[/green] Loaded input dataset with {len(df)} samples, {len(df.columns)} features")
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to load dataset: {e}")
            raise typer.Exit(1)
        
        # Make predictions
        predict_task = progress.add_task("Making predictions...", total=None)
        
        try:
            # Convert to numpy array
            data_array = df.to_numpy()
            
            # Use the model object for prediction
            predictions = model.model_object.predict(data_array)
            scores = model.model_object.decision_function(data_array)
            
            progress.update(predict_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Prediction failed: {e}")
            raise typer.Exit(1)
        
        # Process results
        results_df = df.copy()
        results_df['anomaly_score'] = scores
        results_df['is_anomaly'] = predictions == -1  # Scikit-learn convention: -1 = anomaly
        results_df['prediction'] = predictions
        
        # Apply threshold if provided
        if threshold is not None:
            results_df['is_anomaly'] = results_df['anomaly_score'] > threshold
        
        # Count anomalies
        n_anomalies = results_df['is_anomaly'].sum()
        anomaly_rate = n_anomalies / len(results_df)
        
        # Display results
        results_table = Table(title="[bold blue]Prediction Results[/bold blue]")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Total Samples", str(len(results_df)))
        results_table.add_row("Detected Anomalies", str(n_anomalies))
        results_table.add_row("Anomaly Rate", f"{anomaly_rate:.1%}")
        results_table.add_row("Model Algorithm", model.metadata.algorithm)
        
        if threshold is not None:
            results_table.add_row("Applied Threshold", str(threshold))
        
        console.print(results_table)
        
        # Save output if specified
        if output_file:
            try:
                if output_file.suffix.lower() == '.csv':
                    results_df.to_csv(output_file, index=False)
                elif output_file.suffix.lower() == '.json':
                    results_df.to_json(output_file, orient='records', indent=2)
                elif output_file.suffix.lower() == '.parquet':
                    results_df.to_parquet(output_file, index=False)
                else:
                    print(f"[yellow]⚠[/yellow] Unsupported output format '{output_file.suffix}', using CSV")
                    output_file = output_file.with_suffix('.csv')
                    results_df.to_csv(output_file, index=False)
                
                print(f"[green]✅[/green] Predictions saved to: {output_file}")
                
            except Exception as e:
                print(f"[red]✗[/red] Failed to save predictions: {e}")
                raise typer.Exit(1)
        else:
            # Display sample predictions
            sample_table = Table(title="[bold blue]Sample Predictions (first 10 rows)[/bold blue]")
            
            # Add original columns (limit to first few for display)
            display_cols = df.columns[:3].tolist()  # Show first 3 original columns
            for col in display_cols:
                sample_table.add_column(col, style="dim")
            
            sample_table.add_column("Anomaly Score", style="blue", justify="right")
            sample_table.add_column("Is Anomaly", style="red")
            
            for i in range(min(10, len(results_df))):
                row_data = []
                for col in display_cols:
                    row_data.append(str(results_df[col].iloc[i]))
                row_data.append(f"{results_df['anomaly_score'].iloc[i]:.3f}")
                row_data.append("Yes" if results_df['is_anomaly'].iloc[i] else "No")
                sample_table.add_row(*row_data)
            
            console.print(sample_table)
    
    print(f"[green]✅[/green] Prediction completed successfully!")


# Export command functions for backward compatibility
train_command = train
predict_command = predict