"""Detection commands for Typer CLI."""

import json
import typer
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...domain.services.detection_service import DetectionService
from ...domain.services.ensemble_service import EnsembleService
from ...domain.entities.dataset import Dataset, DatasetType, DatasetMetadata
from ...infrastructure.logging import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(help="Anomaly detection commands")


@app.command()
def run(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input data file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output results file path"),
    algorithm: str = typer.Option("isolation_forest", "--algorithm", "-a", 
                                 help="Detection algorithm to use"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", 
                                       help="Expected contamination ratio"),
    has_labels: bool = typer.Option(False, "--has-labels", help="Dataset includes ground truth labels"),
    label_column: str = typer.Option("label", "--label-column", help="Name of label column"),
) -> None:
    """Run anomaly detection on dataset."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Load dataset
        load_task = progress.add_task("Loading dataset...", total=None)
        
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
            print(f"[green]✓[/green] Loaded dataset with {len(df)} samples, {len(df.columns)} features")
            
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
            dataset_type=DatasetType.INFERENCE,
            labels=labels,
            metadata=DatasetMetadata(
                name=input_file.stem,
                source=str(input_file),
                description=f"Dataset loaded from {input_file}"
            )
        )
        
        # Validate dataset
        validation_issues = dataset.validate()
        if validation_issues:
            print("[yellow]⚠[/yellow] Dataset validation issues found:")
            for issue in validation_issues:
                print(f"  • {issue}")
        
        # Run detection
        detect_task = progress.add_task(f"Running {algorithm} detection...", total=None)
        
        try:
            service = DetectionService()
            data_array = dataset.to_numpy()
            
            # Map algorithm names
            algorithm_map = {
                'isolation_forest': 'iforest',
                'one_class_svm': 'ocsvm',
                'lof': 'lof',
                'autoencoder': 'autoencoder'
            }
            
            detection_result = service.detect_anomalies(
                data=data_array,
                algorithm=algorithm_map.get(algorithm, algorithm),
                contamination=contamination
            )
            
            progress.update(detect_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Detection failed: {e}")
            raise typer.Exit(1)
    
    # Prepare results
    results = {
        "input": str(input_file),
        "algorithm": algorithm,
        "contamination": contamination,
        "dataset_info": {
            "total_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "feature_names": dataset.feature_names
        },
        "detection_results": {
            "anomalies_detected": detection_result.anomaly_count,
            "normal_samples": detection_result.normal_count,
            "anomaly_rate": detection_result.anomaly_rate,
            "anomaly_indices": detection_result.anomalies
        },
        "timestamp": detection_result.timestamp.isoformat() if detection_result.timestamp else None,
        "success": detection_result.success
    }
    
    # Add evaluation metrics if ground truth labels are available
    if has_labels and labels is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        
        pred_labels = detection_result.predictions
        
        results["evaluation_metrics"] = {
            "accuracy": float(accuracy_score(labels, pred_labels)),
            "precision": float(precision_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary')),
            "recall": float(recall_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary')),
            "f1_score": float(f1_score(labels, pred_labels, pos_label=-1, zero_division=0, average='binary')),
        }
    
    # Output results
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[green]✓[/green] Results saved to: {output}")
    
    # Display summary
    table = Table(title="[bold blue]Detection Results[/bold blue]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Algorithm", algorithm)
    table.add_row("Total Samples", str(dataset.n_samples))
    table.add_row("Anomalies Detected", str(detection_result.anomaly_count))
    table.add_row("Anomaly Rate", f"{detection_result.anomaly_rate:.1%}")
    table.add_row("Contamination", str(contamination))
    
    if has_labels and labels is not None and results.get("evaluation_metrics"):
        metrics = results["evaluation_metrics"]
        table.add_row("Accuracy", f"{metrics['accuracy']:.3f}")
        table.add_row("Precision", f"{metrics['precision']:.3f}")
        table.add_row("Recall", f"{metrics['recall']:.3f}")
        table.add_row("F1-Score", f"{metrics['f1_score']:.3f}")
    
    console.print(table)
    print(f"[green]✅[/green] Detection completed successfully!")


@app.command()
def ensemble(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input data file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output results file path"),
    algorithms: List[str] = typer.Option(["isolation_forest", "one_class_svm", "lof"], 
                                        "--algorithms", "-a", help="Algorithms to use in ensemble"),
    method: str = typer.Option("majority", "--method", "-m", 
                              help="Ensemble combination method"),
    contamination: float = typer.Option(0.1, "--contamination", "-c", 
                                       help="Expected contamination ratio"),
) -> None:
    """Run ensemble anomaly detection."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Load dataset
        load_task = progress.add_task("Loading dataset...", total=None)
        
        try:
            if not input_file.exists():
                print(f"[red]✗[/red] Input file '{input_file}' not found")
                raise typer.Exit(1)
            
            if input_file.suffix.lower() == '.csv':
                df = pd.read_csv(input_file)
            elif input_file.suffix.lower() == '.json':
                df = pd.read_json(input_file)
            else:
                print(f"[red]✗[/red] Unsupported file format '{input_file.suffix}'")
                raise typer.Exit(1)
            
            data_array = df.select_dtypes(include=[np.number]).values.astype(np.float64)
            progress.update(load_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to load dataset: {e}")
            raise typer.Exit(1)
        
        # Run ensemble detection
        ensemble_task = progress.add_task(f"Running ensemble with {len(algorithms)} algorithms...", total=None)
        
        try:
            # Map algorithm names
            algorithm_map = {
                'isolation_forest': 'iforest',
                'one_class_svm': 'ocsvm',
                'lof': 'lof',
                'autoencoder': 'autoencoder'
            }
            
            mapped_algorithms = [algorithm_map.get(alg, alg) for alg in algorithms]
            
            # Get individual results
            individual_results = {}
            predictions_list = []
            scores_list = []
            
            for algorithm in mapped_algorithms:
                service = DetectionService()
                result = service.detect_anomalies(
                    data=data_array,
                    algorithm=algorithm,
                    contamination=contamination
                )
                individual_results[algorithm] = result
                predictions_list.append(result.predictions)
                if result.confidence_scores is not None:
                    scores_list.append(result.confidence_scores)
            
            # Combine predictions using ensemble method
            ensemble_service = EnsembleService()
            predictions_array = np.array(predictions_list)
            scores_array = np.array(scores_list) if scores_list else None
            
            if method == 'majority':
                ensemble_predictions = ensemble_service.majority_vote(predictions_array)
            elif method == 'average' and scores_array is not None:
                ensemble_predictions, _ = ensemble_service.average_combination(predictions_array, scores_array)
            elif method == 'weighted_average' and scores_array is not None:
                weights = np.ones(len(algorithms)) / len(algorithms)
                ensemble_predictions, _ = ensemble_service.weighted_combination(
                    predictions_array, scores_array, weights
                )
            elif method == 'max' and scores_array is not None:
                ensemble_predictions, _ = ensemble_service.max_combination(predictions_array, scores_array)
            else:
                ensemble_predictions = ensemble_service.majority_vote(predictions_array)
            
            progress.update(ensemble_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Ensemble detection failed: {e}")
            raise typer.Exit(1)
    
    # Calculate ensemble statistics
    ensemble_anomaly_count = int(np.sum(ensemble_predictions == -1))
    ensemble_normal_count = len(ensemble_predictions) - ensemble_anomaly_count
    ensemble_anomaly_rate = ensemble_anomaly_count / len(ensemble_predictions)
    
    # Prepare results
    results = {
        "input": str(input_file),
        "ensemble_config": {
            "algorithms": algorithms,
            "method": method,
            "contamination": contamination
        },
        "individual_results": {
            alg: {
                "anomalies_detected": result.anomaly_count,
                "anomaly_rate": result.anomaly_rate,
                "success": result.success
            } for alg, result in individual_results.items()
        },
        "ensemble_results": {
            "anomalies_detected": ensemble_anomaly_count,
            "normal_samples": ensemble_normal_count,
            "anomaly_rate": ensemble_anomaly_rate,
            "anomaly_indices": np.where(ensemble_predictions == -1)[0].tolist()
        },
        "dataset_info": {
            "total_samples": len(data_array),
            "n_features": data_array.shape[1]
        },
        "success": True
    }
    
    # Output results
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[green]✓[/green] Results saved to: {output}")
    
    # Display summary
    table = Table(title="[bold blue]Ensemble Detection Results[/bold blue]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Method", method)
    table.add_row("Algorithms", ", ".join(algorithms))
    table.add_row("Total Samples", str(len(data_array)))
    table.add_row("Ensemble Anomalies", str(ensemble_anomaly_count))
    table.add_row("Ensemble Rate", f"{ensemble_anomaly_rate:.1%}")
    
    console.print(table)
    
    # Individual results table
    individual_table = Table(title="[bold blue]Individual Algorithm Results[/bold blue]")
    individual_table.add_column("Algorithm", style="cyan")
    individual_table.add_column("Anomalies", style="green")
    individual_table.add_column("Rate", style="green")
    
    for alg, result in individual_results.items():
        individual_table.add_row(
            alg, 
            str(result.anomaly_count), 
            f"{result.anomaly_rate:.1%}"
        )
    
    console.print(individual_table)
    print(f"[green]✅[/green] Ensemble detection completed successfully!")