"""Threshold optimization commands for Typer CLI."""

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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score

from ...domain.services.detection_service import DetectionService
from ...domain.entities.dataset import Dataset, DatasetType, DatasetMetadata
from ...infrastructure.logging import get_logger

console = Console()
logger = get_logger(__name__)

app = typer.Typer(help="Threshold optimization commands")


@app.command()
def thresholds(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input data file with labels"),
    label_column: str = typer.Option("label", "--label-column", help="Name of label column"),
    algorithm: str = typer.Option("isolation_forest", "--algorithm", "-a", help="Algorithm to optimize"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output optimization results"),
    objective: str = typer.Option("f1", "--objective", help="Optimization objective (f1, precision, recall, balanced)"),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Cross-validation folds"),
    plot: bool = typer.Option(False, "--plot", help="Generate ROC and PR curves"),
) -> None:
    """Optimize detection thresholds for maximum performance."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Load dataset
        load_task = progress.add_task("Loading labeled dataset...", total=None)
        
        try:
            if not input_file.exists():
                print(f"[red]✗[/red] Input file '{input_file}' not found")
                raise typer.Exit(1)
            
            # Load data
            if input_file.suffix.lower() == '.csv':
                df = pd.read_csv(input_file)
            elif input_file.suffix.lower() == '.json':
                df = pd.read_json(input_file)
            else:
                print(f"[red]✗[/red] Unsupported file format '{input_file.suffix}'")
                raise typer.Exit(1)
            
            # Validate label column
            if label_column not in df.columns:
                print(f"[red]✗[/red] Label column '{label_column}' not found in dataset")
                print(f"Available columns: {list(df.columns)}")
                raise typer.Exit(1)
            
            # Separate features and labels
            labels = df[label_column].values
            X = df.drop(columns=[label_column])
            
            # Convert labels to binary (1 for anomaly, 0 for normal)
            unique_labels = np.unique(labels)
            if len(unique_labels) != 2:
                print(f"[red]✗[/red] Expected binary labels, found: {unique_labels}")
                raise typer.Exit(1)
            
            # Assume the minority class is anomalies
            anomaly_class = unique_labels[np.bincount(labels).argmin()]
            y_binary = (labels == anomaly_class).astype(int)
            
            progress.update(load_task, completed=True)
            print(f"[green]✓[/green] Loaded {len(X)} samples with {len(X.columns)} features")
            print(f"[blue]ℹ[/blue] Anomaly class: {anomaly_class} ({np.sum(y_binary)} samples)")
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to load dataset: {e}")
            raise typer.Exit(1)
        
        # Create dataset
        dataset = Dataset(
            data=X,
            dataset_type=DatasetType.EVALUATION,
            labels=y_binary,
            metadata=DatasetMetadata(
                name=input_file.stem,
                source=str(input_file),
                description=f"Threshold optimization dataset from {input_file}"
            )
        )
        
        # Optimize thresholds
        optimize_task = progress.add_task("Optimizing thresholds...", total=None)
        
        try:
            service = DetectionService()
            data_array = dataset.to_numpy()
            
            # Get anomaly scores for different contamination rates
            contamination_rates = np.linspace(0.01, 0.5, 50)
            threshold_results = []
            
            algorithm_map = {
                'isolation_forest': 'iforest',
                'one_class_svm': 'ocsvm',
                'lof': 'lof',
                'autoencoder': 'autoencoder'
            }
            mapped_algorithm = algorithm_map.get(algorithm, algorithm)
            
            for contamination in contamination_rates:
                result = service.detect_anomalies(
                    data=data_array,
                    algorithm=mapped_algorithm,
                    contamination=contamination
                )
                
                # Calculate metrics
                y_pred = (result.predictions == -1).astype(int)
                
                # Calculate precision, recall, F1
                tp = np.sum((y_binary == 1) & (y_pred == 1))
                fp = np.sum((y_binary == 0) & (y_pred == 1))
                fn = np.sum((y_binary == 1) & (y_pred == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                balanced_accuracy = 0.5 * (recall + (1 - fp / (fp + np.sum(y_binary == 0))))
                
                threshold_results.append({
                    'contamination': contamination,
                    'threshold': contamination,  # For isolation forest, contamination ~= threshold
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'balanced_accuracy': balanced_accuracy,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                })
            
            progress.update(optimize_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Threshold optimization failed: {e}")
            raise typer.Exit(1)
    
    # Find optimal threshold based on objective
    results_df = pd.DataFrame(threshold_results)
    
    if objective == 'f1':
        optimal_idx = results_df['f1_score'].idxmax()
        objective_name = "F1-Score"
    elif objective == 'precision':
        optimal_idx = results_df['precision'].idxmax()
        objective_name = "Precision"
    elif objective == 'recall':
        optimal_idx = results_df['recall'].idxmax()
        objective_name = "Recall"
    elif objective == 'balanced':
        optimal_idx = results_df['balanced_accuracy'].idxmax()
        objective_name = "Balanced Accuracy"
    else:
        print(f"[red]✗[/red] Unknown objective: {objective}")
        raise typer.Exit(1)
    
    optimal_result = results_df.iloc[optimal_idx]
    
    # Prepare optimization results
    optimization_results = {
        "input": str(input_file),
        "algorithm": algorithm,
        "objective": objective,
        "optimization_results": {
            "optimal_threshold": float(optimal_result['threshold']),
            "optimal_contamination": float(optimal_result['contamination']),
            "optimal_metrics": {
                "precision": float(optimal_result['precision']),
                "recall": float(optimal_result['recall']),
                "f1_score": float(optimal_result['f1_score']),
                "balanced_accuracy": float(optimal_result['balanced_accuracy'])
            }
        },
        "threshold_analysis": results_df.to_dict('records'),
        "dataset_info": {
            "total_samples": len(y_binary),
            "anomaly_samples": int(np.sum(y_binary)),
            "normal_samples": int(len(y_binary) - np.sum(y_binary)),
            "anomaly_rate": float(np.sum(y_binary) / len(y_binary))
        }
    }
    
    # Output results
    if output:
        with open(output, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        print(f"[green]✓[/green] Optimization results saved to: {output}")
    
    # Display results table
    table = Table(title=f"[bold blue]Threshold Optimization Results - {objective_name}[/bold blue]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Algorithm", algorithm)
    table.add_row("Objective", objective_name)
    table.add_row("Optimal Threshold", f"{optimal_result['threshold']:.4f}")
    table.add_row("Optimal Contamination", f"{optimal_result['contamination']:.4f}")
    table.add_row("Precision", f"{optimal_result['precision']:.3f}")
    table.add_row("Recall", f"{optimal_result['recall']:.3f}")
    table.add_row("F1-Score", f"{optimal_result['f1_score']:.3f}")
    table.add_row("Balanced Accuracy", f"{optimal_result['balanced_accuracy']:.3f}")
    
    console.print(table)
    
    # Generate plots if requested
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Threshold vs metrics
            ax1.plot(results_df['threshold'], results_df['precision'], label='Precision', marker='o')
            ax1.plot(results_df['threshold'], results_df['recall'], label='Recall', marker='s')
            ax1.plot(results_df['threshold'], results_df['f1_score'], label='F1-Score', marker='^')
            ax1.axvline(optimal_result['threshold'], color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Threshold (Contamination)')
            ax1.set_ylabel('Score')
            ax1.set_title('Performance vs Threshold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Precision-Recall curve
            ax2.plot(results_df['recall'], results_df['precision'], marker='o')
            ax2.scatter(optimal_result['recall'], optimal_result['precision'], 
                       color='red', s=100, zorder=5, label='Optimal')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Distribution of metrics
            ax3.hist(results_df['f1_score'], bins=20, alpha=0.7, color='blue', label='F1-Score')
            ax3.axvline(optimal_result['f1_score'], color='red', linestyle='--', alpha=0.7)
            ax3.set_xlabel('F1-Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('F1-Score Distribution')
            ax3.legend()
            
            # Top-K results
            top_results = results_df.nlargest(10, objective.replace('balanced', 'balanced_accuracy'))
            y_pos = np.arange(len(top_results))
            ax4.barh(y_pos, top_results[objective.replace('balanced', 'balanced_accuracy')])
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([f"{t:.3f}" for t in top_results['threshold']])
            ax4.set_xlabel(objective_name)
            ax4.set_ylabel('Threshold')
            ax4.set_title(f'Top 10 Thresholds by {objective_name}')
            
            plt.tight_layout()
            plot_path = input_file.parent / f"threshold_optimization_{algorithm}_{objective}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"[green]✓[/green] Optimization plots saved to: {plot_path}")
            
        except ImportError:
            print("[yellow]⚠[/yellow] Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"[yellow]⚠[/yellow] Failed to generate plots: {e}")
    
    print(f"[green]✅[/green] Threshold optimization completed successfully!")


@app.command()
def roc_analysis(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input data file with labels"),
    label_column: str = typer.Option("label", "--label-column", help="Name of label column"),
    algorithms: List[str] = typer.Option(["isolation_forest"], "--algorithms", "-a", 
                                        help="Algorithms to analyze"),
    output_dir: Path = typer.Option(Path("."), "--output-dir", help="Output directory for plots"),
) -> None:
    """Generate ROC curve analysis for multiple algorithms."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        load_task = progress.add_task("Loading and preparing data...", total=None)
        
        try:
            # Load and prepare data (similar to thresholds command)
            if input_file.suffix.lower() == '.csv':
                df = pd.read_csv(input_file)
            else:
                print(f"[red]✗[/red] Only CSV files supported for ROC analysis")
                raise typer.Exit(1)
            
            if label_column not in df.columns:
                print(f"[red]✗[/red] Label column '{label_column}' not found")
                raise typer.Exit(1)
            
            labels = df[label_column].values
            X = df.drop(columns=[label_column])
            
            unique_labels = np.unique(labels)
            if len(unique_labels) != 2:
                print(f"[red]✗[/red] Expected binary labels, found: {unique_labels}")
                raise typer.Exit(1)
            
            anomaly_class = unique_labels[np.bincount(labels).argmin()]
            y_binary = (labels == anomaly_class).astype(int)
            
            progress.update(load_task, completed=True)
            
        except Exception as e:
            print(f"[red]✗[/red] Failed to load data: {e}")
            raise typer.Exit(1)
        
        # Generate ROC curves
        analysis_task = progress.add_task("Generating ROC analysis...", total=None)
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            algorithm_map = {
                'isolation_forest': 'iforest',
                'one_class_svm': 'ocsvm',
                'lof': 'lof'
            }
            
            service = DetectionService()
            data_array = X.values.astype(np.float64)
            
            roc_results = {}
            
            for algorithm in algorithms:
                mapped_alg = algorithm_map.get(algorithm, algorithm)
                
                # Get anomaly scores
                result = service.detect_anomalies(
                    data=data_array,
                    algorithm=mapped_alg,
                    contamination=0.1
                )
                
                if hasattr(result, 'confidence_scores') and result.confidence_scores is not None:
                    scores = result.confidence_scores
                else:
                    # Use decision function or distance-based scores
                    scores = -(result.predictions)  # Convert predictions to scores
                
                # Calculate ROC curve
                fpr, tpr, roc_thresholds = roc_curve(y_binary, scores)
                roc_auc = auc(fpr, tpr)
                
                # Calculate PR curve
                precision, recall, pr_thresholds = precision_recall_curve(y_binary, scores)
                pr_auc = auc(recall, precision)
                
                # Plot ROC curve
                ax1.plot(fpr, tpr, label=f'{algorithm} (AUC = {roc_auc:.3f})')
                
                # Plot PR curve
                ax2.plot(recall, precision, label=f'{algorithm} (AUC = {pr_auc:.3f})')
                
                roc_results[algorithm] = {
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'precision': precision.tolist(),
                    'recall': recall.tolist()
                }
            
            # Format ROC plot
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curves')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            
            # Format PR plot
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curves')
            ax2.legend(loc="lower left")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            output_dir.mkdir(exist_ok=True)
            plot_path = output_dir / f"roc_analysis_{input_file.stem}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Save results
            results_path = output_dir / f"roc_results_{input_file.stem}.json"
            with open(results_path, 'w') as f:
                json.dump({
                    'algorithms': algorithms,
                    'dataset': str(input_file),
                    'results': roc_results
                }, f, indent=2)
            
            progress.update(analysis_task, completed=True)
            
            print(f"[green]✓[/green] ROC analysis plots saved to: {plot_path}")
            print(f"[green]✓[/green] ROC analysis results saved to: {results_path}")
            
        except ImportError:
            print("[red]✗[/red] Matplotlib required for ROC analysis")
            raise typer.Exit(1)
        except Exception as e:
            print(f"[red]✗[/red] ROC analysis failed: {e}")
            raise typer.Exit(1)
    
    # Display results table
    table = Table(title="[bold blue]ROC Analysis Results[/bold blue]")
    table.add_column("Algorithm", style="cyan")
    table.add_column("ROC AUC", style="green")
    table.add_column("PR AUC", style="green")
    
    for algorithm, results in roc_results.items():
        table.add_row(
            algorithm,
            f"{results['roc_auc']:.3f}",
            f"{results['pr_auc']:.3f}"
        )
    
    console.print(table)
    print(f"[green]✅[/green] ROC analysis completed successfully!")