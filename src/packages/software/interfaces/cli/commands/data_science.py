"""Data Science CLI Commands for statistical analysis and machine learning operations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()

# Create the data science CLI app
data_science_app = typer.Typer(
    name="data-science",
    help="Data science operations including statistical analysis and machine learning",
    rich_markup_mode="rich"
)


@data_science_app.command("analyze")
def analyze_dataset(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for analysis results"
    ),
    analysis_type: str = typer.Option(
        "comprehensive", "--type", "-t", 
        help="Analysis type: [comprehensive|statistical|exploratory|correlation]"
    ),
    format: str = typer.Option(
        "json", "--format", "-f",
        help="Output format: [json|csv|html|pdf]"
    ),
    include_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Include visualization plots"
    ),
    sample_size: Optional[int] = typer.Option(
        None, "--sample", "-s", help="Sample size for large datasets"
    ),
    confidence_level: float = typer.Option(
        0.95, "--confidence", "-c", help="Confidence level for statistical tests"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Perform comprehensive statistical analysis on a data_collection."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load data_collection
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import data science packages
            from packages.data_science.application.services.statistical_analysis_service import StatisticalAnalysisService
            from packages.data_science.application.services.data_analysis_orchestrator import DataAnalysisOrchestrator
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            if sample_size and len(data_collection) > sample_size:
                data_collection = data_collection.sample(n=sample_size, random_state=42)
                console.print(f"[yellow]Sampled {sample_size} rows from {len(data_collection)} total rows[/yellow]")
            
            progress.update(task, description="Performing statistical analysis...")
            
            # Initialize services
            stats_service = StatisticalAnalysisService()
            orchestrator = DataAnalysisOrchestrator()
            
            # Perform analysis
            if analysis_type == "comprehensive":
                results = orchestrator.perform_comprehensive_analysis(
                    data_collection, confidence_level=confidence_level
                )
            elif analysis_type == "statistical":
                results = stats_service.perform_statistical_analysis(
                    data_collection, confidence_level=confidence_level
                )
            elif analysis_type == "exploratory":
                results = orchestrator.perform_exploratory_analysis(data_collection)
            elif analysis_type == "correlation":
                results = stats_service.calculate_correlation_matrix(data_collection)
            else:
                console.print(f"[red]Unknown analysis type: {analysis_type}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Generating reports...")
            
            # Save results
            output_file = output_dir / f"analysis_results.{format}"
            
            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif format == "html":
                from packages.data_science.application.services.reporting_service import ReportingService
                reporting_service = ReportingService()
                reporting_service.generate_html_report(results, output_file)
            
            # Generate plots if requested
            if include_plots:
                plots_dir = output_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                
                from packages.data_science.application.services.visualization_service import VisualizationService
                viz_service = VisualizationService()
                viz_service.generate_analysis_plots(data_collection, results, plots_dir)
            
            progress.update(task, description="Analysis complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required data science packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during analysis: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    # Display results summary
    console.print("\n[green]✓ Analysis completed successfully![/green]")
    console.print(f"Results saved to: {output_file}")
    
    if include_plots:
        console.print(f"Plots saved to: {output_dir / 'plots'}")


@data_science_app.command("train")
def train_model(
    input_file: Path = typer.Argument(..., help="Training data_collection file"),
    target_column: str = typer.Argument(..., help="Target column name"),
    processor_type: str = typer.Option(
        "auto", "--processor", "-m",
        help="Processor type: [auto|classification|regression|clustering|anomaly]"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for processor artifacts"
    ),
    validation_split: float = typer.Option(
        0.2, "--validation", "-v", help="Validation split ratio"
    ),
    hyperparameter_tuning: bool = typer.Option(
        True, "--tune/--no-tune", help="Enable hyperparameter tuning"
    ),
    cross_validation: int = typer.Option(
        5, "--cv", help="Cross-validation folds"
    ),
    max_trials: int = typer.Option(
        50, "--max-trials", help="Maximum hyperparameter tuning trials"
    ),
    early_stopping: bool = typer.Option(
        True, "--early-stopping/--no-early-stopping", help="Enable early stopping"
    ),
    feature_selection: bool = typer.Option(
        True, "--feature-selection/--no-feature-selection", help="Enable automatic feature selection"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Train machine learning models with automatic optimization."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_dir is None:
        output_dir = input_file.parent / f"{input_file.stem}_processor"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing training...", total=None)
        
        try:
            # Import ML packages
            from packages.data_science.application.services.ml_pipeline_orchestrator import MLPipelineOrchestrator
            from packages.data_science.application.services.model_training_service import ModelTrainingService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            if target_column not in data_collection.columns:
                console.print(f"[red]Error: Target column '{target_column}' not found in data_collection[/red]")
                console.print(f"Available columns: {list(data_collection.columns)}")
                raise typer.Exit(1)
            
            progress.update(task, description="Preparing data...")
            
            # Initialize services
            orchestrator = MLPipelineOrchestrator()
            training_service = ModelTrainingService()
            
            # Configure training parameters
            training_config = {
                "target_column": target_column,
                "validation_split": validation_split,
                "hyperparameter_tuning": hyperparameter_tuning,
                "cross_validation_folds": cross_validation,
                "max_trials": max_trials,
                "early_stopping": early_stopping,
                "feature_selection": feature_selection,
                "output_dir": str(output_dir)
            }
            
            progress.update(task, description="Training models...")
            
            # Train processor
            if processor_type == "auto":
                results = orchestrator.auto_train_best_processor(data_collection, training_config)
            else:
                results = training_service.train_processor(
                    data_collection, processor_type, training_config
                )
            
            progress.update(task, description="Evaluating processor performance...")
            
            # Save processor and results
            processor_file = output_dir / "processor.pkl"
            results_file = output_dir / "training_results.json"
            
            training_service.save_processor(results["processor"], processor_file)
            
            with open(results_file, 'w') as f:
                json.dump(results["measurements"], f, indent=2, default=str)
            
            progress.update(task, description="Training complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required ML packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during training: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    # Display results
    console.print("\n[green]✓ Processor training completed successfully![/green]")
    console.print(f"Processor saved to: {processor_file}")
    console.print(f"Results saved to: {results_file}")
    
    # Display performance measurements
    if "measurements" in results:
        table = Table(title="Processor Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in results["measurements"].items():
            table.add_row(metric, str(value))
        
        console.print(table)


@data_science_app.command("predict")
def predict(
    processor_file: Path = typer.Argument(..., help="Trained processor file"),
    input_file: Path = typer.Argument(..., help="Input data_collection for prediction"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for predictions"
    ),
    include_probabilities: bool = typer.Option(
        False, "--probabilities", help="Include prediction probabilities"
    ),
    batch_size: int = typer.Option(
        1000, "--batch-size", help="Batch size for large datasets"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Make predictions using a trained processor."""
    
    if not processor_file.exists():
        console.print(f"[red]Error: Processor file {processor_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_predictions.csv"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading processor and data...", total=None)
        
        try:
            # Import required packages
            from packages.data_science.application.services.model_inference_service import ModelInferenceService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load processor and data
            inference_service = ModelInferenceService()
            processor = inference_service.load_processor(processor_file)
            
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            progress.update(task, description="Making predictions...")
            
            # Make predictions
            predictions = inference_service.predict_batch(
                processor, data_collection, 
                batch_size=batch_size,
                include_probabilities=include_probabilities
            )
            
            progress.update(task, description="Saving predictions...")
            
            # Save predictions
            predictions.to_csv(output_file, index=False)
            
            progress.update(task, description="Prediction complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during prediction: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Predictions completed successfully![/green]")
    console.print(f"Predictions saved to: {output_file}")
    console.print(f"Total predictions: {len(predictions)}")


@data_science_app.command("feature-engineering")
def feature_engineering(
    input_file: Path = typer.Argument(..., help="Input data_collection file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for engineered features"
    ),
    target_column: Optional[str] = typer.Option(
        None, "--target", "-t", help="Target column for supervised feature selection"
    ),
    auto_features: bool = typer.Option(
        True, "--auto/--manual", help="Automatically generate features"
    ),
    polynomial_features: bool = typer.Option(
        False, "--polynomial", help="Generate polynomial features"
    ),
    interaction_features: bool = typer.Option(
        False, "--interactions", help="Generate interaction features"
    ),
    temporal_features: bool = typer.Option(
        False, "--temporal", help="Generate temporal features for datetime columns"
    ),
    text_features: bool = typer.Option(
        False, "--text", help="Generate text features (TF-IDF, embeddings)"
    ),
    feature_selection: bool = typer.Option(
        True, "--selection/--no-selection", help="Perform feature selection"
    ),
    max_features: Optional[int] = typer.Option(
        None, "--max-features", help="Maximum number of features to select"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Automated feature engineering and selection."""
    
    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_features.csv"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data_collection...", total=None)
        
        try:
            # Import feature engineering packages
            from packages.data_science.application.services.feature_engineering_service import FeatureEngineeringService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load data
            adapter = DataSourceAdapter()
            data_collection = adapter.load_data_collection(str(input_file))
            
            progress.update(task, description="Engineering features...")
            
            # Initialize feature engineering service
            fe_service = FeatureEngineeringService()
            
            # Configure feature engineering
            config = {
                "auto_features": auto_features,
                "polynomial_features": polynomial_features,
                "interaction_features": interaction_features,
                "temporal_features": temporal_features,
                "text_features": text_features,
                "feature_selection": feature_selection,
                "max_features": max_features,
                "target_column": target_column
            }
            
            # Perform feature engineering
            engineered_data_collection = fe_service.engineer_features(data_collection, config)
            
            progress.update(task, description="Saving engineered features...")
            
            # Save results
            engineered_data_collection.to_csv(output_file, index=False)
            
            # Generate feature importance report
            if feature_selection and target_column:
                importance_file = output_file.parent / f"{output_file.stem}_importance.json"
                importance = fe_service.get_feature_importance()
                
                with open(importance_file, 'w') as f:
                    json.dump(importance, f, indent=2, default=str)
            
            progress.update(task, description="Feature engineering complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during feature engineering: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    console.print("\n[green]✓ Feature engineering completed successfully![/green]")
    console.print(f"Features saved to: {output_file}")
    console.print(f"Original features: {len(data_collection.columns)}")
    console.print(f"Engineered features: {len(engineered_data_collection.columns)}")


@data_science_app.command("evaluate")
def evaluate_model(
    processor_file: Path = typer.Argument(..., help="Trained processor file"),
    test_file: Path = typer.Argument(..., help="Test data_collection file"),
    target_column: str = typer.Argument(..., help="Target column name"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for evaluation results"
    ),
    measurements: List[str] = typer.Option(
        ["accuracy", "precision", "recall", "f1"], "--metric", help="Evaluation measurements"
    ),
    confusion_matrix: bool = typer.Option(
        True, "--confusion-matrix/--no-confusion-matrix", help="Generate confusion matrix"
    ),
    feature_importance: bool = typer.Option(
        True, "--feature-importance/--no-feature-importance", help="Calculate feature importance"
    ),
    roc_curve: bool = typer.Option(
        True, "--roc-curve/--no-roc-curve", help="Generate ROC curve"
    ),
    calibration_plot: bool = typer.Option(
        False, "--calibration", help="Generate calibration plot"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """Comprehensive processor evaluation and performance analysis."""
    
    if not processor_file.exists():
        console.print(f"[red]Error: Processor file {processor_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if not test_file.exists():
        console.print(f"[red]Error: Test file {test_file} does not exist[/red]")
        raise typer.Exit(1)
    
    if output_dir is None:
        output_dir = processor_file.parent / "evaluation"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading processor and test data...", total=None)
        
        try:
            # Import evaluation packages
            from packages.data_science.application.services.model_evaluation_service import ModelEvaluationService
            from packages.data_science.application.services.model_inference_service import ModelInferenceService
            from packages.data_profiling.application.services.data_source_adapter import DataSourceAdapter
            
            # Load processor and data
            inference_service = ModelInferenceService()
            processor = inference_service.load_processor(processor_file)
            
            adapter = DataSourceAdapter()
            test_data_collection = adapter.load_data_collection(str(test_file))
            
            if target_column not in test_data_collection.columns:
                console.print(f"[red]Error: Target column '{target_column}' not found in test data_collection[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Evaluating processor performance...")
            
            # Initialize evaluation service
            eval_service = ModelEvaluationService()
            
            # Perform evaluation
            evaluation_results = eval_service.comprehensive_evaluation(
                processor, test_data_collection, target_column,
                measurements=measurements,
                generate_plots=True,
                output_dir=output_dir
            )
            
            progress.update(task, description="Generating evaluation report...")
            
            # Save results
            results_file = output_dir / "evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            
            progress.update(task, description="Evaluation complete!")
            
        except ImportError as e:
            console.print(f"[red]Error: Required packages not available: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error during evaluation: {e}[/red]")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    # Display results
    console.print("\n[green]✓ Processor evaluation completed successfully![/green]")
    console.print(f"Results saved to: {output_dir}")
    
    # Display key measurements
    if "measurements" in evaluation_results:
        table = Table(title="Processor Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in evaluation_results["measurements"].items():
            table.add_row(metric, f"{value:.4f}" if isinstance(value, float) else str(value))
        
        console.print(table)


@data_science_app.command("list-models")
def list_models(
    models_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d", help="Models directory to scan"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: [table|json|csv]"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
) -> None:
    """List available trained models."""
    
    if models_dir is None:
        models_dir = Path.cwd() / "models"
    
    if not models_dir.exists():
        console.print(f"[yellow]Models directory {models_dir} does not exist[/yellow]")
        return
    
    # Scan for processor files
    processor_files = []
    for ext in ["*.pkl", "*.joblib", "*.h5", "*.pt", "*.pth"]:
        processor_files.extend(models_dir.rglob(ext))
    
    if not processor_files:
        console.print(f"[yellow]No processor files found in {models_dir}[/yellow]")
        return
    
    models_info = []
    for processor_file in processor_files:
        stat = processor_file.stat()
        models_info.append({
            "name": processor_file.name,
            "path": str(processor_file),
            "size": f"{stat.st_size / 1024 / 1024:.2f} MB",
            "modified": stat.st_mtime
        })
    
    if format == "table":
        table = Table(title="Available Models")
        table.add_column("Processor Name", style="cyan")
        table.add_column("Path", style="blue")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="yellow")
        
        for processor in models_info:
            from datetime import datetime
            modified_date = datetime.fromtimestamp(processor["modified"]).strftime("%Y-%m-%d %H:%M")
            table.add_row(processor["name"], processor["path"], processor["size"], modified_date)
        
        console.print(table)
    
    elif format == "json":
        console.print(json.dumps(models_info, indent=2, default=str))
    
    elif format == "csv":
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=["name", "path", "size", "modified"])
        writer.writeheader()
        writer.writerows(models_info)
        console.print(output.getvalue())


if __name__ == "__main__":
    data_science_app()