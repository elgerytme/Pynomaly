"""Advanced AutoML CLI commands for intelligent hyperparameter optimization."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

# Application imports
from pynomaly.application.services.advanced_automl_service import AdvancedAutoMLService
from pynomaly.application.dto.optimization_dto import (
    OptimizationObjectiveDTO, ResourceConstraintsDTO, OptimizationConfigDTO,
    create_default_objectives, create_default_constraints
)

# Domain imports
from pynomaly.domain.entities import Dataset

# Infrastructure imports
from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
from pynomaly.infrastructure.data_loaders.parquet_loader import ParquetLoader
from pynomaly.infrastructure.config.feature_flags import require_feature
from pynomaly.presentation.cli.container import get_cli_container

console = Console()
app = typer.Typer()


def automl_help():
    """Advanced AutoML and hyperparameter optimization commands."""
    pass


@app.command()
@require_feature("advanced_automl")
def optimize(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file"),
    algorithm_name: str = typer.Argument(..., help="Algorithm name to optimize"),
    objectives: Optional[List[str]] = typer.Option(None, "-o", "--objectives", 
                                                   help="Optimization objectives (accuracy, speed, interpretability, memory_efficiency)"),
    max_time: int = typer.Option(3600, "-t", "--max-time", help="Maximum optimization time in seconds"),
    max_trials: int = typer.Option(100, "-n", "--max-trials", help="Maximum number of optimization trials"),
    max_memory: int = typer.Option(4096, "-m", "--max-memory", help="Maximum memory usage in MB"),
    parallel_jobs: int = typer.Option(1, "-j", "--parallel-jobs", help="Number of parallel optimization jobs"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for optimization results"),
    disable_learning: bool = typer.Option(False, "--disable-learning", help="Disable learning from optimization history"),
    prefer_speed: bool = typer.Option(False, "--prefer-speed", help="Prefer speed over accuracy in optimization"),
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU acceleration if available")
):
    """Optimize algorithm hyperparameters using advanced AutoML.
    
    DATASET_PATH: Path to the dataset file (CSV or Parquet)
    ALGORITHM_NAME: Algorithm to optimize (IsolationForest, LocalOutlierFactor, OneClassSVM)
    
    Examples:
        pynomaly automl optimize data.csv IsolationForest
        pynomaly automl optimize data.csv LOF --max-time 1800 --max-trials 50
        pynomaly automl optimize data.parquet OneClassSVM --objectives accuracy speed
    """
    try:
        # Load dataset
        console.print(f"📊 Loading dataset: {dataset_path}")
        dataset = _load_dataset(dataset_path)
        
        # Configure objectives
        if objectives:
            objective_configs = []
            weights = {"accuracy": 0.4, "speed": 0.3, "interpretability": 0.2, "memory_efficiency": 0.1}
            
            for obj_name in objectives:
                if obj_name not in weights:
                    console.print(f"❌ Unknown objective: {obj_name}", style="red")
                    sys.exit(1)
                
                objective_configs.append(OptimizationObjectiveDTO(
                    name=obj_name,
                    weight=weights[obj_name],
                    direction="maximize",
                    description=f"Optimize {obj_name}"
                ))
        else:
            objective_configs = create_default_objectives()
        
        # Configure resource constraints
        constraints = ResourceConstraintsDTO(
            max_time_seconds=max_time,
            max_trials=max_trials,
            max_memory_mb=max_memory,
            max_cpu_cores=parallel_jobs,
            gpu_available=gpu,
            prefer_speed=prefer_speed
        )
        
        # Initialize AutoML service
        automl_service = AdvancedAutoMLService(
            enable_distributed=parallel_jobs > 1,
            n_parallel_jobs=parallel_jobs
        )
        
        # Run optimization with progress tracking
        console.print("🚀 Starting advanced AutoML optimization...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"Optimizing {algorithm_name}...", 
                total=max_trials
            )
            
            # Run optimization
            start_time = time.time()
            detector, report = asyncio.run(
                automl_service.optimize_detector_advanced(
                    dataset=dataset,
                    algorithm_name=algorithm_name,
                    objectives=objective_configs,
                    constraints=constraints,
                    enable_learning=not disable_learning
                )
            )
            optimization_time = time.time() - start_time
            
            progress.update(task, completed=max_trials)
        
        # Display results
        _display_optimization_results(report, optimization_time)
        
        # Save results if requested
        if output:
            _save_optimization_results(report, output)
            console.print(f"💾 Results saved to: {output}")
        
        console.print("✅ AutoML optimization completed successfully!", style="green")
        
    except Exception as e:
        console.print(f"❌ AutoML optimization failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("advanced_automl")
def compare(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file"),
    algorithms: Optional[List[str]] = typer.Option(None, "-a", "--algorithms",
                                                  help="Algorithms to compare (default: all available)"),
    max_time_per_algorithm: int = typer.Option(1800, "-t", "--max-time-per-algorithm",
                                              help="Maximum time per algorithm in seconds"),
    max_trials_per_algorithm: int = typer.Option(50, "-n", "--max-trials-per-algorithm",
                                                help="Maximum trials per algorithm"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for comparison results")
):
    """Compare multiple algorithms using AutoML optimization.
    
    DATASET_PATH: Path to the dataset file
    
    Examples:
        pynomaly automl compare data.csv
        pynomaly automl compare data.csv --algorithms IsolationForest LOF
    """
    try:
        # Load dataset
        console.print(f"📊 Loading dataset: {dataset_path}")
        dataset = _load_dataset(dataset_path)
        
        # Default algorithms if none specified
        if not algorithms:
            algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        
        # Initialize AutoML service
        automl_service = AdvancedAutoMLService()
        
        # Configure constraints for comparison
        constraints = ResourceConstraintsDTO(
            max_time_seconds=max_time_per_algorithm,
            max_trials=max_trials_per_algorithm,
            max_memory_mb=4096,
            max_cpu_cores=2
        )
        
        results = {}
        
        console.print(f"🔬 Comparing {len(algorithms)} algorithms...")
        
        # Optimize each algorithm
        for i, algorithm in enumerate(algorithms, 1):
            console.print(f"\n📈 [{i}/{len(algorithms)}] Optimizing {algorithm}...")
            
            try:
                detector, report = asyncio.run(
                    automl_service.optimize_detector_advanced(
                        dataset=dataset,
                        algorithm_name=algorithm,
                        objectives=create_default_objectives(),
                        constraints=constraints,
                        enable_learning=True
                    )
                )
                
                results[algorithm] = report
                
                # Show brief results
                best_metrics = report.get("best_metrics", {})
                console.print(f"  ✅ {algorithm}: Best accuracy = {best_metrics.get('objective_0', 0):.3f}")
                
            except Exception as e:
                console.print(f"  ❌ {algorithm} failed: {e}", style="red")
                results[algorithm] = {"error": str(e)}
        
        # Display comparison results
        _display_algorithm_comparison(results)
        
        # Save comparison results
        if output:
            _save_comparison_results(results, output)
            console.print(f"💾 Comparison results saved to: {output}")
        
        console.print("\n✅ Algorithm comparison completed!", style="green")
        
    except Exception as e:
        console.print(f"❌ Algorithm comparison failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("advanced_automl")
def insights(
    storage_path: Path = typer.Option(Path("./automl_storage"), "--storage-path", 
                                     help="AutoML storage path")
):
    """Analyze optimization history and learning insights.
    
    Examples:
        pynomaly automl insights
        pynomaly automl insights --storage-path /path/to/storage
    """
    try:
        # Initialize AutoML service
        automl_service = AdvancedAutoMLService(optimization_storage_path=storage_path)
        
        # Analyze trends
        console.print("🧠 Analyzing optimization trends and learning insights...")
        
        trends_analysis = asyncio.run(automl_service.analyze_optimization_trends())
        
        if "message" in trends_analysis:
            console.print(f"ℹ️ {trends_analysis['message']}", style="yellow")
            return
        
        # Display insights
        _display_learning_insights(trends_analysis)
        
        console.print("✅ Learning insights analysis completed!", style="green")
        
    except Exception as e:
        console.print(f"❌ Learning insights analysis failed: {e}", style="red")
        sys.exit(1)


@app.command("predict-performance")
@require_feature("advanced_automl")
def predict_performance(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file"),
    algorithm_name: str = typer.Argument(..., help="Algorithm to predict performance for"),
    storage_path: Path = typer.Option(Path("./automl_storage"), "--storage-path",
                                     help="AutoML storage path")
):
    """Predict algorithm performance based on dataset characteristics.
    
    DATASET_PATH: Path to the dataset file
    ALGORITHM_NAME: Algorithm to predict performance for
    
    Examples:
        pynomaly automl predict-performance data.csv IsolationForest
    """
    try:
        # Load dataset
        dataset = _load_dataset(dataset_path)
        
        # Initialize AutoML service
        automl_service = AdvancedAutoMLService(optimization_storage_path=storage_path)
        
        console.print(f"🔮 Predicting performance for {algorithm_name}...")
        
        # Analyze dataset characteristics
        dataset_chars = automl_service._analyze_dataset_characteristics(dataset)
        
        # Predict optimal parameters
        predicted_params = automl_service._predict_optimal_parameters(
            dataset_chars, algorithm_name
        )
        
        # Display predictions
        _display_performance_prediction(dataset_chars, algorithm_name, predicted_params)
        
        console.print("✅ Performance prediction completed!", style="green")
        
    except Exception as e:
        console.print(f"❌ Performance prediction failed: {e}", style="red")
        sys.exit(1)


def _load_dataset(dataset_path: Path) -> Dataset:
    """Load dataset from file."""
    try:
        if dataset_path.suffix.lower() == '.csv':
            loader = CSVLoader()
            data = loader.load(dataset_path)
        elif dataset_path.suffix.lower() in ['.parquet', '.pq']:
            loader = ParquetLoader()
            data = loader.load(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
        
        return Dataset(
            name=dataset_path.stem,
            data=data,
            features=[f"feature_{i}" for i in range(data.shape[1])]
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def _display_optimization_results(report: dict, optimization_time: float):
    """Display optimization results in a formatted table."""
    console.print("\n📊 Optimization Results", style="bold")
    
    # Summary panel
    summary = report.get("optimization_summary", {})
    best_metrics = report.get("best_metrics", {})
    
    summary_text = f"""
    Total Trials: {summary.get('total_trials', 0)}
    Successful Trials: {summary.get('successful_trials', 0)}
    Optimization Time: {optimization_time:.2f}s
    Best Accuracy: {best_metrics.get('objective_0', 0):.4f}
    """
    
    console.print(Panel(summary_text, title="Summary", border_style="blue"))
    
    # Best parameters table
    best_params = report.get("best_parameters", {})
    if best_params:
        table = Table(title="Best Parameters")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        for param, value in best_params.items():
            table.add_row(param, str(value))
        
        console.print(table)
    
    # Pareto optimal solutions
    pareto_solutions = report.get("pareto_optimal_solutions", [])
    if pareto_solutions:
        console.print(f"\n🏆 Found {len(pareto_solutions)} Pareto optimal solutions")


def _display_algorithm_comparison(results: dict):
    """Display algorithm comparison results."""
    console.print("\n🔬 Algorithm Comparison Results", style="bold")
    
    # Create comparison table
    table = Table(title="Algorithm Performance Comparison")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Speed Score", style="yellow")
    table.add_column("Status", style="white")
    
    # Sort algorithms by accuracy
    algorithm_scores = []
    for algorithm, report in results.items():
        if "error" in report:
            algorithm_scores.append((algorithm, 0, "Failed"))
        else:
            best_metrics = report.get("best_metrics", {})
            accuracy = best_metrics.get("objective_0", 0)
            speed = best_metrics.get("objective_1", 0)
            algorithm_scores.append((algorithm, accuracy, speed, "Success"))
    
    algorithm_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Add rows to table
    for i, (algorithm, accuracy, speed, status) in enumerate(algorithm_scores):
        if status == "Success":
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            table.add_row(
                f"{rank_emoji} {algorithm}",
                f"{accuracy:.4f}",
                f"{speed:.4f}",
                status
            )
        else:
            table.add_row(algorithm, "N/A", "N/A", "❌ Failed")
    
    console.print(table)
    
    # Winner announcement
    if algorithm_scores and algorithm_scores[0][3] == "Success":
        winner = algorithm_scores[0][0]
        winner_accuracy = algorithm_scores[0][1]
        console.print(f"\n🏆 Winner: {winner} (Accuracy: {winner_accuracy:.4f})", style="bold green")


def _display_learning_insights(trends_analysis: dict):
    """Display learning insights and trends."""
    console.print("\n🧠 Learning Insights", style="bold")
    
    algorithm_trends = trends_analysis.get("algorithm_trends", {})
    total_optimizations = trends_analysis.get("total_optimizations", 0)
    insights = trends_analysis.get("learning_insights", [])
    
    # Summary
    console.print(f"📈 Total optimizations performed: {total_optimizations}")
    
    # Algorithm trends table
    if algorithm_trends:
        table = Table(title="Algorithm Learning Trends")
        table.add_column("Algorithm", style="cyan")
        table.add_column("Optimizations", style="white")
        table.add_column("Avg Performance", style="green")
        table.add_column("Trend", style="yellow")
        table.add_column("Learning Rate", style="blue")
        
        for algorithm, data in algorithm_trends.items():
            trend = data.get("performance_improvement", "unknown")
            trend_emoji = {
                "improving": "📈",
                "declining": "📉", 
                "stable": "➡️"
            }.get(trend, "❓")
            
            table.add_row(
                algorithm,
                str(data.get("total_optimizations", 0)),
                f"{data.get('average_performance', 0):.4f}",
                f"{trend_emoji} {trend}",
                f"{data.get('learning_rate', 0):.4f}"
            )
        
        console.print(table)
    
    # Learning insights
    if insights:
        console.print("\n💡 Key Insights:", style="bold")
        for insight in insights:
            console.print(f"  • {insight}")


def _display_performance_prediction(dataset_chars: dict, algorithm_name: str, predicted_params: dict):
    """Display performance prediction results."""
    console.print(f"\n🔮 Performance Prediction for {algorithm_name}", style="bold")
    
    # Dataset characteristics
    chars_text = f"""
    Samples: {dataset_chars.get('n_samples', 0):,}
    Features: {dataset_chars.get('n_features', 0)}
    Size Category: {dataset_chars.get('size_category', 'unknown')}
    Sparsity: {dataset_chars.get('sparsity', 0):.3f}
    """
    
    console.print(Panel(chars_text, title="Dataset Characteristics", border_style="blue"))
    
    # Predicted parameters
    if predicted_params:
        table = Table(title="Predicted Optimal Parameters")
        table.add_column("Parameter", style="cyan")
        table.add_column("Predicted Value", style="green")
        
        for param, value in predicted_params.items():
            table.add_row(param, f"{value:.4f}" if isinstance(value, float) else str(value))
        
        console.print(table)
    else:
        console.print("ℹ️ No historical data available for prediction", style="yellow")


def _save_optimization_results(report: dict, output_path: Path):
    """Save optimization results to file."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)


def _save_comparison_results(results: dict, output_path: Path):
    """Save comparison results to file.""" 
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)