"""CLI commands for intelligent algorithm selection."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly.application.dto.selection_dto import (
    AlgorithmPerformanceDTO,
    OptimizationConstraintsDTO,
)

# Application imports
from pynomaly.application.services.intelligent_selection_service import (
    IntelligentSelectionService,
)

# Domain imports
from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.config.feature_flags import require_feature

# Infrastructure imports
from pynomaly.infrastructure.data_loaders import CSVLoader, ParquetLoader

console = Console()

# Create Typer app
app = typer.Typer(
    name="selection",
    help="🧠 Intelligent algorithm selection with learning capabilities",
    add_completion=True,
    rich_markup_mode="rich",
)


@app.command()
@require_feature("intelligent_selection")
def recommend(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file (CSV or Parquet)", exists=True),
    max_training_time: Optional[float] = typer.Option(None, "--max-training-time", help="Maximum training time in seconds"),
    max_memory: Optional[float] = typer.Option(None, "--max-memory", help="Maximum memory usage in MB"),
    min_accuracy: Optional[float] = typer.Option(None, "--min-accuracy", help="Minimum required accuracy (0-1)"),
    require_interpretability: bool = typer.Option(
        False,
        "--require-interpretability/--no-interpretability",
        help="Require interpretable algorithms"
    ),
    gpu: bool = typer.Option(False, "--gpu/--no-gpu", help="GPU availability"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for recommendations"),
    top_k: int = typer.Option(5, "--top-k", help="Number of top recommendations to show"),
):
    """Recommend optimal algorithms for a dataset.

    DATASET_PATH: Path to dataset file (CSV or Parquet)

    Examples:
        pynomaly selection recommend data.csv
        pynomaly selection recommend data.csv --max-training-time 60 --min-accuracy 0.8
        pynomaly selection recommend data.csv --require-interpretability --top-k 3
    """
    try:
        # Load dataset
        console.print(f"📊 Loading dataset: {dataset_path}")
        dataset = _load_dataset(dataset_path)

        # Set up constraints
        constraints = OptimizationConstraintsDTO(
            max_training_time_seconds=max_training_time,
            max_memory_mb=max_memory,
            min_accuracy=min_accuracy,
            require_interpretability=require_interpretability,
            gpu_available=gpu,
        )

        # Initialize selection service
        selection_service = IntelligentSelectionService(
            enable_meta_learning=True,
            enable_performance_prediction=True,
            enable_historical_learning=True,
        )

        console.print("🧠 Generating intelligent algorithm recommendations...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Analyzing dataset and generating recommendations...", total=None
            )

            start_time = time.time()

            # Get recommendations
            recommendation = asyncio.run(
                selection_service.recommend_algorithm(
                    dataset=dataset, constraints=constraints
                )
            )

            generation_time = time.time() - start_time
            progress.update(task, completed=100)

        # Display results
        _display_recommendations(recommendation, generation_time, top_k)

        # Save recommendations if requested
        if output:
            _save_recommendations(recommendation, output)
            console.print(f"💾 Recommendations saved to: {output}")

        console.print(
            "✅ Algorithm recommendation completed successfully!", style="green"
        )

    except Exception as e:
        console.print(f"❌ Algorithm recommendation failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("intelligent_selection")
def benchmark(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file (CSV or Parquet)", exists=True),
    algorithms: Optional[List[str]] = typer.Option(None, "-a", "--algorithms", help="Specific algorithms to benchmark"),
    cv_folds: int = typer.Option(3, "--cv-folds", help="Cross-validation folds"),
    max_training_time: Optional[float] = typer.Option(None, "--max-training-time", help="Maximum training time per algorithm"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for benchmark results"),
):
    """Benchmark algorithms on a dataset.

    DATASET_PATH: Path to dataset file (CSV or Parquet)

    Examples:
        pynomaly selection benchmark data.csv
        pynomaly selection benchmark data.csv --algorithms isolation_forest local_outlier_factor
        pynomaly selection benchmark data.csv --cv-folds 5 --max-training-time 120
    """
    try:
        # Load dataset
        dataset = _load_dataset(dataset_path)

        # Set up constraints
        constraints = (
            OptimizationConstraintsDTO(max_training_time_seconds=max_training_time)
            if max_training_time
            else None
        )

        # Initialize selection service
        selection_service = IntelligentSelectionService()

        algorithm_list = algorithms

        console.print(f"⚡ Benchmarking algorithms on dataset: {dataset.name}")
        console.print(f"Cross-validation folds: {cv_folds}")
        if algorithm_list:
            console.print(f"Specific algorithms: {', '.join(algorithm_list)}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running algorithm benchmarks...", total=None)

            start_time = time.time()

            # Run benchmarks
            benchmarks = asyncio.run(
                selection_service.benchmark_algorithms(
                    dataset=dataset,
                    algorithms=algorithm_list,
                    cv_folds=cv_folds,
                    constraints=constraints,
                )
            )

            total_time = time.time() - start_time
            progress.update(task, completed=100)

        # Display results
        _display_benchmark_results(benchmarks, total_time)

        # Save results if requested
        if output:
            _save_benchmark_results(benchmarks, output)
            console.print(f"💾 Benchmark results saved to: {output}")

        console.print(
            "✅ Algorithm benchmarking completed successfully!", style="green"
        )

    except Exception as e:
        console.print(f"❌ Algorithm benchmarking failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("intelligent_selection")
def learn(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file used", exists=True),
    algorithm: str = typer.Argument(..., help="Algorithm that was used"),
    performance_score: float = typer.Option(..., "--performance-score", help="Achieved performance score (0-1)"),
    training_time: Optional[float] = typer.Option(None, "--training-time", help="Training time in seconds"),
    memory_usage: Optional[float] = typer.Option(None, "--memory-usage", help="Memory usage in MB"),
    additional_metrics: Optional[str] = typer.Option(None, "--additional-metrics", help="Additional metrics as JSON string"),
):
    """Learn from algorithm selection result.

    DATASET_PATH: Path to dataset file used
    ALGORITHM: Algorithm that was used

    Examples:
        pynomaly selection learn data.csv isolation_forest --performance-score 0.85
        pynomaly selection learn data.csv autoencoder --performance-score 0.92 --training-time 180
    """
    try:
        # Load dataset
        dataset = _load_dataset(dataset_path)

        # Parse additional metrics
        additional_metrics_dict = {}
        if additional_metrics:
            try:
                additional_metrics_dict = json.loads(additional_metrics)
            except json.JSONDecodeError:
                console.print(
                    "⚠️ Invalid JSON for additional metrics, ignoring", style="yellow"
                )

        # Create performance DTO
        performance = AlgorithmPerformanceDTO(
            primary_metric=performance_score,
            training_time_seconds=training_time or 0.0,
            memory_usage_mb=memory_usage or 0.0,
            secondary_metrics=additional_metrics_dict,
        )

        # Initialize selection service
        selection_service = IntelligentSelectionService()

        console.print(f"📚 Learning from result: {algorithm} on {dataset.name}")
        console.print(f"Performance score: {performance_score:.3f}")

        # Learn from result
        asyncio.run(
            selection_service.learn_from_result(
                dataset=dataset,
                algorithm=algorithm,
                performance=performance,
                selection_context={"source": "cli", "timestamp": time.time()},
            )
        )

        console.print("✅ Learning completed successfully!", style="green")
        console.print("💡 This information will improve future recommendations")

    except Exception as e:
        console.print(f"❌ Learning failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("intelligent_selection")
def insights(
    min_samples: int = typer.Option(10, "--min-samples", help="Minimum samples required for reliable insights"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for insights"),
):
    """Get insights from algorithm selection history.

    Examples:
        pynomaly selection insights
        pynomaly selection insights --min-samples 20
    """
    try:
        # Initialize selection service
        selection_service = IntelligentSelectionService()

        console.print("🔍 Analyzing algorithm selection history...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating learning insights...", total=None)

            # Get insights
            insights = asyncio.run(
                selection_service.get_learning_insights(min_samples=min_samples)
            )

            progress.update(task, completed=100)

        # Display insights
        _display_learning_insights(insights)

        # Save insights if requested
        if output:
            _save_learning_insights(insights, output)
            console.print(f"💾 Insights saved to: {output}")

        console.print("✅ Learning insights generated successfully!", style="green")

    except Exception as e:
        console.print(f"❌ Failed to generate insights: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("intelligent_selection")
def predict_performance(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file", exists=True),
    algorithm: str = typer.Argument(..., help="Algorithm to predict performance for"),
    confidence_level: float = typer.Option(0.95, "--confidence-level", help="Confidence level for prediction interval"),
):
    """Predict algorithm performance on a dataset.

    DATASET_PATH: Path to dataset file
    ALGORITHM: Algorithm to predict performance for

    Examples:
        pynomaly selection predict-performance data.csv isolation_forest
        pynomaly selection predict-performance data.csv autoencoder --confidence-level 0.99
    """
    try:
        # Load dataset
        dataset = _load_dataset(dataset_path)

        # Initialize selection service
        selection_service = IntelligentSelectionService()

        console.print(f"🔮 Predicting performance: {algorithm} on {dataset.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Analyzing dataset and predicting performance...", total=None
            )

            # Get recommendation which includes performance prediction
            recommendation = asyncio.run(
                selection_service.recommend_algorithm(dataset=dataset)
            )

            progress.update(task, completed=100)

        # Extract prediction for specific algorithm
        if algorithm in recommendation.confidence_scores:
            predicted_score = recommendation.confidence_scores[algorithm]

            # Display prediction
            console.print(f"\n🎯 Performance Prediction for {algorithm}:", style="bold")
            console.print(f"Predicted Score: {predicted_score:.3f}")

            if (
                recommendation.predicted_performances
                and algorithm in recommendation.predicted_performances
            ):
                predicted_perf = recommendation.predicted_performances[algorithm]
                console.print(f"Detailed Prediction: {predicted_perf:.3f}")

            # Display confidence information
            confidence_text = f"""
            Confidence Level: {confidence_level:.1%}
            Prediction Confidence: {predicted_score:.3f}

            Note: Predictions are based on historical performance
            and dataset similarity analysis.
            """

            console.print(
                Panel(confidence_text, title="Prediction Details", border_style="blue")
            )

        else:
            console.print(
                f"⚠️ No prediction available for algorithm: {algorithm}", style="yellow"
            )
            console.print("Available algorithms:")
            for algo in recommendation.confidence_scores.keys():
                console.print(f"  • {algo}")

        console.print("✅ Performance prediction completed!", style="green")

    except Exception as e:
        console.print(f"❌ Performance prediction failed: {e}", style="red")
        sys.exit(1)


@app.command()
def status():
    """Show intelligent selection service status.

    Examples:
        pynomaly selection status
    """
    try:
        # Initialize selection service
        selection_service = IntelligentSelectionService()

        # Get service information
        service_info = selection_service.get_service_info()

        console.print("🧠 Intelligent Selection Service Status", style="bold")

        # Feature status table
        features_table = Table(title="Feature Status")
        features_table.add_column("Feature", style="cyan")
        features_table.add_column("Enabled", style="white")
        features_table.add_column("Status", style="green")

        features = [
            (
                "Meta-Learning",
                service_info["meta_learning_enabled"],
                "✓ Trained" if service_info["meta_model_trained"] else "⚠️ Not Trained",
            ),
            (
                "Performance Prediction",
                service_info["performance_prediction_enabled"],
                (
                    "✓ Available"
                    if service_info["performance_predictor_trained"]
                    else "⚠️ Not Available"
                ),
            ),
            (
                "Historical Learning",
                service_info["historical_learning_enabled"],
                (
                    f"✓ {service_info['selection_history_size']} samples"
                    if service_info["selection_history_size"] > 0
                    else "⚠️ No History"
                ),
            ),
        ]

        for feature, enabled, status in features:
            enabled_status = "✓" if enabled else "✗"
            enabled_color = "green" if enabled else "red"

            features_table.add_row(
                feature, f"[{enabled_color}]{enabled_status}[/{enabled_color}]", status
            )

        console.print(features_table)

        # Algorithm registry
        console.print(f"\n📚 Available Algorithms: {service_info['algorithm_count']}")
        algorithms = service_info["available_algorithms"]
        for i, algo in enumerate(algorithms, 1):
            console.print(f"  {i:2d}. {algo}")

        # Storage information
        storage_text = f"""
        Selection History: {service_info["selection_history_size"]} entries
        History Path: {service_info["history_path"]}
        Model Path: {service_info["model_path"]}
        """

        console.print(
            Panel(storage_text, title="Storage Information", border_style="blue")
        )

        console.print("✅ Service status displayed successfully!", style="green")

    except Exception as e:
        console.print(f"❌ Failed to get service status: {e}", style="red")
        sys.exit(1)


def _load_dataset(dataset_path: Path) -> Dataset:
    """Load dataset from file."""
    try:
        if dataset_path.suffix.lower() == ".csv":
            loader = CSVLoader()
            data = loader.load(dataset_path)
        elif dataset_path.suffix.lower() in [".parquet", ".pq"]:
            loader = ParquetLoader()
            data = loader.load(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

        return Dataset(
            name=dataset_path.stem,
            data=data,
            feature_names=(
                list(data.columns)
                if hasattr(data, "columns")
                else [f"feature_{i}" for i in range(data.shape[1])]
            ),
        )

    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def _display_recommendations(recommendation, generation_time: float, top_k: int):
    """Display algorithm recommendations."""
    console.print("\n🧠 Algorithm Recommendations", style="bold")

    # Summary panel
    summary_text = f"""
    Generation Time: {generation_time:.2f}s
    Dataset: {recommendation.dataset_characteristics.n_samples} samples, {recommendation.dataset_characteristics.n_features} features
    Recommendations: {len(recommendation.recommended_algorithms)}
    Top Algorithm: {recommendation.recommended_algorithms[0] if recommendation.recommended_algorithms else "None"}
    """

    console.print(Panel(summary_text, title="Summary", border_style="blue"))

    # Recommendations table
    if recommendation.recommended_algorithms:
        rec_table = Table(title=f"Top {top_k} Algorithm Recommendations")
        rec_table.add_column("Rank", style="cyan")
        rec_table.add_column("Algorithm", style="white")
        rec_table.add_column("Confidence", style="green")
        rec_table.add_column("Predicted Performance", style="yellow")

        for i, algo in enumerate(recommendation.recommended_algorithms[:top_k], 1):
            confidence = recommendation.confidence_scores.get(algo, 0.0)
            predicted_perf = ""

            if (
                recommendation.predicted_performances
                and algo in recommendation.predicted_performances
            ):
                predicted_perf = f"{recommendation.predicted_performances[algo]:.3f}"

            rec_table.add_row(
                str(i), algo, f"{confidence:.3f}", predicted_perf or "N/A"
            )

        console.print(rec_table)

    # Dataset characteristics
    chars = recommendation.dataset_characteristics
    console.print("\n📊 Dataset Characteristics:")
    console.print(f"  Samples: {chars.n_samples:,}")
    console.print(f"  Features: {chars.n_features}")
    console.print(f"  Density: {chars.feature_density:.3f}")
    console.print(f"  Outlier Ratio: {chars.outlier_ratio:.3f}")
    console.print(f"  Correlation: {chars.mean_feature_correlation:.3f}")

    # Reasoning
    if recommendation.reasoning:
        console.print("\n💡 Reasoning:")
        for reason in recommendation.reasoning:
            console.print(f"  • {reason}")


def _display_benchmark_results(benchmarks, total_time: float):
    """Display benchmark results."""
    console.print("\n⚡ Algorithm Benchmark Results", style="bold")

    console.print(f"Total benchmarking time: {total_time:.2f}s")
    console.print(f"Algorithms tested: {len(benchmarks)}")

    if not benchmarks:
        console.print("No benchmark results available", style="yellow")
        return

    # Results table
    results_table = Table(title="Benchmark Results")
    results_table.add_column("Rank", style="cyan")
    results_table.add_column("Algorithm", style="white")
    results_table.add_column("Mean Score", style="green")
    results_table.add_column("Std Dev", style="yellow")
    results_table.add_column("Training Time", style="blue")
    results_table.add_column("Memory (MB)", style="magenta")

    for i, benchmark in enumerate(benchmarks, 1):
        results_table.add_row(
            str(i),
            benchmark.algorithm_name,
            f"{benchmark.mean_score:.3f}",
            f"±{benchmark.std_score:.3f}",
            f"{benchmark.training_time_seconds:.1f}s",
            f"{benchmark.memory_usage_mb:.0f}",
        )

    console.print(results_table)

    # Winner announcement
    if benchmarks:
        winner = benchmarks[0]
        console.print(
            f"\n🏆 Best Algorithm: {winner.algorithm_name} (score: {winner.mean_score:.3f})"
        )


def _display_learning_insights(insights):
    """Display learning insights."""
    console.print("\n🔍 Learning Insights", style="bold")

    # Summary
    summary_text = f"""
    Total Selections: {insights.total_selections}
    Meta-Model Accuracy: {insights.meta_model_accuracy:.3f if insights.meta_model_accuracy else 'N/A'}
    Recommendation Confidence: {insights.recommendation_confidence:.3f}
    Analysis Generated: {insights.generated_at.strftime("%Y-%m-%d %H:%M:%S")}
    """

    console.print(Panel(summary_text, title="Summary", border_style="blue"))

    # Algorithm performance stats
    if insights.algorithm_performance_stats:
        console.print("\n📈 Algorithm Performance Statistics:")

        perf_table = Table()
        perf_table.add_column("Algorithm", style="cyan")
        perf_table.add_column("Mean", style="green")
        perf_table.add_column("Std Dev", style="yellow")
        perf_table.add_column("Count", style="white")

        for algo, stats in insights.algorithm_performance_stats.items():
            perf_table.add_row(
                algo,
                f"{stats['mean']:.3f}",
                f"±{stats['std']:.3f}",
                str(stats["count"]),
            )

        console.print(perf_table)

    # Dataset preferences
    if insights.dataset_type_preferences:
        console.print("\n🎯 Dataset Type Preferences:")
        for category, algorithms in insights.dataset_type_preferences.items():
            if algorithms:
                console.print(
                    f"  {category.replace('_', ' ').title()}: {', '.join(algorithms[:3])}"
                )

    # Feature importance
    if insights.feature_importance_insights:
        console.print("\n🔧 Feature Importance for Selection:")
        sorted_features = sorted(
            insights.feature_importance_insights.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        for feature, importance in sorted_features:
            console.print(f"  {feature}: {importance:.3f}")


def _save_recommendations(recommendation, output_path: Path):
    """Save recommendations to file."""
    recommendation_dict = {
        "recommended_algorithms": recommendation.recommended_algorithms,
        "confidence_scores": recommendation.confidence_scores,
        "reasoning": recommendation.reasoning,
        "dataset_characteristics": recommendation.dataset_characteristics.dict(),
        "selection_context": recommendation.selection_context,
        "timestamp": recommendation.timestamp.isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(recommendation_dict, f, indent=2, default=str)


def _save_benchmark_results(benchmarks, output_path: Path):
    """Save benchmark results to file."""
    benchmark_data = [
        {
            "algorithm_name": b.algorithm_name,
            "mean_score": b.mean_score,
            "std_score": b.std_score,
            "cv_scores": b.cv_scores,
            "training_time_seconds": b.training_time_seconds,
            "memory_usage_mb": b.memory_usage_mb,
            "hyperparameters": b.hyperparameters,
            "additional_metrics": b.additional_metrics,
        }
        for b in benchmarks
    ]

    with open(output_path, "w") as f:
        json.dump(benchmark_data, f, indent=2, default=str)


def _save_learning_insights(insights, output_path: Path):
    """Save learning insights to file."""
    with open(output_path, "w") as f:
        json.dump(insights.dict(), f, indent=2, default=str)
