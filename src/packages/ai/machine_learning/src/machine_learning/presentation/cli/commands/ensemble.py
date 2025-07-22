"""CLI commands for advanced ensemble methods and meta-learning."""

"""
TODO: This file needs dependency injection refactoring.
Replace direct monorepo imports with dependency injection.
Use interfaces/shared/base_entity.py for abstractions.
"""



from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Application imports
from interfaces.application.services.advanced_ensemble_service import (
    AdvancedEnsembleService,
    EnsembleConfiguration,
)

# Domain imports
from interfaces.domain.entities import Dataset
from interfaces.infrastructure.config.feature_flags import require_feature

# Infrastructure imports
from monorepo.infrastructure.data_loaders import CSVLoader, ParquetLoader

console = Console()


@click.group()
def ensemble():
    """Advanced ensemble methods and meta-learning commands."""
    pass


@ensemble.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--algorithms", "-a", multiple=True, help="Algorithms to include in ensemble"
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["voting", "weighted_voting", "stacking", "dynamic_selection"]),
    default="voting",
    help="Ensemble combination strategy",
)
@click.option("--max-size", "-n", type=int, default=5, help="Maximum ensemble size")
@click.option(
    "--diversity-threshold",
    "-d",
    type=float,
    default=0.3,
    help="Minimum diversity threshold",
)
@click.option(
    "--diversity-weight",
    "-w",
    type=float,
    default=0.3,
    help="Diversity vs performance trade-off weight",
)
@click.option(
    "--optimize-weights",
    is_flag=True,
    default=True,
    help="Enable ensemble weight optimization",
)
@click.option(
    "--disable-meta-learning",
    is_flag=True,
    help="Disable meta-learning for algorithm selection",
)
@click.option(
    "--cv-folds", type=int, default=3, help="Cross-validation folds for optimization"
)
@click.option(
    "--output", type=click.Path(path_type=Path), help="Output file for ensemble report"
)
@click.option(
    "--save-ensemble",
    type=click.Path(path_type=Path),
    help="Save ensemble detectors to file",
)
@require_feature("ensemble_optimization")
def create(
    dataset_path: Path,
    algorithms: tuple,
    strategy: str,
    max_size: int,
    diversity_threshold: float,
    diversity_weight: float,
    optimize_weights: bool,
    disable_meta_learning: bool,
    cv_folds: int,
    output: Path | None,
    save_ensemble: Path | None,
):
    """Create an intelligent ensemble using meta-learning and optimization.

    DATASET_PATH: Path to the dataset file (CSV or Parquet)

    Examples:
        machine_learning ensemble create data.csv
        machine_learning ensemble create data.csv --algorithms IsolationForest LOF OneClassSVM
        machine_learning ensemble create data.csv --strategy weighted_voting --max-size 7
    """
    try:
        # Load dataset
        console.print(f"📊 Loading dataset: {dataset_path}")
        dataset = _load_dataset(dataset_path)

        # Configure ensemble
        config = EnsembleConfiguration(
            base_algorithms=list(algorithms) if algorithms else [],
            ensemble_strategy=strategy,
            max_ensemble_size=max_size,
            min_diversity_threshold=diversity_threshold,
            weight_optimization=optimize_weights,
            diversity_weighting=diversity_weight,
            cross_validation_folds=cv_folds,
            meta_learning_enabled=not disable_meta_learning,
        )

        # Initialize ensemble service
        ensemble_service = AdvancedEnsembleService(
            enable_meta_learning=not disable_meta_learning,
            diversity_threshold=diversity_threshold,
        )

        # Create ensemble with progress tracking
        console.print("🤖 Creating intelligent ensemble...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Analyzing dataset and creating ensemble...", total=None
            )

            # Run ensemble creation
            start_time = time.time()

            algorithms_list = list(algorithms) if algorithms else None
            ensemble_detectors, ensemble_report = asyncio.run(
                ensemble_service.create_intelligent_ensemble(
                    dataset=dataset, algorithms=algorithms_list, config=config
                )
            )

            creation_time = time.time() - start_time
            progress.update(task, completed=100)

        # Display results
        _display_ensemble_results(ensemble_report, creation_time)

        # Save results if requested
        if output:
            _save_ensemble_report(ensemble_report, output)
            console.print(f"💾 Ensemble report saved to: {output}")

        if save_ensemble:
            _save_ensemble_detectors(ensemble_detectors, save_ensemble)
            console.print(f"💾 Ensemble detectors saved to: {save_ensemble}")

        console.print("✅ Ensemble creation completed successfully!", style="green")

    except Exception as e:
        console.print(f"❌ Ensemble creation failed: {e}", style="red")
        sys.exit(1)


@ensemble.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--algorithms", "-a", multiple=True, help="Algorithms to compare in ensembles"
)
@click.option(
    "--strategies", "-s", multiple=True, help="Ensemble strategies to compare"
)
@click.option(
    "--max-ensembles",
    "-n",
    type=int,
    default=10,
    help="Maximum number of ensemble configurations to test",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for comparison results",
)
@require_feature("ensemble_optimization")
def compare(
    dataset_path: Path,
    algorithms: tuple,
    strategies: tuple,
    max_ensembles: int,
    output: Path | None,
):
    """Compare different ensemble configurations and strategies.

    DATASET_PATH: Path to the dataset file

    Examples:
        machine_learning ensemble compare data.csv
        machine_learning ensemble compare data.csv --strategies voting weighted_voting stacking
    """
    try:
        # Load dataset
        console.print(f"📊 Loading dataset: {dataset_path}")
        dataset = _load_dataset(dataset_path)

        # Default algorithms and strategies if none specified
        if not algorithms:
            algorithms = [
                "IsolationForest",
                "LocalOutlierFactor",
                "OneClassSVM",
                "EllipticEnvelope",
            ]

        if not strategies:
            strategies = ["voting", "weighted_voting", "stacking"]

        # Initialize ensemble service
        ensemble_service = AdvancedEnsembleService()

        console.print(f"🔬 Comparing {len(strategies)} ensemble strategies...")

        comparison_results = {}

        # Test each strategy
        for i, strategy in enumerate(strategies, 1):
            console.print(
                f"\n📈 [{i}/{len(strategies)}] Testing {strategy} strategy..."
            )

            try:
                config = EnsembleConfiguration(
                    base_algorithms=list(algorithms),
                    ensemble_strategy=strategy,
                    max_ensemble_size=min(5, len(algorithms)),
                    meta_learning_enabled=True,
                )

                start_time = time.time()
                ensemble_detectors, ensemble_report = asyncio.run(
                    ensemble_service.create_intelligent_ensemble(
                        dataset=dataset, algorithms=list(algorithms), config=config
                    )
                )
                creation_time = time.time() - start_time

                # Extract key metrics
                performance_summary = ensemble_report.get("performance_summary", {})
                diversity_analysis = ensemble_report.get("diversity_analysis", {})

                comparison_results[strategy] = {
                    "creation_time": creation_time,
                    "n_detectors": len(ensemble_detectors),
                    "estimated_performance": performance_summary.get(
                        "estimated_ensemble_performance", 0.0
                    ),
                    "diversity_score": diversity_analysis.get("overall_diversity", 0.0),
                    "performance_improvement": performance_summary.get(
                        "performance_improvement", 0.0
                    ),
                    "confidence": performance_summary.get("confidence_score", 0.0),
                    "report": ensemble_report,
                }

                console.print(
                    f"  ✅ {strategy}: Performance = {performance_summary.get('estimated_ensemble_performance', 0):.3f}"
                )

            except Exception as e:
                console.print(f"  ❌ {strategy} failed: {e}", style="red")
                comparison_results[strategy] = {"error": str(e)}

        # Display comparison results
        _display_ensemble_comparison(comparison_results)

        # Save comparison results
        if output:
            _save_comparison_results(comparison_results, output)
            console.print(f"💾 Comparison results saved to: {output}")

        console.print("\n✅ Ensemble comparison completed!", style="green")

    except Exception as e:
        console.print(f"❌ Ensemble comparison failed: {e}", style="red")
        sys.exit(1)


@ensemble.command()
@click.option(
    "--meta-knowledge-path",
    type=click.Path(path_type=Path),
    default=Path("./meta_knowledge"),
    help="Meta-knowledge storage path",
)
@require_feature("meta_learning")
def insights():
    """Analyze meta-learning insights and knowledge base.

    Examples:
        machine_learning ensemble insights
        machine_learning ensemble insights --meta-knowledge-path /path/to/knowledge
    """
    try:
        # Initialize ensemble service
        ensemble_service = AdvancedEnsembleService(
            meta_knowledge_path=meta_knowledge_path
        )

        console.print("🧠 Analyzing meta-learning insights...")

        # Generate insights
        insights = ensemble_service._generate_meta_learning_insights()

        # Display insights
        _display_meta_learning_insights(insights)

        console.print("✅ Meta-learning insights analysis completed!", style="green")

    except Exception as e:
        console.print(f"❌ Meta-learning insights analysis failed: {e}", style="red")
        sys.exit(1)


@ensemble.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.argument("algorithms", nargs=-1)
@click.option(
    "--meta-knowledge-path",
    type=click.Path(path_type=Path),
    default=Path("./meta_knowledge"),
    help="Meta-knowledge storage path",
)
@require_feature("meta_learning")
def predict_performance(
    dataset_path: Path, algorithms: tuple, meta_knowledge_path: Path
):
    """Predict ensemble performance using meta-learning.

    DATASET_PATH: Path to the dataset file
    ALGORITHMS: Algorithms to include in predicted ensemble

    Examples:
        machine_learning ensemble predict-performance data.csv IsolationForest LOF OneClassSVM
    """
    try:
        # Load dataset
        dataset = _load_dataset(dataset_path)

        # Initialize ensemble service
        ensemble_service = AdvancedEnsembleService(
            meta_knowledge_path=meta_knowledge_path
        )

        console.print("🔮 Predicting ensemble performance...")

        # Analyze dataset characteristics
        dataset_chars = ensemble_service._analyze_dataset_for_ensemble(dataset)

        # Predict performance
        prediction = asyncio.run(
            ensemble_service.predict_ensemble_performance(
                dataset_chars, list(algorithms)
            )
        )

        # Display prediction
        _display_performance_prediction(dataset_chars, list(algorithms), prediction)

        console.print("✅ Performance prediction completed!", style="green")

    except Exception as e:
        console.print(f"❌ Performance prediction failed: {e}", style="red")
        sys.exit(1)


@ensemble.command()
@click.argument("algorithms", nargs=-1)
@require_feature("ensemble_optimization")
def diversity(algorithms: tuple):
    """Analyze diversity potential between algorithms.

    ALGORITHMS: Algorithms to analyze for diversity

    Examples:
        machine_learning ensemble diversity IsolationForest LOF OneClassSVM
    """
    try:
        if len(algorithms) < 2:
            console.print(
                "❌ Need at least 2 algorithms for diversity analysis", style="red"
            )
            sys.exit(1)

        # Initialize ensemble service
        ensemble_service = AdvancedEnsembleService()

        console.print(f"📊 Analyzing diversity between {len(algorithms)} algorithms...")

        # Display algorithm compatibility matrix
        _display_algorithm_compatibility(
            algorithms, ensemble_service.algorithm_compatibility
        )

        console.print("✅ Diversity analysis completed!", style="green")

    except Exception as e:
        console.print(f"❌ Diversity analysis failed: {e}", style="red")
        sys.exit(1)


@ensemble.command()
@click.option(
    "--meta-knowledge-path",
    type=click.Path(path_type=Path),
    default=Path("./meta_knowledge"),
    help="Meta-knowledge storage path",
)
@require_feature("meta_learning")
def strategies():
    """List available ensemble strategies and their characteristics.

    Examples:
        machine_learning ensemble strategies
    """
    try:
        # Initialize ensemble service
        ensemble_service = AdvancedEnsembleService()

        console.print("📋 Available Ensemble Strategies:")

        # Create strategies table
        table = Table(title="Ensemble Strategies")
        table.add_column("Strategy", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Training Required", style="yellow")
        table.add_column("Supports Weights", style="green")
        table.add_column("Complexity", style="blue")
        table.add_column("Interpretability", style="magenta")

        for name, strategy in ensemble_service.ensemble_strategies.items():
            table.add_row(
                name,
                strategy.description,
                "✓" if strategy.requires_training else "✗",
                "✓" if strategy.supports_weights else "✗",
                strategy.complexity,
                f"{strategy.interpretability:.2f}",
            )

        console.print(table)

        console.print("✅ Strategy listing completed!", style="green")

    except Exception as e:
        console.print(f"❌ Strategy listing failed: {e}", style="red")
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
            features=[f"feature_{i}" for i in range(data.shape[1])],
        )

    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def _display_ensemble_results(report: dict, creation_time: float):
    """Display ensemble creation results."""
    console.print("\n🤖 Ensemble Creation Results", style="bold")

    # Summary panel
    summary = report.get("ensemble_summary", {})
    performance_summary = report.get("performance_summary", {})
    diversity_analysis = report.get("diversity_analysis", {})

    summary_text = f"""
    Detectors: {summary.get("n_detectors", 0)}
    Strategy: {summary.get("strategy", "unknown")}
    Creation Time: {creation_time:.2f}s
    Estimated Performance: {performance_summary.get("estimated_ensemble_performance", 0):.4f}
    Diversity Score: {diversity_analysis.get("overall_diversity", 0):.4f}
    """

    console.print(Panel(summary_text, title="Summary", border_style="blue"))

    # Ensemble weights
    ensemble_weights = report.get("ensemble_weights", {})
    if ensemble_weights:
        table = Table(title="Ensemble Weights")
        table.add_column("Detector", style="cyan")
        table.add_column("Weight", style="green")

        for detector, weight in ensemble_weights.items():
            table.add_row(detector, f"{weight:.4f}")

        console.print(table)

    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        console.print("\n💡 Recommendations:", style="bold")
        for rec in recommendations:
            console.print(f"  • {rec}")


def _display_ensemble_comparison(results: dict):
    """Display ensemble comparison results."""
    console.print("\n🔬 Ensemble Strategy Comparison", style="bold")

    # Create comparison table
    table = Table(title="Strategy Performance Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Performance", style="green")
    table.add_column("Diversity", style="yellow")
    table.add_column("Improvement", style="blue")
    table.add_column("Time (s)", style="white")
    table.add_column("Status", style="white")

    # Sort strategies by performance
    strategy_scores = []
    for strategy, result in results.items():
        if "error" in result:
            strategy_scores.append((strategy, 0, "Failed"))
        else:
            performance = result.get("estimated_performance", 0)
            strategy_scores.append((strategy, performance, "Success"))

    strategy_scores.sort(key=lambda x: x[1], reverse=True)

    # Add rows to table
    for i, (strategy, performance, status) in enumerate(strategy_scores):
        if status == "Success":
            result = results[strategy]
            rank_emoji = (
                "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            )
            table.add_row(
                f"{rank_emoji} {strategy}",
                f"{result.get('estimated_performance', 0):.4f}",
                f"{result.get('diversity_score', 0):.4f}",
                f"{result.get('performance_improvement', 0):.4f}",
                f"{result.get('creation_time', 0):.2f}",
                status,
            )
        else:
            table.add_row(strategy, "N/A", "N/A", "N/A", "N/A", "❌ Failed")

    console.print(table)

    # Winner announcement
    if strategy_scores and strategy_scores[0][2] == "Success":
        winner = strategy_scores[0][0]
        winner_performance = strategy_scores[0][1]
        console.print(
            f"\n🏆 Best Strategy: {winner} (Performance: {winner_performance:.4f})",
            style="bold green",
        )


def _display_meta_learning_insights(insights: dict):
    """Display meta-learning insights."""
    console.print("\n🧠 Meta-Learning Insights", style="bold")

    if "message" in insights:
        console.print(f"ℹ️ {insights['message']}", style="yellow")
        return

    # Knowledge base summary
    knowledge_size = insights.get("knowledge_base_size", 0)
    learning_confidence = insights.get("learning_confidence", 0)

    console.print(f"📚 Knowledge Base Size: {knowledge_size}")
    console.print(f"🎯 Learning Confidence: {learning_confidence:.3f}")

    # Popular algorithms
    popular_algorithms = insights.get("most_popular_algorithms", [])
    if popular_algorithms:
        console.print("\n🔥 Most Popular Algorithms:")
        table = Table()
        table.add_column("Algorithm", style="cyan")
        table.add_column("Usage Count", style="green")

        for algorithm, count in popular_algorithms:
            table.add_row(algorithm, str(count))

        console.print(table)

    # Performance by algorithm
    avg_performance = insights.get("average_performance_by_algorithm", {})
    if avg_performance:
        console.print("\n📈 Average Performance by Algorithm:")
        perf_table = Table()
        perf_table.add_column("Algorithm", style="cyan")
        perf_table.add_column("Avg Performance", style="green")

        sorted_performance = sorted(
            avg_performance.items(), key=lambda x: x[1], reverse=True
        )
        for algorithm, performance in sorted_performance:
            perf_table.add_row(algorithm, f"{performance:.4f}")

        console.print(perf_table)


def _display_performance_prediction(
    dataset_chars: dict, algorithms: list, prediction: dict
):
    """Display performance prediction results."""
    console.print("\n🔮 Performance Prediction", style="bold")

    # Dataset characteristics summary
    chars_text = f"""
    Samples: {dataset_chars.get("n_samples", 0):,}
    Features: {dataset_chars.get("n_features", 0)}
    Complexity: {dataset_chars.get("data_complexity", 0):.3f}
    Noise Level: {dataset_chars.get("noise_level", 0):.3f}
    """

    console.print(
        Panel(chars_text, title="Dataset Characteristics", border_style="blue")
    )

    # Prediction results
    pred_text = f"""
    Predicted Performance: {prediction.get("predicted_performance", 0):.4f}
    Confidence: {prediction.get("confidence", 0):.3f}
    Similar Cases: {prediction.get("similar_cases_found", 0)}
    Recommendation Strength: {prediction.get("recommendation_strength", "unknown")}
    """

    console.print(Panel(pred_text, title="Prediction Results", border_style="green"))

    # Proposed algorithms
    console.print(f"\n📋 Proposed Algorithms: {', '.join(algorithms)}")


def _display_algorithm_compatibility(algorithms: tuple, compatibility_matrix: dict):
    """Display algorithm compatibility analysis."""
    console.print("\n📊 Algorithm Compatibility Matrix", style="bold")

    # Create compatibility table
    table = Table()
    table.add_column("Algorithm", style="cyan")

    for alg in algorithms:
        table.add_column(alg, style="white")

    for alg1 in algorithms:
        row = [alg1]
        for alg2 in algorithms:
            if alg1 == alg2:
                row.append("1.00")
            else:
                compatibility = compatibility_matrix.get(alg1, {}).get(alg2, 0.5)
                color = (
                    "green"
                    if compatibility < 0.6
                    else "yellow"
                    if compatibility < 0.8
                    else "red"
                )
                row.append(f"[{color}]{compatibility:.2f}[/{color}]")
        table.add_row(*row)

    console.print(table)

    console.print("\n💡 Lower values indicate higher diversity potential", style="dim")


def _save_ensemble_report(report: dict, output_path: Path):
    """Save ensemble report to file."""
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def _save_ensemble_detectors(detectors: list, output_path: Path):
    """Save ensemble detectors to file."""
    # Simplified serialization (in real implementation, would use proper model serialization)
    detector_info = [
        {
            "algorithm": getattr(detector, "algorithm_name", "unknown"),
            "parameters": getattr(detector, "algorithm_params", {}),
            "trained": True,
        }
        for detector in detectors
    ]

    with open(output_path, "w") as f:
        json.dump(detector_info, f, indent=2, default=str)


def _save_comparison_results(results: dict, output_path: Path):
    """Save comparison results to file."""
    # Clean results for JSON serialization
    clean_results = {}
    for strategy, result in results.items():
        if "report" in result:
            # Remove the full report to avoid circular references
            clean_result = {k: v for k, v in result.items() if k != "report"}
            clean_results[strategy] = clean_result
        else:
            clean_results[strategy] = result

    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
