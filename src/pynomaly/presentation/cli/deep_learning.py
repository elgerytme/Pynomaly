"""CLI commands for deep learning anomaly detection."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Literal

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Application imports
from pynomaly.application.services.deep_learning_integration_service import (
    DeepLearningIntegrationService,
    DLOptimizationConfig,
)

# Domain imports
from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.config.feature_flags import require_feature

# Infrastructure imports
from pynomaly.infrastructure.data_loaders import CSVLoader, ParquetLoader

console = Console()

# Create Typer app
app = typer.Typer(
    name="deep-learning",
    help="🧠 Deep learning anomaly detection (PyTorch, TensorFlow, JAX)",
    add_completion=True,
    rich_markup_mode="rich",
)


@app.command()
@require_feature("deep_learning")
def train(
    dataset_path: Path = typer.Argument(..., help="Path to the dataset file (CSV or Parquet)", exists=True),
    algorithm: Literal["autoencoder", "vae", "lstm", "gru", "transformer"] = typer.Option(
        "autoencoder",
        "-a", "--algorithm",
        help="Deep learning algorithm"
    ),
    framework: Optional[Literal["pytorch", "tensorflow", "jax"]] = typer.Option(
        None,
        "-f", "--framework",
        help="Deep learning framework (auto-select if not specified)"
    ),
    epochs: int = typer.Option(100, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size for training"),
    learning_rate: float = typer.Option(0.001, "--learning-rate", help="Learning rate"),
    hidden_dims: Optional[List[int]] = typer.Option(None, "--hidden-dims", help="Hidden layer dimensions"),
    latent_dim: int = typer.Option(16, "--latent-dim", help="Latent space dimension"),
    contamination: float = typer.Option(0.1, "--contamination", help="Expected contamination rate"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Enable GPU acceleration"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for results"),
    save_model: Optional[Path] = typer.Option(None, "--save-model", help="Save trained model to file"),
):
    """Train a deep learning anomaly detection model.

    DATASET_PATH: Path to the dataset file (CSV or Parquet)

    Examples:
        pynomaly deep-learning train data.csv
        pynomaly deep-learning train data.csv --algorithm vae --framework pytorch
        pynomaly deep-learning train data.csv --hidden-dims 128 64 32 --epochs 200
    """
    try:
        # Load dataset
        console.print(f"📊 Loading dataset: {dataset_path}")
        dataset = _load_dataset(dataset_path)

        # Create model configuration
        model_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "latent_dim": latent_dim,
            "contamination": contamination,
            "hidden_dims": hidden_dims
        }

        if hidden_dims:
            model_config["hidden_dims"] = hidden_dims

        # Create optimization configuration
        opt_config = DLOptimizationConfig(
            target_framework=framework,
            enable_gpu=gpu,
            optimization_objectives=["accuracy", "speed"],
        )

        # Initialize service
        dl_service = DeepLearningIntegrationService()

        # Display framework information
        _display_framework_info(dl_service, algorithm)

        # Create and train detector
        console.print(f"🤖 Training {algorithm} model...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training deep learning model...", total=None)

            start_time = time.time()

            # Create detector
            detector = asyncio.run(
                dl_service.create_deep_learning_detector(
                    dataset=dataset,
                    algorithm=algorithm,
                    framework=framework,
                    model_config=model_config,
                    optimization_config=opt_config,
                )
            )

            # Train detector (use sync method for CLI)
            detector.fit(dataset.data.values)

            training_time = time.time() - start_time
            progress.update(task, completed=100)

        # Get model information
        model_info = detector.get_model_info()

        # Display results
        _display_training_results(model_info, training_time, algorithm)

        # Run detection on training data
        console.print("🔍 Running anomaly detection...")
        predictions = detector.predict(dataset.data.values)
        scores = detector.decision_function(dataset.data.values)

        n_anomalies = int(predictions.sum())
        anomaly_rate = n_anomalies / len(predictions)

        # Display detection results
        _display_detection_results(n_anomalies, anomaly_rate, scores)

        # Save results if requested
        if output:
            _save_results(output, predictions, scores, model_info, training_time)
            console.print(f"💾 Results saved to: {output}")

        # Save model if requested
        if save_model:
            detector.save_model(save_model)
            console.print(f"💾 Model saved to: {save_model}")

        console.print(
            "✅ Deep learning training completed successfully!", style="green"
        )

    except Exception as e:
        console.print(f"❌ Training failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("deep_learning")
def benchmark(
    dataset_path: Path = typer.Argument(..., help="Path to the dataset file", exists=True),
    algorithm: Literal["autoencoder", "vae", "lstm", "gru", "transformer"] = typer.Option(
        "autoencoder",
        "-a", "--algorithm",
        help="Deep learning algorithm"
    ),
    frameworks: Optional[List[str]] = typer.Option(
        None,
        "-f", "--frameworks",
        help="Frameworks to benchmark (all available if not specified)"
    ),
    epochs: int = typer.Option(50, "--epochs", help="Number of training epochs"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for benchmark results"),
):
    """Benchmark deep learning frameworks on anomaly detection task.

    DATASET_PATH: Path to the dataset file

    Examples:
        pynomaly deep-learning benchmark data.csv
        pynomaly deep-learning benchmark data.csv --frameworks pytorch tensorflow
        pynomaly deep-learning benchmark data.csv --algorithm vae --epochs 100
    """
    try:
        # Load dataset
        console.print(f"📊 Loading dataset: {dataset_path}")
        dataset = _load_dataset(dataset_path)

        # Initialize service
        dl_service = DeepLearningIntegrationService()

        # Use specified frameworks or all available
        frameworks_to_test = frameworks if frameworks else None

        console.print(f"🔬 Benchmarking {algorithm} across frameworks...")

        # Run benchmark
        model_config = {"epochs": epochs}
        results = asyncio.run(
            dl_service.benchmark_frameworks(
                dataset=dataset,
                algorithm=algorithm,
                frameworks=frameworks_to_test,
                model_config=model_config,
            )
        )

        # Display results
        _display_benchmark_results(results)

        # Save results if requested
        if output:
            _save_benchmark_results(results, output)
            console.print(f"💾 Benchmark results saved to: {output}")

        console.print("✅ Framework benchmark completed!", style="green")

    except Exception as e:
        console.print(f"❌ Benchmark failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("deep_learning")
def recommend(
    dataset_path: Path = typer.Argument(..., help="Path to the dataset file", exists=True),
    priority: Literal["speed", "accuracy", "memory", "balanced"] = typer.Option(
        "balanced",
        "--priority",
        help="Performance priority"
    ),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="GPU availability"),
):
    """Get framework and algorithm recommendations for dataset.

    DATASET_PATH: Path to the dataset file

    Examples:
        pynomaly deep-learning recommend data.csv
        pynomaly deep-learning recommend data.csv --priority speed --gpu
    """
    try:
        # Load dataset
        console.print(f"📊 Analyzing dataset: {dataset_path}")
        dataset = _load_dataset(dataset_path)

        # Initialize service
        dl_service = DeepLearningIntegrationService()

        # Get recommendations
        requirements = {"performance_priority": priority, "gpu_available": gpu}

        recommendations = dl_service.get_performance_recommendations(
            dataset, requirements
        )

        # Display recommendations
        _display_recommendations(recommendations, dataset, priority)

        console.print("✅ Recommendations generated!", style="green")

    except Exception as e:
        console.print(f"❌ Recommendation failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("deep_learning")
def frameworks():
    """List available deep learning frameworks and their capabilities.

    Examples:
        pynomaly deep-learning frameworks
    """
    try:
        # Initialize service
        dl_service = DeepLearningIntegrationService()

        # Get framework information
        frameworks_info = dl_service.get_available_frameworks()

        console.print("🧠 Available Deep Learning Frameworks", style="bold")

        if not frameworks_info:
            console.print("❌ No deep learning frameworks available", style="red")
            console.print("\nTo install frameworks:")
            console.print("  PyTorch:    pip install torch torchvision")
            console.print("  TensorFlow: pip install tensorflow")
            console.print("  JAX:        pip install jax jaxlib")
            return

        # Create frameworks table
        table = Table(title="Deep Learning Frameworks")
        table.add_column("Framework", style="cyan")
        table.add_column("Algorithms", style="white")
        table.add_column("Performance", style="green")
        table.add_column("Use Cases", style="yellow")
        table.add_column("Hardware", style="blue")

        for _name, info in frameworks_info.items():
            table.add_row(
                info.name,
                ", ".join(info.algorithms),
                info.performance_tier.replace("_", " ").title(),
                ", ".join(info.use_cases),
                f"CPU: {info.hardware_requirements.get('cpu', 'any')}",
            )

        console.print(table)

        # Display integration status
        status = dl_service.get_integration_status()

        status_text = f"""
        Available Frameworks: {status["available_frameworks"]}
        Total Algorithms: {status["total_algorithms"]}
        Performance History: {status["performance_history_size"]} entries
        """

        console.print(
            Panel(status_text, title="Integration Status", border_style="blue")
        )

        console.print("✅ Framework information displayed!", style="green")

    except Exception as e:
        console.print(f"❌ Failed to list frameworks: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("deep_learning")
def info(
    algorithm: str = typer.Argument(
        ...,
        help="Algorithm to get information about"
    ),
):
    """Get detailed information about a deep learning algorithm.

    ALGORITHM: Algorithm to get information about

    Examples:
        pynomaly deep-learning info autoencoder
        pynomaly deep-learning info vae
    """
    try:
        # Initialize service
        dl_service = DeepLearningIntegrationService()

        # Get algorithm information
        _display_algorithm_info(algorithm, dl_service)

        console.print(f"✅ Information for {algorithm} displayed!", style="green")

    except Exception as e:
        console.print(f"❌ Failed to get algorithm info: {e}", style="red")
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


def _display_framework_info(dl_service: DeepLearningIntegrationService, algorithm: str):
    """Display available frameworks for algorithm."""
    frameworks = dl_service.get_available_frameworks()

    console.print("\n🧠 Available Frameworks for " + algorithm.upper(), style="bold")

    table = Table()
    table.add_column("Framework", style="cyan")
    table.add_column("Supported", style="green")
    table.add_column("Performance", style="yellow")

    for _name, info in frameworks.items():
        supported = "✓" if algorithm in info.algorithms else "✗"
        color = "green" if algorithm in info.algorithms else "red"

        table.add_row(
            info.name,
            f"[{color}]{supported}[/{color}]",
            info.performance_tier.replace("_", " ").title(),
        )

    console.print(table)


def _display_training_results(model_info: dict, training_time: float, algorithm: str):
    """Display training results."""
    console.print("\n🤖 Training Results", style="bold")

    # Summary panel
    summary_text = f"""
    Algorithm: {algorithm.upper()}
    Framework: {model_info.get("backend", model_info.get("algorithm", "unknown")).split("_")[0]}
    Training Time: {training_time:.2f}s
    Total Parameters: {model_info.get("total_parameters", "N/A")}
    Model Trained: {"✓" if model_info.get("is_trained") else "✗"}
    Threshold: {model_info.get("threshold", "N/A")}
    """

    console.print(Panel(summary_text, title="Model Summary", border_style="blue"))


def _display_detection_results(n_anomalies: int, anomaly_rate: float, scores: list):
    """Display detection results."""
    console.print("\n🔍 Detection Results", style="bold")

    results_text = f"""
    Anomalies Detected: {n_anomalies:,}
    Anomaly Rate: {anomaly_rate:.2%}
    Mean Anomaly Score: {np.mean(scores):.4f}
    Max Anomaly Score: {np.max(scores):.4f}
    Min Anomaly Score: {np.min(scores):.4f}
    """

    console.print(Panel(results_text, title="Detection Summary", border_style="green"))


def _display_benchmark_results(results: list):
    """Display benchmark results."""
    console.print("\n🔬 Benchmark Results", style="bold")

    if not results:
        console.print("No benchmark results available", style="yellow")
        return

    # Create benchmark table
    table = Table(title="Framework Performance Comparison")
    table.add_column("Framework", style="cyan")
    table.add_column("Training Time (s)", style="green")
    table.add_column("Inference Time (ms)", style="yellow")
    table.add_column("Parameters", style="blue")
    table.add_column("Rank", style="white")

    # Sort by training time
    sorted_results = sorted(results, key=lambda x: x.training_time)

    for i, result in enumerate(sorted_results, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."

        table.add_row(
            result.framework.title(),
            f"{result.training_time:.2f}",
            f"{result.inference_time * 1000:.2f}",
            f"{result.parameters_count:,}",
            f"{rank_emoji}",
        )

    console.print(table)

    # Winner announcement
    if sorted_results:
        winner = sorted_results[0]
        console.print(
            f"\n🏆 Fastest Framework: {winner.framework.title()} ({winner.training_time:.2f}s)",
            style="bold green",
        )


def _display_recommendations(recommendations: dict, dataset: Dataset, priority: str):
    """Display framework and algorithm recommendations."""
    console.print(f"\n💡 Recommendations for {priority.upper()} Priority", style="bold")

    # Primary recommendation
    primary = recommendations.get("primary_framework", "Not available")
    console.print(
        f"\n🎯 Primary Framework: [bold green]{primary.title() if primary != 'Not available' else primary}[/bold green]"
    )

    # Alternative frameworks
    alternatives = recommendations.get("alternative_frameworks", [])
    if alternatives:
        console.print(
            f"🔄 Alternatives: {', '.join(alt.title() for alt in alternatives)}"
        )

    # Dataset characteristics
    if hasattr(dataset.data, "shape"):
        n_samples, n_features = dataset.data.shape
        console.print(f"\n📊 Dataset: {n_samples:,} samples, {n_features} features")

    # Algorithm suggestions
    algo_suggestions = recommendations.get("algorithm_suggestions", {})
    if algo_suggestions:
        console.print("\n🧠 Algorithm Suggestions:")
        for category, algorithms in algo_suggestions.items():
            console.print(
                f"  {category.replace('_', ' ').title()}: {', '.join(algorithms)}"
            )

    # Configuration tips
    tips = recommendations.get("configuration_tips", [])
    if tips:
        console.print("\n💡 Configuration Tips:")
        for tip in tips:
            console.print(f"  • {tip}")


def _display_algorithm_info(algorithm: str, dl_service: DeepLearningIntegrationService):
    """Display detailed algorithm information."""
    algorithm_info = {
        "autoencoder": {
            "description": "Neural network that learns to compress and reconstruct data",
            "use_cases": [
                "General anomaly detection",
                "Dimensionality reduction",
                "Data compression",
            ],
            "strengths": [
                "Simple architecture",
                "Good baseline",
                "Interpretable reconstruction error",
            ],
            "weaknesses": [
                "May not capture complex patterns",
                "Sensitive to hyperparameters",
            ],
            "parameters": [
                "hidden_dims",
                "latent_dim",
                "epochs",
                "learning_rate",
                "dropout_rate",
            ],
        },
        "vae": {
            "description": "Variational autoencoder with probabilistic latent space",
            "use_cases": [
                "Probabilistic anomaly detection",
                "Data generation",
                "Uncertainty quantification",
            ],
            "strengths": [
                "Principled probabilistic framework",
                "Good for uncertainty",
                "Generative capabilities",
            ],
            "weaknesses": [
                "More complex than standard autoencoder",
                "Requires careful tuning",
            ],
            "parameters": [
                "encoder_dims",
                "decoder_dims",
                "latent_dim",
                "beta",
                "epochs",
            ],
        },
        "lstm": {
            "description": "Long Short-Term Memory network for sequential data",
            "use_cases": [
                "Time series anomaly detection",
                "Sequential pattern detection",
                "Temporal data",
            ],
            "strengths": [
                "Excellent for temporal patterns",
                "Memory of long sequences",
                "Handles variable length",
            ],
            "weaknesses": [
                "Requires sequential data",
                "Computationally expensive",
                "Many hyperparameters",
            ],
            "parameters": [
                "hidden_dim",
                "num_layers",
                "sequence_length",
                "dropout",
                "epochs",
            ],
        },
        "transformer": {
            "description": "Attention-based architecture for sequence modeling",
            "use_cases": [
                "Advanced sequence analysis",
                "Multi-modal data",
                "Attention-based detection",
            ],
            "strengths": [
                "State-of-the-art for sequences",
                "Parallel processing",
                "Attention visualization",
            ],
            "weaknesses": [
                "High computational cost",
                "Requires large datasets",
                "Complex architecture",
            ],
            "parameters": [
                "d_model",
                "num_heads",
                "num_layers",
                "sequence_length",
                "dropout_rate",
            ],
        },
        "gmm": {
            "description": "Gaussian Mixture Model for probabilistic clustering",
            "use_cases": [
                "Density-based detection",
                "Probabilistic clustering",
                "Statistical modeling",
            ],
            "strengths": [
                "Probabilistic interpretation",
                "Fast inference",
                "Well-understood theory",
            ],
            "weaknesses": [
                "Assumes Gaussian distributions",
                "Sensitive to initialization",
                "Limited complexity",
            ],
            "parameters": [
                "n_components",
                "covariance_type",
                "max_iterations",
                "tolerance",
            ],
        },
        "svdd": {
            "description": "Support Vector Data Description for one-class classification",
            "use_cases": [
                "One-class classification",
                "Boundary-based detection",
                "Kernel methods",
            ],
            "strengths": [
                "Solid theoretical foundation",
                "Flexible kernels",
                "Good generalization",
            ],
            "weaknesses": [
                "Kernel selection important",
                "Can be slow",
                "Memory intensive",
            ],
            "parameters": [
                "hidden_dims",
                "nu",
                "learning_rate",
                "weight_decay",
                "epochs",
            ],
        },
    }

    info = algorithm_info.get(algorithm, {})

    console.print(f"\n🧠 {algorithm.upper()} Algorithm", style="bold")

    if not info:
        console.print(
            "No detailed information available for this algorithm", style="yellow"
        )
        return

    # Description
    desc = info.get("description", "No description available")
    console.print(f"\n📝 Description: {desc}")

    # Use cases
    use_cases = info.get("use_cases", [])
    if use_cases:
        console.print("\n🎯 Use Cases:")
        for use_case in use_cases:
            console.print(f"  • {use_case}")

    # Strengths and weaknesses
    strengths = info.get("strengths", [])
    weaknesses = info.get("weaknesses", [])

    if strengths or weaknesses:
        comparison_table = Table(title="Strengths vs Weaknesses")
        comparison_table.add_column("Strengths", style="green")
        comparison_table.add_column("Weaknesses", style="red")

        max_items = max(len(strengths), len(weaknesses))
        for i in range(max_items):
            strength = strengths[i] if i < len(strengths) else ""
            weakness = weaknesses[i] if i < len(weaknesses) else ""
            comparison_table.add_row(strength, weakness)

        console.print(comparison_table)

    # Key parameters
    parameters = info.get("parameters", [])
    if parameters:
        console.print(f"\n⚙️ Key Parameters: {', '.join(parameters)}")

    # Framework availability
    frameworks = dl_service.get_available_frameworks()
    available_frameworks = [
        name
        for name, framework_info in frameworks.items()
        if algorithm in framework_info.algorithms
    ]

    if available_frameworks:
        console.print(
            f"\n🔧 Available in: {', '.join(fw.title() for fw in available_frameworks)}"
        )
    else:
        console.print("\n❌ Not available in any installed framework", style="red")


def _save_results(
    output_path: Path,
    predictions: list,
    scores: list,
    model_info: dict,
    training_time: float,
):
    """Save results to file."""
    results = {
        "model_info": model_info,
        "training_time": training_time,
        "predictions": predictions.tolist(),
        "anomaly_scores": scores.tolist(),
        "summary": {
            "n_anomalies": int(predictions.sum()),
            "anomaly_rate": float(predictions.mean()),
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _save_benchmark_results(results: list, output_path: Path):
    """Save benchmark results to file."""
    benchmark_data = {
        "benchmark_results": [
            {
                "framework": result.framework,
                "algorithm": result.algorithm,
                "training_time": result.training_time,
                "inference_time": result.inference_time,
                "memory_usage": result.memory_usage,
                "parameters_count": result.parameters_count,
                "dataset_size": result.dataset_size,
            }
            for result in results
        ],
        "summary": {
            "fastest_framework": (
                min(results, key=lambda x: x.training_time).framework
                if results
                else None
            ),
            "total_frameworks_tested": len(results),
            "average_training_time": (
                sum(r.training_time for r in results) / len(results) if results else 0
            ),
        },
    }

    with open(output_path, "w") as f:
        json.dump(benchmark_data, f, indent=2, default=str)


# numpy imported at top of file
