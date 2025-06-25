"""CLI commands for configuration recommendations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly.application.dto.configuration_dto import (
    ConfigurationLevel,
    DatasetCharacteristicsDTO,
)
from pynomaly.infrastructure.config.feature_flags import require_feature
from pynomaly.presentation.cli.container import get_cli_container

# Create Typer app
app = typer.Typer(
    name="recommend",
    help="🧠 Intelligent configuration recommendations based on dataset characteristics",
    rich_markup_mode="rich",
)

console = Console()


@app.command("dataset")
@require_feature("advanced_automl")
def recommend_for_dataset(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file"),
    samples: int | None = typer.Option(
        None, "--samples", help="Number of samples (auto-detected if not provided)"
    ),
    features: int | None = typer.Option(
        None, "--features", help="Number of features (auto-detected if not provided)"
    ),
    min_accuracy: float | None = typer.Option(
        0.7, "--min-accuracy", help="Minimum required accuracy"
    ),
    use_case: str | None = typer.Option(
        None, "--use-case", help="Specific use case (e.g., fraud_detection)"
    ),
    difficulty: str = typer.Option(
        "intermediate",
        "--difficulty",
        help="Difficulty level (beginner/intermediate/advanced)",
    ),
    max_results: int = typer.Option(
        5, "--max-results", help="Maximum number of recommendations"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save recommendations to file"
    ),
    format: str = typer.Option(
        "table", "--format", help="Output format (table/json/yaml)"
    ),
):
    """Get configuration recommendations for a specific dataset."""

    # Validate difficulty level
    difficulty_mapping = {
        "beginner": ConfigurationLevel.BEGINNER,
        "intermediate": ConfigurationLevel.INTERMEDIATE,
        "advanced": ConfigurationLevel.ADVANCED,
    }

    if difficulty not in difficulty_mapping:
        console.print(
            f"[red]Error:[/red] Invalid difficulty level. Use: {list(difficulty_mapping.keys())}"
        )
        raise typer.Exit(1)

    difficulty_level = difficulty_mapping[difficulty]

    # Check if dataset exists
    if not dataset_path.exists():
        console.print(f"[red]Error:[/red] Dataset file not found: {dataset_path}")
        raise typer.Exit(1)

    async def run_recommendation():
        container = get_cli_container()
        recommendation_service = container.recommendation_service()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Analyze dataset
            progress.add_task("Analyzing dataset characteristics...", total=None)
            dataset_characteristics = await _analyze_dataset(
                dataset_path, samples, features
            )

            # Generate recommendations
            progress.add_task("Generating intelligent recommendations...", total=None)

            performance_requirements = (
                {"min_accuracy": min_accuracy} if min_accuracy else None
            )

            recommendations = await recommendation_service.recommend_configurations(
                dataset_characteristics=dataset_characteristics,
                performance_requirements=performance_requirements,
                use_case=use_case,
                difficulty_level=difficulty_level,
                max_recommendations=max_results,
            )

        # Display results
        _display_recommendations(recommendations, format, output)

        # Show dataset characteristics
        _display_dataset_characteristics(dataset_characteristics)

    asyncio.run(run_recommendation())


@app.command("similar")
@require_feature("advanced_automl")
def recommend_similar(
    config_id: str = typer.Argument(
        ..., help="Configuration ID to find similar configurations for"
    ),
    max_results: int = typer.Option(
        5, "--max-results", help="Maximum number of recommendations"
    ),
    similarity_threshold: float = typer.Option(
        0.7, "--threshold", help="Minimum similarity threshold"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save recommendations to file"
    ),
    format: str = typer.Option(
        "table", "--format", help="Output format (table/json/yaml)"
    ),
):
    """Get recommendations similar to an existing configuration."""

    async def run_similar_recommendation():
        container = get_cli_container()
        container.recommendation_service()

        try:
            from uuid import UUID

            UUID(config_id)
        except ValueError:
            console.print("[red]Error:[/red] Invalid configuration ID format")
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Finding similar configurations...", total=None)

            # This would need to be implemented in the recommendation service
            # For now, show placeholder
            console.print(
                "[yellow]Similar configuration search not yet implemented[/yellow]"
            )
            console.print(f"Would search for configurations similar to: {config_id}")
            console.print(f"Similarity threshold: {similarity_threshold}")
            console.print(f"Max results: {max_results}")

    asyncio.run(run_similar_recommendation())


@app.command("predict")
@require_feature("advanced_automl")
def predict_performance(
    config_id: str = typer.Argument(
        ..., help="Configuration ID to predict performance for"
    ),
    dataset_path: Path = typer.Argument(..., help="Path to target dataset"),
    samples: int | None = typer.Option(None, "--samples", help="Number of samples"),
    features: int | None = typer.Option(None, "--features", help="Number of features"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save predictions to file"
    ),
    format: str = typer.Option(
        "table", "--format", help="Output format (table/json/yaml)"
    ),
):
    """Predict performance of a configuration on a specific dataset."""

    if not dataset_path.exists():
        console.print(f"[red]Error:[/red] Dataset file not found: {dataset_path}")
        raise typer.Exit(1)

    async def run_prediction():
        container = get_cli_container()
        recommendation_service = container.recommendation_service()
        repository = container.configuration_repository()

        try:
            from uuid import UUID

            config_uuid = UUID(config_id)
        except ValueError:
            console.print("[red]Error:[/red] Invalid configuration ID format")
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load configuration
            progress.add_task("Loading configuration...", total=None)
            configuration = await repository.load_configuration(config_uuid)

            if not configuration:
                console.print(f"[red]Error:[/red] Configuration not found: {config_id}")
                raise typer.Exit(1)

            # Analyze dataset
            progress.add_task("Analyzing dataset characteristics...", total=None)
            dataset_characteristics = await _analyze_dataset(
                dataset_path, samples, features
            )

            # Predict performance
            progress.add_task("Predicting performance...", total=None)
            predictions = (
                await recommendation_service.predict_configuration_performance(
                    configuration, dataset_characteristics
                )
            )

        # Display results
        _display_performance_predictions(configuration, predictions, format, output)

    asyncio.run(run_prediction())


@app.command("train")
@require_feature("advanced_automl")
def train_models(
    min_configs: int = typer.Option(
        20, "--min-configs", help="Minimum configurations needed for training"
    ),
    test_size: float = typer.Option(
        0.2, "--test-size", help="Fraction of data for testing"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force retraining even if models exist"
    ),
):
    """Train machine learning models for recommendations."""

    async def run_training():
        container = get_cli_container()
        recommendation_service = container.recommendation_service()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Training recommendation models...", total=None)

            results = await recommendation_service.train_recommendation_models(
                min_configurations=min_configs, test_size=test_size
            )

        # Display training results
        if "error" in results:
            console.print(f"[red]Training failed:[/red] {results['error']}")
            if "available_configurations" in results:
                console.print(
                    f"Available configurations: {results['available_configurations']}"
                )
        else:
            console.print("[green]✓[/green] Model training completed successfully!")

            # Show training metrics
            table = Table(title="Training Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Configurations Used", str(results["configurations_used"]))
            table.add_row("Training Size", str(results["train_size"]))
            table.add_row("Test Size", str(results["test_size"]))
            table.add_row(
                "Performance RMSE", f"{results['performance_predictor_rmse']:.4f}"
            )
            table.add_row(
                "Algorithm Accuracy", f"{results['algorithm_selector_accuracy']:.3f}"
            )
            table.add_row("Unique Algorithms", str(results["unique_algorithms"]))

            console.print(table)

            # Show feature importance
            if "feature_importance" in results:
                console.print("\n[bold]Feature Importance:[/bold]")
                importance_table = Table()
                importance_table.add_column("Feature", style="cyan")
                importance_table.add_column("Importance", style="yellow")

                for feature, importance in sorted(
                    results["feature_importance"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]:
                    importance_table.add_row(feature, f"{importance:.4f}")

                console.print(importance_table)

    asyncio.run(run_training())


@app.command("analyze")
@require_feature("advanced_automl")
def analyze_patterns(
    days: int = typer.Option(30, "--days", help="Number of days to analyze"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save analysis to file"
    ),
    format: str = typer.Option(
        "table", "--format", help="Output format (table/json/yaml)"
    ),
):
    """Analyze recommendation patterns and effectiveness."""

    async def run_analysis():
        container = get_cli_container()
        recommendation_service = container.recommendation_service()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Analyzing patterns over {days} days...", total=None)

            analysis = await recommendation_service.analyze_recommendation_patterns(
                time_period_days=days
            )

        # Display analysis results
        _display_pattern_analysis(analysis, format, output)

    asyncio.run(run_analysis())


@app.command("stats")
def show_statistics():
    """Show recommendation service statistics."""

    async def get_stats():
        container = get_cli_container()
        recommendation_service = container.recommendation_service()

        stats = recommendation_service.get_recommendation_statistics()

        # Recommendation stats
        console.print(Panel.fit("[bold]Recommendation Statistics[/bold]", style="blue"))

        rec_table = Table()
        rec_table.add_column("Metric", style="cyan")
        rec_table.add_column("Value", style="green")

        rec_stats = stats["recommendation_stats"]
        rec_table.add_row(
            "Total Recommendations", str(rec_stats["total_recommendations"])
        )
        rec_table.add_row("ML Recommendations", str(rec_stats["ml_recommendations"]))
        rec_table.add_row(
            "Similarity Recommendations", str(rec_stats["similarity_recommendations"])
        )
        rec_table.add_row(
            "Performance Predictions", str(rec_stats["performance_predictions"])
        )
        rec_table.add_row(
            "Successful Predictions", str(rec_stats["successful_predictions"])
        )
        rec_table.add_row(
            "Model Training Count", str(rec_stats["model_training_count"])
        )

        if rec_stats["last_model_training"]:
            rec_table.add_row("Last Model Training", rec_stats["last_model_training"])

        console.print(rec_table)

        # Model status
        console.print("\n[bold]Model Status[/bold]")
        model_status = stats["model_status"]

        status_table = Table()
        status_table.add_column("Model", style="cyan")
        status_table.add_column("Status", style="green")

        status_table.add_row(
            "Performance Predictor",
            (
                "✓ Trained"
                if model_status["performance_predictor_trained"]
                else "✗ Not trained"
            ),
        )
        status_table.add_row(
            "Algorithm Selector",
            (
                "✓ Trained"
                if model_status["algorithm_selector_trained"]
                else "✗ Not trained"
            ),
        )
        status_table.add_row(
            "Feature Scaler",
            (
                "✓ Available"
                if model_status["feature_scaler_available"]
                else "✗ Not available"
            ),
        )

        console.print(status_table)

        # Cache info
        console.print("\n[bold]Cache Information[/bold]")
        cache_info = stats["cache_info"]

        cache_table = Table()
        cache_table.add_column("Item", style="cyan")
        cache_table.add_column("Count", style="yellow")

        cache_table.add_row("Cached Datasets", str(cache_info["cached_datasets"]))
        cache_table.add_row(
            "Cached Configurations", str(cache_info["cached_configurations"])
        )
        cache_table.add_row("Last Cache Update", cache_info["last_cache_update"])

        console.print(cache_table)

    asyncio.run(get_stats())


# Helper functions


async def _analyze_dataset(
    dataset_path: Path, samples: int | None = None, features: int | None = None
) -> DatasetCharacteristicsDTO:
    """Analyze dataset characteristics."""
    import numpy as np
    import pandas as pd

    try:
        # Load dataset
        if dataset_path.suffix.lower() == ".csv":
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix.lower() in [".parquet", ".pq"]:
            df = pd.read_parquet(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

        # Extract characteristics
        n_samples = samples or len(df)
        n_features = features or len(df.columns)

        # Feature types
        feature_types = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_types.append("numeric")
            elif (
                pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == "object"
            ):
                feature_types.append("categorical")
            else:
                feature_types.append("other")

        # Missing values ratio
        missing_values_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))

        # Sparsity (for numeric columns only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sparsity = 0.0
        if len(numeric_cols) > 0:
            zero_count = (df[numeric_cols] == 0).sum().sum()
            total_numeric_values = len(df) * len(numeric_cols)
            sparsity = (
                zero_count / total_numeric_values if total_numeric_values > 0 else 0.0
            )

        # Outlier ratio estimation (simplified)
        outlier_ratio = 0.05  # Default estimation

        return DatasetCharacteristicsDTO(
            n_samples=n_samples,
            n_features=n_features,
            feature_types=feature_types,
            missing_values_ratio=missing_values_ratio,
            outlier_ratio=outlier_ratio,
            class_imbalance=None,  # Would need target column
            sparsity=sparsity,
            correlation_structure="unknown",
        )

    except Exception as e:
        console.print(f"[red]Error analyzing dataset:[/red] {str(e)}")
        raise typer.Exit(1)


def _display_recommendations(recommendations: list, format: str, output: Path | None):
    """Display recommendations in specified format."""

    if format == "json":
        data = [rec.dict() if hasattr(rec, "dict") else rec for rec in recommendations]

        if output:
            with open(output, "w") as f:
                json.dump(data, f, indent=2, default=str)
            console.print(f"[green]✓[/green] Recommendations saved to {output}")
        else:
            console.print(json.dumps(data, indent=2, default=str))

    elif format == "yaml":
        import yaml

        data = [rec.dict() if hasattr(rec, "dict") else rec for rec in recommendations]

        if output:
            with open(output, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
            console.print(f"[green]✓[/green] Recommendations saved to {output}")
        else:
            console.print(yaml.dump(data, default_flow_style=False))

    else:  # table format
        if not recommendations:
            console.print("[yellow]No recommendations found[/yellow]")
            return

        table = Table(title="Configuration Recommendations")
        table.add_column("Algorithm", style="cyan")
        table.add_column("Confidence", style="green")
        table.add_column("Predicted Accuracy", style="yellow")
        table.add_column("Training Time", style="blue")
        table.add_column("Reason", style="white")

        for rec in recommendations:
            confidence = f"{rec.confidence_score:.1%}"
            accuracy = "N/A"
            training_time = f"{rec.estimated_training_time}s"

            if rec.predicted_performance and "accuracy" in rec.predicted_performance:
                accuracy = f"{rec.predicted_performance['accuracy']:.3f}"

            table.add_row(
                rec.algorithm_name,
                confidence,
                accuracy,
                training_time,
                (
                    rec.recommendation_reason[:50] + "..."
                    if len(rec.recommendation_reason) > 50
                    else rec.recommendation_reason
                ),
            )

        console.print(table)


def _display_dataset_characteristics(characteristics: DatasetCharacteristicsDTO):
    """Display dataset characteristics."""
    console.print("\n[bold]Dataset Characteristics[/bold]")

    char_table = Table()
    char_table.add_column("Characteristic", style="cyan")
    char_table.add_column("Value", style="green")

    char_table.add_row("Samples", f"{characteristics.n_samples:,}")
    char_table.add_row("Features", f"{characteristics.n_features:,}")
    char_table.add_row("Missing Values", f"{characteristics.missing_values_ratio:.1%}")
    char_table.add_row("Sparsity", f"{characteristics.sparsity:.1%}")
    char_table.add_row("Outlier Ratio", f"{characteristics.outlier_ratio:.1%}")

    if characteristics.feature_types:
        type_counts = {}
        for ft in characteristics.feature_types:
            type_counts[ft] = type_counts.get(ft, 0) + 1
        char_table.add_row(
            "Feature Types", ", ".join(f"{k}: {v}" for k, v in type_counts.items())
        )

    console.print(char_table)


def _display_performance_predictions(
    configuration, predictions: Dict[str, float], format: str, output: Path | None
):
    """Display performance predictions."""

    if not predictions:
        console.print("[yellow]No performance predictions available[/yellow]")
        return

    if format == "json":
        data = {
            "configuration_id": str(configuration.id),
            "algorithm": configuration.algorithm_config.algorithm_name,
            "predictions": predictions,
        }

        if output:
            with open(output, "w") as f:
                json.dump(data, f, indent=2)
            console.print(f"[green]✓[/green] Predictions saved to {output}")
        else:
            console.print(json.dumps(data, indent=2))

    else:  # table format
        console.print(
            f"\n[bold]Performance Predictions for {configuration.algorithm_config.algorithm_name}[/bold]"
        )

        pred_table = Table()
        pred_table.add_column("Metric", style="cyan")
        pred_table.add_column("Predicted Value", style="green")

        for metric, value in predictions.items():
            if metric == "training_time_seconds":
                pred_table.add_row(metric.replace("_", " ").title(), f"{value:.1f}s")
            else:
                pred_table.add_row(metric.replace("_", " ").title(), f"{value:.3f}")

        console.print(pred_table)


def _display_pattern_analysis(
    analysis: Dict[str, Any], format: str, output: Path | None
):
    """Display pattern analysis results."""

    if format == "json":
        if output:
            with open(output, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            console.print(f"[green]✓[/green] Analysis saved to {output}")
        else:
            console.print(json.dumps(analysis, indent=2, default=str))

    else:  # table format
        console.print(
            f"\n[bold]Pattern Analysis ({analysis['time_period_days']} days)[/bold]"
        )

        # Summary
        summary_table = Table(title="Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row(
            "Total Configurations", str(analysis["total_configurations"])
        )

        if "performance_summary" in analysis:
            perf = analysis["performance_summary"]
            summary_table.add_row("Mean Accuracy", f"{perf['mean_accuracy']:.3f}")
            summary_table.add_row(
                "High Performance Rate", f"{perf['high_performance_rate']:.1%}"
            )

        console.print(summary_table)

        # Algorithm popularity
        if "algorithm_popularity" in analysis:
            console.print("\n[bold]Algorithm Popularity[/bold]")
            algo_table = Table()
            algo_table.add_column("Algorithm", style="cyan")
            algo_table.add_column("Usage Count", style="yellow")

            for algo, count in list(analysis["algorithm_popularity"].items())[:10]:
                algo_table.add_row(algo, str(count))

            console.print(algo_table)


if __name__ == "__main__":
    app()
