"""CLI commands for explainable AI functionality."""

import asyncio
import json
from pathlib import Path

import typer
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pynomaly.application.services.explainable_ai_service import (
    ExplainableAIService,
    ExplanationConfiguration,
)
from pynomaly.domain.entities.explainable_ai import (
    ExplanationMethod,
)
from pynomaly.infrastructure.config.container import Container

console = Console()

# Create the Typer app for explain commands
app = typer.Typer(help="Explainable AI commands for model interpretability")


@app.command()
def predict(
    model_path: str = typer.Option(..., "--model-path", help="Path to trained model file"),
    data_path: str = typer.Option(..., "--data-path", help="Path to data file for explanation"),
    instance_index: int | None = typer.Option(
        None, "--instance-index", help="Index of specific instance to explain"
    ),
    method: str = typer.Option(
        "shap_tree",
        "--method",
        help="Explanation method to use",
        show_default=True,
    ),
    num_features: int = typer.Option(10, "--num-features", help="Number of top features to show"),
    output_path: str | None = typer.Option(None, "--output-path", help="Path to save explanation results"),
    output_format: str = typer.Option(
        "json",
        "--format",
        help="Output format",
        show_default=True,
    ),
    audience: str = typer.Option(
        "technical",
        "--audience",
        help="Target audience for explanation",
        show_default=True,
    ),
    enable_bias_detection: bool = typer.Option(
        False, "--enable-bias-detection", help="Enable bias detection analysis"
    ),
    enable_counterfactuals: bool = typer.Option(
        False,
        "--enable-counterfactuals",
        help="Generate counterfactual explanations",
    ),
    confidence_threshold: float = typer.Option(
        0.8,
        "--confidence-threshold",
        help="Confidence threshold for explanations",
    ),
):
    """Explain a model's prediction for specific instance(s)."""

    async def run_explanation():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize explainable AI service
            task1 = progress.add_task(
                "Initializing explainable AI service...", total=None
            )
            storage_path = Path("./explanations")
            xai_service = ExplainableAIService(storage_path)
            progress.update(task1, completed=True)

            # Load model
            task2 = progress.add_task("Loading model...", total=None)
            try:
                import joblib

                model = joblib.load(model_path)
                progress.update(task2, completed=True)
            except Exception as e:
                console.print(f"[red]Error loading model: {e}[/red]")
                return

            # Load data
            task3 = progress.add_task("Loading data...", total=None)
            try:
                if data_path.endswith(".csv"):
                    data_df = pd.read_csv(data_path)
                    data = data_df.values
                    feature_names = list(data_df.columns)
                elif data_path.endswith(".json"):
                    with open(data_path) as f:
                        data_json = json.load(f)
                    data = np.array(data_json["data"])
                    feature_names = data_json.get(
                        "feature_names", [f"feature_{i}" for i in range(data.shape[1])]
                    )
                else:
                    data = np.load(data_path)
                    feature_names = [f"feature_{i}" for i in range(data.shape[1])]
                progress.update(task3, completed=True)
            except Exception as e:
                console.print(f"[red]Error loading data: {e}[/red]")
                return

            # Configure explanation
            config = ExplanationConfiguration(
                explanation_method=ExplanationMethod(method),
                num_features=num_features,
                enable_bias_detection=enable_bias_detection,
                enable_counterfactual_analysis=enable_counterfactuals,
                confidence_threshold=confidence_threshold,
            )

            # Generate explanations
            task4 = progress.add_task("Generating explanations...", total=None)

            try:
                if instance_index is not None:
                    # Explain single instance
                    if instance_index >= len(data):
                        console.print(
                            f"[red]Instance index {instance_index} out of range[/red]"
                        )
                        return

                    instance = data[instance_index]
                    result = await xai_service.explain_prediction(
                        model, instance, feature_names, config
                    )

                    progress.update(task4, completed=True)

                    # Display results
                    _display_instance_explanation(result, instance_index, audience)

                    # Generate counterfactuals if requested
                    if enable_counterfactuals:
                        counterfactuals = (
                            await xai_service.generate_counterfactual_explanations(
                                model, instance, feature_names
                            )
                        )
                        _display_counterfactuals(counterfactuals)

                else:
                    # Global explanation
                    global_explanation = await xai_service.explain_model_global(
                        model, data, feature_names, config
                    )

                    progress.update(task4, completed=True)

                    # Display results
                    _display_global_explanation(global_explanation, audience)

                # Save results if requested
                if output_path:
                    if instance_index is not None:
                        await _save_explanation_result(
                            result, output_path, output_format
                        )
                    else:
                        await _save_global_explanation(
                            global_explanation, output_path, output_format
                        )

                    console.print(f"[green]Results saved to {output_path}[/green]")

            except Exception as e:
                console.print(f"[red]Error generating explanation: {e}[/red]")
                return

    asyncio.run(run_explanation())


@app.command()
def analyze_bias(
    model_path: str = typer.Option(..., "--model-path", help="Path to trained model file"),
    data_path: str = typer.Option(..., "--data-path", help="Path to training data"),
    protected_attributes: str = typer.Option(
        ...,
        "--protected-attributes",
        help="Comma-separated list of protected attribute names",
    ),
    output_path: str | None = typer.Option(None, "--output-path", help="Path to save bias analysis results"),
    threshold: float = typer.Option(0.3, "--threshold", help="Bias detection threshold"),
):
    """Analyze model for bias and fairness issues."""

    async def run_bias_analysis():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize service
            task1 = progress.add_task("Initializing bias analysis...", total=None)
            storage_path = Path("./explanations")
            xai_service = ExplainableAIService(storage_path)
            progress.update(task1, completed=True)

            # Load model and data
            task2 = progress.add_task("Loading model and data...", total=None)
            try:
                import joblib

                model = joblib.load(model_path)

                if data_path.endswith(".csv"):
                    data_df = pd.read_csv(data_path)
                    data = data_df.values
                    feature_names = list(data_df.columns)
                else:
                    data = np.load(data_path)
                    feature_names = [f"feature_{i}" for i in range(data.shape[1])]

                progress.update(task2, completed=True)
            except Exception as e:
                console.print(f"[red]Error loading model/data: {e}[/red]")
                return

            # Run bias analysis
            task3 = progress.add_task("Analyzing bias...", total=None)
            try:
                protected_attrs = [
                    attr.strip() for attr in protected_attributes.split(",")
                ]

                bias_analysis = await xai_service.detect_explanation_bias(
                    model, data, protected_attrs, feature_names
                )

                progress.update(task3, completed=True)

                # Display results
                _display_bias_analysis(bias_analysis, threshold)

                # Save results if requested
                if output_path:
                    await _save_bias_analysis(bias_analysis, output_path)
                    console.print(
                        f"[green]Bias analysis saved to {output_path}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error analyzing bias: {e}[/red]")
                return

    asyncio.run(run_bias_analysis())


@app.command()
def feature_importance(
    model_path: str = typer.Option(..., "--model-path", help="Path to trained model file"),
    data_path: str = typer.Option(..., "--data-path", help="Path to data file"),
    method: str = typer.Option(
        "permutation_importance",
        "--method",
        help="Feature importance method",
        show_default=True,
    ),
    top_k: int = typer.Option(15, "--top-k", help="Number of top features to display"),
    output_path: str | None = typer.Option(None, "--output-path", help="Path to save feature importance results"),
    plot: bool = typer.Option(False, "--plot", help="Generate feature importance plot"),
):
    """Analyze feature importance for the model."""

    async def run_feature_analysis():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize service
            task1 = progress.add_task("Initializing feature analysis...", total=None)
            storage_path = Path("./explanations")
            xai_service = ExplainableAIService(storage_path)
            progress.update(task1, completed=True)

            # Load model and data
            task2 = progress.add_task("Loading model and data...", total=None)
            try:
                import joblib

                model = joblib.load(model_path)

                if data_path.endswith(".csv"):
                    data_df = pd.read_csv(data_path)
                    data = data_df.values
                    feature_names = list(data_df.columns)
                else:
                    data = np.load(data_path)
                    feature_names = [f"feature_{i}" for i in range(data.shape[1])]

                progress.update(task2, completed=True)
            except Exception as e:
                console.print(f"[red]Error loading model/data: {e}[/red]")
                return

            # Analyze feature importance
            task3 = progress.add_task("Computing feature importance...", total=None)
            try:
                importances = await xai_service.analyze_feature_importance(
                    model, data, feature_names, ExplanationMethod(method)
                )

                progress.update(task3, completed=True)

                # Display results
                _display_feature_importance(importances, top_k, method)

                # Generate plot if requested
                if plot:
                    _generate_importance_plot(importances, top_k, output_path)

                # Save results if requested
                if output_path and not plot:
                    await _save_feature_importance(importances, output_path)
                    console.print(
                        f"[green]Feature importance saved to {output_path}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error analyzing feature importance: {e}[/red]")
                return

    asyncio.run(run_feature_analysis())


@app.command()
def validate_explanations(
    model_path: str = typer.Option(..., "--model-path", help="Path to trained model file"),
    data_path: str = typer.Option(..., "--data-path", help="Path to validation data"),
    method: str = typer.Option(
        "shap_tree",
        "--method",
        help="Explanation method to validate",
        show_default=True,
    ),
    num_samples: int = typer.Option(100, "--num-samples", help="Number of samples for validation"),
    output_path: str | None = typer.Option(None, "--output-path", help="Path to save validation results"),
):
    """Validate explanation quality and trustworthiness."""

    async def run_validation():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize service
            task1 = progress.add_task("Initializing validation...", total=None)
            storage_path = Path("./explanations")
            xai_service = ExplainableAIService(storage_path)
            progress.update(task1, completed=True)

            # Load model and data
            task2 = progress.add_task("Loading model and data...", total=None)
            try:
                import joblib

                model = joblib.load(model_path)

                if data_path.endswith(".csv"):
                    data_df = pd.read_csv(data_path)
                    data = data_df.values
                    feature_names = list(data_df.columns)
                else:
                    data = np.load(data_path)
                    feature_names = [f"feature_{i}" for i in range(data.shape[1])]

                progress.update(task2, completed=True)
            except Exception as e:
                console.print(f"[red]Error loading model/data: {e}[/red]")
                return

            # Run validation
            task3 = progress.add_task("Validating explanations...", total=None)
            try:
                # Sample data for validation
                sample_size = min(num_samples, len(data))
                indices = np.random.choice(len(data), size=sample_size, replace=False)
                sample_data = data[indices]

                trust_scores = []

                for i, instance in enumerate(sample_data):
                    # Generate explanation
                    config = ExplanationConfiguration(
                        explanation_method=ExplanationMethod(method)
                    )

                    result = await xai_service.explain_prediction(
                        model, instance, feature_names, config
                    )

                    # Assess trust
                    trust_score = await xai_service.assess_explanation_trust(
                        result, model, sample_data
                    )

                    trust_scores.append(trust_score)

                    if i % 10 == 0:
                        progress.update(
                            task3,
                            description=f"Validating explanations... {i + 1}/{sample_size}",
                        )

                progress.update(task3, completed=True)

                # Display validation results
                _display_validation_results(trust_scores, method)

                # Save results if requested
                if output_path:
                    await _save_validation_results(trust_scores, output_path)
                    console.print(
                        f"[green]Validation results saved to {output_path}[/green]"
                    )

            except Exception as e:
                console.print(f"[red]Error validating explanations: {e}[/red]")
                return

    asyncio.run(run_validation())


@app.command()
def summary(
    model_id: str | None = typer.Option(None, "--model-id", help="Model ID to get summary for"),
    time_window: int = typer.Option(24, "--time-window", help="Time window in hours"),
    output_path: str | None = typer.Option(None, "--output-path", help="Path to save summary"),
):
    """Get explanation summary for a model."""

    async def run_summary():
        Container()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize service
            task1 = progress.add_task("Generating explanation summary...", total=None)
            storage_path = Path("./explanations")
            xai_service = ExplainableAIService(storage_path)

            from datetime import timedelta
            from uuid import UUID

            try:
                model_uuid = (
                    UUID(model_id) if model_id else UUID(int=12345)
                )  # Default UUID
                time_delta = timedelta(hours=time_window)

                summary_data = await xai_service.get_explanation_summary(
                    model_uuid, time_delta
                )

                progress.update(task1, completed=True)

                # Display summary
                _display_explanation_summary(summary_data)

                # Save if requested
                if output_path:
                    with open(output_path, "w") as f:
                        json.dump(summary_data, f, indent=2)
                    console.print(f"[green]Summary saved to {output_path}[/green]")

            except Exception as e:
                console.print(f"[red]Error generating summary: {e}[/red]")
                return

    asyncio.run(run_summary())


# Helper functions for display


def _display_instance_explanation(result, instance_index: int, audience: str):
    """Display instance explanation results."""
    if not result.success or not result.instance_explanation:
        console.print("[red]Failed to generate explanation[/red]")
        return

    explanation = result.instance_explanation

    # Main explanation panel
    console.print(
        Panel(
            f"[bold blue]Explanation for Instance {instance_index}[/bold blue]\n"
            f"Method: {result.explanation_method.value}\n"
            f"Prediction: {explanation.prediction_value}\n"
            f"Confidence: {explanation.prediction_confidence:.3f}",
            title="Instance Explanation",
        )
    )

    # Feature importance table
    table = Table(title="Feature Importance")
    table.add_column("Rank", style="cyan")
    table.add_column("Feature", style="green")
    table.add_column("Importance", style="yellow")
    table.add_column("Direction", style="blue")
    table.add_column("Category", style="magenta")

    for importance in explanation.get_top_features(10):
        direction = (
            "↑"
            if importance.contribution_direction == "positive"
            else "↓" if importance.contribution_direction == "negative" else "→"
        )
        table.add_row(
            str(importance.rank),
            importance.feature_name,
            f"{importance.importance_value:.4f}",
            direction,
            importance.get_importance_category(),
        )

    console.print(table)

    # Summary for different audiences
    if audience == "business":
        _display_business_summary(explanation)
    elif audience == "regulatory":
        _display_regulatory_summary(result)


def _display_global_explanation(global_explanation, audience: str):
    """Display global explanation results."""
    console.print(
        Panel(
            f"[bold blue]Global Model Explanation[/bold blue]\n"
            f"Method: {global_explanation.explanation_method.value}\n"
            f"Data Coverage: {global_explanation.data_coverage:.2%}\n"
            f"Features Analyzed: {len(global_explanation.global_feature_importances)}",
            title="Global Explanation",
        )
    )

    # Global feature importance
    table = Table(title="Global Feature Importance")
    table.add_column("Rank", style="cyan")
    table.add_column("Feature", style="green")
    table.add_column("Importance", style="yellow")
    table.add_column("Category", style="magenta")

    for importance in global_explanation.get_most_important_features(15):
        table.add_row(
            str(importance.rank),
            importance.feature_name,
            f"{importance.importance_value:.4f}",
            importance.get_importance_category(),
        )

    console.print(table)

    # Global summary
    summary = global_explanation.get_global_summary()
    console.print(
        Panel(
            f"Top Features: {', '.join(summary['top_features'])}\n"
            f"Feature Stability: {summary['feature_stability']:.3f}\n"
            f"Bias Detected: {'Yes' if summary['has_bias'] else 'No'}",
            title="Global Summary",
        )
    )


def _display_bias_analysis(bias_analysis, threshold: float):
    """Display bias analysis results."""
    summary = bias_analysis.get_bias_summary()

    # Bias overview
    status_color = "red" if summary["bias_detected"] else "green"
    console.print(
        Panel(
            f"[bold {status_color}]Bias Score: {summary['overall_bias']:.3f}[/bold {status_color}]\n"
            f"Severity: {summary['bias_severity']}\n"
            f"Bias Detected: {'Yes' if summary['bias_detected'] else 'No'}\n"
            f"Requires Attention: {'Yes' if summary['requires_attention'] else 'No'}",
            title="Bias Analysis Results",
        )
    )

    # Protected attributes
    if bias_analysis.protected_attribute_bias:
        table = Table(title="Protected Attribute Bias")
        table.add_column("Attribute", style="cyan")
        table.add_column("Bias Score", style="yellow")
        table.add_column("Status", style="red")

        for attr, score in bias_analysis.protected_attribute_bias.items():
            status = "HIGH" if score > threshold else "OK"
            color = "red" if score > threshold else "green"
            table.add_row(attr, f"{score:.3f}", f"[{color}]{status}[/{color}]")

        console.print(table)

    # Fairness metrics
    if bias_analysis.fairness_metrics:
        fairness_text = "\n".join(
            [f"{k}: {v:.3f}" for k, v in bias_analysis.fairness_metrics.items()]
        )
        console.print(Panel(fairness_text, title="Fairness Metrics"))


def _display_feature_importance(importances, top_k: int, method: str):
    """Display feature importance results."""
    console.print(
        Panel(
            f"[bold blue]Feature Importance Analysis[/bold blue]\n"
            f"Method: {method}\n"
            f"Top {top_k} Features",
            title="Feature Importance",
        )
    )

    table = Table()
    table.add_column("Rank", style="cyan")
    table.add_column("Feature", style="green")
    table.add_column("Importance", style="yellow")
    table.add_column("Confidence", style="blue")
    table.add_column("Category", style="magenta")

    for importance in importances[:top_k]:
        table.add_row(
            str(importance.rank),
            importance.feature_name,
            f"{importance.importance_value:.4f}",
            f"{importance.confidence:.3f}",
            importance.get_importance_category(),
        )

    console.print(table)


def _display_validation_results(trust_scores, method: str):
    """Display validation results."""
    overall_trust = np.mean([ts.overall_trust_score for ts in trust_scores])
    high_trust_count = sum(1 for ts in trust_scores if ts.trust_level.value == "high")

    console.print(
        Panel(
            f"[bold blue]Explanation Validation Results[/bold blue]\n"
            f"Method: {method}\n"
            f"Samples Validated: {len(trust_scores)}\n"
            f"Average Trust Score: {overall_trust:.3f}\n"
            f"High Trust Explanations: {high_trust_count}/{len(trust_scores)}",
            title="Validation Results",
        )
    )

    # Trust breakdown
    avg_consistency = np.mean([ts.consistency_score for ts in trust_scores])
    avg_stability = np.mean([ts.stability_score for ts in trust_scores])
    avg_fidelity = np.mean([ts.fidelity_score for ts in trust_scores])

    table = Table(title="Trust Component Breakdown")
    table.add_column("Component", style="cyan")
    table.add_column("Average Score", style="yellow")
    table.add_column("Status", style="green")

    components = [
        ("Consistency", avg_consistency),
        ("Stability", avg_stability),
        ("Fidelity", avg_fidelity),
    ]

    for name, score in components:
        status = "Good" if score > 0.8 else "Fair" if score > 0.6 else "Poor"
        color = "green" if score > 0.8 else "yellow" if score > 0.6 else "red"
        table.add_row(name, f"{score:.3f}", f"[{color}]{status}[/{color}]")

    console.print(table)


def _display_counterfactuals(counterfactuals):
    """Display counterfactual explanations."""
    if not counterfactuals:
        console.print("[yellow]No counterfactual explanations generated[/yellow]")
        return

    console.print(
        Panel(
            f"[bold blue]Counterfactual Explanations[/bold blue]\n"
            f"Generated {len(counterfactuals)} counterfactuals",
            title="What-If Analysis",
        )
    )

    for i, cf in enumerate(counterfactuals[:3]):  # Show top 3
        changes_text = "\n".join(
            [
                f"  {feature}: {change['original']:.3f} → {change['counterfactual']:.3f}"
                for feature, change in cf["feature_changes"].items()
            ]
        )

        console.print(
            Panel(
                f"Original Prediction: {cf['original_prediction']}\n"
                f"Counterfactual Prediction: {cf['counterfactual_prediction']}\n"
                f"Feature Changes:\n{changes_text}",
                title=f"Counterfactual {i + 1}",
            )
        )


def _display_business_summary(explanation):
    """Display business-friendly summary."""
    contribution_summary = explanation.get_feature_contribution_summary()

    console.print(
        Panel(
            f"[bold blue]Business Impact Summary[/bold blue]\n"
            f"Prediction Confidence: {explanation.prediction_confidence:.1%}\n"
            f"Key Drivers ({len(contribution_summary['top_positive'])}): "
            f"{', '.join([f.feature_name for f in contribution_summary['top_positive']])}\n"
            f"Risk Factors ({len(contribution_summary['top_negative'])}): "
            f"{', '.join([f.feature_name for f in contribution_summary['top_negative']])}",
            title="Business Summary",
        )
    )


def _display_regulatory_summary(result):
    """Display regulatory compliance summary."""
    console.print(
        Panel(
            f"[bold blue]Regulatory Compliance[/bold blue]\n"
            f"Explanation Method: {result.explanation_method.value}\n"
            f"Trust Score: {result.trust_score.overall_trust_score:.3f if result.trust_score else 'N/A'}\n"
            f"Bias Detected: {'Yes' if result.has_bias_issues() else 'No'}\n"
            f"Explanation Quality: {'High' if result.is_high_quality() else 'Standard'}",
            title="Regulatory Summary",
        )
    )


def _display_explanation_summary(summary_data):
    """Display explanation summary."""
    console.print(
        Panel(
            f"[bold blue]Explanation Activity Summary[/bold blue]\n"
            f"Model ID: {summary_data['model_id']}\n"
            f"Time Window: {summary_data['time_window_hours']:.1f} hours\n"
            f"Total Explanations: {summary_data['explanation_stats']['total_explanations']}\n"
            f"Cache Hit Rate: {summary_data['explanation_stats']['cache_hit_rate']:.1%}",
            title="Summary",
        )
    )


# Helper functions for saving results


async def _save_explanation_result(result, output_path: str, format: str):
    """Save explanation result to file."""
    if format == "json":
        data = result.get_result_summary()
        if result.instance_explanation:
            data["feature_importances"] = [
                {
                    "feature": f.feature_name,
                    "importance": f.importance_value,
                    "rank": f.rank,
                }
                for f in result.instance_explanation.feature_importances
            ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    elif format == "csv" and result.instance_explanation:
        df = pd.DataFrame(
            [
                {
                    "feature": f.feature_name,
                    "importance": f.importance_value,
                    "rank": f.rank,
                    "category": f.get_importance_category(),
                }
                for f in result.instance_explanation.feature_importances
            ]
        )
        df.to_csv(output_path, index=False)


async def _save_global_explanation(explanation, output_path: str, format: str):
    """Save global explanation to file."""
    if format == "json":
        data = explanation.get_global_summary()
        data["feature_importances"] = [
            {
                "feature": f.feature_name,
                "importance": f.importance_value,
                "rank": f.rank,
            }
            for f in explanation.global_feature_importances
        ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


async def _save_bias_analysis(bias_analysis, output_path: str):
    """Save bias analysis to file."""
    data = bias_analysis.get_bias_summary()

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


async def _save_feature_importance(importances, output_path: str):
    """Save feature importance to file."""
    df = pd.DataFrame(
        [
            {
                "feature": f.feature_name,
                "importance": f.importance_value,
                "rank": f.rank,
                "confidence": f.confidence,
                "category": f.get_importance_category(),
            }
            for f in importances
        ]
    )
    df.to_csv(output_path, index=False)


async def _save_validation_results(trust_scores, output_path: str):
    """Save validation results to file."""
    data = {
        "summary": {
            "total_samples": len(trust_scores),
            "average_trust": np.mean([ts.overall_trust_score for ts in trust_scores]),
            "high_trust_count": sum(
                1 for ts in trust_scores if ts.trust_level.value == "high"
            ),
        },
        "detailed_scores": [ts.get_trust_breakdown() for ts in trust_scores],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _generate_importance_plot(importances, top_k: int, output_path: str | None):
    """Generate feature importance plot."""
    try:
        import matplotlib.pyplot as plt

        top_features = importances[:top_k]
        features = [f.feature_name for f in top_features]
        values = [f.importance_value for f in top_features]

        plt.figure(figsize=(10, 6))
        plt.barh(features, values)
        plt.xlabel("Importance")
        plt.title(f"Top {top_k} Feature Importances")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path.replace(".json", ".png").replace(".csv", ".png"))
        else:
            plt.savefig("feature_importance.png")

        console.print("[green]Feature importance plot saved[/green]")

    except ImportError:
        console.print("[yellow]Matplotlib not available for plotting[/yellow]")


if __name__ == "__main__":
    explain_commands()
