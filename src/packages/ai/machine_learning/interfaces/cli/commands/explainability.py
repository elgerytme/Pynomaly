"""CLI commands for explainable AI and model interpretability."""

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

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Application imports
from interfaces.application.services.advanced_explainability_service import (
    AdvancedExplainabilityService,
    BiasAnalysisConfig,
    ExplanationConfig,
    TrustScoreConfig,
)

# Domain imports
from interfaces.domain.entities import Dataset
from interfaces.infrastructure.config.feature_flags import require_feature

# Infrastructure imports
from monorepo.infrastructure.data_loaders import CSVLoader, ParquetLoader
from interfaces.shared.protocols import DetectorProtocol

console = Console()

# Create Typer app
app = typer.Typer(
    name="explainability",
    help="🔍 Explainable AI (model interpretability, bias analysis)",
    add_completion=True,
    rich_markup_mode="rich",
)


@app.command()
@require_feature("advanced_explainability")
def explain(
    detector_path: Path = typer.Argument(
        ..., help="Path to saved detector model", exists=True
    ),
    dataset_path: Path = typer.Argument(
        ..., help="Path to dataset file (CSV or Parquet)", exists=True
    ),
    explanation_type: str = typer.Option(
        "both",
        "-t",
        "--explanation-type",
        help="Type of explanation to generate (options: local, global, both)",
    ),
    methods: list[str] | None = typer.Option(
        None,
        "-m",
        "--methods",
        help="Explanation methods to use (options: shap, lime, permutation, gradient)",
    ),
    n_samples: int = typer.Option(
        10, "--n-samples", help="Number of samples for local explanations"
    ),
    audience: str = typer.Option(
        "technical",
        "--audience",
        help="Target audience for explanations (technical, business, regulatory)",
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Output file for explanation report"
    ),
    output_format: str = typer.Option(
        "json", "--format", help="Output format (json, html, pdf)"
    ),
    visualizations: bool = typer.Option(
        True,
        "--visualizations/--no-visualizations",
        help="Generate visualization plots",
    ),
):
    """Generate comprehensive explanations for model predictions.

    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file (CSV or Parquet)

    Examples:
        anomaly_detection explainability explain model.pkl data.csv
        anomaly_detection explainability explain model.pkl data.csv --methods shap lime
        anomaly_detection explainability explain model.pkl data.csv --explanation-type local --n-samples 20
    """
    try:
        # Load dataset
        console.print(f"📊 Loading dataset: {dataset_path}")
        dataset = _load_dataset(dataset_path)

        # Load detector (simplified - would need proper loading)
        console.print(f"🤖 Loading detector: {detector_path}")
        detector = _load_detector(detector_path)

        # Configure explanation
        config = ExplanationConfig(
            explanation_type=explanation_type,
            method=methods[0] if methods and len(methods) > 0 else "shap",
            n_samples=n_samples,
            feature_names=dataset.feature_names,
            target_audience=audience,
            generate_plots=visualizations,
        )

        # Initialize explainability service
        explainability_service = AdvancedExplainabilityService(
            enable_shap=True,
            enable_lime=True,
            enable_permutation=True,
            cache_explanations=True,
        )

        console.print("🔍 Generating comprehensive explanations...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing model behavior...", total=None)

            start_time = time.time()

            # Generate explanation report
            report = asyncio.run(
                explainability_service.generate_comprehensive_explanation(
                    detector=detector, dataset=dataset, config=config
                )
            )

            generation_time = time.time() - start_time
            progress.update(task, completed=100)

        # Display results
        _display_explanation_results(report, generation_time)

        # Save report if requested
        if output:
            _save_explanation_report(report, output, output_format)
            console.print(f"💾 Explanation report saved to: {output}")

        console.print(
            "✅ Explanation generation completed successfully!", style="green"
        )

    except Exception as e:
        console.print(f"❌ Explanation generation failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("advanced_explainability")
def analyze_bias(
    detector_path: Path = typer.Argument(
        ..., help="Path to saved detector model", exists=True
    ),
    dataset_path: Path = typer.Argument(
        ..., help="Path to dataset file with protected attributes", exists=True
    ),
    protected_attributes: list[str] | None = typer.Option(
        None, "-p", "--protected-attributes", help="Protected attribute column names"
    ),
    metrics: list[str] | None = typer.Option(
        None,
        "-m",
        "--metrics",
        help="Fairness metrics to compute (options: demographic_parity, equalized_odds, statistical_parity)",
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Decision threshold for binary classification"
    ),
    min_group_size: int = typer.Option(
        30, "--min-group-size", help="Minimum group size for analysis"
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Output file for bias analysis results"
    ),
):
    """Analyze model for potential bias and fairness issues.

    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file with protected attributes

    Examples:
        anomaly_detection explainability analyze-bias model.pkl data.csv --protected-attributes gender age
        anomaly_detection explainability analyze-bias model.pkl data.csv -p race -m demographic_parity equalized_odds
    """
    try:
        if not protected_attributes or len(protected_attributes) == 0:
            console.print(
                "❌ At least one protected attribute must be specified", style="red"
            )
            sys.exit(1)

        # Load dataset and detector
        dataset = _load_dataset(dataset_path)
        detector = _load_detector(detector_path)

        # Configure bias analysis
        config = BiasAnalysisConfig(
            protected_attributes=protected_attributes,
            fairness_metrics=(
                metrics if metrics else ["demographic_parity", "equalized_odds"]
            ),
            threshold=threshold,
            min_group_size=min_group_size,
        )

        # Initialize service
        explainability_service = AdvancedExplainabilityService()

        console.print("⚖️ Analyzing model bias and fairness...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Performing bias analysis...", total=None)

            # Run bias analysis
            bias_results = asyncio.run(
                explainability_service.analyze_bias(
                    detector=detector, dataset=dataset, config=config
                )
            )

            progress.update(task, completed=100)

        # Display results
        _display_bias_analysis_results(bias_results)

        # Save results if requested
        if output:
            _save_bias_analysis_results(bias_results, output)
            console.print(f"💾 Bias analysis results saved to: {output}")

        console.print("✅ Bias analysis completed!", style="green")

    except Exception as e:
        console.print(f"❌ Bias analysis failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("advanced_explainability")
def assess_trust(
    detector_path: Path = typer.Argument(
        ..., help="Path to saved detector model", exists=True
    ),
    dataset_path: Path = typer.Argument(..., help="Path to dataset file", exists=True),
    n_perturbations: int = typer.Option(
        100, "--n-perturbations", help="Number of perturbations for stability analysis"
    ),
    perturbation_strength: float = typer.Option(
        0.1, "--perturbation-strength", help="Strength of perturbations (0-1)"
    ),
    consistency: bool = typer.Option(
        True, "--consistency/--no-consistency", help="Enable consistency analysis"
    ),
    stability: bool = typer.Option(
        True, "--stability/--no-stability", help="Enable stability analysis"
    ),
    fidelity: bool = typer.Option(
        True, "--fidelity/--no-fidelity", help="Enable fidelity assessment"
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Output file for trust assessment"
    ),
):
    """Assess trust and reliability of model predictions.

    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file

    Examples:
        anomaly_detection explainability assess-trust model.pkl data.csv
        anomaly_detection explainability assess-trust model.pkl data.csv --n-perturbations 200
    """
    try:
        # Load dataset and detector
        dataset = _load_dataset(dataset_path)
        detector = _load_detector(detector_path)

        # Configure trust assessment
        config = TrustScoreConfig(
            consistency_checks=consistency,
            stability_analysis=stability,
            fidelity_assessment=fidelity,
            n_perturbations=n_perturbations,
            perturbation_strength=perturbation_strength,
        )

        # Initialize service
        explainability_service = AdvancedExplainabilityService()

        console.print("🔒 Assessing model trust and reliability...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Performing trust assessment...", total=None)

            # Get predictions for trust assessment
            X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
            predictions = detector.decision_function(X)

            # Run trust assessment
            trust_result = asyncio.run(
                explainability_service._assess_trust_score(
                    detector=detector, X=X, predictions=predictions, config=config
                )
            )

            progress.update(task, completed=100)

        # Display results
        _display_trust_assessment_results(trust_result)

        # Save results if requested
        if output:
            _save_trust_assessment_results(trust_result, output)
            console.print(f"💾 Trust assessment saved to: {output}")

        console.print("✅ Trust assessment completed!", style="green")

    except Exception as e:
        console.print(f"❌ Trust assessment failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("advanced_explainability")
def feature_importance(
    detector_path: Path = typer.Argument(
        ..., help="Path to saved detector model", exists=True
    ),
    dataset_path: Path = typer.Argument(..., help="Path to dataset file", exists=True),
    method: str = typer.Option(
        "shap", "--method", help="Feature importance method (shap, lime, permutation)"
    ),
    top_k: int = typer.Option(15, "--top-k", help="Number of top features to display"),
    output: Path | None = typer.Option(
        None, "--output", help="Output file for feature importance"
    ),
):
    """Analyze global feature importance for the model.

    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file

    Examples:
        anomaly_detection explainability feature-importance model.pkl data.csv
        anomaly_detection explainability feature-importance model.pkl data.csv --method permutation --top-k 20
    """
    try:
        # Load dataset and detector
        dataset = _load_dataset(dataset_path)
        detector = _load_detector(detector_path)

        # Initialize service
        explainability_service = AdvancedExplainabilityService()

        console.print(f"📈 Computing {method.upper()} feature importance...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Computing feature importance...", total=None)

            X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
            feature_names = dataset.feature_names or [
                f"feature_{i}" for i in range(X.shape[1])
            ]

            # Compute feature importance based on method
            if method == "permutation":
                importance = asyncio.run(
                    explainability_service._compute_permutation_importance(
                        detector, X, feature_names
                    )
                )
            elif method == "shap":
                importance = asyncio.run(
                    explainability_service._compute_shap_global_importance(
                        detector, X, feature_names
                    )
                )
            else:  # lime or fallback
                importance = explainability_service._compute_variance_importance(
                    X, feature_names
                )

            progress.update(task, completed=100)

        if importance:
            # Display feature importance
            _display_feature_importance(importance, top_k, method)

            # Save results if requested
            if output:
                _save_feature_importance(importance, output)
                console.print(f"💾 Feature importance saved to: {output}")
        else:
            console.print(
                f"⚠️ Failed to compute {method} feature importance", style="yellow"
            )

        console.print("✅ Feature importance analysis completed!", style="green")

    except Exception as e:
        console.print(f"❌ Feature importance analysis failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("advanced_explainability")
def status(
    cache_info: bool = typer.Option(
        True, "--cache-info/--no-cache-info", help="Show cache information"
    ),
):
    """Show explainability service status and capabilities.

    Examples:
        anomaly_detection explainability status
    """
    try:
        # Initialize service
        explainability_service = AdvancedExplainabilityService()

        # Get service information
        service_info = explainability_service.get_service_info()

        console.print("🔍 Explainability Service Status", style="bold")

        # Availability table
        availability_table = Table(title="Method Availability")
        availability_table.add_column("Method", style="cyan")
        availability_table.add_column("Available", style="white")
        availability_table.add_column("Enabled", style="green")
        availability_table.add_column("Description", style="white")

        methods = [
            (
                "SHAP",
                service_info["shap_available"],
                service_info["shap_enabled"],
                "SHapley Additive exPlanations",
            ),
            (
                "LIME",
                service_info["lime_available"],
                service_info["lime_enabled"],
                "Local Interpretable Model-agnostic Explanations",
            ),
            (
                "Permutation",
                service_info["sklearn_available"],
                service_info["permutation_enabled"],
                "Permutation Feature Importance",
            ),
        ]

        for method, available, enabled, description in methods:
            avail_status = "✓" if available else "✗"
            avail_color = "green" if available else "red"

            enabled_status = "✓" if enabled else "✗"
            enabled_color = "green" if enabled else "yellow"

            availability_table.add_row(
                method,
                f"[{avail_color}]{avail_status}[/{avail_color}]",
                f"[{enabled_color}]{enabled_status}[/{enabled_color}]",
                description,
            )

        console.print(availability_table)

        # Cache information
        if cache_info and service_info["cache_enabled"]:
            cache_text = f"""
            Cache Enabled: ✓
            Cached Explanations: {service_info["cached_explanations"]}
            Cached Explainers: {service_info["cached_explainers"]}
            """

            console.print(Panel(cache_text, title="Cache Status", border_style="blue"))

        # Capabilities summary
        capabilities = []
        if service_info["shap_enabled"]:
            capabilities.append("Local & Global SHAP explanations")
        if service_info["lime_enabled"]:
            capabilities.append("LIME local explanations")
        if service_info["permutation_enabled"]:
            capabilities.append("Permutation importance")

        capabilities.extend(
            [
                "Bias analysis with fairness metrics",
                "Trust assessment with stability analysis",
                "Feature interaction analysis",
                "Explanation caching for performance",
            ]
        )

        console.print("\n🎯 Available Capabilities:")
        for capability in capabilities:
            console.print(f"  • {capability}")

        console.print("\n✅ Explainability service status displayed!", style="green")

    except Exception as e:
        console.print(f"❌ Failed to get service status: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("advanced_explainability")
def counterfactuals(
    detector_path: Path = typer.Argument(
        ..., help="Path to saved detector model", exists=True
    ),
    dataset_path: Path = typer.Argument(..., help="Path to dataset file", exists=True),
    instance_index: int = typer.Option(
        0, "--instance", help="Index of instance to generate counterfactuals for"
    ),
    num_counterfactuals: int = typer.Option(
        5, "--num-counterfactuals", help="Number of counterfactuals to generate"
    ),
    method: str = typer.Option(
        "lime",
        "--method",
        help="Method to use for counterfactual generation (lime, gradient)",
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Output file for counterfactual results"
    ),
):
    """Generate counterfactual explanations for specific instances.

    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file

    Examples:
        anomaly_detection explainability counterfactuals model.pkl data.csv --instance 10
        anomaly_detection explainability counterfactuals model.pkl data.csv --num-counterfactuals 10
    """
    try:
        # Load dataset and detector
        dataset = _load_dataset(dataset_path)
        detector = _load_detector(detector_path)

        # Get instance
        X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
        if instance_index >= len(X):
            console.print(
                f"❌ Instance index {instance_index} is out of range (max: {len(X)-1})",
                style="red",
            )
            sys.exit(1)

        instance = X[instance_index]
        feature_names = dataset.feature_names or [
            f"feature_{i}" for i in range(len(instance))
        ]

        # Initialize explainability service
        explainability_service = AdvancedExplainabilityService()

        console.print(
            f"🔄 Generating {num_counterfactuals} counterfactual explanations..."
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating counterfactuals...", total=None)

            # Generate counterfactuals based on method
            if method == "lime":
                # Import LIME explainer
                from monorepo.infrastructure.explainers.lime_explainer import (
                    LIMEExplainer,
                )

                lime_explainer = LIMEExplainer(
                    training_data=X,
                    feature_names=feature_names,
                    enable_submodular_pick=True,
                    stability_analysis=True,
                )

                counterfactuals = lime_explainer.generate_counterfactual_explanations(
                    instance=instance,
                    model=detector,
                    feature_names=feature_names,
                    num_counterfactuals=num_counterfactuals,
                )
            else:
                # Use advanced explainability service
                counterfactuals = asyncio.run(
                    explainability_service.generate_counterfactual_explanations(
                        detector=detector,
                        instance=instance,
                        feature_names=feature_names,
                        num_counterfactuals=num_counterfactuals,
                        method=method,
                    )
                )

            progress.update(task, completed=100)

        # Display counterfactual results
        _display_counterfactual_results(counterfactuals, instance, feature_names)

        # Save results if requested
        if output:
            _save_counterfactual_results(counterfactuals, output)
            console.print(f"💾 Counterfactual results saved to: {output}")

        console.print("✅ Counterfactual generation completed!", style="green")

    except Exception as e:
        console.print(f"❌ Counterfactual generation failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("advanced_explainability")
def interactions(
    detector_path: Path = typer.Argument(
        ..., help="Path to saved detector model", exists=True
    ),
    dataset_path: Path = typer.Argument(..., help="Path to dataset file", exists=True),
    max_interactions: int = typer.Option(
        20, "--max-interactions", help="Maximum number of interactions to analyze"
    ),
    method: str = typer.Option(
        "shap", "--method", help="Method to use for interaction analysis (shap, lime)"
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Output file for interaction analysis"
    ),
):
    """Analyze feature interactions using advanced explainability methods.

    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file

    Examples:
        anomaly_detection explainability interactions model.pkl data.csv
        anomaly_detection explainability interactions model.pkl data.csv --max-interactions 50
    """
    try:
        # Load dataset and detector
        dataset = _load_dataset(dataset_path)
        detector = _load_detector(detector_path)

        X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
        feature_names = dataset.feature_names or [
            f"feature_{i}" for i in range(X.shape[1])
        ]

        console.print(f"🔍 Analyzing feature interactions using {method.upper()}...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing interactions...", total=None)

            # Analyze interactions based on method
            if method == "shap":
                # Import SHAP explainer
                from monorepo.infrastructure.explainers.shap_explainer import (
                    SHAPExplainer,
                )

                shap_explainer = SHAPExplainer(
                    explainer_type="auto",
                    background_data=X[:100],  # Use subset for background
                    enable_interactions=True,
                    enable_clustering=True,
                )

                interactions = shap_explainer.get_interaction_values(
                    instances=X,
                    model=detector,
                    feature_names=feature_names,
                    max_interactions=max_interactions,
                )
            else:
                # Use advanced explainability service
                explainability_service = AdvancedExplainabilityService()
                interactions = asyncio.run(
                    explainability_service.analyze_feature_interactions(
                        detector=detector,
                        X=X,
                        feature_names=feature_names,
                        max_interactions=max_interactions,
                        method=method,
                    )
                )

            progress.update(task, completed=100)

        # Display interaction results
        _display_interaction_results(interactions, method)

        # Save results if requested
        if output:
            _save_interaction_results(interactions, output)
            console.print(f"💾 Interaction analysis saved to: {output}")

        console.print("✅ Feature interaction analysis completed!", style="green")

    except Exception as e:
        console.print(f"❌ Feature interaction analysis failed: {e}", style="red")
        sys.exit(1)


@app.command()
@require_feature("advanced_explainability")
def dashboard(
    detector_path: Path = typer.Argument(
        ..., help="Path to saved detector model", exists=True
    ),
    dataset_path: Path = typer.Argument(..., help="Path to dataset file", exists=True),
    port: int = typer.Option(8080, "--port", help="Port to run dashboard on"),
    host: str = typer.Option("localhost", "--host", help="Host to run dashboard on"),
    output: Path | None = typer.Option(
        None, "--output", help="Output directory for dashboard data"
    ),
):
    """Generate interactive explanation dashboard.

    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file

    Examples:
        anomaly_detection explainability dashboard model.pkl data.csv
        anomaly_detection explainability dashboard model.pkl data.csv --port 8080
    """
    try:
        # Load dataset and detector
        dataset = _load_dataset(dataset_path)
        detector = _load_detector(detector_path)

        # Initialize explainability service
        explainability_service = AdvancedExplainabilityService()

        console.print("🎨 Generating interactive explanation dashboard...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Preparing dashboard data...", total=None)

            # Generate dashboard data
            dashboard_data = asyncio.run(
                explainability_service.generate_explanation_dashboard_data(
                    detector=detector,
                    dataset=dataset,
                    include_interactions=True,
                    include_counterfactuals=True,
                    include_clustering=True,
                )
            )

            progress.update(task, completed=100)

        # Display dashboard info
        _display_dashboard_info(dashboard_data, host, port)

        # Save dashboard data if requested
        if output:
            _save_dashboard_data(dashboard_data, output)
            console.print(f"💾 Dashboard data saved to: {output}")

        console.print(
            f"✅ Dashboard prepared! Access at http://{host}:{port}", style="green"
        )
        console.print(
            "Note: Dashboard server implementation would be started here.",
            style="yellow",
        )

    except Exception as e:
        console.print(f"❌ Dashboard generation failed: {e}", style="red")
        sys.exit(1)


@app.command()
def info(
    explanation_type: str = typer.Option(
        ..., help="Type of explanation to get information about"
    ),
):
    """Get detailed information about explanation types and methods.

    EXPLANATION_TYPE: Type of explanation to get information about

    Examples:
        anomaly_detection explainability info local
        anomaly_detection explainability info bias
    """
    try:
        _display_explanation_info(explanation_type)
        console.print(
            f"✅ Information for {explanation_type} explanations displayed!",
            style="green",
        )

    except Exception as e:
        console.print(f"❌ Failed to display information: {e}", style="red")
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


def _load_detector(detector_path: Path) -> DetectorProtocol:
    """Load detector from file (simplified implementation)."""
    try:
        # This is a simplified implementation
        # In practice, would need proper model loading/deserialization
        import pickle

        with open(detector_path, "rb") as f:
            detector = pickle.load(f)

        return detector

    except Exception as e:
        # Fallback: create a mock detector for demonstration
        console.print(f"⚠️ Could not load detector, using mock: {e}", style="yellow")
        return _create_mock_detector()


def _create_mock_detector():
    """Create mock detector for demonstration."""
    from unittest.mock import Mock

    import numpy as np

    mock_detector = Mock()
    mock_detector.decision_function = lambda X: np.random.random(len(X))
    mock_detector.predict = lambda X: (np.random.random(len(X)) > 0.5).astype(int)
    mock_detector.algorithm_name = "MockDetector"
    mock_detector.algorithm_params = {"mock": True}
    mock_detector.is_trained = True

    return mock_detector


def _display_explanation_results(report, generation_time: float):
    """Display explanation report results."""
    console.print("\n🔍 Explanation Report Results", style="bold")

    # Summary panel
    summary_text = f"""
    Generation Time: {generation_time:.2f}s
    Local Explanations: {len(report.local_explanations)}
    Global Explanation: {"✓" if report.global_explanation else "✗"}
    Bias Analysis: {len(report.bias_analysis)} attributes
    Trust Score: {report.trust_assessment.overall_trust_score:.3f}
    Risk Level: {report.trust_assessment.risk_assessment.title()}
    """

    console.print(Panel(summary_text, title="Summary", border_style="blue"))

    # Trust assessment details
    trust = report.trust_assessment
    console.print("\n🔒 Trust Assessment:")
    console.print(f"  Overall Trust Score: {trust.overall_trust_score:.3f}")
    console.print(f"  Consistency: {trust.consistency_score:.3f}")
    console.print(f"  Stability: {trust.stability_score:.3f}")
    console.print(f"  Fidelity: {trust.fidelity_score:.3f}")
    console.print(f"  Risk Level: {trust.risk_assessment.title()}")

    # Feature importance from global explanation
    if report.global_explanation and hasattr(
        report.global_explanation, "feature_importance"
    ):
        console.print("\n📈 Top Features (Global Importance):")
        importance = report.global_explanation.feature_importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        for i, (feature, score) in enumerate(sorted_features, 1):
            console.print(f"  {i:2d}. {feature}: {score:.4f}")

    # Recommendations
    if report.recommendations:
        console.print("\n💡 Recommendations:")
        for rec in report.recommendations:
            console.print(f"  • {rec}")


def _display_bias_analysis_results(bias_results):
    """Display bias analysis results."""
    console.print("\n⚖️ Bias Analysis Results", style="bold")

    if not bias_results:
        console.print("No bias analysis results available", style="yellow")
        return

    for result in bias_results:
        console.print(f"\n📊 Protected Attribute: {result.protected_attribute}")
        console.print(f"Bias Detected: {'Yes' if result.bias_detected else 'No'}")
        console.print(f"Severity: {result.severity.title()}")

        # Fairness metrics
        if result.fairness_metrics:
            metrics_table = Table(
                title=f"Fairness Metrics - {result.protected_attribute}"
            )
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="white")
            metrics_table.add_column("Status", style="white")

            for metric, value in result.fairness_metrics.items():
                status = (
                    "✓ Pass"
                    if value >= 0.8
                    else "⚠️ Concerning"
                    if value >= 0.6
                    else "❌ Fail"
                )
                color = "green" if value >= 0.8 else "yellow" if value >= 0.6 else "red"

                metrics_table.add_row(
                    metric.replace("_", " ").title(),
                    f"{value:.3f}",
                    f"[{color}]{status}[/{color}]",
                )

            console.print(metrics_table)

        # Group statistics
        if result.group_statistics:
            console.print("\n👥 Group Statistics:")
            for group, stats in result.group_statistics.items():
                console.print(
                    f"  {group}: {stats['size']} samples, {stats['positive_rate']:.1%} positive rate"
                )

        # Recommendations
        if result.recommendations:
            console.print("\n💡 Recommendations:")
            for rec in result.recommendations:
                console.print(f"  • {rec}")


def _display_trust_assessment_results(trust_result):
    """Display trust assessment results."""
    console.print("\n🔒 Trust Assessment Results", style="bold")

    # Overall score panel
    score_text = f"""
    Overall Trust Score: {trust_result.overall_trust_score:.3f}
    Risk Level: {trust_result.risk_assessment.title()}

    Individual Factors:
    • Consistency: {trust_result.consistency_score:.3f}
    • Stability: {trust_result.stability_score:.3f}
    • Fidelity: {trust_result.fidelity_score:.3f}
    """

    console.print(Panel(score_text, title="Trust Score Summary", border_style="blue"))

    # Trust factors table
    factors_table = Table(title="Trust Factors")
    factors_table.add_column("Factor", style="cyan")
    factors_table.add_column("Score", style="white")
    factors_table.add_column("Status", style="white")

    for factor, score in trust_result.trust_factors.items():
        status = (
            "✓ Good" if score >= 0.8 else "⚠️ Moderate" if score >= 0.6 else "❌ Poor"
        )
        color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"

        factors_table.add_row(
            factor.replace("_", " ").title(),
            f"{score:.3f}",
            f"[{color}]{status}[/{color}]",
        )

    console.print(factors_table)

    # Risk assessment
    risk_color = (
        "green"
        if trust_result.risk_assessment == "low"
        else "yellow"
        if trust_result.risk_assessment == "medium"
        else "red"
    )
    console.print(
        f"\n🚨 Risk Assessment: [{risk_color}]{trust_result.risk_assessment.upper()}[/{risk_color}]"
    )


def _display_feature_importance(importance: dict, top_k: int, method: str):
    """Display feature importance results."""
    console.print(f"\n📈 {method.upper()} Feature Importance", style="bold")

    # Sort features by importance
    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[
        :top_k
    ]

    # Create importance table
    table = Table(title=f"Top {top_k} Features")
    table.add_column("Rank", style="cyan")
    table.add_column("Feature", style="white")
    table.add_column("Importance", style="green")
    table.add_column("Bar", style="blue")

    max_importance = (
        max(abs(score) for _, score in sorted_features) if sorted_features else 1.0
    )

    for rank, (feature, importance_score) in enumerate(sorted_features, 1):
        # Create simple bar visualization
        bar_length = int(20 * abs(importance_score) / max_importance)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        table.add_row(str(rank), feature, f"{importance_score:.4f}", bar)

    console.print(table)


def _display_explanation_info(explanation_type: str):
    """Display detailed information about explanation types."""
    info_data = {
        "local": {
            "title": "Local Explanations",
            "description": "Explain individual predictions by showing feature contributions",
            "methods": ["SHAP", "LIME", "Gradient-based"],
            "use_cases": [
                "Understanding why a specific sample was classified as anomalous",
                "Debugging model decisions on individual cases",
                "Providing explanations to end users for specific predictions",
            ],
            "outputs": [
                "Feature contribution scores",
                "Confidence measures",
                "Trust scores",
            ],
            "limitations": [
                "May not generalize to other samples",
                "Can be computationally expensive",
            ],
        },
        "global": {
            "title": "Global Explanations",
            "description": "Explain overall model behavior across all data",
            "methods": ["Permutation Importance", "SHAP Global", "Feature Variance"],
            "use_cases": [
                "Understanding which features are most important overall",
                "Model validation and debugging",
                "Feature selection and engineering guidance",
            ],
            "outputs": [
                "Feature importance rankings",
                "Feature interactions",
                "Model summary",
            ],
            "limitations": [
                "May miss local patterns",
                "Averages can hide important details",
            ],
        },
        "bias": {
            "title": "Bias Analysis",
            "description": "Analyze model fairness across different demographic groups",
            "methods": ["Demographic Parity", "Equalized Odds", "Statistical Parity"],
            "use_cases": [
                "Ensuring fair treatment across protected groups",
                "Compliance with anti-discrimination regulations",
                "Identifying and mitigating algorithmic bias",
            ],
            "outputs": [
                "Fairness metrics",
                "Group statistics",
                "Bias severity assessment",
            ],
            "limitations": [
                "Requires labeled protected attributes",
                "May not catch all forms of bias",
            ],
        },
        "trust": {
            "title": "Trust Assessment",
            "description": "Evaluate reliability and trustworthiness of model predictions",
            "methods": [
                "Consistency Analysis",
                "Stability Testing",
                "Fidelity Assessment",
            ],
            "use_cases": [
                "Determining when to rely on model predictions",
                "Risk assessment for critical decisions",
                "Model monitoring and quality assurance",
            ],
            "outputs": ["Trust scores", "Risk levels", "Improvement suggestions"],
            "limitations": [
                "Computationally intensive",
                "May require domain expertise to interpret",
            ],
        },
    }

    info = info_data.get(explanation_type)
    if not info:
        console.print(
            f"No information available for explanation type: {explanation_type}",
            style="red",
        )
        return

    console.print(f"\n{info['title']}", style="bold")
    console.print(f"\n📝 Description: {info['description']}")

    console.print("\n🔧 Methods:")
    for method in info["methods"]:
        console.print(f"  • {method}")

    console.print("\n🎯 Use Cases:")
    for use_case in info["use_cases"]:
        console.print(f"  • {use_case}")

    console.print("\n📊 Outputs:")
    for output in info["outputs"]:
        console.print(f"  • {output}")

    console.print("\n⚠️ Limitations:")
    for limitation in info["limitations"]:
        console.print(f"  • {limitation}")


def _save_explanation_report(report, output_path: Path, format_type: str):
    """Save explanation report to file."""
    try:
        if format_type == "json":
            with open(output_path, "w") as f:
                # Convert report to dict for JSON serialization
                report_dict = {
                    "model_info": report.model_info,
                    "dataset_summary": report.dataset_summary,
                    "local_explanations": [
                        {
                            "sample_id": exp.sample_id,
                            "prediction": exp.prediction,
                            "confidence": exp.confidence,
                            "feature_contributions": exp.feature_contributions,
                            "explanation_method": exp.explanation_method,
                        }
                        for exp in report.local_explanations
                    ],
                    "global_explanation": (
                        {
                            "feature_importance": report.global_explanation.feature_importance,
                            "explanation_method": report.global_explanation.explanation_method,
                            "coverage": report.global_explanation.coverage,
                            "reliability": report.global_explanation.reliability,
                        }
                        if report.global_explanation
                        else None
                    ),
                    "trust_assessment": {
                        "overall_trust_score": report.trust_assessment.overall_trust_score,
                        "consistency_score": report.trust_assessment.consistency_score,
                        "stability_score": report.trust_assessment.stability_score,
                        "fidelity_score": report.trust_assessment.fidelity_score,
                        "risk_assessment": report.trust_assessment.risk_assessment,
                    },
                    "recommendations": report.recommendations,
                }
                json.dump(report_dict, f, indent=2, default=str)
        else:
            # For other formats, save as JSON for now
            _save_explanation_report(report, output_path, "json")

    except Exception as e:
        raise RuntimeError(f"Failed to save explanation report: {e}")


def _save_bias_analysis_results(results, output_path: Path):
    """Save bias analysis results to file."""
    bias_data = []
    for result in results:
        bias_data.append(
            {
                "protected_attribute": result.protected_attribute,
                "bias_detected": result.bias_detected,
                "severity": result.severity,
                "fairness_metrics": result.fairness_metrics,
                "group_statistics": result.group_statistics,
                "recommendations": result.recommendations,
            }
        )

    with open(output_path, "w") as f:
        json.dump(bias_data, f, indent=2, default=str)


def _save_trust_assessment_results(trust_result, output_path: Path):
    """Save trust assessment results to file."""
    trust_data = {
        "overall_trust_score": trust_result.overall_trust_score,
        "consistency_score": trust_result.consistency_score,
        "stability_score": trust_result.stability_score,
        "fidelity_score": trust_result.fidelity_score,
        "trust_factors": trust_result.trust_factors,
        "risk_assessment": trust_result.risk_assessment,
    }

    with open(output_path, "w") as f:
        json.dump(trust_data, f, indent=2, default=str)


def _save_feature_importance(importance: dict, output_path: Path):
    """Save feature importance to file."""
    with open(output_path, "w") as f:
        json.dump(importance, f, indent=2, default=str)


def _display_counterfactual_results(
    counterfactuals: dict, instance: list, feature_names: list
):
    """Display counterfactual explanation results."""
    console.print("\n🔄 Counterfactual Explanations Results", style="bold")

    if "error" in counterfactuals:
        console.print(f"❌ Error: {counterfactuals['error']}", style="red")
        return

    original_instance = counterfactuals.get("original_instance", instance)
    original_prediction = counterfactuals.get("original_prediction", 0)
    counterfactual_instances = counterfactuals.get("counterfactuals", [])

    console.print(f"📊 Original prediction: {original_prediction:.3f}")
    console.print(f"🔄 Generated {len(counterfactual_instances)} counterfactuals")

    if counterfactual_instances:
        # Display top counterfactuals
        console.print("\n🎯 Top Counterfactuals:")
        for i, cf in enumerate(counterfactual_instances[:3], 1):
            console.print(
                f"\n  {i}. Counterfactual (distance: {cf.get('distance', 0):.3f})"
            )
            console.print(f"     New prediction: {cf.get('prediction', 0):.3f}")
            console.print(
                f"     Prediction change: {cf.get('prediction_change', 0):.3f}"
            )

            # Show key differences
            differences = cf.get("differences", [])[:5]  # Show top 5 differences
            if differences:
                console.print("     Key changes:")
                for diff in differences:
                    console.print(
                        f"       {diff['feature_name']}: {diff['original_value']:.3f} → {diff['counterfactual_value']:.3f}"
                    )


def _save_counterfactual_results(counterfactuals: dict, output_path: Path):
    """Save counterfactual results to file."""
    with open(output_path, "w") as f:
        json.dump(counterfactuals, f, indent=2, default=str)


def _display_interaction_results(interactions: dict, method: str):
    """Display feature interaction analysis results."""
    console.print(f"\n🔍 {method.upper()} Feature Interaction Analysis", style="bold")

    if "error" in interactions:
        console.print(f"❌ Error: {interactions['error']}", style="red")
        return

    if not interactions:
        console.print("ℹ️ No interactions found or method not supported", style="yellow")
        return

    # Create interaction table
    table = Table(title="Top Feature Interactions")
    table.add_column("Rank", style="cyan")
    table.add_column("Feature Pair", style="white")
    table.add_column("Interaction Strength", style="green")
    table.add_column("Bar", style="blue")

    # Sort interactions by strength
    sorted_interactions = sorted(
        interactions.items(), key=lambda x: abs(x[1]), reverse=True
    )[:15]

    max_strength = (
        max(abs(strength) for _, strength in sorted_interactions)
        if sorted_interactions
        else 1.0
    )

    for rank, (feature_pair, strength) in enumerate(sorted_interactions, 1):
        # Create simple bar visualization
        bar_length = int(20 * abs(strength) / max_strength)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        table.add_row(str(rank), feature_pair, f"{strength:.4f}", bar)

    console.print(table)


def _save_interaction_results(interactions: dict, output_path: Path):
    """Save interaction analysis results to file."""
    with open(output_path, "w") as f:
        json.dump(interactions, f, indent=2, default=str)


def _display_dashboard_info(dashboard_data: dict, host: str, port: int):
    """Display dashboard information."""
    console.print("\n🎨 Interactive Explanation Dashboard", style="bold")

    # Dashboard summary
    summary_text = f"""
    Components Generated: {len(dashboard_data.get('components', []))}
    Local Explanations: {len(dashboard_data.get('local_explanations', []))}
    Global Explanations: {len(dashboard_data.get('global_explanations', []))}
    Feature Interactions: {len(dashboard_data.get('feature_interactions', {}))}
    Counterfactuals: {len(dashboard_data.get('counterfactuals', []))}
    """

    console.print(Panel(summary_text, title="Dashboard Summary", border_style="blue"))

    # Available features
    features = []
    if dashboard_data.get("local_explanations"):
        features.append("Local explanations with SHAP/LIME")
    if dashboard_data.get("global_explanations"):
        features.append("Global feature importance")
    if dashboard_data.get("feature_interactions"):
        features.append("Feature interaction analysis")
    if dashboard_data.get("counterfactuals"):
        features.append("Counterfactual explanations")
    if dashboard_data.get("bias_analysis"):
        features.append("Bias and fairness analysis")
    if dashboard_data.get("trust_assessment"):
        features.append("Trust and reliability assessment")

    console.print("\n📊 Available Features:")
    for feature in features:
        console.print(f"  • {feature}")

    console.print(f"\n🌐 Dashboard URL: http://{host}:{port}")
    console.print("📱 Interactive visualizations and explanations")


def _save_dashboard_data(dashboard_data: dict, output_path: Path):
    """Save dashboard data to directory."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Save main dashboard data
    with open(output_path / "dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    # Save individual components
    for component_name, component_data in dashboard_data.items():
        if isinstance(component_data, (dict, list)):
            with open(output_path / f"{component_name}.json", "w") as f:
                json.dump(component_data, f, indent=2, default=str)
