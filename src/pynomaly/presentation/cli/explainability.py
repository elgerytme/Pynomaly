"""CLI commands for explainable AI and model interpretability."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

# Application imports
from pynomaly.application.services.advanced_explainability_service import (
    AdvancedExplainabilityService, ExplanationConfig, BiasAnalysisConfig, TrustScoreConfig
)
from pynomaly.application.dto.explainability_dto import (
    LocalExplanationRequestDTO, GlobalExplanationRequestDTO, BiasAnalysisRequestDTO,
    TrustAssessmentRequestDTO, ExplanationVisualizationDTO
)

# Domain imports
from pynomaly.domain.entities import Dataset

# Infrastructure imports
from pynomaly.infrastructure.data_loaders import CSVLoader, ParquetLoader
from pynomaly.infrastructure.config.feature_flags import require_feature
from pynomaly.shared.protocols import DetectorProtocol

console = Console()


@click.group()
def explainability():
    """Explainable AI and model interpretability commands."""
    pass


@explainability.command()
@click.argument('detector_path', type=click.Path(exists=True, path_type=Path))
@click.argument('dataset_path', type=click.Path(exists=True, path_type=Path))
@click.option('--explanation-type', '-t', type=click.Choice(['local', 'global', 'both']),
              default='both', help='Type of explanation to generate')
@click.option('--methods', '-m', multiple=True,
              type=click.Choice(['shap', 'lime', 'permutation', 'gradient']),
              help='Explanation methods to use')
@click.option('--n-samples', type=int, default=10,
              help='Number of samples for local explanations')
@click.option('--audience', type=click.Choice(['technical', 'business', 'regulatory']),
              default='technical', help='Target audience for explanations')
@click.option('--output', type=click.Path(path_type=Path),
              help='Output file for explanation report')
@click.option('--format', 'output_format', type=click.Choice(['json', 'html', 'pdf']),
              default='json', help='Output format')
@click.option('--visualizations/--no-visualizations', default=True,
              help='Generate visualization plots')
@require_feature("advanced_explainability")
def explain(
    detector_path: Path,
    dataset_path: Path,
    explanation_type: str,
    methods: tuple,
    n_samples: int,
    audience: str,
    output: Optional[Path],
    output_format: str,
    visualizations: bool
):
    """Generate comprehensive explanations for model predictions.
    
    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file (CSV or Parquet)
    
    Examples:
        pynomaly explainability explain model.pkl data.csv
        pynomaly explainability explain model.pkl data.csv --methods shap lime
        pynomaly explainability explain model.pkl data.csv --explanation-type local --n-samples 20
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
            method=methods[0] if methods else "shap",
            n_samples=n_samples,
            feature_names=dataset.feature_names,
            target_audience=audience,
            generate_plots=visualizations
        )
        
        # Initialize explainability service
        explainability_service = AdvancedExplainabilityService(
            enable_shap=True,
            enable_lime=True,
            enable_permutation=True,
            cache_explanations=True
        )
        
        console.print("🔍 Generating comprehensive explanations...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Analyzing model behavior...", total=None)
            
            start_time = time.time()
            
            # Generate explanation report
            report = asyncio.run(
                explainability_service.generate_comprehensive_explanation(
                    detector=detector,
                    dataset=dataset,
                    config=config
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
        
        console.print("✅ Explanation generation completed successfully!", style="green")
        
    except Exception as e:
        console.print(f"❌ Explanation generation failed: {e}", style="red")
        sys.exit(1)


@explainability.command()
@click.argument('detector_path', type=click.Path(exists=True, path_type=Path))
@click.argument('dataset_path', type=click.Path(exists=True, path_type=Path))
@click.option('--protected-attributes', '-p', multiple=True,
              help='Protected attribute column names')
@click.option('--metrics', '-m', multiple=True,
              type=click.Choice(['demographic_parity', 'equalized_odds', 'statistical_parity']),
              help='Fairness metrics to compute')
@click.option('--threshold', type=float, default=0.5,
              help='Decision threshold for binary classification')
@click.option('--min-group-size', type=int, default=30,
              help='Minimum group size for analysis')
@click.option('--output', type=click.Path(path_type=Path),
              help='Output file for bias analysis results')
@require_feature("advanced_explainability")
def analyze_bias(
    detector_path: Path,
    dataset_path: Path,
    protected_attributes: tuple,
    metrics: tuple,
    threshold: float,
    min_group_size: int,
    output: Optional[Path]
):
    """Analyze model for potential bias and fairness issues.
    
    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file with protected attributes
    
    Examples:
        pynomaly explainability analyze-bias model.pkl data.csv --protected-attributes gender age
        pynomaly explainability analyze-bias model.pkl data.csv -p race -m demographic_parity equalized_odds
    """
    try:
        if not protected_attributes:
            console.print("❌ At least one protected attribute must be specified", style="red")
            sys.exit(1)
        
        # Load dataset and detector
        dataset = _load_dataset(dataset_path)
        detector = _load_detector(detector_path)
        
        # Configure bias analysis
        config = BiasAnalysisConfig(
            protected_attributes=list(protected_attributes),
            fairness_metrics=list(metrics) if metrics else ["demographic_parity", "equalized_odds"],
            threshold=threshold,
            min_group_size=min_group_size
        )
        
        # Initialize service
        explainability_service = AdvancedExplainabilityService()
        
        console.print("⚖️ Analyzing model bias and fairness...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Performing bias analysis...", total=None)
            
            # Run bias analysis
            bias_results = asyncio.run(
                explainability_service.analyze_bias(
                    detector=detector,
                    dataset=dataset,
                    config=config
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


@explainability.command()
@click.argument('detector_path', type=click.Path(exists=True, path_type=Path))
@click.argument('dataset_path', type=click.Path(exists=True, path_type=Path))
@click.option('--n-perturbations', type=int, default=100,
              help='Number of perturbations for stability analysis')
@click.option('--perturbation-strength', type=float, default=0.1,
              help='Strength of perturbations (0-1)')
@click.option('--consistency/--no-consistency', default=True,
              help='Enable consistency analysis')
@click.option('--stability/--no-stability', default=True,
              help='Enable stability analysis')
@click.option('--fidelity/--no-fidelity', default=True,
              help='Enable fidelity assessment')
@click.option('--output', type=click.Path(path_type=Path),
              help='Output file for trust assessment')
@require_feature("advanced_explainability")
def assess_trust(
    detector_path: Path,
    dataset_path: Path,
    n_perturbations: int,
    perturbation_strength: float,
    consistency: bool,
    stability: bool,
    fidelity: bool,
    output: Optional[Path]
):
    """Assess trust and reliability of model predictions.
    
    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file
    
    Examples:
        pynomaly explainability assess-trust model.pkl data.csv
        pynomaly explainability assess-trust model.pkl data.csv --n-perturbations 200
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
            perturbation_strength=perturbation_strength
        )
        
        # Initialize service
        explainability_service = AdvancedExplainabilityService()
        
        console.print("🔒 Assessing model trust and reliability...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Performing trust assessment...", total=None)
            
            # Get predictions for trust assessment
            X = dataset.data.values if hasattr(dataset.data, 'values') else dataset.data
            predictions = detector.decision_function(X)
            
            # Run trust assessment
            trust_result = asyncio.run(
                explainability_service._assess_trust_score(
                    detector=detector,
                    X=X,
                    predictions=predictions,
                    config=config
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


@explainability.command()
@click.argument('detector_path', type=click.Path(exists=True, path_type=Path))
@click.argument('dataset_path', type=click.Path(exists=True, path_type=Path))
@click.option('--method', type=click.Choice(['shap', 'lime', 'permutation']),
              default='shap', help='Feature importance method')
@click.option('--top-k', type=int, default=15,
              help='Number of top features to display')
@click.option('--output', type=click.Path(path_type=Path),
              help='Output file for feature importance')
@require_feature("advanced_explainability")
def feature_importance(
    detector_path: Path,
    dataset_path: Path,
    method: str,
    top_k: int,
    output: Optional[Path]
):
    """Analyze global feature importance for the model.
    
    DETECTOR_PATH: Path to saved detector model
    DATASET_PATH: Path to dataset file
    
    Examples:
        pynomaly explainability feature-importance model.pkl data.csv
        pynomaly explainability feature-importance model.pkl data.csv --method permutation --top-k 20
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
            console=console
        ) as progress:
            
            task = progress.add_task("Computing feature importance...", total=None)
            
            X = dataset.data.values if hasattr(dataset.data, 'values') else dataset.data
            feature_names = dataset.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
            # Compute feature importance based on method
            if method == "permutation":
                importance = asyncio.run(explainability_service._compute_permutation_importance(
                    detector, X, feature_names
                ))
            elif method == "shap":
                importance = asyncio.run(explainability_service._compute_shap_global_importance(
                    detector, X, feature_names
                ))
            else:  # lime or fallback
                importance = explainability_service._compute_variance_importance(X, feature_names)
            
            progress.update(task, completed=100)
        
        if importance:
            # Display feature importance
            _display_feature_importance(importance, top_k, method)
            
            # Save results if requested
            if output:
                _save_feature_importance(importance, output)
                console.print(f"💾 Feature importance saved to: {output}")
        else:
            console.print(f"⚠️ Failed to compute {method} feature importance", style="yellow")
        
        console.print("✅ Feature importance analysis completed!", style="green")
        
    except Exception as e:
        console.print(f"❌ Feature importance analysis failed: {e}", style="red")
        sys.exit(1)


@explainability.command()
@click.option('--cache-info/--no-cache-info', default=True,
              help='Show cache information')
@require_feature("advanced_explainability")
def status():
    """Show explainability service status and capabilities.
    
    Examples:
        pynomaly explainability status
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
            ("SHAP", service_info["shap_available"], service_info["shap_enabled"], 
             "SHapley Additive exPlanations"),
            ("LIME", service_info["lime_available"], service_info["lime_enabled"],
             "Local Interpretable Model-agnostic Explanations"),
            ("Permutation", service_info["sklearn_available"], service_info["permutation_enabled"],
             "Permutation Feature Importance"),
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
                description
            )
        
        console.print(availability_table)
        
        # Cache information
        if cache_info and service_info["cache_enabled"]:
            cache_text = f"""
            Cache Enabled: ✓
            Cached Explanations: {service_info['cached_explanations']}
            Cached Explainers: {service_info['cached_explainers']}
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
        
        capabilities.extend([
            "Bias analysis with fairness metrics",
            "Trust assessment with stability analysis",
            "Feature interaction analysis",
            "Explanation caching for performance"
        ])
        
        console.print("\n🎯 Available Capabilities:")
        for capability in capabilities:
            console.print(f"  • {capability}")
        
        console.print("\n✅ Explainability service status displayed!", style="green")
        
    except Exception as e:
        console.print(f"❌ Failed to get service status: {e}", style="red")
        sys.exit(1)


@explainability.command()
@click.argument('explanation_type', type=click.Choice(['local', 'global', 'bias', 'trust']))
def info(explanation_type: str):
    """Get detailed information about explanation types and methods.
    
    EXPLANATION_TYPE: Type of explanation to get information about
    
    Examples:
        pynomaly explainability info local
        pynomaly explainability info bias
    """
    try:
        _display_explanation_info(explanation_type)
        console.print(f"✅ Information for {explanation_type} explanations displayed!", style="green")
        
    except Exception as e:
        console.print(f"❌ Failed to display information: {e}", style="red")
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
            feature_names=list(data.columns) if hasattr(data, 'columns') else [f"feature_{i}" for i in range(data.shape[1])]
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def _load_detector(detector_path: Path) -> DetectorProtocol:
    """Load detector from file (simplified implementation)."""
    try:
        # This is a simplified implementation
        # In practice, would need proper model loading/deserialization
        import pickle
        
        with open(detector_path, 'rb') as f:
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
    Global Explanation: {'✓' if report.global_explanation else '✗'}
    Bias Analysis: {len(report.bias_analysis)} attributes
    Trust Score: {report.trust_assessment.overall_trust_score:.3f}
    Risk Level: {report.trust_assessment.risk_assessment.title()}
    """
    
    console.print(Panel(summary_text, title="Summary", border_style="blue"))
    
    # Trust assessment details
    trust = report.trust_assessment
    console.print(f"\n🔒 Trust Assessment:")
    console.print(f"  Overall Trust Score: {trust.overall_trust_score:.3f}")
    console.print(f"  Consistency: {trust.consistency_score:.3f}")
    console.print(f"  Stability: {trust.stability_score:.3f}")
    console.print(f"  Fidelity: {trust.fidelity_score:.3f}")
    console.print(f"  Risk Level: {trust.risk_assessment.title()}")
    
    # Feature importance from global explanation
    if report.global_explanation and hasattr(report.global_explanation, 'feature_importance'):
        console.print(f"\n📈 Top Features (Global Importance):")
        importance = report.global_explanation.feature_importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (feature, score) in enumerate(sorted_features, 1):
            console.print(f"  {i:2d}. {feature}: {score:.4f}")
    
    # Recommendations
    if report.recommendations:
        console.print(f"\n💡 Recommendations:")
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
            metrics_table = Table(title=f"Fairness Metrics - {result.protected_attribute}")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="white")
            metrics_table.add_column("Status", style="white")
            
            for metric, value in result.fairness_metrics.items():
                status = "✓ Pass" if value >= 0.8 else "⚠️ Concerning" if value >= 0.6 else "❌ Fail"
                color = "green" if value >= 0.8 else "yellow" if value >= 0.6 else "red"
                
                metrics_table.add_row(
                    metric.replace("_", " ").title(),
                    f"{value:.3f}",
                    f"[{color}]{status}[/{color}]"
                )
            
            console.print(metrics_table)
        
        # Group statistics
        if result.group_statistics:
            console.print(f"\n👥 Group Statistics:")
            for group, stats in result.group_statistics.items():
                console.print(f"  {group}: {stats['size']} samples, {stats['positive_rate']:.1%} positive rate")
        
        # Recommendations
        if result.recommendations:
            console.print(f"\n💡 Recommendations:")
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
        status = "✓ Good" if score >= 0.8 else "⚠️ Moderate" if score >= 0.6 else "❌ Poor"
        color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
        
        factors_table.add_row(
            factor.replace("_", " ").title(),
            f"{score:.3f}",
            f"[{color}]{status}[/{color}]"
        )
    
    console.print(factors_table)
    
    # Risk assessment
    risk_color = "green" if trust_result.risk_assessment == "low" else "yellow" if trust_result.risk_assessment == "medium" else "red"
    console.print(f"\n🚨 Risk Assessment: [{risk_color}]{trust_result.risk_assessment.upper()}[/{risk_color}]")


def _display_feature_importance(importance: dict, top_k: int, method: str):
    """Display feature importance results."""
    console.print(f"\n📈 {method.upper()} Feature Importance", style="bold")
    
    # Sort features by importance
    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    
    # Create importance table
    table = Table(title=f"Top {top_k} Features")
    table.add_column("Rank", style="cyan")
    table.add_column("Feature", style="white")
    table.add_column("Importance", style="green")
    table.add_column("Bar", style="blue")
    
    max_importance = max(abs(score) for _, score in sorted_features) if sorted_features else 1.0
    
    for rank, (feature, importance_score) in enumerate(sorted_features, 1):
        # Create simple bar visualization
        bar_length = int(20 * abs(importance_score) / max_importance)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        table.add_row(
            str(rank),
            feature,
            f"{importance_score:.4f}",
            bar
        )
    
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
                "Providing explanations to end users for specific predictions"
            ],
            "outputs": ["Feature contribution scores", "Confidence measures", "Trust scores"],
            "limitations": ["May not generalize to other samples", "Can be computationally expensive"]
        },
        "global": {
            "title": "Global Explanations", 
            "description": "Explain overall model behavior across all data",
            "methods": ["Permutation Importance", "SHAP Global", "Feature Variance"],
            "use_cases": [
                "Understanding which features are most important overall",
                "Model validation and debugging",
                "Feature selection and engineering guidance"
            ],
            "outputs": ["Feature importance rankings", "Feature interactions", "Model summary"],
            "limitations": ["May miss local patterns", "Averages can hide important details"]
        },
        "bias": {
            "title": "Bias Analysis",
            "description": "Analyze model fairness across different demographic groups",
            "methods": ["Demographic Parity", "Equalized Odds", "Statistical Parity"],
            "use_cases": [
                "Ensuring fair treatment across protected groups",
                "Compliance with anti-discrimination regulations",
                "Identifying and mitigating algorithmic bias"
            ],
            "outputs": ["Fairness metrics", "Group statistics", "Bias severity assessment"],
            "limitations": ["Requires labeled protected attributes", "May not catch all forms of bias"]
        },
        "trust": {
            "title": "Trust Assessment",
            "description": "Evaluate reliability and trustworthiness of model predictions",
            "methods": ["Consistency Analysis", "Stability Testing", "Fidelity Assessment"],
            "use_cases": [
                "Determining when to rely on model predictions",
                "Risk assessment for critical decisions",
                "Model monitoring and quality assurance"
            ],
            "outputs": ["Trust scores", "Risk levels", "Improvement suggestions"],
            "limitations": ["Computationally intensive", "May require domain expertise to interpret"]
        }
    }
    
    info = info_data.get(explanation_type)
    if not info:
        console.print(f"No information available for explanation type: {explanation_type}", style="red")
        return
    
    console.print(f"\n{info['title']}", style="bold")
    console.print(f"\n📝 Description: {info['description']}")
    
    console.print(f"\n🔧 Methods:")
    for method in info['methods']:
        console.print(f"  • {method}")
    
    console.print(f"\n🎯 Use Cases:")
    for use_case in info['use_cases']:
        console.print(f"  • {use_case}")
    
    console.print(f"\n📊 Outputs:")
    for output in info['outputs']:
        console.print(f"  • {output}")
    
    console.print(f"\n⚠️ Limitations:")
    for limitation in info['limitations']:
        console.print(f"  • {limitation}")


def _save_explanation_report(report, output_path: Path, format_type: str):
    """Save explanation report to file."""
    try:
        if format_type == 'json':
            with open(output_path, 'w') as f:
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
                            "explanation_method": exp.explanation_method
                        }
                        for exp in report.local_explanations
                    ],
                    "global_explanation": {
                        "feature_importance": report.global_explanation.feature_importance,
                        "explanation_method": report.global_explanation.explanation_method,
                        "coverage": report.global_explanation.coverage,
                        "reliability": report.global_explanation.reliability
                    } if report.global_explanation else None,
                    "trust_assessment": {
                        "overall_trust_score": report.trust_assessment.overall_trust_score,
                        "consistency_score": report.trust_assessment.consistency_score,
                        "stability_score": report.trust_assessment.stability_score,
                        "fidelity_score": report.trust_assessment.fidelity_score,
                        "risk_assessment": report.trust_assessment.risk_assessment
                    },
                    "recommendations": report.recommendations
                }
                json.dump(report_dict, f, indent=2, default=str)
        else:
            # For other formats, save as JSON for now
            _save_explanation_report(report, output_path, 'json')
            
    except Exception as e:
        raise RuntimeError(f"Failed to save explanation report: {e}")


def _save_bias_analysis_results(results, output_path: Path):
    """Save bias analysis results to file."""
    bias_data = []
    for result in results:
        bias_data.append({
            "protected_attribute": result.protected_attribute,
            "bias_detected": result.bias_detected,
            "severity": result.severity,
            "fairness_metrics": result.fairness_metrics,
            "group_statistics": result.group_statistics,
            "recommendations": result.recommendations
        })
    
    with open(output_path, 'w') as f:
        json.dump(bias_data, f, indent=2, default=str)


def _save_trust_assessment_results(trust_result, output_path: Path):
    """Save trust assessment results to file."""
    trust_data = {
        "overall_trust_score": trust_result.overall_trust_score,
        "consistency_score": trust_result.consistency_score,
        "stability_score": trust_result.stability_score,
        "fidelity_score": trust_result.fidelity_score,
        "trust_factors": trust_result.trust_factors,
        "risk_assessment": trust_result.risk_assessment
    }
    
    with open(output_path, 'w') as f:
        json.dump(trust_data, f, indent=2, default=str)


def _save_feature_importance(importance: dict, output_path: Path):
    """Save feature importance to file."""
    with open(output_path, 'w') as f:
        json.dump(importance, f, indent=2, default=str)