"""Core explainability data structures and configuration."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExplanationConfig(BaseModel):
    """Configuration for explanation generation."""

    explanation_type: str = Field(
        default="local", description="Explanation type (local, global)"
    )
    method: str = Field(default="shap", description="Explanation method")
    n_samples: int = Field(
        default=1000, ge=10, description="Number of samples for explanation"
    )
    feature_names: list[str] | None = Field(None, description="Feature names")
    target_audience: str = Field(
        default="technical", description="Target audience level"
    )
    include_confidence: bool = Field(
        default=True, description="Include confidence measurements"
    )
    generate_plots: bool = Field(
        default=True, description="Generate visualization plots"
    )


class BiasAnalysisConfig(BaseModel):
    """Configuration for bias analysis."""

    protected_attributes: list[str] = Field(description="Protected attribute names")
    fairness_measurements: list[str] = Field(
        default=["demographic_parity", "equalized_odds"], description="Fairness measurements"
    )
    threshold: float = Field(default=0.5, description="Decision threshold")
    reference_group: str | None = Field(
        None, description="Reference group for comparison"
    )
    min_group_size: int = Field(
        default=50, description="Minimum group size for analysis"
    )


class TrustScoreConfig(BaseModel):
    """Configuration for trust scoring."""

    consistency_checks: bool = Field(
        default=True, description="Enable consistency analysis"
    )
    stability_analysis: bool = Field(
        default=True, description="Enable stability analysis"
    )
    fidelity_assessment: bool = Field(
        default=True, description="Enable fidelity assessment"
    )
    n_perturbations: int = Field(
        default=100, description="Number of perturbations for stability"
    )
    perturbation_strength: float = Field(
        default=0.1, description="Perturbation strength"
    )


class LocalExplanation(BaseModel):
    """Local explanation for individual predictions."""

    sample_id: str = Field(description="Sample identifier")
    prediction: float = Field(description="Processor prediction")
    confidence: float = Field(description="Prediction confidence")
    feature_contributions: dict[str, float] = Field(
        description="Feature contribution scores"
    )
    explanation_method: str = Field(description="Method used for explanation")
    trust_score: float | None = Field(None, description="Trust score for explanation")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class GlobalExplanation(BaseModel):
    """Global explanation for processor behavior."""

    feature_importance: dict[str, float] = Field(
        description="Global feature importance"
    )
    feature_interactions: dict[str, float] = Field(
        default_factory=dict, description="Feature interactions"
    )
    processor_summary: dict[str, Any] = Field(description="Processor summary statistics")
    explanation_method: str = Field(description="Method used for explanation")
    coverage: float = Field(description="Explanation coverage percentage")
    reliability: float = Field(description="Explanation reliability score")


class BiasAnalysisResult(BaseModel):
    """Results of bias analysis."""

    protected_attribute: str = Field(description="Protected attribute analyzed")
    fairness_measurements: dict[str, float] = Field(description="Fairness metric values")
    group_statistics: dict[str, dict[str, float]] = Field(
        description="Statistics by group"
    )
    bias_detected: bool = Field(description="Whether bias was detected")
    severity: str = Field(description="Bias severity level")
    recommendations: list[str] = Field(description="Mitigation recommendations")


class TrustScoreResult(BaseModel):
    """Trust score assessment results."""

    overall_trust_score: float = Field(description="Overall trust score (0-1)")
    consistency_score: float = Field(description="Processor consistency score")
    stability_score: float = Field(description="Prediction stability score")
    fidelity_score: float = Field(description="Explanation fidelity score")
    trust_factors: dict[str, float] = Field(description="Individual trust factors")
    risk_assessment: str = Field(description="Risk level assessment")


class ExplanationReport(BaseModel):
    """Comprehensive explanation report."""

    processor_info: dict[str, Any] = Field(description="Processor information")
    data_collection_summary: dict[str, Any] = Field(description="DataCollection summary")
    local_explanations: list[LocalExplanation] = Field(description="Local explanations")
    global_explanation: GlobalExplanation = Field(description="Global explanation")
    bias_analysis: list[BiasAnalysisResult] = Field(description="Bias analysis results")
    trust_assessment: TrustScoreResult = Field(description="Trust score assessment")
    recommendations: list[str] = Field(description="Overall recommendations")
    generation_time: datetime = Field(
        default_factory=datetime.now, description="Report generation time"
    )
