"""Configuration and data structures for autonomous anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pynomaly_detection.application.services.autonomous_preprocessing import DataQualityReport


@dataclass
class AutonomousConfig:
    """Configuration for autonomous detection."""

    max_samples_analysis: int = 10000
    confidence_threshold: float = 0.8
    max_algorithms: int = 5
    auto_tune_hyperparams: bool = True
    save_results: bool = True
    export_results: bool = False
    export_format: str = "csv"
    verbose: bool = False

    # Preprocessing configuration
    enable_preprocessing: bool = True
    quality_threshold: float = 0.8
    max_preprocessing_time: float = 300.0  # 5 minutes
    preprocessing_strategy: str = "auto"  # auto, aggressive, conservative, minimal

    # Explainability configuration
    enable_explainability: bool = True
    explain_algorithm_choices: bool = True
    explain_anomalies: bool = True
    explanation_method: str = "auto"  # auto, shap, lime, permutation


@dataclass
class AlgorithmExplanation:
    """Explanation for why an algorithm was selected or rejected."""

    algorithm: str
    selected: bool
    confidence: float
    reasoning: str
    data_characteristics: dict[str, Any]
    decision_factors: dict[str, float]
    alternatives_considered: list[str]
    performance_prediction: float
    computational_complexity: str
    memory_requirements: str
    interpretability_score: float


@dataclass
class AnomalyExplanation:
    """Explanation for detected anomalies."""

    sample_id: int
    anomaly_score: float
    contributing_features: dict[str, float]
    feature_importances: dict[str, float]
    normal_range_deviations: dict[str, float]
    similar_normal_samples: list[int]
    explanation_confidence: float
    explanation_method: str


@dataclass
class DataProfile:
    """Data profiling results."""

    n_samples: int
    n_features: int
    numeric_features: int
    categorical_features: int
    temporal_features: int
    missing_values_ratio: float
    data_types: dict[str, str]
    correlation_score: float
    sparsity_ratio: float
    outlier_ratio_estimate: float
    seasonality_detected: bool
    trend_detected: bool
    recommended_contamination: float
    complexity_score: float

    # Preprocessing-related fields
    quality_score: float = 1.0
    quality_report: DataQualityReport | None = None
    preprocessing_recommended: bool = False
    preprocessing_applied: bool = False
    preprocessing_metadata: dict[str, Any] | None = None


@dataclass
class AlgorithmRecommendation:
    """Algorithm recommendation with confidence."""

    algorithm: str
    confidence: float
    reasoning: str
    hyperparams: dict[str, Any]
    expected_performance: float


@dataclass
class AutonomousExplanationReport:
    """Comprehensive explanation report for autonomous detection."""

    dataset_profile: DataProfile
    algorithm_explanations: list[AlgorithmExplanation]
    selected_algorithms: list[str]
    rejected_algorithms: list[str]
    ensemble_explanation: str | None
    anomaly_explanations: list[AnomalyExplanation]
    processing_explanation: str
    recommendations: list[str]
    decision_tree: dict[str, Any]
