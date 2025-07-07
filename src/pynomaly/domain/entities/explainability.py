"""Explainability domain entities for anomaly detection interpretation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class ExplanationType(str, Enum):
    """Types of explanations for anomaly detection."""
    
    GLOBAL = "global"
    LOCAL = "local"
    COUNTERFACTUAL = "counterfactual"
    FEATURE_IMPORTANCE = "feature_importance"
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION = "permutation"
    ATTENTION = "attention"
    DECISION_TREE = "decision_tree"
    RULE_BASED = "rule_based"


class ExplanationScope(str, Enum):
    """Scope of the explanation."""
    
    INSTANCE = "instance"
    FEATURE = "feature"
    MODEL = "model"
    DATASET = "dataset"
    CLUSTER = "cluster"


class ConfidenceLevel(str, Enum):
    """Confidence levels for explanations."""
    
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class FeatureContribution:
    """Individual feature contribution to anomaly score."""
    
    feature_name: str
    contribution_score: float
    feature_value: Any
    baseline_value: Any | None = None
    rank: int | None = None
    confidence: float | None = None
    
    def __post_init__(self) -> None:
        """Validate feature contribution."""
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.rank is not None and self.rank < 1:
            raise ValueError("Rank must be positive")


@dataclass
class CounterfactualExample:
    """Counterfactual example for explanation."""
    
    original_values: dict[str, Any]
    counterfactual_values: dict[str, Any]
    original_score: float
    counterfactual_score: float
    distance: float
    changed_features: list[str]
    explanation: str | None = None
    
    def __post_init__(self) -> None:
        """Validate counterfactual example."""
        if not (0.0 <= self.original_score <= 1.0):
            raise ValueError("Original score must be between 0.0 and 1.0")
        if not (0.0 <= self.counterfactual_score <= 1.0):
            raise ValueError("Counterfactual score must be between 0.0 and 1.0")
        if self.distance < 0:
            raise ValueError("Distance must be non-negative")


@dataclass
class ExplanationRule:
    """Rule-based explanation for anomaly detection."""
    
    condition: str
    support: float
    confidence: float
    lift: float
    description: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate explanation rule."""
        if not (0.0 <= self.support <= 1.0):
            raise ValueError("Support must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.lift < 0:
            raise ValueError("Lift must be non-negative")


@dataclass
class SHAPValues:
    """SHAP (SHapley Additive exPlanations) values."""
    
    feature_values: dict[str, float]
    base_value: float
    prediction: float
    expected_value: float
    feature_names: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate SHAP values."""
        if not self.feature_names:
            self.feature_names = list(self.feature_values.keys())
        
        # Verify additivity property: sum(shap_values) + base_value ≈ prediction
        shap_sum = sum(self.feature_values.values()) + self.base_value
        tolerance = 1e-6
        if abs(shap_sum - self.prediction) > tolerance:
            raise ValueError(
                f"SHAP values do not satisfy additivity: "
                f"{shap_sum} != {self.prediction}"
            )

    def get_top_features(self, n: int = 5) -> list[tuple[str, float]]:
        """Get top N features by absolute SHAP value."""
        sorted_features = sorted(
            self.feature_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]


@dataclass
class ExplanationMetadata:
    """Metadata for explanations."""
    
    model_id: UUID
    model_version: str | None = None
    algorithm: str | None = None
    explainer_version: str | None = None
    computation_time: float | None = None
    memory_usage: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate explanation metadata."""
        if self.computation_time is not None and self.computation_time < 0:
            raise ValueError("Computation time must be non-negative")
        if self.memory_usage is not None and self.memory_usage < 0:
            raise ValueError("Memory usage must be non-negative")


@dataclass
class AnomalyExplanation:
    """Comprehensive explanation for anomaly detection result."""
    
    # Identity
    instance_id: str | UUID
    explanation_type: ExplanationType
    scope: ExplanationScope
    metadata: ExplanationMetadata
    
    # Core explanation data
    anomaly_score: float
    feature_contributions: list[FeatureContribution]
    
    # Auto-generated fields
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Optional explanation components
    shap_values: SHAPValues | None = None
    counterfactuals: list[CounterfactualExample] = field(default_factory=list)
    rules: list[ExplanationRule] = field(default_factory=list)
    
    # Text explanations
    summary: str | None = None
    detailed_explanation: str | None = None
    recommendations: list[str] = field(default_factory=list)
    
    # Confidence and quality metrics
    explanation_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    explanation_quality_score: float | None = None
    
    # Visualization data
    visualization_data: dict[str, Any] = field(default_factory=dict)
    
    # Context
    business_context: dict[str, Any] = field(default_factory=dict)
    technical_context: dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    tags: list[str] = field(default_factory=list)
    custom_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate anomaly explanation."""
        if not (0.0 <= self.anomaly_score <= 1.0):
            raise ValueError("Anomaly score must be between 0.0 and 1.0")
        
        if (self.explanation_quality_score is not None and 
            not (0.0 <= self.explanation_quality_score <= 1.0)):
            raise ValueError("Explanation quality score must be between 0.0 and 1.0")
        
        if not self.feature_contributions:
            raise ValueError("At least one feature contribution is required")

    def get_top_features(self, n: int = 5) -> list[FeatureContribution]:
        """Get top N features by contribution magnitude."""
        return sorted(
            self.feature_contributions,
            key=lambda x: abs(x.contribution_score),
            reverse=True
        )[:n]

    def get_positive_contributions(self) -> list[FeatureContribution]:
        """Get features with positive contributions to anomaly score."""
        return [fc for fc in self.feature_contributions if fc.contribution_score > 0]

    def get_negative_contributions(self) -> list[FeatureContribution]:
        """Get features with negative contributions to anomaly score."""
        return [fc for fc in self.feature_contributions if fc.contribution_score < 0]

    def get_total_contribution(self) -> float:
        """Get total contribution from all features."""
        return sum(fc.contribution_score for fc in self.feature_contributions)

    def get_explanation_strength(self) -> float:
        """Get explanation strength based on top features."""
        if not self.feature_contributions:
            return 0.0
        
        top_features = self.get_top_features(3)
        total_abs_contribution = sum(abs(fc.contribution_score) for fc in top_features)
        total_possible_contribution = sum(abs(fc.contribution_score) for fc in self.feature_contributions)
        
        if total_possible_contribution == 0:
            return 0.0
        
        return total_abs_contribution / total_possible_contribution

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation to the explanation."""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the explanation."""
        if tag not in self.tags:
            self.tags.append(tag)

    def is_high_confidence(self) -> bool:
        """Check if explanation has high confidence."""
        return self.explanation_confidence in [
            ConfidenceLevel.HIGH, 
            ConfidenceLevel.VERY_HIGH
        ]

    def requires_review(self) -> bool:
        """Check if explanation requires human review."""
        return (
            self.explanation_confidence in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]
            or self.get_explanation_strength() < 0.5
            or (self.explanation_quality_score is not None and self.explanation_quality_score < 0.6)
        )


@dataclass
class ExplanationQuery:
    """Query for requesting explanations."""
    
    instance_ids: list[str | UUID]
    explanation_types: list[ExplanationType]
    
    # Query parameters
    top_features: int = 5
    include_counterfactuals: bool = False
    include_rules: bool = False
    include_shap: bool = True
    
    # Filter parameters
    min_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    model_ids: list[UUID] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    
    # Pagination
    limit: int = 100
    offset: int = 0
    
    def __post_init__(self) -> None:
        """Validate explanation query."""
        if not self.instance_ids:
            raise ValueError("At least one instance ID must be provided")
        if not self.explanation_types:
            raise ValueError("At least one explanation type must be provided")
        if self.top_features < 1:
            raise ValueError("Top features must be at least 1")
        if self.limit < 1:
            raise ValueError("Limit must be at least 1")
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")


@dataclass
class ExplanationSummary:
    """Summary of explanation results."""
    
    total_explanations: int
    explanations_by_type: dict[str, int]
    explanations_by_confidence: dict[str, int]
    average_quality_score: float | None = None
    most_important_features: list[str] = field(default_factory=list)
    common_patterns: list[str] = field(default_factory=list)
    summary_generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate explanation summary."""
        if self.total_explanations < 0:
            raise ValueError("Total explanations must be non-negative")
        if (self.average_quality_score is not None and 
            not (0.0 <= self.average_quality_score <= 1.0)):
            raise ValueError("Average quality score must be between 0.0 and 1.0")


@dataclass
class ExplanationTemplate:
    """Template for generating text explanations."""
    
    name: str
    template_text: str
    explanation_type: ExplanationType
    required_fields: list[str]
    
    optional_fields: list[str] = field(default_factory=list)
    language: str = "en"
    format_type: str = "text"  # text, html, markdown
    
    def __post_init__(self) -> None:
        """Validate explanation template."""
        if not self.template_text.strip():
            raise ValueError("Template text cannot be empty")
        if not self.required_fields:
            raise ValueError("At least one required field must be specified")

    def can_generate(self, explanation: AnomalyExplanation) -> bool:
        """Check if template can generate explanation for given data."""
        # Check if all required fields are available
        explanation_dict = {
            "anomaly_score": explanation.anomaly_score,
            "top_feature": explanation.get_top_features(1)[0].feature_name if explanation.feature_contributions else None,
            "contribution_count": len(explanation.feature_contributions),
            "confidence": explanation.explanation_confidence.value,
            # Add more fields as needed
        }
        
        return all(
            field in explanation_dict and explanation_dict[field] is not None
            for field in self.required_fields
        )

    def generate_text(self, explanation: AnomalyExplanation) -> str:
        """Generate explanation text using template."""
        if not self.can_generate(explanation):
            raise ValueError("Cannot generate explanation: missing required fields")
        
        # Simple template substitution (in production, use a proper template engine)
        top_features = explanation.get_top_features(3)
        
        context = {
            "anomaly_score": f"{explanation.anomaly_score:.2f}",
            "top_feature": top_features[0].feature_name if top_features else "Unknown",
            "contribution_count": len(explanation.feature_contributions),
            "confidence": explanation.explanation_confidence.value.replace("_", " ").title(),
        }
        
        text = self.template_text
        for field, value in context.items():
            placeholder = f"{{{field}}}"
            text = text.replace(placeholder, str(value))
        
        return text