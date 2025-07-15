"""Advanced anomaly classification value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnomalySubType(str, Enum):
    """Detailed anomaly sub-types for advanced classification."""
    
    # Point anomaly subtypes
    OUTLIER = "outlier"
    EXTREME_VALUE = "extreme_value"
    NOVELTY = "novelty"
    
    # Contextual anomaly subtypes
    CONDITIONAL = "conditional"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    
    # Collective anomaly subtypes
    SEQUENCE = "sequence"
    PATTERN = "pattern"
    CLUSTER = "cluster"
    
    # Global anomaly subtypes
    SYSTEM_WIDE = "system_wide"
    TREND = "trend"
    SEASONAL = "seasonal"
    
    # Local anomaly subtypes
    REGIONAL = "regional"
    NEIGHBORHOOD = "neighborhood"
    LOCALIZED = "localized"


class ClassificationMethod(str, Enum):
    """Classification methods for anomaly detection."""
    
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"


class ConfidenceLevel(str, Enum):
    """Confidence levels for classification decisions."""
    
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass(frozen=True)
class ClassificationResult:
    """Result of anomaly classification with confidence and metadata."""
    
    predicted_class: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    probability_distribution: dict[str, float] = field(default_factory=dict)
    feature_contributions: dict[str, float] = field(default_factory=dict)
    classification_method: ClassificationMethod = ClassificationMethod.UNSUPERVISED
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate classification result."""
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        
        if self.probability_distribution:
            prob_sum = sum(self.probability_distribution.values())
            if not (0.99 <= prob_sum <= 1.01):  # Allow small floating point errors
                raise ValueError("Probability distribution must sum to 1.0")
        
        if self.feature_contributions:
            for feature, contribution in self.feature_contributions.items():
                if not isinstance(contribution, (int, float)):
                    raise ValueError(f"Feature contribution for {feature} must be numeric")
    
    @classmethod
    def from_confidence_score(cls, predicted_class: str, confidence_score: float) -> ClassificationResult:
        """Create result from confidence score."""
        if confidence_score >= 0.9:
            level = ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW
        
        return cls(
            predicted_class=predicted_class,
            confidence_score=confidence_score,
            confidence_level=level
        )
    
    def is_confident(self) -> bool:
        """Check if classification is confident."""
        return self.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
    
    def requires_review(self) -> bool:
        """Check if classification requires manual review."""
        return self.confidence_level in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]


@dataclass(frozen=True)
class HierarchicalClassification:
    """Hierarchical classification with parent-child relationships."""
    
    primary_category: str
    secondary_category: str | None = None
    tertiary_category: str | None = None
    sub_type: AnomalySubType | None = None
    confidence_scores: dict[str, float] = field(default_factory=dict)
    hierarchy_path: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate hierarchical classification."""
        if not self.primary_category:
            raise ValueError("Primary category cannot be empty")
        
        # Build hierarchy path
        path = [self.primary_category]
        if self.secondary_category:
            path.append(self.secondary_category)
        if self.tertiary_category:
            path.append(self.tertiary_category)
        if self.sub_type:
            path.append(self.sub_type.value)
        
        object.__setattr__(self, 'hierarchy_path', path)
    
    def get_hierarchy_depth(self) -> int:
        """Get depth of the hierarchy."""
        return len(self.hierarchy_path)
    
    def get_full_path(self) -> str:
        """Get full hierarchical path as string."""
        return " > ".join(self.hierarchy_path)
    
    def is_child_of(self, parent_path: str) -> bool:
        """Check if this classification is a child of given parent path."""
        return self.get_full_path().startswith(parent_path)


@dataclass(frozen=True)
class MultiClassClassification:
    """Multi-class classification result with multiple possible classes."""
    
    primary_result: ClassificationResult
    alternative_results: list[ClassificationResult] = field(default_factory=list)
    classification_threshold: float = 0.5
    multi_class_strategy: str = "one_vs_rest"
    
    def __post_init__(self) -> None:
        """Validate multi-class classification."""
        if not (0.0 <= self.classification_threshold <= 1.0):
            raise ValueError("Classification threshold must be between 0.0 and 1.0")
        
        valid_strategies = ["one_vs_rest", "one_vs_one", "multi_class_direct"]
        if self.multi_class_strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")
    
    def get_all_results(self) -> list[ClassificationResult]:
        """Get all classification results."""
        return [self.primary_result] + self.alternative_results
    
    def get_confident_results(self) -> list[ClassificationResult]:
        """Get only confident classification results."""
        return [result for result in self.get_all_results() if result.is_confident()]
    
    def has_ambiguous_classification(self) -> bool:
        """Check if classification is ambiguous (multiple high-confidence results)."""
        confident_results = self.get_confident_results()
        return len(confident_results) > 1
    
    def get_top_n_results(self, n: int) -> list[ClassificationResult]:
        """Get top N classification results by confidence."""
        all_results = self.get_all_results()
        return sorted(all_results, key=lambda x: x.confidence_score, reverse=True)[:n]


@dataclass(frozen=True)
class AdvancedAnomalyClassification:
    """Advanced anomaly classification with multiple dimensions."""
    
    basic_classification: ClassificationResult
    hierarchical_classification: HierarchicalClassification | None = None
    multi_class_classification: MultiClassClassification | None = None
    severity_classification: str = "medium"
    context_classification: dict[str, Any] = field(default_factory=dict)
    temporal_classification: dict[str, Any] = field(default_factory=dict)
    spatial_classification: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate advanced classification."""
        valid_severities = ["low", "medium", "high", "critical"]
        if self.severity_classification not in valid_severities:
            raise ValueError(f"Severity must be one of: {valid_severities}")
    
    def get_primary_class(self) -> str:
        """Get primary classification class."""
        return self.basic_classification.predicted_class
    
    def get_confidence_score(self) -> float:
        """Get primary confidence score."""
        return self.basic_classification.confidence_score
    
    def is_hierarchical(self) -> bool:
        """Check if hierarchical classification is available."""
        return self.hierarchical_classification is not None
    
    def is_multi_class(self) -> bool:
        """Check if multi-class classification is available."""
        return self.multi_class_classification is not None
    
    def has_spatial_context(self) -> bool:
        """Check if spatial classification context is available."""
        return bool(self.spatial_classification)
    
    def has_temporal_context(self) -> bool:
        """Check if temporal classification context is available."""
        return bool(self.temporal_classification)
    
    def get_full_classification_summary(self) -> dict[str, Any]:
        """Get comprehensive classification summary."""
        summary = {
            "primary_class": self.get_primary_class(),
            "confidence_score": self.get_confidence_score(),
            "confidence_level": self.basic_classification.confidence_level.value,
            "severity": self.severity_classification,
            "classification_method": self.basic_classification.classification_method.value,
        }
        
        if self.is_hierarchical():
            summary["hierarchical_path"] = self.hierarchical_classification.get_full_path()
            summary["hierarchy_depth"] = self.hierarchical_classification.get_hierarchy_depth()
        
        if self.is_multi_class():
            summary["alternative_classes"] = [
                result.predicted_class for result in self.multi_class_classification.alternative_results
            ]
            summary["has_ambiguous_classification"] = self.multi_class_classification.has_ambiguous_classification()
        
        if self.has_spatial_context():
            summary["spatial_context"] = self.spatial_classification
        
        if self.has_temporal_context():
            summary["temporal_context"] = self.temporal_classification
        
        return summary
    
    def requires_escalation(self) -> bool:
        """Check if classification requires escalation."""
        return (
            self.severity_classification in ["high", "critical"]
            or self.basic_classification.requires_review()
            or (self.is_multi_class() and self.multi_class_classification.has_ambiguous_classification())
        )