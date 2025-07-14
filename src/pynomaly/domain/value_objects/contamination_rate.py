"""Contamination rate value object."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from pynomaly.domain.exceptions import InvalidValueError, ValidationError


@dataclass(frozen=True)
class ContaminationRate:
    """Immutable value object representing contamination rate.

    Attributes:
        value: The contamination rate (0.0 to 0.5)
        metadata: Additional context about the contamination rate
        confidence_level: Confidence level in the rate estimate (0.0 to 1.0)
        source: Source of the contamination rate (e.g., 'domain_knowledge', 'statistical_analysis')
    """

    value: float
    metadata: dict[str, Any] = None
    confidence_level: float = 0.8
    source: str = "auto"

    # Class constants for common rates
    AUTO: ClassVar[ContaminationRate]
    LOW: ClassVar[ContaminationRate]
    MEDIUM: ClassVar[ContaminationRate]
    HIGH: ClassVar[ContaminationRate]

    def __post_init__(self) -> None:
        """Validate contamination rate with advanced business rules."""
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            # Make a deep copy to prevent mutation
            import copy
            object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

        # Basic validation
        self._validate_basic_constraints()

        # Advanced business rule validation
        self._validate_business_rules()

        # Statistical validation
        self._validate_statistical_properties()

        # Context validation
        self._validate_context_appropriateness()

    def _validate_basic_constraints(self) -> None:
        """Validate basic type and range constraints."""
        if not isinstance(self.value, (int, float)):
            raise InvalidValueError(
                f"Contamination rate must be numeric, got {type(self.value)}"
            )

        if math.isnan(self.value) or math.isinf(self.value):
            raise ValidationError(f"Contamination rate must be finite, got {self.value}")

        if not (0.0 <= self.value <= 0.5):
            raise InvalidValueError(
                f"Contamination rate must be between 0 and 0.5, got {self.value}"
            )

        # Validate confidence level
        if not isinstance(self.confidence_level, (int, float)):
            raise ValidationError(
                f"Confidence level must be numeric, got {type(self.confidence_level)}"
            )

        if not (0.0 <= self.confidence_level <= 1.0):
            raise ValidationError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )

        # Validate source
        if not isinstance(self.source, str):
            raise ValidationError(f"Source must be a string, got {type(self.source)}")

        if len(self.source.strip()) == 0:
            raise ValidationError("Source cannot be empty")

    def _validate_business_rules(self) -> None:
        """Validate advanced business rules for contamination rates."""
        # Validate against known contamination sources
        valid_sources = {
            "auto", "domain_knowledge", "statistical_analysis", "historical_data",
            "expert_opinion", "cross_validation", "grid_search", "heuristic"
        }

        if self.source.lower() not in valid_sources:
            raise ValidationError(
                f"Unknown contamination source: {self.source}. Valid sources: {valid_sources}"
            )

        # Domain-specific business rules
        if "domain" in self.metadata:
            domain = self.metadata["domain"]
            if isinstance(domain, str):
                self._validate_domain_specific_rules(domain)

        # Validate sample size dependency
        if "sample_size" in self.metadata:
            sample_size = self.metadata["sample_size"]
            if isinstance(sample_size, int) and sample_size > 0:
                self._validate_sample_size_relationship(sample_size)

        # Validate feature dimensionality impact
        if "feature_count" in self.metadata:
            feature_count = self.metadata["feature_count"]
            if isinstance(feature_count, int) and feature_count > 0:
                self._validate_dimensionality_impact(feature_count)

    def _validate_domain_specific_rules(self, domain: str) -> None:
        """Validate contamination rate for specific domains."""
        domain_lower = domain.lower()

        if domain_lower == "fraud_detection":
            # Fraud typically has very low contamination rates
            if self.value > 0.05:
                raise ValidationError(
                    f"Fraud detection contamination rate too high ({self.value}). "
                    f"Typical fraud rates are < 5%"
                )
        elif domain_lower == "network_security":
            # Network anomalies can be more frequent
            if self.value > 0.15:
                raise ValidationError(
                    f"Network security contamination rate too high ({self.value}). "
                    f"Typical network anomaly rates are < 15%"
                )
        elif domain_lower == "medical_diagnosis":
            # Medical anomalies require careful consideration
            if self.value > 0.10:
                raise ValidationError(
                    f"Medical diagnosis contamination rate too high ({self.value}). "
                    f"Medical anomaly rates typically < 10%"
                )
        elif domain_lower == "manufacturing":
            # Manufacturing defects can vary widely
            if self.value > 0.30:
                raise ValidationError(
                    f"Manufacturing contamination rate too high ({self.value}). "
                    f"Manufacturing defect rates typically < 30%"
                )

    def _validate_sample_size_relationship(self, sample_size: int) -> None:
        """Validate contamination rate relative to sample size."""
        expected_anomalies = int(sample_size * self.value)

        # Too few anomalies for reliable detection
        if expected_anomalies < 10:
            raise ValidationError(
                f"Sample size ({sample_size}) with contamination rate ({self.value}) "
                f"yields too few anomalies ({expected_anomalies}). Need at least 10 anomalies."
            )

        # Very small datasets need higher contamination rates
        if sample_size < 100 and self.value < 0.05:
            raise ValidationError(
                f"Small sample size ({sample_size}) requires higher contamination rate "
                f"for effective detection. Consider rate > 0.05"
            )

    def _validate_dimensionality_impact(self, feature_count: int) -> None:
        """Validate contamination rate considering feature dimensionality."""
        # High-dimensional data typically requires lower contamination rates
        if feature_count > 100:
            if self.value > 0.1:
                raise ValidationError(
                    f"High-dimensional data ({feature_count} features) typically requires "
                    f"lower contamination rates. Consider rate < 0.1"
                )

        # Curse of dimensionality impact
        if feature_count > 1000 and self.value > 0.05:
            raise ValidationError(
                f"Very high-dimensional data ({feature_count} features) may suffer from "
                f"curse of dimensionality. Consider rate < 0.05"
            )

    def _validate_statistical_properties(self) -> None:
        """Validate statistical properties of contamination rate."""
        # Validate precision
        if isinstance(self.value, float):
            decimal_places = len(str(self.value).split('.')[-1]) if '.' in str(self.value) else 0
            if decimal_places > 4:
                raise ValidationError(
                    f"Contamination rate has excessive precision ({decimal_places} decimal places)"
                )

        # Validate confidence relationship
        if self.confidence_level < 0.5 and self.value > 0.2:
            raise ValidationError(
                f"High contamination rate ({self.value}) with low confidence ({self.confidence_level}) "
                f"may indicate estimation uncertainty"
            )

        # Extremely low rates should have high confidence
        if self.value < 0.01 and self.confidence_level < 0.9:
            raise ValidationError(
                f"Very low contamination rate ({self.value}) requires high confidence "
                f"({self.confidence_level} < 0.9)"
            )

    def _validate_context_appropriateness(self) -> None:
        """Validate contamination rate appropriateness for detection context."""
        # Check for algorithm-specific constraints
        if "algorithm" in self.metadata:
            algorithm = self.metadata["algorithm"]
            if isinstance(algorithm, str):
                self._validate_algorithm_compatibility(algorithm)

        # Check for evaluation context
        if "evaluation_context" in self.metadata:
            context = self.metadata["evaluation_context"]
            if isinstance(context, str):
                self._validate_evaluation_context(context)

    def _validate_algorithm_compatibility(self, algorithm: str) -> None:
        """Validate contamination rate compatibility with specific algorithms."""
        algorithm_lower = algorithm.lower()

        # Isolation Forest works well with low contamination rates
        if "isolation" in algorithm_lower:
            if self.value > 0.3:
                raise ValidationError(
                    f"Isolation Forest typically performs better with contamination rates < 0.3, "
                    f"got {self.value}"
                )

        # LOF is sensitive to contamination rate
        if "lof" in algorithm_lower or "local_outlier" in algorithm_lower:
            if self.value > 0.2:
                raise ValidationError(
                    f"Local Outlier Factor is sensitive to contamination rates > 0.2, "
                    f"got {self.value}"
                )

    def _validate_evaluation_context(self, context: str) -> None:
        """Validate contamination rate for evaluation context."""
        context_lower = context.lower()

        if context_lower == "cross_validation":
            # Cross-validation requires consistent contamination rates
            if self.confidence_level < 0.7:
                raise ValidationError(
                    f"Cross-validation requires higher confidence in contamination rate "
                    f"({self.confidence_level} < 0.7)"
                )
        elif context_lower == "production":
            # Production environments need robust rates
            if self.value > 0.25:
                raise ValidationError(
                    f"Production contamination rate too high ({self.value}). "
                    f"Consider rate < 0.25 for production stability"
                )

    def is_valid(self) -> bool:
        """Check if the contamination rate is valid."""
        return isinstance(self.value, int | float) and 0.0 <= self.value <= 0.5

    def as_percentage(self) -> float:
        """Return contamination rate as a percentage (0-100)."""
        return self.value * 100.0

    def expected_anomalies(self, sample_size: int) -> int:
        """Calculate expected number of anomalies for given sample size."""
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValidationError(f"Sample size must be a positive integer, got {sample_size}")
        return int(sample_size * self.value)

    def validate_for_algorithm(self, algorithm: str) -> bool:
        """Validate contamination rate for specific algorithm."""
        try:
            self._validate_algorithm_compatibility(algorithm)
            return True
        except ValidationError:
            return False

    def get_domain_recommendation(self, domain: str) -> float:
        """Get recommended contamination rate for specific domain."""
        domain_lower = domain.lower()

        recommendations = {
            "fraud_detection": 0.02,
            "network_security": 0.10,
            "medical_diagnosis": 0.05,
            "manufacturing": 0.15,
            "finance": 0.03,
            "cybersecurity": 0.08,
            "quality_control": 0.12,
            "general": 0.10
        }

        return recommendations.get(domain_lower, 0.10)

    def confidence_assessment(self) -> dict[str, Any]:
        """Assess confidence in contamination rate estimate."""
        assessment = {
            "confidence_level": self.confidence_level,
            "source": self.source,
            "reliability": "unknown"
        }

        # Assess reliability based on source
        source_reliability = {
            "domain_knowledge": "high",
            "statistical_analysis": "high",
            "historical_data": "high",
            "expert_opinion": "medium",
            "cross_validation": "high",
            "grid_search": "medium",
            "heuristic": "low",
            "auto": "low"
        }

        assessment["reliability"] = source_reliability.get(self.source.lower(), "unknown")

        # Add recommendations
        if self.confidence_level < 0.6:
            assessment["recommendation"] = "Consider gathering more data or expert input"
        elif self.confidence_level < 0.8:
            assessment["recommendation"] = "Acceptable for most applications"
        else:
            assessment["recommendation"] = "High confidence - suitable for critical applications"

        return assessment

    def __str__(self) -> str:
        """String representation."""
        # Format as percentage, removing trailing zeros after decimal
        percentage = self.value * 100
        if percentage == int(percentage):
            return f"{int(percentage)}.0%"
        else:
            return f"{percentage:.1f}%"

    def __lt__(self, other) -> bool:
        """Less than comparison."""
        if isinstance(other, ContaminationRate):
            return self.value < other.value
        if isinstance(other, (int, float)):
            return self.value < other
        return NotImplemented

    def __le__(self, other) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, ContaminationRate):
            return self.value <= other.value
        if isinstance(other, (int, float)):
            return self.value <= other
        return NotImplemented

    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        if isinstance(other, ContaminationRate):
            return self.value > other.value
        if isinstance(other, (int, float)):
            return self.value > other
        return NotImplemented

    def __ge__(self, other) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, ContaminationRate):
            return self.value >= other.value
        if isinstance(other, (int, float)):
            return self.value >= other
        return NotImplemented

    @classmethod
    def auto(cls) -> ContaminationRate:
        """Create auto contamination rate (typically 0.1)."""
        return cls(0.1)

    @classmethod
    def low(cls) -> ContaminationRate:
        """Create low contamination rate."""
        return cls(0.05)

    @classmethod
    def medium(cls) -> ContaminationRate:
        """Create medium contamination rate."""
        return cls(0.1)

    @classmethod
    def high(cls) -> ContaminationRate:
        """Create high contamination rate."""
        return cls(0.2)


# Initialize class constants
ContaminationRate.AUTO = ContaminationRate(0.1)
ContaminationRate.LOW = ContaminationRate(0.05)
ContaminationRate.MEDIUM = ContaminationRate(0.1)
ContaminationRate.HIGH = ContaminationRate(0.2)
