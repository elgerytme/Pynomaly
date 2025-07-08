"""Anomaly score value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

from pynomaly.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class AnomalyScore:
    """Immutable value object representing an anomaly score.

    Attributes:
        value: The anomaly score value (higher means more anomalous)
        confidence_interval: Confidence interval for the score (optional)
        method: The scoring method used (optional)
    """

    value: float
    confidence_interval: ConfidenceInterval | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        """Validate score after initialization with advanced business rules."""
        self._validate_basic_properties()
        self._validate_business_rules()
        self._validate_confidence_interval_consistency()
        self._validate_method_compatibility()

    def _validate_basic_properties(self) -> None:
        """Validate basic properties of the score."""
        if not isinstance(self.value, int | float):
            raise InvalidValueError(
                f"Score value must be numeric, got {type(self.value)}",
                field="value",
                value=self.value
            )

        if math.isnan(self.value) or math.isinf(self.value):
            raise InvalidValueError(
                f"Score value cannot be NaN or infinite, got {self.value}",
                field="value",
                value=self.value
            )

        if not (0.0 <= self.value <= 1.0):
            raise InvalidValueError(
                f"Score value must be between 0 and 1, got {self.value}",
                field="value",
                value=self.value
            )

    def _validate_business_rules(self) -> None:
        """Validate advanced business rules for anomaly scores."""
        # Business Rule: Scores below 0.01 should be treated as noise
        if 0.0 < self.value < 0.01:
            raise InvalidValueError(
                "Score values between 0 and 0.01 are likely noise and should be normalized to 0.0",
                field="value",
                value=self.value,
                rule="noise_threshold"
            )

        # Business Rule: Exact 1.0 scores should be rare and verified
        if self.value == 1.0 and self.confidence_interval is None:
            raise InvalidValueError(
                "Perfect anomaly scores (1.0) must include confidence intervals for validation",
                field="value",
                value=self.value,
                rule="perfect_score_validation"
            )

        # Business Rule: High precision scores should have method documentation
        if self.value > 0.999 and self.method is None:
            raise InvalidValueError(
                "High precision scores (>0.999) must specify the scoring method",
                field="method",
                value=self.value,
                rule="high_precision_documentation"
            )

    def _validate_confidence_interval_consistency(self) -> None:
        """Validate confidence interval consistency with score."""
        if self.confidence_interval is not None:
            if not self.confidence_interval.contains(self.value):
                raise InvalidValueError(
                    f"Score value ({self.value}) must be within confidence interval "
                    f"[{self.confidence_interval.lower}, {self.confidence_interval.upper}]",
                    field="confidence_interval",
                    value=self.value
                )

            # Business Rule: Confidence intervals should be reasonable
            interval_width = self.confidence_interval.width()
            if interval_width > 0.8:  # 80% of the total range
                raise InvalidValueError(
                    f"Confidence interval too wide ({interval_width:.3f}), indicates unreliable score",
                    field="confidence_interval",
                    value=interval_width,
                    rule="confidence_interval_width"
                )

            # Business Rule: Very narrow intervals on extreme scores need verification
            if (self.value > 0.95 or self.value < 0.05) and interval_width < 0.01:
                raise InvalidValueError(
                    f"Suspiciously narrow confidence interval ({interval_width:.4f}) on extreme score ({self.value})",
                    field="confidence_interval",
                    value=interval_width,
                    rule="extreme_score_narrow_interval"
                )

    def _validate_method_compatibility(self) -> None:
        """Validate method compatibility with score properties."""
        if self.method is not None:
            # Business Rule: Method validation for known scoring methods
            known_methods = {
                'isolation_forest', 'lof', 'svm', 'autoencoder', 'kmeans',
                'pca', 'gaussian_mixture', 'dbscan', 'ensemble'
            }
            
            method_lower = self.method.lower()
            if method_lower not in known_methods:
                # Allow but validate format for custom methods
                if not method_lower.replace('_', '').replace('-', '').isalnum():
                    raise InvalidValueError(
                        f"Method name '{self.method}' contains invalid characters",
                        field="method",
                        value=self.method,
                        rule="method_format"
                    )

    def is_valid(self) -> bool:
        """Check if the score is valid with comprehensive validation."""
        try:
            # Check basic type and numeric properties
            if not isinstance(self.value, (int, float)):
                return False
            
            if math.isnan(self.value) or math.isinf(self.value):
                return False
            
            # Check range
            if not (0.0 <= self.value <= 1.0):
                return False
            
            # Check business rules without raising exceptions
            if 0.0 < self.value < 0.01:
                return False
            
            # Check confidence interval consistency
            if self.confidence_interval is not None:
                if not self.confidence_interval.contains(self.value):
                    return False
                
                interval_width = self.confidence_interval.width()
                if interval_width > 0.8:
                    return False
            
            return True
        except Exception:
            return False

    @property
    def is_confident(self) -> bool:
        """Check if score has confidence intervals."""
        return self.confidence_interval is not None

    @property
    def confidence_width(self) -> float | None:
        """Calculate width of confidence interval."""
        if self.is_confident and self.confidence_interval:
            return self.confidence_interval.width()
        return None

    @property
    def confidence_lower(self) -> float | None:
        """Get lower bound of confidence interval."""
        if self.confidence_interval:
            return self.confidence_interval.lower
        return None

    @property
    def confidence_upper(self) -> float | None:
        """Get upper bound of confidence interval."""
        if self.confidence_interval:
            return self.confidence_interval.upper
        return None

    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if score exceeds a given threshold with validation."""
        if not isinstance(threshold, (int, float)):
            raise InvalidValueError(
                f"Threshold must be numeric, got {type(threshold)}",
                field="threshold",
                value=threshold
            )
        
        if not (0.0 <= threshold <= 1.0):
            raise InvalidValueError(
                f"Threshold must be between 0 and 1, got {threshold}",
                field="threshold",
                value=threshold
            )
        
        return self.value > threshold

    def is_anomaly(self, threshold: float = 0.5) -> bool:
        """Check if score indicates an anomaly above the threshold."""
        return self.exceeds_threshold(threshold)

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if score indicates high confidence anomaly."""
        return self.exceeds_threshold(threshold)

    def severity_level(self) -> str:
        """Categorize anomaly severity based on score value."""
        if self.value >= 0.9:
            return "critical"
        elif self.value >= 0.7:
            return "high"
        elif self.value >= 0.5:
            return "medium"
        elif self.value >= 0.3:
            return "low"
        else:
            return "minimal"

    def confidence_score(self) -> float:
        """Calculate confidence score based on interval width."""
        if self.confidence_interval is None:
            return 0.0  # No confidence interval means no confidence measure
        
        # Confidence is inversely related to interval width
        width = self.confidence_interval.width()
        return max(0.0, 1.0 - width)

    def adjusted_score(self, confidence_weight: float = 0.2) -> float:
        """Calculate confidence-adjusted score."""
        if self.confidence_interval is None:
            return self.value
        
        confidence = self.confidence_score()
        return self.value * (1 - confidence_weight) + confidence * confidence_weight

    def __str__(self) -> str:
        """String representation of the score."""
        return str(self.value)

    def __lt__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value < other.value
        if isinstance(other, (int, float)):
            return self.value < other
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value <= other.value
        if isinstance(other, (int, float)):
            return self.value <= other
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value > other.value
        if isinstance(other, (int, float)):
            return self.value > other
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        """Compare scores by value."""
        if isinstance(other, AnomalyScore):
            return self.value >= other.value
        if isinstance(other, (int, float)):
            return self.value >= other
        return NotImplemented


# Import here to avoid circular imports
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
