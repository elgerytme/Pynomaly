"""Anomaly score value object."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from pynomaly.domain.exceptions import ValidationError


@dataclass(frozen=True)
class AnomalyScore:
    """Immutable value object representing an anomaly score.

    Attributes:
        value: The anomaly score value (higher means more anomalous)
        threshold: The threshold for anomaly classification (default: 0.5)
        metadata: Additional metadata about the score
        confidence_interval: Confidence interval for the score (optional)
        method: The scoring method used (optional)
    """

    value: float
    threshold: float = 0.5
    metadata: dict[str, Any] = None
    confidence_interval: Any = None  # ConfidenceInterval | None
    method: str | None = None

    def __post_init__(self) -> None:
        """Validate score after initialization with advanced business rules."""
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            # Make a deep copy to prevent mutation of original dict
            import copy

            object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

        # Basic validation
        self._validate_basic_constraints()

        # Advanced business rule validation
        self._validate_business_rules()

        # Cross-field validation
        self._validate_field_relationships()

        # Statistical validation
        self._validate_statistical_properties()

    def _validate_basic_constraints(self) -> None:
        """Validate basic type and range constraints."""
        # Validate value
        if not isinstance(self.value, (int, float)):
            raise ValidationError(
                f"Score value must be numeric, got {type(self.value)}"
            )

        if math.isnan(self.value) or math.isinf(self.value):
            raise ValidationError(f"Score value must be finite, got {self.value}")

        if not (0.0 <= self.value <= 1.0):
            raise ValidationError(
                f"Score value must be between 0 and 1, got {self.value}"
            )

        # Validate threshold
        if not isinstance(self.threshold, (int, float)):
            raise ValidationError(
                f"Threshold must be numeric, got {type(self.threshold)}"
            )

        if math.isnan(self.threshold) or math.isinf(self.threshold):
            raise ValidationError(f"Threshold must be finite, got {self.threshold}")

        if not (0.0 <= self.threshold <= 1.0):
            raise ValidationError(
                f"Threshold must be between 0 and 1, got {self.threshold}"
            )

        # Validate metadata
        if not isinstance(self.metadata, dict):
            raise ValidationError(
                f"Metadata must be a dictionary, got {type(self.metadata)}"
            )

        # Validate confidence_interval if present
        if self.confidence_interval is not None:
            if (
                not hasattr(self.confidence_interval, "contains")
                or not hasattr(self.confidence_interval, "lower")
                or not hasattr(self.confidence_interval, "upper")
            ):
                raise ValidationError(
                    "Confidence interval must have 'contains', 'lower', and 'upper' attributes"
                )
            if not self.confidence_interval.contains(self.value):
                raise ValidationError(
                    f"Score value ({self.value}) must be within confidence interval "
                    f"[{self.confidence_interval.lower}, {self.confidence_interval.upper}]"
                )

    def _validate_business_rules(self) -> None:
        """Validate advanced business rules for anomaly scoring."""
        # Validate scoring method if provided
        if self.method is not None:
            if not isinstance(self.method, str):
                raise ValidationError(
                    f"Scoring method must be a string, got {type(self.method)}"
                )
            if len(self.method.strip()) == 0:
                raise ValidationError("Scoring method cannot be empty")

            # Validate against known scoring methods
            valid_methods = {
                "isolation_forest", "local_outlier_factor", "one_class_svm",
                "elliptic_envelope", "auto_encoder", "gaussian_mixture",
                "statistical", "distance_based", "density_based", "ensemble",
                "pyod", "sklearn"  # Add common scoring methods
            }
            if self.method.lower() not in valid_methods:
                raise ValidationError(
                    f"Unknown scoring method: {self.method}. Valid methods: {valid_methods}"
                )

        # Validate metadata structure for required fields
        if "algorithm" in self.metadata:
            if not isinstance(self.metadata["algorithm"], str):
                raise ValidationError("Algorithm metadata must be a string")

        if "feature_count" in self.metadata:
            if not isinstance(self.metadata["feature_count"], int) or self.metadata["feature_count"] <= 0:
                raise ValidationError("Feature count must be a positive integer")

        if "sample_size" in self.metadata:
            if not isinstance(self.metadata["sample_size"], int) or self.metadata["sample_size"] <= 0:
                raise ValidationError("Sample size must be a positive integer")

        # Validate business rule: very high scores should have high confidence
        if self.value > 0.9 and self.confidence_interval is not None:
            width = self.confidence_interval.upper - self.confidence_interval.lower
            if width > 0.3:  # Wide confidence interval for high scores is suspicious
                raise ValidationError(
                    f"High anomaly score ({self.value}) has suspiciously wide confidence interval (width: {width})"
                )

    def _validate_field_relationships(self) -> None:
        """Validate relationships between fields."""
        # Threshold should be reasonable for the detection context
        if self.threshold < 0.1:
            raise ValidationError(
                f"Threshold too low ({self.threshold}). May result in too many false positives."
            )

        if self.threshold > 0.9:
            raise ValidationError(
                f"Threshold too high ({self.threshold}). May result in missed anomalies."
            )

        # If confidence intervals are present, validate their consistency
        if self.confidence_interval is not None:
            lower = self.confidence_interval.lower
            upper = self.confidence_interval.upper

            # Confidence interval should be within valid score range
            if lower < 0.0 or upper > 1.0:
                raise ValidationError(
                    f"Confidence interval [{lower}, {upper}] extends outside valid score range [0, 1]"
                )

            # Confidence interval width should be reasonable
            width = upper - lower
            if width < 0.01:  # Too narrow
                raise ValidationError(
                    f"Confidence interval too narrow (width: {width}). May indicate overconfidence."
                )
            if width > 0.8:  # Too wide
                raise ValidationError(
                    f"Confidence interval too wide (width: {width}). May indicate poor model performance."
                )

    def _validate_statistical_properties(self) -> None:
        """Validate statistical properties of the anomaly score."""
        # Check for extreme values that might indicate data quality issues
        if self.value < 0.001:
            # Very low scores might indicate normal behavior but could also indicate
            # model issues or data preprocessing problems
            if "model_confidence" in self.metadata:
                model_confidence = self.metadata["model_confidence"]
                if isinstance(model_confidence, (int, float)) and model_confidence < 0.7:
                    raise ValidationError(
                        f"Very low anomaly score ({self.value}) combined with low model confidence ({model_confidence})"
                    )

        if self.value > 0.999:
            # Very high scores should be treated with caution
            if "data_quality_score" in self.metadata:
                quality_score = self.metadata["data_quality_score"]
                if isinstance(quality_score, (int, float)) and quality_score < 0.8:
                    raise ValidationError(
                        f"Very high anomaly score ({self.value}) but low data quality score ({quality_score})"
                    )

        # Validate precision for the score value
        if isinstance(self.value, float):
            # Check if the score has reasonable precision (not too many decimal places)
            decimal_places = len(str(self.value).split('.')[-1]) if '.' in str(self.value) else 0
            if decimal_places > 25:  # Allow up to 25 decimal places for high precision algorithms
                raise ValidationError(
                    f"Score value has excessive precision ({decimal_places} decimal places). Consider rounding."
                )

    def is_anomaly(self) -> bool:
        """Check if score indicates an anomaly based on threshold."""
        return self.value > self.threshold

    def confidence_level(self) -> float:
        """Calculate confidence level based on distance from threshold."""
        return abs(self.value - self.threshold)

    @property
    def is_confident(self) -> bool:
        """Check if score has confidence intervals."""
        return self.confidence_interval is not None

    @property
    def confidence_width(self) -> float | None:
        """Calculate width of confidence interval."""
        if self.is_confident and self.confidence_interval:
            return self.confidence_interval.upper - self.confidence_interval.lower
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "value": self.value,
            "threshold": self.threshold,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyScore:
        """Create AnomalyScore from dictionary."""
        if "value" not in data:
            raise ValidationError("Missing required field: value")

        return cls(
            value=data["value"],
            threshold=data.get("threshold", 0.5),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert to JSON representation."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> AnomalyScore:
        """Create AnomalyScore from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of the score."""
        return str(self.value)

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"AnomalyScore(value={self.value}, threshold={self.threshold}, metadata={self.metadata})"

    def __bool__(self) -> bool:
        """Boolean conversion - True if score is non-zero."""
        return self.value != 0.0

    def __hash__(self) -> int:
        """Hash function for use in sets and dicts."""
        return hash(
            (
                self.value,
                self.threshold,
                tuple(sorted(self.metadata.items())),
                id(self.confidence_interval) if self.confidence_interval else None,
                self.method,
            )
        )

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

    def __eq__(self, other: Any) -> bool:
        """Compare scores for equality."""
        if isinstance(other, AnomalyScore):
            return (
                self.value == other.value
                and self.threshold == other.threshold
                and self.metadata == other.metadata
                and self.confidence_interval == other.confidence_interval
                and self.method == other.method
            )
        return False

    def __ne__(self, other: Any) -> bool:
        """Compare scores for inequality."""
        return not self.__eq__(other)

    def is_valid(self) -> bool:
        """Check if the score is valid."""
        return (
            isinstance(self.value, (int, float))
            and not math.isnan(self.value)
            and not math.isinf(self.value)
            and 0.0 <= self.value <= 1.0
        )

    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if score exceeds a given threshold."""
        return self.value > threshold

    def validate_business_context(self, context: dict[str, Any]) -> bool:
        """Validate score within specific business context."""
        if not isinstance(context, dict):
            raise ValidationError("Business context must be a dictionary")

        # Validate based on detection scenario
        scenario = context.get("scenario", "general")
        if scenario == "fraud_detection":
            # Fraud detection requires higher confidence
            if self.value > 0.8 and self.confidence_interval is None:
                raise ValidationError(
                    "High-risk fraud detection scores require confidence intervals"
                )
        elif scenario == "network_security":
            # Network security should have rapid response capability
            if "response_time_ms" in context:
                response_time = context["response_time_ms"]
                if isinstance(response_time, (int, float)) and response_time > 1000:
                    raise ValidationError(
                        f"Network security detection too slow (response time: {response_time}ms)"
                    )
        elif scenario == "medical_diagnosis":
            # Medical diagnosis requires very high confidence
            if self.value > 0.7 and self.confidence_interval is not None:
                width = self.confidence_interval.upper - self.confidence_interval.lower
                if width > 0.1:
                    raise ValidationError(
                        f"Medical diagnosis requires higher confidence (interval width: {width})"
                    )

        return True

    def is_statistically_significant(self, significance_level: float = 0.05) -> bool:
        """Check if anomaly score is statistically significant."""
        if not (0.0 < significance_level < 1.0):
            raise ValidationError(
                f"Significance level must be between 0 and 1, got {significance_level}"
            )

        # If confidence interval is available, check if it excludes normal range
        if self.confidence_interval is not None:
            # Normal range is typically considered [0.0, 0.5] for anomaly scores
            normal_upper_bound = 0.5
            return self.confidence_interval.lower > normal_upper_bound

        # Without confidence interval, use a simple threshold-based approach
        return self.value > (1.0 - significance_level)

    def risk_assessment(self) -> dict[str, Any]:
        """Assess risk level based on anomaly score characteristics."""
        risk_level = "low"
        risk_factors = []

        # High score indicates high risk
        if self.value > 0.8:
            risk_level = "high"
            risk_factors.append("high_anomaly_score")
        elif self.value > 0.6:
            risk_level = "medium"
            risk_factors.append("moderate_anomaly_score")

        # Wide confidence interval increases uncertainty
        if self.confidence_interval is not None:
            width = self.confidence_interval.upper - self.confidence_interval.lower
            if width > 0.4:
                risk_factors.append("high_uncertainty")
                if risk_level == "low":
                    risk_level = "medium"

        # Low model confidence is a risk factor
        if "model_confidence" in self.metadata:
            model_confidence = self.metadata["model_confidence"]
            if isinstance(model_confidence, (int, float)) and model_confidence < 0.7:
                risk_factors.append("low_model_confidence")
                if risk_level == "low":
                    risk_level = "medium"

        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "score": self.value,
            "threshold": self.threshold,
            "requires_manual_review": len(risk_factors) > 1 or risk_level == "high"
        }
