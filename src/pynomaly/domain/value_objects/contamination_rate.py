"""Contamination rate value object."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Tuple

from pynomaly.domain.exceptions import InvalidValueError


@dataclass(frozen=True)
class ContaminationRate:
    """Immutable value object representing contamination rate.

    Attributes:
        value: The contamination rate (0.0 to 0.5)
    """

    value: float

    # Class constants for common rates
    AUTO: ClassVar[ContaminationRate]
    LOW: ClassVar[ContaminationRate]
    MEDIUM: ClassVar[ContaminationRate]
    HIGH: ClassVar[ContaminationRate]

    def __post_init__(self) -> None:
        """Validate contamination rate after initialization."""
        self._validate_basic_constraints()
        self._validate_precision_constraints()
        self._validate_business_rules()

    def _validate_basic_constraints(self) -> None:
        """Validate basic type and range constraints."""
        if not isinstance(self.value, (int, float)):
            raise InvalidValueError(
                f"Contamination rate must be numeric, got {type(self.value)}",
                field="value",
                value=self.value
            )

        if math.isnan(self.value) or math.isinf(self.value):
            raise InvalidValueError(
                "Contamination rate cannot be NaN or infinite",
                field="value",
                value=self.value
            )

        if not (0.0 <= self.value <= 0.5):
            raise InvalidValueError(
                f"Contamination rate must be between 0 and 0.5, got {self.value}",
                field="value",
                value=self.value
            )

    def _validate_business_rules(self) -> None:
        """Validate advanced business rules for contamination rates."""
        # Business Rule: High contamination rates (>40%) should be flagged
        if self.value > 0.4:
            raise InvalidValueError(
                f"Contamination rate {self.value} is unusually high (>40%) and may indicate data issues",
                field="value",
                value=self.value,
                rule="high_contamination_warning"
            )

        # Business Rule: Very low contamination rates (<0.001) should be flagged
        if 0.0 < self.value < 0.001:
            raise InvalidValueError(
                f"Contamination rate {self.value} is extremely low (<0.1%) and may not be meaningful",
                field="value",
                value=self.value,
                rule="low_contamination_warning"
            )

    def _validate_precision_constraints(self) -> None:
        """Validate precision constraints for contamination rates."""
        # Business Rule: Precision should not exceed 6 decimal places
        if self.value != round(self.value, 6):
            raise InvalidValueError(
                f"Contamination rate has excessive precision beyond 6 decimal places, got {self.value}",
                field="value",
                value=self.value,
                rule="precision_constraint"
            )

    def is_valid(self) -> bool:
        """Check if the contamination rate is valid."""
        try:
            self._validate_basic_constraints()
            self._validate_precision_constraints()
            self._validate_business_rules()
            return True
        except InvalidValueError:
            return False

    def as_percentage(self) -> float:
        """Return contamination rate as a percentage (0-100)."""
        return self.value * 100.0

    def as_ratio(self) -> str:
        """Return contamination rate as a ratio string (e.g., '1:10')."""
        if self.value == 0.0:
            return "0:1"
        
        # Calculate ratio as 1:N where N is the inverse of contamination rate
        inverse = 1.0 / self.value
        
        # Round to nearest integer if close enough
        if abs(inverse - round(inverse)) < 0.01:
            return f"1:{int(round(inverse))}"
        else:
            return f"1:{inverse:.1f}"

    def is_standard(self) -> bool:
        """Check if contamination rate is a standard value."""
        standard_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        return any(abs(self.value - rate) < 1e-10 for rate in standard_rates)

    def severity_level(self) -> str:
        """Categorize contamination rate severity."""
        if self.value == 0.0:
            return "none"
        elif self.value < 0.05:
            return "low"
        elif self.value < 0.15:
            return "moderate"
        elif self.value < 0.3:
            return "high"
        else:
            return "critical"

    def expected_anomalies(self, sample_size: int) -> int:
        """Calculate expected number of anomalies for given sample size."""
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise InvalidValueError(
                f"Sample size must be a positive integer, got {sample_size}",
                field="sample_size",
                value=sample_size
            )
        
        return int(self.value * sample_size)

    def confidence_interval(self, sample_size: int, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for contamination rate."""
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise InvalidValueError(
                f"Sample size must be a positive integer, got {sample_size}",
                field="sample_size",
                value=sample_size
            )
        
        if not (0.0 < confidence_level < 1.0):
            raise InvalidValueError(
                f"Confidence level must be between 0 and 1, got {confidence_level}",
                field="confidence_level",
                value=confidence_level
            )
        
        # Use normal approximation for binomial confidence interval
        # This is a simplified calculation - in practice, you'd use scipy.stats
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # Simplified z-scores
        
        # Standard error
        standard_error = math.sqrt(self.value * (1 - self.value) / sample_size)
        
        # Margin of error
        margin_of_error = z_score * standard_error
        
        # Calculate bounds
        lower = max(0.0, self.value - margin_of_error)
        upper = min(0.5, self.value + margin_of_error)
        
        return (lower, upper)

    def __str__(self) -> str:
        """String representation."""
        # Format as percentage, removing trailing zeros after decimal
        percentage = self.value * 100
        if percentage == int(percentage):
            return f"{int(percentage)}.0%"
        else:
            return f"{percentage:.1f}%"

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
