"""Threshold-based severity classifier with extended threshold table."""

from __future__ import annotations

import logging

from pynomaly.domain.value_objects import AnomalyScore

logger = logging.getLogger(__name__)


class ThresholdSeverityClassifier:
    """Threshold-based severity classifier with extended threshold table.

    This classifier uses predefined threshold ranges to classify anomalies
    into severity levels. It supports customizable threshold tables and
    provides a default configuration that maintains current behavior.
    """

    # Default threshold table maintaining current behavior
    DEFAULT_THRESHOLDS = {
        "low": {"min": 0.0, "max": 0.3},
        "medium": {"min": 0.3, "max": 0.7},
        "high": {"min": 0.7, "max": 0.9},
        "critical": {"min": 0.9, "max": 1.0},
    }

    def __init__(
        self,
        threshold_table: dict[str, dict[str, float]] | None = None,
        default_severity: str = "medium",
    ):
        """Initialize the threshold severity classifier.

        Args:
            threshold_table: Custom threshold table mapping severity levels to ranges
            default_severity: Default severity for scores that don't match any threshold
        """
        self.threshold_table = threshold_table or self.DEFAULT_THRESHOLDS.copy()
        self.default_severity = default_severity
        self._validate_threshold_table()

    def _validate_threshold_table(self) -> None:
        """Validate the threshold table configuration."""
        if not self.threshold_table:
            raise ValueError("Threshold table cannot be empty")

        for severity, thresholds in self.threshold_table.items():
            if not isinstance(thresholds, dict):
                raise ValueError(f"Thresholds for {severity} must be a dictionary")

            if "min" not in thresholds or "max" not in thresholds:
                raise ValueError(
                    f"Thresholds for {severity} must have 'min' and 'max' keys"
                )

            min_val = thresholds["min"]
            max_val = thresholds["max"]

            if not isinstance(min_val, (int, float)) or not isinstance(
                max_val, (int, float)
            ):
                raise ValueError(f"Threshold values for {severity} must be numeric")

            if min_val >= max_val:
                raise ValueError(
                    f"Min threshold must be less than max threshold for {severity}"
                )

            if min_val < 0 or max_val > 1:
                raise ValueError(
                    f"Threshold values for {severity} must be in range [0, 1]"
                )

    def classify_single(self, score: float | AnomalyScore) -> str:
        """Classify a single anomaly score into severity level.

        Args:
            score: Anomaly score to classify

        Returns:
            Severity level as string
        """
        if isinstance(score, AnomalyScore):
            score_value = score.value
        else:
            score_value = float(score)

        # Ensure score is in valid range
        score_value = max(0.0, min(1.0, score_value))

        # Find matching severity level
        for severity, thresholds in self.threshold_table.items():
            min_threshold = thresholds["min"]
            max_threshold = thresholds["max"]

            # Use inclusive lower bound, exclusive upper bound (except for the highest level)
            if severity == max(
                self.threshold_table.keys(),
                key=lambda x: self.threshold_table[x]["max"],
            ):
                # For the highest severity level, include the upper bound
                if min_threshold <= score_value <= max_threshold:
                    return severity
            else:
                if min_threshold <= score_value < max_threshold:
                    return severity

        # Fallback to default severity if no match found
        logger.warning(
            f"Score {score_value} did not match any threshold, using default: {self.default_severity}"
        )
        return self.default_severity

    def classify_batch(self, scores: list[float | AnomalyScore]) -> list[str]:
        """Classify a batch of anomaly scores into severity levels.

        Args:
            scores: List of anomaly scores to classify

        Returns:
            List of severity levels
        """
        if not scores:
            return []

        return [self.classify_single(score) for score in scores]

    def get_severity_stats(
        self, scores: list[float | AnomalyScore]
    ) -> dict[str, dict[str, int | float]]:
        """Get statistics about severity distribution.

        Args:
            scores: List of anomaly scores

        Returns:
            Dictionary with severity statistics
        """
        if not scores:
            return {}

        classifications = self.classify_batch(scores)
        total_count = len(classifications)

        stats = {}
        for severity in self.threshold_table.keys():
            count = classifications.count(severity)
            stats[severity] = {
                "count": count,
                "percentage": (count / total_count) * 100 if total_count > 0 else 0.0,
            }

        return stats

    def update_threshold(
        self, severity: str, min_threshold: float, max_threshold: float
    ) -> None:
        """Update threshold for a specific severity level.

        Args:
            severity: Severity level to update
            min_threshold: New minimum threshold
            max_threshold: New maximum threshold
        """
        if severity not in self.threshold_table:
            raise ValueError(
                f"Severity level '{severity}' not found in threshold table"
            )

        if min_threshold >= max_threshold:
            raise ValueError("Min threshold must be less than max threshold")

        if min_threshold < 0 or max_threshold > 1:
            raise ValueError("Threshold values must be in range [0, 1]")

        self.threshold_table[severity]["min"] = min_threshold
        self.threshold_table[severity]["max"] = max_threshold

    def add_severity_level(
        self, severity: str, min_threshold: float, max_threshold: float
    ) -> None:
        """Add a new severity level to the threshold table.

        Args:
            severity: New severity level name
            min_threshold: Minimum threshold for the new level
            max_threshold: Maximum threshold for the new level
        """
        if severity in self.threshold_table:
            raise ValueError(f"Severity level '{severity}' already exists")

        if min_threshold >= max_threshold:
            raise ValueError("Min threshold must be less than max threshold")

        if min_threshold < 0 or max_threshold > 1:
            raise ValueError("Threshold values must be in range [0, 1]")

        self.threshold_table[severity] = {"min": min_threshold, "max": max_threshold}

    def remove_severity_level(self, severity: str) -> None:
        """Remove a severity level from the threshold table.

        Args:
            severity: Severity level to remove
        """
        if severity not in self.threshold_table:
            raise ValueError(
                f"Severity level '{severity}' not found in threshold table"
            )

        if len(self.threshold_table) <= 1:
            raise ValueError("Cannot remove the last severity level")

        del self.threshold_table[severity]

    def get_threshold_table(self) -> dict[str, dict[str, float]]:
        """Get a copy of the current threshold table.

        Returns:
            Copy of the threshold table
        """
        return {
            severity: thresholds.copy()
            for severity, thresholds in self.threshold_table.items()
        }

    def reset_to_default(self) -> None:
        """Reset the threshold table to default values."""
        self.threshold_table = self.DEFAULT_THRESHOLDS.copy()
        self._validate_threshold_table()

    def get_severity_for_threshold_range(
        self, min_val: float, max_val: float
    ) -> str | None:
        """Get severity level that matches a specific threshold range.

        Args:
            min_val: Minimum threshold value
            max_val: Maximum threshold value

        Returns:
            Severity level name if found, None otherwise
        """
        for severity, thresholds in self.threshold_table.items():
            if (
                abs(thresholds["min"] - min_val) < 1e-6
                and abs(thresholds["max"] - max_val) < 1e-6
            ):
                return severity
        return None

    def __repr__(self) -> str:
        """String representation of the classifier."""
        return f"ThresholdSeverityClassifier(thresholds={len(self.threshold_table)}, default={self.default_severity})"
