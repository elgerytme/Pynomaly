"""Domain service for threshold calculation."""

from __future__ import annotations

import numpy as np

from monorepo.domain.value_objects import ContaminationRate


class ThresholdCalculator:
    """Domain service for calculating anomaly detection thresholds.

    This service encapsulates various strategies for determining
    the threshold that separates normal from anomalous data points.
    """

    @staticmethod
    def calculate_by_contamination(
        scores: list[float], contamination_rate: ContaminationRate
    ) -> float:
        """Calculate threshold based on contamination rate.

        Args:
            scores: Raw anomaly scores
            contamination_rate: Expected proportion of anomalies

        Returns:
            Threshold value
        """
        if not scores:
            raise ValueError("Cannot calculate threshold from empty scores")

        sorted_scores = sorted(scores, reverse=True)
        n_samples = len(sorted_scores)

        # Calculate number of anomalies
        n_anomalies = contamination_rate.calculate_threshold_index(n_samples)

        # Handle edge cases
        if n_anomalies == 0:
            # No anomalies - set threshold above max score
            return sorted_scores[0] + 0.1

        if n_anomalies >= n_samples:
            # All anomalies - set threshold below min score
            return sorted_scores[-1] - 0.1

        # Normal case - threshold is between n_anomalies-1 and n_anomalies
        threshold_idx = n_anomalies - 1
        return sorted_scores[threshold_idx]

    @staticmethod
    def calculate_by_percentile(scores: list[float], percentile: float) -> float:
        """Calculate threshold based on percentile.

        Args:
            scores: Raw anomaly scores
            percentile: Percentile for threshold (e.g., 95 for top 5%)

        Returns:
            Threshold value
        """
        if not scores:
            raise ValueError("Cannot calculate threshold from empty scores")

        if not 0 <= percentile <= 100:
            raise ValueError(f"Percentile must be in [0, 100], got {percentile}")

        return float(np.percentile(scores, percentile))

    @staticmethod
    def calculate_by_iqr(scores: list[float], multiplier: float = 1.5) -> float:
        """Calculate threshold using Interquartile Range (IQR) method.

        Args:
            scores: Raw anomaly scores
            multiplier: IQR multiplier (typically 1.5 or 3.0)

        Returns:
            Threshold value
        """
        if not scores:
            raise ValueError("Cannot calculate threshold from empty scores")

        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1

        # Anomalies are typically high scores
        threshold = q3 + multiplier * iqr

        return float(threshold)

    @staticmethod
    def calculate_by_mad(scores: list[float], threshold_factor: float = 3.0) -> float:
        """Calculate threshold using Median Absolute Deviation (MAD).

        Args:
            scores: Raw anomaly scores
            threshold_factor: Number of MADs from median

        Returns:
            Threshold value
        """
        if not scores:
            raise ValueError("Cannot calculate threshold from empty scores")

        scores_array = np.array(scores)
        median = np.median(scores_array)
        mad = np.median(np.abs(scores_array - median))

        # Constant to make MAD consistent with standard deviation
        # for normal distribution
        mad_std = mad * 1.4826

        threshold = median + threshold_factor * mad_std

        return float(threshold)

    @staticmethod
    def calculate_dynamic_threshold(
        scores: list[float],
        min_anomalies: int = 1,
        max_contamination: float = 0.5,
        method: str = "knee",
    ) -> tuple[float, float]:
        """Calculate threshold dynamically based on score distribution.

        Args:
            scores: Raw anomaly scores
            min_anomalies: Minimum number of anomalies to detect
            max_contamination: Maximum allowed contamination rate
            method: Method for finding threshold ('knee', 'gap', 'slope')

        Returns:
            Tuple of (threshold, estimated_contamination_rate)
        """
        if not scores:
            raise ValueError("Cannot calculate threshold from empty scores")

        sorted_scores = sorted(scores, reverse=True)
        n_samples = len(sorted_scores)
        max_anomalies = int(n_samples * max_contamination)

        if method == "knee":
            threshold, n_anomalies = ThresholdCalculator._find_knee_point(
                sorted_scores, min_anomalies, max_anomalies
            )
        elif method == "gap":
            threshold, n_anomalies = ThresholdCalculator._find_largest_gap(
                sorted_scores, min_anomalies, max_anomalies
            )
        elif method == "slope":
            threshold, n_anomalies = ThresholdCalculator._find_slope_change(
                sorted_scores, min_anomalies, max_anomalies
            )
        else:
            raise ValueError(f"Unknown dynamic threshold method: {method}")

        contamination_rate = n_anomalies / n_samples

        return threshold, contamination_rate

    @staticmethod
    def _find_knee_point(
        sorted_scores: list[float], min_anomalies: int, max_anomalies: int
    ) -> tuple[float, int]:
        """Find knee point in the score curve."""
        # Simple knee detection: find point with maximum distance
        # from line connecting first and last valid points

        if max_anomalies <= min_anomalies:
            return sorted_scores[min_anomalies - 1], min_anomalies

        # Create line from first to last point
        x1, y1 = 0, sorted_scores[min_anomalies - 1]
        x2, y2 = max_anomalies - min_anomalies, sorted_scores[max_anomalies - 1]

        max_distance = 0
        best_idx = min_anomalies - 1

        for i in range(min_anomalies - 1, max_anomalies):
            # Calculate perpendicular distance to line
            x0 = i - min_anomalies + 1
            y0 = sorted_scores[i]

            # Distance formula from point to line
            numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if denominator > 0:
                distance = numerator / denominator

                if distance > max_distance:
                    max_distance = distance
                    best_idx = i

        return sorted_scores[best_idx], best_idx + 1

    @staticmethod
    def _find_largest_gap(
        sorted_scores: list[float], min_anomalies: int, max_anomalies: int
    ) -> tuple[float, int]:
        """Find largest gap between consecutive scores."""
        max_gap = 0
        best_idx = min_anomalies - 1

        for i in range(
            min_anomalies - 1, min(max_anomalies - 1, len(sorted_scores) - 1)
        ):
            gap = sorted_scores[i] - sorted_scores[i + 1]

            if gap > max_gap:
                max_gap = gap
                best_idx = i

        # Threshold is between the two scores with largest gap
        threshold = (sorted_scores[best_idx] + sorted_scores[best_idx + 1]) / 2

        return threshold, best_idx + 1

    @staticmethod
    def _find_slope_change(
        sorted_scores: list[float], min_anomalies: int, max_anomalies: int
    ) -> tuple[float, int]:
        """Find point where slope changes most dramatically."""
        if max_anomalies <= min_anomalies + 2:
            return sorted_scores[min_anomalies - 1], min_anomalies

        # Calculate second derivative (change in slope)
        max_slope_change = 0
        best_idx = min_anomalies - 1

        for i in range(min_anomalies, max_anomalies - 1):
            # Approximate second derivative
            if i > 0 and i < len(sorted_scores) - 1:
                second_deriv = (
                    sorted_scores[i - 1] - 2 * sorted_scores[i] + sorted_scores[i + 1]
                )

                if abs(second_deriv) > max_slope_change:
                    max_slope_change = abs(second_deriv)
                    best_idx = i

        return sorted_scores[best_idx], best_idx + 1
