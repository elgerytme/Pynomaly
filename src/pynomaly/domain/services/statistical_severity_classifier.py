"""Statistical severity classifier using z-score calculation."""

from __future__ import annotations

from typing import List, Union
import logging
import numpy as np

from pynomaly.domain.value_objects import AnomalyScore

logger = logging.getLogger(__name__)


class StatisticalSeverityClassifier:
    """Statistical severity classifier using z-score against population.
    
    This classifier uses z-score statistics to classify anomalies
    into severity levels based on the deviation from the population mean.
    """

    SEVERITY_THRESHOLDS = {
        "low": 1.0,
        "medium": 2.0,
        "high": 3.0,
        "critical": float('inf')  # represent extremely high z-scores
    }

    def __init__(self, thresholds: dict[str, float] | None = None,
                 default_severity: str = "medium"):
        """Initialize the statistical severity classifier.

        Args:
            thresholds: Custom z-score thresholds for severity levels
            default_severity: Default severity for scores that don't meet any criteria
        """
        self.severity_thresholds = thresholds or self.SEVERITY_THRESHOLDS.copy()
        self.default_severity = default_severity
        
    def _calculate_z_scores(self, scores: List[float]) -> List[float]:
        """Calculate z-scores for a list of scores.

        Args:
            scores: Raw anomaly scores

        Returns:
            List of z-scores
        """
        scores_array = np.array(scores)
        mean = scores_array.mean()
        std = scores_array.std()

        if std == 0:
            logger.warning("Standard deviation is zero, using zero z-scores.")
            return [0] * len(scores)

        z_scores = (scores_array - mean) / std
        return list(z_scores)
    
    def classify_single(self, score: float, z_score: float | None = None) -> str:
        """Classify a single score using z-score calculation.

        Args:
            score: Anomaly score to classify
            z_score: Pre-calculated z-score (optional)

        Returns:
            Severity level
        """
        if z_score is None:
            z_score = self._calculate_z_scores([score])[0]

        for severity, threshold in self.severity_thresholds.items():
            if z_score <= threshold:
                return severity
        
        return self.default_severity

    def classify_batch(self, scores: List[Union[float, AnomalyScore]]) -> List[str]:
        """Classify a batch of anomaly scores.

        Args:
            scores: List of anomaly scores
        
        Returns:
            List of severity levels
        """
        if not scores:
            return []

        raw_scores = [score.value if isinstance(score, AnomalyScore) else score for score in scores]
        z_scores = self._calculate_z_scores(raw_scores)

        return [self.classify_single(score, z) for score, z in zip(raw_scores, z_scores)]

    def get_severity_stats(self, scores: List[Union[float, AnomalyScore]]) -> dict[str, dict[str, Union[int, float]]]:
        """Calculate statistics on the severity levels of a batch of scores.

        Args:
            scores: List of anomaly scores

        Returns:
            Dictionary containing the count and percentage per severity level
        """
        if not scores:
            return {}

        classifications = self.classify_batch(scores)
        total_count = len(classifications)

        stats = {}
        for severity in self.severity_thresholds.keys():
            count = classifications.count(severity)
            stats[severity] = {
                "count": count,
                "percentage": (count / total_count) * 100 if total_count > 0 else 0.0
            }

        return stats

    def __repr__(self) -> str:
        """String representation of the classifier."""
        return f"StatisticalSeverityClassifier(thresholds={len(self.severity_thresholds)}, default={self.default_severity})"
