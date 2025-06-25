"""Domain service for anomaly scoring logic."""

from __future__ import annotations

import numpy as np

from pynomaly.domain.value_objects import AnomalyScore


class AnomalyScorer:
    """Domain service for calculating and normalizing anomaly scores.

    This service contains the core business logic for scoring anomalies,
    independent of any specific detection algorithm.
    """

    @staticmethod
    def normalize_scores(
        scores: list[float], method: str = "min-max"
    ) -> list[AnomalyScore]:
        """Normalize raw scores to [0, 1] range.

        Args:
            scores: Raw anomaly scores
            method: Normalization method ('min-max', 'z-score', 'percentile')

        Returns:
            List of normalized AnomalyScore objects
        """
        if not scores:
            return []

        scores_array = np.array(scores)

        if method == "min-max":
            normalized = AnomalyScorer._min_max_normalize(scores_array)
        elif method == "z-score":
            normalized = AnomalyScorer._z_score_normalize(scores_array)
        elif method == "percentile":
            normalized = AnomalyScorer._percentile_normalize(scores_array)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return [AnomalyScore(value=float(score), method=method) for score in normalized]

    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        min_score = scores.min()
        max_score = scores.max()

        if min_score == max_score:
            # All scores are the same
            return np.ones_like(scores) * 0.5

        return (scores - min_score) / (max_score - min_score)

    @staticmethod
    def _z_score_normalize(scores: np.ndarray) -> np.ndarray:
        """Z-score normalization with sigmoid to [0, 1]."""
        mean = scores.mean()
        std = scores.std()

        if std == 0:
            # No variation in scores
            return np.ones_like(scores) * 0.5

        z_scores = (scores - mean) / std
        # Apply sigmoid to map to [0, 1]
        return 1 / (1 + np.exp(-z_scores))

    @staticmethod
    def _percentile_normalize(scores: np.ndarray) -> np.ndarray:
        """Percentile rank normalization."""
        from scipy import stats

        return stats.rankdata(scores, method="average") / len(scores)

    @staticmethod
    def calculate_threshold(
        scores: list[AnomalyScore], contamination_rate: float
    ) -> float:
        """Calculate threshold based on contamination rate.

        Args:
            scores: List of anomaly scores
            contamination_rate: Expected proportion of anomalies

        Returns:
            Threshold value
        """
        if not scores:
            raise ValueError("Cannot calculate threshold from empty scores")

        if not 0 <= contamination_rate <= 1:
            raise ValueError(
                f"Contamination rate must be in [0, 1], got {contamination_rate}"
            )

        score_values = sorted([s.value for s in scores], reverse=True)
        n_anomalies = max(1, int(len(score_values) * contamination_rate))

        # Ensure we don't exceed array bounds
        threshold_idx = min(n_anomalies - 1, len(score_values) - 1)

        return score_values[threshold_idx]

    @staticmethod
    def add_confidence_intervals(
        scores: list[AnomalyScore],
        confidence_level: float = 0.95,
        method: str = "bootstrap",
    ) -> list[AnomalyScore]:
        """Add confidence intervals to scores.

        Args:
            scores: List of anomaly scores
            confidence_level: Confidence level (e.g., 0.95)
            method: Method for CI calculation ('bootstrap', 'empirical')

        Returns:
            Scores with confidence intervals
        """
        if not scores:
            return []

        if method == "bootstrap":
            return AnomalyScorer._bootstrap_confidence_intervals(
                scores, confidence_level
            )
        elif method == "empirical":
            return AnomalyScorer._empirical_confidence_intervals(
                scores, confidence_level
            )
        else:
            raise ValueError(f"Unknown confidence interval method: {method}")

    @staticmethod
    def _bootstrap_confidence_intervals(
        scores: list[AnomalyScore], confidence_level: float, n_bootstrap: int = 1000
    ) -> list[AnomalyScore]:
        """Calculate bootstrap confidence intervals."""
        score_values = np.array([s.value for s in scores])
        n_samples = len(score_values)

        # Bootstrap resampling
        bootstrap_scores = np.zeros((n_bootstrap, n_samples))
        rng = np.random.RandomState(42)  # For reproducibility

        for i in range(n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            bootstrap_scores[i] = score_values[indices]

        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        result = []
        for i, score in enumerate(scores):
            bootstrap_values = bootstrap_scores[:, i]
            ci_lower = np.percentile(bootstrap_values, lower_percentile)
            ci_upper = np.percentile(bootstrap_values, upper_percentile)

            result.append(
                AnomalyScore(
                    value=score.value,
                    confidence_lower=float(ci_lower),
                    confidence_upper=float(ci_upper),
                    method=score.method,
                )
            )

        return result

    @staticmethod
    def _empirical_confidence_intervals(
        scores: list[AnomalyScore], confidence_level: float
    ) -> list[AnomalyScore]:
        """Calculate empirical confidence intervals based on score distribution."""
        score_values = np.array([s.value for s in scores])

        # Use empirical distribution
        mean = score_values.mean()
        std = score_values.std()

        # For normal approximation
        from scipy import stats

        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * std / np.sqrt(len(score_values))

        result = []
        for score in scores:
            # Simple approach: use global statistics
            ci_lower = max(0, score.value - margin)
            ci_upper = min(1, score.value + margin)

            result.append(
                AnomalyScore(
                    value=score.value,
                    confidence_lower=float(ci_lower),
                    confidence_upper=float(ci_upper),
                    method=score.method,
                )
            )

        return result

    @staticmethod
    def rank_scores(
        scores: list[AnomalyScore], ascending: bool = False
    ) -> list[tuple[int, AnomalyScore]]:
        """Rank scores and return with their original indices.

        Args:
            scores: List of anomaly scores
            ascending: If True, rank from lowest to highest

        Returns:
            List of (original_index, score) tuples sorted by score
        """
        indexed_scores = list(enumerate(scores))
        return sorted(indexed_scores, key=lambda x: x[1].value, reverse=not ascending)
