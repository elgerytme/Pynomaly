"""Domain service for ensemble aggregation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from monorepo.domain.value_objects import AnomalyScore


class EnsembleAggregator:
    """Domain service for aggregating results from multiple detectors.

    This service implements various strategies for combining
    predictions from ensemble members.
    """

    @staticmethod
    def aggregate_scores(
        scores_dict: dict[str, list[AnomalyScore]],
        weights: dict[str, float] | None = None,
        method: str = "average",
    ) -> list[AnomalyScore]:
        """Aggregate scores from multiple detectors.

        Args:
            scores_dict: Dictionary mapping detector names to score lists
            weights: Optional weights for each detector
            method: Aggregation method ('average', 'median', 'max', 'weighted')

        Returns:
            Aggregated anomaly scores
        """
        if not scores_dict:
            return []

        # Validate all detectors have same number of scores
        n_samples = len(next(iter(scores_dict.values())))
        for name, scores in scores_dict.items():
            if len(scores) != n_samples:
                raise ValueError(
                    f"Detector '{name}' has {len(scores)} scores, expected {n_samples}"
                )

        if method == "average":
            return EnsembleAggregator._average_scores(scores_dict)
        elif method == "median":
            return EnsembleAggregator._median_scores(scores_dict)
        elif method == "max":
            return EnsembleAggregator._max_scores(scores_dict)
        elif method == "weighted":
            if weights is None:
                # Default to equal weights
                weights = dict.fromkeys(scores_dict, 1.0)
            return EnsembleAggregator._weighted_average_scores(scores_dict, weights)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    @staticmethod
    def _average_scores(
        scores_dict: dict[str, list[AnomalyScore]],
    ) -> list[AnomalyScore]:
        """Calculate average scores across detectors."""
        n_samples = len(next(iter(scores_dict.values())))
        n_detectors = len(scores_dict)

        aggregated = []
        for i in range(n_samples):
            values = [scores[i].value for scores in scores_dict.values()]
            avg_score = sum(values) / n_detectors

            # If all scores have confidence intervals, aggregate them too
            if all(scores[i].is_confident for scores in scores_dict.values()):
                lower_values = [
                    scores[i].confidence_lower for scores in scores_dict.values()
                ]
                upper_values = [
                    scores[i].confidence_upper for scores in scores_dict.values()
                ]

                aggregated.append(
                    AnomalyScore(
                        value=avg_score,
                        confidence_lower=sum(lower_values) / n_detectors,  # type: ignore
                        confidence_upper=sum(upper_values) / n_detectors,  # type: ignore
                        method="ensemble_average",
                    )
                )
            else:
                aggregated.append(
                    AnomalyScore(value=avg_score, method="ensemble_average")
                )

        return aggregated

    @staticmethod
    def _median_scores(
        scores_dict: dict[str, list[AnomalyScore]],
    ) -> list[AnomalyScore]:
        """Calculate median scores across detectors."""
        n_samples = len(next(iter(scores_dict.values())))

        aggregated = []
        for i in range(n_samples):
            values = [scores[i].value for scores in scores_dict.values()]
            median_score = float(np.median(values))

            aggregated.append(
                AnomalyScore(value=median_score, method="ensemble_median")
            )

        return aggregated

    @staticmethod
    def _max_scores(scores_dict: dict[str, list[AnomalyScore]]) -> list[AnomalyScore]:
        """Take maximum score across detectors."""
        n_samples = len(next(iter(scores_dict.values())))

        aggregated = []
        for i in range(n_samples):
            values = [scores[i].value for scores in scores_dict.values()]
            max_score = max(values)

            aggregated.append(AnomalyScore(value=max_score, method="ensemble_max"))

        return aggregated

    @staticmethod
    def _weighted_average_scores(
        scores_dict: dict[str, list[AnomalyScore]], weights: dict[str, float]
    ) -> list[AnomalyScore]:
        """Calculate weighted average of scores."""
        n_samples = len(next(iter(scores_dict.values())))

        # Normalize weights
        total_weight = sum(weights.values())
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        aggregated = []
        for i in range(n_samples):
            weighted_sum = sum(
                scores[i].value * norm_weights.get(name, 0)
                for name, scores in scores_dict.items()
            )

            aggregated.append(
                AnomalyScore(value=weighted_sum, method="ensemble_weighted")
            )

        return aggregated

    @staticmethod
    def aggregate_labels(
        labels_dict: dict[str, np.ndarray],
        weights: dict[str, float] | None = None,
        method: str = "majority",
    ) -> np.ndarray:
        """Aggregate binary labels from multiple detectors.

        Args:
            labels_dict: Dictionary mapping detector names to label arrays
            weights: Optional weights for voting
            method: Aggregation method ('majority', 'unanimous', 'any')

        Returns:
            Aggregated binary labels
        """
        if not labels_dict:
            return np.array([])

        # Convert to list of arrays for easier processing
        all_labels = list(labels_dict.values())
        len(all_labels[0])

        if method == "majority":
            return EnsembleAggregator._majority_vote(labels_dict, weights)
        elif method == "unanimous":
            # All detectors must agree it's an anomaly
            return np.all(all_labels, axis=0).astype(int)
        elif method == "any":
            # Any detector flags as anomaly
            return np.any(all_labels, axis=0).astype(int)
        else:
            raise ValueError(f"Unknown label aggregation method: {method}")

    @staticmethod
    def _majority_vote(
        labels_dict: dict[str, np.ndarray], weights: dict[str, float] | None = None
    ) -> np.ndarray:
        """Perform (weighted) majority voting."""
        n_samples = len(next(iter(labels_dict.values())))
        aggregated = np.zeros(n_samples, dtype=int)

        if weights is None:
            # Simple majority vote
            for i in range(n_samples):
                votes = [labels[i] for labels in labels_dict.values()]
                aggregated[i] = 1 if sum(votes) > len(votes) / 2 else 0
        else:
            # Weighted majority vote
            total_weight = sum(weights.values())
            threshold = total_weight / 2

            for i in range(n_samples):
                weighted_votes = sum(
                    labels[i] * weights.get(name, 0)
                    for name, labels in labels_dict.items()
                )
                aggregated[i] = 1 if weighted_votes > threshold else 0

        return aggregated

    @staticmethod
    def calculate_agreement(
        labels_dict: dict[str, np.ndarray],
    ) -> tuple[float, np.ndarray]:
        """Calculate agreement between detectors.

        Args:
            labels_dict: Dictionary mapping detector names to label arrays

        Returns:
            Tuple of (overall_agreement_rate, per_sample_agreement)
        """
        if len(labels_dict) < 2:
            return 1.0, np.ones(len(next(iter(labels_dict.values()))))

        all_labels = np.array(list(labels_dict.values()))
        n_detectors, n_samples = all_labels.shape

        # Per-sample agreement: fraction of detectors that agree with majority
        per_sample_agreement = np.zeros(n_samples)

        for i in range(n_samples):
            votes = all_labels[:, i]
            # Count most common vote
            vote_counts = Counter(votes)
            majority_count = max(vote_counts.values())
            per_sample_agreement[i] = majority_count / n_detectors

        # Overall agreement
        overall_agreement = per_sample_agreement.mean()

        return float(overall_agreement), per_sample_agreement

    @staticmethod
    def rank_detectors(
        scores_dict: dict[str, list[AnomalyScore]],
        true_labels: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        """Rank detectors by their contribution or performance.

        Args:
            scores_dict: Dictionary mapping detector names to score lists
            true_labels: Optional ground truth labels for performance ranking

        Returns:
            List of (detector_name, rank_score) tuples, sorted by rank
        """
        if true_labels is not None:
            # Rank by performance (e.g., correlation with true labels)
            rankings = []

            for name, scores in scores_dict.items():
                score_values = np.array([s.value for s in scores])
                # Use Spearman correlation as ranking metric
                from scipy.stats import spearmanr

                corr, _ = spearmanr(score_values, true_labels)
                rankings.append((name, abs(corr)))

            return sorted(rankings, key=lambda x: x[1], reverse=True)
        else:
            # Rank by diversity (how different from average)
            avg_scores = EnsembleAggregator._average_scores(scores_dict)
            avg_values = np.array([s.value for s in avg_scores])

            rankings = []
            for name, scores in scores_dict.items():
                score_values = np.array([s.value for s in scores])
                # Calculate mean absolute deviation from average
                diversity = np.mean(np.abs(score_values - avg_values))
                rankings.append((name, diversity))

            return sorted(rankings, key=lambda x: x[1], reverse=True)
