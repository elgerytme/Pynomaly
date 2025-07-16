#!/usr/bin/env python3
"""
Anomaly Ranking and Filtering Strategies Example

This comprehensive example demonstrates advanced techniques for ranking and filtering
anomalies across multiple detectors, including confidence-based filtering, consensus
analysis, top-K selection, and production-ready filtering workflows.
"""

import asyncio
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.value_objects import ConfidenceInterval
from pynomaly.infrastructure.config import create_container
from pynomaly.infrastructure.resilience import ml_resilient


class RankingMethod(Enum):
    """Enumeration of anomaly ranking methods."""

    AVERAGE_SCORE = "average_score"
    WEIGHTED_AVERAGE = "weighted_average"
    BORDA_COUNT = "borda_count"
    CONSENSUS_RANK = "consensus_rank"
    CONCORDANCE_RANK = "concordance_rank"
    MAXIMUM_SCORE = "maximum_score"
    MINIMUM_SCORE = "minimum_score"
    HARMONIC_MEAN = "harmonic_mean"
    GEOMETRIC_MEAN = "geometric_mean"


class FilteringStrategy(Enum):
    """Enumeration of filtering strategies."""

    TOP_K = "top_k"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    CONSENSUS_LEVEL = "consensus_level"
    OUTLIER_REMOVAL = "outlier_removal"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    COMBINED_FILTER = "combined_filter"
    STATISTICAL_FILTER = "statistical_filter"


@dataclass
class RankedAnomaly:
    """Ranked anomaly with comprehensive metadata."""

    sample_index: int
    final_score: float
    individual_scores: list[float]
    rank: int
    confidence_interval: ConfidenceInterval | None
    consensus_level: float
    detector_agreement: dict[str, bool]
    metadata: dict[str, Any]


@dataclass
class FilteringResult:
    """Result of anomaly filtering."""

    filtered_anomalies: list[RankedAnomaly]
    filter_statistics: dict[str, Any]
    threshold_used: float
    rejection_reasons: dict[int, str]


class AnomalyRankingSystem:
    """Advanced system for ranking and filtering anomalies."""

    def __init__(self, container):
        self.container = container
        self.detection_service = container.detection_service()
        self.ranking_cache = {}
        self.performance_weights = {}

    @ml_resilient(timeout_seconds=300, max_attempts=2)
    async def get_detector_predictions(
        self, detector_ids: list[str], dataset: Dataset
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Get predictions and scores from all detectors with resilience."""
        predictions = {}
        scores = {}

        for detector_id in detector_ids:
            try:
                result = await self.detection_service.detect_anomalies(
                    detector_id=detector_id, dataset=dataset
                )

                # Binary predictions
                pred_array = np.zeros(len(dataset.data))
                pred_array[result.anomaly_indices] = 1
                predictions[detector_id] = pred_array

                # Anomaly scores (normalized)
                if (
                    hasattr(result, "anomaly_scores")
                    and result.anomaly_scores is not None
                ):
                    raw_scores = np.array(result.anomaly_scores)
                    # Robust normalization
                    q75, q25 = np.percentile(raw_scores, [75, 25])
                    iqr = q75 - q25
                    if iqr > 0:
                        normalized_scores = (raw_scores - q25) / iqr
                        normalized_scores = np.clip(normalized_scores, 0, 1)
                    else:
                        normalized_scores = (raw_scores - np.min(raw_scores)) / (
                            np.max(raw_scores) - np.min(raw_scores) + 1e-8
                        )
                    scores[detector_id] = normalized_scores
                else:
                    scores[detector_id] = pred_array.astype(float)

            except Exception as e:
                print(f"Warning: Detector {detector_id} failed: {e}")
                # Add zero arrays for failed detectors
                predictions[detector_id] = np.zeros(len(dataset.data))
                scores[detector_id] = np.zeros(len(dataset.data))

        return predictions, scores

    def calculate_performance_weights(
        self,
        predictions: dict[str, np.ndarray],
        true_labels: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Calculate performance-based weights for detectors."""
        if true_labels is None:
            # Equal weights if no ground truth
            return dict.fromkeys(predictions.keys(), 1.0)

        weights = {}
        for detector_id, pred in predictions.items():
            # Calculate F1 score
            tp = np.sum((pred == 1) & (true_labels == 1))
            fp = np.sum((pred == 1) & (true_labels == 0))
            fn = np.sum((pred == 0) & (true_labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            weights[detector_id] = max(f1, 0.1)  # Minimum weight

        # Normalize weights
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()}

    def rank_anomalies_average_score(
        self, scores: dict[str, np.ndarray], weights: dict[str, float] | None = None
    ) -> np.ndarray:
        """Rank anomalies by average score."""
        score_matrix = np.array(list(scores.values()))

        if weights is not None:
            weight_array = np.array(
                [weights.get(detector_id, 1.0) for detector_id in scores.keys()]
            )
            final_scores = np.average(score_matrix, axis=0, weights=weight_array)
        else:
            final_scores = np.mean(score_matrix, axis=0)

        return final_scores

    def rank_anomalies_borda_count(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Rank anomalies using Borda count method."""
        score_matrix = np.array(list(scores.values()))
        n_detectors, n_samples = score_matrix.shape

        # Convert scores to ranks for each detector
        rank_matrix = np.zeros_like(score_matrix)
        for i, detector_scores in enumerate(score_matrix):
            # Higher scores get higher ranks
            rank_matrix[i] = np.argsort(np.argsort(detector_scores))

        # Sum ranks (Borda count)
        borda_scores = np.sum(rank_matrix, axis=0)

        # Normalize to [0, 1]
        borda_scores = borda_scores / np.max(borda_scores)

        return borda_scores

    def rank_anomalies_consensus(
        self, scores: dict[str, np.ndarray], threshold: float = 0.7
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rank anomalies by consensus level."""
        score_matrix = np.array(list(scores.values()))

        # Calculate consensus: how many detectors agree this is anomalous
        binary_matrix = (score_matrix > threshold).astype(int)
        consensus_levels = np.mean(binary_matrix, axis=0)

        # Final score combines consensus and average score
        avg_scores = np.mean(score_matrix, axis=0)
        consensus_weighted_scores = consensus_levels * avg_scores

        return consensus_weighted_scores, consensus_levels

    def rank_anomalies_concordance(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Rank anomalies by concordance (agreement in relative ordering)."""
        score_matrix = np.array(list(scores.values()))
        n_detectors, n_samples = score_matrix.shape

        # Calculate pairwise concordance for each sample
        concordance_scores = np.zeros(n_samples)

        for i in range(n_samples):
            sample_scores = score_matrix[:, i]

            # Calculate how well this sample's relative ranking agrees across detectors
            concordance = 0
            comparisons = 0

            for j in range(n_samples):
                if i != j:
                    other_scores = score_matrix[:, j]

                    # Count agreements in pairwise comparisons
                    agreements = np.sum(
                        (sample_scores > other_scores)
                        == (np.mean(sample_scores) > np.mean(other_scores))
                    )
                    concordance += agreements
                    comparisons += n_detectors

            concordance_scores[i] = concordance / comparisons if comparisons > 0 else 0

        return concordance_scores

    def rank_anomalies_geometric_mean(
        self, scores: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Rank anomalies using geometric mean (good for multiplicative effects)."""
        score_matrix = np.array(list(scores.values()))

        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        score_matrix = score_matrix + epsilon

        # Geometric mean
        log_scores = np.log(score_matrix)
        mean_log_scores = np.mean(log_scores, axis=0)
        geometric_means = np.exp(mean_log_scores)

        # Normalize
        geometric_means = (geometric_means - np.min(geometric_means)) / (
            np.max(geometric_means) - np.min(geometric_means) + epsilon
        )

        return geometric_means

    def rank_anomalies_harmonic_mean(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        """Rank anomalies using harmonic mean (penalizes low scores)."""
        score_matrix = np.array(list(scores.values()))

        # Add epsilon to avoid division by zero
        epsilon = 1e-8
        score_matrix = score_matrix + epsilon

        # Harmonic mean
        reciprocal_sum = np.sum(1 / score_matrix, axis=0)
        harmonic_means = len(score_matrix) / reciprocal_sum

        # Normalize
        harmonic_means = (harmonic_means - np.min(harmonic_means)) / (
            np.max(harmonic_means) - np.min(harmonic_means) + epsilon
        )

        return harmonic_means

    def calculate_confidence_intervals(
        self, scores: dict[str, np.ndarray]
    ) -> list[ConfidenceInterval]:
        """Calculate confidence intervals for anomaly scores."""
        score_matrix = np.array(list(scores.values()))

        mean_scores = np.mean(score_matrix, axis=0)
        std_scores = np.std(score_matrix, axis=0)

        confidence_intervals = []
        for i in range(len(mean_scores)):
            # 95% confidence interval
            margin = 1.96 * std_scores[i] / np.sqrt(len(score_matrix))
            lower = max(0, mean_scores[i] - margin)
            upper = min(1, mean_scores[i] + margin)

            ci = ConfidenceInterval(lower=lower, upper=upper, confidence_level=0.95)
            confidence_intervals.append(ci)

        return confidence_intervals

    async def rank_anomalies(
        self,
        detector_ids: list[str],
        dataset: Dataset,
        method: RankingMethod,
        true_labels: np.ndarray | None = None,
    ) -> list[RankedAnomaly]:
        """Comprehensive anomaly ranking using specified method."""

        # Get predictions and scores
        predictions, scores = await self.get_detector_predictions(detector_ids, dataset)

        # Calculate performance weights
        weights = self.calculate_performance_weights(predictions, true_labels)

        # Apply ranking method
        if method == RankingMethod.AVERAGE_SCORE:
            final_scores = self.rank_anomalies_average_score(scores)
            consensus_levels = np.ones(len(final_scores))

        elif method == RankingMethod.WEIGHTED_AVERAGE:
            final_scores = self.rank_anomalies_average_score(scores, weights)
            consensus_levels = np.ones(len(final_scores))

        elif method == RankingMethod.BORDA_COUNT:
            final_scores = self.rank_anomalies_borda_count(scores)
            consensus_levels = np.ones(len(final_scores))

        elif method == RankingMethod.CONSENSUS_RANK:
            final_scores, consensus_levels = self.rank_anomalies_consensus(scores)

        elif method == RankingMethod.CONCORDANCE_RANK:
            final_scores = self.rank_anomalies_concordance(scores)
            consensus_levels = np.ones(len(final_scores))

        elif method == RankingMethod.GEOMETRIC_MEAN:
            final_scores = self.rank_anomalies_geometric_mean(scores)
            consensus_levels = np.ones(len(final_scores))

        elif method == RankingMethod.HARMONIC_MEAN:
            final_scores = self.rank_anomalies_harmonic_mean(scores)
            consensus_levels = np.ones(len(final_scores))

        elif method == RankingMethod.MAXIMUM_SCORE:
            score_matrix = np.array(list(scores.values()))
            final_scores = np.max(score_matrix, axis=0)
            consensus_levels = np.ones(len(final_scores))

        elif method == RankingMethod.MINIMUM_SCORE:
            score_matrix = np.array(list(scores.values()))
            final_scores = np.min(score_matrix, axis=0)
            consensus_levels = np.ones(len(final_scores))

        else:
            # Default to average
            final_scores = self.rank_anomalies_average_score(scores)
            consensus_levels = np.ones(len(final_scores))

        # Calculate confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(scores)

        # Create ranked anomalies
        ranked_anomalies = []
        sorted_indices = np.argsort(final_scores)[::-1]  # Sort descending

        for rank, sample_idx in enumerate(sorted_indices):
            # Individual scores for this sample
            individual_scores = [
                scores[detector_id][sample_idx] for detector_id in detector_ids
            ]

            # Detector agreement (which detectors flagged this as anomaly)
            detector_agreement = {}
            for detector_id in detector_ids:
                detector_agreement[detector_id] = (
                    predictions[detector_id][sample_idx] == 1
                )

            # Metadata
            metadata = {
                "ranking_method": method.value,
                "individual_scores_std": np.std(individual_scores),
                "score_range": np.max(individual_scores) - np.min(individual_scores),
                "agreeing_detectors": sum(detector_agreement.values()),
                "weight_contribution": sum(
                    [
                        weights[detector_id] * scores[detector_id][sample_idx]
                        for detector_id in detector_ids
                    ]
                ),
            }

            ranked_anomaly = RankedAnomaly(
                sample_index=sample_idx,
                final_score=final_scores[sample_idx],
                individual_scores=individual_scores,
                rank=rank + 1,
                confidence_interval=confidence_intervals[sample_idx],
                consensus_level=consensus_levels[sample_idx],
                detector_agreement=detector_agreement,
                metadata=metadata,
            )

            ranked_anomalies.append(ranked_anomaly)

        return ranked_anomalies

    def filter_anomalies_top_k(
        self, ranked_anomalies: list[RankedAnomaly], k: int
    ) -> FilteringResult:
        """Filter top-K anomalies."""
        filtered = ranked_anomalies[:k]
        rejection_reasons = {
            anomaly.sample_index: f"Rank {anomaly.rank} > {k}"
            for anomaly in ranked_anomalies[k:]
        }

        statistics = {
            "total_candidates": len(ranked_anomalies),
            "selected": len(filtered),
            "rejection_rate": (len(ranked_anomalies) - len(filtered))
            / len(ranked_anomalies),
            "score_range_selected": (
                (filtered[0].final_score, filtered[-1].final_score)
                if filtered
                else (0, 0)
            ),
        }

        return FilteringResult(
            filtered_anomalies=filtered,
            filter_statistics=statistics,
            threshold_used=filtered[-1].final_score if filtered else 0,
            rejection_reasons=rejection_reasons,
        )

    def filter_anomalies_confidence_threshold(
        self, ranked_anomalies: list[RankedAnomaly], confidence_threshold: float = 0.8
    ) -> FilteringResult:
        """Filter anomalies based on confidence threshold."""
        filtered = []
        rejection_reasons = {}

        for anomaly in ranked_anomalies:
            # Use confidence interval width as confidence measure
            confidence = 1 - anomaly.confidence_interval.width()

            if confidence >= confidence_threshold:
                filtered.append(anomaly)
            else:
                rejection_reasons[
                    anomaly.sample_index
                ] = f"Confidence {confidence:.3f} < {confidence_threshold}"

        statistics = {
            "total_candidates": len(ranked_anomalies),
            "selected": len(filtered),
            "avg_confidence_selected": (
                np.mean([1 - a.confidence_interval.width() for a in filtered])
                if filtered
                else 0
            ),
            "confidence_threshold": confidence_threshold,
        }

        return FilteringResult(
            filtered_anomalies=filtered,
            filter_statistics=statistics,
            threshold_used=confidence_threshold,
            rejection_reasons=rejection_reasons,
        )

    def filter_anomalies_consensus_level(
        self, ranked_anomalies: list[RankedAnomaly], min_consensus: float = 0.6
    ) -> FilteringResult:
        """Filter anomalies based on consensus level."""
        filtered = []
        rejection_reasons = {}

        for anomaly in ranked_anomalies:
            if anomaly.consensus_level >= min_consensus:
                filtered.append(anomaly)
            else:
                rejection_reasons[
                    anomaly.sample_index
                ] = f"Consensus {anomaly.consensus_level:.3f} < {min_consensus}"

        statistics = {
            "total_candidates": len(ranked_anomalies),
            "selected": len(filtered),
            "avg_consensus_selected": (
                np.mean([a.consensus_level for a in filtered]) if filtered else 0
            ),
            "min_consensus": min_consensus,
        }

        return FilteringResult(
            filtered_anomalies=filtered,
            filter_statistics=statistics,
            threshold_used=min_consensus,
            rejection_reasons=rejection_reasons,
        )

    def filter_anomalies_statistical(
        self, ranked_anomalies: list[RankedAnomaly], z_threshold: float = 2.0
    ) -> FilteringResult:
        """Filter anomalies using statistical outlier detection."""
        if not ranked_anomalies:
            return FilteringResult([], {}, 0, {})

        scores = [a.final_score for a in ranked_anomalies]
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        filtered = []
        rejection_reasons = {}

        for anomaly in ranked_anomalies:
            z_score = (anomaly.final_score - mean_score) / (std_score + 1e-8)

            if z_score >= z_threshold:
                filtered.append(anomaly)
            else:
                rejection_reasons[
                    anomaly.sample_index
                ] = f"Z-score {z_score:.3f} < {z_threshold}"

        statistics = {
            "total_candidates": len(ranked_anomalies),
            "selected": len(filtered),
            "mean_score": mean_score,
            "std_score": std_score,
            "z_threshold": z_threshold,
        }

        return FilteringResult(
            filtered_anomalies=filtered,
            filter_statistics=statistics,
            threshold_used=mean_score + z_threshold * std_score,
            rejection_reasons=rejection_reasons,
        )

    def filter_anomalies_combined(
        self,
        ranked_anomalies: list[RankedAnomaly],
        top_k: int = 50,
        min_consensus: float = 0.5,
        confidence_threshold: float = 0.7,
    ) -> FilteringResult:
        """Combined filtering using multiple criteria."""
        # Stage 1: Top-K
        stage1 = self.filter_anomalies_top_k(ranked_anomalies, top_k)

        # Stage 2: Consensus level
        stage2 = self.filter_anomalies_consensus_level(
            stage1.filtered_anomalies, min_consensus
        )

        # Stage 3: Confidence threshold
        stage3 = self.filter_anomalies_confidence_threshold(
            stage2.filtered_anomalies, confidence_threshold
        )

        # Combine rejection reasons
        all_rejections = {}
        all_rejections.update(stage1.rejection_reasons)
        all_rejections.update(stage2.rejection_reasons)
        all_rejections.update(stage3.rejection_reasons)

        statistics = {
            "total_candidates": len(ranked_anomalies),
            "after_top_k": len(stage1.filtered_anomalies),
            "after_consensus": len(stage2.filtered_anomalies),
            "final_selected": len(stage3.filtered_anomalies),
            "combined_criteria": f"Top-{top_k} AND consensus≥{min_consensus} AND confidence≥{confidence_threshold}",
        }

        return FilteringResult(
            filtered_anomalies=stage3.filtered_anomalies,
            filter_statistics=statistics,
            threshold_used=stage3.threshold_used,
            rejection_reasons=all_rejections,
        )


def create_multi_pattern_dataset():
    """Create dataset with multiple anomaly patterns for comprehensive testing."""
    np.random.seed(42)

    # Large normal population
    n_normal = 2000

    # Normal data with different patterns
    normal_1 = np.random.randn(800, 8)  # Standard
    normal_2 = np.random.randn(600, 8) * 0.5 + [1, 1, 0, 0, 0, 0, 0, 0]  # Dense cluster
    normal_3 = np.random.randn(600, 8)  # Correlated features
    normal_3[:, 1] = normal_3[:, 0] * 0.8 + np.random.randn(600) * 0.2
    normal_3[:, 2] = -normal_3[:, 0] * 0.6 + np.random.randn(600) * 0.3

    normal_data = np.vstack([normal_1, normal_2, normal_3])

    # Diverse anomaly patterns
    anomaly_patterns = {
        "extreme_outliers": np.random.randn(30, 8) * 4 + [6, 6, 6, 6, 6, 6, 6, 6],
        "moderate_outliers": np.random.randn(25, 8) * 2 + [3, 3, 3, 3, 0, 0, 0, 0],
        "subtle_outliers": normal_1[:20] + [1, 2, 0.5, 0, 0, 0, 0, 0],
        "cluster_outliers": np.random.randn(20, 8) * 0.3 + [-4, -4, 0, 0, 0, 0, 0, 0],
        "correlation_breaks": np.array(
            [
                [x, -x * 2, y, z, w, v, u, t]
                for x, y, z, w, v, u, t in zip(
                    np.linspace(4, 7, 25),
                    *[np.random.randn(25) for _ in range(6)],
                    strict=False,
                )
            ]
        ),
        "multi_dimensional": np.random.randn(15, 8) * [0.5, 3, 0.5, 3, 0.5, 3, 0.5, 3]
        + [0, 0, 4, 0, 4, 0, 4, 0],
        "contextual": np.array(
            [
                [np.sin(i / 3), np.cos(i / 3), i / 8, *np.random.randn(5)]
                for i in range(30, 50)
            ]
        )
        * 3,
    }

    # Assign severity levels
    severity_mapping = {
        "extreme_outliers": "high",
        "moderate_outliers": "medium",
        "subtle_outliers": "low",
        "cluster_outliers": "medium",
        "correlation_breaks": "high",
        "multi_dimensional": "high",
        "contextual": "medium",
    }

    # Combine data
    all_anomalies = np.vstack(list(anomaly_patterns.values()))
    data = np.vstack([normal_data, all_anomalies])

    # Create labels and metadata
    labels = np.zeros(len(data))
    severity_labels = ["normal"] * n_normal

    current_idx = n_normal
    for pattern_name, pattern_data in anomaly_patterns.items():
        pattern_length = len(pattern_data)
        labels[current_idx : current_idx + pattern_length] = 1
        severity_labels.extend([severity_mapping[pattern_name]] * pattern_length)
        current_idx += pattern_length

    # Shuffle
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    severity_labels = [severity_labels[i] for i in indices]

    # Create DataFrame
    columns = [f"feature_{i + 1}" for i in range(8)]
    df = pd.DataFrame(data, columns=columns)

    return df, labels, severity_labels, anomaly_patterns


async def create_specialized_detectors(container, dataset: Dataset) -> list[str]:
    """Create specialized detectors for ranking demonstration."""
    detector_configs = [
        {
            "name": "IsolationForest (Extreme)",
            "algorithm": "IsolationForest",
            "params": {
                "contamination": 0.05,
                "n_estimators": 300,
                "max_samples": 256,
                "random_state": 42,
            },
            "specialty": "extreme_outliers",
        },
        {
            "name": "LOF (Local)",
            "algorithm": "LOF",
            "params": {"contamination": 0.08, "n_neighbors": 35},
            "specialty": "local_outliers",
        },
        {
            "name": "COPOD (Statistical)",
            "algorithm": "COPOD",
            "params": {"contamination": 0.07},
            "specialty": "statistical_outliers",
        },
        {
            "name": "KNN (Distance)",
            "algorithm": "KNN",
            "params": {"contamination": 0.08, "n_neighbors": 25, "method": "largest"},
            "specialty": "distance_based",
        },
        {
            "name": "PCA (Subspace)",
            "algorithm": "PCA",
            "params": {"contamination": 0.06, "n_components": 5, "weighted": True},
            "specialty": "correlation_breaks",
        },
        {
            "name": "CBLOF (Cluster)",
            "algorithm": "CBLOF",
            "params": {"contamination": 0.07, "n_clusters": 12, "use_weights": True},
            "specialty": "cluster_outliers",
        },
        {
            "name": "HBOS (Histogram)",
            "algorithm": "HBOS",
            "params": {"contamination": 0.06, "n_bins": 15, "alpha": 0.1},
            "specialty": "multi_dimensional",
        },
        {
            "name": "OCSVM (Boundary)",
            "algorithm": "OCSVM",
            "params": {"contamination": 0.05, "kernel": "rbf", "gamma": "scale"},
            "specialty": "boundary_detection",
        },
    ]

    detector_ids = []
    detector_repo = container.detector_repository()
    detection_service = container.detection_service()

    print("\n2. Creating specialized detectors for ranking analysis...")

    for config in detector_configs:
        print(f"   Training {config['name']} (specialty: {config['specialty']})...")

        detector = Detector(
            name=config["name"],
            algorithm=config["algorithm"],
            parameters=config["params"],
            metadata={
                "specialty": config["specialty"],
                "optimization": "ranking_analysis",
                "expected_performance": config.get("expected_performance", "medium"),
            },
        )
        detector_repo.save(detector)

        try:
            await detection_service.train_detector(
                detector_id=detector.id, dataset=dataset
            )
            detector_ids.append(detector.id)
        except Exception as e:
            print(f"   Warning: Failed to train {config['name']}: {e}")

    print(f"   Successfully trained {len(detector_ids)} specialized detectors")
    return detector_ids


async def demonstrate_ranking_methods(
    ranking_system: AnomalyRankingSystem,
    detector_ids: list[str],
    dataset: Dataset,
    true_labels: np.ndarray,
):
    """Demonstrate all ranking methods."""
    print("\n3. Comprehensive ranking method comparison...")

    ranking_methods = [
        RankingMethod.AVERAGE_SCORE,
        RankingMethod.WEIGHTED_AVERAGE,
        RankingMethod.BORDA_COUNT,
        RankingMethod.CONSENSUS_RANK,
        RankingMethod.CONCORDANCE_RANK,
        RankingMethod.GEOMETRIC_MEAN,
        RankingMethod.HARMONIC_MEAN,
    ]

    ranking_results = {}

    for method in ranking_methods:
        print(f"\n   Ranking method: {method.value}")

        ranked_anomalies = await ranking_system.rank_anomalies(
            detector_ids, dataset, method, true_labels
        )

        ranking_results[method] = ranked_anomalies

        # Analyze top anomalies
        top_10 = ranked_anomalies[:10]

        # Calculate precision at 10
        true_anomalies_top10 = sum(
            [1 for a in top_10 if true_labels[a.sample_index] == 1]
        )
        precision_at_10 = true_anomalies_top10 / 10

        print(f"   - Top 10 precision: {precision_at_10:.3f}")
        print(
            f"   - Score range: {top_10[0].final_score:.3f} - {top_10[-1].final_score:.3f}"
        )
        print(
            f"   - Avg consensus in top 10: {np.mean([a.consensus_level for a in top_10]):.3f}"
        )
        print(
            f"   - Avg confidence width: {np.mean([a.confidence_interval.width() for a in top_10]):.3f}"
        )

    return ranking_results


async def demonstrate_filtering_strategies(
    ranking_system: AnomalyRankingSystem, ranked_anomalies: list[RankedAnomaly]
):
    """Demonstrate filtering strategies."""
    print("\n4. Advanced filtering strategy comparison...")

    filtering_strategies = [
        ("Top-20", FilteringStrategy.TOP_K, {"k": 20}),
        (
            "High Confidence",
            FilteringStrategy.CONFIDENCE_THRESHOLD,
            {"confidence_threshold": 0.8},
        ),
        ("Strong Consensus", FilteringStrategy.CONSENSUS_LEVEL, {"min_consensus": 0.7}),
        (
            "Statistical Filter",
            FilteringStrategy.STATISTICAL_FILTER,
            {"z_threshold": 2.5},
        ),
        (
            "Combined Filter",
            FilteringStrategy.COMBINED_FILTER,
            {"top_k": 50, "min_consensus": 0.6, "confidence_threshold": 0.75},
        ),
    ]

    for strategy_name, strategy, params in filtering_strategies:
        print(f"\n   Strategy: {strategy_name}")

        if strategy == FilteringStrategy.TOP_K:
            result = ranking_system.filter_anomalies_top_k(
                ranked_anomalies, params["k"]
            )
        elif strategy == FilteringStrategy.CONFIDENCE_THRESHOLD:
            result = ranking_system.filter_anomalies_confidence_threshold(
                ranked_anomalies, params["confidence_threshold"]
            )
        elif strategy == FilteringStrategy.CONSENSUS_LEVEL:
            result = ranking_system.filter_anomalies_consensus_level(
                ranked_anomalies, params["min_consensus"]
            )
        elif strategy == FilteringStrategy.STATISTICAL_FILTER:
            result = ranking_system.filter_anomalies_statistical(
                ranked_anomalies, params["z_threshold"]
            )
        elif strategy == FilteringStrategy.COMBINED_FILTER:
            result = ranking_system.filter_anomalies_combined(
                ranked_anomalies,
                params["top_k"],
                params["min_consensus"],
                params["confidence_threshold"],
            )

        print(f"   - Selected: {result.filter_statistics['selected']} anomalies")
        print(
            f"   - Rejection rate: {result.filter_statistics.get('rejection_rate', 0):.1%}"
        )
        print(f"   - Threshold used: {result.threshold_used:.3f}")

        if result.filtered_anomalies:
            avg_score = np.mean([a.final_score for a in result.filtered_anomalies])
            print(f"   - Average score of selected: {avg_score:.3f}")


async def analyze_ranking_effectiveness(
    ranking_results: dict, true_labels: np.ndarray, severity_labels: list[str]
):
    """Analyze effectiveness of different ranking methods."""
    print("\n5. Ranking effectiveness analysis...")

    # Calculate metrics for different top-K values
    top_k_values = [5, 10, 20, 50]

    print("\n   Precision at different K values:")
    print("   Method\\K\t\t" + "\t".join([f"@{k}" for k in top_k_values]))
    print("   " + "-" * 60)

    for method, ranked_anomalies in ranking_results.items():
        precisions = []

        for k in top_k_values:
            top_k_anomalies = ranked_anomalies[:k]
            true_anomalies_in_top_k = sum(
                [1 for a in top_k_anomalies if true_labels[a.sample_index] == 1]
            )
            precision = true_anomalies_in_top_k / k
            precisions.append(precision)

        precision_str = "\t".join([f"{p:.3f}" for p in precisions])
        print(f"   {method.value:<20}\t{precision_str}")

    # Severity analysis
    print("\n   Severity level detection in top 20:")
    print("   Method\\Severity\t\tHigh\tMedium\tLow")
    print("   " + "-" * 50)

    for method, ranked_anomalies in ranking_results.items():
        top_20 = ranked_anomalies[:20]

        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for anomaly in top_20:
            if true_labels[anomaly.sample_index] == 1:
                severity = severity_labels[anomaly.sample_index]
                if severity in severity_counts:
                    severity_counts[severity] += 1

        print(
            f"   {method.value:<20}\t\t{severity_counts['high']}\t{severity_counts['medium']}\t{severity_counts['low']}"
        )


async def main():
    """Demonstrate anomaly ranking and filtering strategies."""
    print("🎯 Pynomaly Anomaly Ranking and Filtering Strategies\n")
    print("This example demonstrates comprehensive anomaly ranking and filtering:")
    print("• Multiple ranking algorithms (Borda count, consensus, concordance)")
    print("• Performance-weighted aggregation with statistical measures")
    print("• Confidence interval-based uncertainty quantification")
    print("• Multi-criteria filtering strategies with combined approaches")
    print("• Severity-aware analysis and precision evaluation")
    print("• Production-ready filtering workflows with rejection tracking\n")

    # Initialize container
    container = create_container()

    # Create comprehensive dataset
    print("1. Creating multi-pattern dataset for ranking analysis...")
    (
        data,
        true_labels,
        severity_labels,
        anomaly_patterns,
    ) = create_multi_pattern_dataset()
    print(f"   Dataset: {len(data)} samples, {data.shape[1]} features")
    print(
        f"   True anomalies: {np.sum(true_labels)} ({np.mean(true_labels) * 100:.1f}%)"
    )
    print("   Anomaly patterns included:")
    for pattern_name, pattern_data in anomaly_patterns.items():
        print(f"   - {pattern_name}: {len(pattern_data)} samples")

    # Severity distribution
    severity_dist = {
        sev: severity_labels.count(sev) for sev in ["normal", "low", "medium", "high"]
    }
    print(f"   Severity distribution: {severity_dist}")

    # Create dataset entity
    dataset = Dataset(
        name="Multi-Pattern Ranking Dataset",
        data=data,
        metadata={
            "anomaly_patterns": list(anomaly_patterns.keys()),
            "n_true_anomalies": int(np.sum(true_labels)),
            "severity_distribution": severity_dist,
            "analysis_purpose": "ranking_and_filtering",
        },
    )

    # Save dataset
    dataset_repo = container.dataset_repository()
    dataset_repo.save(dataset)

    # Create specialized detectors
    detector_ids = await create_specialized_detectors(container, dataset)

    # Initialize ranking system
    ranking_system = AnomalyRankingSystem(container)

    # Demonstrate ranking methods
    ranking_results = await demonstrate_ranking_methods(
        ranking_system, detector_ids, dataset, true_labels
    )

    # Use best ranking method for filtering demonstration
    best_method = RankingMethod.WEIGHTED_AVERAGE
    best_ranked_anomalies = ranking_results[best_method]

    # Demonstrate filtering strategies
    await demonstrate_filtering_strategies(ranking_system, best_ranked_anomalies)

    # Analyze ranking effectiveness
    await analyze_ranking_effectiveness(ranking_results, true_labels, severity_labels)

    # Production workflow example
    print("\n6. Production-ready workflow example...")

    # Stage 1: Rank using consensus method
    consensus_ranked = ranking_results[RankingMethod.CONSENSUS_RANK]

    # Stage 2: Apply combined filtering
    production_filter = ranking_system.filter_anomalies_combined(
        consensus_ranked, top_k=30, min_consensus=0.6, confidence_threshold=0.75
    )

    print("   Production workflow results:")
    print(f"   - Input candidates: {len(consensus_ranked)}")
    print(
        f"   - After combined filtering: {production_filter.filter_statistics['final_selected']}"
    )
    print(f"   - Final anomalies selected: {len(production_filter.filtered_anomalies)}")

    if production_filter.filtered_anomalies:
        # Calculate precision of final selection
        final_true_anomalies = sum(
            [
                1
                for a in production_filter.filtered_anomalies
                if true_labels[a.sample_index] == 1
            ]
        )
        final_precision = final_true_anomalies / len(
            production_filter.filtered_anomalies
        )

        print(f"   - Final precision: {final_precision:.3f}")
        print(
            f"   - Score range: {production_filter.filtered_anomalies[0].final_score:.3f} - {production_filter.filtered_anomalies[-1].final_score:.3f}"
        )

        # Top 5 anomalies for manual review
        print("\n   Top 5 anomalies for manual review:")
        for i, anomaly in enumerate(production_filter.filtered_anomalies[:5]):
            true_label = (
                "ANOMALY" if true_labels[anomaly.sample_index] == 1 else "NORMAL"
            )
            severity = severity_labels[anomaly.sample_index]
            agreeing_detectors = anomaly.metadata["agreeing_detectors"]

            print(
                f"   {i + 1}. Sample {anomaly.sample_index}: Score={anomaly.final_score:.3f}, "
                f"Consensus={anomaly.consensus_level:.3f}, Detectors={agreeing_detectors}/8, "
                f"Actual={true_label}, Severity={severity}"
            )

    print("\n🎯 Anomaly ranking and filtering analysis completed!")
    print("\nThis demonstrates how sophisticated ranking and filtering can prioritize")
    print("anomalies for efficient manual review in production environments.")


if __name__ == "__main__":
    asyncio.run(main())
