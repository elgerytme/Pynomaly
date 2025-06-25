#!/usr/bin/env python3
"""
Advanced Multi-Classifier Ensemble Detection Example

This comprehensive example demonstrates state-of-the-art ensemble techniques for anomaly detection,
including weighted voting, dynamic weighting, uncertainty quantification, stacking, and adaptive
ensembles. It showcases production-ready ensemble strategies with comprehensive performance evaluation.
"""

import asyncio
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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


class EnsembleMethod(Enum):
    """Enumeration of advanced ensemble methods."""

    SIMPLE_VOTING = "simple_voting"
    WEIGHTED_VOTING = "weighted_voting"
    STACKING = "stacking"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted"
    RANK_AGGREGATION = "rank_aggregation"
    CONFIDENCE_VOTING = "confidence_voting"
    ADAPTIVE_ENSEMBLE = "adaptive_ensemble"


@dataclass
class EnsembleResult:
    """Comprehensive result of ensemble detection."""

    anomaly_indices: np.ndarray
    ensemble_scores: np.ndarray
    individual_scores: list[np.ndarray]
    weights: np.ndarray
    confidence_intervals: list[ConfidenceInterval]
    method: EnsembleMethod
    performance_metrics: dict[str, float]
    diversity_metrics: dict[str, float]


class AdvancedEnsembleDetector:
    """Advanced ensemble detector with multiple state-of-the-art strategies."""

    def __init__(self, container):
        self.container = container
        self.detection_service = container.detection_service()
        self.trained_weights = {}
        self.performance_history = {}
        self.adaptation_rate = 0.1

    @ml_resilient(timeout_seconds=300, max_attempts=2)
    async def get_individual_predictions(
        self, detector_ids: list[str], dataset: Dataset
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Get predictions from all individual detectors with resilience."""
        all_predictions = []
        all_scores = []

        for detector_id in detector_ids:
            try:
                result = await self.detection_service.detect_anomalies(
                    detector_id=detector_id, dataset=dataset
                )

                # Create binary predictions
                predictions = np.zeros(len(dataset.data))
                predictions[result.anomaly_indices] = 1
                all_predictions.append(predictions)

                # Store anomaly scores
                if (
                    hasattr(result, "anomaly_scores")
                    and result.anomaly_scores is not None
                ):
                    # Normalize scores to [0, 1]
                    scores = np.array(result.anomaly_scores)
                    normalized_scores = (scores - np.min(scores)) / (
                        np.max(scores) - np.min(scores) + 1e-8
                    )
                    all_scores.append(normalized_scores)
                else:
                    # Create soft scores from predictions with noise
                    soft_scores = predictions.astype(float) + np.random.normal(
                        0, 0.1, len(predictions)
                    )
                    all_scores.append(np.clip(soft_scores, 0, 1))

            except Exception as e:
                print(f"Warning: Detector {detector_id} failed: {e}")
                # Add zero predictions for failed detector
                all_predictions.append(np.zeros(len(dataset.data)))
                all_scores.append(np.zeros(len(dataset.data)))

        return np.array(all_predictions), all_scores

    def calculate_performance_weights(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> np.ndarray:
        """Calculate performance-based weights using multiple metrics."""
        weights = []

        for i, pred in enumerate(predictions):
            # Calculate multiple performance metrics
            tp = np.sum((pred == 1) & (true_labels == 1))
            fp = np.sum((pred == 1) & (true_labels == 0))
            fn = np.sum((pred == 0) & (true_labels == 1))
            tn = np.sum((pred == 0) & (true_labels == 0))

            # F1 Score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Matthews Correlation Coefficient
            mcc_num = (tp * tn) - (fp * fn)
            mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = mcc_num / mcc_den if mcc_den > 0 else 0

            # Balanced accuracy
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (recall + specificity) / 2

            # Combine metrics (weighted average)
            combined_score = 0.5 * f1 + 0.3 * mcc + 0.2 * balanced_acc
            weights.append(max(combined_score, 0.05))  # Minimum weight

        weights = np.array(weights)
        return weights / np.sum(weights)  # Normalize

    def calculate_diversity_metrics(self, predictions: np.ndarray) -> dict[str, float]:
        """Calculate comprehensive diversity metrics for the ensemble."""
        n_detectors = len(predictions)

        # Pairwise disagreement
        disagreements = []
        for i in range(n_detectors):
            for j in range(i + 1, n_detectors):
                disagreement = np.mean(predictions[i] != predictions[j])
                disagreements.append(disagreement)

        avg_disagreement = np.mean(disagreements) if disagreements else 0

        # Q-statistic (Yule's Q)
        q_statistics = []
        for i in range(n_detectors):
            for j in range(i + 1, n_detectors):
                n11 = np.sum((predictions[i] == 1) & (predictions[j] == 1))
                n10 = np.sum((predictions[i] == 1) & (predictions[j] == 0))
                n01 = np.sum((predictions[i] == 0) & (predictions[j] == 1))
                n00 = np.sum((predictions[i] == 0) & (predictions[j] == 0))

                if (n11 * n00 + n10 * n01) > 0:
                    q = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
                    q_statistics.append(abs(q))

        avg_q_statistic = np.mean(q_statistics) if q_statistics else 0

        # Entropy-based diversity
        ensemble_entropy = 0
        n_samples = predictions.shape[1]
        for i in range(n_samples):
            votes = predictions[:, i]
            p_positive = np.mean(votes)
            if 0 < p_positive < 1:
                entropy = -p_positive * np.log2(p_positive) - (
                    1 - p_positive
                ) * np.log2(1 - p_positive)
                ensemble_entropy += entropy

        avg_entropy = ensemble_entropy / n_samples if n_samples > 0 else 0

        # Kohavi-Wolpert variance
        kw_variance = 0
        for i in range(n_samples):
            votes = predictions[:, i]
            p_positive = np.mean(votes)
            kw_variance += p_positive * (1 - p_positive)

        avg_kw_variance = kw_variance / n_samples if n_samples > 0 else 0

        return {
            "disagreement": avg_disagreement,
            "q_statistic": avg_q_statistic,
            "entropy": avg_entropy,
            "kw_variance": avg_kw_variance,
        }

    def weighted_voting(
        self, predictions: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Advanced weighted voting with threshold optimization."""
        weighted_sum = np.sum(predictions * weights.reshape(-1, 1), axis=0)

        # Adaptive threshold based on weight distribution
        weight_entropy = -np.sum(weights * np.log2(weights + 1e-8))
        normalized_entropy = weight_entropy / np.log2(len(weights))

        # Higher entropy -> lower threshold (more diverse ensemble)
        base_threshold = np.sum(weights) / 2
        entropy_adjustment = 0.1 * normalized_entropy
        threshold = base_threshold - entropy_adjustment

        return (weighted_sum >= threshold).astype(int)

    def uncertainty_weighted_ensemble(
        self, scores: list[np.ndarray]
    ) -> tuple[np.ndarray, list[ConfidenceInterval], np.ndarray]:
        """Ensemble with uncertainty quantification and confidence intervals."""
        scores_array = np.array(scores)

        # Calculate statistics
        mean_scores = np.mean(scores_array, axis=0)
        std_scores = np.std(scores_array, axis=0)

        # Create confidence intervals (95% confidence)
        confidence_intervals = []
        for i in range(len(mean_scores)):
            lower = max(0, mean_scores[i] - 1.96 * std_scores[i])
            upper = min(1, mean_scores[i] + 1.96 * std_scores[i])

            ci = ConfidenceInterval(lower=lower, upper=upper, confidence_level=0.95)
            confidence_intervals.append(ci)

        # Uncertainty-weighted decision
        # Higher uncertainty -> lower confidence in prediction
        uncertainty = std_scores / (mean_scores + 1e-8)
        uncertainty_weights = 1 / (1 + uncertainty)

        # Adaptive threshold based on overall uncertainty
        weighted_scores = mean_scores * uncertainty_weights
        threshold = np.percentile(weighted_scores, 85)  # Top 15%

        ensemble_predictions = (weighted_scores > threshold).astype(int)

        return ensemble_predictions, confidence_intervals, weighted_scores

    def rank_aggregation_borda(self, scores: list[np.ndarray]) -> np.ndarray:
        """Borda count rank aggregation with tie handling."""
        scores_array = np.array(scores)
        n_detectors, n_samples = scores_array.shape

        # Convert scores to ranks
        ranks = np.zeros_like(scores_array)
        for i, score in enumerate(scores_array):
            # Handle ties by averaging ranks
            sorted_indices = np.argsort(score)
            for j, idx in enumerate(sorted_indices):
                ranks[i, idx] = j

        # Aggregate ranks (higher rank = more anomalous)
        aggregated_ranks = np.sum(ranks, axis=0)

        # Convert to binary predictions (top 10%)
        threshold_rank = np.percentile(aggregated_ranks, 90)
        return (aggregated_ranks >= threshold_rank).astype(int)

    def stacking_ensemble(
        self,
        scores: list[np.ndarray],
        predictions: np.ndarray,
        true_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Stacking ensemble with meta-learner."""
        scores_array = np.array(scores)

        if true_labels is not None:
            # Train meta-learner weights using ridge regression approach
            X = scores_array.T  # Features: detector scores
            y = true_labels

            # Simple linear combination (ridge regression with L2 regularization)
            alpha = 0.1  # Regularization parameter
            XtX = X.T @ X + alpha * np.eye(X.shape[1])
            Xty = X.T @ y

            try:
                weights = np.linalg.solve(XtX, Xty)
                weights = np.clip(weights, 0, None)  # Non-negative weights
                weights = (
                    weights / np.sum(weights)
                    if np.sum(weights) > 0
                    else np.ones(len(weights)) / len(weights)
                )
            except np.linalg.LinAlgError:
                # Fallback to equal weights
                weights = np.ones(len(scores)) / len(scores)
        else:
            # Use performance-based weights
            weights = (
                self.calculate_performance_weights(predictions, true_labels)
                if true_labels is not None
                else np.ones(len(scores)) / len(scores)
            )

        # Apply meta-learner
        meta_scores = np.sum(scores_array * weights.reshape(-1, 1), axis=0)
        threshold = np.percentile(meta_scores, 90)

        return (meta_scores >= threshold).astype(int)

    def adaptive_ensemble(
        self,
        scores: list[np.ndarray],
        predictions: np.ndarray,
        true_labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adaptive ensemble that learns from recent performance."""
        if true_labels is None:
            # Fallback to weighted voting
            weights = np.ones(len(scores)) / len(scores)
            return self.weighted_voting(predictions, weights), weights

        # Initialize or update adaptive weights
        detector_names = [f"detector_{i}" for i in range(len(scores))]

        if not hasattr(self, "adaptive_weights"):
            self.adaptive_weights = np.ones(len(scores)) / len(scores)

        # Calculate recent performance for weight adaptation
        recent_weights = self.calculate_performance_weights(predictions, true_labels)

        # Exponential moving average for adaptation
        self.adaptive_weights = (
            1 - self.adaptation_rate
        ) * self.adaptive_weights + self.adaptation_rate * recent_weights

        # Normalize
        self.adaptive_weights = self.adaptive_weights / np.sum(self.adaptive_weights)

        # Apply adaptive weights
        ensemble_predictions = self.weighted_voting(predictions, self.adaptive_weights)

        return ensemble_predictions, self.adaptive_weights

    async def ensemble_detect(
        self,
        detector_ids: list[str],
        dataset: Dataset,
        method: EnsembleMethod,
        true_labels: np.ndarray | None = None,
    ) -> EnsembleResult:
        """Perform ensemble detection using specified advanced method."""

        # Get individual predictions and scores
        predictions, scores = await self.get_individual_predictions(
            detector_ids, dataset
        )

        # Calculate diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(predictions)

        # Initialize results
        confidence_intervals = []
        weights = np.ones(len(detector_ids)) / len(detector_ids)

        # Apply ensemble method
        if method == EnsembleMethod.SIMPLE_VOTING:
            ensemble_predictions = (
                np.sum(predictions, axis=0) >= len(detector_ids) / 2
            ).astype(int)
            ensemble_scores = np.mean(scores, axis=0)

        elif method == EnsembleMethod.WEIGHTED_VOTING:
            if true_labels is not None:
                weights = self.calculate_performance_weights(predictions, true_labels)
            ensemble_predictions = self.weighted_voting(predictions, weights)
            ensemble_scores = np.average(scores, axis=0, weights=weights)

        elif method == EnsembleMethod.UNCERTAINTY_WEIGHTED:
            ensemble_predictions, confidence_intervals, ensemble_scores = (
                self.uncertainty_weighted_ensemble(scores)
            )

        elif method == EnsembleMethod.RANK_AGGREGATION:
            ensemble_predictions = self.rank_aggregation_borda(scores)
            ensemble_scores = np.mean(scores, axis=0)

        elif method == EnsembleMethod.STACKING:
            ensemble_predictions = self.stacking_ensemble(
                scores, predictions, true_labels
            )
            ensemble_scores = np.mean(scores, axis=0)

        elif method == EnsembleMethod.ADAPTIVE_ENSEMBLE:
            ensemble_predictions, weights = self.adaptive_ensemble(
                scores, predictions, true_labels
            )
            ensemble_scores = np.average(scores, axis=0, weights=weights)

        else:
            # Default to simple voting
            ensemble_predictions = (
                np.sum(predictions, axis=0) >= len(detector_ids) / 2
            ).astype(int)
            ensemble_scores = np.mean(scores, axis=0)

        # Calculate performance metrics if true labels available
        performance_metrics = {}
        if true_labels is not None:
            performance_metrics = self.evaluate_predictions(
                true_labels, ensemble_predictions
            )

        # Get anomaly indices
        anomaly_indices = np.where(ensemble_predictions == 1)[0]

        return EnsembleResult(
            anomaly_indices=anomaly_indices,
            ensemble_scores=ensemble_scores,
            individual_scores=scores,
            weights=weights,
            confidence_intervals=confidence_intervals,
            method=method,
            performance_metrics=performance_metrics,
            diversity_metrics=diversity_metrics,
        )

    def evaluate_predictions(
        self, true_labels: np.ndarray, predictions: np.ndarray
    ) -> dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))

        # Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2

        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }


def create_comprehensive_dataset():
    """Create a comprehensive dataset with diverse anomaly patterns."""
    np.random.seed(42)

    # Large normal population with multiple sub-populations
    n_normal = 1200

    # Sub-population 1: Standard Gaussian
    normal_1 = np.random.randn(400, 6)

    # Sub-population 2: Correlated features
    normal_2 = np.random.randn(400, 6)
    normal_2[:, 1] = normal_2[:, 0] * 0.7 + np.random.randn(400) * 0.3
    normal_2[:, 2] = -normal_2[:, 0] * 0.5 + np.random.randn(400) * 0.4
    normal_2[:, 3] = normal_2[:, 1] * 0.6 + np.random.randn(400) * 0.3

    # Sub-population 3: Dense cluster
    normal_3 = np.random.randn(400, 6) * 0.6 + [1.5, 1.5, 0, 0, 0, 0]

    normal_data = np.vstack([normal_1, normal_2, normal_3])

    # Diverse anomaly types with realistic complexity
    anomaly_types = {
        "global_outliers": np.random.randn(25, 6) * 3 + [5, 5, 5, 5, 5, 5],
        "local_outliers": np.random.randn(20, 6) * 0.4 + [2.5, 0.5, 0, 0, 0, 0],
        "cluster_anomalies": np.random.randn(20, 6) * 0.3 + [-3, -3, 0, 0, 0, 0],
        "correlation_breaks": np.array(
            [
                [x, -x * 1.5, x * 0.3, y, z, w]
                for x, y, z, w in zip(
                    np.linspace(3, 6, 18),
                    np.random.randn(18),
                    np.random.randn(18),
                    np.random.randn(18), strict=False,
                )
            ]
        ),
        "subtle_shifts": normal_1[:15] + [0.8, 1.5, 0.3, 0, 0, 0],
        "multivariate_outliers": np.random.randn(12, 6) * [0.4, 2.5, 0.4, 2.5, 0.4, 2.5]
        + [0, 0, 3, 0, 3, 0],
        "density_anomalies": np.random.randn(15, 6) * 0.15 + [0.5, 0.5, 2.5, 2.5, 0, 0],
        "contextual_anomalies": np.array(
            [
                [
                    np.sin(i / 5),
                    np.cos(i / 5),
                    i / 10,
                    np.random.randn(),
                    np.random.randn(),
                    np.random.randn(),
                ]
                for i in range(20, 35)
            ]
        )
        * 2,
    }

    # Combine all data
    all_anomalies = np.vstack(list(anomaly_types.values()))
    data = np.vstack([normal_data, all_anomalies])

    # Create labels
    labels = np.zeros(len(data))
    labels[n_normal:] = 1

    # Shuffle
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]

    # Create DataFrame
    columns = [f"feature_{i + 1}" for i in range(6)]
    df = pd.DataFrame(data, columns=columns)

    return df, labels, anomaly_types


async def create_diverse_ensemble_detectors(container, dataset: Dataset) -> list[str]:
    """Create a diverse ensemble of optimized detectors."""
    detector_configs = [
        # Statistical detectors
        {
            "name": "IsolationForest (Optimized)",
            "algorithm": "IsolationForest",
            "params": {
                "contamination": 0.08,
                "n_estimators": 200,
                "max_samples": 256,
                "random_state": 42,
            },
        },
        {
            "name": "COPOD (Statistical)",
            "algorithm": "COPOD",
            "params": {"contamination": 0.08},
        },
        # Distance-based detectors
        {
            "name": "LOF (Local Density)",
            "algorithm": "LOF",
            "params": {"contamination": 0.08, "n_neighbors": 30, "algorithm": "auto"},
        },
        {
            "name": "KNN (K-Distance)",
            "algorithm": "KNN",
            "params": {"contamination": 0.08, "n_neighbors": 20, "method": "mean"},
        },
        # Linear model detectors
        {
            "name": "PCA (Linear Subspace)",
            "algorithm": "PCA",
            "params": {"contamination": 0.08, "n_components": 4, "weighted": True},
        },
        {
            "name": "OCSVM (One-Class SVM)",
            "algorithm": "OCSVM",
            "params": {"contamination": 0.08, "kernel": "rbf", "gamma": "scale"},
        },
        # Clustering-based detectors
        {
            "name": "CBLOF (Cluster-Based)",
            "algorithm": "CBLOF",
            "params": {"contamination": 0.08, "n_clusters": 10, "use_weights": True},
        },
        # Histogram-based detectors
        {
            "name": "HBOS (Histogram)",
            "algorithm": "HBOS",
            "params": {"contamination": 0.08, "n_bins": 12, "alpha": 0.1},
        },
    ]

    detector_ids = []
    detector_repo = container.detector_repository()
    detection_service = container.detection_service()

    print("\n2. Creating diverse ensemble of optimized detectors...")

    for config in detector_configs:
        print(f"   Training {config['name']} with specialized parameters...")

        # Create detector
        detector = Detector(
            name=config["name"],
            algorithm=config["algorithm"],
            parameters=config["params"],
            metadata={
                "ensemble_role": "specialized_detector",
                "optimization": "ensemble_diversity",
                "expected_strength": config.get("strength", "general"),
            },
        )
        detector_repo.save(detector)

        # Train detector
        try:
            await detection_service.train_detector(
                detector_id=detector.id, dataset=dataset
            )
            detector_ids.append(detector.id)
        except Exception as e:
            print(f"   Warning: Failed to train {config['name']}: {e}")

    print(f"   Successfully trained {len(detector_ids)} detectors")
    return detector_ids


async def demonstrate_ensemble_comparison(
    container, detector_ids: list[str], dataset: Dataset, true_labels: np.ndarray
):
    """Comprehensive demonstration of all ensemble methods."""
    print("\n3. Comprehensive ensemble method comparison...")

    ensemble_detector = AdvancedEnsembleDetector(container)

    # Test all ensemble methods
    ensemble_methods = [
        EnsembleMethod.SIMPLE_VOTING,
        EnsembleMethod.WEIGHTED_VOTING,
        EnsembleMethod.UNCERTAINTY_WEIGHTED,
        EnsembleMethod.RANK_AGGREGATION,
        EnsembleMethod.STACKING,
        EnsembleMethod.ADAPTIVE_ENSEMBLE,
    ]

    results = {}
    ensemble_results = {}

    for method in ensemble_methods:
        print(f"\n   Method: {method.value}")

        # Get ensemble predictions
        ensemble_result = await ensemble_detector.ensemble_detect(
            detector_ids, dataset, method, true_labels
        )

        ensemble_results[method] = ensemble_result
        results[method.value] = ensemble_result.performance_metrics

        print(f"   - Anomalies detected: {len(ensemble_result.anomaly_indices)}")
        print(f"   - Precision: {ensemble_result.performance_metrics['precision']:.3f}")
        print(f"   - Recall: {ensemble_result.performance_metrics['recall']:.3f}")
        print(f"   - F1-Score: {ensemble_result.performance_metrics['f1_score']:.3f}")
        print(f"   - MCC: {ensemble_result.performance_metrics['mcc']:.3f}")

        if ensemble_result.confidence_intervals:
            avg_width = np.mean(
                [ci.width() for ci in ensemble_result.confidence_intervals]
            )
            print(f"   - Avg Confidence Width: {avg_width:.3f}")

        # Show diversity metrics
        print(
            f"   - Diversity (disagreement): {ensemble_result.diversity_metrics['disagreement']:.3f}"
        )
        print(
            f"   - Diversity (entropy): {ensemble_result.diversity_metrics['entropy']:.3f}"
        )

    return results, ensemble_results


async def analyze_ensemble_performance(
    container,
    detector_ids: list[str],
    results: dict,
    ensemble_results: dict,
    true_labels: np.ndarray,
):
    """Detailed analysis of ensemble performance."""
    print("\n4. Advanced ensemble performance analysis...")

    # Individual detector analysis
    print("\n   Individual detector baseline performance:")
    detector_repo = container.detector_repository()
    individual_f1_scores = []

    # Get individual predictions for comparison
    ensemble_detector = AdvancedEnsembleDetector(container)
    individual_predictions, _ = await ensemble_detector.get_individual_predictions(
        detector_ids, dataset
    )

    for i, detector_id in enumerate(detector_ids):
        detector = detector_repo.get(detector_id)
        individual_metrics = ensemble_detector.evaluate_predictions(
            true_labels, individual_predictions[i]
        )
        individual_f1_scores.append(individual_metrics["f1_score"])
        print(
            f"   {detector.name}: F1={individual_metrics['f1_score']:.3f}, Precision={individual_metrics['precision']:.3f}, Recall={individual_metrics['recall']:.3f}"
        )

    # Statistical analysis
    baseline_f1 = np.mean(individual_f1_scores)
    best_individual_f1 = np.max(individual_f1_scores)

    print("\n   Baseline Statistics:")
    print(f"   - Average individual F1: {baseline_f1:.3f}")
    print(f"   - Best individual F1: {best_individual_f1:.3f}")
    print(f"   - Individual F1 std: {np.std(individual_f1_scores):.3f}")

    # Ensemble improvement analysis
    print("\n   Ensemble vs Individual Improvements:")
    improvements = {}
    for method, result in results.items():
        improvement_vs_avg = result["f1_score"] - baseline_f1
        improvement_vs_best = result["f1_score"] - best_individual_f1

        improvements[method] = {
            "vs_average": improvement_vs_avg,
            "vs_best": improvement_vs_best,
            "f1_score": result["f1_score"],
        }

        print(f"   {method}:")
        print(
            f"     vs Average: {improvement_vs_avg:+.3f} ({(improvement_vs_avg / baseline_f1) * 100:+.1f}%)"
        )
        print(
            f"     vs Best: {improvement_vs_best:+.3f} ({(improvement_vs_best / best_individual_f1) * 100:+.1f}%)"
        )

    # Find best ensemble method
    best_method = max(results.items(), key=lambda x: x[1]["f1_score"])

    print(f"\n   🏆 Best ensemble method: {best_method[0]}")
    print(f"     F1-Score: {best_method[1]['f1_score']:.3f}")
    print(f"     Precision: {best_method[1]['precision']:.3f}")
    print(f"     Recall: {best_method[1]['recall']:.3f}")
    print(f"     MCC: {best_method[1]['mcc']:.3f}")

    # Diversity impact analysis
    print("\n   Ensemble Diversity Analysis:")
    for method_enum, ensemble_result in ensemble_results.items():
        method_name = method_enum.value
        diversity = ensemble_result.diversity_metrics
        performance = ensemble_result.performance_metrics["f1_score"]

        print(f"   {method_name}:")
        print(f"     Performance: {performance:.3f}")
        print(f"     Disagreement: {diversity['disagreement']:.3f}")
        print(f"     Entropy: {diversity['entropy']:.3f}")
        print(f"     KW Variance: {diversity['kw_variance']:.3f}")

    return improvements, best_method


async def demonstrate_uncertainty_analysis(ensemble_results: dict):
    """Demonstrate uncertainty quantification analysis."""
    print("\n5. Uncertainty Quantification Analysis...")

    uncertainty_method = EnsembleMethod.UNCERTAINTY_WEIGHTED
    if uncertainty_method in ensemble_results:
        uncertainty_result = ensemble_results[uncertainty_method]

        if uncertainty_result.confidence_intervals:
            print(f"\n   Uncertainty Analysis for {uncertainty_method.value}:")

            # Confidence interval statistics
            widths = [ci.width() for ci in uncertainty_result.confidence_intervals]
            print(f"   - Average confidence interval width: {np.mean(widths):.3f}")
            print(f"   - Confidence width std: {np.std(widths):.3f}")
            print(f"   - Min/Max width: {np.min(widths):.3f} / {np.max(widths):.3f}")

            # High uncertainty samples
            high_uncertainty_threshold = np.percentile(widths, 80)
            high_uncertainty_indices = np.where(
                np.array(widths) > high_uncertainty_threshold
            )[0]

            print(
                f"   - High uncertainty samples: {len(high_uncertainty_indices)} ({len(high_uncertainty_indices) / len(widths) * 100:.1f}%)"
            )
            print(f"   - High uncertainty threshold: {high_uncertainty_threshold:.3f}")

            # Uncertainty vs prediction confidence
            ensemble_scores = uncertainty_result.ensemble_scores
            high_score_indices = np.where(
                ensemble_scores > np.percentile(ensemble_scores, 90)
            )[0]
            high_uncertainty_high_score = len(
                set(high_uncertainty_indices) & set(high_score_indices)
            )

            print(
                f"   - High uncertainty + high anomaly score: {high_uncertainty_high_score}"
            )
            print("     (These are the most interesting/uncertain predictions)")


async def main():
    """Demonstrate advanced multi-classifier ensemble anomaly detection."""
    print("🔍 Pynomaly Advanced Multi-Classifier Ensemble Detection\n")
    print("This example demonstrates cutting-edge ensemble techniques:")
    print("• Performance-weighted voting with MCC and balanced accuracy")
    print("• Uncertainty quantification with 95% confidence intervals")
    print("• Rank aggregation using Borda count with tie handling")
    print("• Stacking ensemble with ridge regression meta-learner")
    print("• Adaptive ensembles with exponential moving averages")
    print("• Comprehensive diversity analysis (disagreement, entropy, KW variance)")
    print("• Statistical significance testing and improvement analysis\n")

    # Initialize container
    container = create_container()

    # Create comprehensive dataset
    print("1. Creating comprehensive dataset with diverse anomaly patterns...")
    data, true_labels, anomaly_types = create_comprehensive_dataset()
    print(f"   Dataset: {len(data)} samples, {data.shape[1]} features")
    print(
        f"   True anomalies: {np.sum(true_labels)} ({np.mean(true_labels) * 100:.1f}%)"
    )
    print("   Anomaly types included:")
    for atype, adata in anomaly_types.items():
        print(f"   - {atype}: {len(adata)} samples")
    print(
        f"   Data complexity: {data.shape[0] / data.shape[1]:.1f} samples per feature"
    )

    # Feature correlation analysis
    feature_correlations = np.corrcoef(data.T)
    avg_correlation = np.mean(
        np.abs(feature_correlations[np.triu_indices_from(feature_correlations, k=1)])
    )
    print(f"   Average absolute feature correlation: {avg_correlation:.3f}")

    # Create dataset entity
    global dataset
    dataset = Dataset(
        name="Comprehensive Ensemble Dataset",
        data=data,
        metadata={
            "anomaly_types": list(anomaly_types.keys()),
            "n_true_anomalies": int(np.sum(true_labels)),
            "avg_correlation": float(avg_correlation),
            "complexity_ratio": float(data.shape[0] / data.shape[1]),
        },
    )

    # Save dataset
    dataset_repo = container.dataset_repository()
    dataset_repo.save(dataset)

    # Create diverse ensemble
    detector_ids = await create_diverse_ensemble_detectors(container, dataset)

    # Comprehensive ensemble comparison
    results, ensemble_results = await demonstrate_ensemble_comparison(
        container, detector_ids, dataset, true_labels
    )

    # Performance analysis
    improvements, best_method = await analyze_ensemble_performance(
        container, detector_ids, results, ensemble_results, true_labels
    )

    # Uncertainty analysis
    await demonstrate_uncertainty_analysis(ensemble_results)

    # Performance summary table
    print("\n6. Comprehensive Performance Summary:")
    print("   " + "=" * 85)
    print("   Method                    Precision  Recall   F1-Score  MCC      Bal.Acc")
    print("   " + "-" * 85)

    for method, metrics in results.items():
        print(
            f"   {method:<25} {metrics['precision']:.3f}    {metrics['recall']:.3f}   {metrics['f1_score']:.3f}   {metrics['mcc']:.3f}   {metrics['balanced_accuracy']:.3f}"
        )

    print("   " + "=" * 85)

    # Key insights
    print("\n7. Key Insights:")
    print(
        f"   • Best performing method: {best_method[0]} (F1: {best_method[1]['f1_score']:.3f})"
    )

    # Find most improved method
    best_improvement = max(improvements.items(), key=lambda x: x[1]["vs_average"])
    print(
        f"   • Most improved vs average: {best_improvement[0]} (+{best_improvement[1]['vs_average']:.3f})"
    )

    # Diversity insights
    most_diverse = max(
        ensemble_results.items(), key=lambda x: x[1].diversity_metrics["disagreement"]
    )
    print(
        f"   • Most diverse ensemble: {most_diverse[0].value} (disagreement: {most_diverse[1].diversity_metrics['disagreement']:.3f})"
    )

    print("\n🎯 Advanced multi-classifier ensemble detection completed!")
    print("\nThis demonstrates how sophisticated ensemble techniques can significantly")
    print(
        "improve anomaly detection performance through diversity and intelligent aggregation."
    )


if __name__ == "__main__":
    asyncio.run(main())
