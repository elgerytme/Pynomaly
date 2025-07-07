"""
Active learning service for sample selection and model improvement.

This module provides the core active learning algorithms for selecting
the most informative samples for human annotation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from pynomaly.domain.entities.active_learning_session import SamplingStrategy
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.human_feedback import HumanFeedback
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval


class SampleSelectionProtocol(Protocol):
    """Protocol for sample selection strategies."""

    def select_samples(
        self,
        detection_results: List[DetectionResult],
        features: np.ndarray,
        n_samples: int,
    ) -> List[int]:
        """Select sample indices for annotation."""
        ...


class ActiveLearningService:
    """
    Domain service for active learning and human-in-the-loop training.

    Provides various sample selection strategies to identify the most
    informative samples for human annotation.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize active learning service.

        Args:
            random_seed: Random seed for reproducible results
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def select_samples_by_uncertainty(
        self,
        detection_results: List[DetectionResult],
        n_samples: int,
        uncertainty_method: str = "entropy",
    ) -> List[int]:
        """
        Select samples with highest uncertainty for annotation.

        Args:
            detection_results: Detection results to select from
            n_samples: Number of samples to select
            uncertainty_method: Method for measuring uncertainty

        Returns:
            List of indices of selected samples
        """
        if not detection_results:
            return []

        n_samples = min(n_samples, len(detection_results))
        scores = np.array([result.score.value for result in detection_results])

        if uncertainty_method == "entropy":
            # Calculate entropy-based uncertainty
            uncertainties = self._calculate_entropy_uncertainty(scores)
        elif uncertainty_method == "margin":
            # Distance from decision boundary (0.5)
            uncertainties = -np.abs(scores - 0.5)
        elif uncertainty_method == "least_confident":
            # Least confident samples (closest to 0.5)
            uncertainties = -np.minimum(scores, 1 - scores)
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")

        # Select samples with highest uncertainty
        selected_indices = np.argsort(uncertainties)[-n_samples:]
        return selected_indices.tolist()

    def select_samples_by_diversity(
        self, features: np.ndarray, n_samples: int, diversity_method: str = "kmeans"
    ) -> List[int]:
        """
        Select diverse samples to cover the feature space.

        Args:
            features: Feature matrix for samples
            n_samples: Number of samples to select
            diversity_method: Method for measuring diversity

        Returns:
            List of indices of selected samples
        """
        if features.shape[0] == 0:
            return []

        n_samples = min(n_samples, features.shape[0])

        if diversity_method == "kmeans":
            # Use k-means clustering to find diverse samples
            selected_indices = self._select_by_kmeans_diversity(features, n_samples)
        elif diversity_method == "max_distance":
            # Greedy selection based on maximum distance
            selected_indices = self._select_by_max_distance(features, n_samples)
        elif diversity_method == "random":
            # Random selection as baseline
            selected_indices = np.random.choice(
                features.shape[0], size=n_samples, replace=False
            ).tolist()
        else:
            raise ValueError(f"Unknown diversity method: {diversity_method}")

        return selected_indices

    def select_samples_by_committee_disagreement(
        self, ensemble_results: List[List[DetectionResult]], n_samples: int
    ) -> List[int]:
        """
        Select samples where ensemble models disagree most.

        Args:
            ensemble_results: Results from multiple models
            n_samples: Number of samples to select

        Returns:
            List of indices of selected samples
        """
        if not ensemble_results or not ensemble_results[0]:
            return []

        n_samples = min(n_samples, len(ensemble_results[0]))

        # Calculate disagreement for each sample
        disagreements = []
        for i in range(len(ensemble_results[0])):
            sample_scores = [
                model_results[i].score.value for model_results in ensemble_results
            ]
            disagreement = np.std(sample_scores)
            disagreements.append(disagreement)

        disagreements = np.array(disagreements)

        # Select samples with highest disagreement
        selected_indices = np.argsort(disagreements)[-n_samples:]
        return selected_indices.tolist()

    def select_samples_by_expected_model_change(
        self,
        detection_results: List[DetectionResult],
        features: np.ndarray,
        model_gradients: Optional[np.ndarray] = None,
        n_samples: int = 10,
    ) -> List[int]:
        """
        Select samples that would cause the largest model update.

        Args:
            detection_results: Current detection results
            features: Feature matrix
            model_gradients: Gradients for each sample (if available)
            n_samples: Number of samples to select

        Returns:
            List of indices of selected samples
        """
        if not detection_results:
            return []

        n_samples = min(n_samples, len(detection_results))

        if model_gradients is not None:
            # Use provided gradients
            gradient_norms = np.linalg.norm(model_gradients, axis=1)
            selected_indices = np.argsort(gradient_norms)[-n_samples:]
        else:
            # Approximate expected model change using uncertainty
            scores = np.array([result.score.value for result in detection_results])
            uncertainties = self._calculate_entropy_uncertainty(scores)

            # Weight by feature magnitude (samples with larger features may cause bigger changes)
            feature_magnitudes = np.linalg.norm(features, axis=1)
            expected_changes = uncertainties * feature_magnitudes

            selected_indices = np.argsort(expected_changes)[-n_samples:]

        return selected_indices.tolist()

    def combine_selection_strategies(
        self,
        detection_results: List[DetectionResult],
        features: np.ndarray,
        strategies: Dict[SamplingStrategy, float],
        n_samples: int,
    ) -> List[int]:
        """
        Combine multiple selection strategies with weights.

        Args:
            detection_results: Detection results to select from
            features: Feature matrix
            strategies: Dictionary mapping strategies to weights
            n_samples: Number of samples to select

        Returns:
            List of indices of selected samples
        """
        if not detection_results:
            return []

        n_total = len(detection_results)
        combined_scores = np.zeros(n_total)

        # Normalize weights
        total_weight = sum(strategies.values())
        normalized_strategies = {
            strategy: weight / total_weight for strategy, weight in strategies.items()
        }

        for strategy, weight in normalized_strategies.items():
            if strategy == SamplingStrategy.UNCERTAINTY:
                scores = np.array([result.score.value for result in detection_results])
                uncertainties = self._calculate_entropy_uncertainty(scores)
                combined_scores += weight * uncertainties

            elif strategy == SamplingStrategy.DIVERSITY:
                diversity_scores = self._calculate_diversity_scores(features)
                combined_scores += weight * diversity_scores

            elif strategy == SamplingStrategy.MARGIN:
                scores = np.array([result.score.value for result in detection_results])
                margin_scores = -np.abs(scores - 0.5)
                combined_scores += weight * margin_scores

            elif strategy == SamplingStrategy.ENTROPY:
                scores = np.array([result.score.value for result in detection_results])
                entropy_scores = self._calculate_entropy_uncertainty(scores)
                combined_scores += weight * entropy_scores

        # Select top samples
        n_samples = min(n_samples, n_total)
        selected_indices = np.argsort(combined_scores)[-n_samples:]
        return selected_indices.tolist()

    def calculate_annotation_value(
        self,
        sample_index: int,
        detection_results: List[DetectionResult],
        features: np.ndarray,
        existing_feedback: List[HumanFeedback],
    ) -> float:
        """
        Calculate the expected value of annotating a specific sample.

        Args:
            sample_index: Index of sample to evaluate
            detection_results: All detection results
            features: Feature matrix
            existing_feedback: Previously collected feedback

        Returns:
            Expected annotation value score
        """
        if sample_index >= len(detection_results):
            return 0.0

        result = detection_results[sample_index]

        # Base uncertainty value
        uncertainty = self._calculate_entropy_uncertainty(
            np.array([result.score.value])
        )[0]

        # Diversity value (distance to annotated samples)
        diversity_value = self._calculate_diversity_value(
            sample_index, features, existing_feedback
        )

        # Novelty value (different from existing feedback patterns)
        novelty_value = self._calculate_novelty_value(result, existing_feedback)

        # Combine values with weights
        total_value = 0.4 * uncertainty + 0.3 * diversity_value + 0.3 * novelty_value

        return total_value

    def update_model_with_feedback(
        self, feedback_list: List[HumanFeedback], learning_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate model update parameters based on human feedback.

        Args:
            feedback_list: List of human feedback
            learning_rate: Learning rate for updates

        Returns:
            Dictionary with update statistics
        """
        if not feedback_list:
            return {
                "total_corrections": 0,
                "average_confidence": 0.0,
                "update_magnitude": 0.0,
            }

        # Calculate corrections and weights
        corrections = [fb for fb in feedback_list if fb.is_correction()]
        total_corrections = len(corrections)

        # Calculate average confidence
        confidence_values = {"low": 0.3, "medium": 0.6, "high": 0.9, "expert": 1.0}
        avg_confidence = np.mean(
            [confidence_values[fb.confidence.value] for fb in feedback_list]
        )

        # Calculate update magnitude based on corrections and confidence
        if corrections:
            correction_magnitudes = []
            for correction in corrections:
                if correction.original_prediction:
                    corrected_score = correction.get_corrected_score()
                    if corrected_score:
                        magnitude = abs(
                            corrected_score.value - correction.original_prediction.value
                        )
                        weight = correction.get_feedback_weight()
                        correction_magnitudes.append(magnitude * weight)

            update_magnitude = (
                np.mean(correction_magnitudes) if correction_magnitudes else 0.0
            )
        else:
            update_magnitude = 0.0

        return {
            "total_corrections": total_corrections,
            "average_confidence": avg_confidence,
            "update_magnitude": update_magnitude * learning_rate,
            "feedback_quality": avg_confidence,
            "effective_samples": sum(fb.get_feedback_weight() for fb in feedback_list),
        }

    def _calculate_entropy_uncertainty(self, scores: np.ndarray) -> np.ndarray:
        """Calculate entropy-based uncertainty for anomaly scores."""
        # Clip scores to avoid log(0)
        epsilon = 1e-15
        clipped_scores = np.clip(scores, epsilon, 1 - epsilon)

        # Calculate binary entropy
        entropies = -(
            clipped_scores * np.log2(clipped_scores)
            + (1 - clipped_scores) * np.log2(1 - clipped_scores)
        )

        return entropies

    def _select_by_kmeans_diversity(
        self, features: np.ndarray, n_samples: int
    ) -> List[int]:
        """Select diverse samples using k-means clustering."""
        if n_samples >= features.shape[0]:
            return list(range(features.shape[0]))

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_samples, random_state=self.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # Select sample closest to each cluster center
        selected_indices = []
        for cluster_id in range(n_samples):
            cluster_mask = cluster_labels == cluster_id
            if not np.any(cluster_mask):
                continue

            cluster_features = features[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            center = kmeans.cluster_centers_[cluster_id]

            # Find closest sample to cluster center
            distances = np.linalg.norm(cluster_features - center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)

        return selected_indices

    def _select_by_max_distance(
        self, features: np.ndarray, n_samples: int
    ) -> List[int]:
        """Select samples using greedy max-distance approach."""
        selected_indices = []

        # Start with random sample
        selected_indices.append(np.random.randint(0, features.shape[0]))

        # Greedily select samples that are farthest from already selected
        for _ in range(n_samples - 1):
            if len(selected_indices) >= features.shape[0]:
                break

            max_min_distance = -1
            best_candidate = -1

            for candidate in range(features.shape[0]):
                if candidate in selected_indices:
                    continue

                # Calculate minimum distance to already selected samples
                min_distance = float("inf")
                for selected_idx in selected_indices:
                    distance = np.linalg.norm(
                        features[candidate] - features[selected_idx]
                    )
                    min_distance = min(min_distance, distance)

                # Keep track of candidate with maximum minimum distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate

            if best_candidate >= 0:
                selected_indices.append(best_candidate)

        return selected_indices

    def _calculate_diversity_scores(self, features: np.ndarray) -> np.ndarray:
        """Calculate diversity scores for all samples."""
        if features.shape[0] <= 1:
            return np.ones(features.shape[0])

        # Calculate pairwise distances
        distances = pdist(features, metric="euclidean")
        distance_matrix = squareform(distances)

        # Average distance to all other samples as diversity score
        diversity_scores = np.mean(distance_matrix, axis=1)

        # Normalize to [0, 1]
        if np.max(diversity_scores) > np.min(diversity_scores):
            diversity_scores = (diversity_scores - np.min(diversity_scores)) / (
                np.max(diversity_scores) - np.min(diversity_scores)
            )

        return diversity_scores

    def _calculate_diversity_value(
        self,
        sample_index: int,
        features: np.ndarray,
        existing_feedback: List[HumanFeedback],
    ) -> float:
        """Calculate diversity value relative to annotated samples."""
        if not existing_feedback:
            return 1.0  # Maximum diversity if no annotations yet

        # Get indices of annotated samples
        annotated_indices = []
        for feedback in existing_feedback:
            # Try to find the sample index (this would need to be tracked)
            # For now, use a simplified approach
            pass

        if not annotated_indices:
            return 1.0

        # Calculate minimum distance to annotated samples
        sample_features = features[sample_index]
        min_distance = float("inf")

        for annotated_idx in annotated_indices:
            if annotated_idx < features.shape[0]:
                distance = np.linalg.norm(sample_features - features[annotated_idx])
                min_distance = min(min_distance, distance)

        # Normalize distance (higher distance = higher diversity value)
        return min(1.0, min_distance / np.std(features))

    def _calculate_novelty_value(
        self, result: DetectionResult, existing_feedback: List[HumanFeedback]
    ) -> float:
        """Calculate novelty value based on feedback patterns."""
        if not existing_feedback:
            return 1.0  # Maximum novelty if no feedback yet

        # Calculate how different this result is from existing feedback patterns
        existing_scores = []
        for feedback in existing_feedback:
            if feedback.original_prediction:
                existing_scores.append(feedback.original_prediction.value)

        if not existing_scores:
            return 1.0

        # Calculate distance from existing score distribution
        existing_scores = np.array(existing_scores)
        score_distance = min(np.abs(existing_scores - result.score.value))

        # Normalize (higher distance = higher novelty)
        return min(1.0, score_distance * 2)  # Scale factor for sensitivity
