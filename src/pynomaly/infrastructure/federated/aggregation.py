"""Advanced aggregation algorithms for federated learning."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances

from pynomaly.domain.models.federated import (
    AggregationMethod,
    FederatedParticipant,
    ModelUpdate,
)


class AggregationStrategy(ABC):
    """Abstract base class for federated aggregation strategies."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def aggregate(
        self,
        updates: Dict[UUID, ModelUpdate],
        participants: Dict[UUID, FederatedParticipant],
        **kwargs,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Aggregate model updates from participants.

        Returns:
            Tuple of (aggregated_parameters, aggregation_metrics)
        """
        pass

    def _get_parameter_names(self, updates: Dict[UUID, ModelUpdate]) -> List[str]:
        """Get parameter names from updates."""
        if not updates:
            return []

        first_update = next(iter(updates.values()))
        return list(first_update.parameters.keys())

    def _validate_updates(self, updates: Dict[UUID, ModelUpdate]) -> bool:
        """Validate that all updates have consistent parameter structure."""
        if not updates:
            return False

        first_update = next(iter(updates.values()))
        first_param_names = set(first_update.parameters.keys())

        for update in updates.values():
            if set(update.parameters.keys()) != first_param_names:
                return False

            # Check parameter shapes
            for param_name in first_param_names:
                if (
                    update.parameters[param_name].shape
                    != first_update.parameters[param_name].shape
                ):
                    return False

        return True


class FederatedAveragingAggregation(AggregationStrategy):
    """Standard Federated Averaging (FedAvg) aggregation."""

    def __init__(self):
        super().__init__("FederatedAveraging")

    async def aggregate(
        self,
        updates: Dict[UUID, ModelUpdate],
        participants: Dict[UUID, FederatedParticipant],
        **kwargs,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Aggregate using weighted average based on data size."""

        if not self._validate_updates(updates):
            raise ValueError("Invalid or inconsistent model updates")

        # Calculate weights based on data size
        total_data_size = sum(
            participants[pid].data_size for pid in updates.keys() if pid in participants
        )

        if total_data_size == 0:
            # Fallback to simple average
            weights = {pid: 1.0 / len(updates) for pid in updates.keys()}
        else:
            weights = {
                pid: participants[pid].data_size / total_data_size
                for pid in updates.keys()
                if pid in participants
            }

        # Aggregate parameters
        param_names = self._get_parameter_names(updates)
        aggregated_params = {}

        for param_name in param_names:
            weighted_sum = None

            for participant_id, update in updates.items():
                if participant_id in weights:
                    param_value = update.parameters[param_name]
                    weight = weights[participant_id]

                    weighted_param = param_value * weight

                    if weighted_sum is None:
                        weighted_sum = weighted_param
                    else:
                        weighted_sum += weighted_param

            if weighted_sum is not None:
                aggregated_params[param_name] = weighted_sum

        # Calculate aggregation metrics
        metrics = {
            "num_participants": len(updates),
            "total_data_size": total_data_size,
            "weight_entropy": self._calculate_weight_entropy(weights),
            "parameter_count": sum(p.size for p in aggregated_params.values()),
        }

        return aggregated_params, metrics

    def _calculate_weight_entropy(self, weights: Dict[UUID, float]) -> float:
        """Calculate entropy of participant weights."""
        weight_values = list(weights.values())
        if not weight_values:
            return 0.0

        # Normalize weights
        total_weight = sum(weight_values)
        if total_weight == 0:
            return 0.0

        normalized_weights = [w / total_weight for w in weight_values]

        # Calculate entropy
        entropy = -sum(w * np.log(w + 1e-10) for w in normalized_weights if w > 0)
        return entropy


class TrimmedMeanAggregation(AggregationStrategy):
    """Trimmed mean aggregation for Byzantine robustness."""

    def __init__(self, trim_ratio: float = 0.2):
        super().__init__("TrimmedMean")
        self.trim_ratio = trim_ratio

    async def aggregate(
        self,
        updates: Dict[UUID, ModelUpdate],
        participants: Dict[UUID, FederatedParticipant],
        **kwargs,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Aggregate using trimmed mean to handle outliers."""

        if not self._validate_updates(updates):
            raise ValueError("Invalid or inconsistent model updates")

        param_names = self._get_parameter_names(updates)
        aggregated_params = {}

        for param_name in param_names:
            # Collect all parameter values
            param_values = []
            for update in updates.values():
                param_values.append(update.parameters[param_name])

            if param_values:
                # Calculate trimmed mean
                param_array = np.array(param_values)
                aggregated_params[param_name] = stats.trim_mean(
                    param_array, self.trim_ratio, axis=0
                )

        # Calculate robustness metrics
        outlier_count = self._count_outliers(updates, aggregated_params)

        metrics = {
            "num_participants": len(updates),
            "trim_ratio": self.trim_ratio,
            "outlier_count": outlier_count,
            "robustness_score": 1.0 - (outlier_count / len(updates)),
        }

        return aggregated_params, metrics

    def _count_outliers(
        self,
        updates: Dict[UUID, ModelUpdate],
        aggregated_params: Dict[str, np.ndarray],
    ) -> int:
        """Count outlier participants based on parameter deviation."""

        if not updates or not aggregated_params:
            return 0

        # Calculate deviation for each participant
        participant_deviations = {}

        for participant_id, update in updates.items():
            total_deviation = 0.0
            param_count = 0

            for param_name, aggregated_param in aggregated_params.items():
                if param_name in update.parameters:
                    participant_param = update.parameters[param_name]
                    deviation = np.linalg.norm(participant_param - aggregated_param)
                    total_deviation += deviation
                    param_count += 1

            if param_count > 0:
                participant_deviations[participant_id] = total_deviation / param_count

        if not participant_deviations:
            return 0

        # Use IQR method to identify outliers
        deviations = list(participant_deviations.values())
        q75, q25 = np.percentile(deviations, [75, 25])
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr

        outlier_count = sum(
            1 for deviation in deviations if deviation > outlier_threshold
        )

        return outlier_count


class KrumAggregation(AggregationStrategy):
    """Krum aggregation for Byzantine-robust federated learning."""

    def __init__(self, num_byzantines: int = 1):
        super().__init__("Krum")
        self.num_byzantines = num_byzantines

    async def aggregate(
        self,
        updates: Dict[UUID, ModelUpdate],
        participants: Dict[UUID, FederatedParticipant],
        **kwargs,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Aggregate using Krum algorithm."""

        if not self._validate_updates(updates):
            raise ValueError("Invalid or inconsistent model updates")

        if len(updates) <= 2 * self.num_byzantines:
            raise ValueError("Insufficient participants for Krum aggregation")

        # Convert updates to flat vectors for distance calculation
        participant_ids = list(updates.keys())
        param_vectors = []

        for participant_id in participant_ids:
            update = updates[participant_id]
            # Flatten all parameters into single vector
            flat_params = np.concatenate(
                [param.flatten() for param in update.parameters.values()]
            )
            param_vectors.append(flat_params)

        param_vectors = np.array(param_vectors)

        # Calculate pairwise distances
        distances = pairwise_distances(param_vectors, metric="euclidean")

        # Calculate Krum scores
        krum_scores = {}
        num_closest = len(participant_ids) - self.num_byzantines - 2

        for i, participant_id in enumerate(participant_ids):
            # Get distances to other participants
            participant_distances = distances[i]
            # Exclude self (distance = 0)
            other_distances = np.concatenate(
                [participant_distances[:i], participant_distances[i + 1 :]]
            )

            # Sum of distances to num_closest participants
            closest_distances = np.partition(other_distances, num_closest)[:num_closest]
            krum_scores[participant_id] = np.sum(closest_distances)

        # Select participant with minimum Krum score
        selected_participant = min(krum_scores.keys(), key=lambda x: krum_scores[x])
        selected_update = updates[selected_participant]

        # Return selected participant's parameters
        aggregated_params = selected_update.parameters.copy()

        metrics = {
            "num_participants": len(updates),
            "num_byzantines": self.num_byzantines,
            "selected_participant": str(selected_participant),
            "krum_score": krum_scores[selected_participant],
            "score_variance": np.var(list(krum_scores.values())),
        }

        return aggregated_params, metrics


class MultiKrumAggregation(KrumAggregation):
    """Multi-Krum aggregation selecting multiple participants."""

    def __init__(self, num_byzantines: int = 1, num_selected: int = 3):
        super().__init__(num_byzantines)
        self.name = "MultiKrum"
        self.num_selected = num_selected

    async def aggregate(
        self,
        updates: Dict[UUID, ModelUpdate],
        participants: Dict[UUID, FederatedParticipant],
        **kwargs,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Aggregate using Multi-Krum algorithm."""

        if not self._validate_updates(updates):
            raise ValueError("Invalid or inconsistent model updates")

        if len(updates) <= 2 * self.num_byzantines:
            raise ValueError("Insufficient participants for Multi-Krum aggregation")

        if self.num_selected > len(updates) - self.num_byzantines:
            self.num_selected = len(updates) - self.num_byzantines

        # Calculate Krum scores (reuse parent implementation)
        _, parent_metrics = await super().aggregate(updates, participants)

        # Convert updates to flat vectors
        participant_ids = list(updates.keys())
        param_vectors = []

        for participant_id in participant_ids:
            update = updates[participant_id]
            flat_params = np.concatenate(
                [param.flatten() for param in update.parameters.values()]
            )
            param_vectors.append(flat_params)

        param_vectors = np.array(param_vectors)
        distances = pairwise_distances(param_vectors, metric="euclidean")

        # Calculate Krum scores
        krum_scores = {}
        num_closest = len(participant_ids) - self.num_byzantines - 2

        for i, participant_id in enumerate(participant_ids):
            participant_distances = distances[i]
            other_distances = np.concatenate(
                [participant_distances[:i], participant_distances[i + 1 :]]
            )
            closest_distances = np.partition(other_distances, num_closest)[:num_closest]
            krum_scores[participant_id] = np.sum(closest_distances)

        # Select top participants with lowest Krum scores
        sorted_participants = sorted(krum_scores.keys(), key=lambda x: krum_scores[x])
        selected_participants = sorted_participants[: self.num_selected]

        # Average the selected participants' parameters
        param_names = self._get_parameter_names(updates)
        aggregated_params = {}

        for param_name in param_names:
            param_sum = None

            for participant_id in selected_participants:
                param_value = updates[participant_id].parameters[param_name]

                if param_sum is None:
                    param_sum = param_value.copy()
                else:
                    param_sum += param_value

            if param_sum is not None:
                aggregated_params[param_name] = param_sum / len(selected_participants)

        metrics = {
            "num_participants": len(updates),
            "num_byzantines": self.num_byzantines,
            "num_selected": self.num_selected,
            "selected_participants": [str(pid) for pid in selected_participants],
            "avg_krum_score": np.mean(
                [krum_scores[pid] for pid in selected_participants]
            ),
            "score_variance": np.var(list(krum_scores.values())),
        }

        return aggregated_params, metrics


class GeometricMedianAggregation(AggregationStrategy):
    """Geometric median aggregation for robust federated learning."""

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        super().__init__("GeometricMedian")
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    async def aggregate(
        self,
        updates: Dict[UUID, ModelUpdate],
        participants: Dict[UUID, FederatedParticipant],
        **kwargs,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Aggregate using geometric median."""

        if not self._validate_updates(updates):
            raise ValueError("Invalid or inconsistent model updates")

        param_names = self._get_parameter_names(updates)
        aggregated_params = {}

        for param_name in param_names:
            # Collect parameter values
            param_values = []
            for update in updates.values():
                param_values.append(update.parameters[param_name].flatten())

            param_matrix = np.array(param_values)

            # Calculate geometric median
            median_params = self._geometric_median(param_matrix)

            # Reshape back to original shape
            original_shape = (
                updates[next(iter(updates.keys()))].parameters[param_name].shape
            )
            aggregated_params[param_name] = median_params.reshape(original_shape)

        # Calculate robustness metrics
        total_deviation = self._calculate_total_deviation(updates, aggregated_params)

        metrics = {
            "num_participants": len(updates),
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "total_deviation": total_deviation,
            "robustness_score": 1.0 / (1.0 + total_deviation),
        }

        return aggregated_params, metrics

    def _geometric_median(self, points: np.ndarray) -> np.ndarray:
        """Calculate geometric median using Weiszfeld's algorithm."""

        # Initialize with arithmetic mean
        median = np.mean(points, axis=0)

        for iteration in range(self.max_iterations):
            # Calculate distances
            distances = np.linalg.norm(points - median, axis=1)

            # Avoid division by zero
            distances = np.maximum(distances, 1e-8)

            # Calculate weights
            weights = 1.0 / distances
            weight_sum = np.sum(weights)

            # Update median
            new_median = np.sum((weights.reshape(-1, 1) * points), axis=0) / weight_sum

            # Check convergence
            if np.linalg.norm(new_median - median) < self.tolerance:
                break

            median = new_median

        return median

    def _calculate_total_deviation(
        self,
        updates: Dict[UUID, ModelUpdate],
        aggregated_params: Dict[str, np.ndarray],
    ) -> float:
        """Calculate total deviation from aggregated parameters."""

        total_deviation = 0.0

        for update in updates.values():
            participant_deviation = 0.0

            for param_name, aggregated_param in aggregated_params.items():
                if param_name in update.parameters:
                    participant_param = update.parameters[param_name]
                    deviation = np.linalg.norm(participant_param - aggregated_param)
                    participant_deviation += deviation

            total_deviation += participant_deviation

        return total_deviation


class FederatedAggregationService:
    """Service for managing federated aggregation strategies."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Register available aggregation strategies
        self.strategies = {
            AggregationMethod.WEIGHTED_AVERAGE: FederatedAveragingAggregation(),
            AggregationMethod.SIMPLE_AVERAGE: FederatedAveragingAggregation(),
            AggregationMethod.TRIMMED_MEAN: TrimmedMeanAggregation(),
            AggregationMethod.BYZANTINE_RESILIENT: KrumAggregation(),
        }

        # Advanced strategies
        self.advanced_strategies = {
            "krum": KrumAggregation(),
            "multi_krum": MultiKrumAggregation(),
            "geometric_median": GeometricMedianAggregation(),
            "trimmed_mean_20": TrimmedMeanAggregation(trim_ratio=0.2),
            "trimmed_mean_30": TrimmedMeanAggregation(trim_ratio=0.3),
        }

    async def aggregate(
        self,
        method: AggregationMethod,
        updates: Dict[UUID, ModelUpdate],
        participants: Dict[UUID, FederatedParticipant],
        **kwargs,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Aggregate model updates using specified method."""

        if method not in self.strategies:
            raise ValueError(f"Unsupported aggregation method: {method}")

        strategy = self.strategies[method]

        try:
            aggregated_params, metrics = await strategy.aggregate(
                updates, participants, **kwargs
            )

            self.logger.info(
                f"Aggregated {len(updates)} updates using {strategy.name}: "
                f"{metrics}"
            )

            return aggregated_params, metrics

        except Exception as e:
            self.logger.error(f"Aggregation failed with {strategy.name}: {e}")
            raise

    async def aggregate_advanced(
        self,
        strategy_name: str,
        updates: Dict[UUID, ModelUpdate],
        participants: Dict[UUID, FederatedParticipant],
        **kwargs,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Aggregate using advanced strategy by name."""

        if strategy_name not in self.advanced_strategies:
            raise ValueError(f"Unknown advanced strategy: {strategy_name}")

        strategy = self.advanced_strategies[strategy_name]

        return await strategy.aggregate(updates, participants, **kwargs)

    def get_available_strategies(self) -> List[str]:
        """Get list of available aggregation strategies."""

        basic_strategies = [method.value for method in self.strategies.keys()]
        advanced_strategies = list(self.advanced_strategies.keys())

        return basic_strategies + advanced_strategies

    def benchmark_strategies(
        self,
        updates: Dict[UUID, ModelUpdate],
        participants: Dict[UUID, FederatedParticipant],
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark different aggregation strategies."""

        results = {}

        # Test basic strategies
        for method, strategy in self.strategies.items():
            try:
                import time

                start_time = time.time()

                aggregated_params, metrics = asyncio.run(
                    strategy.aggregate(updates, participants)
                )

                execution_time = time.time() - start_time

                results[method.value] = {
                    "execution_time": execution_time,
                    "metrics": metrics,
                    "success": True,
                }

            except Exception as e:
                results[method.value] = {
                    "execution_time": 0,
                    "metrics": {},
                    "success": False,
                    "error": str(e),
                }

        return results
