"""Real-time processor ensemble optimization service with dynamic selection and multi-armed bandit algorithms."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from monorepo.application.services.advanced_ensemble_service import (
    AdvancedEnsembleService,
)
from monorepo.application.services.drift_detection_service import DriftDetectionService
from monorepo.domain.entities import Dataset, Detector

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """Ensemble optimization strategies."""

    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "upper_confidence_bound"
    EPSILON_GREEDY = "epsilon_greedy"
    CONTEXTUAL_BANDIT = "contextual_bandit"
    GRADIENT_DESCENT = "gradient_descent"


class ModelArm(BaseModel):
    """Processor arm for multi-armed bandit optimization."""

    processor_id: str
    algorithm_name: str
    total_rewards: float = 0.0
    total_pulls: int = 0
    recent_rewards: list[float] = Field(default_factory=list)
    last_selected: datetime | None = None
    confidence_intervals: tuple[float, float] = (0.0, 1.0)
    context_features: dict[str, float] = Field(default_factory=dict)

    def get_average_reward(self) -> float:
        """Get average reward for this arm."""
        return self.total_rewards / max(self.total_pulls, 1)

    def get_ucb_score(self, total_pulls: int, c: float = 1.0) -> float:
        """Calculate Upper Confidence Bound score."""
        if self.total_pulls == 0:
            return float("inf")  # Prioritize unexplored arms

        avg_reward = self.get_average_reward()
        exploration_term = c * np.sqrt(np.log(total_pulls) / self.total_pulls)
        return avg_reward + exploration_term


@dataclass
class EnsemblePerformanceMetrics:
    """Performance measurements for ensemble tracking."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    prediction_latency: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    throughput: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataContext:
    """Context information about incoming data for contextual bandits."""

    feature_drift_score: float = 0.0
    concept_drift_score: float = 0.0
    data_complexity: float = 0.0
    noise_level: float = 0.0
    sample_size: int = 0
    feature_count: int = 0
    anomaly_density: float = 0.0
    temporal_patterns: dict[str, float] = field(default_factory=dict)


class RealTimeEnsembleOptimizer:
    """Real-time ensemble optimization with dynamic processor selection."""

    def __init__(
        self,
        base_ensemble_service: AdvancedEnsembleService,
        drift_processing_service: DriftDetectionService,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.UCB,
        optimization_window: int = 1000,
        min_samples_for_optimization: int = 100,
        exploration_rate: float = 0.1,
        learning_rate: float = 0.01,
    ):
        """Initialize real-time ensemble optimizer.

        Args:
            base_ensemble_service: Base ensemble service for creating ensembles
            drift_processing_service: Service for detecting data drift
            optimization_strategy: Strategy for ensemble optimization
            optimization_window: Number of predictions to consider for optimization
            min_samples_for_optimization: Minimum samples before optimization starts
            exploration_rate: Exploration rate for epsilon-greedy strategy
            learning_rate: Learning rate for gradient-based optimization
        """
        self.base_ensemble_service = base_ensemble_service
        self.drift_processing_service = drift_processing_service
        self.optimization_strategy = optimization_strategy
        self.optimization_window = optimization_window
        self.min_samples_for_optimization = min_samples_for_optimization
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate

        # Processor arms for bandit algorithms
        self.processor_arms: dict[str, ModelArm] = {}

        # Performance tracking
        self.performance_history: deque = deque(maxlen=optimization_window)
        self.current_ensemble: list[Detector] | None = None
        self.ensemble_weights: dict[str, float] = {}

        # Optimization state
        self.total_predictions = 0
        self.last_optimization = datetime.now()
        self.optimization_interval = timedelta(minutes=5)

        # Context tracking for contextual bandits
        self.current_context: DataContext | None = None
        self.context_history: deque = deque(maxlen=100)

        # Drift processing state
        self.drift_detected = False
        self.last_drift_check = datetime.now()

        # Performance monitoring
        self.performance_buffer: list[EnsemblePerformanceMetrics] = []

        logger.info(
            f"Initialized real-time ensemble optimizer with strategy: {optimization_strategy}"
        )

    async def initialize_ensemble(self, initial_data_collection: DataCollection) -> list[Detector]:
        """Initialize the ensemble with base algorithms."""
        logger.info("Initializing base ensemble")

        # Create initial ensemble using base service
        ensemble, report = await self.base_ensemble_service.create_intelligent_ensemble(
            initial_data_collection
        )

        # Initialize processor arms
        for i, detector in enumerate(ensemble):
            arm_id = f"processor_{i}"
            algorithm_name = getattr(detector, "algorithm_name", f"algorithm_{i}")

            self.processor_arms[arm_id] = ModelArm(
                processor_id=arm_id, algorithm_name=algorithm_name
            )

        self.current_ensemble = ensemble

        # Initialize equal weights
        weight = 1.0 / len(ensemble)
        self.ensemble_weights = {f"processor_{i}": weight for i in range(len(ensemble))}

        logger.info(f"Initialized ensemble with {len(ensemble)} models")
        return ensemble

    async def predict_with_optimization(
        self, data: DataCollection, feedback: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Make predictions while optimizing ensemble in real-time.

        Args:
            data: Input data for prediction
            feedback: Optional ground truth labels for learning

        Returns:
            Tuple of (predictions, optimization_info)
        """
        start_time = time.time()

        # Update context
        await self._update_context(data)

        # Check for drift
        await self._check_drift(data)

        # Select optimal ensemble composition
        selected_processors = await self._select_optimal_processors()

        # Make predictions
        predictions = await self._make_ensemble_prediction(data, selected_processors)

        # Calculate performance measurements
        prediction_time = time.time() - start_time
        measurements = EnsemblePerformanceMetrics(
            prediction_latency=prediction_time,
            throughput=len(data.data) / prediction_time if prediction_time > 0 else 0,
        )

        # Update processor arms with feedback
        if feedback is not None:
            await self._update_processor_arms(predictions, feedback, selected_processors)

            # Calculate accuracy measurements
            measurements.accuracy = (
                np.mean(predictions == feedback) if len(feedback) > 0 else 0.0
            )

        # Store performance measurements
        self.performance_buffer.append(measurements)
        self.total_predictions += len(data.data)

        # Trigger optimization if needed
        optimization_info = await self._trigger_optimization_if_needed()

        # Add prediction info
        optimization_info.update(
            {
                "selected_processors": [arm.processor_id for arm in selected_processors],
                "prediction_latency": prediction_time,
                "total_predictions": self.total_predictions,
                "drift_detected": self.drift_detected,
                "current_context": (
                    self.current_context.dict() if self.current_context else None
                ),
            }
        )

        return predictions, optimization_info

    async def _update_context(self, data: DataCollection) -> None:
        """Update context information for contextual bandits."""
        # Calculate data characteristics
        context = DataContext(
            sample_size=len(data.data),
            feature_count=data.data.shape[1] if len(data.data.shape) > 1 else 1,
            data_complexity=self._calculate_data_complexity(data.data),
            noise_level=self._estimate_noise_level(data.data),
            anomaly_density=self._estimate_anomaly_density(data.data),
        )

        # Add drift scores if available
        if hasattr(self.drift_processing_service, "get_latest_drift_scores"):
            drift_scores = await self.drift_processing_service.get_latest_drift_scores()
            if drift_scores:
                context.feature_drift_score = drift_scores.get("feature_drift", 0.0)
                context.concept_drift_score = drift_scores.get("concept_drift", 0.0)

        # Add temporal patterns
        current_time = datetime.now()
        context.temporal_patterns = {
            "hour_of_day": current_time.hour / 24.0,
            "day_of_week": current_time.weekday() / 7.0,
            "time_since_last_prediction": (
                ((current_time - self.last_optimization).total_seconds() / 3600.0)
                if hasattr(self, "last_optimization")
                else 0.0
            ),
        }

        self.current_context = context
        self.context_history.append(context)

    def _calculate_data_complexity(self, data: np.ndarray) -> float:
        """Calculate intrinsic data complexity."""
        try:
            # Use variance-based complexity measure
            if len(data) > 1:
                feature_variances = np.var(data, axis=0)
                mean_variance = np.mean(feature_variances)
                variance_of_variances = np.var(feature_variances)

                # Complexity as coefficient of variation of variances
                if mean_variance > 0:
                    return min(1.0, variance_of_variances / mean_variance)
            return 0.5
        except Exception:
            return 0.5

    def _estimate_noise_level(self, data: np.ndarray) -> float:
        """Estimate noise level in data."""
        try:
            if len(data) > 10:
                # Use interquartile range method
                q75, q25 = np.percentile(data, [75, 25], axis=0)
                iqr = q75 - q25
                median_iqr = np.median(iqr[iqr > 0]) if np.any(iqr > 0) else 1.0

                # Noise as ratio of outliers to IQR
                outliers = np.sum(
                    np.any(np.abs(data - np.median(data, axis=0)) > 1.5 * iqr, axis=1)
                )
                noise_level = outliers / len(data)
                return min(1.0, noise_level)
            return 0.5
        except Exception:
            return 0.5

    def _estimate_anomaly_density(self, data: np.ndarray) -> float:
        """Estimate density of anomalies in data."""
        try:
            if len(data) > 5:
                # Use z-score method
                z_scores = np.abs(
                    (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
                )
                max_z_scores = np.max(z_scores, axis=1)
                anomalies = np.sum(max_z_scores > 3.0)  # 3-sigma rule
                return min(0.5, anomalies / len(data))
            return 0.1
        except Exception:
            return 0.1

    async def _check_drift(self, data: DataCollection) -> None:
        """Check for data drift and trigger reoptimization if needed."""
        current_time = datetime.now()

        # Check drift periodically
        if current_time - self.last_drift_check > timedelta(minutes=1):
            try:
                # Use drift processing service
                drift_results = await self.drift_processing_service.detect_drift(data)

                # Check if significant drift detected
                drift_threshold = 0.7
                self.drift_detected = (
                    drift_results.get("feature_drift_score", 0.0) > drift_threshold
                    or drift_results.get("concept_drift_score", 0.0) > drift_threshold
                )

                if self.drift_detected:
                    logger.warning(
                        "Significant data drift detected, triggering ensemble reoptimization"
                    )
                    await self._trigger_ensemble_reoptimization(data)

                self.last_drift_check = current_time

            except Exception as e:
                logger.error(f"Error during drift processing: {e}")

    async def _trigger_ensemble_reoptimization(self, data: DataCollection) -> None:
        """Trigger complete ensemble reoptimization due to drift."""
        logger.info("Reoptimizing ensemble due to drift processing")

        # Create new ensemble
        (
            new_ensemble,
            report,
        ) = await self.base_ensemble_service.create_intelligent_ensemble(data)

        # Update processor arms with new models
        old_arms = self.processor_arms.copy()
        self.processor_arms.clear()

        for i, detector in enumerate(new_ensemble):
            arm_id = f"processor_{i}"
            algorithm_name = getattr(detector, "algorithm_name", f"algorithm_{i}")

            # Transfer knowledge from old arms if similar algorithm exists
            old_arm = None
            for old_id, old_arm_obj in old_arms.items():
                if old_arm_obj.algorithm_name == algorithm_name:
                    old_arm = old_arm_obj
                    break

            if old_arm:
                # Transfer partial knowledge
                self.processor_arms[arm_id] = ModelArm(
                    processor_id=arm_id,
                    algorithm_name=algorithm_name,
                    total_rewards=old_arm.total_rewards * 0.5,  # Decay old rewards
                    total_pulls=max(1, old_arm.total_pulls // 2),  # Reduce pull count
                    recent_rewards=old_arm.recent_rewards[-10:],  # Keep recent rewards
                )
            else:
                # New algorithm
                self.processor_arms[arm_id] = ModelArm(
                    processor_id=arm_id, algorithm_name=algorithm_name
                )

        self.current_ensemble = new_ensemble
        self.drift_detected = False

        # Reset weights
        weight = 1.0 / len(new_ensemble)
        self.ensemble_weights = {f"processor_{i}": weight for i in range(len(new_ensemble))}

    async def _select_optimal_processors(self) -> list[ModelArm]:
        """Select optimal models based on optimization strategy."""
        if not self.processor_arms:
            return []

        if self.optimization_strategy == OptimizationStrategy.MULTI_ARMED_BANDIT:
            return await self._select_with_multi_armed_bandit()
        elif self.optimization_strategy == OptimizationStrategy.UCB:
            return await self._select_with_ucb()
        elif self.optimization_strategy == OptimizationStrategy.EPSILON_GREEDY:
            return await self._select_with_epsilon_greedy()
        elif self.optimization_strategy == OptimizationStrategy.THOMPSON_SAMPLING:
            return await self._select_with_thompson_sampling()
        elif self.optimization_strategy == OptimizationStrategy.CONTEXTUAL_BANDIT:
            return await self._select_with_contextual_bandit()
        else:
            # Default: select all models with equal weights
            return list(self.processor_arms.values())

    async def _select_with_multi_armed_bandit(self) -> list[ModelArm]:
        """Select models using multi-armed bandit algorithm."""
        total_pulls = sum(arm.total_pulls for arm in self.processor_arms.values())

        if total_pulls < self.min_samples_for_optimization:
            # Exploration phase: select all models
            return list(self.processor_arms.values())

        # Calculate UCB scores for all arms
        arm_scores = []
        for arm in self.processor_arms.values():
            ucb_score = arm.get_ucb_score(total_pulls)
            arm_scores.append((arm, ucb_score))

        # Sort by UCB score and select top models
        arm_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top 3-5 models
        max_processors = min(5, len(arm_scores))
        selected_arms = [arm for arm, _ in arm_scores[:max_processors]]

        return selected_arms

    async def _select_with_ucb(self) -> list[ModelArm]:
        """Select models using Upper Confidence Bound."""
        return await self._select_with_multi_armed_bandit()  # Same implementation

    async def _select_with_epsilon_greedy(self) -> list[ModelArm]:
        """Select models using epsilon-greedy strategy."""
        if np.random.random() < self.exploration_rate:
            # Exploration: random selection
            return list(self.processor_arms.values())
        else:
            # Exploitation: select best performing models
            sorted_arms = sorted(
                self.processor_arms.values(),
                key=lambda arm: arm.get_average_reward(),
                reverse=True,
            )
            # Select top 3 models
            return sorted_arms[:3]

    async def _select_with_thompson_sampling(self) -> list[ModelArm]:
        """Select models using Thompson sampling."""
        # Sample from beta distribution for each arm
        arm_samples = []

        for arm in self.processor_arms.values():
            if arm.total_pulls > 0:
                # Beta distribution parameters
                successes = arm.total_rewards
                failures = arm.total_pulls - successes

                # Sample from beta distribution
                sample = np.random.beta(max(1, successes + 1), max(1, failures + 1))
            else:
                # Uniform sample for unexplored arms
                sample = np.random.random()

            arm_samples.append((arm, sample))

        # Sort by sampled values and select top models
        arm_samples.sort(key=lambda x: x[1], reverse=True)

        # Select top models (adaptive number based on diversity)
        max_processors = min(4, len(arm_samples))
        selected_arms = [arm for arm, _ in arm_samples[:max_processors]]

        return selected_arms

    async def _select_with_contextual_bandit(self) -> list[ModelArm]:
        """Select models using contextual bandit with current context."""
        if not self.current_context:
            return await self._select_with_ucb()

        # Calculate context-aware scores for each arm
        context_vector = self._vectorize_context(self.current_context)
        arm_scores = []

        for arm in self.processor_arms.values():
            # Update arm's context features (simplified linear processor)
            if not arm.context_features:
                # Initialize with random weights
                arm.context_features = {
                    f"feature_{i}": np.random.normal(0, 0.1)
                    for i in range(len(context_vector))
                }

            # Calculate context score
            context_score = sum(
                arm.context_features.get(f"feature_{i}", 0.0) * context_vector[i]
                for i in range(len(context_vector))
            )

            # Combine with historical performance
            historical_score = arm.get_average_reward()

            # Weighted combination
            combined_score = 0.7 * historical_score + 0.3 * context_score

            arm_scores.append((arm, combined_score))

        # Sort and select top models
        arm_scores.sort(key=lambda x: x[1], reverse=True)

        max_processors = min(4, len(arm_scores))
        selected_arms = [arm for arm, _ in arm_scores[:max_processors]]

        return selected_arms

    def _vectorize_context(self, context: DataContext) -> np.ndarray:
        """Convert context to feature vector."""
        features = [
            context.feature_drift_score,
            context.concept_drift_score,
            context.data_complexity,
            context.noise_level,
            np.log(context.sample_size + 1) / 10.0,  # Log-scaled sample size
            np.log(context.feature_count + 1) / 5.0,  # Log-scaled feature count
            context.anomaly_density,
        ]

        # Add temporal features
        features.extend(context.temporal_patterns.values())

        return np.array(features)

    async def _make_ensemble_prediction(
        self, data: DataCollection, selected_processors: list[ModelArm]
    ) -> np.ndarray:
        """Make ensemble prediction using selected models."""
        if not selected_processors or not self.current_ensemble:
            return np.array([])

        # Get predictions from selected models
        predictions = []
        processor_weights = []

        for arm in selected_processors:
            processor_index = int(arm.processor_id.split("_")[1])
            if processor_index < len(self.current_ensemble):
                detector = self.current_ensemble[processor_index]

                try:
                    pred = detector.predict(data)
                    predictions.append(pred)

                    # Use adaptive weight based on recent performance
                    weight = self.ensemble_weights.get(arm.processor_id, 1.0)
                    performance_weight = (
                        arm.get_average_reward() if arm.total_pulls > 0 else 0.5
                    )
                    adaptive_weight = 0.7 * weight + 0.3 * performance_weight

                    processor_weights.append(adaptive_weight)

                except Exception as e:
                    logger.error(
                        f"Error making prediction with processor {arm.processor_id}: {e}"
                    )
                    continue

        if not predictions:
            return np.array([])

        # Weighted ensemble prediction
        predictions = np.array(predictions)
        weights = np.array(processor_weights)
        weights = weights / np.sum(weights)  # Normalize weights

        # Weighted average
        ensemble_prediction = np.average(predictions, axis=0, weights=weights)

        return ensemble_prediction

    async def _update_processor_arms(
        self,
        predictions: np.ndarray,
        feedback: np.ndarray,
        selected_processors: list[ModelArm],
    ) -> None:
        """Update processor arms with feedback."""
        if len(predictions) == 0 or len(feedback) == 0:
            return

        # Calculate reward for ensemble
        ensemble_reward = np.mean(predictions == feedback) if len(feedback) > 0 else 0.0

        # Update each selected processor arm
        for arm in selected_processors:
            processor_index = int(arm.processor_id.split("_")[1])
            if processor_index < len(self.current_ensemble):
                try:
                    # Get individual processor prediction
                    detector = self.current_ensemble[processor_index]
                    individual_pred = detector.predict(
                        DataCollection(name="feedback", data=np.array([feedback]), features=[])
                    )

                    # Calculate individual reward
                    individual_reward = (
                        np.mean(individual_pred == feedback)
                        if len(feedback) > 0
                        else 0.0
                    )

                    # Update arm statistics
                    arm.total_rewards += individual_reward
                    arm.total_pulls += 1
                    arm.recent_rewards.append(individual_reward)
                    arm.last_selected = datetime.now()

                    # Keep only recent rewards
                    if len(arm.recent_rewards) > 50:
                        arm.recent_rewards = arm.recent_rewards[-50:]

                    # Update context features for contextual bandits
                    if (
                        self.optimization_strategy
                        == OptimizationStrategy.CONTEXTUAL_BANDIT
                    ):
                        await self._update_contextual_features(arm, individual_reward)

                except Exception as e:
                    logger.error(f"Error updating arm {arm.processor_id}: {e}")

    async def _update_contextual_features(self, arm: ModelArm, reward: float) -> None:
        """Update contextual features for an arm using gradient descent."""
        if not self.current_context or not arm.context_features:
            return

        try:
            context_vector = self._vectorize_context(self.current_context)

            # Predicted reward using current features
            predicted_reward = sum(
                arm.context_features.get(f"feature_{i}", 0.0) * context_vector[i]
                for i in range(len(context_vector))
            )

            # Gradient descent update
            error = reward - predicted_reward

            for i, feature_value in enumerate(context_vector):
                feature_key = f"feature_{i}"
                if feature_key in arm.context_features:
                    # Update feature weight
                    arm.context_features[feature_key] += (
                        self.learning_rate * error * feature_value
                    )

                    # Clip weights to reasonable range
                    arm.context_features[feature_key] = np.clip(
                        arm.context_features[feature_key], -2.0, 2.0
                    )

        except Exception as e:
            logger.error(f"Error updating contextual features for {arm.processor_id}: {e}")

    async def _trigger_optimization_if_needed(self) -> dict[str, Any]:
        """Trigger optimization if conditions are met."""
        current_time = datetime.now()
        optimization_info = {
            "optimization_triggered": False,
            "optimization_reason": None,
            "processor_arm_stats": self._get_processor_arm_statistics(),
        }

        # Check if optimization should be triggered
        should_optimize = (
            current_time - self.last_optimization > self.optimization_interval
            or self.total_predictions % self.optimization_window == 0
            or self.drift_detected
        )

        if (
            should_optimize
            and self.total_predictions >= self.min_samples_for_optimization
        ):
            optimization_info["optimization_triggered"] = True

            if self.drift_detected:
                optimization_info["optimization_reason"] = "drift_detected"
            elif current_time - self.last_optimization > self.optimization_interval:
                optimization_info["optimization_reason"] = "time_interval"
            else:
                optimization_info["optimization_reason"] = "prediction_count"

            # Perform optimization
            await self._optimize_ensemble_weights()
            self.last_optimization = current_time

        return optimization_info

    async def _optimize_ensemble_weights(self) -> None:
        """Optimize ensemble weights based on recent performance."""
        if not self.processor_arms:
            return

        logger.info("Optimizing ensemble weights")

        # Calculate new weights based on recent performance
        total_performance = 0.0
        arm_performances = {}

        for arm_id, arm in self.processor_arms.items():
            if arm.total_pulls > 0:
                # Use recent performance with decay
                recent_performance = (
                    np.mean(arm.recent_rewards[-20:]) if arm.recent_rewards else 0.5
                )
                overall_performance = arm.get_average_reward()

                # Weighted combination favoring recent performance
                combined_performance = (
                    0.7 * recent_performance + 0.3 * overall_performance
                )

                arm_performances[arm_id] = combined_performance
                total_performance += combined_performance
            else:
                arm_performances[arm_id] = (
                    0.1  # Small default weight for unexplored arms
                )
                total_performance += 0.1

        # Normalize weights
        if total_performance > 0:
            for arm_id in arm_performances:
                self.ensemble_weights[arm_id] = (
                    arm_performances[arm_id] / total_performance
                )

        # Apply weight smoothing to prevent extreme weights
        self._smooth_ensemble_weights()

        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")

    def _smooth_ensemble_weights(self) -> None:
        """Apply smoothing to ensemble weights to prevent extreme values."""
        min_weight = 0.05  # Minimum weight for any processor

        # Ensure minimum weight
        for arm_id in self.ensemble_weights:
            if self.ensemble_weights[arm_id] < min_weight:
                self.ensemble_weights[arm_id] = min_weight

        # Renormalize
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            for arm_id in self.ensemble_weights:
                self.ensemble_weights[arm_id] /= total_weight

    def _get_model_arm_statistics(self) -> dict[str, Any]:
        """Get statistics about processor arms."""
        stats = {
            "total_arms": len(self.processor_arms),
            "arms_with_experience": sum(
                1 for arm in self.processor_arms.values() if arm.total_pulls > 0
            ),
            "average_pulls": (
                np.mean([arm.total_pulls for arm in self.processor_arms.values()])
                if self.processor_arms
                else 0
            ),
            "average_reward": (
                np.mean([arm.get_average_reward() for arm in self.processor_arms.values()])
                if self.processor_arms
                else 0
            ),
            "best_performing_arm": None,
            "worst_performing_arm": None,
            "arm_details": {},
        }

        if self.processor_arms:
            # Find best and worst performing arms
            best_arm = max(
                self.processor_arms.values(), key=lambda arm: arm.get_average_reward()
            )
            worst_arm = min(
                self.processor_arms.values(), key=lambda arm: arm.get_average_reward()
            )

            stats["best_performing_arm"] = {
                "processor_id": best_arm.processor_id,
                "algorithm": best_arm.algorithm_name,
                "average_reward": best_arm.get_average_reward(),
                "total_pulls": best_arm.total_pulls,
            }

            stats["worst_performing_arm"] = {
                "processor_id": worst_arm.processor_id,
                "algorithm": worst_arm.algorithm_name,
                "average_reward": worst_arm.get_average_reward(),
                "total_pulls": worst_arm.total_pulls,
            }

            # Detailed statistics for each arm
            for arm in self.processor_arms.values():
                stats["arm_details"][arm.processor_id] = {
                    "algorithm": arm.algorithm_name,
                    "average_reward": arm.get_average_reward(),
                    "total_pulls": arm.total_pulls,
                    "recent_performance": (
                        np.mean(arm.recent_rewards[-10:]) if arm.recent_rewards else 0.0
                    ),
                    "last_selected": (
                        arm.last_selected.isoformat() if arm.last_selected else None
                    ),
                    "current_weight": self.ensemble_weights.get(arm.processor_id, 0.0),
                }

        return stats

    async def get_optimization_report(self) -> dict[str, Any]:
        """Get comprehensive optimization report."""
        current_time = datetime.now()

        report = {
            "optimizer_status": {
                "strategy": self.optimization_strategy.value,
                "total_predictions": self.total_predictions,
                "last_optimization": self.last_optimization.isoformat(),
                "drift_detected": self.drift_detected,
                "optimization_window": self.optimization_window,
                "exploration_rate": self.exploration_rate,
            },
            "ensemble_composition": {
                "total_processors": (
                    len(self.current_ensemble) if self.current_ensemble else 0
                ),
                "active_processors": len(
                    [arm for arm in self.processor_arms.values() if arm.total_pulls > 0]
                ),
                "current_weights": self.ensemble_weights.copy(),
            },
            "performance_measurements": {
                "recent_performance": self._calculate_recent_performance(),
                "processor_arm_statistics": self._get_processor_arm_statistics(),
                "optimization_effectiveness": self._calculate_optimization_effectiveness(),
            },
            "context_analysis": {
                "current_context": (
                    self.current_context.dict() if self.current_context else None
                ),
                "context_trends": self._analyze_context_trends(),
            },
            "recommendations": self._generate_optimization_recommendations(),
        }

        return report

    def _calculate_recent_performance(self) -> dict[str, float]:
        """Calculate recent performance measurements."""
        if not self.performance_buffer:
            return {"no_data": True}

        recent_measurements = self.performance_buffer[-20:]  # Last 20 predictions

        return {
            "average_accuracy": np.mean(
                [m.accuracy for m in recent_measurements if m.accuracy > 0]
            ),
            "average_latency": np.mean([m.prediction_latency for m in recent_measurements]),
            "average_throughput": np.mean(
                [m.throughput for m in recent_measurements if m.throughput > 0]
            ),
            "performance_trend": self._calculate_performance_trend(recent_measurements),
        }

    def _calculate_performance_trend(
        self, measurements: list[EnsemblePerformanceMetrics]
    ) -> str:
        """Calculate performance trend."""
        if len(measurements) < 5:
            return "insufficient_data"

        accuracies = [m.accuracy for m in measurements if m.accuracy > 0]
        if len(accuracies) < 3:
            return "insufficient_data"

        # Simple trend analysis
        recent_half = accuracies[len(accuracies) // 2 :]
        early_half = accuracies[: len(accuracies) // 2]

        recent_avg = np.mean(recent_half)
        early_avg = np.mean(early_half)

        improvement = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0

        if improvement > 0.05:
            return "improving"
        elif improvement < -0.05:
            return "declining"
        else:
            return "stable"

    def _calculate_optimization_effectiveness(self) -> dict[str, float]:
        """Calculate how effective the optimization has been."""
        if not self.processor_arms:
            return {"no_data": True}

        # Calculate variance in arm performance
        arm_rewards = [
            arm.get_average_reward()
            for arm in self.processor_arms.values()
            if arm.total_pulls > 0
        ]

        if len(arm_rewards) < 2:
            return {"insufficient_data": True}

        reward_variance = np.var(arm_rewards)
        reward_range = max(arm_rewards) - min(arm_rewards)
        exploration_ratio = len(
            [arm for arm in self.processor_arms.values() if arm.total_pulls > 0]
        ) / len(self.processor_arms)

        return {
            "reward_variance": reward_variance,
            "reward_range": reward_range,
            "exploration_ratio": exploration_ratio,
            "optimization_effectiveness": min(
                1.0, reward_range + exploration_ratio * 0.5
            ),
        }

    def _analyze_context_trends(self) -> dict[str, Any]:
        """Analyze trends in context over time."""
        if len(self.context_history) < 5:
            return {"insufficient_data": True}

        recent_contexts = list(self.context_history)[-10:]

        trends = {}
        for field in [
            "feature_drift_score",
            "concept_drift_score",
            "data_complexity",
            "noise_level",
            "anomaly_density",
        ]:
            values = [getattr(ctx, field) for ctx in recent_contexts]
            if len(values) > 1:
                # Simple trend calculation
                trend = np.polyfit(range(len(values)), values, 1)[0]
                trends[field] = {
                    "trend": (
                        "increasing"
                        if trend > 0.01
                        else "decreasing"
                        if trend < -0.01
                        else "stable"
                    ),
                    "slope": trend,
                    "current_value": values[-1],
                    "average_value": np.mean(values),
                }

        return trends

    def _generate_optimization_recommendations(self) -> list[str]:
        """Generate recommendations for optimization improvement."""
        recommendations = []

        # Check exploration vs exploitation balance
        if self.processor_arms:
            unexplored_arms = [
                arm for arm in self.processor_arms.values() if arm.total_pulls == 0
            ]
            if len(unexplored_arms) > len(self.processor_arms) * 0.3:
                recommendations.append(
                    "Consider increasing exploration rate to better evaluate all models"
                )

        # Check if drift is frequent
        if self.drift_detected:
            recommendations.append(
                "Frequent drift detected - consider more adaptive algorithms or shorter retraining cycles"
            )

        # Check performance variance
        effectiveness = self._calculate_optimization_effectiveness()
        if (
            not effectiveness.get("no_data")
            and effectiveness.get("reward_range", 0) < 0.1
        ):
            recommendations.append(
                "Low performance variance among models - consider adding more diverse algorithms"
            )

        # Check recent performance trend
        recent_perf = self._calculate_recent_performance()
        if (
            not recent_perf.get("no_data")
            and recent_perf.get("performance_trend") == "declining"
        ):
            recommendations.append(
                "Performance declining - consider retraining models or adjusting hyperparameters"
            )

        if not recommendations:
            recommendations.append(
                "Optimization appears to be working well - continue monitoring"
            )

        return recommendations
