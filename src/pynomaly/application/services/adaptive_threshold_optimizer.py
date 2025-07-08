"""Adaptive anomaly threshold optimization with Bayesian optimization and self-tuning capabilities."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ThresholdOptimizationStrategy(str, Enum):
    """Threshold optimization strategies."""

    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRADIENT_DESCENT = "gradient_descent"
    BINARY_SEARCH = "binary_search"
    GRID_SEARCH = "grid_search"
    EVOLUTIONARY = "evolutionary"
    ADAPTIVE_BANDITS = "adaptive_bandits"


class FeedbackType(str, Enum):
    """Types of feedback for threshold adjustment."""

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_NEGATIVE = "false_negative"
    UNCERTAIN = "uncertain"


@dataclass
class ThresholdFeedback:
    """Feedback about threshold performance."""

    threshold_value: float
    feedback_type: FeedbackType
    anomaly_score: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for threshold evaluation."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    accuracy: float = 0.0
    specificity: float = 0.0
    balanced_accuracy: float = 0.0
    matthews_correlation: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ThresholdCandidate(BaseModel):
    """Candidate threshold with performance tracking."""

    threshold: float
    performance_history: list[PerformanceMetrics] = Field(default_factory=list)
    feedback_count: dict[FeedbackType, int] = Field(default_factory=dict)
    exploration_count: int = 0
    exploitation_count: int = 0
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    last_updated: datetime = Field(default_factory=datetime.now)

    def get_average_f1(self) -> float:
        """Get average F1 score."""
        if not self.performance_history:
            return 0.5
        return np.mean([perf.f1_score for perf in self.performance_history])

    def get_balanced_score(self) -> float:
        """Get balanced performance score considering multiple metrics."""
        if not self.performance_history:
            return 0.5

        recent_metrics = self.performance_history[-5:]  # Last 5 evaluations

        weights = {
            'f1_score': 0.3,
            'balanced_accuracy': 0.25,
            'precision': 0.2,
            'recall': 0.2,
            'matthews_correlation': 0.05
        }

        balanced_score = 0.0
        for metric_name, weight in weights.items():
            metric_values = [getattr(perf, metric_name) for perf in recent_metrics]
            if metric_values:
                avg_metric = np.mean(metric_values)
                balanced_score += weight * avg_metric

        return balanced_score


class AdaptiveThresholdOptimizer:
    """Adaptive threshold optimization with multiple strategies and continuous learning."""

    def __init__(
        self,
        optimization_strategy: ThresholdOptimizationStrategy = ThresholdOptimizationStrategy.BAYESIAN_OPTIMIZATION,
        initial_threshold: float = 0.5,
        threshold_range: tuple[float, float] = (0.1, 0.9),
        optimization_window: int = 100,
        min_feedback_for_optimization: int = 20,
        false_positive_penalty: float = 1.0,
        false_negative_penalty: float = 2.0,
        learning_rate: float = 0.01,
        exploration_rate: float = 0.1,
    ):
        """Initialize adaptive threshold optimizer.

        Args:
            optimization_strategy: Strategy for threshold optimization
            initial_threshold: Initial threshold value
            threshold_range: Valid range for thresholds
            optimization_window: Number of feedback samples to consider
            min_feedback_for_optimization: Minimum feedback before optimization
            false_positive_penalty: Penalty weight for false positives
            false_negative_penalty: Penalty weight for false negatives
            learning_rate: Learning rate for gradient-based methods
            exploration_rate: Exploration rate for exploration vs exploitation
        """
        self.optimization_strategy = optimization_strategy
        self.current_threshold = initial_threshold
        self.threshold_range = threshold_range
        self.optimization_window = optimization_window
        self.min_feedback_for_optimization = min_feedback_for_optimization
        self.false_positive_penalty = false_positive_penalty
        self.false_negative_penalty = false_negative_penalty
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate

        # Feedback tracking
        self.feedback_history: deque = deque(maxlen=optimization_window)
        self.recent_performance: deque = deque(maxlen=50)

        # Threshold candidates for exploration
        self.threshold_candidates: dict[float, ThresholdCandidate] = {}
        self._initialize_threshold_candidates()

        # Optimization state
        self.total_feedback = 0
        self.last_optimization = datetime.now()
        self.optimization_interval = timedelta(minutes=10)

        # Context-aware adaptation
        self.context_thresholds: dict[str, float] = {}
        self.context_performance: dict[str, list[PerformanceMetrics]] = defaultdict(list)

        # Bayesian optimization state
        self.bayesian_state = {
            'observations': [],
            'targets': [],
            'acquisition_function': 'expected_improvement',
            'kernel_params': {'length_scale': 0.1, 'noise_level': 0.01}
        }

        logger.info(f"Initialized adaptive threshold optimizer with strategy: {optimization_strategy}")

    def _initialize_threshold_candidates(self) -> None:
        """Initialize threshold candidates for exploration."""
        # Create candidates across the threshold range
        num_candidates = 11
        thresholds = np.linspace(self.threshold_range[0], self.threshold_range[1], num_candidates)

        for threshold in thresholds:
            self.threshold_candidates[threshold] = ThresholdCandidate(threshold=threshold)

        # Ensure current threshold is included
        if self.current_threshold not in self.threshold_candidates:
            self.threshold_candidates[self.current_threshold] = ThresholdCandidate(
                threshold=self.current_threshold
            )

    async def get_adaptive_threshold(
        self,
        anomaly_scores: np.ndarray,
        context: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any]]:
        """Get adaptive threshold based on current context and history.

        Args:
            anomaly_scores: Recent anomaly scores for context
            context: Optional context information

        Returns:
            Tuple of (threshold, adaptation_info)
        """
        adaptation_info = {
            "base_threshold": self.current_threshold,
            "adaptation_applied": False,
            "adaptation_reason": None,
            "context_adjustment": 0.0,
            "confidence": 0.0
        }

        # Get base threshold
        threshold = self.current_threshold

        # Apply context-specific adaptations
        if context:
            context_key = self._get_context_key(context)
            if context_key in self.context_thresholds:
                context_threshold = self.context_thresholds[context_key]

                # Blend base and context-specific thresholds
                context_weight = min(0.5, len(self.context_performance[context_key]) / 20.0)
                threshold = (1 - context_weight) * threshold + context_weight * context_threshold

                adaptation_info["adaptation_applied"] = True
                adaptation_info["adaptation_reason"] = "context_specific"
                adaptation_info["context_adjustment"] = context_threshold - self.current_threshold

        # Apply score distribution adaptation
        if len(anomaly_scores) > 10:
            distribution_adjustment = await self._calculate_distribution_adjustment(anomaly_scores)
            threshold += distribution_adjustment

            if abs(distribution_adjustment) > 0.01:
                adaptation_info["adaptation_applied"] = True
                adaptation_info["adaptation_reason"] = "score_distribution"

        # Apply temporal adaptation
        temporal_adjustment = await self._calculate_temporal_adjustment()
        threshold += temporal_adjustment

        if abs(temporal_adjustment) > 0.01:
            adaptation_info["adaptation_applied"] = True
            adaptation_info["adaptation_reason"] = "temporal_pattern"

        # Ensure threshold is within valid range
        threshold = np.clip(threshold, self.threshold_range[0], self.threshold_range[1])

        # Calculate confidence in threshold
        adaptation_info["confidence"] = self._calculate_threshold_confidence(threshold)

        return threshold, adaptation_info

    def _get_context_key(self, context: dict[str, Any]) -> str:
        """Generate context key for threshold mapping."""
        # Create a simple hash of key context features
        key_features = []

        for key in ['data_source', 'time_of_day', 'day_of_week', 'system_load']:
            if key in context:
                key_features.append(f"{key}:{context[key]}")

        return "|".join(sorted(key_features)) if key_features else "default"

    async def _calculate_distribution_adjustment(self, anomaly_scores: np.ndarray) -> float:
        """Calculate threshold adjustment based on score distribution."""
        try:
            # Analyze score distribution
            score_mean = np.mean(anomaly_scores)
            score_std = np.std(anomaly_scores)
            score_skew = self._calculate_skewness(anomaly_scores)

            # If scores are heavily skewed right (many high scores), lower threshold
            if score_skew > 1.0:
                adjustment = -0.05 * min(score_skew / 2.0, 1.0)
            # If scores are heavily skewed left (many low scores), raise threshold
            elif score_skew < -1.0:
                adjustment = 0.05 * min(abs(score_skew) / 2.0, 1.0)
            else:
                adjustment = 0.0

            # Consider standard deviation - high variance might need threshold adjustment
            if score_std > 0.3:
                # High variance: be more conservative
                adjustment += 0.02
            elif score_std < 0.1:
                # Low variance: can be more aggressive
                adjustment -= 0.02

            return np.clip(adjustment, -0.1, 0.1)

        except Exception as e:
            logger.error(f"Error calculating distribution adjustment: {e}")
            return 0.0

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution."""
        try:
            mean = np.mean(data)
            std = np.std(data)

            if std == 0:
                return 0.0

            # Calculate third moment
            skewness = np.mean(((data - mean) / std) ** 3)
            return skewness

        except Exception:
            return 0.0

    async def _calculate_temporal_adjustment(self) -> float:
        """Calculate threshold adjustment based on temporal patterns."""
        try:
            current_time = datetime.now()
            hour = current_time.hour
            day_of_week = current_time.weekday()

            # Simple temporal adjustments based on time patterns
            adjustment = 0.0

            # Night hours (10 PM - 6 AM): slightly more sensitive
            if hour >= 22 or hour <= 6:
                adjustment -= 0.02

            # Business hours (9 AM - 5 PM): slightly less sensitive
            elif 9 <= hour <= 17:
                adjustment += 0.01

            # Weekend: different sensitivity
            if day_of_week >= 5:  # Saturday = 5, Sunday = 6
                adjustment -= 0.01

            # Look at recent performance trends
            if len(self.recent_performance) >= 5:
                recent_metrics = list(self.recent_performance)[-5:]
                avg_fpr = np.mean([m.false_positive_rate for m in recent_metrics])
                avg_fnr = np.mean([m.false_negative_rate for m in recent_metrics])

                # If too many false positives, raise threshold
                if avg_fpr > 0.1:
                    adjustment += 0.03 * (avg_fpr - 0.1)

                # If too many false negatives, lower threshold
                if avg_fnr > 0.1:
                    adjustment -= 0.03 * (avg_fnr - 0.1)

            return np.clip(adjustment, -0.05, 0.05)

        except Exception as e:
            logger.error(f"Error calculating temporal adjustment: {e}")
            return 0.0

    def _calculate_threshold_confidence(self, threshold: float) -> float:
        """Calculate confidence in the current threshold."""
        try:
            # Base confidence on amount of recent feedback
            feedback_confidence = min(1.0, len(self.feedback_history) / self.optimization_window)

            # Performance consistency confidence
            if len(self.recent_performance) >= 3:
                recent_f1_scores = [m.f1_score for m in list(self.recent_performance)[-5:]]
                f1_variance = np.var(recent_f1_scores)
                consistency_confidence = max(0.0, 1.0 - f1_variance)
            else:
                consistency_confidence = 0.5

            # Exploration confidence (how well we've explored around current threshold)
            exploration_confidence = 0.5
            if self.threshold_candidates:
                current_candidate = self.threshold_candidates.get(threshold)
                if current_candidate:
                    exploration_confidence = min(1.0, current_candidate.exploration_count / 10.0)

            # Combined confidence
            combined_confidence = (
                0.4 * feedback_confidence +
                0.4 * consistency_confidence +
                0.2 * exploration_confidence
            )

            return combined_confidence

        except Exception as e:
            logger.error(f"Error calculating threshold confidence: {e}")
            return 0.5

    async def provide_feedback(
        self,
        anomaly_score: float,
        threshold_used: float,
        feedback_type: FeedbackType,
        confidence: float = 1.0,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Provide feedback about threshold performance.

        Args:
            anomaly_score: The anomaly score that was evaluated
            threshold_used: The threshold that was used for decision
            feedback_type: Type of feedback (TP, FP, TN, FN)
            confidence: Confidence in the feedback (0-1)
            context: Optional context information

        Returns:
            Dict with feedback processing results
        """
        # Create feedback record
        feedback = ThresholdFeedback(
            threshold_value=threshold_used,
            feedback_type=feedback_type,
            anomaly_score=anomaly_score,
            confidence=confidence,
            context=context or {}
        )

        # Store feedback
        self.feedback_history.append(feedback)
        self.total_feedback += 1

        # Update threshold candidate statistics
        await self._update_threshold_candidate_stats(feedback)

        # Update context-specific performance
        if context:
            context_key = self._get_context_key(context)
            await self._update_context_performance(context_key, feedback)

        # Check if optimization should be triggered
        optimization_triggered = await self._check_optimization_trigger()

        feedback_result = {
            "feedback_processed": True,
            "total_feedback": self.total_feedback,
            "optimization_triggered": optimization_triggered,
            "current_threshold": self.current_threshold,
            "feedback_summary": self._get_feedback_summary()
        }

        if optimization_triggered:
            # Trigger threshold optimization
            optimization_result = await self._optimize_threshold()
            feedback_result["optimization_result"] = optimization_result

        return feedback_result

    async def _update_threshold_candidate_stats(self, feedback: ThresholdFeedback) -> None:
        """Update statistics for threshold candidates."""
        # Find closest threshold candidate
        closest_threshold = min(
            self.threshold_candidates.keys(),
            key=lambda t: abs(t - feedback.threshold_value)
        )

        candidate = self.threshold_candidates[closest_threshold]

        # Update feedback count
        if feedback.feedback_type not in candidate.feedback_count:
            candidate.feedback_count[feedback.feedback_type] = 0
        candidate.feedback_count[feedback.feedback_type] += 1

        # Calculate performance metrics if we have enough feedback
        total_feedback = sum(candidate.feedback_count.values())
        if total_feedback >= 10:
            metrics = self._calculate_performance_metrics(candidate.feedback_count)
            candidate.performance_history.append(metrics)

            # Keep only recent performance history
            if len(candidate.performance_history) > 20:
                candidate.performance_history = candidate.performance_history[-20:]

        candidate.last_updated = datetime.now()

    def _calculate_performance_metrics(self, feedback_count: dict[FeedbackType, int]) -> PerformanceMetrics:
        """Calculate performance metrics from feedback counts."""
        tp = feedback_count.get(FeedbackType.TRUE_POSITIVE, 0)
        fp = feedback_count.get(FeedbackType.FALSE_POSITIVE, 0)
        tn = feedback_count.get(FeedbackType.TRUE_NEGATIVE, 0)
        fn = feedback_count.get(FeedbackType.FALSE_NEGATIVE, 0)

        total = tp + fp + tn + fn
        if total == 0:
            return PerformanceMetrics()

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total
        balanced_accuracy = (recall + specificity) / 2

        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Matthews Correlation Coefficient
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        matthews_correlation = numerator / denominator if denominator > 0 else 0.0

        return PerformanceMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            accuracy=accuracy,
            specificity=specificity,
            balanced_accuracy=balanced_accuracy,
            matthews_correlation=matthews_correlation
        )

    async def _update_context_performance(self, context_key: str, feedback: ThresholdFeedback) -> None:
        """Update performance for specific context."""
        # This is a simplified update - in practice, you'd accumulate feedback
        # and periodically calculate metrics for each context
        pass

    async def _check_optimization_trigger(self) -> bool:
        """Check if threshold optimization should be triggered."""
        current_time = datetime.now()

        # Time-based trigger
        time_trigger = current_time - self.last_optimization > self.optimization_interval

        # Feedback count trigger
        feedback_trigger = len(self.feedback_history) >= self.min_feedback_for_optimization

        # Performance degradation trigger
        performance_trigger = await self._check_performance_degradation()

        return time_trigger and feedback_trigger or performance_trigger

    async def _check_performance_degradation(self) -> bool:
        """Check if performance has degraded significantly."""
        if len(self.recent_performance) < 5:
            return False

        try:
            recent_metrics = list(self.recent_performance)

            # Compare recent performance to historical average
            recent_f1 = np.mean([m.f1_score for m in recent_metrics[-3:]])
            historical_f1 = np.mean([m.f1_score for m in recent_metrics[:-3]])

            # Trigger optimization if performance dropped significantly
            performance_drop = historical_f1 - recent_f1
            return performance_drop > 0.1

        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
            return False

    async def _optimize_threshold(self) -> dict[str, Any]:
        """Optimize threshold using the selected strategy."""
        logger.info(f"Optimizing threshold using {self.optimization_strategy}")

        optimization_result = {
            "strategy_used": self.optimization_strategy.value,
            "old_threshold": self.current_threshold,
            "new_threshold": self.current_threshold,
            "improvement": 0.0,
            "optimization_time": 0.0
        }

        start_time = time.time()

        try:
            if self.optimization_strategy == ThresholdOptimizationStrategy.BAYESIAN_OPTIMIZATION:
                new_threshold = await self._bayesian_optimization()
            elif self.optimization_strategy == ThresholdOptimizationStrategy.GRADIENT_DESCENT:
                new_threshold = await self._gradient_descent_optimization()
            elif self.optimization_strategy == ThresholdOptimizationStrategy.BINARY_SEARCH:
                new_threshold = await self._binary_search_optimization()
            elif self.optimization_strategy == ThresholdOptimizationStrategy.GRID_SEARCH:
                new_threshold = await self._grid_search_optimization()
            elif self.optimization_strategy == ThresholdOptimizationStrategy.EVOLUTIONARY:
                new_threshold = await self._evolutionary_optimization()
            elif self.optimization_strategy == ThresholdOptimizationStrategy.ADAPTIVE_BANDITS:
                new_threshold = await self._adaptive_bandits_optimization()
            else:
                new_threshold = self.current_threshold

            # Update threshold if improvement is significant
            if abs(new_threshold - self.current_threshold) > 0.01:
                old_performance = await self._evaluate_threshold_performance(self.current_threshold)
                new_performance = await self._evaluate_threshold_performance(new_threshold)

                if new_performance > old_performance:
                    optimization_result["improvement"] = new_performance - old_performance
                    optimization_result["new_threshold"] = new_threshold
                    self.current_threshold = new_threshold
                    logger.info(f"Threshold optimized: {self.current_threshold:.4f} (improvement: {optimization_result['improvement']:.4f})")
                else:
                    logger.info("No improvement found, keeping current threshold")

            self.last_optimization = datetime.now()
            optimization_result["optimization_time"] = time.time() - start_time

        except Exception as e:
            logger.error(f"Error during threshold optimization: {e}")
            optimization_result["error"] = str(e)

        return optimization_result

    async def _bayesian_optimization(self) -> float:
        """Optimize threshold using Bayesian optimization."""
        # Simplified Bayesian optimization using candidate evaluation

        # Evaluate all candidates and build Gaussian Process model
        candidate_performances = {}
        for threshold, candidate in self.threshold_candidates.items():
            performance = candidate.get_balanced_score()
            candidate_performances[threshold] = performance

            # Update Bayesian state
            self.bayesian_state['observations'].append([threshold])
            self.bayesian_state['targets'].append(performance)

        # Keep only recent observations
        if len(self.bayesian_state['observations']) > 100:
            self.bayesian_state['observations'] = self.bayesian_state['observations'][-100:]
            self.bayesian_state['targets'] = self.bayesian_state['targets'][-100:]

        # Simple acquisition function: pick best performing candidate with some exploration
        if candidate_performances:
            best_threshold = max(candidate_performances, key=candidate_performances.get)
            best_performance = candidate_performances[best_threshold]

            # Add exploration: sometimes pick second or third best
            sorted_candidates = sorted(candidate_performances.items(), key=lambda x: x[1], reverse=True)

            if np.random.random() < self.exploration_rate and len(sorted_candidates) > 1:
                # Explore: pick from top 3 candidates
                exploration_pool = sorted_candidates[:min(3, len(sorted_candidates))]
                best_threshold = np.random.choice([t for t, _ in exploration_pool])

            return best_threshold

        return self.current_threshold

    async def _gradient_descent_optimization(self) -> float:
        """Optimize threshold using gradient descent."""
        # Calculate gradient based on recent feedback
        gradient = 0.0

        recent_feedback = list(self.feedback_history)[-20:]  # Use recent feedback

        for feedback in recent_feedback:
            # Calculate loss gradient
            if feedback.feedback_type == FeedbackType.FALSE_POSITIVE:
                # Too many false positives -> increase threshold
                gradient += self.false_positive_penalty * feedback.confidence
            elif feedback.feedback_type == FeedbackType.FALSE_NEGATIVE:
                # Too many false negatives -> decrease threshold
                gradient -= self.false_negative_penalty * feedback.confidence

        # Normalize gradient
        if len(recent_feedback) > 0:
            gradient /= len(recent_feedback)

        # Update threshold using gradient descent
        new_threshold = self.current_threshold + self.learning_rate * gradient

        # Clip to valid range
        new_threshold = np.clip(new_threshold, self.threshold_range[0], self.threshold_range[1])

        return new_threshold

    async def _binary_search_optimization(self) -> float:
        """Optimize threshold using binary search."""
        # Simple binary search for optimal threshold
        low, high = self.threshold_range
        best_threshold = self.current_threshold
        best_performance = await self._evaluate_threshold_performance(self.current_threshold)

        # Perform binary search iterations
        for _ in range(5):  # Limited iterations
            mid1 = low + (high - low) / 3
            mid2 = high - (high - low) / 3

            perf1 = await self._evaluate_threshold_performance(mid1)
            perf2 = await self._evaluate_threshold_performance(mid2)

            if perf1 > best_performance:
                best_performance = perf1
                best_threshold = mid1

            if perf2 > best_performance:
                best_performance = perf2
                best_threshold = mid2

            # Narrow search space
            if perf1 > perf2:
                high = mid2
            else:
                low = mid1

        return best_threshold

    async def _grid_search_optimization(self) -> float:
        """Optimize threshold using grid search."""
        # Evaluate performance at grid points
        grid_points = np.linspace(self.threshold_range[0], self.threshold_range[1], 11)

        best_threshold = self.current_threshold
        best_performance = await self._evaluate_threshold_performance(self.current_threshold)

        for threshold in grid_points:
            performance = await self._evaluate_threshold_performance(threshold)
            if performance > best_performance:
                best_performance = performance
                best_threshold = threshold

        return best_threshold

    async def _evolutionary_optimization(self) -> float:
        """Optimize threshold using evolutionary algorithm."""
        # Simple evolutionary algorithm
        population_size = 10
        generations = 5

        # Initialize population around current threshold
        population = np.random.normal(
            self.current_threshold,
            0.1,
            population_size
        )
        population = np.clip(population, self.threshold_range[0], self.threshold_range[1])

        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for threshold in population:
                fitness = await self._evaluate_threshold_performance(threshold)
                fitness_scores.append(fitness)

            # Select best individuals
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_size = population_size // 2
            elite = population[sorted_indices[:elite_size]]

            # Create new population
            new_population = elite.copy()

            # Add mutations
            for i in range(len(elite), population_size):
                parent = elite[i % elite_size]
                mutation = np.random.normal(0, 0.05)
                child = np.clip(
                    parent + mutation,
                    self.threshold_range[0],
                    self.threshold_range[1]
                )
                new_population = np.append(new_population, child)

            population = new_population

        # Return best individual
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]

    async def _adaptive_bandits_optimization(self) -> float:
        """Optimize threshold using multi-armed bandits."""
        # Use UCB (Upper Confidence Bound) for threshold selection
        total_evaluations = sum(
            candidate.exploration_count + candidate.exploitation_count
            for candidate in self.threshold_candidates.values()
        )

        best_threshold = self.current_threshold
        best_ucb = -float('inf')

        for threshold, candidate in self.threshold_candidates.items():
            # Calculate UCB score
            avg_performance = candidate.get_balanced_score()

            n_evaluations = candidate.exploration_count + candidate.exploitation_count
            if n_evaluations == 0:
                ucb_score = float('inf')  # Prioritize unexplored arms
            else:
                confidence_bonus = np.sqrt(2 * np.log(total_evaluations) / n_evaluations)
                ucb_score = avg_performance + confidence_bonus

            if ucb_score > best_ucb:
                best_ucb = ucb_score
                best_threshold = threshold

        # Update exploration count for selected threshold
        if best_threshold in self.threshold_candidates:
            self.threshold_candidates[best_threshold].exploration_count += 1

        return best_threshold

    async def _evaluate_threshold_performance(self, threshold: float) -> float:
        """Evaluate performance of a specific threshold."""
        # Find or create candidate for this threshold
        candidate = None
        closest_threshold = min(
            self.threshold_candidates.keys(),
            key=lambda t: abs(t - threshold)
        )

        if abs(closest_threshold - threshold) < 0.01:
            candidate = self.threshold_candidates[closest_threshold]

        if candidate and candidate.performance_history:
            return candidate.get_balanced_score()

        # Fallback: estimate performance based on recent feedback
        if not self.feedback_history:
            return 0.5

        # Simulate threshold performance based on recent feedback
        simulated_feedback = defaultdict(int)

        for feedback in self.feedback_history:
            # Determine what the decision would have been with this threshold
            predicted_anomaly = feedback.anomaly_score > threshold
            actual_anomaly = feedback.feedback_type in [FeedbackType.TRUE_POSITIVE, FeedbackType.FALSE_NEGATIVE]

            if predicted_anomaly and actual_anomaly:
                simulated_feedback[FeedbackType.TRUE_POSITIVE] += 1
            elif predicted_anomaly and not actual_anomaly:
                simulated_feedback[FeedbackType.FALSE_POSITIVE] += 1
            elif not predicted_anomaly and actual_anomaly:
                simulated_feedback[FeedbackType.FALSE_NEGATIVE] += 1
            else:
                simulated_feedback[FeedbackType.TRUE_NEGATIVE] += 1

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(simulated_feedback)
        return metrics.f1_score

    def _get_feedback_summary(self) -> dict[str, Any]:
        """Get summary of recent feedback."""
        if not self.feedback_history:
            return {"no_feedback": True}

        recent_feedback = list(self.feedback_history)[-50:]
        feedback_counts = defaultdict(int)

        for feedback in recent_feedback:
            feedback_counts[feedback.feedback_type.value] += 1

        total_feedback = len(recent_feedback)

        return {
            "total_recent_feedback": total_feedback,
            "feedback_distribution": dict(feedback_counts),
            "false_positive_rate": feedback_counts[FeedbackType.FALSE_POSITIVE.value] / total_feedback,
            "false_negative_rate": feedback_counts[FeedbackType.FALSE_NEGATIVE.value] / total_feedback,
            "accuracy_estimate": (
                feedback_counts[FeedbackType.TRUE_POSITIVE.value] +
                feedback_counts[FeedbackType.TRUE_NEGATIVE.value]
            ) / total_feedback if total_feedback > 0 else 0.0
        }

    async def get_optimization_report(self) -> dict[str, Any]:
        """Get comprehensive optimization report."""
        report = {
            "optimizer_status": {
                "strategy": self.optimization_strategy.value,
                "current_threshold": self.current_threshold,
                "threshold_range": self.threshold_range,
                "total_feedback": self.total_feedback,
                "last_optimization": self.last_optimization.isoformat(),
                "optimization_interval": self.optimization_interval.total_seconds() / 60
            },
            "threshold_candidates": {
                threshold: {
                    "performance_score": candidate.get_balanced_score(),
                    "exploration_count": candidate.exploration_count,
                    "feedback_count": candidate.feedback_count,
                    "last_updated": candidate.last_updated.isoformat()
                }
                for threshold, candidate in self.threshold_candidates.items()
            },
            "performance_summary": self._get_feedback_summary(),
            "recent_performance": [
                {
                    "f1_score": perf.f1_score,
                    "precision": perf.precision,
                    "recall": perf.recall,
                    "timestamp": perf.timestamp.isoformat()
                }
                for perf in list(self.recent_performance)[-10:]
            ],
            "optimization_recommendations": self._generate_optimization_recommendations()
        }

        return report

    def _generate_optimization_recommendations(self) -> list[str]:
        """Generate recommendations for optimization improvement."""
        recommendations = []

        # Check feedback balance
        feedback_summary = self._get_feedback_summary()
        if not feedback_summary.get("no_feedback"):
            fpr = feedback_summary.get("false_positive_rate", 0)
            fnr = feedback_summary.get("false_negative_rate", 0)

            if fpr > 0.15:
                recommendations.append("High false positive rate detected - consider increasing threshold or reviewing detection criteria")

            if fnr > 0.15:
                recommendations.append("High false negative rate detected - consider decreasing threshold or improving model sensitivity")

            if fpr < 0.05 and fnr < 0.05:
                recommendations.append("Excellent threshold performance - consider fine-tuning for marginal improvements")

        # Check exploration vs exploitation
        total_explorations = sum(c.exploration_count for c in self.threshold_candidates.values())
        total_exploitations = sum(c.exploitation_count for c in self.threshold_candidates.values())

        if total_explorations < total_exploitations / 10:
            recommendations.append("Low exploration detected - consider increasing exploration rate")

        # Check optimization frequency
        time_since_optimization = (datetime.now() - self.last_optimization).total_seconds() / 60
        if time_since_optimization > self.optimization_interval.total_seconds() / 60 * 2:
            recommendations.append("Long time since last optimization - consider more frequent optimization")

        if not recommendations:
            recommendations.append("Threshold optimization appears to be working well")

        return recommendations
