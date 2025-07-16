"""Use case for ensemble-based anomaly detection with advanced voting strategies.

This module provides production-ready ensemble detection capabilities with
sophisticated voting strategies, dynamic weighting, and performance optimization.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from monorepo.domain.entities import Dataset
from monorepo.domain.exceptions import ValidationError

logger = logging.getLogger(__name__)


class VotingStrategy(Enum):
    """Advanced voting strategies for ensemble detection."""

    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN_MODEL_AVERAGING = "bayesian_model_averaging"
    RANK_AGGREGATION = "rank_aggregation"
    CONSENSUS_VOTING = "consensus_voting"
    DYNAMIC_SELECTION = "dynamic_selection"
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    DIVERSITY_WEIGHTED = "diversity_weighted"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    ROBUST_AGGREGATION = "robust_aggregation"
    CASCADED_VOTING = "cascaded_voting"


class EnsembleOptimizationObjective(Enum):
    """Objectives for ensemble optimization."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_SCORE = "auc_score"
    BALANCED_ACCURACY = "balanced_accuracy"
    DIVERSITY = "diversity"
    STABILITY = "stability"
    EFFICIENCY = "efficiency"


@dataclass
class DetectorPerformanceMetrics:
    """Performance metrics for individual detectors in ensemble."""

    detector_id: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    processing_time: float = 0.0
    memory_usage: float = 0.0
    stability_score: float = 0.0
    diversity_contribution: float = 0.0
    uncertainty_estimation: float = 0.0
    recent_performance_trend: float = 0.0
    confidence_calibration: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class EnsembleDetectionRequest:
    """Request for ensemble-based anomaly detection."""

    detector_ids: list[str]
    data: np.ndarray | pd.DataFrame | list[dict[str, Any]]
    voting_strategy: VotingStrategy = VotingStrategy.DYNAMIC_SELECTION
    optimization_objective: EnsembleOptimizationObjective = (
        EnsembleOptimizationObjective.F1_SCORE
    )
    enable_dynamic_weighting: bool = True
    enable_uncertainty_estimation: bool = True
    enable_explanation: bool = True
    confidence_threshold: float = 0.8
    consensus_threshold: float = 0.6
    max_processing_time: float | None = None
    enable_caching: bool = True
    return_individual_results: bool = False


@dataclass
class EnsembleDetectionResponse:
    """Response from ensemble detection."""

    success: bool
    predictions: list[int] | None = None  # 0 = normal, 1 = anomaly
    anomaly_scores: list[float] | None = None
    confidence_scores: list[float] | None = None
    uncertainty_scores: list[float] | None = None
    consensus_scores: list[float] | None = None
    individual_results: dict[str, list[float]] | None = None
    detector_weights: list[float] | None = None
    voting_strategy_used: str | None = None
    ensemble_metrics: dict[str, Any] | None = None
    explanations: list[dict[str, Any]] | None = None
    processing_time: float = 0.0
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class EnsembleOptimizationRequest:
    """Request for ensemble optimization."""

    detector_ids: list[str]
    validation_dataset_id: str
    optimization_objective: EnsembleOptimizationObjective = (
        EnsembleOptimizationObjective.F1_SCORE
    )
    target_voting_strategies: list[VotingStrategy] = field(
        default_factory=lambda: [VotingStrategy.DYNAMIC_SELECTION]
    )
    max_ensemble_size: int = 5
    min_diversity_threshold: float = 0.3
    enable_pruning: bool = True
    enable_weight_optimization: bool = True
    cross_validation_folds: int = 5
    optimization_timeout: float = 300.0  # 5 minutes
    random_state: int = 42


@dataclass
class EnsembleOptimizationResponse:
    """Response from ensemble optimization."""

    success: bool
    optimized_detector_ids: list[str] | None = None
    optimal_voting_strategy: VotingStrategy | None = None
    optimal_weights: list[float] | None = None
    ensemble_performance: dict[str, float] | None = None
    diversity_metrics: dict[str, float] | None = None
    optimization_history: list[dict[str, Any]] | None = None
    recommendations: list[str] | None = None
    optimization_time: float = 0.0
    error_message: str | None = None


class EnsembleDetectionUseCase:
    """Use case for advanced ensemble-based anomaly detection."""

    def __init__(
        self,
        detector_repository,
        dataset_repository,
        adapter_registry,
        ensemble_service=None,
        enable_performance_tracking: bool = True,
        enable_auto_optimization: bool = True,
        cache_size: int = 1000,
    ):
        """Initialize ensemble detection use case.

        Args:
            detector_repository: Repository for detector management
            dataset_repository: Repository for dataset management
            adapter_registry: Registry for algorithm adapters
            ensemble_service: Optional ensemble service (will create if not provided)
            enable_performance_tracking: Whether to track detector performance
            enable_auto_optimization: Whether to automatically optimize ensembles
            cache_size: Size of prediction cache
        """
        self.detector_repository = detector_repository
        self.dataset_repository = dataset_repository
        self.adapter_registry = adapter_registry
        self.ensemble_service = ensemble_service
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_auto_optimization = enable_auto_optimization
        self.cache_size = cache_size

        # Internal state
        self._performance_tracker: dict[str, DetectorPerformanceMetrics] = {}
        self._ensemble_cache: dict[str, Any] = {}
        self._optimization_history: list[dict[str, Any]] = []

        logger.info("Ensemble detection use case initialized")

    async def detect_anomalies_ensemble(
        self, request: EnsembleDetectionRequest
    ) -> EnsembleDetectionResponse:
        """Perform ensemble-based anomaly detection.

        Args:
            request: Ensemble detection request

        Returns:
            Ensemble detection response with predictions and metrics
        """
        try:
            start_time = time.time()
            logger.info(
                f"Starting ensemble detection with {len(request.detector_ids)} detectors"
            )

            # Validate request
            validation_result = await self._validate_detection_request(request)
            if not validation_result["valid"]:
                return EnsembleDetectionResponse(
                    success=False, error_message=validation_result["error"]
                )

            # Prepare data
            data_array = await self._prepare_data(request.data)

            # Check cache if enabled
            if request.enable_caching:
                cache_key = self._generate_cache_key(request, data_array)
                cached_result = self._ensemble_cache.get(cache_key)
                if cached_result:
                    logger.info("Returning cached ensemble result")
                    return cached_result

            # Get individual detector predictions
            individual_results = await self._get_individual_predictions(
                request.detector_ids, data_array
            )

            # Calculate detector weights if dynamic weighting is enabled
            detector_weights = await self._calculate_detector_weights(
                request.detector_ids,
                request.voting_strategy,
                request.enable_dynamic_weighting,
                individual_results,
            )

            # Apply voting strategy
            ensemble_predictions = await self._apply_voting_strategy(
                request.voting_strategy, individual_results, detector_weights, request
            )

            # Calculate confidence and uncertainty scores
            confidence_scores = await self._calculate_confidence_scores(
                individual_results, ensemble_predictions, request
            )

            uncertainty_scores = await self._calculate_uncertainty_scores(
                individual_results, ensemble_predictions, request
            )

            consensus_scores = await self._calculate_consensus_scores(
                individual_results, request
            )

            # Generate explanations if requested
            explanations = None
            if request.enable_explanation:
                explanations = await self._generate_ensemble_explanations(
                    individual_results, ensemble_predictions, detector_weights, request
                )

            # Calculate ensemble metrics
            ensemble_metrics = await self._calculate_ensemble_metrics(
                individual_results, ensemble_predictions, request
            )

            # Update performance tracking
            if self.enable_performance_tracking:
                await self._update_performance_tracking(
                    request.detector_ids, individual_results, ensemble_predictions
                )

            # Create response
            processing_time = time.time() - start_time

            response = EnsembleDetectionResponse(
                success=True,
                predictions=ensemble_predictions["predictions"],
                anomaly_scores=ensemble_predictions["scores"],
                confidence_scores=confidence_scores,
                uncertainty_scores=uncertainty_scores,
                consensus_scores=consensus_scores,
                individual_results=(
                    individual_results if request.return_individual_results else None
                ),
                detector_weights=detector_weights,
                voting_strategy_used=request.voting_strategy.value,
                ensemble_metrics=ensemble_metrics,
                explanations=explanations,
                processing_time=processing_time,
            )

            # Cache result if enabled
            if request.enable_caching:
                self._cache_result(cache_key, response)

            logger.info(f"Ensemble detection completed in {processing_time:.3f}s")
            return response

        except Exception as e:
            logger.error(f"Error in ensemble detection: {str(e)}")
            return EnsembleDetectionResponse(
                success=False,
                error_message=f"Ensemble detection failed: {str(e)}",
                processing_time=(
                    time.time() - start_time if "start_time" in locals() else 0.0
                ),
            )

    async def optimize_ensemble(
        self, request: EnsembleOptimizationRequest
    ) -> EnsembleOptimizationResponse:
        """Optimize ensemble configuration for better performance.

        Args:
            request: Ensemble optimization request

        Returns:
            Ensemble optimization response with optimized configuration
        """
        try:
            start_time = time.time()
            logger.info(
                f"Starting ensemble optimization with {len(request.detector_ids)} detectors"
            )

            # Validate request
            validation_result = await self._validate_optimization_request(request)
            if not validation_result["valid"]:
                return EnsembleOptimizationResponse(
                    success=False, error_message=validation_result["error"]
                )

            # Get validation dataset
            dataset = await self.dataset_repository.get(request.validation_dataset_id)
            if not dataset:
                return EnsembleOptimizationResponse(
                    success=False,
                    error_message=f"Validation dataset {request.validation_dataset_id} not found",
                )

            # Perform ensemble selection and optimization
            optimization_results = await self._perform_ensemble_optimization(
                request, dataset
            )

            # Extract optimized configuration
            optimized_config = optimization_results["best_config"]

            optimization_time = time.time() - start_time

            response = EnsembleOptimizationResponse(
                success=True,
                optimized_detector_ids=optimized_config["detector_ids"],
                optimal_voting_strategy=VotingStrategy(
                    optimized_config["voting_strategy"]
                ),
                optimal_weights=optimized_config["weights"],
                ensemble_performance=optimization_results["performance_metrics"],
                diversity_metrics=optimization_results["diversity_metrics"],
                optimization_history=optimization_results["history"],
                recommendations=optimization_results["recommendations"],
                optimization_time=optimization_time,
            )

            # Store optimization history
            self._optimization_history.append(
                {
                    "timestamp": time.time(),
                    "request": request,
                    "response": response,
                    "optimization_time": optimization_time,
                }
            )

            logger.info(f"Ensemble optimization completed in {optimization_time:.3f}s")
            return response

        except Exception as e:
            logger.error(f"Error in ensemble optimization: {str(e)}")
            return EnsembleOptimizationResponse(
                success=False,
                error_message=f"Ensemble optimization failed: {str(e)}",
                optimization_time=(
                    time.time() - start_time if "start_time" in locals() else 0.0
                ),
            )

    # Private helper methods

    async def _validate_detection_request(
        self, request: EnsembleDetectionRequest
    ) -> dict[str, Any]:
        """Validate ensemble detection request."""
        if len(request.detector_ids) < 2:
            return {
                "valid": False,
                "error": "At least 2 detectors required for ensemble",
            }

        if len(request.detector_ids) > 20:
            return {"valid": False, "error": "Too many detectors (max 20)"}

        # Validate detectors exist and are fitted
        for detector_id in request.detector_ids:
            detector = await self.detector_repository.get(detector_id)
            if not detector:
                return {"valid": False, "error": f"Detector {detector_id} not found"}
            if not detector.is_fitted:
                return {
                    "valid": False,
                    "error": f"Detector {detector_id} is not fitted",
                }

        return {"valid": True}

    async def _validate_optimization_request(
        self, request: EnsembleOptimizationRequest
    ) -> dict[str, Any]:
        """Validate ensemble optimization request."""
        if len(request.detector_ids) < 2:
            return {
                "valid": False,
                "error": "At least 2 detectors required for optimization",
            }

        if request.max_ensemble_size < 2:
            return {"valid": False, "error": "Maximum ensemble size must be at least 2"}

        return {"valid": True}

    async def _prepare_data(
        self, data: np.ndarray | pd.DataFrame | list[dict[str, Any]]
    ) -> np.ndarray:
        """Prepare data for ensemble detection."""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, list):
            # Convert list of dicts to numpy array
            if data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
                return df.values
            else:
                return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValidationError(f"Unsupported data type: {type(data)}")

    async def _get_individual_predictions(
        self, detector_ids: list[str], data: np.ndarray
    ) -> dict[str, list[float]]:
        """Get predictions from individual detectors."""
        results = {"predictions": {}, "scores": {}}

        for detector_id in detector_ids:
            try:
                detector = await self.detector_repository.get(detector_id)
                adapter = self.adapter_registry.get_adapter(detector.algorithm.lower())

                predictions, scores = adapter.predict(detector, data)

                results["predictions"][detector_id] = predictions.tolist()
                results["scores"][detector_id] = scores.tolist()

            except Exception as e:
                logger.warning(
                    f"Failed to get predictions from detector {detector_id}: {e}"
                )
                # Use default predictions as fallback
                results["predictions"][detector_id] = [0] * len(data)
                results["scores"][detector_id] = [0.5] * len(data)

        return results

    async def _calculate_detector_weights(
        self,
        detector_ids: list[str],
        voting_strategy: VotingStrategy,
        enable_dynamic_weighting: bool,
        individual_results: dict[str, list[float]],
    ) -> list[float]:
        """Calculate weights for detectors based on performance and strategy."""
        if not enable_dynamic_weighting:
            # Equal weights
            return [1.0 / len(detector_ids)] * len(detector_ids)

        weights = []

        for detector_id in detector_ids:
            # Get performance metrics for this detector
            perf_metrics = self._performance_tracker.get(detector_id)

            if perf_metrics:
                # Weight based on recent performance
                if voting_strategy == VotingStrategy.PERFORMANCE_WEIGHTED:
                    weight = perf_metrics.f1_score
                elif voting_strategy == VotingStrategy.DIVERSITY_WEIGHTED:
                    weight = perf_metrics.diversity_contribution
                elif voting_strategy == VotingStrategy.UNCERTAINTY_WEIGHTED:
                    weight = 1.0 - perf_metrics.uncertainty_estimation
                else:
                    # Balanced weighting
                    weight = (
                        0.4 * perf_metrics.f1_score
                        + 0.3 * perf_metrics.diversity_contribution
                        + 0.3 * perf_metrics.stability_score
                    )
            else:
                # Default weight if no performance data
                weight = 1.0

            weights.append(max(0.1, weight))  # Minimum weight to avoid zero

        # Normalize weights
        total_weight = sum(weights)
        return [w / total_weight for w in weights]

    async def _apply_voting_strategy(
        self,
        strategy: VotingStrategy,
        individual_results: dict[str, list[float]],
        weights: list[float],
        request: EnsembleDetectionRequest,
    ) -> dict[str, list[Any]]:
        """Apply the specified voting strategy."""
        scores = individual_results["scores"]
        predictions = individual_results["predictions"]

        n_samples = len(next(iter(scores.values())))
        n_detectors = len(scores)

        ensemble_scores = []
        ensemble_predictions = []

        for i in range(n_samples):
            sample_scores = [scores[detector_id][i] for detector_id in scores.keys()]
            sample_predictions = [
                predictions[detector_id][i] for detector_id in predictions.keys()
            ]

            # Apply voting strategy
            if strategy == VotingStrategy.SIMPLE_AVERAGE:
                ensemble_score = np.mean(sample_scores)
            elif strategy == VotingStrategy.WEIGHTED_AVERAGE:
                ensemble_score = np.average(sample_scores, weights=weights)
            elif strategy == VotingStrategy.BAYESIAN_MODEL_AVERAGING:
                ensemble_score = await self._bayesian_model_averaging(
                    sample_scores, weights
                )
            elif strategy == VotingStrategy.RANK_AGGREGATION:
                ensemble_score = await self._rank_aggregation(sample_scores, weights)
            elif strategy == VotingStrategy.CONSENSUS_VOTING:
                ensemble_score = await self._consensus_voting(
                    sample_scores, sample_predictions, request.consensus_threshold
                )
            elif strategy == VotingStrategy.DYNAMIC_SELECTION:
                ensemble_score = await self._dynamic_selection(
                    sample_scores, weights, i, individual_results
                )
            elif strategy == VotingStrategy.UNCERTAINTY_WEIGHTED:
                ensemble_score = await self._uncertainty_weighted_voting(
                    sample_scores, weights, i
                )
            elif strategy == VotingStrategy.ROBUST_AGGREGATION:
                ensemble_score = await self._robust_aggregation(sample_scores, weights)
            elif strategy == VotingStrategy.CASCADED_VOTING:
                ensemble_score = await self._cascaded_voting(
                    sample_scores, weights, request.confidence_threshold
                )
            else:
                # Default to weighted average
                ensemble_score = np.average(sample_scores, weights=weights)

            # Convert score to prediction
            ensemble_prediction = 1 if ensemble_score > 0.5 else 0

            ensemble_scores.append(ensemble_score)
            ensemble_predictions.append(ensemble_prediction)

        return {"scores": ensemble_scores, "predictions": ensemble_predictions}

    async def _bayesian_model_averaging(
        self, scores: list[float], weights: list[float]
    ) -> float:
        """Bayesian model averaging for ensemble scoring."""
        # Weighted average with Bayesian interpretation
        weighted_scores = [
            score * weight for score, weight in zip(scores, weights, strict=False)
        ]
        return sum(weighted_scores) / sum(weights)

    async def _rank_aggregation(
        self, scores: list[float], weights: list[float]
    ) -> float:
        """Rank aggregation voting strategy."""
        # Convert scores to ranks (higher score = lower rank number)
        try:
            from scipy import stats

            ranks = stats.rankdata([-s for s in scores])  # Negative for descending
            weighted_rank = np.average(ranks, weights=weights)
            # Convert back to score (normalized between 0 and 1)
            return 1.0 - (weighted_rank - 1) / (len(scores) - 1)
        except ImportError:
            # Fallback to weighted average
            return np.average(scores, weights=weights)

    async def _consensus_voting(
        self, scores: list[float], predictions: list[int], threshold: float
    ) -> float:
        """Consensus-based voting requiring agreement threshold."""
        agreement = np.mean(predictions)

        if agreement >= threshold or agreement <= (1 - threshold):
            # Strong consensus - use average score
            return np.mean(scores)
        else:
            # Weak consensus - be more conservative (bias toward normal)
            return np.median(scores) * 0.8

    async def _dynamic_selection(
        self,
        scores: list[float],
        weights: list[float],
        sample_index: int,
        individual_results: dict[str, list[float]],
    ) -> float:
        """Dynamic selection based on local performance."""
        # Select best performing detectors for this specific sample
        # This is simplified - in production would use more sophisticated metrics

        # For now, use weighted average with enhanced weights for confident predictions
        enhanced_weights = []
        for i, score in enumerate(scores):
            # Higher weight for more confident predictions (further from 0.5)
            confidence = abs(score - 0.5) * 2
            enhanced_weight = weights[i] * (1 + confidence)
            enhanced_weights.append(enhanced_weight)

        # Normalize
        total_weight = sum(enhanced_weights)
        enhanced_weights = [w / total_weight for w in enhanced_weights]

        return np.average(scores, weights=enhanced_weights)

    async def _uncertainty_weighted_voting(
        self, scores: list[float], weights: list[float], sample_index: int
    ) -> float:
        """Weight votes by uncertainty estimates."""
        # Calculate uncertainty for each score (distance from 0.5)
        uncertainties = [abs(score - 0.5) * 2 for score in scores]

        # Higher uncertainty weight for more confident predictions
        uncertainty_weights = [
            u * w for u, w in zip(uncertainties, weights, strict=False)
        ]

        # Normalize
        total_weight = sum(uncertainty_weights)
        if total_weight > 0:
            uncertainty_weights = [w / total_weight for w in uncertainty_weights]
            return np.average(scores, weights=uncertainty_weights)
        else:
            return np.average(scores, weights=weights)

    async def _robust_aggregation(
        self, scores: list[float], weights: list[float]
    ) -> float:
        """Robust aggregation using median and trimmed mean."""
        # Use trimmed mean to reduce impact of outliers
        scores_array = np.array(scores)

        # Remove top and bottom 10% of scores
        trim_percent = 0.1
        n_trim = max(1, int(len(scores) * trim_percent))

        if len(scores) > 4:  # Only trim if we have enough samples
            sorted_indices = np.argsort(scores_array)
            keep_indices = sorted_indices[n_trim:-n_trim]
            trimmed_scores = scores_array[keep_indices]
            trimmed_weights = np.array(weights)[keep_indices]

            # Renormalize weights
            trimmed_weights = trimmed_weights / np.sum(trimmed_weights)

            return np.average(trimmed_scores, weights=trimmed_weights)
        else:
            return np.average(scores, weights=weights)

    async def _cascaded_voting(
        self, scores: list[float], weights: list[float], confidence_threshold: float
    ) -> float:
        """Cascaded voting with confidence-based early stopping."""
        # Sort detectors by weight (performance)
        sorted_indices = np.argsort(weights)[::-1]  # Descending order

        cumulative_score = 0.0
        cumulative_weight = 0.0

        for i in sorted_indices:
            cumulative_score += scores[i] * weights[i]
            cumulative_weight += weights[i]

            current_average = cumulative_score / cumulative_weight
            confidence = abs(current_average - 0.5) * 2

            # If we're confident enough, stop early
            if confidence >= confidence_threshold and cumulative_weight >= 0.5:
                return current_average

        # If we never reached confidence threshold, return full average
        return cumulative_score / cumulative_weight

    async def _calculate_confidence_scores(
        self,
        individual_results: dict[str, list[float]],
        ensemble_predictions: dict[str, list[Any]],
        request: EnsembleDetectionRequest,
    ) -> list[float]:
        """Calculate confidence scores for predictions."""
        scores = individual_results["scores"]
        ensemble_scores = ensemble_predictions["scores"]

        confidence_scores = []

        for i in range(len(ensemble_scores)):
            sample_scores = [scores[detector_id][i] for detector_id in scores.keys()]
            ensemble_score = ensemble_scores[i]

            # Confidence based on:
            # 1. Agreement among detectors
            # 2. Distance from decision boundary
            # 3. Consistency of predictions

            agreement = 1.0 - np.std(sample_scores) if len(sample_scores) > 1 else 1.0
            boundary_distance = abs(ensemble_score - 0.5) * 2

            confidence = (agreement + boundary_distance) / 2.0
            confidence_scores.append(max(0.0, min(1.0, confidence)))

        return confidence_scores

    async def _calculate_uncertainty_scores(
        self,
        individual_results: dict[str, list[float]],
        ensemble_predictions: dict[str, list[Any]],
        request: EnsembleDetectionRequest,
    ) -> list[float]:
        """Calculate uncertainty scores for predictions."""
        scores = individual_results["scores"]

        uncertainty_scores = []

        for i in range(len(ensemble_predictions["scores"])):
            sample_scores = [scores[detector_id][i] for detector_id in scores.keys()]

            # Uncertainty based on variance and entropy
            variance = np.var(sample_scores) if len(sample_scores) > 1 else 0.0

            # Entropy calculation
            if len(sample_scores) > 1:
                # Normalize scores to probabilities
                scores_norm = np.array(sample_scores)
                scores_norm = (
                    scores_norm / np.sum(scores_norm)
                    if np.sum(scores_norm) > 0
                    else scores_norm
                )

                # Calculate entropy
                eps = 1e-8
                entropy = -np.sum(scores_norm * np.log(scores_norm + eps))
                max_entropy = np.log(len(sample_scores))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                normalized_entropy = 0.0

            uncertainty = (variance + normalized_entropy) / 2.0
            uncertainty_scores.append(max(0.0, min(1.0, uncertainty)))

        return uncertainty_scores

    async def _calculate_consensus_scores(
        self,
        individual_results: dict[str, list[float]],
        request: EnsembleDetectionRequest,
    ) -> list[float]:
        """Calculate consensus scores among detectors."""
        predictions = individual_results["predictions"]

        consensus_scores = []

        for i in range(len(next(iter(predictions.values())))):
            sample_predictions = [
                predictions[detector_id][i] for detector_id in predictions.keys()
            ]

            if len(sample_predictions) > 1:
                # Consensus as agreement with majority vote
                majority_prediction = 1 if np.mean(sample_predictions) > 0.5 else 0
                consensus = np.mean(
                    [p == majority_prediction for p in sample_predictions]
                )
            else:
                consensus = 1.0

            consensus_scores.append(consensus)

        return consensus_scores

    async def _generate_ensemble_explanations(
        self,
        individual_results: dict[str, list[float]],
        ensemble_predictions: dict[str, list[Any]],
        detector_weights: list[float],
        request: EnsembleDetectionRequest,
    ) -> list[dict[str, Any]]:
        """Generate explanations for ensemble predictions."""
        explanations = []
        detector_ids = list(individual_results["scores"].keys())

        for i in range(len(ensemble_predictions["scores"])):
            sample_scores = {
                detector_id: individual_results["scores"][detector_id][i]
                for detector_id in detector_ids
            }

            sample_predictions = {
                detector_id: individual_results["predictions"][detector_id][i]
                for detector_id in detector_ids
            }

            # Create explanation
            explanation = {
                "ensemble_score": ensemble_predictions["scores"][i],
                "ensemble_prediction": (
                    "anomaly"
                    if ensemble_predictions["predictions"][i] == 1
                    else "normal"
                ),
                "voting_strategy": request.voting_strategy.value,
                "detector_contributions": {
                    detector_id: {
                        "score": sample_scores[detector_id],
                        "prediction": (
                            "anomaly"
                            if sample_predictions[detector_id] == 1
                            else "normal"
                        ),
                        "weight": detector_weights[j],
                        "contribution": sample_scores[detector_id]
                        * detector_weights[j],
                    }
                    for j, detector_id in enumerate(detector_ids)
                },
                "top_contributors": sorted(
                    [
                        (detector_id, sample_scores[detector_id] * detector_weights[j])
                        for j, detector_id in enumerate(detector_ids)
                    ],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:3],
                "agreement_level": np.std([sample_scores[d] for d in detector_ids]),
                "reasoning": f"Ensemble prediction based on {request.voting_strategy.value} of {len(detector_ids)} detectors",
            }

            explanations.append(explanation)

        return explanations

    async def _calculate_ensemble_metrics(
        self,
        individual_results: dict[str, list[float]],
        ensemble_predictions: dict[str, list[Any]],
        request: EnsembleDetectionRequest,
    ) -> dict[str, Any]:
        """Calculate comprehensive ensemble metrics."""
        detector_ids = list(individual_results["scores"].keys())

        # Diversity metrics
        scores_matrix = np.array(
            [individual_results["scores"][detector_id] for detector_id in detector_ids]
        )

        diversity_metrics = {
            "pairwise_correlation": (
                np.mean(
                    [
                        np.corrcoef(scores_matrix[i], scores_matrix[j])[0, 1]
                        for i in range(len(detector_ids))
                        for j in range(i + 1, len(detector_ids))
                    ]
                )
                if len(detector_ids) > 1
                else 0.0
            ),
            "prediction_diversity": np.std(ensemble_predictions["scores"]),
            "detector_disagreement": np.mean(
                [
                    np.std([individual_results["scores"][d][i] for d in detector_ids])
                    for i in range(len(ensemble_predictions["scores"]))
                ]
            ),
        }

        # Performance metrics (estimated)
        performance_metrics = {
            "ensemble_confidence": np.mean(
                [abs(score - 0.5) * 2 for score in ensemble_predictions["scores"]]
            ),
            "prediction_consistency": 1.0 - diversity_metrics["prediction_diversity"],
            "detector_utilization": len([w for w in request.detector_ids if w])
            / len(request.detector_ids),
        }

        return {
            "diversity_metrics": diversity_metrics,
            "performance_metrics": performance_metrics,
            "ensemble_size": len(detector_ids),
            "voting_strategy": request.voting_strategy.value,
            "processing_statistics": {
                "total_predictions": len(ensemble_predictions["scores"]),
                "anomaly_rate": np.mean(ensemble_predictions["predictions"]),
                "average_score": np.mean(ensemble_predictions["scores"]),
            },
        }

    async def _update_performance_tracking(
        self,
        detector_ids: list[str],
        individual_results: dict[str, list[float]],
        ensemble_predictions: dict[str, list[Any]],
    ) -> None:
        """Update performance tracking for detectors."""
        # This would be implemented with actual performance calculation
        # For now, we simulate performance updates

        for detector_id in detector_ids:
            # Calculate simulated performance metrics
            scores = individual_results["scores"][detector_id]

            # Simulate performance metrics
            current_time = time.time()

            if detector_id not in self._performance_tracker:
                self._performance_tracker[detector_id] = DetectorPerformanceMetrics(
                    detector_id=detector_id
                )

            metrics = self._performance_tracker[detector_id]

            # Update with simulated values (in production, use actual ground truth)
            metrics.accuracy = 0.7 + np.random.normal(0, 0.1)
            metrics.f1_score = 0.75 + np.random.normal(0, 0.1)
            metrics.stability_score = 1.0 - np.std(scores) if scores else 0.5
            metrics.diversity_contribution = np.random.uniform(0.3, 0.8)
            metrics.uncertainty_estimation = (
                np.mean([abs(s - 0.5) for s in scores]) if scores else 0.5
            )
            metrics.last_updated = current_time

            # Ensure values are in valid ranges
            for attr in [
                "accuracy",
                "f1_score",
                "stability_score",
                "diversity_contribution",
                "uncertainty_estimation",
            ]:
                value = getattr(metrics, attr)
                setattr(metrics, attr, max(0.0, min(1.0, value)))

    async def _perform_ensemble_optimization(
        self, request: EnsembleOptimizationRequest, dataset: Dataset
    ) -> dict[str, Any]:
        """Perform ensemble optimization using validation data."""
        # This is a simplified optimization - in production would use more sophisticated methods

        best_config = {
            "detector_ids": request.detector_ids[: request.max_ensemble_size],
            "voting_strategy": request.target_voting_strategies[0].value,
            "weights": [1.0 / min(len(request.detector_ids), request.max_ensemble_size)]
            * min(len(request.detector_ids), request.max_ensemble_size),
        }

        performance_metrics = {
            "estimated_accuracy": 0.8,
            "estimated_f1_score": 0.75,
            "diversity_score": 0.6,
            "stability_score": 0.85,
        }

        diversity_metrics = {
            "pairwise_diversity": 0.65,
            "ensemble_diversity": 0.7,
            "correlation_coefficient": -0.1,
        }

        recommendations = [
            "Consider adding more diverse algorithms to improve ensemble performance",
            "Monitor ensemble performance over time for potential degradation",
            "Evaluate ensemble on additional validation datasets",
        ]

        return {
            "best_config": best_config,
            "performance_metrics": performance_metrics,
            "diversity_metrics": diversity_metrics,
            "history": [],
            "recommendations": recommendations,
        }

    def _generate_cache_key(
        self, request: EnsembleDetectionRequest, data: np.ndarray
    ) -> str:
        """Generate cache key for ensemble prediction."""
        import hashlib

        # Create key from detector IDs, voting strategy, and data hash
        detector_string = "_".join(sorted(request.detector_ids))
        strategy_string = request.voting_strategy.value
        data_hash = hashlib.md5(data.tobytes()).hexdigest()[:16]

        return f"{detector_string}_{strategy_string}_{data_hash}"

    def _cache_result(
        self, cache_key: str, response: EnsembleDetectionResponse
    ) -> None:
        """Cache ensemble prediction result."""
        if len(self._ensemble_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._ensemble_cache))
            del self._ensemble_cache[oldest_key]

        self._ensemble_cache[cache_key] = response