"""Ensemble meta-adapter for combining multiple anomaly detection algorithms."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult
from pynomaly.domain.exceptions import DetectorNotFittedError, FittingError
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.shared.protocols import DetectorProtocol, EnsembleDetectorProtocol


class AggregationMethod(Enum):
    """Ensemble aggregation methods."""

    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    MAJORITY_VOTE = "majority_vote"
    ADAPTIVE = "adaptive"
    STACKING = "stacking"


@dataclass
class DetectorConfig:
    """Configuration for a detector in the ensemble."""

    detector: DetectorProtocol
    weight: float = 1.0
    enabled: bool = True
    performance_score: Optional[float] = None


@dataclass
class EnsembleMetadata:
    """Metadata for ensemble detection."""

    n_detectors: int
    aggregation_method: str
    individual_scores: Dict[str, List[float]]
    individual_predictions: Dict[str, List[int]]
    confidence_scores: List[float]
    agreement_scores: List[float]


class EnsembleMetaAdapter(EnsembleDetectorProtocol):
    """Meta-adapter for ensemble anomaly detection."""

    def __init__(
        self,
        name: str = "EnsembleDetector",
        contamination_rate: Optional[ContaminationRate] = None,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        """Initialize ensemble meta-adapter.

        Args:
            name: Name of the ensemble detector
            contamination_rate: Expected contamination rate
            aggregation_method: Method for combining predictions
            enable_parallel: Whether to run detectors in parallel
            max_workers: Maximum number of parallel workers
        """
        self._name = name
        self._contamination_rate = contamination_rate or ContaminationRate.auto()
        self._aggregation_method = aggregation_method
        self._enable_parallel = enable_parallel
        self._max_workers = max_workers

        # Detector management
        self._detector_configs: Dict[str, DetectorConfig] = {}
        self._is_fitted = False
        self._ensemble_metadata: Optional[EnsembleMetadata] = None

        # Performance tracking
        self._performance_history: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Get the ensemble detector name."""
        return self._name

    @property
    def contamination_rate(self) -> ContaminationRate:
        """Get the contamination rate."""
        return self._contamination_rate

    @property
    def is_fitted(self) -> bool:
        """Check if the ensemble has been fitted."""
        return self._is_fitted

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get ensemble parameters."""
        return {
            "aggregation_method": self._aggregation_method.value,
            "enable_parallel": self._enable_parallel,
            "max_workers": self._max_workers,
            "n_detectors": len(self._detector_configs),
            "detector_names": list(self._detector_configs.keys()),
        }

    @property
    def base_detectors(self) -> List[DetectorProtocol]:
        """Get the base detectors in the ensemble."""
        return [
            config.detector
            for config in self._detector_configs.values()
            if config.enabled
        ]

    def add_detector(self, detector: DetectorProtocol, weight: float = 1.0) -> None:
        """Add a detector to the ensemble.

        Args:
            detector: Detector to add
            weight: Weight for this detector's votes
        """
        config = DetectorConfig(detector=detector, weight=weight, enabled=True)
        self._detector_configs[detector.name] = config

        # Reset fitted state when ensemble composition changes
        self._is_fitted = False

    def remove_detector(self, detector_name: str) -> None:
        """Remove a detector from the ensemble.

        Args:
            detector_name: Name of detector to remove
        """
        if detector_name in self._detector_configs:
            del self._detector_configs[detector_name]
            self._is_fitted = False

    def enable_detector(self, detector_name: str, enabled: bool = True) -> None:
        """Enable or disable a detector.

        Args:
            detector_name: Name of detector to enable/disable
            enabled: Whether to enable the detector
        """
        if detector_name in self._detector_configs:
            self._detector_configs[detector_name].enabled = enabled

    def update_detector_weight(self, detector_name: str, weight: float) -> None:
        """Update the weight of a detector.

        Args:
            detector_name: Name of detector to update
            weight: New weight value
        """
        if detector_name in self._detector_configs:
            self._detector_configs[detector_name].weight = weight

    def get_detector_weights(self) -> Dict[str, float]:
        """Get weights of all detectors in the ensemble.

        Returns:
            Dictionary mapping detector names to weights
        """
        return {
            name: config.weight
            for name, config in self._detector_configs.items()
            if config.enabled
        }

    def fit(self, dataset: Dataset) -> None:
        """Fit all detectors in the ensemble.

        Args:
            dataset: Dataset to fit on

        Raises:
            FittingError: If fitting fails
        """
        if not self._detector_configs:
            raise FittingError(
                detector_name=self._name,
                reason="No detectors in ensemble",
                dataset_name=dataset.name,
            )

        start_time = time.perf_counter()

        try:
            enabled_configs = [
                (name, config)
                for name, config in self._detector_configs.items()
                if config.enabled
            ]

            if self._enable_parallel and len(enabled_configs) > 1:
                self._fit_parallel(dataset, enabled_configs)
            else:
                self._fit_sequential(dataset, enabled_configs)

            self._is_fitted = True
            fitting_time = time.perf_counter() - start_time

            # Track performance
            self._performance_history.append(
                {
                    "operation": "fit",
                    "dataset_name": dataset.name,
                    "n_samples": len(dataset.data),
                    "n_detectors": len(enabled_configs),
                    "fitting_time": fitting_time,
                    "timestamp": time.time(),
                }
            )

        except Exception as e:
            raise FittingError(
                detector_name=self._name,
                reason=f"Ensemble fitting failed: {str(e)}",
                dataset_name=dataset.name,
            ) from e

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies using the ensemble.

        Args:
            dataset: Dataset to analyze

        Returns:
            Ensemble detection result
        """
        if not self._is_fitted:
            raise DetectorNotFittedError(detector_name=self._name, operation="detect")

        start_time = time.perf_counter()

        try:
            enabled_configs = [
                (name, config)
                for name, config in self._detector_configs.items()
                if config.enabled
            ]

            # Get individual results
            if self._enable_parallel and len(enabled_configs) > 1:
                individual_results = self._detect_parallel(dataset, enabled_configs)
            else:
                individual_results = self._detect_sequential(dataset, enabled_configs)

            # Aggregate results
            ensemble_result = self._aggregate_results(dataset, individual_results)

            detection_time = time.perf_counter() - start_time

            # Update metadata
            ensemble_result.metadata.update(
                {
                    "ensemble_detection_time": detection_time,
                    "aggregation_method": self._aggregation_method.value,
                    "n_detectors": len(individual_results),
                }
            )

            return ensemble_result

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000

            # Return empty result with error
            return DetectionResult(
                detector_id=hash(self._name),
                dataset_id=dataset.id,
                anomalies=[],
                scores=[],
                labels=[],
                threshold=0.5,
                execution_time_ms=execution_time,
                metadata={
                    "ensemble_name": self._name,
                    "error": str(e),
                    "status": "failed",
                },
            )

    def score(self, dataset: Dataset) -> List[AnomalyScore]:
        """Calculate ensemble anomaly scores.

        Args:
            dataset: Dataset to score

        Returns:
            List of ensemble anomaly scores
        """
        if not self._is_fitted:
            raise DetectorNotFittedError(detector_name=self._name, operation="score")

        enabled_configs = [
            (name, config)
            for name, config in self._detector_configs.items()
            if config.enabled
        ]

        # Get individual scores
        all_scores = {}
        for name, config in enabled_configs:
            try:
                scores = config.detector.score(dataset)
                all_scores[name] = [score.value for score in scores]
            except Exception:
                # Skip failed detectors
                continue

        if not all_scores:
            return []

        # Aggregate scores
        aggregated_scores = self._aggregate_scores(all_scores, enabled_configs)

        return [
            AnomalyScore(
                value=float(score), method=f"ensemble_{self._aggregation_method.value}"
            )
            for score in aggregated_scores
        ]

    def fit_detect(self, dataset: Dataset) -> DetectionResult:
        """Fit ensemble and detect anomalies in one step.

        Args:
            dataset: Dataset to fit and analyze

        Returns:
            Detection result
        """
        self.fit(dataset)
        return self.detect(dataset)

    def get_params(self) -> Dict[str, Any]:
        """Get ensemble parameters."""
        return self.parameters

    def set_params(self, **params: Any) -> None:
        """Set ensemble parameters."""
        if "aggregation_method" in params:
            method_value = params["aggregation_method"]
            if isinstance(method_value, str):
                self._aggregation_method = AggregationMethod(method_value)
            elif isinstance(method_value, AggregationMethod):
                self._aggregation_method = method_value

        if "enable_parallel" in params:
            self._enable_parallel = params["enable_parallel"]

        if "max_workers" in params:
            self._max_workers = params["max_workers"]

    def _fit_parallel(
        self, dataset: Dataset, configs: List[Tuple[str, DetectorConfig]]
    ) -> None:
        """Fit detectors in parallel."""
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(config.detector.fit, dataset): name
                for name, config in configs
            }

            for future in as_completed(futures):
                detector_name = futures[future]
                try:
                    future.result()  # This will raise any exceptions
                except Exception as e:
                    raise FittingError(
                        detector_name=detector_name,
                        reason=str(e),
                        dataset_name=dataset.name,
                    ) from e

    def _fit_sequential(
        self, dataset: Dataset, configs: List[Tuple[str, DetectorConfig]]
    ) -> None:
        """Fit detectors sequentially."""
        for name, config in configs:
            try:
                config.detector.fit(dataset)
            except Exception as e:
                raise FittingError(
                    detector_name=name, reason=str(e), dataset_name=dataset.name
                ) from e

    def _detect_parallel(
        self, dataset: Dataset, configs: List[Tuple[str, DetectorConfig]]
    ) -> Dict[str, DetectionResult]:
        """Run detection in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(config.detector.detect, dataset): name
                for name, config in configs
            }

            for future in as_completed(futures):
                detector_name = futures[future]
                try:
                    result = future.result()
                    results[detector_name] = result
                except Exception:
                    # Skip failed detectors in ensemble
                    continue

        return results

    def _detect_sequential(
        self, dataset: Dataset, configs: List[Tuple[str, DetectorConfig]]
    ) -> Dict[str, DetectionResult]:
        """Run detection sequentially."""
        results = {}

        for name, config in configs:
            try:
                result = config.detector.detect(dataset)
                results[name] = result
            except Exception:
                # Skip failed detectors in ensemble
                continue

        return results

    def _aggregate_results(
        self, dataset: Dataset, individual_results: Dict[str, DetectionResult]
    ) -> DetectionResult:
        """Aggregate individual detection results into ensemble result."""
        if not individual_results:
            raise FittingError(
                detector_name=self._name,
                reason="No valid individual results to aggregate",
                dataset_name=dataset.name,
            )

        # Extract scores and predictions
        all_scores = {}
        all_predictions = {}

        for name, result in individual_results.items():
            all_scores[name] = [score.value for score in result.scores]
            all_predictions[name] = result.labels

        # Get detector weights
        weights = {}
        for name in individual_results.keys():
            if name in self._detector_configs:
                weights[name] = self._detector_configs[name].weight
            else:
                weights[name] = 1.0

        # Aggregate scores
        aggregated_scores = self._aggregate_scores_with_method(all_scores, weights)

        # Aggregate predictions
        aggregated_labels = self._aggregate_predictions_with_method(
            all_predictions, weights
        )

        # Calculate threshold
        threshold = self._calculate_ensemble_threshold(aggregated_scores)

        # Create ensemble anomaly scores
        ensemble_scores = [
            AnomalyScore(
                value=float(score), method=f"ensemble_{self._aggregation_method.value}"
            )
            for score in aggregated_scores
        ]

        # Create anomaly entities
        anomalies = self._create_ensemble_anomalies(
            dataset, aggregated_labels, ensemble_scores, individual_results
        )

        # Calculate agreement and confidence metrics
        agreement_scores = self._calculate_agreement_scores(all_predictions)
        confidence_scores = self._calculate_confidence_scores(
            all_scores, agreement_scores
        )

        # Create ensemble metadata
        ensemble_metadata = EnsembleMetadata(
            n_detectors=len(individual_results),
            aggregation_method=self._aggregation_method.value,
            individual_scores=all_scores,
            individual_predictions=all_predictions,
            confidence_scores=confidence_scores,
            agreement_scores=agreement_scores,
        )

        return DetectionResult(
            detector_id=hash(self._name),
            dataset_id=dataset.id,
            anomalies=anomalies,
            scores=ensemble_scores,
            labels=aggregated_labels,
            threshold=threshold,
            execution_time_ms=sum(
                r.execution_time_ms for r in individual_results.values()
            ),
            metadata={
                "ensemble_name": self._name,
                "n_detectors": len(individual_results),
                "detector_names": list(individual_results.keys()),
                "aggregation_method": self._aggregation_method.value,
                "mean_agreement": (
                    float(np.mean(agreement_scores)) if agreement_scores else 0.0
                ),
                "mean_confidence": (
                    float(np.mean(confidence_scores)) if confidence_scores else 0.0
                ),
                "individual_results": {
                    name: {
                        "n_anomalies": len(result.anomalies),
                        "execution_time_ms": result.execution_time_ms,
                        "threshold": result.threshold,
                    }
                    for name, result in individual_results.items()
                },
            },
        )

    def _aggregate_scores_with_method(
        self, all_scores: Dict[str, List[float]], weights: Dict[str, float]
    ) -> List[float]:
        """Aggregate scores using the specified method."""
        if not all_scores:
            return []

        # Convert to numpy arrays for easier computation
        score_matrix = np.array([scores for scores in all_scores.values()])
        weight_array = np.array([weights.get(name, 1.0) for name in all_scores.keys()])

        if self._aggregation_method == AggregationMethod.AVERAGE:
            return np.mean(score_matrix, axis=0).tolist()

        elif self._aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            weighted_scores = score_matrix * weight_array.reshape(-1, 1)
            return (np.sum(weighted_scores, axis=0) / np.sum(weight_array)).tolist()

        elif self._aggregation_method == AggregationMethod.MAX:
            return np.max(score_matrix, axis=0).tolist()

        elif self._aggregation_method == AggregationMethod.MIN:
            return np.min(score_matrix, axis=0).tolist()

        elif self._aggregation_method == AggregationMethod.MEDIAN:
            return np.median(score_matrix, axis=0).tolist()

        elif self._aggregation_method == AggregationMethod.ADAPTIVE:
            # Use weighted average with dynamic weights based on agreement
            agreement_weights = self._calculate_dynamic_weights(all_scores)
            combined_weights = weight_array * agreement_weights
            weighted_scores = score_matrix * combined_weights.reshape(-1, 1)
            return (np.sum(weighted_scores, axis=0) / np.sum(combined_weights)).tolist()

        else:
            # Default to weighted average
            weighted_scores = score_matrix * weight_array.reshape(-1, 1)
            return (np.sum(weighted_scores, axis=0) / np.sum(weight_array)).tolist()

    def _aggregate_predictions_with_method(
        self, all_predictions: Dict[str, List[int]], weights: Dict[str, float]
    ) -> List[int]:
        """Aggregate predictions using majority voting with weights."""
        if not all_predictions:
            return []

        prediction_matrix = np.array([preds for preds in all_predictions.values()])
        weight_array = np.array(
            [weights.get(name, 1.0) for name in all_predictions.keys()]
        )

        # Weighted majority voting
        weighted_votes = prediction_matrix * weight_array.reshape(-1, 1)
        vote_sums = np.sum(weighted_votes, axis=0)
        total_weight = np.sum(weight_array)

        # Threshold at 50% of total weight
        return (vote_sums > total_weight / 2).astype(int).tolist()

    def _aggregate_scores(
        self,
        all_scores: Dict[str, List[float]],
        configs: List[Tuple[str, DetectorConfig]],
    ) -> List[float]:
        """Aggregate scores from individual detectors."""
        weights = {
            name: config.weight for name, config in configs if name in all_scores
        }
        return self._aggregate_scores_with_method(all_scores, weights)

    def _calculate_ensemble_threshold(self, scores: List[float]) -> float:
        """Calculate ensemble threshold based on contamination rate."""
        if not scores:
            return 0.5

        threshold_idx = int(len(scores) * (1 - self._contamination_rate.value))
        threshold_idx = max(0, min(threshold_idx, len(scores) - 1))

        sorted_scores = sorted(scores)
        return sorted_scores[threshold_idx]

    def _create_ensemble_anomalies(
        self,
        dataset: Dataset,
        labels: List[int],
        scores: List[AnomalyScore],
        individual_results: Dict[str, DetectionResult],
    ) -> List[Anomaly]:
        """Create ensemble anomaly entities."""
        anomalies = []
        anomaly_indices = [i for i, label in enumerate(labels) if label == 1]

        for idx in anomaly_indices:
            if idx >= len(dataset.data):
                continue

            # Collect individual detector votes for this point
            individual_votes = {}
            for name, result in individual_results.items():
                if idx < len(result.labels):
                    individual_votes[name] = {
                        "prediction": result.labels[idx],
                        "score": (
                            result.scores[idx].value
                            if idx < len(result.scores)
                            else 0.0
                        ),
                    }

            anomaly = Anomaly(
                score=scores[idx],
                data_point=dataset.data.iloc[idx].to_dict(),
                detector_name=self._name,
                metadata={
                    "index": idx,
                    "ensemble_method": self._aggregation_method.value,
                    "individual_votes": individual_votes,
                    "n_detectors_agree": sum(
                        1
                        for votes in individual_votes.values()
                        if votes["prediction"] == 1
                    ),
                    "agreement_ratio": (
                        sum(
                            1
                            for votes in individual_votes.values()
                            if votes["prediction"] == 1
                        )
                        / len(individual_votes)
                        if individual_votes
                        else 0.0
                    ),
                },
            )

            anomalies.append(anomaly)

        return anomalies

    def _calculate_agreement_scores(
        self, all_predictions: Dict[str, List[int]]
    ) -> List[float]:
        """Calculate agreement scores between detectors."""
        if len(all_predictions) < 2:
            return [1.0] * len(next(iter(all_predictions.values())))

        prediction_matrix = np.array([preds for preds in all_predictions.values()])
        n_samples = prediction_matrix.shape[1]

        agreement_scores = []
        for i in range(n_samples):
            sample_predictions = prediction_matrix[:, i]
            # Calculate agreement as the proportion of detectors that agree with the majority
            unique, counts = np.unique(sample_predictions, return_counts=True)
            max_count = np.max(counts)
            agreement = max_count / len(sample_predictions)
            agreement_scores.append(agreement)

        return agreement_scores

    def _calculate_confidence_scores(
        self, all_scores: Dict[str, List[float]], agreement_scores: List[float]
    ) -> List[float]:
        """Calculate confidence scores combining score variance and agreement."""
        if not all_scores:
            return []

        score_matrix = np.array([scores for scores in all_scores.values()])

        # Calculate score variance for each sample
        score_vars = np.var(score_matrix, axis=0)

        # Normalize variance to [0, 1] and invert (lower variance = higher confidence)
        max_var = np.max(score_vars) if np.max(score_vars) > 0 else 1.0
        normalized_vars = score_vars / max_var
        variance_confidence = 1.0 - normalized_vars

        # Combine with agreement scores
        confidence_scores = (variance_confidence + np.array(agreement_scores)) / 2.0

        return confidence_scores.tolist()

    def _calculate_dynamic_weights(
        self, all_scores: Dict[str, List[float]]
    ) -> np.ndarray:
        """Calculate dynamic weights based on score consistency."""
        if len(all_scores) < 2:
            return np.ones(len(all_scores))

        score_matrix = np.array([scores for scores in all_scores.values()])

        # Calculate inverse of variance for each detector across all samples
        detector_vars = np.var(score_matrix, axis=1)

        # Avoid division by zero
        detector_vars = np.where(detector_vars == 0, 1e-8, detector_vars)

        # Inverse variance weighting
        weights = 1.0 / detector_vars

        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)

        return weights
