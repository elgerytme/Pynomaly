"""
Multi-Modal Anomaly Detection Service.

This service orchestrates anomaly detection across multiple data modalities
(text, time-series, tabular, graph) and provides data fusion capabilities.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from pynomaly.domain.entities import DetectionResult
from pynomaly.domain.models.multimodal import (
    FusionStrategy,
    ModalityType,
    MultiModalData,
)
from pynomaly.domain.value_objects import AnomalyScore, AnomalyType
from pynomaly.infrastructure.adapters.pygod_adapter import PyGODDetector
from pynomaly.infrastructure.adapters.text_adapter import TextAnomalyDetector
from pynomaly.infrastructure.adapters.time_series_adapter import TimeSeriesDetector
from pynomaly.shared.protocols.detector_protocol import DetectorProtocol

logger = logging.getLogger(__name__)


class ModalityDetectorConfig(BaseModel):
    """Configuration for a single modality detector."""

    modality_type: ModalityType
    detector_type: str
    detector_config: dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    enabled: bool = Field(default=True)


class MultiModalDetectionConfig(BaseModel):
    """Configuration for multi-modal detection service."""

    fusion_strategy: FusionStrategy = Field(default=FusionStrategy.WEIGHTED_AVERAGE)
    modality_configs: list[ModalityDetectorConfig] = Field(default_factory=list)

    # Fusion parameters
    fusion_weights: dict[str, float] | None = None
    adaptive_weighting: bool = Field(default=True)
    quality_threshold: float = Field(default=0.5)

    # Parallel processing
    max_workers: int = Field(default=4)
    timeout: float | None = None

    # Cross-modal analysis
    enable_cross_modal_analysis: bool = Field(default=False)
    cross_modal_threshold: float = Field(default=0.7)

    # Output configuration
    return_individual_results: bool = Field(default=True)
    return_fusion_details: bool = Field(default=True)


class ModalityDetector:
    """Wrapper for modality-specific detectors."""

    def __init__(self, config: ModalityDetectorConfig):
        self.config = config
        self.detector = self._create_detector()
        self.is_fitted = False
        self.performance_score = 1.0  # Quality metric for adaptive weighting

    def _create_detector(self) -> DetectorProtocol:
        """Create the appropriate detector for this modality."""
        if self.config.modality_type == ModalityType.TEXT:
            return TextAnomalyDetector(**self.config.detector_config)
        elif self.config.modality_type == ModalityType.TIME_SERIES:
            return TimeSeriesDetector(**self.config.detector_config)
        elif self.config.modality_type == ModalityType.GRAPH:
            return PyGODDetector(**self.config.detector_config)
        else:
            raise NotImplementedError(
                f"Detector for modality {self.config.modality_type} not implemented"
            )

    def fit(self, data: Any) -> ModalityDetector:
        """Fit the detector on modality-specific data."""
        try:
            logger.info(f"Fitting {self.config.modality_type} detector")
            self.detector.fit(data)
            self.is_fitted = True
            logger.info(f"{self.config.modality_type} detector fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit {self.config.modality_type} detector: {e}")
            self.performance_score = 0.0
            raise
        return self

    def detect(self, data: Any) -> DetectionResult:
        """Perform anomaly detection on modality-specific data."""
        if not self.is_fitted:
            raise ValueError(
                f"{self.config.modality_type} detector must be fitted before detection"
            )

        try:
            result = self.detector.detect(data)
            # Update performance score based on detection quality
            self._update_performance_score(result)
            return result
        except Exception as e:
            logger.error(f"Detection failed for {self.config.modality_type}: {e}")
            self.performance_score = 0.0
            raise

    def _update_performance_score(self, result: DetectionResult) -> None:
        """Update performance score based on detection results."""
        # Simple quality metric based on anomaly distribution
        if result.anomaly_scores:
            scores = [score.value for score in result.anomaly_scores]
            score_variance = np.var(scores) if len(scores) > 1 else 0.5
            # Higher variance indicates better discrimination
            self.performance_score = min(1.0, score_variance * 2)
        else:
            self.performance_score = 0.5


class FusionEngine:
    """Engine for fusing multi-modal detection results."""

    def __init__(self, config: MultiModalDetectionConfig):
        self.config = config

    def fuse_results(
        self,
        results: dict[ModalityType, DetectionResult],
        detectors: dict[ModalityType, ModalityDetector],
    ) -> DetectionResult:
        """Fuse detection results from multiple modalities."""
        if not results:
            raise ValueError("No results to fuse")

        # Extract scores from all modalities
        all_scores = self._extract_scores(results)

        # Calculate fusion weights
        weights = self._calculate_fusion_weights(detectors)

        # Apply fusion strategy
        fused_scores = self._apply_fusion_strategy(all_scores, weights)

        # Determine anomalies based on fused scores
        anomaly_indices = self._determine_anomalies(fused_scores)

        # Create fused result
        return DetectionResult(
            anomaly_indices=anomaly_indices,
            anomaly_scores=[AnomalyScore(value=score) for score in fused_scores],
            anomaly_type=AnomalyType.MULTIMODAL,
            metadata={
                "fusion_strategy": self.config.fusion_strategy.value,
                "modalities": [modality.value for modality in results.keys()],
                "fusion_weights": weights,
                "individual_results": results
                if self.config.return_individual_results
                else None,
                "num_samples": len(fused_scores),
                "num_anomalies": len(anomaly_indices),
            },
        )

    def _extract_scores(
        self, results: dict[ModalityType, DetectionResult]
    ) -> dict[ModalityType, list[float]]:
        """Extract anomaly scores from detection results."""
        scores = {}
        for modality, result in results.items():
            if result.anomaly_scores:
                scores[modality] = [score.value for score in result.anomaly_scores]
            else:
                # Fallback: create binary scores from anomaly indices
                num_samples = (
                    len(result.anomaly_indices) if result.anomaly_indices else 1
                )
                binary_scores = [0.0] * num_samples
                for idx in result.anomaly_indices:
                    if idx < len(binary_scores):
                        binary_scores[idx] = 1.0
                scores[modality] = binary_scores
        return scores

    def _calculate_fusion_weights(
        self, detectors: dict[ModalityType, ModalityDetector]
    ) -> dict[ModalityType, float]:
        """Calculate weights for fusion based on configuration and performance."""
        weights = {}

        if self.config.fusion_weights:
            # Use configured weights
            for modality, detector in detectors.items():
                weights[modality] = self.config.fusion_weights.get(
                    modality.value, detector.config.weight
                )
        else:
            # Use detector configuration weights
            for modality, detector in detectors.items():
                weights[modality] = detector.config.weight

        # Apply adaptive weighting if enabled
        if self.config.adaptive_weighting:
            for modality, detector in detectors.items():
                performance_factor = detector.performance_score
                weights[modality] *= performance_factor

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {
                modality: weight / total_weight for modality, weight in weights.items()
            }

        return weights

    def _apply_fusion_strategy(
        self,
        all_scores: dict[ModalityType, list[float]],
        weights: dict[ModalityType, float],
    ) -> list[float]:
        """Apply the specified fusion strategy."""
        if self.config.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(all_scores, weights)
        elif self.config.fusion_strategy == FusionStrategy.MAX:
            return self._max_fusion(all_scores)
        elif self.config.fusion_strategy == FusionStrategy.ATTENTION:
            return self._attention_fusion(all_scores, weights)
        elif self.config.fusion_strategy == FusionStrategy.ENSEMBLE:
            return self._ensemble_fusion(all_scores, weights)
        else:
            raise NotImplementedError(
                f"Fusion strategy {self.config.fusion_strategy} not implemented"
            )

    def _weighted_average_fusion(
        self,
        all_scores: dict[ModalityType, list[float]],
        weights: dict[ModalityType, float],
    ) -> list[float]:
        """Weighted average fusion of anomaly scores."""
        if not all_scores:
            return []

        # Determine the number of samples
        num_samples = max(len(scores) for scores in all_scores.values())

        fused_scores = []
        for i in range(num_samples):
            weighted_sum = 0.0
            total_weight = 0.0

            for modality, scores in all_scores.items():
                if i < len(scores):
                    weight = weights.get(modality, 0.0)
                    weighted_sum += scores[i] * weight
                    total_weight += weight

            if total_weight > 0:
                fused_scores.append(weighted_sum / total_weight)
            else:
                fused_scores.append(0.0)

        return fused_scores

    def _max_fusion(self, all_scores: dict[ModalityType, list[float]]) -> list[float]:
        """Maximum score fusion."""
        if not all_scores:
            return []

        num_samples = max(len(scores) for scores in all_scores.values())

        fused_scores = []
        for i in range(num_samples):
            max_score = 0.0
            for scores in all_scores.values():
                if i < len(scores):
                    max_score = max(max_score, scores[i])
            fused_scores.append(max_score)

        return fused_scores

    def _attention_fusion(
        self,
        all_scores: dict[ModalityType, list[float]],
        weights: dict[ModalityType, float],
    ) -> list[float]:
        """Attention-based fusion (simplified version)."""
        # For now, use weighted average with attention-like weighting
        return self._weighted_average_fusion(all_scores, weights)

    def _ensemble_fusion(
        self,
        all_scores: dict[ModalityType, list[float]],
        weights: dict[ModalityType, float],
    ) -> list[float]:
        """Ensemble fusion combining multiple strategies."""
        # Combine weighted average and max fusion
        weighted_scores = self._weighted_average_fusion(all_scores, weights)
        max_scores = self._max_fusion(all_scores)

        # Ensemble combination
        fused_scores = []
        for w_score, m_score in zip(weighted_scores, max_scores, strict=False):
            ensemble_score = 0.7 * w_score + 0.3 * m_score
            fused_scores.append(ensemble_score)

        return fused_scores

    def _determine_anomalies(self, scores: list[float]) -> list[int]:
        """Determine anomaly indices based on fused scores."""
        if not scores:
            return []

        # Use threshold-based approach
        threshold = np.percentile(scores, 90)  # Top 10% as anomalies
        anomaly_indices = [i for i, score in enumerate(scores) if score >= threshold]

        return anomaly_indices


class MultiModalDetectionService:
    """Main service for multi-modal anomaly detection."""

    def __init__(self, config: MultiModalDetectionConfig | None = None):
        self.config = config or MultiModalDetectionConfig()
        self.detectors: dict[ModalityType, ModalityDetector] = {}
        self.fusion_engine = FusionEngine(self.config)
        self.is_fitted = False

    def add_modality(
        self,
        modality_type: ModalityType,
        detector_type: str,
        detector_config: dict[str, Any] | None = None,
        weight: float = 1.0,
    ) -> MultiModalDetectionService:
        """Add a modality detector to the service."""
        config = ModalityDetectorConfig(
            modality_type=modality_type,
            detector_type=detector_type,
            detector_config=detector_config or {},
            weight=weight,
        )

        self.detectors[modality_type] = ModalityDetector(config)
        logger.info(f"Added {modality_type} modality detector")

        return self

    def fit(self, multi_modal_data: MultiModalData) -> MultiModalDetectionService:
        """Fit all modality detectors on multi-modal data."""
        logger.info("Fitting multi-modal detection service")

        if self.config.max_workers > 1:
            # Parallel fitting
            self._fit_parallel(multi_modal_data)
        else:
            # Sequential fitting
            self._fit_sequential(multi_modal_data)

        self.is_fitted = True
        logger.info("Multi-modal detection service fitted successfully")

        return self

    def _fit_parallel(self, multi_modal_data: MultiModalData) -> None:
        """Fit detectors in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}

            for modality_type, detector in self.detectors.items():
                if modality_type in multi_modal_data.modalities:
                    data = multi_modal_data.modalities[modality_type]
                    future = executor.submit(detector.fit, data.data)
                    futures[future] = modality_type

            for future in as_completed(futures, timeout=self.config.timeout):
                modality_type = futures[future]
                try:
                    future.result()
                    logger.info(f"Successfully fitted {modality_type} detector")
                except Exception as e:
                    logger.error(f"Failed to fit {modality_type} detector: {e}")

    def _fit_sequential(self, multi_modal_data: MultiModalData) -> None:
        """Fit detectors sequentially."""
        for modality_type, detector in self.detectors.items():
            if modality_type in multi_modal_data.modalities:
                data = multi_modal_data.modalities[modality_type]
                try:
                    detector.fit(data.data)
                    logger.info(f"Successfully fitted {modality_type} detector")
                except Exception as e:
                    logger.error(f"Failed to fit {modality_type} detector: {e}")

    def detect(self, multi_modal_data: MultiModalData) -> DetectionResult:
        """Perform multi-modal anomaly detection."""
        if not self.is_fitted:
            raise ValueError("Service must be fitted before detection")

        logger.info("Performing multi-modal anomaly detection")

        # Detect anomalies in each modality
        if self.config.max_workers > 1:
            results = self._detect_parallel(multi_modal_data)
        else:
            results = self._detect_sequential(multi_modal_data)

        if not results:
            raise ValueError("No detection results obtained")

        # Fuse results
        fused_result = self.fusion_engine.fuse_results(results, self.detectors)

        logger.info(
            f"Multi-modal detection completed. Found {len(fused_result.anomaly_indices)} anomalies"
        )

        return fused_result

    def _detect_parallel(
        self, multi_modal_data: MultiModalData
    ) -> dict[ModalityType, DetectionResult]:
        """Perform detection in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}

            for modality_type, detector in self.detectors.items():
                if modality_type in multi_modal_data.modalities:
                    data = multi_modal_data.modalities[modality_type]
                    future = executor.submit(detector.detect, data.data)
                    futures[future] = modality_type

            for future in as_completed(futures, timeout=self.config.timeout):
                modality_type = futures[future]
                try:
                    result = future.result()
                    results[modality_type] = result
                    logger.info(f"Successfully detected anomalies in {modality_type}")
                except Exception as e:
                    logger.error(f"Detection failed for {modality_type}: {e}")

        return results

    def _detect_sequential(
        self, multi_modal_data: MultiModalData
    ) -> dict[ModalityType, DetectionResult]:
        """Perform detection sequentially."""
        results = {}

        for modality_type, detector in self.detectors.items():
            if modality_type in multi_modal_data.modalities:
                data = multi_modal_data.modalities[modality_type]
                try:
                    result = detector.detect(data.data)
                    results[modality_type] = result
                    logger.info(f"Successfully detected anomalies in {modality_type}")
                except Exception as e:
                    logger.error(f"Detection failed for {modality_type}: {e}")

        return results


# Factory functions for easy service creation
def create_text_service(**kwargs) -> MultiModalDetectionService:
    """Create a service with text anomaly detection."""
    service = MultiModalDetectionService(**kwargs)
    service.add_modality(ModalityType.TEXT, "text_detector")
    return service


def create_multi_modal_service(
    modalities: list[tuple[ModalityType, str, dict[str, Any]]], **kwargs
) -> MultiModalDetectionService:
    """Create a multi-modal service with specified modalities."""
    service = MultiModalDetectionService(**kwargs)

    for modality_type, detector_type, detector_config in modalities:
        service.add_modality(modality_type, detector_type, detector_config)

    return service
