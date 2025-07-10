"""Application service for coordinating detection operations."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.services import AnomalyScorer, ThresholdCalculator
from pynomaly.shared.protocols import (
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)

from .detection_engine import DetectionEngine
from .detection_orchestrator import DetectionOrchestrator


class DetectionService:
    """Service for coordinating anomaly detection workflows using orchestrator and engine."""

    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        result_repository: DetectionResultRepositoryProtocol,
        anomaly_scorer: AnomalyScorer,
        threshold_calculator: ThresholdCalculator,
    ):
        """Initialize detection service.

        Args:
            detector_repository: Repository for detectors
            result_repository: Repository for results
            anomaly_scorer: Service for score processing
            threshold_calculator: Service for threshold calculation
        """
        self.detection_engine = DetectionEngine(
            detector_repository=detector_repository,
            result_repository=result_repository,
            anomaly_scorer=anomaly_scorer,
            threshold_calculator=threshold_calculator,
        )
        self.detection_orchestrator = DetectionOrchestrator(
            detector_repository=detector_repository,
            result_repository=result_repository,
            detection_engine=self.detection_engine,
        )

    async def detect_with_multiple_detectors(
        self, detector_ids: list[UUID], dataset: Dataset, save_results: bool = True
    ) -> dict[UUID, DetectionResult]:
        """Run detection with multiple detectors in parallel.

        Args:
            detector_ids: List of detector IDs to use
            dataset: Dataset to analyze
            save_results: Whether to persist results

        Returns:
            Dictionary mapping detector IDs to results
        """
        return await self.detection_orchestrator.detect_with_multiple_detectors(
            detector_ids, dataset, save_results
        )

    async def detect_with_custom_threshold(
        self,
        detector_id: UUID,
        dataset: Dataset,
        threshold_method: str = "percentile",
        threshold_params: dict | None = None,
    ) -> DetectionResult:
        """Detect anomalies with custom threshold calculation.

        Args:
            detector_id: Detector to use
            dataset: Dataset to analyze
            threshold_method: Method for threshold calculation
            threshold_params: Parameters for threshold method

        Returns:
            Detection result with custom threshold
        """
        return await self.detection_engine.detect_with_custom_threshold(
            detector_id, dataset, threshold_method, threshold_params
        )

    async def recompute_with_confidence(
        self, result_id: UUID, confidence_level: float = 0.95, method: str = "bootstrap"
    ) -> DetectionResult:
        """Recompute result with confidence intervals.

        Args:
            result_id: Original result ID
            confidence_level: Confidence level for intervals
            method: Method for CI calculation

        Returns:
            Updated result with confidence intervals
        """
        return await self.detection_engine.recompute_with_confidence(
            result_id, confidence_level, method
        )

    async def get_detection_history(
        self,
        detector_id: UUID | None = None,
        dataset_id: UUID | None = None,
        limit: int = 10,
    ) -> list[DetectionResult]:
        """Get historical detection results.

        Args:
            detector_id: Filter by detector
            dataset_id: Filter by dataset
            limit: Maximum results to return

        Returns:
            List of detection results
        """
        return await self.detection_orchestrator.get_detection_history(
            detector_id, dataset_id, limit
        )

    async def compare_detectors(
        self,
        detector_ids: list[UUID],
        dataset: Dataset,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare performance of multiple detectors.

        Args:
            detector_ids: Detectors to compare
            dataset: Dataset with labels for comparison
            metrics: Metrics to calculate

        Returns:
            Comparison results
        """
        return await self.detection_orchestrator.compare_detectors(
            detector_ids, dataset, metrics
        )
