"""Detection orchestrator for coordinating detection workflows."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from pynomaly_detection.domain.entities import Dataset, DetectionResult
from pynomaly_detection.domain.exceptions import DetectorNotFittedError
from pynomaly_detection.shared.protocols import (
    DetectionResultRepositoryProtocol,
    DetectorProtocol,
    DetectorRepositoryProtocol,
)

from .detection_engine import DetectionEngine


class DetectionOrchestrator:
    """Orchestrates detection workflows and manages multiple detectors."""

    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        result_repository: DetectionResultRepositoryProtocol,
        detection_engine: DetectionEngine,
    ):
        """Initialize detection orchestrator.

        Args:
            detector_repository: Repository for detectors
            result_repository: Repository for results
            detection_engine: Engine for core detection logic
        """
        self.detector_repository = detector_repository
        self.result_repository = result_repository
        self.detection_engine = detection_engine

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
        # Load all detectors
        detectors = []
        for detector_id in detector_ids:
            detector = await self.detector_repository.find_by_id(detector_id)

            if detector is None:
                raise ValueError(f"Detector {detector_id} not found")
            if not detector.is_fitted:
                raise DetectorNotFittedError(
                    detector_name=detector.name, operation="detect"
                )
            detectors.append(detector)

        # Run detections in parallel
        detection_tasks = [
            self._detect_async(detector, dataset) for detector in detectors
        ]

        results = await asyncio.gather(*detection_tasks)

        # Create result dictionary
        result_dict = dict(zip(detector_ids, results, strict=False))

        # Save results if requested
        if save_results:
            for result in results:
                await self.result_repository.save(result)

        return result_dict

    async def _detect_async(
        self, detector: DetectorProtocol, dataset: Dataset
    ) -> DetectionResult:
        """Async wrapper for detection."""
        # In a real implementation, this might use thread pool for CPU-bound work
        return detector.detect(dataset)

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
        if detector_id:
            results = await self.result_repository.find_by_detector(detector_id)
        elif dataset_id:
            results = await self.result_repository.find_by_dataset(dataset_id)
        else:
            results = await self.result_repository.find_recent(limit)

        # Sort by timestamp and limit
        results = sorted(results, key=lambda r: r.timestamp, reverse=True)[:limit]

        return results

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
        if not dataset.has_target:
            raise ValueError("Dataset must have labels for comparison")

        # Default metrics
        if metrics is None:
            metrics = ["precision", "recall", "f1", "auc_roc"]

        # Run detections
        results = await self.detect_with_multiple_detectors(
            detector_ids, dataset, save_results=False
        )

        # Calculate metrics for each detector
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        true_labels = dataset.target.values  # type: ignore
        comparison = {"detectors": {}, "summary": {}}

        for detector_id, result in results.items():
            if hasattr(
                self.detector_repository, "find_by_id"
            ) and asyncio.iscoroutinefunction(self.detector_repository.find_by_id):
                detector = await self.detector_repository.find_by_id(detector_id)
            else:
                detector = self.detector_repository.find_by_id(detector_id)
            detector_metrics = {
                "name": detector.name if detector else str(detector_id),
                "n_anomalies": result.n_anomalies,
                "anomaly_rate": result.anomaly_rate,
                "threshold": result.threshold,
            }

            # Calculate requested metrics
            if "precision" in metrics:
                detector_metrics["precision"] = float(
                    precision_score(true_labels, result.labels)
                )

            if "recall" in metrics:
                detector_metrics["recall"] = float(
                    recall_score(true_labels, result.labels)
                )

            if "f1" in metrics:
                detector_metrics["f1"] = float(f1_score(true_labels, result.labels))

            if "auc_roc" in metrics:
                try:
                    score_values = [s.value for s in result.scores]
                    detector_metrics["auc_roc"] = float(
                        roc_auc_score(true_labels, score_values)
                    )
                except ValueError:
                    detector_metrics["auc_roc"] = 0.5

            comparison["detectors"][str(detector_id)] = detector_metrics

        # Summary statistics
        for metric in metrics:
            values = [
                d[metric] for d in comparison["detectors"].values() if metric in d
            ]
            if values:
                import numpy as np

                comparison["summary"][f"{metric}_mean"] = float(np.mean(values))
                comparison["summary"][f"{metric}_std"] = float(np.std(values))
                comparison["summary"][f"{metric}_best"] = float(max(values))

        return comparison
