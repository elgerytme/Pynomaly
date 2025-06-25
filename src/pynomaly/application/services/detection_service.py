"""Application service for coordinating detection operations."""

from __future__ import annotations

import asyncio
from uuid import UUID

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.exceptions import DetectorNotFittedError
from pynomaly.domain.services import AnomalyScorer, ThresholdCalculator
from pynomaly.shared.protocols import (
    DetectionResultRepositoryProtocol,
    DetectorProtocol,
    DetectorRepositoryProtocol,
)


class DetectionService:
    """Service for orchestrating anomaly detection workflows."""

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
        self.detector_repository = detector_repository
        self.result_repository = result_repository
        self.anomaly_scorer = anomaly_scorer
        self.threshold_calculator = threshold_calculator

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
            # Handle async repository pattern
            if hasattr(
                self.detector_repository, "find_by_id"
            ) and asyncio.iscoroutinefunction(self.detector_repository.find_by_id):
                detector = await self.detector_repository.find_by_id(detector_id)
            else:
                detector = self.detector_repository.find_by_id(detector_id)

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
                # Handle async repository pattern
                if hasattr(
                    self.result_repository, "save"
                ) and asyncio.iscoroutinefunction(self.result_repository.save):
                    await self.result_repository.save(result)
                else:
                    self.result_repository.save(result)

        return result_dict

    async def _detect_async(
        self, detector: DetectorProtocol, dataset: Dataset
    ) -> DetectionResult:
        """Async wrapper for detection."""
        # In a real implementation, this might use thread pool for CPU-bound work
        return detector.detect(dataset)

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
        # Load detector
        if hasattr(
            self.detector_repository, "find_by_id"
        ) and asyncio.iscoroutinefunction(self.detector_repository.find_by_id):
            detector = await self.detector_repository.find_by_id(detector_id)
        else:
            detector = self.detector_repository.find_by_id(detector_id)
        if detector is None:
            raise ValueError(f"Detector {detector_id} not found")

        # Get scores
        scores = detector.score(dataset)
        score_values = [s.value for s in scores]

        # Calculate threshold
        threshold_params = threshold_params or {}

        if threshold_method == "percentile":
            percentile = threshold_params.get("percentile", 95)
            threshold = self.threshold_calculator.calculate_by_percentile(
                score_values, percentile
            )
        elif threshold_method == "iqr":
            multiplier = threshold_params.get("multiplier", 1.5)
            threshold = self.threshold_calculator.calculate_by_iqr(
                score_values, multiplier
            )
        elif threshold_method == "mad":
            factor = threshold_params.get("factor", 3.0)
            threshold = self.threshold_calculator.calculate_by_mad(score_values, factor)
        elif threshold_method == "dynamic":
            method = threshold_params.get("method", "knee")
            threshold, _ = self.threshold_calculator.calculate_dynamic_threshold(
                score_values, method=method
            )
        else:
            # Default to contamination rate
            threshold = self.threshold_calculator.calculate_by_contamination(
                score_values, detector.contamination_rate
            )

        # Apply threshold to create labels
        import numpy as np

        labels = np.array([1 if s.value > threshold else 0 for s in scores])

        # Create anomalies list
        from pynomaly.domain.entities import Anomaly

        anomalies = []
        anomaly_indices = np.where(labels == 1)[0]

        for idx in anomaly_indices:
            anomaly = Anomaly(
                score=scores[idx],
                data_point=dataset.data.iloc[idx].to_dict(),
                detector_name=detector.name,
            )
            anomalies.append(anomaly)

        # Create result
        result = DetectionResult(
            detector_id=detector.id,
            dataset_id=dataset.id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=threshold,
            metadata={
                "threshold_method": threshold_method,
                "threshold_params": threshold_params,
            },
        )

        return result

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
        # Load original result
        if hasattr(
            self.result_repository, "find_by_id"
        ) and asyncio.iscoroutinefunction(self.result_repository.find_by_id):
            result = await self.result_repository.find_by_id(result_id)
        else:
            result = self.result_repository.find_by_id(result_id)
        if result is None:
            raise ValueError(f"Result {result_id} not found")

        # Add confidence intervals to scores
        scores_with_ci = self.anomaly_scorer.add_confidence_intervals(
            result.scores, confidence_level=confidence_level, method=method
        )

        # Update result
        result.scores = scores_with_ci

        # Create confidence intervals for the result
        from pynomaly.domain.value_objects import ConfidenceInterval

        confidence_intervals = []

        for score in scores_with_ci:
            if score.is_confident:
                ci = ConfidenceInterval(
                    lower=score.confidence_lower,  # type: ignore
                    upper=score.confidence_upper,  # type: ignore
                    level=confidence_level,
                )
                confidence_intervals.append(ci)

        result.confidence_intervals = confidence_intervals
        result.add_metadata("confidence_level", confidence_level)
        result.add_metadata("confidence_method", method)

        # Save updated result
        if hasattr(self.result_repository, "save") and asyncio.iscoroutinefunction(
            self.result_repository.save
        ):
            await self.result_repository.save(result)
        else:
            self.result_repository.save(result)

        return result

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
            if hasattr(
                self.result_repository, "find_by_detector"
            ) and asyncio.iscoroutinefunction(self.result_repository.find_by_detector):
                results = await self.result_repository.find_by_detector(detector_id)
            else:
                results = self.result_repository.find_by_detector(detector_id)
        elif dataset_id:
            if hasattr(
                self.result_repository, "find_by_dataset"
            ) and asyncio.iscoroutinefunction(self.result_repository.find_by_dataset):
                results = await self.result_repository.find_by_dataset(dataset_id)
            else:
                results = self.result_repository.find_by_dataset(dataset_id)
        else:
            if hasattr(
                self.result_repository, "find_recent"
            ) and asyncio.iscoroutinefunction(self.result_repository.find_recent):
                results = await self.result_repository.find_recent(limit)
            else:
                results = self.result_repository.find_recent(limit)

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
