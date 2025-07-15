"""Detection engine for core detection logic and result processing."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.services import AnomalyScorer, ThresholdCalculator
from pynomaly.shared.protocols import (
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)


class DetectionEngine:
    """Engine for core detection logic and result processing."""

    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        result_repository: DetectionResultRepositoryProtocol,
        anomaly_scorer: AnomalyScorer,
        threshold_calculator: ThresholdCalculator,
    ):
        """Initialize detection engine.

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
        detector = await self.detector_repository.find_by_id(detector_id)
        if detector is None:
            raise ValueError(f"Detector {detector_id} not found")

        # Get scores
        scores = detector.score(dataset)
        score_values = [s.value for s in scores]

        # Calculate threshold
        threshold = self._calculate_threshold(
            score_values, threshold_method, threshold_params
        )

        # Apply threshold to create labels
        import numpy as np

        labels = np.array([1 if s.value > threshold else 0 for s in scores])

        # Create anomalies list
        anomalies = self._create_anomalies(scores, labels, dataset, detector)

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

    def _calculate_threshold(
        self,
        score_values: list[float],
        threshold_method: str,
        threshold_params: dict | None = None,
    ) -> float:
        """Calculate threshold based on method and parameters."""
        threshold_params = threshold_params or {}

        if threshold_method == "percentile":
            percentile = threshold_params.get("percentile", 95)
            return self.threshold_calculator.calculate_by_percentile(
                score_values, percentile
            )
        elif threshold_method == "iqr":
            multiplier = threshold_params.get("multiplier", 1.5)
            return self.threshold_calculator.calculate_by_iqr(score_values, multiplier)
        elif threshold_method == "mad":
            factor = threshold_params.get("factor", 3.0)
            return self.threshold_calculator.calculate_by_mad(score_values, factor)
        elif threshold_method == "dynamic":
            method = threshold_params.get("method", "knee")
            threshold, _ = self.threshold_calculator.calculate_dynamic_threshold(
                score_values, method=method
            )
            return threshold
        else:
            # Default to contamination rate
            contamination = threshold_params.get("contamination", 0.1)
            return self.threshold_calculator.calculate_by_contamination(
                score_values, contamination
            )

    def _create_anomalies(
        self, scores, labels, dataset: Dataset, detector
    ) -> list[Any]:
        """Create anomalies list from scores and labels."""
        import numpy as np

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

        return anomalies

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
        result = await self.result_repository.find_by_id(result_id)
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
        await self.result_repository.save(result)

        return result
