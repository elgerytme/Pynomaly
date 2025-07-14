"""Use case for detecting anomalies."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pynomaly.application.services.enhanced_algorithm_adapter_registry import (
    EnhancedAlgorithmAdapterRegistry,
)
from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult
from pynomaly.domain.exceptions import DatasetError, DetectorNotFittedError
from pynomaly.domain.services import FeatureValidator
from pynomaly.shared.protocols import DetectorRepositoryProtocol


@dataclass
class DetectAnomaliesRequest:
    """Request for anomaly detection."""

    detector_id: UUID
    dataset: Dataset
    validate_features: bool = True
    save_results: bool = True


@dataclass
class DetectAnomaliesResponse:
    """Response from anomaly detection."""

    result: DetectionResult
    quality_report: dict | None = None
    warnings: list[str] | None = None

    # Backward compatibility properties
    @property
    def anomaly_indices(self) -> list[int]:
        """Get indices of detected anomalies."""
        return [i for i, label in enumerate(self.result.labels) if label == 1]

    @property
    def anomaly_scores(self) -> list[float]:
        """Get anomaly scores."""
        return [score.value for score in self.result.scores]


class DetectAnomaliesUseCase:
    """Use case for detecting anomalies in a dataset."""

    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        feature_validator: FeatureValidator,
        adapter_registry: EnhancedAlgorithmAdapterRegistry | None = None,
    ):
        """Initialize the use case.

        Args:
            detector_repository: Repository for loading detectors
            feature_validator: Service for validating features
            adapter_registry: Registry for algorithm adapters
        """
        self.detector_repository = detector_repository
        self.feature_validator = feature_validator
        self.adapter_registry = adapter_registry or EnhancedAlgorithmAdapterRegistry()

    async def execute(self, request: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
        """Execute anomaly detection.

        Args:
            request: Detection request

        Returns:
            Detection response with results

        Raises:
            DetectorNotFittedError: If detector is not fitted
            DatasetError: If dataset validation fails
        """
        # Load detector
        detector = await self.detector_repository.find_by_id(request.detector_id)
        if detector is None:
            raise ValueError(f"Detector {request.detector_id} not found")

        # Check if detector is fitted
        if not detector.is_fitted:
            raise DetectorNotFittedError(
                detector_name=detector.name, operation="detect"
            )

        warnings = []
        quality_report = None

        # Validate features if requested
        if request.validate_features:
            # Check data quality
            quality_report = self.feature_validator.check_data_quality(request.dataset)

            # Add warnings for quality issues
            if quality_report["quality_score"] < 0.8:
                warnings.append(
                    f"Data quality score is low: {quality_report['quality_score']:.2f}"
                )

            suggestions = self.feature_validator.suggest_preprocessing(quality_report)
            if suggestions:
                warnings.extend(suggestions)

        # Perform detection using adapter registry
        import time

        start_time = time.time()

        try:
            # Get scores and predictions from adapter (now async)
            scores = await self.adapter_registry.score_with_detector(
                detector, request.dataset
            )
            predictions = await self.adapter_registry.predict_with_detector(
                detector, request.dataset
            )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Create anomalies list
            anomalies = []
            for i, (score, pred) in enumerate(zip(scores, predictions, strict=False)):
                if pred == 1:  # Is anomaly
                    anomaly = Anomaly(
                        score=score,
                        data_point=request.dataset.data.iloc[i].to_dict(),
                        detector_name=detector.name,
                    )
                    anomalies.append(anomaly)

            # Calculate threshold (approximate from scores and predictions)
            import numpy as np

            if len(scores) == 0:  # Handle empty scores
                threshold = 0.0
            else:
                score_values = [s.value for s in scores]
                anomaly_score_values = [
                    score_values[i]
                    for i in range(len(predictions))
                    if predictions[i] == 1
                ]

                if anomaly_score_values:
                    threshold = min(anomaly_score_values)
                else:
                    threshold = np.percentile(score_values, 95) if score_values else 0.0

            # Create detection result
            result = DetectionResult(
                detector_id=detector.id,
                dataset_id=request.dataset.id,
                anomalies=anomalies,
                scores=scores,
                labels=predictions,
                threshold=threshold,
                metadata={
                    "execution_time_ms": execution_time_ms,
                    "algorithm": detector.algorithm_name,
                },
            )

        except Exception as e:
            raise DatasetError(
                message=f"Detection failed for dataset '{request.dataset.name}': {str(e)}",
                details={
                    "dataset_name": request.dataset.name,
                    "operation": "detection",
                },
                cause=e,
            ) from e

        # Save results if requested
        if request.save_results:
            # This would typically use a result repository
            # For now, we'll just add metadata
            result.add_metadata("saved", True)

        return DetectAnomaliesResponse(
            result=result,
            quality_report=quality_report,
            warnings=warnings if warnings else None,
        )
