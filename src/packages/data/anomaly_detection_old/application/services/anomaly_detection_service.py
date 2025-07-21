"""Anomaly detection service for coordinating detection operations."""

from __future__ import annotations

import logging
from typing import Any

from ...domain.entities import Dataset, DetectionResult, Detector
from ..protocols.adapter_protocols import ApplicationAlgorithmFactoryProtocol
from ..protocols.repository_protocols import (
    DatasetRepositoryProtocol,
    DetectorRepositoryProtocol,
)

logger = logging.getLogger(__name__)


class AnomalyDetectionService:
    """Service for anomaly detection operations."""

    def __init__(
        self,
        dataset_repository: DatasetRepositoryProtocol,
        detector_repository: DetectorRepositoryProtocol,
        algorithm_factory: ApplicationAlgorithmFactoryProtocol,
    ):
        """Initialize anomaly detection service.

        Args:
            dataset_repository: Repository for datasets
            detector_repository: Repository for detectors
            algorithm_factory: Factory for creating algorithm adapters
        """
        self.dataset_repository = dataset_repository
        self.detector_repository = detector_repository
        self.algorithm_factory = algorithm_factory

    async def detect_anomalies(
        self,
        dataset: Dataset,
        detector: Detector | None = None,
        algorithm_name: str = "IsolationForest",
        parameters: dict[str, Any] | None = None,
    ) -> DetectionResult:
        """Detect anomalies in dataset.

        Args:
            dataset: Dataset to analyze
            detector: Pre-configured detector (optional)
            algorithm_name: Algorithm to use if no detector provided
            parameters: Algorithm parameters

        Returns:
            Detection result with anomalies and scores
        """
        try:
            # Use provided detector or create new one
            if detector is None:
                from ...domain.protocols.detection_protocols import (
                    DetectionAlgorithm,
                    DetectionConfig,
                )

                # Create config from parameters
                algorithm = DetectionAlgorithm(algorithm_name.lower())
                config = DetectionConfig(
                    algorithm=algorithm,
                    **(parameters or {})
                )
                adapter = self.algorithm_factory.create_algorithm(config)

                detector = Detector(
                    name=f"{algorithm_name}_detector",
                    algorithm_name=algorithm_name,
                    hyperparameters=parameters or {},
                )
            else:
                from ...domain.protocols.detection_protocols import (
                    DetectionAlgorithm,
                    DetectionConfig,
                )

                algorithm = DetectionAlgorithm(detector.algorithm_name.lower())
                config = DetectionConfig(
                    algorithm=algorithm,
                    **detector.hyperparameters
                )
                adapter = self.algorithm_factory.create_algorithm(config)

            # Fit and predict using the adapter
            adapter.fit(dataset.data)
            predictions = adapter.predict(dataset.data)
            scores = adapter.decision_function(dataset.data)
            
            # Create detection result
            from ...domain.value_objects.anomaly_score import AnomalyScore
            from ...domain.value_objects.contamination_rate import ContaminationRate
            from ...domain.value_objects.confidence_interval import ConfidenceInterval
            from ...domain.entities.anomaly import Anomaly
            import numpy as np
            
            # Convert predictions to anomalies
            anomaly_indices = np.where(predictions == 1)[0]
            anomalies = []
            
            for idx in anomaly_indices:
                anomaly = Anomaly(
                    data_point=dataset.data[idx],
                    score=AnomalyScore(scores[idx]),
                    feature_contributions=None,
                    context={"algorithm": algorithm_name, "index": int(idx)}
                )
                anomalies.append(anomaly)
            
            result = DetectionResult(
                dataset_id=dataset.id,
                anomalies=anomalies,
                contamination_rate=ContaminationRate(config.contamination_rate),
                confidence_interval=ConfidenceInterval(
                    lower_bound=float(np.min(scores)),
                    upper_bound=float(np.max(scores)),
                    confidence_level=0.95
                ),
                processing_time_seconds=0.0,  # Would be calculated in real implementation
                algorithm_used=algorithm_name,
                metadata={
                    "total_samples": len(dataset.data),
                    "total_anomalies": len(anomalies),
                    "anomaly_rate": len(anomalies) / len(dataset.data) if len(dataset.data) > 0 else 0.0
                }
            )

            # Store results
            await self.dataset_repository.save(dataset)
            await self.detector_repository.save(detector)

            return result

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise

    async def get_detector_by_id(self, detector_id: str) -> Detector | None:
        """Get detector by ID.

        Args:
            detector_id: ID of detector to retrieve

        Returns:
            Detector if found, None otherwise
        """
        from uuid import UUID
        return await self.detector_repository.find_by_id(UUID(detector_id))

    async def get_dataset_by_id(self, dataset_id: str) -> Dataset | None:
        """Get dataset by ID.

        Args:
            dataset_id: ID of dataset to retrieve

        Returns:
            Dataset if found, None otherwise
        """
        from uuid import UUID
        return await self.dataset_repository.find_by_id(UUID(dataset_id))

    async def list_detectors(self) -> list[Detector]:
        """List all available detectors.

        Returns:
            List of all detectors
        """
        return self.detector_repository.find_all()

    async def list_datasets(self) -> list[Dataset]:
        """List all available datasets.

        Returns:
            List of all datasets
        """
        return self.dataset_repository.find_all()