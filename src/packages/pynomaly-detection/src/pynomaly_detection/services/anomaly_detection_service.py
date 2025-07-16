"""Anomaly detection service for coordinating detection operations."""

from __future__ import annotations

import logging
from typing import Any

from pynomaly_detection.domain.entities import Dataset, DetectionResult, Detector
from pynomaly_detection.infrastructure.adapters.algorithm_factory import AlgorithmFactory
from pynomaly_detection.infrastructure.persistence.memory_repository import MemoryRepository

logger = logging.getLogger(__name__)


class AnomalyDetectionService:
    """Service for anomaly detection operations."""

    def __init__(
        self,
        dataset_repository: MemoryRepository | None = None,
        detector_repository: MemoryRepository | None = None,
        algorithm_factory: AlgorithmFactory | None = None,
    ):
        """Initialize anomaly detection service.

        Args:
            dataset_repository: Repository for datasets
            detector_repository: Repository for detectors
            algorithm_factory: Factory for creating algorithm adapters
        """
        self.dataset_repository = dataset_repository or MemoryRepository()
        self.detector_repository = detector_repository or MemoryRepository()
        self.algorithm_factory = algorithm_factory or AlgorithmFactory()

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
                adapter = self.algorithm_factory.create_adapter(
                    algorithm_name,
                    parameters or {}
                )
                detector = Detector(
                    name=f"{algorithm_name}_detector",
                    algorithm_name=algorithm_name,
                    hyperparameters=parameters or {},
                )
            else:
                adapter = self.algorithm_factory.create_adapter(
                    detector.algorithm_name,
                    detector.hyperparameters,
                )

            # Fit and detect
            adapter.fit(dataset)
            result = adapter.detect(dataset)

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
        return await self.detector_repository.find_by_id(detector_id)

    async def get_dataset_by_id(self, dataset_id: str) -> Dataset | None:
        """Get dataset by ID.

        Args:
            dataset_id: ID of dataset to retrieve

        Returns:
            Dataset if found, None otherwise
        """
        return await self.dataset_repository.find_by_id(dataset_id)

    async def list_detectors(self) -> list[Detector]:
        """List all available detectors.

        Returns:
            List of all detectors
        """
        return await self.detector_repository.list_all()

    async def list_datasets(self) -> list[Dataset]:
        """List all available datasets.

        Returns:
            List of all datasets
        """
        return await self.dataset_repository.list_all()
