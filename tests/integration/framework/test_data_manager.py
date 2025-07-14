"""
Integration Test Data Manager

This module provides comprehensive test data management for integration tests,
including realistic datasets, streaming data, and external service mocks.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from faker import Faker

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.infrastructure.config.container import Container

logger = logging.getLogger(__name__)


class IntegrationTestDataManager:
    """Manages test data for integration tests."""

    def __init__(self, container: Container):
        """Initialize test data manager."""
        self.container = container
        self.fake = Faker()
        self.test_data_dir = Path("tests/integration/data")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

        # Data generators
        self.dataset_generator = TestDatasetGenerator(self.fake)
        self.detector_generator = TestDetectorGenerator(self.fake)
        self.streaming_generator = StreamingDataGenerator(self.fake)

        # Storage for created test data
        self.created_datasets: list[Dataset] = []
        self.created_detectors: list[Detector] = []
        self.created_files: list[Path] = []

    async def setup(self) -> None:
        """Set up test data manager."""
        logger.info("Setting up integration test data manager")

        # Pre-generate common test datasets
        await self._generate_common_datasets()

        # Pre-generate common test detectors
        await self._generate_common_detectors()

        logger.info("Test data manager setup complete")

    async def cleanup(self) -> None:
        """Clean up all test data."""
        logger.info("Cleaning up integration test data")

        # Clean up repositories
        await self._cleanup_repositories()

        # Clean up files
        await self._cleanup_files()

        logger.info("Test data cleanup complete")

    async def _generate_common_datasets(self) -> None:
        """Generate common datasets for testing."""
        dataset_configs = [
            {
                "name": "small_normal_dataset",
                "size": 1000,
                "anomaly_rate": 0.05,
                "features": 10,
                "data_type": "tabular",
            },
            {
                "name": "medium_normal_dataset",
                "size": 10000,
                "anomaly_rate": 0.1,
                "features": 20,
                "data_type": "tabular",
            },
            {
                "name": "time_series_dataset",
                "size": 5000,
                "anomaly_rate": 0.08,
                "features": 1,
                "data_type": "time_series",
            },
            {
                "name": "high_dimensional_dataset",
                "size": 2000,
                "anomaly_rate": 0.12,
                "features": 100,
                "data_type": "high_dimensional",
            },
        ]

        for config in dataset_configs:
            dataset = await self.create_dataset(**config)
            self.created_datasets.append(dataset)

    async def _generate_common_detectors(self) -> None:
        """Generate common detectors for testing."""
        detector_configs = [
            {
                "name": "test_isolation_forest",
                "algorithm": "IsolationForest",
                "parameters": {"contamination": 0.1, "random_state": 42},
            },
            {
                "name": "test_local_outlier_factor",
                "algorithm": "LocalOutlierFactor",
                "parameters": {"contamination": 0.1, "n_neighbors": 20},
            },
            {
                "name": "test_one_class_svm",
                "algorithm": "OneClassSVM",
                "parameters": {"nu": 0.1, "gamma": "scale"},
            },
        ]

        for config in detector_configs:
            detector = await self.create_detector(**config)
            self.created_detectors.append(detector)

    async def create_dataset(
        self,
        name: str,
        size: int,
        anomaly_rate: float = 0.1,
        features: int = 10,
        data_type: str = "tabular",
        save_to_file: bool = True,
    ) -> Dataset:
        """Create a test dataset with specified characteristics."""
        logger.info(
            f"Creating dataset: {name} (size={size}, anomaly_rate={anomaly_rate})"
        )

        # Generate data based on type
        if data_type == "tabular":
            data = self.dataset_generator.generate_tabular_data(
                size, features, anomaly_rate
            )
        elif data_type == "time_series":
            data = self.dataset_generator.generate_time_series_data(size, anomaly_rate)
        elif data_type == "high_dimensional":
            data = self.dataset_generator.generate_high_dimensional_data(
                size, features, anomaly_rate
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Save to file if requested
        file_path = None
        if save_to_file:
            file_path = self.test_data_dir / f"{name}.csv"
            data.to_csv(file_path, index=False)
            self.created_files.append(file_path)

        # Create dataset entity
        dataset = Dataset(
            name=name,
            description=f"Test dataset: {data_type} with {features} features",
            file_path=str(file_path) if file_path else None,
            metadata={
                "size": size,
                "features": features,
                "anomaly_rate": anomaly_rate,
                "data_type": data_type,
                "generated_at": datetime.now().isoformat(),
            },
        )

        # Store in repository
        dataset_repo = self.container.dataset_repository()
        await dataset_repo.save(dataset)

        return dataset

    async def create_detector(
        self,
        name: str,
        algorithm: str,
        parameters: dict[str, Any],
        fit_immediately: bool = False,
    ) -> Detector:
        """Create a test detector with specified configuration."""
        logger.info(f"Creating detector: {name} (algorithm={algorithm})")

        # Create detector entity
        detector = Detector(
            name=name,
            algorithm_name=algorithm,
            parameters=parameters,
            description=f"Test detector: {algorithm}",
            metadata={
                "created_for_testing": True,
                "algorithm": algorithm,
                "created_at": datetime.now().isoformat(),
            },
        )

        # Store in repository
        detector_repo = self.container.detector_repository()
        await detector_repo.save(detector)

        # Optionally fit the detector
        if fit_immediately:
            await self._fit_detector(detector)

        return detector

    async def create_detection_result(
        self,
        detector_id: str,
        dataset_id: str,
        anomaly_count: int,
        execution_time_ms: float = 100.0,
    ) -> DetectionResult:
        """Create a test detection result."""
        logger.info(f"Creating detection result for detector {detector_id}")

        # Create result entity
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            n_anomalies=anomaly_count,
            anomaly_rate=anomaly_count / 1000,  # Assume 1000 samples
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now(),
            metadata={"test_result": True, "created_at": datetime.now().isoformat()},
        )

        # Store in repository
        result_repo = self.container.result_repository()
        await result_repo.save(result)

        return result

    async def create_streaming_data_source(
        self,
        name: str,
        event_rate: int = 100,
        duration_seconds: int = 60,
        anomaly_rate: float = 0.1,
    ) -> dict[str, Any]:
        """Create a streaming data source for testing."""
        logger.info(f"Creating streaming data source: {name}")

        # Generate streaming events
        events = self.streaming_generator.generate_streaming_events(
            event_rate, duration_seconds, anomaly_rate
        )

        # Save events to file
        file_path = self.test_data_dir / f"{name}_streaming.json"
        with open(file_path, "w") as f:
            json.dump(events, f, indent=2, default=str)

        self.created_files.append(file_path)

        return {
            "name": name,
            "file_path": str(file_path),
            "event_count": len(events),
            "event_rate": event_rate,
            "duration_seconds": duration_seconds,
            "anomaly_rate": anomaly_rate,
        }

    async def get_test_dataset(self, name: str) -> Dataset | None:
        """Get a test dataset by name."""
        for dataset in self.created_datasets:
            if dataset.name == name:
                return dataset
        return None

    async def get_test_detector(self, name: str) -> Detector | None:
        """Get a test detector by name."""
        for detector in self.created_detectors:
            if detector.name == name:
                return detector
        return None

    async def _fit_detector(self, detector: Detector) -> None:
        """Fit a detector on test data."""
        # This would typically involve training the detector
        # For now, just mark it as fitted
        detector.is_fitted = True
        detector.metadata["fitted_at"] = datetime.now().isoformat()

        # Update in repository
        detector_repo = self.container.detector_repository()
        await detector_repo.save(detector)

    async def _cleanup_repositories(self) -> None:
        """Clean up data from repositories."""
        # Clean up datasets
        dataset_repo = self.container.dataset_repository()
        for dataset in self.created_datasets:
            await dataset_repo.delete(dataset.id)

        # Clean up detectors
        detector_repo = self.container.detector_repository()
        for detector in self.created_detectors:
            await detector_repo.delete(detector.id)

        # Clean up results
        result_repo = self.container.result_repository()
        # Results would be cleaned up automatically when detectors are deleted

    async def _cleanup_files(self) -> None:
        """Clean up test files."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted test file: {file_path}")


class TestDatasetGenerator:
    """Generates realistic test datasets."""

    def __init__(self, fake: Faker):
        """Initialize dataset generator."""
        self.fake = fake

    def generate_tabular_data(
        self, size: int, features: int, anomaly_rate: float
    ) -> pd.DataFrame:
        """Generate tabular dataset with normal and anomalous data."""
        # Calculate number of anomalies
        n_anomalies = int(size * anomaly_rate)
        n_normal = size - n_anomalies

        # Generate normal data
        normal_data = np.random.normal(0, 1, (n_normal, features))

        # Generate anomalous data (shifted and scaled)
        anomalous_data = np.random.normal(3, 2, (n_anomalies, features))

        # Combine data
        data = np.vstack([normal_data, anomalous_data])

        # Create labels
        labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

        # Create DataFrame
        columns = [f"feature_{i}" for i in range(features)]
        df = pd.DataFrame(data, columns=columns)
        df["label"] = labels
        df["timestamp"] = pd.date_range(
            start=datetime.now() - timedelta(days=1), periods=size, freq="1min"
        )

        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)

        return df

    def generate_time_series_data(self, size: int, anomaly_rate: float) -> pd.DataFrame:
        """Generate time series dataset."""
        # Generate time index
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=size), periods=size, freq="1min"
        )

        # Generate base time series with trend and seasonality
        t = np.arange(size)
        trend = 0.001 * t
        seasonal = 0.5 * np.sin(2 * np.pi * t / 60)  # 60-minute cycle
        noise = np.random.normal(0, 0.1, size)

        values = trend + seasonal + noise

        # Add anomalies
        n_anomalies = int(size * anomaly_rate)
        anomaly_indices = np.random.choice(size, n_anomalies, replace=False)

        for idx in anomaly_indices:
            # Create point anomalies
            values[idx] += np.random.choice([-1, 1]) * np.random.uniform(2, 5)

        # Create labels
        labels = np.zeros(size)
        labels[anomaly_indices] = 1

        # Create DataFrame
        df = pd.DataFrame({"timestamp": timestamps, "value": values, "label": labels})

        return df

    def generate_high_dimensional_data(
        self, size: int, features: int, anomaly_rate: float
    ) -> pd.DataFrame:
        """Generate high-dimensional dataset."""
        # Generate normal data with correlation structure
        correlation_matrix = np.random.rand(features, features)
        correlation_matrix = np.dot(correlation_matrix, correlation_matrix.T)

        n_anomalies = int(size * anomaly_rate)
        n_normal = size - n_anomalies

        # Generate normal data
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(features), cov=correlation_matrix, size=n_normal
        )

        # Generate anomalous data (different correlation structure)
        anomalous_correlation = np.eye(features) * 10
        anomalous_data = np.random.multivariate_normal(
            mean=np.ones(features) * 2, cov=anomalous_correlation, size=n_anomalies
        )

        # Combine data
        data = np.vstack([normal_data, anomalous_data])

        # Create labels
        labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

        # Create DataFrame
        columns = [f"dim_{i}" for i in range(features)]
        df = pd.DataFrame(data, columns=columns)
        df["label"] = labels
        df["timestamp"] = pd.date_range(
            start=datetime.now() - timedelta(days=1), periods=size, freq="1min"
        )

        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)

        return df


class TestDetectorGenerator:
    """Generates test detectors with various configurations."""

    def __init__(self, fake: Faker):
        """Initialize detector generator."""
        self.fake = fake

    def generate_detector_config(self, algorithm: str) -> dict[str, Any]:
        """Generate configuration for a specific algorithm."""
        configs = {
            "IsolationForest": {
                "contamination": np.random.uniform(0.05, 0.15),
                "n_estimators": np.random.choice([50, 100, 200]),
                "random_state": 42,
            },
            "LocalOutlierFactor": {
                "contamination": np.random.uniform(0.05, 0.15),
                "n_neighbors": np.random.choice([10, 20, 30]),
                "algorithm": "auto",
            },
            "OneClassSVM": {
                "nu": np.random.uniform(0.05, 0.15),
                "gamma": np.random.choice(["scale", "auto"]),
                "kernel": "rbf",
            },
            "DBSCAN": {
                "eps": np.random.uniform(0.3, 0.7),
                "min_samples": np.random.choice([3, 5, 10]),
            },
        }

        return configs.get(algorithm, {})


class StreamingDataGenerator:
    """Generates streaming data for real-time testing."""

    def __init__(self, fake: Faker):
        """Initialize streaming data generator."""
        self.fake = fake

    def generate_streaming_events(
        self, event_rate: int, duration_seconds: int, anomaly_rate: float
    ) -> list[dict[str, Any]]:
        """Generate streaming events."""
        total_events = event_rate * duration_seconds
        events = []

        start_time = datetime.now()

        for i in range(total_events):
            # Calculate event timestamp
            timestamp = start_time + timedelta(seconds=i / event_rate)

            # Determine if this is an anomaly
            is_anomaly = np.random.random() < anomaly_rate

            # Generate event data
            event = {
                "id": str(uuid4()),
                "timestamp": timestamp.isoformat(),
                "value": self._generate_event_value(is_anomaly),
                "metadata": {
                    "source": "test_generator",
                    "event_type": "anomaly" if is_anomaly else "normal",
                    "sequence_number": i,
                },
                "label": 1 if is_anomaly else 0,
            }

            events.append(event)

        return events

    def _generate_event_value(self, is_anomaly: bool) -> dict[str, Any]:
        """Generate event value based on whether it's an anomaly."""
        if is_anomaly:
            return {
                "sensor_reading": np.random.normal(100, 50),
                "temperature": np.random.normal(80, 20),
                "pressure": np.random.normal(1000, 200),
                "anomaly_score": np.random.uniform(0.8, 1.0),
            }
        else:
            return {
                "sensor_reading": np.random.normal(50, 10),
                "temperature": np.random.normal(25, 5),
                "pressure": np.random.normal(1013, 50),
                "anomaly_score": np.random.uniform(0.0, 0.3),
            }


class ContractTestDataProvider:
    """Provides test data for contract testing."""

    def __init__(self, fake: Faker):
        """Initialize contract test data provider."""
        self.fake = fake

    def get_api_contract_data(self, endpoint: str) -> dict[str, Any]:
        """Get test data for API contract testing."""
        contracts = {
            "/api/v1/detectors": {
                "valid_request": {
                    "name": "test_detector",
                    "algorithm": "IsolationForest",
                    "parameters": {"contamination": 0.1},
                },
                "invalid_request": {
                    "name": "",
                    "algorithm": "InvalidAlgorithm",
                    "parameters": {},
                },
                "expected_response": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "name": "test_detector",
                    "algorithm": "IsolationForest",
                    "created_at": "2023-01-01T00:00:00Z",
                },
            },
            "/api/v1/datasets": {
                "valid_request": {
                    "name": "test_dataset",
                    "file_path": "/path/to/data.csv",
                    "description": "Test dataset",
                },
                "invalid_request": {"name": "", "file_path": "", "description": None},
                "expected_response": {
                    "id": "123e4567-e89b-12d3-a456-426614174001",
                    "name": "test_dataset",
                    "created_at": "2023-01-01T00:00:00Z",
                },
            },
        }

        return contracts.get(endpoint, {})
