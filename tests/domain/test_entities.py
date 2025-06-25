"""Tests for domain entities."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import InvalidDataError
from pynomaly.domain.value_objects import AnomalyScore


class MockDetector(Detector):
    """Concrete test implementation of Detector."""

    def fit(self, dataset: Dataset) -> None:
        """Dummy fit implementation."""
        self.is_fitted = True
        self.trained_at = datetime.utcnow()

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Dummy detect implementation."""
        from pynomaly.domain.entities.detection_result import DetectionResult

        return DetectionResult(
            detector_id=self.id,
            dataset_id=dataset.id,
            scores=[AnomalyScore(0.5) for _ in range(len(dataset.data))],
            labels=[0 for _ in range(len(dataset.data))],
            threshold=0.5,
        )

    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Dummy score implementation."""
        return [AnomalyScore(0.5) for _ in range(len(dataset.data))]


class TestDetector:
    """Test Detector entity."""

    def test_create_detector(self):
        """Test creating a detector."""
        detector = MockDetector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.1},
        )

        assert detector.name == "Test Detector"
        assert detector.algorithm_name == "IsolationForest"
        assert detector.parameters["contamination"] == 0.1
        assert not detector.is_fitted
        assert isinstance(detector.id, UUID)
        assert isinstance(detector.created_at, datetime)

    def test_detector_validation(self):
        """Test detector validation."""
        with pytest.raises(ValueError):
            MockDetector(name="", algorithm_name="IsolationForest")

        with pytest.raises(ValueError):
            MockDetector(name="Test", algorithm_name="")

    def test_update_parameters(self):
        """Test updating detector parameters."""
        detector = MockDetector(name="Test", algorithm_name="LOF")
        detector.parameters["n_neighbors"] = 20

        assert detector.parameters["n_neighbors"] == 20


class TestDataset:
    """Test Dataset entity."""

    def test_create_dataset(self):
        """Test creating a dataset."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],
                "target": [0, 0, 1, 0, 1],
            }
        )

        dataset = Dataset(name="Test Dataset", data=df, target_column="target")

        assert dataset.name == "Test Dataset"
        assert dataset.n_samples == 5
        assert dataset.n_features == 2
        assert dataset.has_target
        assert dataset.target_column == "target"

    def test_dataset_validation(self):
        """Test dataset validation."""
        with pytest.raises(ValueError):
            Dataset(name="", data=pd.DataFrame())

        with pytest.raises(InvalidDataError):
            Dataset(name="Test", data=pd.DataFrame())  # Empty DataFrame

    def test_dataset_split(self):
        """Test splitting dataset."""
        df = pd.DataFrame({"feature1": range(100), "feature2": range(100, 200)})

        dataset = Dataset(name="Test", data=df)
        train, test = dataset.split(test_size=0.2, random_state=42)

        assert train.n_samples == 80
        assert test.n_samples == 20
        assert train.name == "Test_train"
        assert test.name == "Test_test"

    def test_get_feature_types(self):
        """Test getting feature types."""
        df = pd.DataFrame(
            {
                "numeric1": [1.0, 2.0, 3.0],
                "numeric2": [1, 2, 3],
                "categorical": ["A", "B", "C"],
                "mixed": [1, "2", 3.0],
            }
        )

        dataset = Dataset(name="Test", data=df)

        numeric_features = dataset.get_numeric_features()
        categorical_features = dataset.get_categorical_features()

        assert "numeric1" in numeric_features
        assert "numeric2" in numeric_features
        assert "categorical" in categorical_features
        assert "mixed" in categorical_features


class TestAnomaly:
    """Test Anomaly entity."""

    def test_create_anomaly(self):
        """Test creating an anomaly."""
        score = AnomalyScore(value=0.95)
        anomaly = Anomaly(
            score=score,
            data_point={"feature1": 10.5, "feature2": -3.2},
            detector_name="test_detector",
        )

        assert anomaly.score.value == 0.95
        assert anomaly.data_point == {"feature1": 10.5, "feature2": -3.2}
        assert anomaly.detector_name == "test_detector"
        assert isinstance(anomaly.id, UUID)
        assert isinstance(anomaly.timestamp, datetime)
        assert anomaly.severity == "critical"  # score > 0.9

    def test_anomaly_validation(self):
        """Test anomaly validation."""
        score = AnomalyScore(value=0.5)

        # Test empty detector name
        with pytest.raises(ValueError):
            Anomaly(score=score, data_point={"feature1": 1.0}, detector_name="")

        # Test invalid score type
        with pytest.raises(TypeError):
            Anomaly(
                score=0.5,  # Should be AnomalyScore
                data_point={"feature1": 1.0},
                detector_name="test",
            )

        # Test invalid data_point type
        with pytest.raises(TypeError):
            Anomaly(score=score, data_point="not_a_dict", detector_name="test")


class TestDetectionResult:
    """Test DetectionResult entity."""

    def test_create_detection_result(self):
        """Test creating a detection result."""
        detector_id = UUID("12345678-1234-5678-1234-567812345678")
        dataset_id = UUID("87654321-4321-8765-4321-876543218765")

        # Create sample data
        scores = [
            AnomalyScore(0.9),
            AnomalyScore(0.92),
            AnomalyScore(0.88),
            AnomalyScore(0.95),
        ]
        anomalies = [
            Anomaly(
                score=scores[1],  # Second score for first anomaly
                data_point={"feature1": 1.0},
                detector_name="test_detector",
            ),
            Anomaly(
                score=scores[3],  # Fourth score for second anomaly
                data_point={"feature1": 2.0},
                detector_name="test_detector",
            ),
        ]
        labels = np.array(
            [0, 1, 0, 1]
        )  # Binary classification - 2 anomalies match 2 Anomaly objects

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.85,
            execution_time_ms=250,
        )

        assert result.detector_id == detector_id
        assert result.dataset_id == dataset_id
        assert len(result.anomalies) == 2
        assert len(result.scores) == 4
        assert result.threshold == 0.85
        assert result.execution_time_ms == 250

    def test_detection_result_validation(self):
        """Test detection result validation."""
        detector_id = UUID("12345678-1234-5678-1234-567812345678")
        dataset_id = UUID("87654321-4321-8765-4321-876543218765")

        # Create valid base data
        scores = [AnomalyScore(0.9)]
        anomalies = [
            Anomaly(
                score=scores[0],
                data_point={"feature1": 1.0},
                detector_name="test_detector",
            )
        ]

        # Test mismatched dimensions (scores vs labels)
        with pytest.raises(ValueError):
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=anomalies,
                scores=scores,  # 1 score
                labels=np.array([0, 1]),  # 2 labels - mismatch!
                threshold=0.8,
            )

        # Test invalid label values (not binary)
        with pytest.raises(ValueError):
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=anomalies,
                scores=scores,
                labels=np.array([2]),  # Invalid - not 0 or 1
                threshold=0.8,
            )
