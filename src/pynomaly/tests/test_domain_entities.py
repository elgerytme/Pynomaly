#!/usr/bin/env python3
"""
Unit tests for domain entities
"""
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.exceptions import ValidationError
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.contamination_rate import ContaminationRate


class TestDataset:
    """Tests for Dataset entity."""

    def test_create_dataset_from_dataframe(self):
        """Test creating dataset from pandas DataFrame."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })

        dataset = Dataset(name="Test Dataset", data=df)
        assert dataset.name == "Test Dataset"
        assert dataset.data.shape == (5, 3)
        assert dataset.n_samples == 5
        assert dataset.n_features == 3
        assert dataset.id is not None

    def test_dataset_with_metadata(self):
        """Test dataset with metadata."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        metadata = {
            "source": "test_data",
            "created_by": "unit_test",
            "version": "1.0"
        }

        dataset = Dataset(name="Test", data=df, metadata=metadata)
        assert dataset.metadata["source"] == "test_data"
        assert dataset.metadata["version"] == "1.0"

    def test_dataset_feature_types(self):
        """Test dataset feature type detection."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0],
            'integer': [1, 2, 3],
            'categorical': ['A', 'B', 'C']
        })

        dataset = Dataset(name="Mixed Types", data=df)
        numeric_features = dataset.get_numeric_features()
        categorical_features = dataset.get_categorical_features()
        assert len(numeric_features) == 2  # numeric and integer
        assert len(categorical_features) == 1  # categorical

    def test_dataset_basic_statistics(self):
        """Test dataset basic statistics."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        dataset = Dataset(name="Stats Test", data=df)
        summary = dataset.summary()
        assert 'n_samples' in summary
        assert 'n_features' in summary
        assert summary['n_samples'] == 5
        assert summary['n_features'] == 2

    def test_invalid_dataset(self):
        """Test invalid dataset creation."""
        # Empty name
        with pytest.raises(ValueError):
            Dataset(name="", data=pd.DataFrame({'x': [1, 2, 3]}))

        # Empty dataset - should raise InvalidDataError
        from pynomaly.domain.exceptions import InvalidDataError
        with pytest.raises(InvalidDataError):
            Dataset(name="Test", data=pd.DataFrame())


class TestDetector:
    """Tests for Detector entity."""

    def test_create_detector(self):
        """Test creating a detector."""
        detector = Detector(
            name="Test Detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate(value=0.1)
        )

        assert detector.name == "Test Detector"
        assert detector.algorithm_name == "isolation_forest"
        assert detector.contamination_rate.value == 0.1
        assert detector.id is not None

    def test_detector_with_parameters(self):
        """Test detector with algorithm parameters."""
        params = {
            "n_estimators": 100,
            "max_samples": "auto",
            "random_state": 42
        }

        detector = Detector(
            name="Configured Detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate(value=0.05),
            parameters=params
        )

        assert detector.parameters["n_estimators"] == 100
        assert detector.parameters["random_state"] == 42

    def test_detector_validation(self):
        """Test detector validation."""
        # Empty name should raise ValueError
        with pytest.raises(ValueError):
            Detector(
                name="",
                algorithm_name="isolation_forest",
                contamination_rate=ContaminationRate(value=0.1)
            )


class TestAnomaly:
    """Tests for Anomaly entity."""

    def test_create_anomaly(self):
        """Test creating an anomaly."""
        score = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
        data_point = {"feature1": 1.5, "feature2": -2.0}

        anomaly = Anomaly(
            score=score,
            data_point=data_point,
            detector_name="Test Detector"
        )

        assert anomaly.score.value == 0.8
        assert anomaly.data_point["feature1"] == 1.5
        assert anomaly.detector_name == "Test Detector"
        assert anomaly.id is not None
        assert isinstance(anomaly.timestamp, datetime)

    def test_anomaly_with_explanation(self):
        """Test anomaly with explanation."""
        score = AnomalyScore(value=0.9, threshold=0.5, method="pyod")
        data_point = {"x": 100, "y": 200}
        explanation = {
            "feature_contributions": {"x": 0.6, "y": 0.4},
            "method": "shap"
        }

        anomaly = Anomaly(
            score=score,
            data_point=data_point,
            detector_name="Explainable Detector",
            explanation=explanation
        )

        assert anomaly.explanation["method"] == "shap"
        assert anomaly.explanation["feature_contributions"]["x"] == 0.6

    def test_anomaly_severity_classification(self):
        """Test anomaly severity classification."""
        # Critical severity (score > 0.9)
        critical_score = AnomalyScore(value=0.95, threshold=0.5, method="sklearn")
        critical_anomaly = Anomaly(
            score=critical_score,
            data_point={"x": 1},
            detector_name="Test"
        )
        assert critical_anomaly.severity == "critical"

        # High severity (0.7 < score <= 0.9)
        high_score = AnomalyScore(value=0.8, threshold=0.5, method="sklearn")
        high_anomaly = Anomaly(
            score=high_score,
            data_point={"x": 1},
            detector_name="Test"
        )
        assert high_anomaly.severity == "high"

        # Medium severity (0.5 < score <= 0.7)
        med_score = AnomalyScore(value=0.6, threshold=0.5, method="sklearn")
        med_anomaly = Anomaly(
            score=med_score,
            data_point={"x": 1},
            detector_name="Test"
        )
        assert med_anomaly.severity == "medium"

        # Low severity
        low_score = AnomalyScore(value=0.45, threshold=0.5, method="sklearn")
        low_anomaly = Anomaly(
            score=low_score,
            data_point={"x": 1},
            detector_name="Test"
        )
        assert low_anomaly.severity == "low"


class TestDetectionResult:
    """Tests for DetectionResult entity."""

    def test_create_detection_result(self):
        """Test creating a detection result."""
        # Create test scores and anomalies
        scores = [
            AnomalyScore(value=0.2, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.8, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.3, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.9, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.1, threshold=0.5, method="sklearn")
        ]

        anomalies = [
            Anomaly(score=scores[1], data_point={"x": 1}, detector_name="Test"),
            Anomaly(score=scores[3], data_point={"x": 2}, detector_name="Test")
        ]


        result = DetectionResult(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            anomalies=anomalies,
            scores=scores,
            labels=np.array([0, 1, 0, 1, 0]),  # 2 anomalies at indices 1 and 3
            threshold=0.5
        )

        assert result.n_samples == 5
        assert result.n_anomalies == 2
        assert result.anomaly_rate == 0.4  # 2/5 = 40%
        assert result.id is not None

    def test_detection_result_statistics(self):
        """Test detection result statistics."""
        scores = [
            AnomalyScore(value=0.1, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.3, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.7, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.9, threshold=0.5, method="sklearn")
        ]

        anomalies = [
            Anomaly(score=scores[2], data_point={"x": 1}, detector_name="Test"),
            Anomaly(score=scores[3], data_point={"x": 2}, detector_name="Test")
        ]


        result = DetectionResult(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            anomalies=anomalies,
            scores=scores,
            labels=np.array([0, 0, 1, 1]),  # 2 anomalies at indices 2 and 3
            threshold=0.5
        )

        stats = result.score_statistics
        assert 'mean' in stats
        assert 'max' in stats
        assert result.anomaly_rate == 0.5

    def test_detection_result_with_metadata(self):
        """Test detection result with metadata."""
        scores = [
            AnomalyScore(value=0.3, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.4, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.5, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.6, threshold=0.5, method="sklearn"),
            AnomalyScore(value=0.7, threshold=0.5, method="sklearn")
        ]
        anomalies = []

        metadata = {
            "algorithm": "isolation_forest",
            "execution_time": 0.123,
            "contamination": 0.1
        }


        result = DetectionResult(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            anomalies=anomalies,
            scores=scores,
            labels=np.array([0, 0, 0, 0, 0]),  # No anomalies
            threshold=0.5,
            metadata=metadata
        )

        assert result.metadata["algorithm"] == "isolation_forest"
        assert result.metadata["execution_time"] == 0.123

    def test_invalid_detection_result(self):
        """Test invalid detection result."""

        # Empty scores should be invalid
        with pytest.raises(ValidationError):
            DetectionResult(
                detector_id=uuid4(),
                dataset_id=uuid4(),
                anomalies=[],
                scores=[],
                labels=np.array([]),
                threshold=0.5
            )


def test_entity_immutability():
    """Test that entities have proper identity and comparison."""
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

    dataset1 = Dataset(name="Test", data=df)
    dataset2 = Dataset(name="Test", data=df)

    # Different entities even with same data
    assert dataset1.id != dataset2.id
    assert dataset1 != dataset2

    # Same entity should equal itself
    assert dataset1 == dataset1


def test_entity_string_representation():
    """Test entity string representations."""
    df = pd.DataFrame({'x': [1, 2, 3]})
    dataset = Dataset(name="Test Dataset", data=df)

    str_repr = str(dataset)
    assert "Test Dataset" in str_repr
    assert "shape=(3, 1)" in str_repr

    repr_str = repr(dataset)
    assert "Dataset" in repr_str
