"""Comprehensive domain entity tests for anomaly detection package."""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import UUID

from anomaly_detection.domain.entities.anomaly import Anomaly, AnomalyType, AnomalySeverity
from anomaly_detection.domain.entities.dataset import Dataset
from anomaly_detection.domain.entities.model import (
    AnomalyModel, ModelStatus, ModelMetrics, ModelConfiguration
)
from anomaly_detection.domain.value_objects.data_point import DataPoint
from anomaly_detection.domain.value_objects.anomaly_score import AnomalyScore


class TestAnomalyEntity:
    """Test cases for Anomaly domain entity."""
    
    def test_anomaly_creation_basic(self):
        """Test basic anomaly creation."""
        anomaly = Anomaly(
            data_point_id="dp_001",
            anomaly_type=AnomalyType.STATISTICAL,
            severity=AnomalySeverity.HIGH,
            score=0.95,
            description="Statistical outlier detected"
        )
        
        assert anomaly.data_point_id == "dp_001"
        assert anomaly.anomaly_type == AnomalyType.STATISTICAL
        assert anomaly.severity == AnomalySeverity.HIGH
        assert anomaly.score == 0.95
        assert anomaly.description == "Statistical outlier detected"
        assert isinstance(anomaly.anomaly_id, str)
        assert isinstance(anomaly.detected_at, datetime)
    
    def test_anomaly_creation_with_context(self):
        """Test anomaly creation with context data."""
        context = {
            "feature_contributions": {"temperature": 0.8, "humidity": 0.2},
            "model_version": "v1.2.3",
            "threshold": 0.9
        }
        
        anomaly = Anomaly(
            data_point_id="dp_002",
            anomaly_type=AnomalyType.MULTIVARIATE,
            severity=AnomalySeverity.CRITICAL,
            score=0.98,
            description="Multi-feature anomaly",
            context=context
        )
        
        assert anomaly.context == context
        assert anomaly.context["feature_contributions"]["temperature"] == 0.8
    
    def test_anomaly_validation(self):
        """Test anomaly validation rules."""
        # Test invalid score range
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            Anomaly(
                data_point_id="dp_003",
                anomaly_type=AnomalyType.STATISTICAL,
                severity=AnomalySeverity.LOW,
                score=1.5
            )
        
        # Test negative score
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            Anomaly(
                data_point_id="dp_004",
                anomaly_type=AnomalyType.STATISTICAL,
                severity=AnomalySeverity.LOW,
                score=-0.1
            )
    
    def test_anomaly_severity_ordering(self):
        """Test anomaly severity comparison."""
        low_anomaly = Anomaly(
            data_point_id="dp_005",
            anomaly_type=AnomalyType.STATISTICAL,
            severity=AnomalySeverity.LOW,
            score=0.3
        )
        
        high_anomaly = Anomaly(
            data_point_id="dp_006",
            anomaly_type=AnomalyType.STATISTICAL,
            severity=AnomalySeverity.HIGH,
            score=0.9
        )
        
        # Test severity values for ordering
        severity_values = {
            AnomalySeverity.LOW: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.CRITICAL: 4
        }
        
        assert severity_values[high_anomaly.severity] > severity_values[low_anomaly.severity]
    
    def test_anomaly_to_dict(self):
        """Test anomaly serialization to dictionary."""
        anomaly = Anomaly(
            data_point_id="dp_007",
            anomaly_type=AnomalyType.TIME_SERIES,
            severity=AnomalySeverity.MEDIUM,
            score=0.75,
            description="Time series anomaly",
            context={"window_size": 10}
        )
        
        anomaly_dict = anomaly.to_dict()
        
        assert anomaly_dict["data_point_id"] == "dp_007"
        assert anomaly_dict["anomaly_type"] == "time_series"
        assert anomaly_dict["severity"] == "medium"
        assert anomaly_dict["score"] == 0.75
        assert anomaly_dict["context"]["window_size"] == 10
        assert "anomaly_id" in anomaly_dict
        assert "detected_at" in anomaly_dict


class TestDatasetEntity:
    """Test cases for Dataset domain entity."""
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        dataset = Dataset(
            name="sensor_data",
            source_path="/data/sensors.csv"
        )
        
        assert dataset.name == "sensor_data"
        assert dataset.source_path == "/data/sensors.csv"
        assert isinstance(dataset.dataset_id, str)
        assert isinstance(dataset.created_at, datetime)
    
    def test_dataset_validation(self):
        """Test dataset validation."""
        # Test empty name
        with pytest.raises(ValueError, match="Dataset name cannot be empty"):
            Dataset(name="", source_path="/data/test.csv")
    
    def test_dataset_metadata(self):
        """Test dataset metadata handling."""
        metadata = {
            "source_system": "IoT sensors",
            "collection_frequency": "1min",
            "data_quality_score": 0.95
        }
        
        dataset = Dataset(
            name="iot_data",
            source_path="/data/iot.csv",
            metadata=metadata
        )
        
        assert dataset.metadata["source_system"] == "IoT sensors"
        assert dataset.metadata["data_quality_score"] == 0.95
    
    def test_dataset_statistics(self):
        """Test dataset statistics methods."""
        dataset = Dataset(name="weather_data", source_path="/data/weather.csv")
        
        # Test basic properties
        assert hasattr(dataset, 'dataset_id')
        assert hasattr(dataset, 'created_at')
        assert hasattr(dataset, 'name')
        assert hasattr(dataset, 'source_path')


class TestAnomalyModelEntity:
    """Test cases for AnomalyModel domain entity."""
    
    def test_model_creation(self):
        """Test basic model creation."""
        config = ModelConfiguration(
            algorithm="isolation_forest",
            hyperparameters={"n_estimators": 100, "contamination": 0.1},
            feature_columns=["temperature", "humidity", "pressure"]
        )
        
        model = AnomalyModel(
            name="sensor_anomaly_detector",
            configuration=config,
            version="1.0.0"
        )
        
        assert model.name == "sensor_anomaly_detector"
        assert model.version == "1.0.0"
        assert model.status == ModelStatus.CREATED
        assert model.configuration.algorithm == "isolation_forest"
        assert len(model.configuration.feature_columns) == 3
        assert isinstance(model.model_id, str)
        assert isinstance(model.created_at, datetime)
    
    def test_model_status_transitions(self):
        """Test valid model status transitions."""
        config = ModelConfiguration(
            algorithm="lof",
            hyperparameters={"n_neighbors": 20},
            feature_columns=["value"]
        )
        
        model = AnomalyModel(name="test_model", configuration=config, version="1.0.0")
        
        # Test valid transitions
        model.start_training()
        assert model.status == ModelStatus.TRAINING
        
        model.complete_training()
        assert model.status == ModelStatus.TRAINED
        
        model.deploy()
        assert model.status == ModelStatus.DEPLOYED
        
        model.retire()
        assert model.status == ModelStatus.RETIRED
    
    def test_model_invalid_transitions(self):
        """Test invalid model status transitions."""
        config = ModelConfiguration(
            algorithm="isolation_forest",
            hyperparameters={},
            feature_columns=["value"]
        )
        
        model = AnomalyModel(name="test_model", configuration=config, version="1.0.0")
        
        # Cannot deploy without training
        with pytest.raises(ValueError, match="Cannot deploy model that is not trained"):
            model.deploy()
        
        # Cannot retire non-deployed model
        with pytest.raises(ValueError, match="Cannot retire model that is not deployed"):
            model.retire()
    
    def test_model_metrics(self):
        """Test model metrics handling."""
        config = ModelConfiguration(
            algorithm="isolation_forest",
            hyperparameters={"n_estimators": 50},
            feature_columns=["temperature", "humidity"]
        )
        
        model = AnomalyModel(name="metrics_test", configuration=config, version="1.0.0")
        
        # Add training metrics
        training_metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            custom_metrics={"auc_roc": 0.94, "training_time": 120.5}
        )
        
        model.add_metrics("training", training_metrics)
        
        assert model.get_metrics("training").accuracy == 0.95
        assert model.get_metrics("training").custom_metrics["auc_roc"] == 0.94
        
        # Test non-existent metrics
        assert model.get_metrics("validation") is None
    
    def test_model_configuration_validation(self):
        """Test model configuration validation."""
        # Test empty feature columns
        with pytest.raises(ValueError, match="Feature columns cannot be empty"):
            ModelConfiguration(
                algorithm="isolation_forest",
                hyperparameters={},
                feature_columns=[]
            )
        
        # Test invalid algorithm
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            ModelConfiguration(
                algorithm="invalid_algorithm",
                hyperparameters={},
                feature_columns=["value"]
            )


class TestDataPointValueObject:
    """Test cases for DataPoint value object."""
    
    def test_data_point_creation(self):
        """Test data point creation."""
        features = {"temperature": 25.5, "humidity": 60.0, "pressure": 1013.25}
        timestamp = datetime.utcnow()
        
        data_point = DataPoint(
            features=features,
            timestamp=timestamp,
            source_id="sensor_001"
        )
        
        assert data_point.features == features
        assert data_point.timestamp == timestamp
        assert data_point.source_id == "sensor_001"
        assert isinstance(data_point.point_id, str)
    
    def test_data_point_feature_access(self):
        """Test data point feature access methods."""
        features = {"temp": 22.0, "humidity": 45.0, "wind": 5.2}
        data_point = DataPoint(features=features, timestamp=datetime.utcnow())
        
        assert data_point.get_feature("temp") == 22.0
        assert data_point.get_feature("humidity") == 45.0
        assert data_point.get_feature("nonexistent") is None
        assert data_point.get_feature("nonexistent", default=0.0) == 0.0
    
    def test_data_point_validation(self):
        """Test data point validation."""
        # Test empty features
        with pytest.raises(ValueError, match="Features cannot be empty"):
            DataPoint(features={}, timestamp=datetime.utcnow())
        
        # Test None features
        with pytest.raises(ValueError, match="Features cannot be None"):
            DataPoint(features=None, timestamp=datetime.utcnow())
    
    def test_data_point_immutability(self):
        """Test that data point is immutable."""
        features = {"temperature": 25.0}
        data_point = DataPoint(features=features, timestamp=datetime.utcnow())
        
        # Modifying original features should not affect data point
        features["temperature"] = 30.0
        assert data_point.get_feature("temperature") == 25.0


class TestAnomalyScoreValueObject:
    """Test cases for AnomalyScore value object."""
    
    def test_anomaly_score_creation(self):
        """Test anomaly score creation."""
        score = AnomalyScore(
            value=0.85,
            confidence=0.92,
            method="isolation_forest"
        )
        
        assert score.value == 0.85
        assert score.confidence == 0.92
        assert score.method == "isolation_forest"
    
    def test_anomaly_score_validation(self):
        """Test anomaly score validation."""
        # Test invalid score value
        with pytest.raises(ValueError, match="Score value must be between 0 and 1"):
            AnomalyScore(value=1.5, confidence=0.9, method="test")
        
        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            AnomalyScore(value=0.5, confidence=1.2, method="test")
    
    def test_anomaly_score_comparison(self):
        """Test anomaly score comparison methods."""
        score1 = AnomalyScore(value=0.8, confidence=0.9, method="method1")
        score2 = AnomalyScore(value=0.6, confidence=0.95, method="method2")
        
        assert score1.is_higher_than(score2)
        assert not score2.is_higher_than(score1)
        assert score1.is_higher_than(0.7)
        assert not score1.is_higher_than(0.9)
    
    def test_anomaly_score_threshold_check(self):
        """Test anomaly score threshold checking."""
        score = AnomalyScore(value=0.75, confidence=0.88, method="test_method")
        
        assert score.exceeds_threshold(0.7)
        assert not score.exceeds_threshold(0.8)
        assert score.exceeds_threshold(0.75)  # Equal case
    
    def test_anomaly_score_categorization(self):
        """Test anomaly score categorization."""
        low_score = AnomalyScore(value=0.2, confidence=0.8, method="test")
        medium_score = AnomalyScore(value=0.6, confidence=0.9, method="test")
        high_score = AnomalyScore(value=0.9, confidence=0.95, method="test")
        
        assert low_score.get_severity_category() == AnomalySeverity.LOW
        assert medium_score.get_severity_category() == AnomalySeverity.MEDIUM
        assert high_score.get_severity_category() == AnomalySeverity.HIGH