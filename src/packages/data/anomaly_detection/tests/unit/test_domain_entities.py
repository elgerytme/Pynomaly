"""Comprehensive domain entity tests for anomaly detection package."""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import UUID

from anomaly_detection.domain.entities.anomaly import Anomaly, AnomalyType, AnomalySeverity
from anomaly_detection.domain.entities.dataset import Dataset
from anomaly_detection.domain.value_objects.algorithm_config import AlgorithmConfig, AlgorithmType
from anomaly_detection.domain.value_objects.detection_metrics import DetectionMetrics


class TestAnomalyEntity:
    """Test cases for Anomaly domain entity."""
    
    def test_anomaly_creation_basic(self):
        """Test basic anomaly creation."""
        anomaly = Anomaly(
            index=0,
            confidence_score=0.95,
            anomaly_type=AnomalyType.POINT,
            severity=AnomalySeverity.HIGH
        )
        
        assert anomaly.index == 0
        assert anomaly.confidence_score == 0.95
        assert anomaly.anomaly_type == AnomalyType.POINT
        assert anomaly.severity == AnomalySeverity.HIGH
        assert anomaly.metadata == {}  # Should be initialized empty
    
    def test_anomaly_creation_with_metadata(self):
        """Test anomaly creation with metadata."""
        import numpy as np
        
        feature_contributions = {"temperature": 0.8, "humidity": 0.2}
        metadata = {
            "model_version": "v1.2.3",
            "threshold": 0.9,
            "processing_time": 0.05
        }
        
        anomaly = Anomaly(
            index=1,
            confidence_score=0.98,
            anomaly_type=AnomalyType.COLLECTIVE,
            severity=AnomalySeverity.CRITICAL,
            feature_values=np.array([25.5, 60.0]),
            feature_contributions=feature_contributions,
            timestamp=datetime.utcnow(),
            metadata=metadata,
            explanation="Multi-feature anomaly detected"
        )
        
        assert anomaly.feature_contributions == feature_contributions
        assert anomaly.metadata == metadata
        assert anomaly.explanation == "Multi-feature anomaly detected"
        assert anomaly.feature_values is not None
        assert len(anomaly.feature_values) == 2
    
    def test_anomaly_validation(self):
        """Test anomaly validation rules."""
        # Test valid confidence score ranges
        valid_anomaly = Anomaly(
            index=2,
            confidence_score=0.5,
            anomaly_type=AnomalyType.CONTEXTUAL,
            severity=AnomalySeverity.MEDIUM
        )
        assert valid_anomaly.confidence_score == 0.5
        
        # Test boundary values
        boundary_low = Anomaly(
            index=3,
            confidence_score=0.0,
            anomaly_type=AnomalyType.POINT
        )
        assert boundary_low.confidence_score == 0.0
        
        boundary_high = Anomaly(
            index=4,
            confidence_score=1.0,
            anomaly_type=AnomalyType.POINT
        )
        assert boundary_high.confidence_score == 1.0
    
    def test_anomaly_severity_ordering(self):
        """Test anomaly severity comparison."""
        low_anomaly = Anomaly(
            index=5,
            confidence_score=0.3,
            anomaly_type=AnomalyType.POINT,
            severity=AnomalySeverity.LOW
        )
        
        high_anomaly = Anomaly(
            index=6,
            confidence_score=0.9,
            anomaly_type=AnomalyType.POINT,
            severity=AnomalySeverity.HIGH
        )
        
        # Test severity enumeration values
        assert low_anomaly.severity == AnomalySeverity.LOW
        assert high_anomaly.severity == AnomalySeverity.HIGH
        assert low_anomaly.severity.value == "low"
        assert high_anomaly.severity.value == "high"
    
    def test_anomaly_metadata_update(self):
        """Test anomaly metadata updates."""
        anomaly = Anomaly(
            index=7,
            confidence_score=0.75,
            anomaly_type=AnomalyType.CONTEXTUAL,
            severity=AnomalySeverity.MEDIUM
        )
        
        # Test initial empty metadata
        assert anomaly.metadata == {}
        
        # Test metadata can be updated (since it's mutable)
        anomaly.metadata["window_size"] = 10
        anomaly.metadata["algorithm"] = "isolation_forest"
        
        assert anomaly.metadata["window_size"] == 10
        assert anomaly.metadata["algorithm"] == "isolation_forest"


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


class TestAlgorithmConfigValueObject:
    """Test cases for AlgorithmConfig value object."""
    
    def test_algorithm_config_creation(self):
        """Test basic algorithm config creation."""
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            contamination=0.1,
            random_state=42,
            hyperparameters={"n_estimators": 100, "max_samples": "auto"}
        )
        
        assert config.algorithm_type == AlgorithmType.ISOLATION_FOREST
        assert config.contamination == 0.1
        assert config.random_state == 42
        assert config.hyperparameters["n_estimators"] == 100
        assert config.is_valid
    
    def test_algorithm_config_validation(self):
        """Test algorithm config validation."""
        # Test invalid contamination - too high
        with pytest.raises(ValueError, match="Contamination must be between 0.0 and 0.5"):
            AlgorithmConfig(
                algorithm_type=AlgorithmType.LOCAL_OUTLIER_FACTOR,
                contamination=0.8
            )
        
        # Test invalid contamination - negative
        with pytest.raises(ValueError, match="Contamination must be between 0.0 and 0.5"):
            AlgorithmConfig(
                algorithm_type=AlgorithmType.LOCAL_OUTLIER_FACTOR,
                contamination=-0.1
            )
        
        # Test valid boundary values
        config_min = AlgorithmConfig(
            algorithm_type=AlgorithmType.ONE_CLASS_SVM,
            contamination=0.0
        )
        assert config_min.is_valid
        
        config_max = AlgorithmConfig(
            algorithm_type=AlgorithmType.ONE_CLASS_SVM,
            contamination=0.5
        )
        assert config_max.is_valid
    
    def test_algorithm_config_sklearn_params(self):
        """Test sklearn parameter generation."""
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            contamination=0.15,
            random_state=123,
            hyperparameters={"n_estimators": 200, "max_features": 1.0}
        )
        
        params = config.get_sklearn_params()
        
        assert params["contamination"] == 0.15
        assert params["random_state"] == 123
        assert params["n_estimators"] == 200
        assert params["max_features"] == 1.0
        
        # Test without random state
        config_no_random = AlgorithmConfig(
            algorithm_type=AlgorithmType.LOCAL_OUTLIER_FACTOR,
            contamination=0.1
        )
        
        params_no_random = config_no_random.get_sklearn_params()
        assert "random_state" not in params_no_random
        assert params_no_random["contamination"] == 0.1
    
class TestDetectionMetricsValueObject:
    """Test cases for DetectionMetrics value object."""
    
    def test_detection_metrics_creation(self):
        """Test detection metrics creation."""
        metrics = DetectionMetrics(
            accuracy=0.95,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            auc_roc=0.94,
            false_positive_rate=0.05,
            false_negative_rate=0.08
        )
        
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.88
        assert metrics.recall == 0.92
        assert metrics.f1_score == 0.90
        assert metrics.auc_roc == 0.94
        assert metrics.false_positive_rate == 0.05
        assert metrics.false_negative_rate == 0.08
    
    def test_detection_metrics_validation(self):
        """Test detection metrics validation."""
        # Test invalid accuracy - too high
        with pytest.raises(ValueError, match="accuracy must be between 0.0 and 1.0"):
            DetectionMetrics(
                accuracy=1.5,
                precision=0.8,
                recall=0.9,
                f1_score=0.85
            )
        
        # Test invalid precision - negative
        with pytest.raises(ValueError, match="precision must be between 0.0 and 1.0"):
            DetectionMetrics(
                accuracy=0.9,
                precision=-0.1,
                recall=0.9,
                f1_score=0.85
            )
        
        # Test invalid AUC-ROC
        with pytest.raises(ValueError, match="AUC-ROC must be between 0.0 and 1.0"):
            DetectionMetrics(
                accuracy=0.9,
                precision=0.8,
                recall=0.9,
                f1_score=0.85,
                auc_roc=1.2
            )


    def test_detection_metrics_performance_assessment(self):
        """Test performance assessment methods."""
        # Good performance metrics
        good_metrics = DetectionMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77
        )
        assert good_metrics.is_good_performance
        
        # Poor performance metrics
        poor_metrics = DetectionMetrics(
            accuracy=0.60,
            precision=0.50,
            recall=0.65,
            f1_score=0.55
        )
        assert not poor_metrics.is_good_performance
    
    def test_detection_metrics_to_dict(self):
        """Test metrics serialization to dictionary."""
        metrics = DetectionMetrics(
            accuracy=0.92,
            precision=0.89,
            recall=0.94,
            f1_score=0.91,
            auc_roc=0.96
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["accuracy"] == 0.92
        assert metrics_dict["precision"] == 0.89
        assert metrics_dict["recall"] == 0.94
        assert metrics_dict["f1_score"] == 0.91
        assert metrics_dict["auc_roc"] == 0.96
        assert metrics_dict["false_positive_rate"] is None
        assert metrics_dict["false_negative_rate"] is None