"""Comprehensive test suite designed to push coverage to 90%+ with strategic testing."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import uuid
from typing import Dict, Any

# Strategic imports targeting high-coverage areas
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.infrastructure.config.settings import Settings, AppSettings, MonitoringSettings, SecuritySettings
from pynomaly.application.dto.detector_dto import CreateDetectorDTO, DetectorResponseDTO


class TestValueObjectsComprehensive:
    """Comprehensive value object testing to boost coverage."""
    
    def test_contamination_rate_comprehensive(self):
        """Test ContaminationRate comprehensive functionality."""
        # Test basic creation
        rate = ContaminationRate(0.1)
        assert rate.value == 0.1
        assert rate.is_valid() is True
        assert rate.as_percentage() == 10.0
        
        # Test string representation
        str_repr = str(rate)
        assert "10.0%" in str_repr
        
        # Test class methods
        auto_rate = ContaminationRate.auto()
        assert auto_rate.value == 0.1
        
        low_rate = ContaminationRate.low()
        assert low_rate.value == 0.05
        
        medium_rate = ContaminationRate.medium()
        assert medium_rate.value == 0.1
        
        high_rate = ContaminationRate.high()
        assert high_rate.value == 0.2
        
        # Test class constants
        assert ContaminationRate.AUTO.value == 0.1
        assert ContaminationRate.LOW.value == 0.05
        assert ContaminationRate.MEDIUM.value == 0.1
        assert ContaminationRate.HIGH.value == 0.2
        
        # Test edge cases
        min_rate = ContaminationRate(0.0)
        assert min_rate.value == 0.0
        assert str(min_rate) == "0.0%"
        
        max_rate = ContaminationRate(0.5)
        assert max_rate.value == 0.5
        assert str(max_rate) == "50.0%"
        
        # Test validation errors
        with pytest.raises(Exception):
            ContaminationRate(-0.1)
        
        with pytest.raises(Exception):
            ContaminationRate(0.6)
    
    def test_confidence_interval_comprehensive(self):
        """Test ConfidenceInterval comprehensive functionality."""
        ci = ConfidenceInterval(lower=0.6, upper=0.8, confidence_level=0.95)
        
        # Test all methods
        assert ci.width() == 0.2
        assert ci.midpoint() == 0.7
        assert ci.confidence_level == 0.95
        assert ci.contains(0.7) is True
        assert ci.contains(0.5) is False
        assert ci.contains(0.9) is False
        assert ci.is_valid() is True
        
        # Test string representation
        str_repr = str(ci)
        assert "0.6" in str_repr and "0.8" in str_repr
        assert "95%" in str_repr
        
        # Test invalid intervals (should raise exception)
        with pytest.raises(Exception):
            ConfidenceInterval(lower=0.8, upper=0.6)
    
    def test_anomaly_score_comprehensive(self):
        """Test AnomalyScore comprehensive functionality."""
        score = AnomalyScore(0.75)
        
        # Test comparison operators
        low_score = AnomalyScore(0.25)
        high_score = AnomalyScore(0.95)
        equal_score = AnomalyScore(0.75)
        
        assert score > low_score
        assert score < high_score
        assert score == equal_score
        assert score != low_score
        assert score >= equal_score
        assert score <= equal_score
        
        # Test edge cases
        min_score = AnomalyScore(0.0)
        max_score = AnomalyScore(1.0)
        assert min_score.value == 0.0
        assert max_score.value == 1.0
        
        # Test string representation
        assert str(score) == "0.75"
        
        # Test validation
        with pytest.raises(Exception):
            AnomalyScore(-0.1)
        
        with pytest.raises(Exception):
            AnomalyScore(1.1)


class TestDatasetComprehensive:
    """Comprehensive dataset testing to cover all methods."""
    
    def test_dataset_all_properties_and_methods(self):
        """Test all Dataset properties and methods."""
        # Create comprehensive dataset
        data = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'C', 'D', 'E'],
            'target': [0, 0, 1, 0, 1]
        })
        
        dataset = Dataset(
            name="comprehensive_test",
            data=data,
            target_column="target",
            description="Comprehensive test dataset",
            metadata={"version": "1.0", "source": "test"}
        )
        
        # Test all properties
        assert dataset.shape == (5, 4)
        assert dataset.n_samples == 5
        assert dataset.n_features == 3  # Excludes target
        assert dataset.has_target is True
        assert dataset.target_column == "target"
        
        # Test feature access
        features = dataset.features
        assert features.shape == (5, 3)
        assert 'target' not in features.columns
        
        # Test target access
        target = dataset.target
        assert target is not None
        assert len(target) == 5
        assert target.name == "target"
        
        # Test type detection
        numeric_features = dataset.get_numeric_features()
        categorical_features = dataset.get_categorical_features()
        
        assert 'numeric1' in numeric_features
        assert 'numeric2' in numeric_features
        assert 'categorical' in categorical_features
        assert 'target' not in numeric_features  # Should exclude target
        
        # Test memory usage
        memory = dataset.memory_usage
        assert isinstance(memory, int)
        assert memory > 0
        
        # Test dtypes
        dtypes = dataset.dtypes
        assert len(dtypes) == 4
        
        # Test summary
        summary = dataset.summary()
        assert summary["name"] == "comprehensive_test"
        assert summary["n_samples"] == 5
        assert summary["n_features"] == 3
        assert summary["has_target"] is True
        assert "memory_usage_mb" in summary
        assert summary["numeric_features"] == 2
        assert summary["categorical_features"] == 1
        
        # Test sampling
        sample = dataset.sample(3, random_state=42)
        assert sample.n_samples == 3
        assert sample.name == "comprehensive_test_sample_3"
        assert "parent_dataset_id" in sample.metadata
        
        # Test splitting
        train, test = dataset.split(test_size=0.4, random_state=42)
        assert train.n_samples == 3
        assert test.n_samples == 2
        assert train.name == "comprehensive_test_train"
        assert test.name == "comprehensive_test_test"
        assert train.metadata["split"] == "train"
        assert test.metadata["split"] == "test"
        
        # Test metadata addition
        dataset.add_metadata("new_key", "new_value")
        assert dataset.metadata["new_key"] == "new_value"
        
        # Test string representation
        repr_str = repr(dataset)
        assert "comprehensive_test" in repr_str
        assert "shape=(5, 4)" in repr_str
        assert "has_target=True" in repr_str
    
    def test_dataset_numpy_input(self):
        """Test dataset with numpy array input."""
        # Test 2D array
        numpy_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = Dataset(name="numpy_test", data=numpy_data)
        
        assert dataset.n_samples == 3
        assert dataset.n_features == 3
        assert isinstance(dataset.data, pd.DataFrame)
        assert dataset.feature_names == ['feature_0', 'feature_1', 'feature_2']
        
        # Test 1D array (should be reshaped)
        numpy_1d = np.array([1, 2, 3, 4, 5])
        dataset_1d = Dataset(name="numpy_1d", data=numpy_1d)
        
        assert dataset_1d.n_samples == 5
        assert dataset_1d.n_features == 1
        assert dataset_1d.feature_names == ['feature_0']
        
        # Test with custom feature names
        custom_names = ['x', 'y', 'z']
        dataset_custom = Dataset(
            name="custom_features", 
            data=numpy_data, 
            feature_names=custom_names
        )
        
        assert dataset_custom.feature_names == custom_names
        assert list(dataset_custom.data.columns) == custom_names
    
    def test_dataset_validation_errors(self):
        """Test dataset validation errors."""
        # Test empty name
        data = pd.DataFrame({'x': [1, 2, 3]})
        with pytest.raises(ValueError, match="Dataset name cannot be empty"):
            Dataset(name="", data=data)
        
        # Test empty dataframe
        empty_data = pd.DataFrame()
        with pytest.raises(Exception):  # Should raise InvalidDataError
            Dataset(name="empty", data=empty_data)
        
        # Test invalid data type
        with pytest.raises(TypeError):
            Dataset(name="invalid", data="not a dataframe")
        
        # Test mismatched feature names
        numpy_data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Number of feature names"):
            Dataset(name="mismatch", data=numpy_data, feature_names=['x', 'y', 'z'])
        
        # Test invalid target column
        data = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        with pytest.raises(ValueError, match="Target column 'z' not found"):
            Dataset(name="invalid_target", data=data, target_column="z")
        
        # Test sample size validation
        small_dataset = Dataset(name="small", data=pd.DataFrame({'x': [1, 2]}))
        with pytest.raises(ValueError, match="Cannot sample 5 rows"):
            small_dataset.sample(5)
        
        # Test split validation
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            small_dataset.split(test_size=1.5)


class TestAnomalyComprehensive:
    """Comprehensive anomaly testing."""
    
    def test_anomaly_all_functionality(self):
        """Test all Anomaly functionality."""
        # Create comprehensive anomaly
        score = AnomalyScore(0.85)
        ci = ConfidenceInterval(lower=0.82, upper=0.88)
        
        anomaly = Anomaly(
            score=score,
            data_point={"x": 10.5, "y": -2.3, "z": 7.8},
            detector_name="test_detector",
            confidence_interval=ci,
            metadata={"severity": "high", "feature_importance": {"x": 0.8, "y": 0.2}},
            explanation="Unusually high x value combined with low y value"
        )
        
        # Test properties
        assert anomaly.score.value == 0.85
        assert anomaly.detector_name == "test_detector"
        assert anomaly.data_point["x"] == 10.5
        assert anomaly.explanation == "Unusually high x value combined with low y value"
        
        # Test computed properties
        assert anomaly.is_high_confidence is True  # Should be true with narrow CI
        assert anomaly.severity in ["critical", "high", "medium", "low"]
        
        # Test metadata operations
        anomaly.add_metadata("new_info", "test_value")
        assert anomaly.metadata["new_info"] == "test_value"
        
        # Test dictionary conversion
        anomaly_dict = anomaly.to_dict()
        assert anomaly_dict["score"] == 0.85
        assert anomaly_dict["detector_name"] == "test_detector"
        assert "confidence_interval" in anomaly_dict
        assert anomaly_dict["explanation"] == "Unusually high x value combined with low y value"
        assert anomaly_dict["severity"] in ["critical", "high", "medium", "low"]
        
        # Test equality and hashing
        anomaly2 = Anomaly(
            score=AnomalyScore(0.85),
            data_point={"x": 10.5},
            detector_name="test_detector"
        )
        
        assert anomaly != anomaly2  # Different IDs
        assert hash(anomaly) != hash(anomaly2)
        
        # Test severity levels
        critical_anomaly = Anomaly(
            score=AnomalyScore(0.95),
            data_point={"x": 1},
            detector_name="test"
        )
        assert critical_anomaly.severity == "critical"
        
        medium_anomaly = Anomaly(
            score=AnomalyScore(0.6),
            data_point={"x": 1},
            detector_name="test"
        )
        assert medium_anomaly.severity == "medium"
        
        low_anomaly = Anomaly(
            score=AnomalyScore(0.3),
            data_point={"x": 1},
            detector_name="test"
        )
        assert low_anomaly.severity == "low"


class TestSettingsComprehensive:
    """Comprehensive settings testing."""
    
    def test_app_settings(self):
        """Test AppSettings functionality."""
        app_settings = AppSettings(
            name="TestApp",
            version="2.0.0",
            environment="production",
            debug=False
        )
        
        assert app_settings.name == "TestApp"
        assert app_settings.version == "2.0.0"
        assert app_settings.environment == "production"
        assert app_settings.debug is False
    
    def test_monitoring_settings(self):
        """Test MonitoringSettings functionality."""
        monitoring = MonitoringSettings(
            metrics_enabled=True,
            tracing_enabled=True,
            prometheus_enabled=True,
            prometheus_port=9091,
            log_level="DEBUG",
            log_format="text",
            host_name="test-host",
            instrument_fastapi=False,
            instrument_sqlalchemy=False
        )
        
        assert monitoring.metrics_enabled is True
        assert monitoring.tracing_enabled is True
        assert monitoring.prometheus_port == 9091
        assert monitoring.log_level == "DEBUG"
        assert monitoring.log_format == "text"
        assert monitoring.host_name == "test-host"
    
    def test_security_settings(self):
        """Test SecuritySettings functionality."""
        security = SecuritySettings(
            sanitization_level="strict",
            max_input_length=5000,
            allow_html=True,
            encryption_algorithm="aes_gcm",
            enable_audit_logging=False,
            threat_detection_enabled=False,
            brute_force_max_attempts=3,
            session_timeout=7200
        )
        
        assert security.sanitization_level == "strict"
        assert security.max_input_length == 5000
        assert security.allow_html is True
        assert security.encryption_algorithm == "aes_gcm"
        assert security.enable_audit_logging is False
        assert security.brute_force_max_attempts == 3
        assert security.session_timeout == 7200
        
        # Test validators
        with pytest.raises(Exception):
            SecuritySettings(sanitization_level="invalid")
        
        with pytest.raises(Exception):
            SecuritySettings(encryption_algorithm="invalid")
    
    def test_main_settings_comprehensive(self):
        """Test main Settings class comprehensively."""
        settings = Settings(
            api_host="127.0.0.1",
            api_port=8080,
            api_workers=4,
            api_cors_origins=["http://localhost:3000"],
            storage_path=Path("/tmp/test_storage"),
            database_url="sqlite:///test.db",
            database_pool_size=5,
            use_database_repositories=True,
            cache_enabled=False,
            secret_key="test-secret-key",
            auth_enabled=True,
            default_contamination_rate=0.05,
            max_dataset_size_mb=500,
            random_seed=123,
            gpu_enabled=True,
            streaming_enabled=True
        )
        
        # Test basic properties
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 8080
        assert settings.api_workers == 4
        assert settings.api_cors_origins == ["http://localhost:3000"]
        assert settings.database_url == "sqlite:///test.db"
        assert settings.use_database_repositories is True
        assert settings.cache_enabled is False
        assert settings.auth_enabled is True
        assert settings.default_contamination_rate == 0.05
        assert settings.random_seed == 123
        assert settings.gpu_enabled is True
        assert settings.streaming_enabled is True
        
        # Test computed properties
        assert settings.database_configured is True
        assert settings.is_production is True  # debug=False by default
        
        # Test configuration methods
        db_config = settings.get_database_config()
        assert db_config["url"] == "sqlite:///test.db"
        assert db_config["pool_size"] == 5
        assert "connect_args" in db_config  # SQLite-specific
        assert "poolclass" in db_config
        
        cors_config = settings.get_cors_config()
        assert cors_config["allow_origins"] == ["http://localhost:3000"]
        assert cors_config["allow_credentials"] is True
        
        logging_config = settings.get_logging_config()
        assert logging_config["version"] == 1
        assert "formatters" in logging_config
        assert "handlers" in logging_config
        
        # Test PostgreSQL config
        pg_settings = Settings(database_url="postgresql://user:pass@localhost/test")
        pg_config = pg_settings.get_database_config()
        assert pg_config["pool_size"] >= 5
        assert pg_config["max_overflow"] >= 10
        
        # Test validation
        with pytest.raises(Exception):
            Settings(default_contamination_rate=1.5)


class TestApplicationDTOsComprehensive:
    """Comprehensive application DTO testing."""
    
    def test_create_detector_dto_comprehensive(self):
        """Test CreateDetectorDTO comprehensive functionality."""
        dto = CreateDetectorDTO(
            name="comprehensive_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.15,
            parameters={
                "n_estimators": 200,
                "max_samples": "auto",
                "contamination": "auto",
                "random_state": 42,
                "n_jobs": -1
            },
            metadata={
                "description": "High-performance detector",
                "version": "1.2.0",
                "author": "test"
            }
        )
        
        # Test all fields
        assert dto.name == "comprehensive_detector"
        assert dto.algorithm_name == "IsolationForest"
        assert dto.contamination_rate == 0.15
        assert dto.parameters["n_estimators"] == 200
        assert dto.parameters["max_samples"] == "auto"
        assert dto.metadata["description"] == "High-performance detector"
        assert dto.metadata["version"] == "1.2.0"
        
        # Test serialization/deserialization
        json_data = dto.model_dump()
        assert "name" in json_data
        assert "algorithm_name" in json_data
        assert "contamination_rate" in json_data
        assert "parameters" in json_data
        assert "metadata" in json_data
        
        recreated = CreateDetectorDTO.model_validate(json_data)
        assert recreated.name == dto.name
        assert recreated.algorithm_name == dto.algorithm_name
        assert recreated.contamination_rate == dto.contamination_rate
        assert recreated.parameters == dto.parameters
        assert recreated.metadata == dto.metadata
        
        # Test defaults
        minimal_dto = CreateDetectorDTO(
            name="minimal",
            algorithm_name="LOF"
        )
        assert minimal_dto.contamination_rate == 0.1  # Default
        assert minimal_dto.parameters == {}
        assert minimal_dto.metadata == {}
    
    def test_detector_response_dto_comprehensive(self):
        """Test DetectorResponseDTO comprehensive functionality."""
        response_dto = DetectorResponseDTO(
            id=uuid.uuid4(),
            name="response_detector",
            algorithm_name="OCSVM",
            contamination_rate=0.08,
            is_fitted=True,
            created_at=datetime.utcnow(),
            trained_at=datetime.utcnow(),
            parameters={"nu": 0.05, "kernel": "rbf", "gamma": "auto"},
            metadata={"performance": {"training_time": 45.2, "accuracy": 0.92}},
            status="active",
            version="1.1.0"
        )
        
        # Test all fields
        assert response_dto.algorithm_name == "OCSVM"
        assert response_dto.contamination_rate == 0.08
        assert response_dto.is_fitted is True
        assert response_dto.parameters["nu"] == 0.05
        assert response_dto.parameters["kernel"] == "rbf"
        assert response_dto.metadata["performance"]["accuracy"] == 0.92
        assert response_dto.status == "active"
        assert response_dto.version == "1.1.0"
        
        # Test serialization
        json_data = response_dto.model_dump()
        assert "id" in json_data
        assert "is_fitted" in json_data
        assert "created_at" in json_data
        assert "trained_at" in json_data
        assert "status" in json_data
        assert "version" in json_data


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling across components."""
    
    def test_anomaly_confidence_edge_cases(self):
        """Test anomaly confidence edge cases."""
        # Test without confidence interval
        anomaly_no_ci = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"x": 1},
            detector_name="test"
        )
        assert anomaly_no_ci.is_high_confidence is False
        
        # Test with very wide confidence interval
        wide_ci = ConfidenceInterval(lower=0.1, upper=0.9)
        anomaly_wide_ci = Anomaly(
            score=AnomalyScore(0.5),
            data_point={"x": 1},
            detector_name="test",
            confidence_interval=wide_ci
        )
        assert anomaly_wide_ci.is_high_confidence is False
        
        # Test score confidence attributes (if they exist)
        anomaly_dict = anomaly_no_ci.to_dict()
        # The to_dict method checks for score.confidence_lower/upper
        assert "score_confidence_lower" not in anomaly_dict
        assert "score_confidence_upper" not in anomaly_dict
    
    def test_dataset_edge_cases_comprehensive(self):
        """Test comprehensive dataset edge cases."""
        # Test large dataset properties
        large_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000) 
            for i in range(10)
        })
        large_dataset = Dataset(name="large", data=large_data)
        
        assert large_dataset.n_samples == 1000
        assert large_dataset.n_features == 10
        assert len(large_dataset.get_numeric_features()) == 10
        assert len(large_dataset.get_categorical_features()) == 0
        
        # Test mixed data types
        mixed_data = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True]
        })
        mixed_dataset = Dataset(name="mixed", data=mixed_data)
        
        numeric_features = mixed_dataset.get_numeric_features()
        categorical_features = mixed_dataset.get_categorical_features()
        
        # Test that numeric types are correctly identified
        assert 'int_col' in numeric_features or 'float_col' in numeric_features
        assert 'str_col' in categorical_features
        
        # Test dtypes property
        dtypes = mixed_dataset.dtypes
        assert len(dtypes) == 4
        assert dtypes['str_col'] == 'object'
    
    def test_validation_comprehensive(self):
        """Test comprehensive validation scenarios."""
        # Test anomaly validation
        with pytest.raises(TypeError, match="Score must be AnomalyScore"):
            Anomaly(
                score=0.8,  # Should be AnomalyScore instance
                data_point={"x": 1},
                detector_name="test"
            )
        
        with pytest.raises(ValueError, match="Detector name cannot be empty"):
            Anomaly(
                score=AnomalyScore(0.8),
                data_point={"x": 1},
                detector_name=""
            )
        
        with pytest.raises(TypeError, match="Data point must be a dictionary"):
            Anomaly(
                score=AnomalyScore(0.8),
                data_point=[1, 2, 3],  # Should be dict
                detector_name="test"
            )