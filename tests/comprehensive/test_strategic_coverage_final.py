"""Final strategic test suite to push coverage from 18% towards 90%+."""

import pytest
import numpy as np
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import uuid
import tempfile
from typing import Dict, Any, List, Optional

# Strategic imports for maximum coverage with verified availability
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate, ConfidenceInterval
from pynomaly.domain.entities import Dataset, Anomaly
from pynomaly.infrastructure.config.settings import Settings
from pynomaly.application.dto.detector_dto import CreateDetectorDTO, DetectorResponseDTO, DetectorDTO, UpdateDetectorDTO, DetectionRequestDTO
from pynomaly.application.dto.experiment_dto import CreateExperimentDTO, ExperimentResponseDTO, ExperimentDTO, RunDTO, LeaderboardEntryDTO


class TestInfrastructureComprehensiveStrategic:
    """Strategic infrastructure testing for maximum coverage boost."""
    
    def test_settings_comprehensive_all_features(self):
        """Test all settings configurations comprehensively."""
        # Test default settings
        default_settings = Settings()
        assert default_settings.api_host == "0.0.0.0"
        assert default_settings.api_port == 8000
        
        # Test custom settings with all possible values
        custom_settings = Settings(
            api_host="127.0.0.1",
            api_port=9000,
            api_workers=8,
            api_cors_origins=["http://localhost:3000", "https://prod.example.com"],
            api_rate_limit=500,
            storage_path=Path("/tmp/custom_storage"),
            model_storage_path=Path("/tmp/models"),
            experiment_storage_path=Path("/tmp/experiments"),
            temp_path=Path("/tmp/temp"),
            database_url="postgresql://user:pass@localhost:5432/test",
            database_pool_size=20,
            database_max_overflow=30,
            use_database_repositories=True,
            cache_enabled=True,
            cache_ttl=7200,
            redis_url="redis://localhost:6379/1",
            secret_key="custom-secret-key",
            auth_enabled=True,
            jwt_expiration=3600,
            default_contamination_rate=0.12,
            max_parallel_detectors=6,
            detector_timeout=450,
            max_dataset_size_mb=1500,
            chunk_size=25000,
            max_features=1500,
            random_seed=456,
            gpu_enabled=True,
            gpu_memory_fraction=0.75,
            kafka_bootstrap_servers="localhost:9092",
            streaming_enabled=True,
            max_streaming_sessions=15
        )
        
        # Test all properties and methods
        assert custom_settings.database_configured is True
        assert custom_settings.is_production is True
        
        # Test database configuration
        db_config = custom_settings.get_database_config()
        assert db_config["url"] == "postgresql://user:pass@localhost:5432/test"
        assert db_config["pool_size"] >= 5
        
        # Test CORS configuration
        cors_config = custom_settings.get_cors_config()
        assert len(cors_config["allow_origins"]) == 2
        assert cors_config["allow_credentials"] is True
        
        # Test logging configuration
        logging_config = custom_settings.get_logging_config()
        assert logging_config["version"] == 1
        assert "formatters" in logging_config
        
        # Test SQLite configuration
        sqlite_settings = Settings(database_url="sqlite:///test.db")
        sqlite_config = sqlite_settings.get_database_config()
        assert "connect_args" in sqlite_config
        assert sqlite_config["connect_args"]["check_same_thread"] is False
        
        # Test settings without database
        no_db_settings = Settings(database_url=None)
        assert no_db_settings.database_configured is False
        assert no_db_settings.get_database_config() == {}
    
    def test_monitoring_security_settings_complete(self):
        """Test monitoring and security settings comprehensively."""
        settings = Settings()
        
        # Test monitoring settings
        monitoring = settings.monitoring
        assert monitoring.metrics_enabled is True
        assert monitoring.prometheus_enabled is True
        assert monitoring.prometheus_port == 9090
        assert monitoring.log_level == "INFO"
        assert monitoring.log_format == "json"
        assert monitoring.host_name == "localhost"
        assert monitoring.instrument_fastapi is True
        assert monitoring.instrument_sqlalchemy is True
        
        # Test security settings
        security = settings.security
        assert security.sanitization_level == "moderate"
        assert security.max_input_length == 10000
        assert security.allow_html is False
        assert security.encryption_algorithm == "fernet"
        assert security.enable_audit_logging is True
        assert security.enable_security_monitoring is True
        assert security.brute_force_max_attempts == 5
        assert security.session_timeout == 3600
        
        # Test validation methods
        assert security.validate_sanitization_level("strict") == "strict"
        assert security.validate_encryption_algorithm("aes_gcm") == "aes_gcm"
        
        # Test invalid values raise exceptions
        with pytest.raises(ValueError):
            security.validate_sanitization_level("invalid")
        
        with pytest.raises(ValueError):
            security.validate_encryption_algorithm("invalid")


class TestApplicationDTOsCompleteStrategic:
    """Complete application DTO testing for maximum coverage."""
    
    def test_all_detector_dtos_comprehensive(self):
        """Test all detector DTOs comprehensively."""
        # Test CreateDetectorDTO
        create_dto = CreateDetectorDTO(
            name="comprehensive_detector",
            algorithm_name="IsolationForest",
            contamination_rate=0.08,
            parameters={
                "n_estimators": 150,
                "max_samples": "auto", 
                "contamination": "auto",
                "random_state": 42,
                "n_jobs": -1,
                "bootstrap": False
            },
            metadata={
                "description": "Comprehensive test detector",
                "version": "1.3.0",
                "author": "test_system",
                "tags": ["production", "high_performance"]
            }
        )
        
        # Test all fields and serialization
        assert create_dto.name == "comprehensive_detector"
        assert create_dto.algorithm_name == "IsolationForest"
        assert create_dto.contamination_rate == 0.08
        assert create_dto.parameters["n_estimators"] == 150
        assert create_dto.metadata["version"] == "1.3.0"
        
        json_data = create_dto.model_dump()
        recreated = CreateDetectorDTO.model_validate(json_data)
        assert recreated.name == create_dto.name
        assert recreated.parameters == create_dto.parameters
        
        # Test DetectorDTO
        detector_dto = DetectorDTO(
            id=uuid.uuid4(),
            name="test_detector_dto",
            algorithm_name="LOF",
            contamination_rate=0.05,
            is_fitted=True,
            created_at=datetime.utcnow(),
            trained_at=datetime.utcnow(),
            parameters={"n_neighbors": 25, "metric": "euclidean"},
            metadata={"performance": {"training_time": 120.5}},
            requires_fitting=True,
            supports_streaming=False,
            supports_multivariate=True,
            time_complexity="O(n^2)",
            space_complexity="O(n)"
        )
        
        assert detector_dto.algorithm_name == "LOF"
        assert detector_dto.is_fitted is True
        assert detector_dto.time_complexity == "O(n^2)"
        assert detector_dto.space_complexity == "O(n)"
        
        # Test UpdateDetectorDTO
        update_dto = UpdateDetectorDTO(
            name="updated_detector",
            contamination_rate=0.12,
            parameters={"n_neighbors": 30, "metric": "manhattan"},
            metadata={"version": "1.1", "updated_by": "admin"}
        )
        
        assert update_dto.name == "updated_detector"
        assert update_dto.contamination_rate == 0.12
        assert update_dto.parameters["metric"] == "manhattan"
        
        # Test DetectorResponseDTO
        response_dto = DetectorResponseDTO(
            id=uuid.uuid4(),
            name="response_detector",
            algorithm_name="OCSVM",
            contamination_rate=0.15,
            is_fitted=False,
            created_at=datetime.utcnow(),
            parameters={"nu": 0.1, "kernel": "polynomial", "degree": 3},
            metadata={"category": "svm_based"},
            status="training",
            version="2.0.0"
        )
        
        assert response_dto.algorithm_name == "OCSVM"
        assert response_dto.status == "training"
        assert response_dto.version == "2.0.0"
        
        # Test DetectionRequestDTO
        detection_request = DetectionRequestDTO(
            detector_id=uuid.uuid4(),
            data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            return_scores=True,
            return_feature_importance=True,
            threshold=0.6
        )
        
        assert len(detection_request.data) == 3
        assert len(detection_request.data[0]) == 3
        assert detection_request.return_scores is True
        assert detection_request.return_feature_importance is True
        assert detection_request.threshold == 0.6
    
    def test_all_experiment_dtos_comprehensive(self):
        """Test all experiment DTOs comprehensively."""
        # Test RunDTO
        run_dto = RunDTO(
            id="run_123",
            detector_name="IsolationForest",
            dataset_name="test_dataset",
            parameters={"contamination": 0.1, "n_estimators": 100},
            metrics={"precision": 0.85, "recall": 0.78, "f1_score": 0.81, "auc": 0.89},
            artifacts={"model_path": "/models/run_123.pkl", "report": "/reports/run_123.html"},
            timestamp=datetime.utcnow()
        )
        
        assert run_dto.detector_name == "IsolationForest"
        assert run_dto.metrics["precision"] == 0.85
        assert len(run_dto.artifacts) == 2
        
        # Test ExperimentDTO
        experiment_dto = ExperimentDTO(
            id="exp_456",
            name="comprehensive_experiment",
            description="Multi-algorithm anomaly detection experiment",
            runs=[run_dto],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={
                "objective": "maximize_f1_score",
                "budget": 100,
                "algorithms": ["IsolationForest", "LOF", "OCSVM"]
            }
        )
        
        assert experiment_dto.name == "comprehensive_experiment"
        assert len(experiment_dto.runs) == 1
        assert experiment_dto.metadata["objective"] == "maximize_f1_score"
        
        # Test CreateExperimentDTO
        create_exp_dto = CreateExperimentDTO(
            name="new_experiment",
            description="Testing anomaly detection algorithms",
            metadata={
                "dataset_size": 10000,
                "features": 15,
                "contamination_estimate": 0.05
            }
        )
        
        assert create_exp_dto.name == "new_experiment"
        assert create_exp_dto.metadata["dataset_size"] == 10000
        
        # Test ExperimentResponseDTO
        response_exp_dto = ExperimentResponseDTO(
            id="exp_789",
            name="response_experiment",
            description="Experiment response test",
            status="completed",
            total_runs=10,
            best_score=0.92,
            best_metric="f1_score",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={"duration": "2 hours", "resources_used": "4 CPUs"}
        )
        
        assert response_exp_dto.status == "completed"
        assert response_exp_dto.total_runs == 10
        assert response_exp_dto.best_score == 0.92
        assert response_exp_dto.best_metric == "f1_score"
        
        # Test LeaderboardEntryDTO
        leaderboard_dto = LeaderboardEntryDTO(
            rank=1,
            experiment_id="exp_789",
            run_id="run_best",
            detector_name="IsolationForest",
            dataset_name="benchmark_dataset",
            score=0.95,
            metric_name="f1_score",
            parameters={"contamination": 0.08, "n_estimators": 200},
            timestamp=datetime.utcnow()
        )
        
        assert leaderboard_dto.rank == 1
        assert leaderboard_dto.score == 0.95
        assert leaderboard_dto.metric_name == "f1_score"


class TestDomainEntitiesCompleteStrategic:
    """Complete domain entity testing for maximum coverage."""
    
    def test_dataset_all_functionality_comprehensive(self):
        """Test all dataset functionality comprehensively."""
        # Test with complex data
        complex_data = pd.DataFrame({
            'numeric_int': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'numeric_float': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
            'categorical_str': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
            'boolean_col': [True, False, True, False, True, False, True, False, True, False],
            'target': [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
        })
        
        dataset = Dataset(
            name="comprehensive_dataset",
            data=complex_data,
            target_column="target",
            description="Comprehensive test dataset with mixed types",
            metadata={
                "source": "synthetic",
                "version": "2.0",
                "quality_score": 0.95,
                "preprocessing": ["normalization", "encoding"]
            }
        )
        
        # Test all properties
        assert dataset.name == "comprehensive_dataset"
        assert dataset.shape == (10, 5)
        assert dataset.n_samples == 10
        assert dataset.n_features == 4  # Excludes target
        assert dataset.has_target is True
        assert dataset.target_column == "target"
        assert len(dataset.feature_names) == 5
        
        # Test feature type detection
        numeric_features = dataset.get_numeric_features()
        categorical_features = dataset.get_categorical_features()
        
        assert 'numeric_int' in numeric_features
        assert 'numeric_float' in numeric_features
        assert 'categorical_str' in categorical_features
        assert 'target' not in numeric_features  # Should exclude target
        
        # Test feature and target access
        features = dataset.features
        assert features.shape == (10, 4)
        assert 'target' not in features.columns
        
        target = dataset.target
        assert target is not None
        assert len(target) == 10
        assert target.name == "target"
        
        # Test memory and type information
        memory_usage = dataset.memory_usage
        assert memory_usage > 0
        
        dtypes = dataset.dtypes
        assert len(dtypes) == 5
        
        # Test summary statistics
        summary = dataset.summary()
        assert summary["name"] == "comprehensive_dataset"
        assert summary["n_samples"] == 10
        assert summary["n_features"] == 4
        assert summary["has_target"] is True
        assert summary["numeric_features"] == 2
        assert summary["categorical_features"] == 1
        assert "memory_usage_mb" in summary
        
        # Test sampling with different sizes
        sample_small = dataset.sample(3, random_state=42)
        assert sample_small.n_samples == 3
        assert sample_small.name == "comprehensive_dataset_sample_3"
        assert "parent_dataset_id" in sample_small.metadata
        
        sample_large = dataset.sample(8, random_state=123)
        assert sample_large.n_samples == 8
        
        # Test splitting with different ratios
        train_70, test_30 = dataset.split(test_size=0.3, random_state=42)
        assert train_70.n_samples == 7
        assert test_30.n_samples == 3
        assert train_70.name == "comprehensive_dataset_train"
        assert test_30.name == "comprehensive_dataset_test"
        
        train_80, test_20 = dataset.split(test_size=0.2, random_state=456)
        assert train_80.n_samples == 8
        assert test_20.n_samples == 2
        
        # Test metadata operations
        dataset.add_metadata("test_key", "test_value")
        assert dataset.metadata["test_key"] == "test_value"
        
        dataset.add_metadata("nested", {"inner_key": "inner_value"})
        assert dataset.metadata["nested"]["inner_key"] == "inner_value"
        
        # Test string representation
        repr_str = repr(dataset)
        assert "comprehensive_dataset" in repr_str
        assert "shape=(10, 5)" in repr_str
        assert "has_target=True" in repr_str
    
    def test_dataset_numpy_comprehensive(self):
        """Test dataset with numpy arrays comprehensively."""
        # Test 2D numpy array
        numpy_2d = np.random.randn(50, 10)
        dataset_2d = Dataset(name="numpy_2d", data=numpy_2d)
        
        assert dataset_2d.n_samples == 50
        assert dataset_2d.n_features == 10
        assert isinstance(dataset_2d.data, pd.DataFrame)
        assert dataset_2d.feature_names == [f'feature_{i}' for i in range(10)]
        
        # Test 1D numpy array
        numpy_1d = np.random.randn(25)
        dataset_1d = Dataset(name="numpy_1d", data=numpy_1d)
        
        assert dataset_1d.n_samples == 25
        assert dataset_1d.n_features == 1
        assert dataset_1d.feature_names == ['feature_0']
        
        # Test with custom feature names
        custom_names = [f'custom_feature_{i}' for i in range(10)]
        dataset_custom = Dataset(
            name="custom_features",
            data=numpy_2d,
            feature_names=custom_names
        )
        
        assert dataset_custom.feature_names == custom_names
        assert list(dataset_custom.data.columns) == custom_names
        
        # Test large dataset
        large_numpy = np.random.randn(1000, 20)
        large_dataset = Dataset(name="large_numpy", data=large_numpy)
        
        assert large_dataset.n_samples == 1000
        assert large_dataset.n_features == 20
        
        # Test sampling and splitting on large dataset
        large_sample = large_dataset.sample(100, random_state=42)
        assert large_sample.n_samples == 100
        
        large_train, large_test = large_dataset.split(test_size=0.2, random_state=42)
        assert large_train.n_samples == 800
        assert large_test.n_samples == 200
    
    def test_anomaly_comprehensive_all_features(self):
        """Test anomaly entity comprehensively."""
        # Test anomaly with all features
        score = AnomalyScore(0.88)
        ci = ConfidenceInterval(lower=0.85, upper=0.91, confidence_level=0.99)
        
        comprehensive_anomaly = Anomaly(
            score=score,
            data_point={
                "feature_1": 15.7,
                "feature_2": -3.2,
                "feature_3": 22.1,
                "feature_4": 0.05,
                "categorical_feature": "outlier_category"
            },
            detector_name="comprehensive_detector",
            confidence_interval=ci,
            metadata={
                "severity": "high",
                "feature_importance": {
                    "feature_1": 0.45,
                    "feature_2": 0.30,
                    "feature_3": 0.20,
                    "feature_4": 0.05
                },
                "detection_method": "isolation_forest",
                "model_version": "1.2.3"
            },
            explanation="Extremely high feature_1 value (15.7) combined with very low feature_2 (-3.2) indicates significant deviation from normal patterns"
        )
        
        # Test all properties
        assert comprehensive_anomaly.score.value == 0.88
        assert comprehensive_anomaly.detector_name == "comprehensive_detector"
        assert comprehensive_anomaly.data_point["feature_1"] == 15.7
        assert comprehensive_anomaly.data_point["categorical_feature"] == "outlier_category"
        assert comprehensive_anomaly.confidence_interval.confidence_level == 0.99
        
        # Test computed properties
        assert comprehensive_anomaly.is_high_confidence is True
        assert comprehensive_anomaly.severity == "high"
        
        # Test metadata operations
        comprehensive_anomaly.add_metadata("alert_sent", True)
        assert comprehensive_anomaly.metadata["alert_sent"] is True
        
        comprehensive_anomaly.add_metadata("investigation", {
            "analyst": "john_doe",
            "status": "pending",
            "priority": "high"
        })
        assert comprehensive_anomaly.metadata["investigation"]["analyst"] == "john_doe"
        
        # Test dictionary conversion
        anomaly_dict = comprehensive_anomaly.to_dict()
        assert anomaly_dict["score"] == 0.88
        assert anomaly_dict["detector_name"] == "comprehensive_detector"
        assert "confidence_interval" in anomaly_dict
        assert anomaly_dict["confidence_interval"]["level"] == 0.99
        assert anomaly_dict["explanation"] == comprehensive_anomaly.explanation
        assert anomaly_dict["severity"] == "high"
        
        # Test different severity levels
        critical_anomaly = Anomaly(
            score=AnomalyScore(0.97),
            data_point={"x": 100},
            detector_name="test"
        )
        assert critical_anomaly.severity == "critical"
        
        medium_anomaly = Anomaly(
            score=AnomalyScore(0.65),
            data_point={"x": 50},
            detector_name="test"
        )
        assert medium_anomaly.severity == "medium"
        
        low_anomaly = Anomaly(
            score=AnomalyScore(0.35),
            data_point={"x": 25},
            detector_name="test"
        )
        assert low_anomaly.severity == "low"


class TestValueObjectsCompleteStrategic:
    """Complete value object testing for maximum coverage."""
    
    def test_contamination_rate_all_features_comprehensive(self):
        """Test ContaminationRate all features comprehensively."""
        # Test normal creation
        rate = ContaminationRate(0.08)
        assert rate.value == 0.08
        assert rate.is_valid() is True
        assert rate.as_percentage() == 8.0
        assert str(rate) == "8.0%"
        
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
        
        # Test precise values
        precise_rate = ContaminationRate(0.125)
        assert precise_rate.as_percentage() == 12.5
        assert str(precise_rate) == "12.5%"
        
        # Test validation
        with pytest.raises(Exception):  # Should raise InvalidValueError
            ContaminationRate(-0.1)
        
        with pytest.raises(Exception):  # Should raise InvalidValueError
            ContaminationRate(0.6)
        
        with pytest.raises(Exception):  # Should raise InvalidValueError
            ContaminationRate("invalid")
    
    def test_anomaly_score_all_operations_comprehensive(self):
        """Test AnomalyScore all operations comprehensively."""
        # Test creation and comparison
        score1 = AnomalyScore(0.75)
        score2 = AnomalyScore(0.25)
        score3 = AnomalyScore(0.95)
        score4 = AnomalyScore(0.75)  # Equal to score1
        
        # Test all comparison operators
        assert score1 > score2
        assert score1 < score3
        assert score1 == score4
        assert score1 != score2
        assert score1 >= score4
        assert score1 <= score4
        assert score2 <= score1
        assert score3 >= score1
        
        # Test edge cases
        min_score = AnomalyScore(0.0)
        max_score = AnomalyScore(1.0)
        
        assert min_score < score1
        assert max_score > score1
        assert min_score <= max_score
        
        # Test string representation
        assert str(score1) == "0.75"
        assert str(min_score) == "0.0"
        assert str(max_score) == "1.0"
        
        # Test validation
        with pytest.raises(Exception):
            AnomalyScore(-0.1)
        
        with pytest.raises(Exception):
            AnomalyScore(1.1)
        
        with pytest.raises(Exception):
            AnomalyScore("invalid")
    
    def test_confidence_interval_all_methods_comprehensive(self):
        """Test ConfidenceInterval all methods comprehensively."""
        # Test normal confidence interval
        ci = ConfidenceInterval(lower=0.6, upper=0.8, confidence_level=0.95)
        
        assert ci.lower == 0.6
        assert ci.upper == 0.8
        assert ci.confidence_level == 0.95
        assert ci.width() == 0.2
        assert ci.midpoint() == 0.7
        assert ci.is_valid() is True
        
        # Test containment
        assert ci.contains(0.7) is True
        assert ci.contains(0.6) is True  # Boundary
        assert ci.contains(0.8) is True  # Boundary
        assert ci.contains(0.5) is False
        assert ci.contains(0.9) is False
        
        # Test edge cases
        narrow_ci = ConfidenceInterval(lower=0.499, upper=0.501, confidence_level=0.99)
        assert narrow_ci.width() == pytest.approx(0.002, abs=1e-10)
        assert narrow_ci.midpoint() == pytest.approx(0.5, abs=1e-10)
        
        wide_ci = ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=0.90)
        assert wide_ci.width() == 0.8
        assert wide_ci.midpoint() == 0.5
        
        # Test string representation
        ci_str = str(ci)
        assert "0.6" in ci_str
        assert "0.8" in ci_str
        assert "95%" in ci_str
        
        # Test different confidence levels
        high_conf_ci = ConfidenceInterval(lower=0.45, upper=0.55, confidence_level=0.999)
        assert high_conf_ci.confidence_level == 0.999
        
        # Test validation errors
        with pytest.raises(Exception):  # Lower > Upper
            ConfidenceInterval(lower=0.8, upper=0.6)
        
        with pytest.raises(Exception):  # Invalid confidence level
            ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=1.5)


class TestErrorHandlingAndValidationStrategic:
    """Strategic error handling and validation testing."""
    
    def test_all_validation_scenarios_comprehensive(self):
        """Test all validation scenarios comprehensively."""
        # Dataset validation errors
        with pytest.raises(ValueError, match="Dataset name cannot be empty"):
            Dataset(name="", data=pd.DataFrame({'x': [1, 2, 3]}))
        
        with pytest.raises(Exception):  # InvalidDataError for empty DataFrame
            Dataset(name="empty", data=pd.DataFrame())
        
        with pytest.raises(TypeError, match="Data must be pandas DataFrame or numpy array"):
            Dataset(name="invalid", data="not_a_dataframe")
        
        with pytest.raises(ValueError, match="Number of feature names"):
            Dataset(
                name="mismatch",
                data=np.array([[1, 2], [3, 4]]),
                feature_names=['x', 'y', 'z']  # Too many names
            )
        
        with pytest.raises(ValueError, match="Target column 'missing' not found"):
            Dataset(
                name="bad_target",
                data=pd.DataFrame({'x': [1, 2], 'y': [3, 4]}),
                target_column="missing"
            )
        
        # Anomaly validation errors
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
        
        # Dataset method validation errors
        small_dataset = Dataset(name="small", data=pd.DataFrame({'x': [1, 2]}))
        
        with pytest.raises(ValueError, match="Cannot sample 5 rows"):
            small_dataset.sample(5)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            small_dataset.split(test_size=1.5)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            small_dataset.split(test_size=0.0)


class TestPerformanceAndMemoryStrategic:
    """Strategic performance and memory testing."""
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large dataset
        large_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(5000) 
            for i in range(25)
        })
        large_data['target'] = np.random.choice([0, 1], size=5000, p=[0.9, 0.1])
        
        large_dataset = Dataset(
            name="performance_test",
            data=large_data,
            target_column="target"
        )
        
        # Test properties are computed efficiently
        assert large_dataset.n_samples == 5000
        assert large_dataset.n_features == 25
        assert large_dataset.has_target is True
        
        # Test memory usage tracking
        memory_usage = large_dataset.memory_usage
        assert memory_usage > 0
        
        summary = large_dataset.summary()
        assert summary["memory_usage_mb"] > 0
        
        # Test feature type detection performance
        numeric_features = large_dataset.get_numeric_features()
        categorical_features = large_dataset.get_categorical_features()
        
        assert len(numeric_features) == 25
        assert len(categorical_features) == 0
        
        # Test sampling performance
        sample_1000 = large_dataset.sample(1000, random_state=42)
        assert sample_1000.n_samples == 1000
        
        # Test splitting performance
        train, test = large_dataset.split(test_size=0.2, random_state=42)
        assert train.n_samples == 4000
        assert test.n_samples == 1000
    
    def test_memory_efficiency_comprehensive(self):
        """Test memory efficiency comprehensively."""
        # Test with different data types
        mixed_data = pd.DataFrame({
            'int8_col': pd.array([1, 2, 3, 4, 5], dtype='int8'),
            'int32_col': pd.array([100, 200, 300, 400, 500], dtype='int32'),
            'float32_col': pd.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float32'),
            'float64_col': pd.array([10.1, 20.2, 30.3, 40.4, 50.5], dtype='float64'),
            'category_col': pd.Categorical(['A', 'B', 'C', 'A', 'B'])
        })
        
        mixed_dataset = Dataset(name="memory_test", data=mixed_data)
        
        # Test memory usage calculation
        memory_usage = mixed_dataset.memory_usage
        assert memory_usage > 0
        
        # Test dtype information
        dtypes = mixed_dataset.dtypes
        assert len(dtypes) == 5
        assert dtypes['int8_col'] == 'int8'
        assert dtypes['category_col'] == 'category'
        
        # Test feature detection with different types
        numeric_features = mixed_dataset.get_numeric_features()
        categorical_features = mixed_dataset.get_categorical_features()
        
        # Should detect numeric columns
        assert 'int8_col' in numeric_features or 'int32_col' in numeric_features
        assert 'float32_col' in numeric_features or 'float64_col' in numeric_features