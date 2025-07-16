"""Tests for Result DTOs."""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from monorepo.application.dto.result_dto import AnomalyDTO, DetectionResultDTO


class TestAnomalyDTO:
    """Test suite for AnomalyDTO."""

    def test_valid_creation(self):
        """Test creating a valid anomaly DTO."""
        anomaly_id = uuid4()
        timestamp = datetime.now()
        data_point = {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}
        metadata = {"algorithm": "IsolationForest", "confidence": 0.95}

        dto = AnomalyDTO(
            id=anomaly_id,
            score=0.85,
            data_point=data_point,
            detector_name="fraud_detector",
            timestamp=timestamp,
            severity="high",
            explanation="Unusual pattern detected in features 1 and 2",
            metadata=metadata,
            confidence_lower=0.7,
            confidence_upper=0.9,
        )

        assert dto.id == anomaly_id
        assert dto.score == 0.85
        assert dto.data_point == data_point
        assert dto.detector_name == "fraud_detector"
        assert dto.timestamp == timestamp
        assert dto.severity == "high"
        assert dto.explanation == "Unusual pattern detected in features 1 and 2"
        assert dto.metadata == metadata
        assert dto.confidence_lower == 0.7
        assert dto.confidence_upper == 0.9

    def test_default_values(self):
        """Test default values."""
        anomaly_id = uuid4()
        timestamp = datetime.now()
        data_point = {"value": 10.0}

        dto = AnomalyDTO(
            id=anomaly_id,
            score=0.6,
            data_point=data_point,
            detector_name="test_detector",
            timestamp=timestamp,
            severity="medium",
        )

        assert dto.explanation is None
        assert dto.metadata == {}
        assert dto.confidence_lower is None
        assert dto.confidence_upper is None

    def test_score_validation(self):
        """Test score validation."""
        anomaly_id = uuid4()
        timestamp = datetime.now()
        data_point = {"value": 5.0}

        # Valid score
        dto = AnomalyDTO(
            id=anomaly_id,
            score=0.5,
            data_point=data_point,
            detector_name="test_detector",
            timestamp=timestamp,
            severity="low",
        )
        assert dto.score == 0.5

        # Invalid: negative score
        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=anomaly_id,
                score=-0.1,
                data_point=data_point,
                detector_name="test_detector",
                timestamp=timestamp,
                severity="low",
            )

        # Invalid: score greater than 1
        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=anomaly_id,
                score=1.1,
                data_point=data_point,
                detector_name="test_detector",
                timestamp=timestamp,
                severity="low",
            )

    def test_severity_validation(self):
        """Test severity validation."""
        anomaly_id = uuid4()
        timestamp = datetime.now()
        data_point = {"value": 5.0}

        # Valid severities
        valid_severities = ["low", "medium", "high", "critical"]
        for severity in valid_severities:
            dto = AnomalyDTO(
                id=anomaly_id,
                score=0.5,
                data_point=data_point,
                detector_name="test_detector",
                timestamp=timestamp,
                severity=severity,
            )
            assert dto.severity == severity

        # Invalid severity
        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=anomaly_id,
                score=0.5,
                data_point=data_point,
                detector_name="test_detector",
                timestamp=timestamp,
                severity="invalid",
            )

    def test_confidence_validation(self):
        """Test confidence interval validation."""
        anomaly_id = uuid4()
        timestamp = datetime.now()
        data_point = {"value": 5.0}

        # Valid confidence intervals
        dto = AnomalyDTO(
            id=anomaly_id,
            score=0.8,
            data_point=data_point,
            detector_name="test_detector",
            timestamp=timestamp,
            severity="high",
            confidence_lower=0.6,
            confidence_upper=0.9,
        )
        assert dto.confidence_lower == 0.6
        assert dto.confidence_upper == 0.9

        # Invalid: confidence_lower negative
        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=anomaly_id,
                score=0.8,
                data_point=data_point,
                detector_name="test_detector",
                timestamp=timestamp,
                severity="high",
                confidence_lower=-0.1,
            )

        # Invalid: confidence_upper greater than 1
        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=anomaly_id,
                score=0.8,
                data_point=data_point,
                detector_name="test_detector",
                timestamp=timestamp,
                severity="high",
                confidence_upper=1.1,
            )

    def test_complex_data_point(self):
        """Test complex data point structures."""
        anomaly_id = uuid4()
        timestamp = datetime.now()

        # Complex nested data point
        complex_data = {
            "basic_features": {
                "feature1": 1.5,
                "feature2": 2.3,
                "feature3": 0.8,
            },
            "categorical_features": {
                "category_a": "value1",
                "category_b": "value2",
            },
            "time_series_features": {
                "ts_feature1": [1.0, 1.1, 1.2, 1.3],
                "ts_feature2": [0.5, 0.6, 0.7, 0.8],
            },
            "metadata": {
                "source": "sensor_123",
                "timestamp": "2024-01-01T12:00:00",
            },
        }

        dto = AnomalyDTO(
            id=anomaly_id,
            score=0.9,
            data_point=complex_data,
            detector_name="complex_detector",
            timestamp=timestamp,
            severity="critical",
        )

        assert dto.data_point["basic_features"]["feature1"] == 1.5
        assert dto.data_point["categorical_features"]["category_a"] == "value1"
        assert len(dto.data_point["time_series_features"]["ts_feature1"]) == 4
        assert dto.data_point["metadata"]["source"] == "sensor_123"

    def test_metadata_handling(self):
        """Test metadata handling."""
        anomaly_id = uuid4()
        timestamp = datetime.now()
        data_point = {"value": 5.0}

        # Rich metadata
        metadata = {
            "algorithm": "IsolationForest",
            "model_version": "2.1.0",
            "confidence": 0.95,
            "processing_time_ms": 150.0,
            "feature_importance": {
                "feature1": 0.6,
                "feature2": 0.4,
            },
            "anomaly_type": "point_anomaly",
            "context": {
                "previous_anomalies": 3,
                "time_window": "1h",
            },
        }

        dto = AnomalyDTO(
            id=anomaly_id,
            score=0.85,
            data_point=data_point,
            detector_name="advanced_detector",
            timestamp=timestamp,
            severity="high",
            metadata=metadata,
        )

        assert dto.metadata["algorithm"] == "IsolationForest"
        assert dto.metadata["model_version"] == "2.1.0"
        assert dto.metadata["confidence"] == 0.95
        assert dto.metadata["feature_importance"]["feature1"] == 0.6
        assert dto.metadata["context"]["previous_anomalies"] == 3

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            AnomalyDTO(
                score=0.5,
                data_point={"value": 1.0},
                detector_name="test",
                timestamp=datetime.now(),
                severity="medium",
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            AnomalyDTO(
                id=uuid4(),
                score=0.5,
                data_point={"value": 1.0},
                detector_name="test",
                timestamp=datetime.now(),
                severity="medium",
                unknown_field="value",
            )


class TestDetectionResultDTO:
    """Test suite for DetectionResultDTO."""

    def test_valid_creation(self):
        """Test creating a valid detection result DTO."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        run_id = uuid4()
        created_at = datetime.now()

        # Create some anomalies
        anomalies = [
            AnomalyDTO(
                id=uuid4(),
                score=0.9,
                data_point={"feature1": 10.0, "feature2": 20.0},
                detector_name="test_detector",
                timestamp=datetime.now(),
                severity="critical",
            ),
            AnomalyDTO(
                id=uuid4(),
                score=0.7,
                data_point={"feature1": 5.0, "feature2": 15.0},
                detector_name="test_detector",
                timestamp=datetime.now(),
                severity="high",
            ),
        ]

        metadata = {"algorithm": "IsolationForest", "version": "1.0"}
        parameters = {"n_estimators": 100, "contamination": 0.1}

        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            run_id=run_id,
            created_at=created_at,
            duration_seconds=45.5,
            anomalies=anomalies,
            total_samples=1000,
            anomaly_count=2,
            contamination_rate=0.002,
            mean_score=0.3,
            max_score=0.9,
            min_score=0.1,
            threshold=0.6,
            metadata=metadata,
            parameters=parameters,
        )

        assert dto.id == result_id
        assert dto.detector_id == detector_id
        assert dto.dataset_id == dataset_id
        assert dto.run_id == run_id
        assert dto.created_at == created_at
        assert dto.duration_seconds == 45.5
        assert len(dto.anomalies) == 2
        assert dto.total_samples == 1000
        assert dto.anomaly_count == 2
        assert dto.contamination_rate == 0.002
        assert dto.mean_score == 0.3
        assert dto.max_score == 0.9
        assert dto.min_score == 0.1
        assert dto.threshold == 0.6
        assert dto.metadata == metadata
        assert dto.parameters == parameters

    def test_default_values(self):
        """Test default values."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=30.0,
            anomalies=[],
            total_samples=500,
            anomaly_count=0,
            contamination_rate=0.0,
            mean_score=0.2,
            max_score=0.8,
            min_score=0.0,
            threshold=0.5,
        )

        assert dto.run_id is None
        assert dto.metadata == {}
        assert dto.parameters == {}

    def test_contamination_rate_validation(self):
        """Test contamination rate validation."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        # Valid contamination rate
        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=30.0,
            anomalies=[],
            total_samples=500,
            anomaly_count=0,
            contamination_rate=0.05,
            mean_score=0.2,
            max_score=0.8,
            min_score=0.0,
            threshold=0.5,
        )
        assert dto.contamination_rate == 0.05

        # Invalid: negative contamination rate
        with pytest.raises(ValidationError):
            DetectionResultDTO(
                id=result_id,
                detector_id=detector_id,
                dataset_id=dataset_id,
                created_at=created_at,
                duration_seconds=30.0,
                anomalies=[],
                total_samples=500,
                anomaly_count=0,
                contamination_rate=-0.1,
                mean_score=0.2,
                max_score=0.8,
                min_score=0.0,
                threshold=0.5,
            )

        # Invalid: contamination rate greater than 1
        with pytest.raises(ValidationError):
            DetectionResultDTO(
                id=result_id,
                detector_id=detector_id,
                dataset_id=dataset_id,
                created_at=created_at,
                duration_seconds=30.0,
                anomalies=[],
                total_samples=500,
                anomaly_count=0,
                contamination_rate=1.1,
                mean_score=0.2,
                max_score=0.8,
                min_score=0.0,
                threshold=0.5,
            )

    def test_multiple_anomalies(self):
        """Test detection result with multiple anomalies."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        # Create multiple anomalies with different severities
        anomalies = []
        severities = ["low", "medium", "high", "critical"]
        scores = [0.5, 0.6, 0.8, 0.95]

        for i, (severity, score) in enumerate(zip(severities, scores, strict=False)):
            anomaly = AnomalyDTO(
                id=uuid4(),
                score=score,
                data_point={"feature1": float(i), "feature2": float(i * 2)},
                detector_name="multi_detector",
                timestamp=datetime.now(),
                severity=severity,
                explanation=f"Anomaly {i+1} detected",
            )
            anomalies.append(anomaly)

        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=120.0,
            anomalies=anomalies,
            total_samples=10000,
            anomaly_count=4,
            contamination_rate=0.0004,
            mean_score=0.25,
            max_score=0.95,
            min_score=0.05,
            threshold=0.7,
        )

        assert len(dto.anomalies) == 4
        assert dto.anomalies[0].severity == "low"
        assert dto.anomalies[1].severity == "medium"
        assert dto.anomalies[2].severity == "high"
        assert dto.anomalies[3].severity == "critical"
        assert dto.anomalies[3].score == 0.95

    def test_score_statistics(self):
        """Test score statistics tracking."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        # Test with extreme values
        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=60.0,
            anomalies=[],
            total_samples=1000,
            anomaly_count=0,
            contamination_rate=0.0,
            mean_score=0.35,
            max_score=1.0,
            min_score=0.0,
            threshold=0.8,
        )

        assert dto.mean_score == 0.35
        assert dto.max_score == 1.0
        assert dto.min_score == 0.0
        assert dto.threshold == 0.8

        # Test with realistic values
        dto_realistic = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=90.0,
            anomalies=[],
            total_samples=5000,
            anomaly_count=25,
            contamination_rate=0.005,
            mean_score=0.42,
            max_score=0.89,
            min_score=0.12,
            threshold=0.65,
        )

        assert dto_realistic.mean_score == 0.42
        assert dto_realistic.max_score == 0.89
        assert dto_realistic.min_score == 0.12
        assert dto_realistic.threshold == 0.65

    def test_performance_metrics(self):
        """Test performance-related metrics."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        # Test with various duration values
        durations = [0.1, 1.0, 10.0, 60.0, 300.0]  # seconds
        sample_counts = [100, 1000, 10000, 100000, 1000000]

        for duration, samples in zip(durations, sample_counts, strict=False):
            dto = DetectionResultDTO(
                id=result_id,
                detector_id=detector_id,
                dataset_id=dataset_id,
                created_at=created_at,
                duration_seconds=duration,
                anomalies=[],
                total_samples=samples,
                anomaly_count=int(samples * 0.01),
                contamination_rate=0.01,
                mean_score=0.3,
                max_score=0.9,
                min_score=0.1,
                threshold=0.6,
            )

            assert dto.duration_seconds == duration
            assert dto.total_samples == samples
            assert dto.anomaly_count == int(samples * 0.01)

    def test_complex_metadata(self):
        """Test complex metadata structures."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        complex_metadata = {
            "algorithm": "IsolationForest",
            "model_version": "2.1.0",
            "parameters": {
                "n_estimators": 100,
                "max_samples": "auto",
                "contamination": 0.1,
                "random_state": 42,
            },
            "data_preprocessing": {
                "scaling": "StandardScaler",
                "feature_selection": True,
                "pca_components": 10,
            },
            "performance": {
                "training_time_ms": 5000.0,
                "prediction_time_ms": 150.0,
                "memory_usage_mb": 128.5,
            },
            "quality_metrics": {
                "silhouette_score": 0.45,
                "isolation_score": 0.62,
                "contamination_estimate": 0.095,
            },
        }

        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=75.0,
            anomalies=[],
            total_samples=2000,
            anomaly_count=10,
            contamination_rate=0.005,
            mean_score=0.28,
            max_score=0.92,
            min_score=0.08,
            threshold=0.7,
            metadata=complex_metadata,
        )

        assert dto.metadata["algorithm"] == "IsolationForest"
        assert dto.metadata["parameters"]["n_estimators"] == 100
        assert dto.metadata["data_preprocessing"]["scaling"] == "StandardScaler"
        assert dto.metadata["performance"]["training_time_ms"] == 5000.0
        assert dto.metadata["quality_metrics"]["silhouette_score"] == 0.45

    def test_complex_parameters(self):
        """Test complex parameter structures."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        complex_parameters = {
            "detection_params": {
                "algorithm": "IsolationForest",
                "n_estimators": 200,
                "max_samples": 0.8,
                "contamination": 0.05,
                "max_features": 1.0,
                "bootstrap": False,
                "n_jobs": -1,
                "random_state": 42,
                "verbose": 0,
            },
            "preprocessing_params": {
                "normalize": True,
                "standardize": True,
                "remove_outliers": False,
                "feature_selection": {
                    "method": "variance_threshold",
                    "threshold": 0.01,
                },
            },
            "optimization_params": {
                "hyperparameter_tuning": True,
                "cross_validation": {
                    "folds": 5,
                    "strategy": "kfold",
                },
                "grid_search": {
                    "n_estimators": [100, 200, 300],
                    "contamination": [0.05, 0.1, 0.15],
                },
            },
        }

        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=180.0,
            anomalies=[],
            total_samples=5000,
            anomaly_count=25,
            contamination_rate=0.005,
            mean_score=0.32,
            max_score=0.88,
            min_score=0.12,
            threshold=0.6,
            parameters=complex_parameters,
        )

        assert dto.parameters["detection_params"]["algorithm"] == "IsolationForest"
        assert dto.parameters["detection_params"]["n_estimators"] == 200
        assert dto.parameters["preprocessing_params"]["normalize"] is True
        assert dto.parameters["optimization_params"]["hyperparameter_tuning"] is True
        assert (
            len(dto.parameters["optimization_params"]["grid_search"]["n_estimators"])
            == 3
        )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        # Edge case: No anomalies detected
        dto_no_anomalies = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=30.0,
            anomalies=[],
            total_samples=1000,
            anomaly_count=0,
            contamination_rate=0.0,
            mean_score=0.2,
            max_score=0.4,
            min_score=0.0,
            threshold=0.8,
        )

        assert len(dto_no_anomalies.anomalies) == 0
        assert dto_no_anomalies.anomaly_count == 0
        assert dto_no_anomalies.contamination_rate == 0.0

        # Edge case: All samples are anomalies
        dto_all_anomalies = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=60.0,
            anomalies=[],  # We'd need to create 100 anomalies, but this tests the structure
            total_samples=100,
            anomaly_count=100,
            contamination_rate=1.0,
            mean_score=0.9,
            max_score=1.0,
            min_score=0.8,
            threshold=0.1,
        )

        assert dto_all_anomalies.contamination_rate == 1.0
        assert dto_all_anomalies.total_samples == 100
        assert dto_all_anomalies.anomaly_count == 100

        # Edge case: Very small dataset
        dto_small_dataset = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=0.1,
            anomalies=[],
            total_samples=1,
            anomaly_count=1,
            contamination_rate=1.0,
            mean_score=0.95,
            max_score=0.95,
            min_score=0.95,
            threshold=0.5,
        )

        assert dto_small_dataset.total_samples == 1
        assert dto_small_dataset.mean_score == dto_small_dataset.max_score
        assert dto_small_dataset.max_score == dto_small_dataset.min_score

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValidationError):
            DetectionResultDTO(
                detector_id=uuid4(),
                dataset_id=uuid4(),
                created_at=datetime.now(),
                duration_seconds=30.0,
                anomalies=[],
                total_samples=1000,
                anomaly_count=0,
                contamination_rate=0.0,
                mean_score=0.2,
                max_score=0.8,
                min_score=0.0,
                threshold=0.5,
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            DetectionResultDTO(
                id=uuid4(),
                detector_id=uuid4(),
                dataset_id=uuid4(),
                created_at=datetime.now(),
                duration_seconds=30.0,
                anomalies=[],
                total_samples=1000,
                anomaly_count=0,
                contamination_rate=0.0,
                mean_score=0.2,
                max_score=0.8,
                min_score=0.0,
                threshold=0.5,
                unknown_field="value",
            )


class TestResultDTOIntegration:
    """Test integration scenarios for result DTOs."""

    def test_detection_result_with_anomalies(self):
        """Test detection result with comprehensive anomaly data."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        # Create anomalies with different characteristics
        anomalies = []
        for i in range(5):
            anomaly = AnomalyDTO(
                id=uuid4(),
                score=0.6 + (i * 0.08),  # Increasing scores
                data_point={
                    "feature1": float(i * 2),
                    "feature2": float(i * 3),
                    "feature3": float(i * 1.5),
                },
                detector_name="comprehensive_detector",
                timestamp=datetime.now(),
                severity=["low", "medium", "high", "high", "critical"][i],
                explanation=f"Anomaly {i+1}: deviation in feature patterns",
                metadata={
                    "anomaly_type": [
                        "point",
                        "collective",
                        "contextual",
                        "point",
                        "collective",
                    ][i],
                    "confidence": 0.7 + (i * 0.05),
                    "feature_contributions": {
                        "feature1": 0.3 + (i * 0.1),
                        "feature2": 0.4 + (i * 0.05),
                        "feature3": 0.3 + (i * 0.08),
                    },
                },
                confidence_lower=0.5 + (i * 0.05),
                confidence_upper=0.8 + (i * 0.04),
            )
            anomalies.append(anomaly)

        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=95.5,
            anomalies=anomalies,
            total_samples=10000,
            anomaly_count=5,
            contamination_rate=0.0005,
            mean_score=0.31,
            max_score=0.92,
            min_score=0.08,
            threshold=0.65,
            metadata={
                "algorithm": "IsolationForest",
                "model_performance": {
                    "precision": 0.85,
                    "recall": 0.78,
                    "f1_score": 0.81,
                },
                "data_characteristics": {
                    "n_features": 3,
                    "feature_types": ["numerical", "numerical", "numerical"],
                    "missing_values": False,
                },
            },
            parameters={
                "n_estimators": 100,
                "contamination": 0.05,
                "random_state": 42,
            },
        )

        # Verify the overall structure
        assert len(dto.anomalies) == 5
        assert dto.anomaly_count == 5
        assert dto.contamination_rate == 0.0005

        # Verify individual anomalies
        assert dto.anomalies[0].severity == "low"
        assert dto.anomalies[4].severity == "critical"
        assert dto.anomalies[0].score == 0.6
        assert dto.anomalies[4].score == 0.92

        # Verify metadata integration
        assert dto.metadata["algorithm"] == "IsolationForest"
        assert dto.metadata["model_performance"]["precision"] == 0.85
        assert dto.anomalies[0].metadata["anomaly_type"] == "point"
        assert dto.anomalies[4].metadata["anomaly_type"] == "collective"

    def test_serialization_deserialization(self):
        """Test DTO serialization and deserialization."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        # Create anomaly
        anomaly = AnomalyDTO(
            id=uuid4(),
            score=0.85,
            data_point={"feature1": 5.0, "feature2": 10.0},
            detector_name="test_detector",
            timestamp=datetime.now(),
            severity="high",
            explanation="Test anomaly",
            metadata={"test": "value"},
        )

        # Create detection result
        original_dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=45.0,
            anomalies=[anomaly],
            total_samples=1000,
            anomaly_count=1,
            contamination_rate=0.001,
            mean_score=0.3,
            max_score=0.85,
            min_score=0.1,
            threshold=0.7,
            metadata={"algorithm": "IsolationForest"},
            parameters={"n_estimators": 100},
        )

        # Serialize to dict
        dto_dict = original_dto.model_dump()

        # Verify serialization
        assert dto_dict["id"] == str(result_id)
        assert dto_dict["detector_id"] == str(detector_id)
        assert dto_dict["dataset_id"] == str(dataset_id)
        assert dto_dict["total_samples"] == 1000
        assert dto_dict["anomaly_count"] == 1
        assert len(dto_dict["anomalies"]) == 1
        assert dto_dict["anomalies"][0]["severity"] == "high"

        # Deserialize from dict
        restored_dto = DetectionResultDTO.model_validate(dto_dict)

        # Verify deserialization
        assert restored_dto.id == original_dto.id
        assert restored_dto.detector_id == original_dto.detector_id
        assert restored_dto.dataset_id == original_dto.dataset_id
        assert restored_dto.total_samples == original_dto.total_samples
        assert restored_dto.anomaly_count == original_dto.anomaly_count
        assert len(restored_dto.anomalies) == len(original_dto.anomalies)
        assert restored_dto.anomalies[0].severity == original_dto.anomalies[0].severity
        assert restored_dto.metadata == original_dto.metadata
        assert restored_dto.parameters == original_dto.parameters

    def test_batch_processing_scenario(self):
        """Test scenario with batch processing results."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        # Simulate batch processing with multiple anomalies
        batch_size = 10
        anomalies = []

        for i in range(batch_size):
            anomaly = AnomalyDTO(
                id=uuid4(),
                score=0.5 + (i * 0.04),  # Gradually increasing scores
                data_point={
                    "batch_id": i,
                    "feature1": float(i),
                    "feature2": float(i * 2),
                    "timestamp": f"2024-01-01T{i:02d}:00:00",
                },
                detector_name="batch_detector",
                timestamp=datetime.now(),
                severity="medium" if i < 5 else "high",
                explanation=f"Batch anomaly {i+1}",
                metadata={
                    "batch_number": i + 1,
                    "processing_order": i,
                    "batch_confidence": 0.8 + (i * 0.01),
                },
            )
            anomalies.append(anomaly)

        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=250.0,  # Longer for batch processing
            anomalies=anomalies,
            total_samples=50000,  # Large batch
            anomaly_count=batch_size,
            contamination_rate=0.0002,
            mean_score=0.35,
            max_score=0.86,
            min_score=0.05,
            threshold=0.6,
            metadata={
                "batch_processing": True,
                "batch_size": batch_size,
                "total_batches": 1,
                "processing_mode": "parallel",
                "average_batch_time": 25.0,
            },
            parameters={
                "algorithm": "IsolationForest",
                "batch_size": batch_size,
                "parallel_processing": True,
                "n_jobs": 4,
            },
        )

        # Verify batch processing structure
        assert len(dto.anomalies) == batch_size
        assert dto.metadata["batch_processing"] is True
        assert dto.metadata["batch_size"] == batch_size
        assert dto.parameters["parallel_processing"] is True

        # Verify anomaly progression
        assert dto.anomalies[0].score == 0.5
        assert dto.anomalies[-1].score == 0.86
        assert dto.anomalies[0].severity == "medium"
        assert dto.anomalies[-1].severity == "high"

        # Verify metadata consistency
        for i, anomaly in enumerate(dto.anomalies):
            assert anomaly.metadata["batch_number"] == i + 1
            assert anomaly.metadata["processing_order"] == i
            assert anomaly.data_point["batch_id"] == i

    def test_real_time_processing_scenario(self):
        """Test scenario with real-time processing results."""
        result_id = uuid4()
        detector_id = uuid4()
        dataset_id = uuid4()
        created_at = datetime.now()

        # Single anomaly from real-time processing
        anomaly = AnomalyDTO(
            id=uuid4(),
            score=0.92,
            data_point={
                "timestamp": "2024-01-01T12:34:56",
                "sensor_id": "sensor_001",
                "temperature": 85.5,
                "pressure": 1023.4,
                "humidity": 45.2,
                "location": "production_line_1",
            },
            detector_name="realtime_detector",
            timestamp=datetime.now(),
            severity="critical",
            explanation="Critical temperature anomaly detected",
            metadata={
                "processing_mode": "real_time",
                "alert_triggered": True,
                "response_time_ms": 50.0,
                "confidence": 0.98,
                "feature_contributions": {
                    "temperature": 0.85,
                    "pressure": 0.10,
                    "humidity": 0.05,
                },
            },
            confidence_lower=0.88,
            confidence_upper=0.96,
        )

        dto = DetectionResultDTO(
            id=result_id,
            detector_id=detector_id,
            dataset_id=dataset_id,
            created_at=created_at,
            duration_seconds=0.055,  # Very fast real-time processing
            anomalies=[anomaly],
            total_samples=1,
            anomaly_count=1,
            contamination_rate=1.0,
            mean_score=0.92,
            max_score=0.92,
            min_score=0.92,
            threshold=0.8,
            metadata={
                "processing_mode": "real_time",
                "stream_processing": True,
                "latency_ms": 50.0,
                "throughput_samples_per_second": 18.18,
            },
            parameters={
                "algorithm": "OnlineIsolationForest",
                "window_size": 1000,
                "learning_rate": 0.01,
                "adaptation_enabled": True,
            },
        )

        # Verify real-time processing characteristics
        assert dto.duration_seconds < 0.1  # Very fast processing
        assert dto.total_samples == 1
        assert dto.anomaly_count == 1
        assert dto.contamination_rate == 1.0
        assert dto.metadata["processing_mode"] == "real_time"
        assert dto.metadata["stream_processing"] is True
        assert dto.parameters["algorithm"] == "OnlineIsolationForest"

        # Verify anomaly characteristics
        assert dto.anomalies[0].severity == "critical"
        assert dto.anomalies[0].score == 0.92
        assert dto.anomalies[0].metadata["alert_triggered"] is True
        assert dto.anomalies[0].metadata["response_time_ms"] == 50.0
        assert dto.anomalies[0].data_point["sensor_id"] == "sensor_001"
