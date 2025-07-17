"""
Tests for streaming DTOs.

This module provides comprehensive tests for all streaming data transfer objects,
ensuring proper validation, serialization, and type safety for real-time anomaly detection.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from src.monorepo.application.dto.streaming_dto import (
    BackpressureConfigDTO,
    CheckpointConfigDTO,
    StreamConfigurationDTO,
    StreamDataBatchDTO,
    StreamDataPointDTO,
    StreamDetectionRequestDTO,
    StreamDetectionResponseDTO,
    StreamErrorDTO,
    StreamMetricsDTO,
    StreamStatusDTO,
    WindowConfigDTO,
)


class TestStreamDataPointDTO:
    """Test cases for StreamDataPointDTO."""

    def test_create_valid_data_point(self):
        """Test creating a valid stream data point."""
        timestamp = datetime.now(UTC)
        features = {"feature1": 1.5, "feature2": 2.0}

        dto = StreamDataPointDTO(
            timestamp=timestamp, features=features, metadata={"source": "sensor1"}
        )

        assert dto.timestamp == timestamp
        assert dto.features == features
        assert dto.metadata == {"source": "sensor1"}
        assert dto.anomaly_score is None
        assert dto.is_anomaly is None

    def test_data_point_with_prediction(self):
        """Test data point with anomaly prediction."""
        timestamp = datetime.now(UTC)
        features = {"feature1": 1.5}

        dto = StreamDataPointDTO(
            timestamp=timestamp, features=features, anomaly_score=0.85, is_anomaly=True
        )

        assert dto.anomaly_score == 0.85
        assert dto.is_anomaly is True

    def test_invalid_features_type(self):
        """Test validation for invalid features type."""
        with pytest.raises(ValueError, match="Features must be a dictionary"):
            StreamDataPointDTO(timestamp=datetime.now(UTC), features="invalid")

    def test_invalid_anomaly_score_range(self):
        """Test validation for anomaly score range."""
        with pytest.raises(ValueError, match="Anomaly score must be between 0 and 1"):
            StreamDataPointDTO(
                timestamp=datetime.now(UTC),
                features={"feature1": 1.0},
                anomaly_score=1.5,
            )

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        timestamp = datetime.now(UTC)
        dto = StreamDataPointDTO(
            timestamp=timestamp, features={"feature1": 1.5}, anomaly_score=0.7
        )

        result = dto.to_dict()

        assert result["timestamp"] == timestamp.isoformat()
        assert result["features"] == {"feature1": 1.5}
        assert result["anomaly_score"] == 0.7

    def test_from_dict_creation(self):
        """Test creation from dictionary."""
        timestamp = datetime.now(UTC)
        data = {
            "timestamp": timestamp.isoformat(),
            "features": {"feature1": 1.5},
            "anomaly_score": 0.7,
        }

        dto = StreamDataPointDTO.from_dict(data)

        assert abs((dto.timestamp - timestamp).total_seconds()) < 1
        assert dto.features == {"feature1": 1.5}
        assert dto.anomaly_score == 0.7


class TestStreamDataBatchDTO:
    """Test cases for StreamDataBatchDTO."""

    def test_create_valid_batch(self):
        """Test creating a valid data batch."""
        data_points = [
            StreamDataPointDTO(
                timestamp=datetime.now(UTC), features={"feature1": 1.0}
            ),
            StreamDataPointDTO(
                timestamp=datetime.now(UTC), features={"feature1": 2.0}
            ),
        ]

        dto = StreamDataBatchDTO(
            batch_id="batch_123",
            data_points=data_points,
            window_start=datetime.now(UTC),
            window_end=datetime.now(UTC),
        )

        assert dto.batch_id == "batch_123"
        assert len(dto.data_points) == 2
        assert dto.batch_size == 2
        assert dto.window_start is not None
        assert dto.window_end is not None

    def test_empty_batch_validation(self):
        """Test validation for empty batch."""
        with pytest.raises(ValueError, match="Batch cannot be empty"):
            StreamDataBatchDTO(
                batch_id="batch_123",
                data_points=[],
                window_start=datetime.now(UTC),
                window_end=datetime.now(UTC),
            )

    def test_batch_size_property(self):
        """Test batch size calculation."""
        data_points = [Mock() for _ in range(5)]
        dto = StreamDataBatchDTO(
            batch_id="batch_123",
            data_points=data_points,
            window_start=datetime.now(UTC),
            window_end=datetime.now(UTC),
        )

        assert dto.batch_size == 5

    def test_to_pandas_conversion(self):
        """Test conversion to pandas DataFrame."""
        timestamp1 = datetime.now(UTC)
        timestamp2 = datetime.now(UTC)

        data_points = [
            StreamDataPointDTO(
                timestamp=timestamp1, features={"feature1": 1.0, "feature2": 2.0}
            ),
            StreamDataPointDTO(
                timestamp=timestamp2, features={"feature1": 1.5, "feature2": 2.5}
            ),
        ]

        dto = StreamDataBatchDTO(
            batch_id="batch_123",
            data_points=data_points,
            window_start=timestamp1,
            window_end=timestamp2,
        )

        df = dto.to_pandas()

        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "feature1" in df.columns
        assert "feature2" in df.columns
        assert df.iloc[0]["feature1"] == 1.0
        assert df.iloc[1]["feature1"] == 1.5


class TestStreamDetectionRequestDTO:
    """Test cases for StreamDetectionRequestDTO."""

    def test_create_valid_request(self):
        """Test creating a valid detection request."""
        batch = Mock(spec=StreamDataBatchDTO)
        config = Mock(spec=StreamConfigurationDTO)

        dto = StreamDetectionRequestDTO(
            request_id="req_123",
            batch=batch,
            configuration=config,
            detector_id="detector_456",
        )

        assert dto.request_id == "req_123"
        assert dto.batch == batch
        assert dto.configuration == config
        assert dto.detector_id == "detector_456"
        assert dto.timestamp is not None

    def test_request_validation(self):
        """Test request validation."""
        with pytest.raises(ValueError, match="Request ID cannot be empty"):
            StreamDetectionRequestDTO(
                request_id="",
                batch=Mock(),
                configuration=Mock(),
                detector_id="detector_456",
            )


class TestStreamDetectionResponseDTO:
    """Test cases for StreamDetectionResponseDTO."""

    def test_create_valid_response(self):
        """Test creating a valid detection response."""
        results = [Mock() for _ in range(3)]

        dto = StreamDetectionResponseDTO(
            request_id="req_123",
            results=results,
            processing_time_ms=150.5,
            detector_id="detector_456",
            success=True,
        )

        assert dto.request_id == "req_123"
        assert len(dto.results) == 3
        assert dto.processing_time_ms == 150.5
        assert dto.detector_id == "detector_456"
        assert dto.success is True
        assert dto.error_message is None

    def test_error_response(self):
        """Test creating an error response."""
        dto = StreamDetectionResponseDTO(
            request_id="req_123",
            results=[],
            processing_time_ms=50.0,
            detector_id="detector_456",
            success=False,
            error_message="Processing failed",
        )

        assert dto.success is False
        assert dto.error_message == "Processing failed"

    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        dto = StreamDetectionResponseDTO(
            request_id="req_123",
            results=[Mock() for _ in range(100)],
            processing_time_ms=100.0,
            detector_id="detector_456",
            success=True,
        )

        # Should process 100 points in 100ms = 1000 points/second
        assert dto.throughput_points_per_second == 1000.0


class TestStreamConfigurationDTO:
    """Test cases for StreamConfigurationDTO."""

    def test_create_valid_configuration(self):
        """Test creating a valid stream configuration."""
        backpressure = BackpressureConfigDTO(
            enabled=True, max_queue_size=1000, drop_policy="oldest"
        )

        window = WindowConfigDTO(type="tumbling", size_ms=5000, slide_ms=1000)

        checkpoint = CheckpointConfigDTO(
            enabled=True, interval_ms=10000, storage_path="/tmp/checkpoints"
        )

        dto = StreamConfigurationDTO(
            stream_id="stream_123",
            batch_size=100,
            timeout_ms=30000,
            backpressure=backpressure,
            window=window,
            checkpoint=checkpoint,
        )

        assert dto.stream_id == "stream_123"
        assert dto.batch_size == 100
        assert dto.timeout_ms == 30000
        assert dto.backpressure == backpressure
        assert dto.window == window
        assert dto.checkpoint == checkpoint

    def test_batch_size_validation(self):
        """Test batch size validation."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            StreamConfigurationDTO(
                stream_id="stream_123", batch_size=0, timeout_ms=30000
            )

    def test_timeout_validation(self):
        """Test timeout validation."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            StreamConfigurationDTO(
                stream_id="stream_123", batch_size=100, timeout_ms=-1000
            )


class TestStreamMetricsDTO:
    """Test cases for StreamMetricsDTO."""

    def test_create_valid_metrics(self):
        """Test creating valid stream metrics."""
        dto = StreamMetricsDTO(
            stream_id="stream_123",
            messages_processed=1000,
            messages_failed=5,
            average_processing_time_ms=50.5,
            throughput_per_second=200.0,
            backpressure_events=2,
            window_start=datetime.now(UTC),
            window_end=datetime.now(UTC),
        )

        assert dto.stream_id == "stream_123"
        assert dto.messages_processed == 1000
        assert dto.messages_failed == 5
        assert dto.success_rate == 0.995  # (1000-5)/1000
        assert dto.average_processing_time_ms == 50.5
        assert dto.throughput_per_second == 200.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        dto = StreamMetricsDTO(
            stream_id="stream_123",
            messages_processed=100,
            messages_failed=10,
            average_processing_time_ms=50.0,
            throughput_per_second=100.0,
            backpressure_events=0,
            window_start=datetime.now(UTC),
            window_end=datetime.now(UTC),
        )

        assert dto.success_rate == 0.9  # (100-10)/100

    def test_zero_messages_processed(self):
        """Test handling of zero messages processed."""
        dto = StreamMetricsDTO(
            stream_id="stream_123",
            messages_processed=0,
            messages_failed=0,
            average_processing_time_ms=0.0,
            throughput_per_second=0.0,
            backpressure_events=0,
            window_start=datetime.now(UTC),
            window_end=datetime.now(UTC),
        )

        assert dto.success_rate == 1.0  # No failures when no messages processed


class TestStreamStatusDTO:
    """Test cases for StreamStatusDTO."""

    def test_create_valid_status(self):
        """Test creating valid stream status."""
        dto = StreamStatusDTO(
            stream_id="stream_123",
            status="running",
            uptime_seconds=3600,
            last_processed_timestamp=datetime.now(UTC),
            current_lag_ms=100,
            health_check_status="healthy",
        )

        assert dto.stream_id == "stream_123"
        assert dto.status == "running"
        assert dto.uptime_seconds == 3600
        assert dto.current_lag_ms == 100
        assert dto.health_check_status == "healthy"
        assert dto.is_healthy is True

    def test_unhealthy_status(self):
        """Test unhealthy status detection."""
        dto = StreamStatusDTO(
            stream_id="stream_123",
            status="error",
            uptime_seconds=3600,
            last_processed_timestamp=datetime.now(UTC),
            current_lag_ms=5000,
            health_check_status="unhealthy",
        )

        assert dto.is_healthy is False

    def test_status_validation(self):
        """Test status validation."""
        valid_statuses = ["starting", "running", "paused", "stopped", "error"]

        for status in valid_statuses:
            dto = StreamStatusDTO(
                stream_id="stream_123",
                status=status,
                uptime_seconds=0,
                last_processed_timestamp=datetime.now(UTC),
                current_lag_ms=0,
                health_check_status="healthy",
            )
            assert dto.status == status

        with pytest.raises(ValueError, match="Invalid status"):
            StreamStatusDTO(
                stream_id="stream_123",
                status="invalid_status",
                uptime_seconds=0,
                last_processed_timestamp=datetime.now(UTC),
                current_lag_ms=0,
                health_check_status="healthy",
            )


class TestStreamErrorDTO:
    """Test cases for StreamErrorDTO."""

    def test_create_valid_error(self):
        """Test creating valid stream error."""
        dto = StreamErrorDTO(
            stream_id="stream_123",
            error_type="processing_error",
            error_message="Failed to process batch",
            timestamp=datetime.now(UTC),
            severity="high",
            context={"batch_id": "batch_456", "detector_id": "detector_789"},
        )

        assert dto.stream_id == "stream_123"
        assert dto.error_type == "processing_error"
        assert dto.error_message == "Failed to process batch"
        assert dto.severity == "high"
        assert dto.context["batch_id"] == "batch_456"
        assert dto.is_recoverable is None

    def test_severity_validation(self):
        """Test severity validation."""
        valid_severities = ["low", "medium", "high", "critical"]

        for severity in valid_severities:
            dto = StreamErrorDTO(
                stream_id="stream_123",
                error_type="test_error",
                error_message="Test message",
                timestamp=datetime.now(UTC),
                severity=severity,
            )
            assert dto.severity == severity

        with pytest.raises(ValueError, match="Invalid severity"):
            StreamErrorDTO(
                stream_id="stream_123",
                error_type="test_error",
                error_message="Test message",
                timestamp=datetime.now(UTC),
                severity="invalid_severity",
            )


class TestBackpressureConfigDTO:
    """Test cases for BackpressureConfigDTO."""

    def test_create_valid_backpressure_config(self):
        """Test creating valid backpressure configuration."""
        dto = BackpressureConfigDTO(
            enabled=True,
            max_queue_size=1000,
            drop_policy="oldest",
            threshold_percentage=0.8,
        )

        assert dto.enabled is True
        assert dto.max_queue_size == 1000
        assert dto.drop_policy == "oldest"
        assert dto.threshold_percentage == 0.8

    def test_drop_policy_validation(self):
        """Test drop policy validation."""
        valid_policies = ["oldest", "newest", "random", "reject"]

        for policy in valid_policies:
            dto = BackpressureConfigDTO(
                enabled=True, max_queue_size=1000, drop_policy=policy
            )
            assert dto.drop_policy == policy

        with pytest.raises(ValueError, match="Invalid drop policy"):
            BackpressureConfigDTO(
                enabled=True, max_queue_size=1000, drop_policy="invalid_policy"
            )


class TestWindowConfigDTO:
    """Test cases for WindowConfigDTO."""

    def test_create_tumbling_window(self):
        """Test creating tumbling window configuration."""
        dto = WindowConfigDTO(type="tumbling", size_ms=5000)

        assert dto.type == "tumbling"
        assert dto.size_ms == 5000
        assert dto.slide_ms is None

    def test_create_sliding_window(self):
        """Test creating sliding window configuration."""
        dto = WindowConfigDTO(type="sliding", size_ms=10000, slide_ms=2000)

        assert dto.type == "sliding"
        assert dto.size_ms == 10000
        assert dto.slide_ms == 2000

    def test_window_type_validation(self):
        """Test window type validation."""
        valid_types = ["tumbling", "sliding", "session"]

        for window_type in valid_types:
            dto = WindowConfigDTO(type=window_type, size_ms=5000)
            assert dto.type == window_type

        with pytest.raises(ValueError, match="Invalid window type"):
            WindowConfigDTO(type="invalid_type", size_ms=5000)

    def test_sliding_window_slide_requirement(self):
        """Test that sliding windows require slide_ms."""
        with pytest.raises(ValueError, match="Sliding windows require slide_ms"):
            WindowConfigDTO(type="sliding", size_ms=5000)


class TestCheckpointConfigDTO:
    """Test cases for CheckpointConfigDTO."""

    def test_create_valid_checkpoint_config(self):
        """Test creating valid checkpoint configuration."""
        dto = CheckpointConfigDTO(
            enabled=True,
            interval_ms=10000,
            storage_path="/tmp/checkpoints",
            compression=True,
            retention_count=10,
        )

        assert dto.enabled is True
        assert dto.interval_ms == 10000
        assert dto.storage_path == "/tmp/checkpoints"
        assert dto.compression is True
        assert dto.retention_count == 10

    def test_interval_validation(self):
        """Test interval validation."""
        with pytest.raises(ValueError, match="Checkpoint interval must be positive"):
            CheckpointConfigDTO(
                enabled=True, interval_ms=-1000, storage_path="/tmp/checkpoints"
            )

    def test_retention_validation(self):
        """Test retention count validation."""
        with pytest.raises(ValueError, match="Retention count must be positive"):
            CheckpointConfigDTO(
                enabled=True,
                interval_ms=10000,
                storage_path="/tmp/checkpoints",
                retention_count=0,
            )


class TestDTOSerialization:
    """Test serialization and deserialization of DTOs."""

    def test_stream_data_point_json_roundtrip(self):
        """Test JSON serialization roundtrip for StreamDataPointDTO."""
        original = StreamDataPointDTO(
            timestamp=datetime.now(UTC),
            features={"feature1": 1.5, "feature2": 2.0},
            anomaly_score=0.75,
            is_anomaly=True,
            metadata={"source": "sensor1"},
        )

        # Convert to dict (JSON-serializable)
        data = original.to_dict()

        # Reconstruct from dict
        reconstructed = StreamDataPointDTO.from_dict(data)

        assert abs((reconstructed.timestamp - original.timestamp).total_seconds()) < 1
        assert reconstructed.features == original.features
        assert reconstructed.anomaly_score == original.anomaly_score
        assert reconstructed.is_anomaly == original.is_anomaly
        assert reconstructed.metadata == original.metadata

    def test_configuration_serialization(self):
        """Test configuration DTO serialization."""
        config = StreamConfigurationDTO(
            stream_id="stream_123",
            batch_size=100,
            timeout_ms=30000,
            backpressure=BackpressureConfigDTO(
                enabled=True, max_queue_size=1000, drop_policy="oldest"
            ),
            window=WindowConfigDTO(type="tumbling", size_ms=5000),
            checkpoint=CheckpointConfigDTO(
                enabled=True, interval_ms=10000, storage_path="/tmp/checkpoints"
            ),
        )

        # Should be able to convert to dict for serialization
        data = config.to_dict() if hasattr(config, "to_dict") else config.__dict__

        assert data["stream_id"] == "stream_123"
        assert data["batch_size"] == 100
