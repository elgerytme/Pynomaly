"""Tests for Streaming DTOs."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from pynomaly.application.dto.streaming_dto import (
    StreamDataBatchDTO,
    StreamDataPointDTO,
    StreamErrorDTO,
    StreamingBatchResultDTO,
    StreamingConfigurationDTO,
    StreamingControlDTO,
    StreamingHealthCheckDTO,
    StreamingMetricsDTO,
    StreamingRequestDTO,
    StreamingResponseDTO,
    StreamingResultDTO,
    StreamingSampleDTO,
)


class TestStreamDataPointDTO:
    """Test suite for StreamDataPointDTO."""

    def test_valid_creation(self):
        """Test creating a valid stream data point DTO."""
        timestamp = datetime.now()
        features = {"temperature": 25.5, "pressure": 1013.25, "humidity": 60.0}
        metadata = {"sensor_id": "sensor_001", "location": "building_a"}

        dto = StreamDataPointDTO(
            timestamp=timestamp,
            features=features,
            metadata=metadata,
            anomaly_score=0.75,
            is_anomaly=True,
        )

        assert dto.timestamp == timestamp
        assert dto.features == features
        assert dto.metadata == metadata
        assert dto.anomaly_score == 0.75
        assert dto.is_anomaly is True

    def test_default_values(self):
        """Test default values."""
        features = {"value": 10.0}

        dto = StreamDataPointDTO(features=features)

        assert dto.timestamp is None
        assert dto.features == features
        assert dto.metadata == {}
        assert dto.anomaly_score is None
        assert dto.is_anomaly is None

    def test_features_validation(self):
        """Test features validation."""
        # Valid features
        valid_features = {"feature1": 1.0, "feature2": 2.5, "feature3": 3}
        dto = StreamDataPointDTO(features=valid_features)
        assert dto.features == valid_features

        # Invalid: empty features
        with pytest.raises(ValidationError):
            StreamDataPointDTO(features={})

        # Invalid: non-numeric features
        with pytest.raises(ValidationError):
            StreamDataPointDTO(features={"feature1": "invalid", "feature2": 2.0})

        # Invalid: mixed types with string
        with pytest.raises(ValidationError):
            StreamDataPointDTO(features={"feature1": 1.0, "feature2": "string"})

    def test_anomaly_score_validation(self):
        """Test anomaly score validation."""
        features = {"value": 10.0}

        # Valid anomaly scores
        valid_scores = [0.0, 0.5, 1.0, 0.75, 0.99]
        for score in valid_scores:
            dto = StreamDataPointDTO(features=features, anomaly_score=score)
            assert dto.anomaly_score == score

        # Invalid: negative score
        with pytest.raises(ValidationError):
            StreamDataPointDTO(features=features, anomaly_score=-0.1)

        # Invalid: score greater than 1
        with pytest.raises(ValidationError):
            StreamDataPointDTO(features=features, anomaly_score=1.1)

    def test_to_dict_serialization(self):
        """Test to_dict serialization."""
        timestamp = datetime.now()
        features = {"temperature": 25.5, "pressure": 1013.25}
        metadata = {"sensor_id": "sensor_001"}

        dto = StreamDataPointDTO(
            timestamp=timestamp,
            features=features,
            metadata=metadata,
            anomaly_score=0.8,
            is_anomaly=True,
        )

        result = dto.to_dict()

        assert result["timestamp"] == timestamp.isoformat()
        assert result["features"] == features
        assert result["metadata"] == metadata
        assert result["anomaly_score"] == 0.8
        assert result["is_anomaly"] is True

    def test_from_dict_deserialization(self):
        """Test from_dict deserialization."""
        timestamp_str = "2024-01-01T12:00:00"
        data = {
            "timestamp": timestamp_str,
            "features": {"temperature": 25.5, "pressure": 1013.25},
            "metadata": {"sensor_id": "sensor_001"},
            "anomaly_score": 0.8,
            "is_anomaly": True,
        }

        dto = StreamDataPointDTO.from_dict(data)

        assert dto.timestamp.isoformat() == timestamp_str
        assert dto.features == data["features"]
        assert dto.metadata == data["metadata"]
        assert dto.anomaly_score == 0.8
        assert dto.is_anomaly is True

    def test_from_dict_with_timezone(self):
        """Test from_dict with timezone handling."""
        data = {
            "timestamp": "2024-01-01T12:00:00Z",
            "features": {"value": 10.0},
        }

        dto = StreamDataPointDTO.from_dict(data)

        assert dto.timestamp is not None
        assert dto.features == {"value": 10.0}

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            StreamDataPointDTO(
                features={"value": 10.0},
                unknown_field="value",
            )


class TestStreamDataBatchDTO:
    """Test suite for StreamDataBatchDTO."""

    def test_valid_creation(self):
        """Test creating a valid stream data batch DTO."""
        data_points = [
            StreamDataPointDTO(features={"value": 10.0}, anomaly_score=0.5),
            StreamDataPointDTO(features={"value": 20.0}, anomaly_score=0.7),
            StreamDataPointDTO(features={"value": 30.0}, anomaly_score=0.3),
        ]

        batch_id = "batch_001"
        timestamp = datetime.now()
        window_start = datetime.now()
        window_end = datetime.now()

        dto = StreamDataBatchDTO(
            batch_id=batch_id,
            data_points=data_points,
            timestamp=timestamp,
            window_start=window_start,
            window_end=window_end,
        )

        assert dto.batch_id == batch_id
        assert len(dto.data_points) == 3
        assert dto.timestamp == timestamp
        assert dto.window_start == window_start
        assert dto.window_end == window_end

    def test_default_values(self):
        """Test default values."""
        data_points = [StreamDataPointDTO(features={"value": 10.0})]

        dto = StreamDataBatchDTO(data_points=data_points)

        assert dto.batch_id is None
        assert dto.timestamp is None
        assert dto.window_start is None
        assert dto.window_end is None

    def test_batch_size_property(self):
        """Test batch_size property."""
        data_points = [StreamDataPointDTO(features={"value": i}) for i in range(5)]

        dto = StreamDataBatchDTO(data_points=data_points)

        assert dto.batch_size == 5

    def test_data_points_validation(self):
        """Test data_points validation."""
        # Valid data points
        valid_data_points = [
            StreamDataPointDTO(features={"value": 10.0}),
            StreamDataPointDTO(features={"value": 20.0}),
        ]

        dto = StreamDataBatchDTO(data_points=valid_data_points)
        assert len(dto.data_points) == 2

        # Invalid: empty data points
        with pytest.raises(ValidationError):
            StreamDataBatchDTO(data_points=[])

    def test_to_pandas_conversion(self):
        """Test to_pandas conversion."""
        # Skip test if pandas is not available
        pytest.importorskip("pandas")

        data_points = [
            StreamDataPointDTO(
                features={"temp": 25.0, "pressure": 1013.0},
                anomaly_score=0.5,
                is_anomaly=False,
                metadata={"sensor": "s1"},
            ),
            StreamDataPointDTO(
                features={"temp": 30.0, "pressure": 1015.0},
                anomaly_score=0.8,
                is_anomaly=True,
                metadata={"sensor": "s2"},
            ),
        ]

        dto = StreamDataBatchDTO(data_points=data_points)
        df = dto.to_pandas()

        assert len(df) == 2
        assert "temp" in df.columns
        assert "pressure" in df.columns
        assert "anomaly_score" in df.columns
        assert "is_anomaly" in df.columns
        assert "meta_sensor" in df.columns

        # Test values
        assert df.iloc[0]["temp"] == 25.0
        assert df.iloc[0]["anomaly_score"] == 0.5
        assert df.iloc[0]["is_anomaly"] is False
        assert df.iloc[0]["meta_sensor"] == "s1"

    def test_to_pandas_empty_batch(self):
        """Test to_pandas with empty batch."""
        pytest.importorskip("pandas")

        # Create empty batch (this should not be possible due to validation,
        # but test the method directly)
        dto = StreamDataBatchDTO.__new__(StreamDataBatchDTO)
        dto.data_points = []

        df = dto.to_pandas()
        assert len(df) == 0

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        data_points = [StreamDataPointDTO(features={"value": 10.0})]

        with pytest.raises(ValidationError):
            StreamDataBatchDTO(
                data_points=data_points,
                unknown_field="value",
            )


class TestStreamingConfigurationDTO:
    """Test suite for StreamingConfigurationDTO."""

    def test_valid_creation(self):
        """Test creating a valid streaming configuration DTO."""
        dto = StreamingConfigurationDTO(
            strategy="adaptive_batch",
            backpressure_strategy="adaptive_sampling",
            mode="continuous",
            max_buffer_size=5000,
            min_batch_size=10,
            max_batch_size=50,
            batch_timeout_ms=200,
            high_watermark=0.9,
            low_watermark=0.2,
            max_processing_time_ms=2000,
            max_concurrent_batches=3,
            adaptive_scaling_enabled=True,
            enable_quality_monitoring=True,
            quality_check_interval_ms=10000,
            drift_detection_enabled=True,
            enable_result_buffering=True,
            result_buffer_size=2000,
            enable_metrics_collection=True,
        )

        assert dto.strategy == "adaptive_batch"
        assert dto.backpressure_strategy == "adaptive_sampling"
        assert dto.mode == "continuous"
        assert dto.max_buffer_size == 5000
        assert dto.min_batch_size == 10
        assert dto.max_batch_size == 50
        assert dto.batch_timeout_ms == 200
        assert dto.high_watermark == 0.9
        assert dto.low_watermark == 0.2
        assert dto.max_processing_time_ms == 2000
        assert dto.max_concurrent_batches == 3
        assert dto.adaptive_scaling_enabled is True
        assert dto.enable_quality_monitoring is True
        assert dto.quality_check_interval_ms == 10000
        assert dto.drift_detection_enabled is True
        assert dto.enable_result_buffering is True
        assert dto.result_buffer_size == 2000
        assert dto.enable_metrics_collection is True

    def test_default_values(self):
        """Test default values."""
        dto = StreamingConfigurationDTO()

        assert dto.strategy == "adaptive_batch"
        assert dto.backpressure_strategy == "adaptive_sampling"
        assert dto.mode == "continuous"
        assert dto.max_buffer_size == 10000
        assert dto.min_batch_size == 1
        assert dto.max_batch_size == 100
        assert dto.batch_timeout_ms == 100
        assert dto.high_watermark == 0.8
        assert dto.low_watermark == 0.3
        assert dto.max_processing_time_ms == 1000
        assert dto.max_concurrent_batches == 5
        assert dto.adaptive_scaling_enabled is True
        assert dto.enable_quality_monitoring is True
        assert dto.quality_check_interval_ms == 5000
        assert dto.drift_detection_enabled is True
        assert dto.enable_result_buffering is True
        assert dto.result_buffer_size == 1000
        assert dto.enable_metrics_collection is True

    def test_strategy_validation(self):
        """Test strategy validation."""
        # Valid strategies
        valid_strategies = [
            "real_time",
            "micro_batch",
            "adaptive_batch",
            "windowed",
            "ensemble_stream",
        ]

        for strategy in valid_strategies:
            dto = StreamingConfigurationDTO(strategy=strategy)
            assert dto.strategy == strategy

        # Invalid strategy
        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(strategy="invalid_strategy")

    def test_backpressure_strategy_validation(self):
        """Test backpressure strategy validation."""
        # Valid backpressure strategies
        valid_strategies = [
            "drop_oldest",
            "drop_newest",
            "adaptive_sampling",
            "circuit_breaker",
            "elastic_scaling",
        ]

        for strategy in valid_strategies:
            dto = StreamingConfigurationDTO(backpressure_strategy=strategy)
            assert dto.backpressure_strategy == strategy

        # Invalid backpressure strategy
        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(backpressure_strategy="invalid_strategy")

    def test_mode_validation(self):
        """Test mode validation."""
        # Valid modes
        valid_modes = ["continuous", "burst", "scheduled", "event_driven"]

        for mode in valid_modes:
            dto = StreamingConfigurationDTO(mode=mode)
            assert dto.mode == mode

        # Invalid mode
        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(mode="invalid_mode")

    def test_range_validations(self):
        """Test range validations."""
        # Valid ranges
        dto = StreamingConfigurationDTO(
            max_buffer_size=1000,
            min_batch_size=5,
            max_batch_size=50,
            batch_timeout_ms=500,
            high_watermark=0.7,
            low_watermark=0.2,
            max_processing_time_ms=2000,
            max_concurrent_batches=10,
            quality_check_interval_ms=3000,
            result_buffer_size=500,
        )

        assert dto.max_buffer_size == 1000
        assert dto.min_batch_size == 5
        assert dto.max_batch_size == 50
        assert dto.batch_timeout_ms == 500
        assert dto.high_watermark == 0.7
        assert dto.low_watermark == 0.2
        assert dto.max_processing_time_ms == 2000
        assert dto.max_concurrent_batches == 10
        assert dto.quality_check_interval_ms == 3000
        assert dto.result_buffer_size == 500

        # Invalid ranges
        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(max_buffer_size=50)  # Too small

        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(min_batch_size=0)  # Too small

        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(batch_timeout_ms=0)  # Too small

        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(high_watermark=1.1)  # Too large

        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(low_watermark=-0.1)  # Too small

    def test_batch_size_consistency(self):
        """Test batch size consistency validation."""
        # Valid: max_batch_size >= min_batch_size
        dto = StreamingConfigurationDTO(
            min_batch_size=10,
            max_batch_size=20,
        )
        assert dto.min_batch_size == 10
        assert dto.max_batch_size == 20

        # Invalid: max_batch_size < min_batch_size
        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(
                min_batch_size=20,
                max_batch_size=10,
            )

    def test_watermark_consistency(self):
        """Test watermark consistency validation."""
        # Valid: high_watermark > low_watermark
        dto = StreamingConfigurationDTO(
            low_watermark=0.3,
            high_watermark=0.8,
        )
        assert dto.low_watermark == 0.3
        assert dto.high_watermark == 0.8

        # Invalid: high_watermark <= low_watermark
        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(
                low_watermark=0.8,
                high_watermark=0.3,
            )

        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(
                low_watermark=0.5,
                high_watermark=0.5,
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            StreamingConfigurationDTO(unknown_field="value")


class TestStreamingSampleDTO:
    """Test suite for StreamingSampleDTO."""

    def test_valid_creation_with_array(self):
        """Test creating a valid streaming sample DTO with array data."""
        dto = StreamingSampleDTO(
            id="sample_001",
            data=[1.0, 2.5, 3.8, 4.2],
            timestamp=1234567890.0,
            metadata={"sensor_id": "sensor_001"},
            priority=5,
        )

        assert dto.id == "sample_001"
        assert dto.data == [1.0, 2.5, 3.8, 4.2]
        assert dto.timestamp == 1234567890.0
        assert dto.metadata == {"sensor_id": "sensor_001"}
        assert dto.priority == 5

    def test_valid_creation_with_dict(self):
        """Test creating a valid streaming sample DTO with dictionary data."""
        dto = StreamingSampleDTO(
            id="sample_002",
            data={"temperature": 25.5, "pressure": 1013.25, "humidity": 60.0},
            timestamp=1234567890.0,
            metadata={"location": "building_a"},
            priority=3,
        )

        assert dto.id == "sample_002"
        assert dto.data == {"temperature": 25.5, "pressure": 1013.25, "humidity": 60.0}
        assert dto.timestamp == 1234567890.0
        assert dto.metadata == {"location": "building_a"}
        assert dto.priority == 3

    def test_default_values(self):
        """Test default values."""
        dto = StreamingSampleDTO(data=[1.0, 2.0, 3.0])

        assert dto.id is None
        assert dto.data == [1.0, 2.0, 3.0]
        assert dto.timestamp is None
        assert dto.metadata == {}
        assert dto.priority == 0

    def test_data_validation_array(self):
        """Test data validation for array format."""
        # Valid array
        dto = StreamingSampleDTO(data=[1.0, 2, 3.5])
        assert dto.data == [1.0, 2, 3.5]

        # Invalid: empty array
        with pytest.raises(ValidationError):
            StreamingSampleDTO(data=[])

        # Invalid: non-numeric elements
        with pytest.raises(ValidationError):
            StreamingSampleDTO(data=[1.0, "invalid", 3.0])

    def test_data_validation_dict(self):
        """Test data validation for dictionary format."""
        # Valid dictionary
        dto = StreamingSampleDTO(data={"feature1": 1.0, "feature2": 2.5})
        assert dto.data == {"feature1": 1.0, "feature2": 2.5}

        # Invalid: empty dictionary
        with pytest.raises(ValidationError):
            StreamingSampleDTO(data={})

        # Invalid: non-numeric values
        with pytest.raises(ValidationError):
            StreamingSampleDTO(data={"feature1": 1.0, "feature2": "invalid"})

    def test_priority_validation(self):
        """Test priority validation."""
        # Valid priorities
        for priority in range(11):  # 0-10
            dto = StreamingSampleDTO(data=[1.0], priority=priority)
            assert dto.priority == priority

        # Invalid: negative priority
        with pytest.raises(ValidationError):
            StreamingSampleDTO(data=[1.0], priority=-1)

        # Invalid: priority too high
        with pytest.raises(ValidationError):
            StreamingSampleDTO(data=[1.0], priority=11)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            StreamingSampleDTO(
                data=[1.0, 2.0],
                unknown_field="value",
            )


class TestStreamingRequestDTO:
    """Test suite for StreamingRequestDTO."""

    def test_valid_creation(self):
        """Test creating a valid streaming request DTO."""
        configuration = StreamingConfigurationDTO(
            strategy="real_time",
            max_buffer_size=5000,
        )

        dto = StreamingRequestDTO(
            detector_id="detector_001",
            configuration=configuration,
            enable_ensemble=True,
            ensemble_detector_ids=["detector_002", "detector_003"],
            callback_settings={"webhook_url": "https://example.com/webhook"},
        )

        assert dto.detector_id == "detector_001"
        assert dto.configuration.strategy == "real_time"
        assert dto.enable_ensemble is True
        assert dto.ensemble_detector_ids == ["detector_002", "detector_003"]
        assert dto.callback_settings == {"webhook_url": "https://example.com/webhook"}

    def test_default_values(self):
        """Test default values."""
        configuration = StreamingConfigurationDTO()

        dto = StreamingRequestDTO(
            detector_id="detector_001",
            configuration=configuration,
        )

        assert dto.detector_id == "detector_001"
        assert dto.enable_ensemble is False
        assert dto.ensemble_detector_ids == []
        assert dto.callback_settings == {}

    def test_ensemble_validation(self):
        """Test ensemble validation."""
        configuration = StreamingConfigurationDTO()

        # Valid: ensemble enabled with detector IDs
        dto = StreamingRequestDTO(
            detector_id="detector_001",
            configuration=configuration,
            enable_ensemble=True,
            ensemble_detector_ids=["detector_002", "detector_003"],
        )

        assert dto.enable_ensemble is True
        assert dto.ensemble_detector_ids == ["detector_002", "detector_003"]

        # Invalid: ensemble enabled without detector IDs
        with pytest.raises(ValidationError):
            StreamingRequestDTO(
                detector_id="detector_001",
                configuration=configuration,
                enable_ensemble=True,
                ensemble_detector_ids=[],
            )

        # Valid: ensemble disabled with empty detector IDs
        dto_no_ensemble = StreamingRequestDTO(
            detector_id="detector_001",
            configuration=configuration,
            enable_ensemble=False,
            ensemble_detector_ids=[],
        )

        assert dto_no_ensemble.enable_ensemble is False
        assert dto_no_ensemble.ensemble_detector_ids == []

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        configuration = StreamingConfigurationDTO()

        with pytest.raises(ValidationError):
            StreamingRequestDTO(
                detector_id="detector_001",
                configuration=configuration,
                unknown_field="value",
            )


class TestStreamingResultDTO:
    """Test suite for StreamingResultDTO."""

    def test_valid_creation(self):
        """Test creating a valid streaming result DTO."""
        dto = StreamingResultDTO(
            sample_id="sample_001",
            prediction=1,
            anomaly_score=0.85,
            confidence=0.92,
            processing_time=0.005,
            detector_id="detector_001",
            timestamp=1234567890.0,
            metadata={"model_version": "v1.2.3"},
            quality_indicators={"stability": 0.8, "reliability": 0.9},
        )

        assert dto.sample_id == "sample_001"
        assert dto.prediction == 1
        assert dto.anomaly_score == 0.85
        assert dto.confidence == 0.92
        assert dto.processing_time == 0.005
        assert dto.detector_id == "detector_001"
        assert dto.timestamp == 1234567890.0
        assert dto.metadata == {"model_version": "v1.2.3"}
        assert dto.quality_indicators == {"stability": 0.8, "reliability": 0.9}

    def test_default_values(self):
        """Test default values."""
        dto = StreamingResultDTO(
            sample_id="sample_002",
            prediction=0,
            anomaly_score=0.3,
            confidence=0.7,
            processing_time=0.002,
            detector_id="detector_002",
            timestamp=1234567890.0,
        )

        assert dto.sample_id == "sample_002"
        assert dto.prediction == 0
        assert dto.anomaly_score == 0.3
        assert dto.confidence == 0.7
        assert dto.processing_time == 0.002
        assert dto.detector_id == "detector_002"
        assert dto.timestamp == 1234567890.0
        assert dto.metadata == {}
        assert dto.quality_indicators == {}

    def test_prediction_validation(self):
        """Test prediction validation."""
        # Valid predictions
        for prediction in [0, 1]:
            dto = StreamingResultDTO(
                sample_id="sample_test",
                prediction=prediction,
                anomaly_score=0.5,
                confidence=0.8,
                processing_time=0.001,
                detector_id="detector_test",
                timestamp=1234567890.0,
            )
            assert dto.prediction == prediction

        # Invalid: negative prediction
        with pytest.raises(ValidationError):
            StreamingResultDTO(
                sample_id="sample_test",
                prediction=-1,
                anomaly_score=0.5,
                confidence=0.8,
                processing_time=0.001,
                detector_id="detector_test",
                timestamp=1234567890.0,
            )

        # Invalid: prediction > 1
        with pytest.raises(ValidationError):
            StreamingResultDTO(
                sample_id="sample_test",
                prediction=2,
                anomaly_score=0.5,
                confidence=0.8,
                processing_time=0.001,
                detector_id="detector_test",
                timestamp=1234567890.0,
            )

    def test_score_validation(self):
        """Test score validation."""
        # Valid scores
        valid_scores = [0.0, 0.5, 1.0, 0.99, 0.01]
        for score in valid_scores:
            dto = StreamingResultDTO(
                sample_id="sample_test",
                prediction=1,
                anomaly_score=score,
                confidence=score,
                processing_time=0.001,
                detector_id="detector_test",
                timestamp=1234567890.0,
            )
            assert dto.anomaly_score == score
            assert dto.confidence == score

        # Invalid: negative score
        with pytest.raises(ValidationError):
            StreamingResultDTO(
                sample_id="sample_test",
                prediction=1,
                anomaly_score=-0.1,
                confidence=0.8,
                processing_time=0.001,
                detector_id="detector_test",
                timestamp=1234567890.0,
            )

        # Invalid: score > 1
        with pytest.raises(ValidationError):
            StreamingResultDTO(
                sample_id="sample_test",
                prediction=1,
                anomaly_score=1.1,
                confidence=0.8,
                processing_time=0.001,
                detector_id="detector_test",
                timestamp=1234567890.0,
            )

    def test_processing_time_validation(self):
        """Test processing time validation."""
        # Valid processing times
        valid_times = [0.0, 0.001, 1.0, 10.0]
        for time in valid_times:
            dto = StreamingResultDTO(
                sample_id="sample_test",
                prediction=1,
                anomaly_score=0.5,
                confidence=0.8,
                processing_time=time,
                detector_id="detector_test",
                timestamp=1234567890.0,
            )
            assert dto.processing_time == time

        # Invalid: negative processing time
        with pytest.raises(ValidationError):
            StreamingResultDTO(
                sample_id="sample_test",
                prediction=1,
                anomaly_score=0.5,
                confidence=0.8,
                processing_time=-0.001,
                detector_id="detector_test",
                timestamp=1234567890.0,
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            StreamingResultDTO(
                sample_id="sample_test",
                prediction=1,
                anomaly_score=0.5,
                confidence=0.8,
                processing_time=0.001,
                detector_id="detector_test",
                timestamp=1234567890.0,
                unknown_field="value",
            )


class TestStreamingControlDTO:
    """Test suite for StreamingControlDTO."""

    def test_valid_creation(self):
        """Test creating a valid streaming control DTO."""
        dto = StreamingControlDTO(
            action="configure",
            stream_id="stream_001",
            configuration_updates={"max_batch_size": 200},
            force=True,
        )

        assert dto.action == "configure"
        assert dto.stream_id == "stream_001"
        assert dto.configuration_updates == {"max_batch_size": 200}
        assert dto.force is True

    def test_default_values(self):
        """Test default values."""
        dto = StreamingControlDTO(action="start")

        assert dto.action == "start"
        assert dto.stream_id is None
        assert dto.configuration_updates is None
        assert dto.force is False

    def test_action_validation(self):
        """Test action validation."""
        # Valid actions
        valid_actions = ["start", "stop", "pause", "resume", "configure"]
        for action in valid_actions:
            dto = StreamingControlDTO(action=action)
            assert dto.action == action

        # Invalid action
        with pytest.raises(ValidationError):
            StreamingControlDTO(action="invalid_action")

    def test_stream_id_requirement(self):
        """Test stream_id requirement validation."""
        # Actions that require stream_id
        required_actions = ["stop", "pause", "resume", "configure"]
        for action in required_actions:
            # Valid: action with stream_id
            dto = StreamingControlDTO(action=action, stream_id="stream_001")
            assert dto.action == action
            assert dto.stream_id == "stream_001"

            # Invalid: action without stream_id
            with pytest.raises(ValidationError):
                StreamingControlDTO(action=action)

        # Action that doesn't require stream_id
        dto_start = StreamingControlDTO(action="start")
        assert dto_start.action == "start"
        assert dto_start.stream_id is None

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            StreamingControlDTO(
                action="start",
                unknown_field="value",
            )


class TestStreamingHealthCheckDTO:
    """Test suite for StreamingHealthCheckDTO."""

    def test_valid_creation(self):
        """Test creating a valid streaming health check DTO."""
        timestamp = datetime.now()
        error_rates = {"connection": 0.01, "processing": 0.02}
        resource_health = {"cpu": "healthy", "memory": "healthy", "disk": "warning"}
        performance_indicators = {"avg_latency": 0.005, "throughput": 1000.0}
        recommendations = ["Increase buffer size", "Add more processing nodes"]
        alerts = [{"type": "warning", "message": "High memory usage"}]

        dto = StreamingHealthCheckDTO(
            status="healthy",
            timestamp=timestamp,
            active_streams_count=5,
            total_throughput=5000.0,
            error_rates=error_rates,
            resource_health=resource_health,
            performance_indicators=performance_indicators,
            recommendations=recommendations,
            alerts=alerts,
        )

        assert dto.status == "healthy"
        assert dto.timestamp == timestamp
        assert dto.active_streams_count == 5
        assert dto.total_throughput == 5000.0
        assert dto.error_rates == error_rates
        assert dto.resource_health == resource_health
        assert dto.performance_indicators == performance_indicators
        assert dto.recommendations == recommendations
        assert dto.alerts == alerts

    def test_default_values(self):
        """Test default values."""
        dto = StreamingHealthCheckDTO(
            status="healthy",
            active_streams_count=3,
            total_throughput=1500.0,
            error_rates={"connection": 0.0},
            resource_health={"cpu": "healthy"},
            performance_indicators={"throughput": 1500.0},
        )

        assert dto.status == "healthy"
        assert isinstance(dto.timestamp, datetime)
        assert dto.active_streams_count == 3
        assert dto.total_throughput == 1500.0
        assert dto.error_rates == {"connection": 0.0}
        assert dto.resource_health == {"cpu": "healthy"}
        assert dto.performance_indicators == {"throughput": 1500.0}
        assert dto.recommendations == []
        assert dto.alerts == []

    def test_active_streams_count_validation(self):
        """Test active_streams_count validation."""
        # Valid counts
        valid_counts = [0, 1, 5, 100]
        for count in valid_counts:
            dto = StreamingHealthCheckDTO(
                status="healthy",
                active_streams_count=count,
                total_throughput=1000.0,
                error_rates={},
                resource_health={},
                performance_indicators={},
            )
            assert dto.active_streams_count == count

        # Invalid: negative count
        with pytest.raises(ValidationError):
            StreamingHealthCheckDTO(
                status="healthy",
                active_streams_count=-1,
                total_throughput=1000.0,
                error_rates={},
                resource_health={},
                performance_indicators={},
            )

    def test_total_throughput_validation(self):
        """Test total_throughput validation."""
        # Valid throughput
        valid_throughputs = [0.0, 100.0, 1000.0, 10000.0]
        for throughput in valid_throughputs:
            dto = StreamingHealthCheckDTO(
                status="healthy",
                active_streams_count=1,
                total_throughput=throughput,
                error_rates={},
                resource_health={},
                performance_indicators={},
            )
            assert dto.total_throughput == throughput

        # Invalid: negative throughput
        with pytest.raises(ValidationError):
            StreamingHealthCheckDTO(
                status="healthy",
                active_streams_count=1,
                total_throughput=-100.0,
                error_rates={},
                resource_health={},
                performance_indicators={},
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            StreamingHealthCheckDTO(
                status="healthy",
                active_streams_count=1,
                total_throughput=1000.0,
                error_rates={},
                resource_health={},
                performance_indicators={},
                unknown_field="value",
            )


class TestStreamingDTOIntegration:
    """Test integration scenarios for streaming DTOs."""

    def test_complete_streaming_workflow(self):
        """Test complete streaming workflow with all DTOs."""
        # Create streaming configuration
        configuration = StreamingConfigurationDTO(
            strategy="adaptive_batch",
            mode="continuous",
            max_buffer_size=5000,
            min_batch_size=10,
            max_batch_size=50,
            batch_timeout_ms=200,
            high_watermark=0.85,
            low_watermark=0.25,
            enable_quality_monitoring=True,
            drift_detection_enabled=True,
        )

        # Create streaming request
        request = StreamingRequestDTO(
            detector_id="detector_001",
            configuration=configuration,
            enable_ensemble=True,
            ensemble_detector_ids=["detector_002", "detector_003"],
            callback_settings={"webhook_url": "https://example.com/webhook"},
        )

        # Create streaming response
        response = StreamingResponseDTO(
            success=True,
            stream_id="stream_001",
            configuration=configuration,
            performance_metrics={
                "estimated_throughput": 1000.0,
                "initial_latency": 0.005,
            },
            estimated_throughput=1000.0,
            resource_allocation={
                "cpu_cores": 4,
                "memory_mb": 2048,
                "buffer_size": 5000,
            },
        )

        # Create streaming samples
        samples = [
            StreamingSampleDTO(
                id=f"sample_{i}",
                data={"temperature": 20.0 + i, "pressure": 1013.0 + i * 0.1},
                timestamp=1234567890.0 + i,
                priority=1 if i % 10 == 0 else 0,
            )
            for i in range(20)
        ]

        # Create streaming results
        results = [
            StreamingResultDTO(
                sample_id=f"sample_{i}",
                prediction=1 if i % 15 == 0 else 0,
                anomaly_score=0.9 if i % 15 == 0 else 0.1 + (i % 10) * 0.05,
                confidence=0.85 + (i % 10) * 0.01,
                processing_time=0.001 + (i % 5) * 0.0005,
                detector_id="detector_001",
                timestamp=1234567890.0 + i,
                metadata={"model_version": "v1.2.3"},
                quality_indicators={"stability": 0.8, "reliability": 0.9},
            )
            for i in range(20)
        ]

        # Create batch result
        batch_result = StreamingBatchResultDTO(
            stream_id="stream_001",
            batch_id="batch_001",
            results=results,
            batch_processing_time=0.025,
            batch_size=20,
            anomaly_count=1,  # Only sample_0 is anomaly (i % 15 == 0)
            quality_metrics={"overall_quality": 0.92, "consistency": 0.88},
            timestamp=1234567890.0 + 20,
        )

        # Create streaming metrics
        metrics = StreamingMetricsDTO(
            stream_id="stream_001",
            samples_processed=20,
            samples_dropped=0,
            anomalies_detected=1,
            average_processing_time=0.002,
            current_buffer_size=100,
            buffer_utilization=0.02,
            throughput_per_second=800.0,
            backpressure_active=False,
            circuit_breaker_open=False,
            error_rate=0.0,
            quality_score=0.92,
            last_updated=1234567890.0 + 25,
            latency_percentiles={"p50": 0.002, "p95": 0.004, "p99": 0.006},
            detector_performance={"detector_001": 0.95},
            resource_utilization={"cpu": 0.3, "memory": 0.4},
        )

        # Create health check
        health_check = StreamingHealthCheckDTO(
            status="healthy",
            active_streams_count=1,
            total_throughput=800.0,
            error_rates={"connection": 0.0, "processing": 0.0},
            resource_health={"cpu": "healthy", "memory": "healthy", "disk": "healthy"},
            performance_indicators={"avg_latency": 0.002, "throughput": 800.0},
            recommendations=[],
            alerts=[],
        )

        # Verify workflow consistency
        assert request.detector_id == "detector_001"
        assert request.configuration.strategy == "adaptive_batch"
        assert request.enable_ensemble is True
        assert len(request.ensemble_detector_ids) == 2

        assert response.success is True
        assert response.stream_id == "stream_001"
        assert response.configuration.strategy == "adaptive_batch"
        assert response.estimated_throughput == 1000.0

        assert len(samples) == 20
        assert all(sample.data["temperature"] >= 20.0 for sample in samples)
        assert samples[0].priority == 1  # Every 10th sample has priority 1

        assert len(results) == 20
        assert sum(result.prediction for result in results) == 1  # Only one anomaly
        assert all(0.0 <= result.anomaly_score <= 1.0 for result in results)
        assert all(0.0 <= result.confidence <= 1.0 for result in results)

        assert batch_result.stream_id == "stream_001"
        assert batch_result.batch_size == 20
        assert batch_result.anomaly_count == 1
        assert len(batch_result.results) == 20

        assert metrics.stream_id == "stream_001"
        assert metrics.samples_processed == 20
        assert metrics.anomalies_detected == 1
        assert metrics.samples_dropped == 0
        assert metrics.error_rate == 0.0

        assert health_check.status == "healthy"
        assert health_check.active_streams_count == 1
        assert health_check.total_throughput == 800.0
        assert len(health_check.recommendations) == 0
        assert len(health_check.alerts) == 0

    def test_streaming_control_operations(self):
        """Test streaming control operations."""
        # Start streaming
        start_control = StreamingControlDTO(action="start")
        assert start_control.action == "start"
        assert start_control.stream_id is None

        # Pause streaming
        pause_control = StreamingControlDTO(
            action="pause",
            stream_id="stream_001",
        )
        assert pause_control.action == "pause"
        assert pause_control.stream_id == "stream_001"

        # Resume streaming
        resume_control = StreamingControlDTO(
            action="resume",
            stream_id="stream_001",
        )
        assert resume_control.action == "resume"
        assert resume_control.stream_id == "stream_001"

        # Configure streaming
        configure_control = StreamingControlDTO(
            action="configure",
            stream_id="stream_001",
            configuration_updates={
                "max_batch_size": 200,
                "batch_timeout_ms": 300,
                "enable_quality_monitoring": False,
            },
        )
        assert configure_control.action == "configure"
        assert configure_control.stream_id == "stream_001"
        assert configure_control.configuration_updates["max_batch_size"] == 200
        assert configure_control.configuration_updates["batch_timeout_ms"] == 300
        assert (
            configure_control.configuration_updates["enable_quality_monitoring"]
            is False
        )

        # Stop streaming
        stop_control = StreamingControlDTO(
            action="stop",
            stream_id="stream_001",
            force=True,
        )
        assert stop_control.action == "stop"
        assert stop_control.stream_id == "stream_001"
        assert stop_control.force is True

    def test_error_handling_scenario(self):
        """Test error handling scenario."""
        # Create error condition
        error_dto = StreamErrorDTO(
            stream_id="stream_001",
            error_code="BUFFER_OVERFLOW",
            error_message="Buffer overflow detected, dropping incoming samples",
            timestamp=datetime.now(),
            severity="warning",
        )

        # Create degraded health check
        degraded_health = StreamingHealthCheckDTO(
            status="degraded",
            active_streams_count=1,
            total_throughput=500.0,  # Reduced throughput
            error_rates={
                "connection": 0.0,
                "processing": 0.05,
            },  # Some processing errors
            resource_health={"cpu": "healthy", "memory": "warning", "disk": "healthy"},
            performance_indicators={"avg_latency": 0.008, "throughput": 500.0},
            recommendations=[
                "Increase buffer size to handle load spikes",
                "Consider adding more processing capacity",
            ],
            alerts=[
                {
                    "type": "warning",
                    "message": "Memory usage above 80%",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
        )

        # Create metrics showing degraded performance
        degraded_metrics = StreamingMetricsDTO(
            stream_id="stream_001",
            samples_processed=1000,
            samples_dropped=50,  # Some samples dropped
            anomalies_detected=25,
            average_processing_time=0.008,  # Increased processing time
            current_buffer_size=4500,  # Near buffer limit
            buffer_utilization=0.9,  # High buffer utilization
            throughput_per_second=500.0,  # Reduced throughput
            backpressure_active=True,  # Backpressure activated
            circuit_breaker_open=False,
            error_rate=0.05,  # Some errors
            quality_score=0.85,  # Reduced quality
            last_updated=1234567890.0,
            latency_percentiles={"p50": 0.005, "p95": 0.015, "p99": 0.025},
            detector_performance={"detector_001": 0.90},
            resource_utilization={"cpu": 0.7, "memory": 0.85},
        )

        # Verify error scenario
        assert error_dto.error_code == "BUFFER_OVERFLOW"
        assert error_dto.severity == "warning"
        assert "overflow" in error_dto.error_message.lower()

        assert degraded_health.status == "degraded"
        assert degraded_health.total_throughput == 500.0
        assert degraded_health.error_rates["processing"] == 0.05
        assert degraded_health.resource_health["memory"] == "warning"
        assert len(degraded_health.recommendations) == 2
        assert len(degraded_health.alerts) == 1

        assert degraded_metrics.samples_dropped == 50
        assert degraded_metrics.buffer_utilization == 0.9
        assert degraded_metrics.backpressure_active is True
        assert degraded_metrics.error_rate == 0.05
        assert degraded_metrics.quality_score == 0.85

    def test_serialization_integration(self):
        """Test serialization integration across streaming DTOs."""
        # Create complex streaming configuration
        config = StreamingConfigurationDTO(
            strategy="ensemble_stream",
            backpressure_strategy="elastic_scaling",
            mode="event_driven",
            max_buffer_size=8192,
            min_batch_size=5,
            max_batch_size=100,
            batch_timeout_ms=500,
            high_watermark=0.9,
            low_watermark=0.1,
            max_processing_time_ms=3000,
            max_concurrent_batches=10,
            adaptive_scaling_enabled=True,
            enable_quality_monitoring=True,
            quality_check_interval_ms=2000,
            drift_detection_enabled=True,
            enable_result_buffering=True,
            result_buffer_size=5000,
            enable_metrics_collection=True,
        )

        # Serialize and verify
        config_dict = config.model_dump()
        assert config_dict["strategy"] == "ensemble_stream"
        assert config_dict["backpressure_strategy"] == "elastic_scaling"
        assert config_dict["mode"] == "event_driven"
        assert config_dict["max_buffer_size"] == 8192
        assert config_dict["high_watermark"] == 0.9
        assert config_dict["low_watermark"] == 0.1
        assert config_dict["adaptive_scaling_enabled"] is True

        # Deserialize and verify integrity
        restored_config = StreamingConfigurationDTO.model_validate(config_dict)
        assert restored_config.strategy == config.strategy
        assert restored_config.backpressure_strategy == config.backpressure_strategy
        assert restored_config.mode == config.mode
        assert restored_config.max_buffer_size == config.max_buffer_size
        assert restored_config.min_batch_size == config.min_batch_size
        assert restored_config.max_batch_size == config.max_batch_size
        assert restored_config.batch_timeout_ms == config.batch_timeout_ms
        assert restored_config.high_watermark == config.high_watermark
        assert restored_config.low_watermark == config.low_watermark
        assert (
            restored_config.adaptive_scaling_enabled == config.adaptive_scaling_enabled
        )
        assert (
            restored_config.enable_quality_monitoring
            == config.enable_quality_monitoring
        )
        assert restored_config.drift_detection_enabled == config.drift_detection_enabled
