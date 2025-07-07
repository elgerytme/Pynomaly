"""Comprehensive test suite for streaming detection use case."""

import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.application.use_cases.streaming_detection_use_case import (
    BackpressureStrategy,
    StreamingConfiguration,
    StreamingDetectionUseCase,
    StreamingMetrics,
    StreamingMode,
    StreamingRequest,
    StreamingResponse,
    StreamingResult,
    StreamingSample,
    StreamingStrategy,
)
from pynomaly.domain.entities import Detector
from pynomaly.domain.exceptions import ValidationError


class TestStreamingDetectionUseCase:
    """Test suite for streaming detection use case functionality."""

    @pytest.fixture
    def mock_repositories(self):
        """Mock repositories for testing."""
        detector_repo = Mock()
        adapter_registry = Mock()
        ensemble_service = Mock()
        return detector_repo, adapter_registry, ensemble_service

    @pytest.fixture
    def streaming_use_case(self, mock_repositories):
        """Create streaming detection use case with mocked dependencies."""
        detector_repo, adapter_registry, ensemble_service = mock_repositories

        use_case = StreamingDetectionUseCase(
            detector_repository=detector_repo,
            adapter_registry=adapter_registry,
            ensemble_service=ensemble_service,
            enable_distributed_processing=True,
            max_concurrent_streams=5,
        )

        return use_case

    @pytest.fixture
    def sample_detector(self):
        """Create sample detector for testing."""
        detector = Mock(spec=Detector)
        detector.id = "test_detector"
        detector.algorithm = "IsolationForest"
        detector.is_fitted = True
        detector.model = Mock()
        return detector

    @pytest.fixture
    def streaming_config(self):
        """Create streaming configuration for testing."""
        return StreamingConfiguration(
            strategy=StreamingStrategy.ADAPTIVE_BATCH,
            backpressure_strategy=BackpressureStrategy.ADAPTIVE_SAMPLING,
            mode=StreamingMode.CONTINUOUS,
            max_buffer_size=1000,
            min_batch_size=5,
            max_batch_size=50,
            batch_timeout_ms=100,
        )

    @pytest.fixture
    def streaming_request(self, sample_detector, streaming_config):
        """Create streaming request for testing."""
        return StreamingRequest(
            detector_id=sample_detector.id, configuration=streaming_config
        )

    @pytest.mark.asyncio
    async def test_start_streaming_success(
        self, streaming_use_case, streaming_request, sample_detector
    ):
        """Test successful streaming start."""
        # Setup mocks
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )

        # Execute
        response = await streaming_use_case.start_streaming(streaming_request)

        # Verify
        assert response.success is True
        assert response.stream_id != ""
        assert response.configuration is not None
        assert len(streaming_use_case._active_streams) == 1
        assert response.stream_id in streaming_use_case._stream_metrics

        # Cleanup
        await streaming_use_case.stop_streaming(response.stream_id)

    @pytest.mark.asyncio
    async def test_start_streaming_detector_not_found(
        self, streaming_use_case, streaming_request
    ):
        """Test streaming start with non-existent detector."""
        # Setup mocks
        streaming_use_case.detector_repository.get = AsyncMock(return_value=None)

        # Execute
        response = await streaming_use_case.start_streaming(streaming_request)

        # Verify
        assert response.success is False
        assert "not found" in response.error_message
        assert len(streaming_use_case._active_streams) == 0

    @pytest.mark.asyncio
    async def test_start_streaming_detector_not_fitted(
        self, streaming_use_case, streaming_request, sample_detector
    ):
        """Test streaming start with unfitted detector."""
        # Setup mocks
        sample_detector.is_fitted = False
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )

        # Execute
        response = await streaming_use_case.start_streaming(streaming_request)

        # Verify
        assert response.success is False
        assert "not fitted" in response.error_message

    @pytest.mark.asyncio
    async def test_start_streaming_max_concurrent_limit(
        self, streaming_use_case, streaming_request, sample_detector
    ):
        """Test maximum concurrent streams limit."""
        # Setup mocks
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )

        # Start maximum number of streams
        stream_ids = []
        for i in range(streaming_use_case.max_concurrent_streams):
            request = StreamingRequest(
                detector_id=f"detector_{i}",
                configuration=streaming_request.configuration,
            )
            response = await streaming_use_case.start_streaming(request)
            assert response.success is True
            stream_ids.append(response.stream_id)

        # Try to start one more (should fail)
        response = await streaming_use_case.start_streaming(streaming_request)
        assert response.success is False
        assert "Maximum concurrent streams" in response.error_message

        # Cleanup
        for stream_id in stream_ids:
            await streaming_use_case.stop_streaming(stream_id)

    @pytest.mark.asyncio
    async def test_stop_streaming_success(
        self, streaming_use_case, streaming_request, sample_detector
    ):
        """Test successful streaming stop."""
        # Setup and start stream
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )
        response = await streaming_use_case.start_streaming(streaming_request)
        stream_id = response.stream_id

        # Verify stream is active
        assert stream_id in streaming_use_case._active_streams

        # Stop stream
        success = await streaming_use_case.stop_streaming(stream_id)

        # Verify
        assert success is True
        assert stream_id not in streaming_use_case._active_streams
        assert (
            stream_id in streaming_use_case._stream_metrics
        )  # Metrics kept for analysis

    @pytest.mark.asyncio
    async def test_stop_streaming_not_found(self, streaming_use_case):
        """Test stopping non-existent stream."""
        success = await streaming_use_case.stop_streaming("non_existent_stream")
        assert success is False

    @pytest.mark.asyncio
    async def test_add_sample_success(
        self, streaming_use_case, streaming_request, sample_detector
    ):
        """Test successful sample addition."""
        # Setup and start stream
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )
        response = await streaming_use_case.start_streaming(streaming_request)
        stream_id = response.stream_id

        # Create sample
        sample = StreamingSample(
            data=np.array([1.0, 2.0, 3.0]), metadata={"source": "test"}
        )

        # Add sample
        success = await streaming_use_case.add_sample(stream_id, sample)

        # Verify
        assert success is True

        # Check buffer has sample
        stream_state = streaming_use_case._active_streams[stream_id]
        assert len(stream_state["buffer"]) == 1

        # Cleanup
        await streaming_use_case.stop_streaming(stream_id)

    @pytest.mark.asyncio
    async def test_add_sample_backpressure_drop_oldest(
        self, streaming_use_case, sample_detector
    ):
        """Test backpressure with drop oldest strategy."""
        # Create config with small buffer and drop oldest strategy
        config = StreamingConfiguration(
            max_buffer_size=3,
            high_watermark=0.7,  # Trigger at 70%
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
        )

        request = StreamingRequest(detector_id=sample_detector.id, configuration=config)

        # Setup and start stream
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )
        response = await streaming_use_case.start_streaming(request)
        stream_id = response.stream_id

        # Add samples to fill buffer beyond watermark
        samples = []
        for i in range(5):
            sample = StreamingSample(
                id=f"sample_{i}", data=np.array([float(i), float(i + 1), float(i + 2)])
            )
            samples.append(sample)
            await streaming_use_case.add_sample(stream_id, sample)

        # Check that buffer size is limited and backpressure is active
        stream_state = streaming_use_case._active_streams[stream_id]
        assert len(stream_state["buffer"]) <= config.max_buffer_size
        assert stream_state["backpressure_active"] is True

        # Check that samples were dropped
        metrics = streaming_use_case._stream_metrics[stream_id]
        assert metrics.samples_dropped > 0

        # Cleanup
        await streaming_use_case.stop_streaming(stream_id)

    @pytest.mark.asyncio
    async def test_add_sample_backpressure_drop_newest(
        self, streaming_use_case, sample_detector
    ):
        """Test backpressure with drop newest strategy."""
        # Create config with drop newest strategy
        config = StreamingConfiguration(
            max_buffer_size=3,
            high_watermark=0.7,
            backpressure_strategy=BackpressureStrategy.DROP_NEWEST,
        )

        request = StreamingRequest(detector_id=sample_detector.id, configuration=config)

        # Setup and start stream
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )
        response = await streaming_use_case.start_streaming(request)
        stream_id = response.stream_id

        # Fill buffer beyond watermark
        for i in range(5):
            sample = StreamingSample(id=f"sample_{i}", data=np.array([float(i)]))
            result = await streaming_use_case.add_sample(stream_id, sample)
            if i >= 3:  # Should start dropping
                assert result is False  # Sample should be dropped

        # Check metrics
        metrics = streaming_use_case._stream_metrics[stream_id]
        assert metrics.samples_dropped > 0

        # Cleanup
        await streaming_use_case.stop_streaming(stream_id)

    @pytest.mark.asyncio
    async def test_add_sample_circuit_breaker(
        self, streaming_use_case, sample_detector
    ):
        """Test backpressure with circuit breaker strategy."""
        # Create config with circuit breaker strategy
        config = StreamingConfiguration(
            max_buffer_size=3,
            high_watermark=0.7,
            backpressure_strategy=BackpressureStrategy.CIRCUIT_BREAKER,
        )

        request = StreamingRequest(detector_id=sample_detector.id, configuration=config)

        # Setup and start stream
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )
        response = await streaming_use_case.start_streaming(request)
        stream_id = response.stream_id

        # Fill buffer beyond watermark to trigger circuit breaker
        for i in range(5):
            sample = StreamingSample(data=np.array([float(i)]))
            await streaming_use_case.add_sample(stream_id, sample)

        # Check that circuit breaker is open
        stream_state = streaming_use_case._active_streams[stream_id]
        metrics = streaming_use_case._stream_metrics[stream_id]
        assert stream_state["circuit_breaker_open"] is True
        assert metrics.circuit_breaker_open is True

        # Cleanup
        await streaming_use_case.stop_streaming(stream_id)

    @pytest.mark.asyncio
    async def test_processing_batch_real_time_strategy(
        self, streaming_use_case, sample_detector
    ):
        """Test real-time processing strategy."""
        # Create config with real-time strategy
        config = StreamingConfiguration(
            strategy=StreamingStrategy.REAL_TIME, enable_result_buffering=True
        )

        request = StreamingRequest(detector_id=sample_detector.id, configuration=config)

        # Setup mocks
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )

        adapter = Mock()
        adapter.predict.return_value = (
            np.array([1]),  # prediction
            np.array([0.8]),  # score
        )
        streaming_use_case.adapter_registry.get_adapter.return_value = adapter

        # Start stream
        response = await streaming_use_case.start_streaming(request)
        stream_id = response.stream_id

        # Add sample
        sample = StreamingSample(data=np.array([1.0, 2.0, 3.0]))
        await streaming_use_case.add_sample(stream_id, sample)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Get results
        results = await streaming_use_case.get_results(stream_id)

        # Verify processing occurred
        assert len(results) > 0
        result = results[0]
        assert result.sample_id == sample.id
        assert result.prediction == 1
        assert result.anomaly_score == 0.8
        assert result.detector_id == sample_detector.id

        # Cleanup
        await streaming_use_case.stop_streaming(stream_id)

    @pytest.mark.asyncio
    async def test_processing_batch_adaptive_strategy(
        self, streaming_use_case, sample_detector
    ):
        """Test adaptive batch processing strategy."""
        # Create config with adaptive strategy
        config = StreamingConfiguration(
            strategy=StreamingStrategy.ADAPTIVE_BATCH,
            min_batch_size=2,
            max_batch_size=5,
            enable_result_buffering=True,
        )

        request = StreamingRequest(detector_id=sample_detector.id, configuration=config)

        # Setup mocks
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )

        adapter = Mock()
        adapter.predict.return_value = (
            np.array([0, 1]),  # predictions
            np.array([0.3, 0.7]),  # scores
        )
        streaming_use_case.adapter_registry.get_adapter.return_value = adapter

        # Start stream
        response = await streaming_use_case.start_streaming(request)
        stream_id = response.stream_id

        # Add multiple samples
        for i in range(3):
            sample = StreamingSample(
                data=np.array([float(i), float(i + 1), float(i + 2)])
            )
            await streaming_use_case.add_sample(stream_id, sample)

        # Wait for processing
        await asyncio.sleep(0.3)

        # Get results
        results = await streaming_use_case.get_results(stream_id)

        # Verify batch processing occurred
        assert len(results) > 0

        # Cleanup
        await streaming_use_case.stop_streaming(stream_id)

    @pytest.mark.asyncio
    async def test_get_stream_metrics(
        self, streaming_use_case, streaming_request, sample_detector
    ):
        """Test getting stream metrics."""
        # Setup and start stream
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )
        response = await streaming_use_case.start_streaming(streaming_request)
        stream_id = response.stream_id

        # Get metrics
        metrics = await streaming_use_case.get_stream_metrics(stream_id)

        # Verify
        assert metrics is not None
        assert metrics.stream_id == stream_id
        assert metrics.samples_processed >= 0
        assert metrics.samples_dropped >= 0
        assert metrics.anomalies_detected >= 0
        assert metrics.throughput_per_second >= 0.0

        # Cleanup
        await streaming_use_case.stop_streaming(stream_id)

    @pytest.mark.asyncio
    async def test_list_active_streams(
        self, streaming_use_case, streaming_request, sample_detector
    ):
        """Test listing active streams."""
        # Initially no streams
        streams = await streaming_use_case.list_active_streams()
        assert len(streams) == 0

        # Start streams
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )

        stream_ids = []
        for i in range(3):
            request = StreamingRequest(
                detector_id=f"detector_{i}",
                configuration=streaming_request.configuration,
            )
            response = await streaming_use_case.start_streaming(request)
            stream_ids.append(response.stream_id)

        # List active streams
        active_streams = await streaming_use_case.list_active_streams()
        assert len(active_streams) == 3
        for stream_id in stream_ids:
            assert stream_id in active_streams

        # Cleanup
        for stream_id in stream_ids:
            await streaming_use_case.stop_streaming(stream_id)

    @pytest.mark.asyncio
    async def test_sample_data_formats(self, streaming_use_case, sample_detector):
        """Test different sample data formats."""
        config = StreamingConfiguration(
            strategy=StreamingStrategy.REAL_TIME, enable_result_buffering=True
        )

        request = StreamingRequest(detector_id=sample_detector.id, configuration=config)

        # Setup mocks
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )

        adapter = Mock()
        adapter.predict.return_value = (np.array([0]), np.array([0.5]))
        streaming_use_case.adapter_registry.get_adapter.return_value = adapter

        # Start stream
        response = await streaming_use_case.start_streaming(request)
        stream_id = response.stream_id

        # Test numpy array format
        sample1 = StreamingSample(data=np.array([1.0, 2.0, 3.0]))
        success1 = await streaming_use_case.add_sample(stream_id, sample1)
        assert success1 is True

        # Test dictionary format
        sample2 = StreamingSample(
            data={"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}
        )
        success2 = await streaming_use_case.add_sample(stream_id, sample2)
        assert success2 is True

        # Wait for processing
        await asyncio.sleep(0.2)

        # Get results
        results = await streaming_use_case.get_results(stream_id)
        assert len(results) >= 2  # Should have processed both samples

        # Cleanup
        await streaming_use_case.stop_streaming(stream_id)

    @pytest.mark.asyncio
    async def test_ensemble_streaming_validation(
        self, streaming_use_case, sample_detector
    ):
        """Test ensemble streaming validation."""
        # Create ensemble request
        ensemble_detector = Mock(spec=Detector)
        ensemble_detector.id = "ensemble_detector"
        ensemble_detector.algorithm = "RandomForest"
        ensemble_detector.is_fitted = True

        request = StreamingRequest(
            detector_id=sample_detector.id,
            configuration=StreamingConfiguration(),
            enable_ensemble=True,
            ensemble_detector_ids=[ensemble_detector.id],
        )

        # Setup mocks for both detectors
        async def get_detector(detector_id):
            if detector_id == sample_detector.id:
                return sample_detector
            elif detector_id == ensemble_detector.id:
                return ensemble_detector
            return None

        streaming_use_case.detector_repository.get = get_detector

        # Test successful ensemble validation
        response = await streaming_use_case.start_streaming(request)
        assert response.success is True

        # Cleanup
        await streaming_use_case.stop_streaming(response.stream_id)

    @pytest.mark.asyncio
    async def test_ensemble_streaming_missing_detector(
        self, streaming_use_case, sample_detector
    ):
        """Test ensemble streaming with missing ensemble detector."""
        request = StreamingRequest(
            detector_id=sample_detector.id,
            configuration=StreamingConfiguration(),
            enable_ensemble=True,
            ensemble_detector_ids=["missing_detector"],
        )

        # Setup mocks - only return main detector
        streaming_use_case.detector_repository.get = AsyncMock(
            side_effect=lambda detector_id: (
                sample_detector if detector_id == sample_detector.id else None
            )
        )

        # Test ensemble validation failure
        response = await streaming_use_case.start_streaming(request)
        assert response.success is False
        assert "missing_detector" in response.error_message

    @pytest.mark.asyncio
    async def test_streaming_configuration_validation(
        self, streaming_use_case, sample_detector
    ):
        """Test streaming configuration validation."""
        # Test invalid batch size configuration
        invalid_config = StreamingConfiguration(
            min_batch_size=10, max_batch_size=5  # Invalid: min > max
        )

        request = StreamingRequest(
            detector_id=sample_detector.id, configuration=invalid_config
        )

        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )

        response = await streaming_use_case.start_streaming(request)
        assert response.success is False
        assert (
            "min_batch_size cannot be greater than max_batch_size"
            in response.error_message
        )

    @pytest.mark.asyncio
    async def test_streaming_configuration_validation_watermarks(
        self, streaming_use_case, sample_detector
    ):
        """Test streaming configuration watermark validation."""
        # Test invalid watermark configuration
        invalid_config = StreamingConfiguration(
            high_watermark=0.3, low_watermark=0.8  # Invalid: high < low
        )

        request = StreamingRequest(
            detector_id=sample_detector.id, configuration=invalid_config
        )

        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )

        response = await streaming_use_case.start_streaming(request)
        assert response.success is False
        assert (
            "high_watermark must be greater than low_watermark"
            in response.error_message
        )

    @pytest.mark.asyncio
    async def test_streaming_performance_metrics_update(
        self, streaming_use_case, sample_detector
    ):
        """Test streaming performance metrics updates."""
        config = StreamingConfiguration(
            strategy=StreamingStrategy.REAL_TIME,
            enable_metrics_collection=True,
            enable_result_buffering=True,
        )

        request = StreamingRequest(detector_id=sample_detector.id, configuration=config)

        # Setup mocks
        streaming_use_case.detector_repository.get = AsyncMock(
            return_value=sample_detector
        )

        adapter = Mock()
        adapter.predict.return_value = (np.array([1]), np.array([0.9]))  # anomaly
        streaming_use_case.adapter_registry.get_adapter.return_value = adapter

        # Start stream
        response = await streaming_use_case.start_streaming(request)
        stream_id = response.stream_id

        # Add samples
        for i in range(5):
            sample = StreamingSample(data=np.array([float(i)]))
            await streaming_use_case.add_sample(stream_id, sample)

        # Wait for processing
        await asyncio.sleep(0.3)

        # Check metrics were updated
        metrics = await streaming_use_case.get_stream_metrics(stream_id)
        assert metrics.samples_processed > 0
        assert metrics.anomalies_detected > 0
        assert metrics.average_processing_time > 0.0
        assert metrics.throughput_per_second > 0.0

        # Cleanup
        await streaming_use_case.stop_streaming(stream_id)

    def test_streaming_enums(self):
        """Test streaming enumeration values."""
        # Test StreamingStrategy
        strategies = list(StreamingStrategy)
        assert StreamingStrategy.REAL_TIME in strategies
        assert StreamingStrategy.MICRO_BATCH in strategies
        assert StreamingStrategy.ADAPTIVE_BATCH in strategies
        assert StreamingStrategy.WINDOWED in strategies
        assert StreamingStrategy.ENSEMBLE_STREAM in strategies

        # Test BackpressureStrategy
        backpressure_strategies = list(BackpressureStrategy)
        assert BackpressureStrategy.DROP_OLDEST in backpressure_strategies
        assert BackpressureStrategy.DROP_NEWEST in backpressure_strategies
        assert BackpressureStrategy.ADAPTIVE_SAMPLING in backpressure_strategies
        assert BackpressureStrategy.CIRCUIT_BREAKER in backpressure_strategies
        assert BackpressureStrategy.ELASTIC_SCALING in backpressure_strategies

        # Test StreamingMode
        modes = list(StreamingMode)
        assert StreamingMode.CONTINUOUS in modes
        assert StreamingMode.BURST in modes
        assert StreamingMode.SCHEDULED in modes
        assert StreamingMode.EVENT_DRIVEN in modes

    def test_streaming_sample_creation(self):
        """Test streaming sample creation and properties."""
        # Test with numpy data
        data = np.array([1.0, 2.0, 3.0])
        sample = StreamingSample(
            data=data,
            metadata={"source": "sensor_1", "location": "warehouse"},
            priority=1,
        )

        assert sample.id is not None
        assert len(sample.id) > 0
        assert np.array_equal(sample.data, data)
        assert sample.metadata["source"] == "sensor_1"
        assert sample.priority == 1
        assert sample.timestamp > 0

        # Test with dictionary data
        dict_data = {"temperature": 25.5, "humidity": 60.0, "pressure": 1013.25}
        sample2 = StreamingSample(data=dict_data)

        assert sample2.data == dict_data
        assert sample2.priority == 0  # default

    def test_streaming_result_creation(self):
        """Test streaming result creation and properties."""
        result = StreamingResult(
            sample_id="test_sample_123",
            prediction=1,
            anomaly_score=0.85,
            confidence=0.9,
            processing_time=0.002,
            detector_id="test_detector",
            metadata={"batch_id": "batch_001"},
        )

        assert result.sample_id == "test_sample_123"
        assert result.prediction == 1
        assert result.anomaly_score == 0.85
        assert result.confidence == 0.9
        assert result.processing_time == 0.002
        assert result.detector_id == "test_detector"
        assert result.metadata["batch_id"] == "batch_001"
        assert result.timestamp > 0

    def test_streaming_configuration_defaults(self):
        """Test streaming configuration default values."""
        config = StreamingConfiguration()

        assert config.strategy == StreamingStrategy.ADAPTIVE_BATCH
        assert config.backpressure_strategy == BackpressureStrategy.ADAPTIVE_SAMPLING
        assert config.mode == StreamingMode.CONTINUOUS
        assert config.max_buffer_size == 10000
        assert config.min_batch_size == 1
        assert config.max_batch_size == 100
        assert config.batch_timeout_ms == 100
        assert config.high_watermark == 0.8
        assert config.low_watermark == 0.3
        assert config.enable_quality_monitoring is True
        assert config.enable_result_buffering is True
        assert config.enable_metrics_collection is True

    def test_streaming_metrics_initialization(self):
        """Test streaming metrics initialization."""
        metrics = StreamingMetrics(stream_id="test_stream")

        assert metrics.stream_id == "test_stream"
        assert metrics.samples_processed == 0
        assert metrics.samples_dropped == 0
        assert metrics.anomalies_detected == 0
        assert metrics.average_processing_time == 0.0
        assert metrics.current_buffer_size == 0
        assert metrics.buffer_utilization == 0.0
        assert metrics.throughput_per_second == 0.0
        assert metrics.backpressure_active is False
        assert metrics.circuit_breaker_open is False
        assert metrics.error_rate == 0.0
        assert metrics.quality_score == 1.0
        assert metrics.last_updated > 0
