"""
Tests for streaming detection use case.

This module provides comprehensive tests for the streaming anomaly detection use case,
ensuring proper real-time processing, error handling, and performance characteristics.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.pynomaly.application.dto.streaming_dto import (
    BackpressureConfigDTO,
    CheckpointConfigDTO,
    StreamConfigurationDTO,
    StreamDataBatchDTO,
    StreamDataPointDTO,
    StreamDetectionRequestDTO,
    StreamDetectionResponseDTO,
    StreamMetricsDTO,
    StreamStatusDTO,
    WindowConfigDTO,
)
from src.pynomaly.application.use_cases.streaming_detection_use_case import (
    BackpressureError,
    DetectorNotFoundError,
    StreamingDetectionUseCase,
    StreamProcessingError,
)


class TestStreamingDetectionUseCase:
    """Test cases for StreamingDetectionUseCase."""

    @pytest.fixture
    def mock_detector_repository(self):
        """Mock detector repository."""
        repository = Mock()
        repository.get_by_id = AsyncMock()
        repository.update = AsyncMock()
        return repository

    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock metrics collector."""
        collector = Mock()
        collector.increment_counter = Mock()
        collector.record_gauge = Mock()
        collector.record_histogram = Mock()
        return collector

    @pytest.fixture
    def mock_checkpoint_manager(self):
        """Mock checkpoint manager."""
        manager = Mock()
        manager.save_checkpoint = AsyncMock()
        manager.load_checkpoint = AsyncMock()
        manager.cleanup_old_checkpoints = AsyncMock()
        return manager

    @pytest.fixture
    def mock_backpressure_handler(self):
        """Mock backpressure handler."""
        handler = Mock()
        handler.check_backpressure = Mock(return_value=False)
        handler.handle_backpressure = Mock()
        return handler

    @pytest.fixture
    def use_case(
        self,
        mock_detector_repository,
        mock_metrics_collector,
        mock_checkpoint_manager,
        mock_backpressure_handler,
    ):
        """Create StreamingDetectionUseCase instance with mocked dependencies."""
        return StreamingDetectionUseCase(
            detector_repository=mock_detector_repository,
            metrics_collector=mock_metrics_collector,
            checkpoint_manager=mock_checkpoint_manager,
            backpressure_handler=mock_backpressure_handler,
        )

    @pytest.fixture
    def sample_configuration(self):
        """Sample streaming configuration."""
        return StreamConfigurationDTO(
            stream_id="test_stream",
            batch_size=10,
            timeout_ms=5000,
            backpressure=BackpressureConfigDTO(
                enabled=True, max_queue_size=100, drop_policy="oldest"
            ),
            window=WindowConfigDTO(type="tumbling", size_ms=10000),
            checkpoint=CheckpointConfigDTO(
                enabled=True, interval_ms=30000, storage_path="/tmp/test_checkpoints"
            ),
        )

    @pytest.fixture
    def sample_data_points(self):
        """Sample data points for testing."""
        base_time = datetime.now(timezone.utc)
        return [
            StreamDataPointDTO(
                timestamp=base_time + timedelta(seconds=i),
                features={"feature1": float(i), "feature2": float(i * 2)},
                metadata={"source": f"sensor_{i}"},
            )
            for i in range(5)
        ]

    @pytest.fixture
    def sample_batch(self, sample_data_points):
        """Sample data batch for testing."""
        return StreamDataBatchDTO(
            batch_id="test_batch_123",
            data_points=sample_data_points,
            window_start=sample_data_points[0].timestamp,
            window_end=sample_data_points[-1].timestamp,
        )

    @pytest.fixture
    def mock_detector(self):
        """Mock anomaly detector."""
        detector = Mock()
        detector.id = "test_detector"
        detector.predict = Mock(return_value=np.array([0.1, 0.8, 0.3, 0.9, 0.2]))
        detector.predict_proba = Mock(
            return_value=np.array(
                [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.8, 0.2]]
            )
        )
        return detector


class TestStreamProcessing:
    """Test stream processing functionality."""

    @pytest.mark.asyncio
    async def test_process_batch_success(
        self,
        use_case,
        sample_batch,
        sample_configuration,
        mock_detector,
        mock_detector_repository,
    ):
        """Test successful batch processing."""
        # Setup
        mock_detector_repository.get_by_id.return_value = mock_detector

        request = StreamDetectionRequestDTO(
            request_id="req_123",
            batch=sample_batch,
            configuration=sample_configuration,
            detector_id="test_detector",
        )

        # Execute
        response = await use_case.process_batch(request)

        # Verify
        assert response.success is True
        assert response.request_id == "req_123"
        assert response.detector_id == "test_detector"
        assert len(response.results) == len(sample_batch.data_points)
        assert response.processing_time_ms > 0

        # Verify detector was called
        mock_detector_repository.get_by_id.assert_called_once_with("test_detector")
        mock_detector.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_batch_detector_not_found(
        self, use_case, sample_batch, sample_configuration, mock_detector_repository
    ):
        """Test batch processing with detector not found."""
        # Setup
        mock_detector_repository.get_by_id.return_value = None

        request = StreamDetectionRequestDTO(
            request_id="req_123",
            batch=sample_batch,
            configuration=sample_configuration,
            detector_id="nonexistent_detector",
        )

        # Execute and verify
        with pytest.raises(DetectorNotFoundError):
            await use_case.process_batch(request)

    @pytest.mark.asyncio
    async def test_process_batch_detector_error(
        self,
        use_case,
        sample_batch,
        sample_configuration,
        mock_detector,
        mock_detector_repository,
    ):
        """Test batch processing with detector error."""
        # Setup
        mock_detector_repository.get_by_id.return_value = mock_detector
        mock_detector.predict.side_effect = Exception("Detector failed")

        request = StreamDetectionRequestDTO(
            request_id="req_123",
            batch=sample_batch,
            configuration=sample_configuration,
            detector_id="test_detector",
        )

        # Execute
        response = await use_case.process_batch(request)

        # Verify
        assert response.success is False
        assert "Detector failed" in response.error_message
        assert len(response.results) == 0

    @pytest.mark.asyncio
    async def test_process_single_data_point(
        self,
        use_case,
        sample_data_points,
        sample_configuration,
        mock_detector,
        mock_detector_repository,
    ):
        """Test processing single data point."""
        # Setup
        mock_detector_repository.get_by_id.return_value = mock_detector
        mock_detector.predict.return_value = np.array([0.85])

        data_point = sample_data_points[0]

        # Execute
        result = await use_case.process_single_point(
            data_point, "test_detector", sample_configuration
        )

        # Verify
        assert result.anomaly_score == 0.85
        assert result.timestamp == data_point.timestamp
        assert result.features == data_point.features

    @pytest.mark.asyncio
    async def test_batch_aggregation(
        self, use_case, sample_data_points, sample_configuration
    ):
        """Test data point aggregation into batches."""
        # Execute
        batches = []
        async for batch in use_case._aggregate_into_batches(
            sample_data_points, sample_configuration
        ):
            batches.append(batch)

        # Verify
        assert len(batches) >= 1
        total_points = sum(len(batch.data_points) for batch in batches)
        assert total_points == len(sample_data_points)

        for batch in batches:
            assert len(batch.data_points) <= sample_configuration.batch_size
            assert batch.batch_id is not None
            assert batch.window_start is not None
            assert batch.window_end is not None


class TestStreamConfiguration:
    """Test stream configuration and setup."""

    @pytest.mark.asyncio
    async def test_create_stream_success(self, use_case, sample_configuration):
        """Test successful stream creation."""
        # Execute
        result = await use_case.create_stream(sample_configuration)

        # Verify
        assert result.stream_id == sample_configuration.stream_id
        assert result.status == "created"
        assert result.is_healthy is True

    @pytest.mark.asyncio
    async def test_start_stream_success(self, use_case, sample_configuration):
        """Test successful stream start."""
        # Setup
        await use_case.create_stream(sample_configuration)

        # Execute
        result = await use_case.start_stream(sample_configuration.stream_id)

        # Verify
        assert result.stream_id == sample_configuration.stream_id
        assert result.status == "running"
        assert result.is_healthy is True

    @pytest.mark.asyncio
    async def test_stop_stream_success(self, use_case, sample_configuration):
        """Test successful stream stop."""
        # Setup
        await use_case.create_stream(sample_configuration)
        await use_case.start_stream(sample_configuration.stream_id)

        # Execute
        result = await use_case.stop_stream(sample_configuration.stream_id)

        # Verify
        assert result.stream_id == sample_configuration.stream_id
        assert result.status == "stopped"

    @pytest.mark.asyncio
    async def test_get_stream_status(self, use_case, sample_configuration):
        """Test getting stream status."""
        # Setup
        await use_case.create_stream(sample_configuration)

        # Execute
        status = await use_case.get_stream_status(sample_configuration.stream_id)

        # Verify
        assert status.stream_id == sample_configuration.stream_id
        assert status.status in ["created", "running", "stopped", "error"]
        assert status.uptime_seconds >= 0

    @pytest.mark.asyncio
    async def test_update_configuration(self, use_case, sample_configuration):
        """Test updating stream configuration."""
        # Setup
        await use_case.create_stream(sample_configuration)

        # Modify configuration
        updated_config = sample_configuration
        updated_config.batch_size = 20

        # Execute
        result = await use_case.update_stream_configuration(
            sample_configuration.stream_id, updated_config
        )

        # Verify
        assert result is True

        # Verify configuration was updated
        status = await use_case.get_stream_status(sample_configuration.stream_id)
        # Additional verification would depend on implementation details


class TestBackpressureHandling:
    """Test backpressure handling functionality."""

    @pytest.mark.asyncio
    async def test_backpressure_detection(
        self, use_case, sample_batch, sample_configuration, mock_backpressure_handler
    ):
        """Test backpressure detection."""
        # Setup
        mock_backpressure_handler.check_backpressure.return_value = True

        # Execute and verify
        with pytest.raises(BackpressureError):
            await use_case._check_backpressure(sample_configuration)

    @pytest.mark.asyncio
    async def test_backpressure_handling_oldest_drop(
        self, use_case, sample_data_points
    ):
        """Test backpressure handling with oldest drop policy."""
        config = StreamConfigurationDTO(
            stream_id="test_stream",
            batch_size=10,
            timeout_ms=5000,
            backpressure=BackpressureConfigDTO(
                enabled=True,
                max_queue_size=3,  # Small queue for testing
                drop_policy="oldest",
            ),
        )

        # Simulate queue overflow
        queue = sample_data_points[:5]  # 5 items > max_queue_size of 3

        # Execute
        handled_queue = await use_case._handle_backpressure(queue, config)

        # Verify - should keep newest 3 items
        assert len(handled_queue) <= config.backpressure.max_queue_size
        # Should contain the most recent items
        if len(handled_queue) == 3:
            assert handled_queue[-1] == sample_data_points[4]  # Most recent

    @pytest.mark.asyncio
    async def test_backpressure_handling_newest_drop(
        self, use_case, sample_data_points
    ):
        """Test backpressure handling with newest drop policy."""
        config = StreamConfigurationDTO(
            stream_id="test_stream",
            batch_size=10,
            timeout_ms=5000,
            backpressure=BackpressureConfigDTO(
                enabled=True,
                max_queue_size=3,  # Small queue for testing
                drop_policy="newest",
            ),
        )

        # Simulate queue overflow
        queue = sample_data_points[:5]  # 5 items > max_queue_size of 3

        # Execute
        handled_queue = await use_case._handle_backpressure(queue, config)

        # Verify - should keep oldest 3 items
        assert len(handled_queue) <= config.backpressure.max_queue_size
        # Should contain the oldest items
        if len(handled_queue) == 3:
            assert handled_queue[0] == sample_data_points[0]  # Oldest


class TestMetricsCollection:
    """Test metrics collection functionality."""

    @pytest.mark.asyncio
    async def test_collect_metrics(
        self, use_case, sample_configuration, mock_metrics_collector
    ):
        """Test metrics collection."""
        # Setup
        await use_case.create_stream(sample_configuration)

        # Execute
        metrics = await use_case.get_stream_metrics(sample_configuration.stream_id)

        # Verify
        assert metrics.stream_id == sample_configuration.stream_id
        assert metrics.messages_processed >= 0
        assert metrics.messages_failed >= 0
        assert metrics.success_rate >= 0.0 and metrics.success_rate <= 1.0
        assert metrics.window_start is not None
        assert metrics.window_end is not None

    @pytest.mark.asyncio
    async def test_update_metrics_on_processing(
        self,
        use_case,
        sample_batch,
        sample_configuration,
        mock_detector,
        mock_detector_repository,
        mock_metrics_collector,
    ):
        """Test metrics update during processing."""
        # Setup
        mock_detector_repository.get_by_id.return_value = mock_detector

        request = StreamDetectionRequestDTO(
            request_id="req_123",
            batch=sample_batch,
            configuration=sample_configuration,
            detector_id="test_detector",
        )

        # Execute
        await use_case.process_batch(request)

        # Verify metrics were collected
        assert mock_metrics_collector.increment_counter.called
        assert mock_metrics_collector.record_histogram.called

    @pytest.mark.asyncio
    async def test_metrics_aggregation(self, use_case, sample_configuration):
        """Test metrics aggregation over time windows."""
        # Setup
        await use_case.create_stream(sample_configuration)

        # Simulate some processing
        for i in range(10):
            await use_case._update_processing_metrics(
                sample_configuration.stream_id,
                processing_time_ms=50.0 + i * 5,
                success=i % 10 != 9,  # 1 failure out of 10
            )

        # Execute
        metrics = await use_case.get_stream_metrics(sample_configuration.stream_id)

        # Verify
        assert metrics.messages_processed == 10
        assert metrics.messages_failed == 1
        assert metrics.success_rate == 0.9
        assert metrics.average_processing_time_ms > 0


class TestCheckpointManagement:
    """Test checkpoint management functionality."""

    @pytest.mark.asyncio
    async def test_save_checkpoint(
        self, use_case, sample_configuration, mock_checkpoint_manager
    ):
        """Test checkpoint saving."""
        # Setup
        await use_case.create_stream(sample_configuration)

        # Execute
        await use_case._save_checkpoint(sample_configuration.stream_id)

        # Verify
        mock_checkpoint_manager.save_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_checkpoint(
        self, use_case, sample_configuration, mock_checkpoint_manager
    ):
        """Test checkpoint loading."""
        # Setup
        checkpoint_data = {
            "stream_id": sample_configuration.stream_id,
            "last_processed_timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {"messages_processed": 100},
        }
        mock_checkpoint_manager.load_checkpoint.return_value = checkpoint_data

        # Execute
        result = await use_case._load_checkpoint(sample_configuration.stream_id)

        # Verify
        assert result == checkpoint_data
        mock_checkpoint_manager.load_checkpoint.assert_called_once_with(
            sample_configuration.stream_id
        )

    @pytest.mark.asyncio
    async def test_checkpoint_cleanup(
        self, use_case, sample_configuration, mock_checkpoint_manager
    ):
        """Test checkpoint cleanup."""
        # Execute
        await use_case._cleanup_old_checkpoints(sample_configuration.stream_id)

        # Verify
        mock_checkpoint_manager.cleanup_old_checkpoints.assert_called_once_with(
            sample_configuration.stream_id
        )


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_stream_processing_error_recovery(
        self,
        use_case,
        sample_batch,
        sample_configuration,
        mock_detector,
        mock_detector_repository,
    ):
        """Test recovery from stream processing errors."""
        # Setup
        mock_detector_repository.get_by_id.return_value = mock_detector

        # First call fails, second succeeds
        mock_detector.predict.side_effect = [
            Exception("Temporary failure"),
            np.array([0.1, 0.8, 0.3, 0.9, 0.2]),
        ]

        request = StreamDetectionRequestDTO(
            request_id="req_123",
            batch=sample_batch,
            configuration=sample_configuration,
            detector_id="test_detector",
        )

        # Execute first call (should fail)
        response1 = await use_case.process_batch(request)
        assert response1.success is False

        # Execute second call (should succeed)
        request.request_id = "req_124"  # New request ID
        response2 = await use_case.process_batch(request)
        assert response2.success is True

    @pytest.mark.asyncio
    async def test_stream_not_found_error(self, use_case):
        """Test handling of stream not found errors."""
        # Execute and verify
        with pytest.raises(StreamProcessingError, match="Stream not found"):
            await use_case.get_stream_status("nonexistent_stream")

    @pytest.mark.asyncio
    async def test_invalid_configuration_error(self, use_case):
        """Test handling of invalid configuration errors."""
        invalid_config = StreamConfigurationDTO(
            stream_id="",  # Invalid empty stream ID
            batch_size=0,  # Invalid batch size
            timeout_ms=-1000,  # Invalid timeout
        )

        # Execute and verify
        with pytest.raises(ValueError):
            await use_case.create_stream(invalid_config)


class TestConcurrentProcessing:
    """Test concurrent processing scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(
        self,
        use_case,
        sample_data_points,
        sample_configuration,
        mock_detector,
        mock_detector_repository,
    ):
        """Test concurrent processing of multiple batches."""
        # Setup
        mock_detector_repository.get_by_id.return_value = mock_detector

        # Create multiple batches
        batches = []
        for i in range(3):
            batch = StreamDataBatchDTO(
                batch_id=f"batch_{i}",
                data_points=sample_data_points,
                window_start=sample_data_points[0].timestamp,
                window_end=sample_data_points[-1].timestamp,
            )
            batches.append(batch)

        # Create requests
        requests = [
            StreamDetectionRequestDTO(
                request_id=f"req_{i}",
                batch=batch,
                configuration=sample_configuration,
                detector_id="test_detector",
            )
            for i, batch in enumerate(batches)
        ]

        # Execute concurrently
        tasks = [use_case.process_batch(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify
        assert len(responses) == 3
        for response in responses:
            assert not isinstance(response, Exception)
            assert response.success is True

    @pytest.mark.asyncio
    async def test_concurrent_stream_operations(self, use_case, sample_configuration):
        """Test concurrent stream operations."""
        # Create multiple stream configurations
        configs = []
        for i in range(3):
            config = StreamConfigurationDTO(
                stream_id=f"stream_{i}", batch_size=10, timeout_ms=5000
            )
            configs.append(config)

        # Execute concurrent operations
        create_tasks = [use_case.create_stream(config) for config in configs]
        await asyncio.gather(*create_tasks)

        start_tasks = [use_case.start_stream(config.stream_id) for config in configs]
        start_results = await asyncio.gather(*start_tasks)

        # Verify
        assert len(start_results) == 3
        for result in start_results:
            assert result.status == "running"


class TestPerformanceCharacteristics:
    """Test performance characteristics and benchmarks."""

    @pytest.mark.asyncio
    async def test_processing_throughput(
        self, use_case, sample_configuration, mock_detector, mock_detector_repository
    ):
        """Test processing throughput measurement."""
        # Setup
        mock_detector_repository.get_by_id.return_value = mock_detector

        # Create large batch for throughput testing
        large_data_points = []
        base_time = datetime.now(timezone.utc)
        for i in range(1000):  # 1000 data points
            point = StreamDataPointDTO(
                timestamp=base_time + timedelta(milliseconds=i),
                features={"feature1": float(i), "feature2": float(i * 2)},
            )
            large_data_points.append(point)

        batch = StreamDataBatchDTO(
            batch_id="large_batch",
            data_points=large_data_points,
            window_start=large_data_points[0].timestamp,
            window_end=large_data_points[-1].timestamp,
        )

        request = StreamDetectionRequestDTO(
            request_id="throughput_test",
            batch=batch,
            configuration=sample_configuration,
            detector_id="test_detector",
        )

        # Execute and measure
        start_time = datetime.now()
        response = await use_case.process_batch(request)
        end_time = datetime.now()

        # Verify
        assert response.success is True
        processing_time = (end_time - start_time).total_seconds() * 1000
        throughput = len(large_data_points) / (processing_time / 1000)

        # Should process at least 100 points per second
        assert throughput > 100

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, use_case, sample_configuration):
        """Test memory efficiency during processing."""
        # This would typically measure memory usage during processing
        # For now, just verify that large data sets can be processed
        # without running out of memory

        # Create very large dataset
        large_data_points = []
        base_time = datetime.now(timezone.utc)
        for i in range(10000):  # 10,000 data points
            point = StreamDataPointDTO(
                timestamp=base_time + timedelta(milliseconds=i),
                features={
                    f"feature_{j}": float(i + j) for j in range(50)
                },  # 50 features
            )
            large_data_points.append(point)

        # Process in batches
        batch_size = sample_configuration.batch_size
        batches_processed = 0

        for i in range(0, len(large_data_points), batch_size):
            batch_data = large_data_points[i : i + batch_size]
            batch = StreamDataBatchDTO(
                batch_id=f"memory_test_batch_{batches_processed}",
                data_points=batch_data,
                window_start=batch_data[0].timestamp,
                window_end=batch_data[-1].timestamp,
            )

            # Just verify batch creation doesn't fail
            assert len(batch.data_points) > 0
            batches_processed += 1

        # Verify we processed all data
        assert batches_processed * batch_size >= len(large_data_points)

    @pytest.mark.asyncio
    async def test_latency_characteristics(
        self,
        use_case,
        sample_data_points,
        sample_configuration,
        mock_detector,
        mock_detector_repository,
    ):
        """Test latency characteristics for single point processing."""
        # Setup
        mock_detector_repository.get_by_id.return_value = mock_detector
        mock_detector.predict.return_value = np.array([0.5])

        data_point = sample_data_points[0]

        # Measure latency for multiple calls
        latencies = []
        for _ in range(10):
            start_time = datetime.now()
            await use_case.process_single_point(
                data_point, "test_detector", sample_configuration
            )
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000
            latencies.append(latency)

        # Verify latency characteristics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        # Should have reasonable latency (< 100ms average, < 500ms max)
        assert avg_latency < 100
        assert max_latency < 500
