"""Integration tests for real-time processing features."""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import uuid

from pynomaly.domain.entities import Dataset, DetectionResult, Anomaly, Detector
from pynomaly.domain.value_objects import AnomalyScore, AnomalyType
from pynomaly.features.real_time_processing import (
    StreamingPipeline,
    RealTimeDetector,
    EventProcessor,
    StreamBuffer,
    StreamProcessor,
    StreamEvent,
    StreamEventType,
    StreamingConfig,
    ProcessingMode,
    get_stream_processor,
)


class MockDetector(Detector):
    """Mock detector for real-time testing."""
    
    def __init__(self, algorithm: str = "mock_realtime_detector"):
        self.algorithm = algorithm
        self.is_fitted = True  # Always fitted for real-time
        self.detection_count = 0
        self.anomaly_threshold = 0.7
    
    def fit(self, dataset: Dataset) -> None:
        """Mock fit method."""
        self.is_fitted = True
    
    def predict(self, dataset: Dataset) -> DetectionResult:
        """Mock predict method."""
        self.detection_count += 1
        anomalies = []
        
        # Simulate anomaly detection based on data values
        for i, row in dataset.data.iterrows():
            # Simple rule: anomaly if any feature > 2 or < -2
            feature_values = [v for v in row.values if isinstance(v, (int, float))]
            if feature_values and (max(feature_values) > 2 or min(feature_values) < -2):
                anomaly = Anomaly(
                    id=f"rt_anomaly_{self.detection_count}_{i}",
                    score=AnomalyScore(min(0.9, max(abs(v) for v in feature_values) / 3)),
                    type=AnomalyType.POINT,
                    timestamp=datetime.now(),
                    data_point=row.to_dict(),
                )
                anomalies.append(anomaly)
        
        return DetectionResult(
            anomalies=anomalies,
            threshold=self.anomaly_threshold,
            metadata={
                "algorithm": self.algorithm,
                "execution_time_ms": 10.0,
                "detection_count": self.detection_count,
            }
        )
    
    async def detect(self, dataset: Dataset) -> DetectionResult:
        """Async detect method for real-time processing."""
        return self.predict(dataset)


@pytest.fixture
def streaming_config():
    """Create streaming configuration for testing."""
    return StreamingConfig(
        buffer_size=100,
        batch_size=10,
        batch_timeout_seconds=1.0,
        processing_mode=ProcessingMode.REAL_TIME,
        enable_backpressure=True,
        max_memory_mb=128,
        heartbeat_interval_seconds=5.0,
        error_threshold=10,
        window_size_minutes=5
    )


@pytest.fixture
def mock_detector():
    """Create mock detector for testing."""
    return MockDetector()


@pytest.fixture
def sample_data_points():
    """Create sample data points for streaming."""
    data_points = []
    
    # Normal data points
    for i in range(10):
        data_points.append({
            "timestamp": datetime.now() - timedelta(minutes=i),
            "feature_1": np.random.randn(),
            "feature_2": np.random.randn(),
            "feature_3": np.random.randn(),
            "id": f"normal_{i}"
        })
    
    # Anomalous data points
    for i in range(3):
        data_points.append({
            "timestamp": datetime.now() - timedelta(minutes=i),
            "feature_1": np.random.randn() + 3,  # Anomalous value
            "feature_2": np.random.randn(),
            "feature_3": np.random.randn(),
            "id": f"anomaly_{i}"
        })
    
    return data_points


@pytest.mark.asyncio
class TestStreamBufferIntegration:
    """Integration tests for stream buffer."""
    
    async def test_stream_buffer_basic_operations(self):
        """Test basic stream buffer operations."""
        buffer = StreamBuffer(max_size=10)
        
        # Test adding events
        events = []
        for i in range(5):
            event = StreamEvent(
                event_id=f"event_{i}",
                event_type=StreamEventType.DATA_POINT,
                timestamp=datetime.now(),
                data={"value": i, "feature": f"test_{i}"},
                source="test_source"
            )
            events.append(event)
            success = await buffer.add(event)
            assert success
        
        # Check buffer size
        size = await buffer.size()
        assert size == 5
        
        # Test batch retrieval
        batch = await buffer.get_batch(3)
        assert len(batch) == 3
        assert all(event.event_type == StreamEventType.DATA_POINT for event in batch)
        
        # Check remaining size
        remaining_size = await buffer.size()
        assert remaining_size == 2
        
        # Test buffer statistics
        stats = await buffer.get_stats()
        assert stats["total_received"] == 5
        assert stats["total_processed"] == 3
        assert stats["current_size"] == 2
        assert stats["utilization"] == 0.2  # 2/10
    
    async def test_stream_buffer_overflow_handling(self):
        """Test stream buffer overflow handling."""
        buffer = StreamBuffer(max_size=3)
        
        # Fill buffer to capacity
        for i in range(3):
            event = StreamEvent(
                event_id=f"event_{i}",
                event_type=StreamEventType.DATA_POINT,
                timestamp=datetime.now(),
                data={"value": i},
                source="test_source"
            )
            success = await buffer.add(event)
            assert success
        
        # Try to add one more (should trigger overflow)
        overflow_event = StreamEvent(
            event_id="overflow_event",
            event_type=StreamEventType.DATA_POINT,
            timestamp=datetime.now(),
            data={"value": "overflow"},
            source="test_source"
        )
        success = await buffer.add(overflow_event)
        assert not success  # Should fail due to overflow
        
        # Check statistics
        stats = await buffer.get_stats()
        assert stats["buffer_overflows"] == 1
        assert stats["total_received"] == 3  # Only successful additions
    
    async def test_stream_buffer_concurrent_access(self):
        """Test concurrent access to stream buffer."""
        buffer = StreamBuffer(max_size=50)
        
        # Producer task
        async def producer(producer_id: int):
            for i in range(10):
                event = StreamEvent(
                    event_id=f"producer_{producer_id}_event_{i}",
                    event_type=StreamEventType.DATA_POINT,
                    timestamp=datetime.now(),
                    data={"producer": producer_id, "value": i},
                    source=f"producer_{producer_id}"
                )
                await buffer.add(event)
                await asyncio.sleep(0.001)  # Small delay
        
        # Consumer task
        async def consumer(consumer_id: int):
            consumed = []
            for _ in range(15):  # Try to consume 15 events
                if not await buffer.is_empty():
                    batch = await buffer.get_batch(1)
                    if batch:
                        consumed.extend(batch)
                await asyncio.sleep(0.002)  # Small delay
            return consumed
        
        # Run concurrent producers and consumers
        tasks = []
        
        # Start 2 producers
        for i in range(2):
            tasks.append(asyncio.create_task(producer(i)))
        
        # Start 1 consumer
        consumer_task = asyncio.create_task(consumer(0))
        tasks.append(consumer_task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Verify results
        consumed_events = consumer_task.result()
        stats = await buffer.get_stats()
        
        assert len(consumed_events) <= 20  # At most 20 events produced
        assert stats["total_received"] == 20  # 2 producers * 10 events each
        assert stats["total_processed"] >= len(consumed_events)


@pytest.mark.asyncio
class TestEventProcessorIntegration:
    """Integration tests for event processor."""
    
    async def test_event_processor_basic_functionality(self, streaming_config):
        """Test basic event processor functionality."""
        processor = EventProcessor(streaming_config)
        handled_events = []
        
        # Register event handler
        async def data_point_handler(event: StreamEvent):
            handled_events.append(event)
        
        processor.register_handler(StreamEventType.DATA_POINT, data_point_handler)
        
        # Create and process events
        events = []
        for i in range(5):
            event = StreamEvent(
                event_id=f"test_event_{i}",
                event_type=StreamEventType.DATA_POINT,
                timestamp=datetime.now(),
                data={"value": i},
                source="test_processor"
            )
            events.append(event)
            success = await processor.process_event(event)
            assert success
        
        # Verify all events were handled
        assert len(handled_events) == 5
        assert all(event.data["value"] in range(5) for event in handled_events)
        
        # Check processing statistics
        stats = await processor.get_processing_stats()
        assert stats["events_processed"] == 5
        assert stats["events_failed"] == 0
        assert stats["error_rate"] == 0.0
    
    async def test_event_processor_batch_processing(self, streaming_config):
        """Test event processor batch processing."""
        processor = EventProcessor(streaming_config)
        processed_data = []
        
        # Register batch handler
        async def batch_handler(event: StreamEvent):
            processed_data.append(event.data)
        
        processor.register_handler(StreamEventType.DATA_POINT, batch_handler)
        
        # Create batch of events
        events = []
        for i in range(10):
            event = StreamEvent(
                event_id=f"batch_event_{i}",
                event_type=StreamEventType.DATA_POINT,
                timestamp=datetime.now(),
                data={"batch_id": i, "value": i * 2},
                source="batch_test"
            )
            events.append(event)
        
        # Process batch
        results = await processor.process_batch(events)
        
        # Verify batch processing results
        assert results["success"] == 10
        assert results["failed"] == 0
        assert len(processed_data) == 10
        
        # Verify data integrity
        for i, data in enumerate(processed_data):
            assert data["batch_id"] == i
            assert data["value"] == i * 2
    
    async def test_event_processor_error_handling(self, streaming_config):
        """Test event processor error handling."""
        processor = EventProcessor(streaming_config)
        
        # Register handler that throws errors
        async def error_handler(event: StreamEvent):
            if event.data.get("should_fail", False):
                raise ValueError(f"Intentional error for event {event.event_id}")
        
        processor.register_handler(StreamEventType.DATA_POINT, error_handler)
        
        # Create events (some that will fail)
        events = []
        for i in range(5):
            event = StreamEvent(
                event_id=f"error_test_event_{i}",
                event_type=StreamEventType.DATA_POINT,
                timestamp=datetime.now(),
                data={"value": i, "should_fail": i % 2 == 0},  # Even indices fail
                source="error_test"
            )
            events.append(event)
        
        # Process events
        results = await processor.process_batch(events)
        
        # Verify error handling
        assert results["success"] == 2  # Odd indices (1, 3)
        assert results["failed"] == 3   # Even indices (0, 2, 4)
        
        # Check error statistics
        stats = await processor.get_processing_stats()
        assert stats["events_failed"] == 3
        assert len(stats["recent_errors"]) >= 3
        assert stats["error_rate"] > 0.0


@pytest.mark.asyncio
class TestRealTimeDetectorIntegration:
    """Integration tests for real-time detector."""
    
    async def test_real_time_detector_streaming(self, mock_detector, streaming_config):
        """Test real-time detector with streaming data."""
        rt_detector = RealTimeDetector(mock_detector, streaming_config)
        
        # Process normal data points
        normal_results = []
        for i in range(5):
            data_point = {
                "feature_1": np.random.randn() * 0.5,  # Normal values
                "feature_2": np.random.randn() * 0.5,
                "feature_3": np.random.randn() * 0.5,
                "timestamp": datetime.now(),
            }
            
            result = await rt_detector.detect_streaming(data_point)
            normal_results.append(result)
        
        # Process anomalous data points
        anomaly_results = []
        for i in range(3):
            data_point = {
                "feature_1": np.random.randn() + 3,  # Anomalous values
                "feature_2": np.random.randn() + 3,
                "feature_3": np.random.randn() + 3,
                "timestamp": datetime.now(),
            }
            
            result = await rt_detector.detect_streaming(data_point)
            anomaly_results.append(result)
        
        # Verify detection results
        # Normal data should have fewer/no anomalies
        normal_anomaly_count = sum(len(r.anomalies) if r else 0 for r in normal_results)
        
        # Anomalous data should have more anomalies
        anomaly_anomaly_count = sum(len(r.anomalies) if r else 0 for r in anomaly_results)
        
        assert anomaly_anomaly_count >= normal_anomaly_count
        
        # Check detector statistics
        stats = await rt_detector.get_detection_stats()
        assert stats["total_detections"] == 8  # 5 normal + 3 anomalous
        assert stats["anomalies_detected"] >= 0
        assert stats["window_size"] == 8  # All data points in sliding window
        assert stats["processing_performance"]["samples"] == 8
    
    async def test_real_time_detector_performance(self, mock_detector, streaming_config):
        """Test real-time detector performance characteristics."""
        rt_detector = RealTimeDetector(mock_detector, streaming_config)
        
        # Process many data points to test performance
        processing_times = []
        for i in range(20):
            data_point = {
                "feature_1": np.random.randn(),
                "feature_2": np.random.randn(),
                "feature_3": np.random.randn(),
                "timestamp": datetime.now(),
                "id": f"perf_test_{i}"
            }
            
            start_time = datetime.now()
            result = await rt_detector.detect_streaming(data_point)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            processing_times.append(processing_time)
        
        # Verify performance characteristics
        stats = await rt_detector.get_detection_stats()
        performance = stats["processing_performance"]
        
        assert performance["samples"] == 20
        assert performance["avg_time_ms"] > 0
        assert performance["min_time_ms"] >= 0
        assert performance["max_time_ms"] >= performance["min_time_ms"]
        
        # Check that processing times are reasonable (under 1 second)
        assert all(time < 1000 for time in processing_times)
        assert performance["avg_time_ms"] < 1000


@pytest.mark.asyncio
class TestStreamingPipelineIntegration:
    """Integration tests for streaming pipeline."""
    
    async def test_streaming_pipeline_lifecycle(self, mock_detector, streaming_config, sample_data_points):
        """Test complete streaming pipeline lifecycle."""
        pipeline = StreamingPipeline(mock_detector, streaming_config)
        
        # Start pipeline
        started = await pipeline.start()
        assert started
        assert pipeline.is_running
        
        # Submit data points
        submitted_count = 0
        for data_point in sample_data_points[:10]:  # Submit first 10 points
            success = await pipeline.submit_data(data_point, source="test_pipeline")
            if success:
                submitted_count += 1
        
        assert submitted_count > 0
        
        # Wait for processing
        await asyncio.sleep(2)  # Give time for processing
        
        # Check pipeline status
        status = await pipeline.get_pipeline_status()
        assert status["is_running"]
        assert status["buffer_stats"]["total_received"] >= submitted_count
        assert status["detector_stats"]["total_detections"] >= 0
        
        # Stop pipeline
        stopped = await pipeline.stop()
        assert stopped
        assert not pipeline.is_running
        
        # Verify final status
        final_status = await pipeline.get_pipeline_status()
        assert not final_status["is_running"]
    
    async def test_streaming_pipeline_backpressure(self, mock_detector, sample_data_points):
        """Test streaming pipeline backpressure handling."""
        # Create config with small buffer to test backpressure
        config = StreamingConfig(
            buffer_size=5,  # Small buffer
            batch_size=2,
            batch_timeout_seconds=0.1,
            enable_backpressure=True
        )
        
        pipeline = StreamingPipeline(mock_detector, config)
        
        # Start pipeline
        await pipeline.start()
        
        # Rapidly submit many data points to trigger backpressure
        submission_results = []
        for i in range(15):  # More than buffer size
            data_point = {
                "feature_1": np.random.randn(),
                "feature_2": np.random.randn(),
                "timestamp": datetime.now(),
                "id": f"backpressure_test_{i}"
            }
            success = await pipeline.submit_data(data_point, source="backpressure_test")
            submission_results.append(success)
            
            # Submit rapidly
            await asyncio.sleep(0.001)
        
        # Some submissions should fail due to backpressure
        failed_submissions = submission_results.count(False)
        successful_submissions = submission_results.count(True)
        
        assert successful_submissions > 0
        # May or may not have failures depending on timing
        
        # Wait for buffer to drain
        await asyncio.sleep(1)
        
        # Check buffer statistics
        status = await pipeline.get_pipeline_status()
        buffer_stats = status["buffer_stats"]
        
        # Buffer should have handled overflow gracefully
        assert "buffer_overflows" in buffer_stats
        
        # Stop pipeline
        await pipeline.stop()
    
    async def test_streaming_pipeline_event_handling(self, mock_detector, streaming_config):
        """Test streaming pipeline event handling."""
        pipeline = StreamingPipeline(mock_detector, streaming_config)
        detected_anomalies = []
        
        # Register custom anomaly handler
        async def custom_anomaly_handler(event: StreamEvent):
            if event.event_type == StreamEventType.ANOMALY_DETECTED:
                detected_anomalies.append(event.data)
        
        pipeline.event_processor.register_handler(
            StreamEventType.ANOMALY_DETECTED, 
            custom_anomaly_handler
        )
        
        # Start pipeline
        await pipeline.start()
        
        # Submit anomalous data points
        anomalous_data_points = [
            {
                "feature_1": 4.0,  # Anomalous value
                "feature_2": 3.5,  # Anomalous value
                "feature_3": -3.0, # Anomalous value
                "timestamp": datetime.now(),
                "id": f"anomaly_event_test_{i}"
            }
            for i in range(3)
        ]
        
        for data_point in anomalous_data_points:
            await pipeline.submit_data(data_point, source="anomaly_test")
        
        # Wait for processing and event handling
        await asyncio.sleep(2)
        
        # Verify anomaly events were detected and handled
        assert len(detected_anomalies) > 0
        
        # Verify event data structure
        for anomaly_data in detected_anomalies:
            assert "anomalies" in anomaly_data
            assert "detection_result" in anomaly_data
            assert "original_event_id" in anomaly_data
        
        # Stop pipeline
        await pipeline.stop()


@pytest.mark.asyncio
class TestStreamProcessorIntegration:
    """Integration tests for stream processor."""
    
    async def test_stream_processor_multiple_pipelines(self, mock_detector, streaming_config, sample_data_points):
        """Test stream processor with multiple pipelines."""
        processor = StreamProcessor()
        
        # Create multiple pipelines
        pipeline_ids = ["pipeline_1", "pipeline_2", "pipeline_3"]
        
        for pipeline_id in pipeline_ids:
            created = await processor.create_pipeline(
                pipeline_id, 
                mock_detector, 
                streaming_config
            )
            assert created
        
        # Start all pipelines
        for pipeline_id in pipeline_ids:
            started = await processor.start_pipeline(pipeline_id)
            assert started
        
        # Submit data to different pipelines
        for i, data_point in enumerate(sample_data_points[:9]):  # 3 points per pipeline
            pipeline_id = pipeline_ids[i % 3]
            success = await processor.submit_data(
                pipeline_id, 
                data_point, 
                source=f"multi_pipeline_test_{pipeline_id}"
            )
            assert success
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check status of all pipelines
        all_status = await processor.get_all_pipeline_status()
        assert all_status["total_pipelines"] == 3
        assert all_status["active_pipelines"] == 3
        assert len(all_status["pipelines"]) == 3
        
        # Verify each pipeline is running and has processed data
        for pipeline_id in pipeline_ids:
            pipeline_status = all_status["pipelines"][pipeline_id]
            assert pipeline_status["is_running"]
            assert pipeline_status["buffer_stats"]["total_received"] >= 1
        
        # Stop all pipelines
        shutdown_success = await processor.shutdown_all()
        assert shutdown_success
        
        # Verify all pipelines are stopped
        final_status = await processor.get_all_pipeline_status()
        assert final_status["active_pipelines"] == 0
    
    async def test_stream_processor_pipeline_management(self, mock_detector, streaming_config):
        """Test stream processor pipeline management operations."""
        processor = StreamProcessor()
        
        # Create pipeline
        created = await processor.create_pipeline("mgmt_test_pipeline", mock_detector, streaming_config)
        assert created
        
        # Try to create duplicate pipeline (should fail)
        duplicate_created = await processor.create_pipeline("mgmt_test_pipeline", mock_detector, streaming_config)
        assert not duplicate_created
        
        # Start pipeline
        started = await processor.start_pipeline("mgmt_test_pipeline")
        assert started
        
        # Submit some data
        test_data = {
            "feature_1": 1.5,
            "feature_2": -0.8,
            "timestamp": datetime.now(),
            "id": "mgmt_test_data"
        }
        
        submit_success = await processor.submit_data("mgmt_test_pipeline", test_data, "management_test")
        assert submit_success
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Stop specific pipeline
        stopped = await processor.stop_pipeline("mgmt_test_pipeline")
        assert stopped
        
        # Try to submit data to stopped pipeline (should fail)
        submit_to_stopped = await processor.submit_data("mgmt_test_pipeline", test_data, "stopped_test")
        assert not submit_to_stopped
        
        # Try operations on non-existent pipeline
        nonexistent_start = await processor.start_pipeline("nonexistent_pipeline")
        assert not nonexistent_start
        
        nonexistent_stop = await processor.stop_pipeline("nonexistent_pipeline")
        assert not nonexistent_stop
        
        nonexistent_submit = await processor.submit_data("nonexistent_pipeline", test_data, "nonexistent_test")
        assert not nonexistent_submit
    
    async def test_global_stream_processor_integration(self, mock_detector, streaming_config):
        """Test global stream processor integration."""
        # Test global processor retrieval
        processor1 = get_stream_processor()
        processor2 = get_stream_processor()
        
        # Verify singleton behavior
        assert processor1 is processor2
        assert isinstance(processor1, StreamProcessor)
        
        # Test global processor functionality
        created = await processor1.create_pipeline("global_test_pipeline", mock_detector, streaming_config)
        assert created
        
        # Verify persistence across references
        all_status = await processor2.get_all_pipeline_status()
        assert all_status["total_pipelines"] >= 1
        assert "global_test_pipeline" in all_status["pipelines"]
        
        # Clean up
        await processor1.stop_pipeline("global_test_pipeline")


@pytest.mark.asyncio
class TestRealTimeProcessingErrorHandling:
    """Test error handling in real-time processing."""
    
    async def test_detector_error_handling(self, streaming_config):
        """Test error handling when detector fails."""
        
        class FailingDetector(Detector):
            def __init__(self):
                self.algorithm = "failing_detector"
                self.is_fitted = True
            
            def fit(self, dataset: Dataset) -> None:
                pass
            
            def predict(self, dataset: Dataset) -> DetectionResult:
                raise RuntimeError("Detector intentionally failed")
            
            async def detect(self, dataset: Dataset) -> DetectionResult:
                raise RuntimeError("Async detector intentionally failed")
        
        failing_detector = FailingDetector()
        rt_detector = RealTimeDetector(failing_detector, streaming_config)
        
        # Test that detector failures are handled gracefully
        data_point = {
            "feature_1": 1.0,
            "feature_2": 2.0,
            "timestamp": datetime.now()
        }
        
        with pytest.raises(Exception):  # Should propagate the error
            await rt_detector.detect_streaming(data_point)
    
    async def test_pipeline_error_recovery(self, mock_detector):
        """Test pipeline error recovery mechanisms."""
        config = StreamingConfig(
            buffer_size=10,
            batch_size=3,
            batch_timeout_seconds=0.5,
            error_threshold=2  # Low threshold for testing
        )
        
        pipeline = StreamingPipeline(mock_detector, config)
        
        # Register error-prone handler
        error_count = 0
        
        async def error_prone_handler(event: StreamEvent):
            nonlocal error_count
            error_count += 1
            if error_count <= 3:  # First 3 events cause errors
                raise ValueError(f"Intentional error #{error_count}")
        
        pipeline.event_processor.register_handler(StreamEventType.DATA_POINT, error_prone_handler)
        
        # Start pipeline
        await pipeline.start()
        
        # Submit data that will cause errors
        for i in range(5):
            data_point = {
                "feature_1": i,
                "timestamp": datetime.now(),
                "id": f"error_recovery_test_{i}"
            }
            await pipeline.submit_data(data_point, "error_test")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check processor statistics for error handling
        processor_stats = await pipeline.event_processor.get_processing_stats()
        assert processor_stats["events_failed"] >= 3  # At least 3 errors
        assert processor_stats["events_processed"] >= 2  # Some successful processing
        
        # Pipeline should still be running despite errors
        status = await pipeline.get_pipeline_status()
        assert status["is_running"]
        
        # Stop pipeline
        await pipeline.stop()
    
    async def test_buffer_memory_limits(self, mock_detector):
        """Test buffer behavior under memory pressure."""
        config = StreamingConfig(
            buffer_size=1000,  # Large buffer
            max_memory_mb=1,   # Very low memory limit
            enable_backpressure=True
        )
        
        pipeline = StreamingPipeline(mock_detector, config)
        await pipeline.start()
        
        # Submit large amount of data
        large_data_submissions = 0
        for i in range(100):
            large_data_point = {
                "feature_1": np.random.randn(),
                "feature_2": np.random.randn(),
                "large_data": "x" * 1000,  # Large string data
                "timestamp": datetime.now(),
                "id": f"memory_test_{i}"
            }
            
            success = await pipeline.submit_data(large_data_point, "memory_test")
            if success:
                large_data_submissions += 1
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Buffer should handle memory pressure gracefully
        buffer_stats = await pipeline.buffer.get_stats()
        
        # Some data should have been processed or rejected due to memory limits
        assert buffer_stats["total_received"] <= large_data_submissions
        
        await pipeline.stop()


@pytest.mark.asyncio
class TestRealTimeProcessingPerformance:
    """Performance tests for real-time processing."""
    
    async def test_high_throughput_processing(self, mock_detector):
        """Test high throughput data processing."""
        config = StreamingConfig(
            buffer_size=1000,
            batch_size=50,
            batch_timeout_seconds=0.1,
            processing_mode=ProcessingMode.MICRO_BATCH
        )
        
        pipeline = StreamingPipeline(mock_detector, config)
        await pipeline.start()
        
        # Submit high volume of data
        start_time = datetime.now()
        submitted_count = 0
        
        for i in range(500):  # High volume
            data_point = {
                "feature_1": np.random.randn(),
                "feature_2": np.random.randn(),
                "feature_3": np.random.randn(),
                "timestamp": datetime.now(),
                "id": f"throughput_test_{i}"
            }
            
            success = await pipeline.submit_data(data_point, "throughput_test")
            if success:
                submitted_count += 1
            
            # No delay - maximum submission rate
        
        submission_time = (datetime.now() - start_time).total_seconds()
        
        # Wait for processing to complete
        await asyncio.sleep(3)
        
        # Check final statistics
        status = await pipeline.get_pipeline_status()
        buffer_stats = status["buffer_stats"]
        detector_stats = status["detector_stats"]
        
        # Calculate throughput
        submission_rate = submitted_count / submission_time if submission_time > 0 else 0
        
        # Verify high throughput was achieved
        assert submitted_count > 400  # Most submissions should succeed
        assert submission_rate > 100   # Should handle >100 submissions/second
        assert buffer_stats["total_processed"] > 0
        
        # Performance should be reasonable
        if detector_stats["processing_performance"]["samples"] > 0:
            avg_processing_time = detector_stats["processing_performance"]["avg_time_ms"]
            assert avg_processing_time < 100  # Should be under 100ms average
        
        await pipeline.stop()
    
    async def test_concurrent_pipeline_performance(self, mock_detector, streaming_config):
        """Test performance with multiple concurrent pipelines."""
        processor = StreamProcessor()
        
        # Create multiple pipelines
        pipeline_count = 5
        pipeline_ids = [f"perf_pipeline_{i}" for i in range(pipeline_count)]
        
        # Create and start all pipelines
        for pipeline_id in pipeline_ids:
            await processor.create_pipeline(pipeline_id, mock_detector, streaming_config)
            await processor.start_pipeline(pipeline_id)
        
        # Submit data concurrently to all pipelines
        async def submit_data_to_pipeline(pipeline_id: str, data_count: int):
            successful_submissions = 0
            for i in range(data_count):
                data_point = {
                    "feature_1": np.random.randn(),
                    "feature_2": np.random.randn(),
                    "pipeline_id": pipeline_id,
                    "data_index": i,
                    "timestamp": datetime.now()
                }
                
                success = await processor.submit_data(pipeline_id, data_point, f"concurrent_test_{pipeline_id}")
                if success:
                    successful_submissions += 1
            
            return successful_submissions
        
        # Run concurrent submissions
        start_time = datetime.now()
        tasks = [submit_data_to_pipeline(pid, 50) for pid in pipeline_ids]
        submission_results = await asyncio.gather(*tasks)
        concurrent_time = (datetime.now() - start_time).total_seconds()
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Verify concurrent performance
        total_submissions = sum(submission_results)
        concurrent_throughput = total_submissions / concurrent_time if concurrent_time > 0 else 0
        
        assert total_submissions > pipeline_count * 40  # Most submissions should succeed
        assert concurrent_throughput > 50  # Should handle reasonable concurrent load
        
        # Check that all pipelines processed data
        all_status = await processor.get_all_pipeline_status()
        for pipeline_id in pipeline_ids:
            pipeline_status = all_status["pipelines"][pipeline_id]
            assert pipeline_status["buffer_stats"]["total_received"] > 0
        
        # Clean up
        await processor.shutdown_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])