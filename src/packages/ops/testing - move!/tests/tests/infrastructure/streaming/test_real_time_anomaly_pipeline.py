"""Tests for real-time anomaly detection pipeline."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import numpy as np
import pytest

from monorepo.infrastructure.streaming.real_time_anomaly_pipeline import (
    AlertSeverity,
    DataPoint,
    DataSource,
    KafkaDataSource,
    RealTimeAnomalyPipeline,
    StreamingAlert,
    StreamingAnomalyDetector,
    StreamingConfig,
    StreamingMetrics,
    StreamingWindow,
    WebSocketDataSource,
)


@pytest.fixture
def sample_data_point():
    """Create a sample data point."""
    return DataPoint(
        timestamp=datetime.utcnow(),
        data={
            "feature_1": 1.5,
            "feature_2": 2.0,
            "feature_3": -0.5,
        },
        source_id="test_source",
    )


@pytest.fixture
def streaming_config():
    """Create streaming configuration."""
    return StreamingConfig(
        pipeline_id="test_pipeline",
        batch_size=10,
        window_size=100,
        window_slide=10,
        max_buffer_size=1000,
        processing_timeout=5.0,
        enable_backpressure=True,
        enable_metrics=True,
        alert_thresholds={
            "anomaly_rate": 0.1,
            "processing_latency": 1000.0,
            "error_rate": 0.05,
        },
    )


@pytest.fixture
def mock_data_source():
    """Create a mock data source."""
    data_source = AsyncMock(spec=DataSource)
    data_source.connect = AsyncMock()
    data_source.disconnect = AsyncMock()

    # Mock async generator
    async def mock_consume():
        for i in range(100):
            yield DataPoint(
                timestamp=datetime.utcnow(),
                data={
                    "feature_1": np.random.normal(0, 1),
                    "feature_2": np.random.normal(0, 1),
                    "feature_3": np.random.normal(0, 1),
                },
                source_id="mock_source",
            )
            await asyncio.sleep(0.01)  # Small delay to simulate real streaming

    data_source.consume.return_value = mock_consume()
    return data_source


class TestStreamingWindow:
    """Test streaming window functionality."""

    def test_streaming_window_initialization(self):
        """Test streaming window initialization."""
        window = StreamingWindow(size=100, slide=10)
        assert window.size == 100
        assert window.slide == 10
        assert len(window.buffer) == 0
        assert window.slide_count == 0

    def test_add_data_points(self, sample_data_point):
        """Test adding data points to window."""
        window = StreamingWindow(size=3, slide=1)

        # Add first point
        should_process = window.add(sample_data_point)
        assert not should_process
        assert len(window.buffer) == 1

        # Add second point
        should_process = window.add(sample_data_point)
        assert not should_process
        assert len(window.buffer) == 2

        # Add third point - window should be ready
        should_process = window.add(sample_data_point)
        assert should_process
        assert len(window.buffer) == 3

    def test_sliding_window(self, sample_data_point):
        """Test sliding window behavior."""
        window = StreamingWindow(size=3, slide=2)

        # Fill window
        for _ in range(3):
            window.add(sample_data_point)

        # Get window data
        data = window.get_window_data()
        assert len(data) == 3

        # Slide window
        window.slide_window()
        assert len(window.buffer) == 1  # Should remove 2 points
        assert window.slide_count == 0


class TestStreamingAnomalyDetector:
    """Test streaming anomaly detector."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = StreamingAnomalyDetector(
            detector_algorithm="isolation_forest",
            contamination=0.1,
            window_size=100,
            retraining_interval=1000,
        )

        assert detector.detector_algorithm == "isolation_forest"
        assert detector.contamination == 0.1
        assert detector.window_size == 100
        assert detector.retraining_interval == 1000
        assert detector.detector is not None
        assert not detector.is_trained

    def test_unsupported_algorithm(self):
        """Test unsupported algorithm raises error."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            StreamingAnomalyDetector(detector_algorithm="unsupported_algo")

    @pytest.mark.asyncio
    async def test_process_batch_empty(self):
        """Test processing empty batch."""
        detector = StreamingAnomalyDetector()
        results = await detector.process_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_process_batch_untrained(self):
        """Test processing batch with untrained detector."""
        detector = StreamingAnomalyDetector(window_size=50)

        # Create sample data points
        data_points = []
        for i in range(10):
            data_points.append(
                DataPoint(
                    timestamp=datetime.utcnow(),
                    data={
                        "feature_1": np.random.normal(0, 1),
                        "feature_2": np.random.normal(0, 1),
                    },
                    source_id="test",
                )
            )

        # Process batch - should return empty results since detector is untrained
        results = await detector.process_batch(data_points)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_process_batch_trained(self):
        """Test processing batch with trained detector."""
        detector = StreamingAnomalyDetector(
            window_size=20,
            retraining_interval=20,
        )

        # Create training data to train the detector
        training_points = []
        for i in range(25):  # More than window size to trigger training
            training_points.append(
                DataPoint(
                    timestamp=datetime.utcnow(),
                    data={
                        "feature_1": np.random.normal(0, 1),
                        "feature_2": np.random.normal(0, 1),
                    },
                    source_id="test",
                )
            )

        # Process batch to train detector
        results = await detector.process_batch(training_points)

        # Detector should be trained now
        assert detector.is_trained
        assert len(results) > 0

        # Check result structure
        result = results[0]
        assert result.algorithm_name == "isolation_forest"
        assert result.metadata["is_streaming"] is True


class TestDataSources:
    """Test data source implementations."""

    def test_kafka_data_source_initialization(self):
        """Test Kafka data source initialization."""
        kafka_source = KafkaDataSource(
            bootstrap_servers="localhost:9092",
            topic="test_topic",
            group_id="test_group",
        )

        assert kafka_source.bootstrap_servers == "localhost:9092"
        assert kafka_source.topic == "test_topic"
        assert kafka_source.group_id == "test_group"

    def test_websocket_data_source_initialization(self):
        """Test WebSocket data source initialization."""
        ws_source = WebSocketDataSource(
            websocket_url="wss://example.com/ws",
            headers={"Authorization": "Bearer token"},
        )

        assert ws_source.websocket_url == "wss://example.com/ws"
        assert ws_source.headers["Authorization"] == "Bearer token"

    @pytest.mark.asyncio
    async def test_kafka_connect_disconnect(self):
        """Test Kafka connect and disconnect."""
        kafka_source = KafkaDataSource(
            bootstrap_servers="localhost:9092",
            topic="test_topic",
            group_id="test_group",
        )

        # These are mock implementations, so they should succeed
        await kafka_source.connect()
        await kafka_source.disconnect()

    @pytest.mark.asyncio
    async def test_websocket_connect_disconnect(self):
        """Test WebSocket connect and disconnect."""
        ws_source = WebSocketDataSource(websocket_url="wss://example.com/ws")

        # These are mock implementations, so they should succeed
        await ws_source.connect()
        await ws_source.disconnect()

    @pytest.mark.asyncio
    async def test_kafka_consume_data(self):
        """Test Kafka data consumption."""
        kafka_source = KafkaDataSource(
            bootstrap_servers="localhost:9092",
            topic="test_topic",
            group_id="test_group",
        )

        await kafka_source.connect()

        # Consume a few data points
        consumed_count = 0
        async for data_point in kafka_source.consume():
            assert isinstance(data_point, DataPoint)
            assert data_point.source_id == "kafka_simulator"
            assert "feature_1" in data_point.data

            consumed_count += 1
            if consumed_count >= 5:  # Consume only 5 points for testing
                break

        await kafka_source.disconnect()

    @pytest.mark.asyncio
    async def test_websocket_consume_data(self):
        """Test WebSocket data consumption."""
        ws_source = WebSocketDataSource(websocket_url="wss://example.com/ws")

        await ws_source.connect()

        # Consume a few data points
        consumed_count = 0
        async for data_point in ws_source.consume():
            assert isinstance(data_point, DataPoint)
            assert data_point.source_id == "websocket_simulator"
            assert "sensor_1" in data_point.data

            consumed_count += 1
            if consumed_count >= 5:  # Consume only 5 points for testing
                break

        await ws_source.disconnect()


class TestRealTimeAnomalyPipeline:
    """Test real-time anomaly pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, streaming_config, mock_data_source):
        """Test pipeline initialization."""
        detector = StreamingAnomalyDetector()

        pipeline = RealTimeAnomalyPipeline(
            config=streaming_config,
            data_source=mock_data_source,
            detector=detector,
        )

        assert pipeline.config == streaming_config
        assert pipeline.data_source == mock_data_source
        assert pipeline.detector == detector
        assert not pipeline.is_running
        assert isinstance(pipeline.metrics, StreamingMetrics)

    @pytest.mark.asyncio
    async def test_pipeline_start_stop(self, streaming_config, mock_data_source):
        """Test pipeline start and stop."""
        detector = StreamingAnomalyDetector()

        pipeline = RealTimeAnomalyPipeline(
            config=streaming_config,
            data_source=mock_data_source,
            detector=detector,
        )

        # Start pipeline
        await pipeline.start()
        assert pipeline.is_running
        assert pipeline.start_time is not None

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop pipeline
        await pipeline.stop()
        assert not pipeline.is_running

        # Verify data source was connected and disconnected
        mock_data_source.connect.assert_called_once()
        mock_data_source.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_processing(self, streaming_config, mock_data_source):
        """Test pipeline data processing."""
        # Create detector with small window for faster training
        detector = StreamingAnomalyDetector(
            window_size=10,
            retraining_interval=10,
        )

        # Track alerts
        alerts_received = []

        def alert_handler(alert: StreamingAlert):
            alerts_received.append(alert)

        pipeline = RealTimeAnomalyPipeline(
            config=streaming_config,
            data_source=mock_data_source,
            detector=detector,
            alert_handler=alert_handler,
        )

        # Start pipeline
        await pipeline.start()

        # Let it process data for a short time
        await asyncio.sleep(0.5)

        # Stop pipeline
        await pipeline.stop()

        # Check that some data was processed
        assert pipeline.metrics.processed_records > 0

        # Check pipeline status
        status = pipeline.get_status()
        assert status["pipeline_id"] == "test_pipeline"
        assert not status["is_running"]
        assert status["metrics"]["processed_records"] > 0

    @pytest.mark.asyncio
    async def test_alert_generation(self, streaming_config):
        """Test alert generation with anomalous data."""

        # Create a data source that produces anomalous data
        class AnomalousDataSource(DataSource):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def consume(self):
                # Produce normal data first
                for i in range(50):
                    yield DataPoint(
                        timestamp=datetime.utcnow(),
                        data={
                            "feature_1": np.random.normal(0, 1),
                            "feature_2": np.random.normal(0, 1),
                        },
                        source_id="anomalous_source",
                    )
                    await asyncio.sleep(0.001)

                # Then produce anomalous data
                for i in range(10):
                    yield DataPoint(
                        timestamp=datetime.utcnow(),
                        data={
                            "feature_1": np.random.normal(10, 1),  # Anomalous
                            "feature_2": np.random.normal(10, 1),  # Anomalous
                        },
                        source_id="anomalous_source",
                    )
                    await asyncio.sleep(0.001)

        detector = StreamingAnomalyDetector(
            window_size=20,
            retraining_interval=20,
            contamination=0.1,
        )

        alerts_received = []

        def alert_handler(alert: StreamingAlert):
            alerts_received.append(alert)

        pipeline = RealTimeAnomalyPipeline(
            config=streaming_config,
            data_source=AnomalousDataSource(),
            detector=detector,
            alert_handler=alert_handler,
        )

        # Start pipeline
        await pipeline.start()

        # Let it process data
        await asyncio.sleep(2.0)

        # Stop pipeline
        await pipeline.stop()

        # Should have detected some anomalies
        assert pipeline.metrics.anomalies_detected > 0

        # Should have generated some alerts
        assert len(alerts_received) > 0

        # Check alert structure
        alert = alerts_received[0]
        assert isinstance(alert, StreamingAlert)
        assert alert.alert_type == "anomaly_detected"
        assert alert.severity in [
            AlertSeverity.LOW,
            AlertSeverity.MEDIUM,
            AlertSeverity.HIGH,
            AlertSeverity.CRITICAL,
        ]

    @pytest.mark.asyncio
    async def test_backpressure_handling(self, streaming_config):
        """Test backpressure handling."""

        # Create a fast data source
        class FastDataSource(DataSource):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def consume(self):
                for i in range(2000):  # Produce lots of data quickly
                    yield DataPoint(
                        timestamp=datetime.utcnow(),
                        data={"feature_1": i},
                        source_id="fast_source",
                    )
                    # No sleep - produce data as fast as possible

        # Reduce buffer size to trigger backpressure
        config = streaming_config
        config.max_buffer_size = 100

        detector = StreamingAnomalyDetector()

        pipeline = RealTimeAnomalyPipeline(
            config=config,
            data_source=FastDataSource(),
            detector=detector,
        )

        # Start pipeline
        await pipeline.start()

        # Let it run briefly
        await asyncio.sleep(0.5)

        # Stop pipeline
        await pipeline.stop()

        # Should have triggered backpressure events
        assert pipeline.metrics.backpressure_events > 0

    @pytest.mark.asyncio
    async def test_metrics_collection(self, streaming_config, mock_data_source):
        """Test metrics collection."""
        detector = StreamingAnomalyDetector()

        pipeline = RealTimeAnomalyPipeline(
            config=streaming_config,
            data_source=mock_data_source,
            detector=detector,
        )

        # Start pipeline
        await pipeline.start()

        # Let it run and collect metrics
        await asyncio.sleep(0.5)

        # Stop pipeline
        await pipeline.stop()

        # Check metrics
        metrics = pipeline.get_metrics()
        assert metrics.processed_records > 0
        assert metrics.pipeline_uptime.total_seconds() > 0
        assert metrics.last_processed_at is not None

    def test_alert_severity_determination(self, streaming_config, mock_data_source):
        """Test alert severity determination."""
        detector = StreamingAnomalyDetector()

        pipeline = RealTimeAnomalyPipeline(
            config=streaming_config,
            data_source=mock_data_source,
            detector=detector,
        )

        # Test different anomaly scores
        assert pipeline._determine_alert_severity(0.5) == AlertSeverity.LOW
        assert pipeline._determine_alert_severity(1.2) == AlertSeverity.MEDIUM
        assert pipeline._determine_alert_severity(1.7) == AlertSeverity.HIGH
        assert pipeline._determine_alert_severity(2.5) == AlertSeverity.CRITICAL

        # Test negative scores
        assert pipeline._determine_alert_severity(-1.7) == AlertSeverity.HIGH
        assert pipeline._determine_alert_severity(-2.5) == AlertSeverity.CRITICAL


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_data_source_connection_error(self, streaming_config):
        """Test handling of data source connection errors."""

        class FailingDataSource(DataSource):
            async def connect(self):
                raise ConnectionError("Failed to connect to data source")

            async def disconnect(self):
                pass

            async def consume(self):
                # This shouldn't be called
                yield DataPoint(timestamp=datetime.utcnow(), data={})

        detector = StreamingAnomalyDetector()

        pipeline = RealTimeAnomalyPipeline(
            config=streaming_config,
            data_source=FailingDataSource(),
            detector=detector,
        )

        # Starting pipeline should fail
        with pytest.raises(ConnectionError):
            await pipeline.start()

        # Pipeline should not be running
        assert not pipeline.is_running

    @pytest.mark.asyncio
    async def test_processing_timeout(self, streaming_config):
        """Test processing timeout handling."""
        # Set very short timeout
        config = streaming_config
        config.processing_timeout = 0.001  # 1ms timeout

        # Create slow detector
        class SlowDetector(StreamingAnomalyDetector):
            async def process_batch(self, data_points):
                await asyncio.sleep(0.1)  # Sleep longer than timeout
                return []

        mock_data_source = AsyncMock(spec=DataSource)
        mock_data_source.connect = AsyncMock()
        mock_data_source.disconnect = AsyncMock()

        async def mock_consume():
            for i in range(10):
                yield DataPoint(
                    timestamp=datetime.utcnow(),
                    data={"feature_1": i},
                    source_id="test",
                )
                await asyncio.sleep(0.01)

        mock_data_source.consume.return_value = mock_consume()

        detector = SlowDetector()

        pipeline = RealTimeAnomalyPipeline(
            config=config,
            data_source=mock_data_source,
            detector=detector,
        )

        # Start pipeline
        await pipeline.start()

        # Let it run and encounter timeouts
        await asyncio.sleep(0.5)

        # Stop pipeline
        await pipeline.stop()

        # Should have recorded timeout errors
        assert pipeline.metrics.error_count > 0


class TestIntegration:
    """Test integration scenarios."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline operation."""
        # Create realistic configuration
        config = StreamingConfig(
            pipeline_id="integration_test",
            batch_size=5,
            window_size=20,
            window_slide=5,
            max_buffer_size=100,
            processing_timeout=5.0,
            enable_metrics=True,
            alert_thresholds={
                "anomaly_rate": 0.2,
                "processing_latency": 1000.0,
                "error_rate": 0.1,
            },
        )

        # Create data source with mix of normal and anomalous data
        class MixedDataSource(DataSource):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def consume(self):
                # Normal data
                for i in range(100):
                    yield DataPoint(
                        timestamp=datetime.utcnow(),
                        data={
                            "feature_1": np.random.normal(0, 1),
                            "feature_2": np.random.normal(0, 1),
                            "feature_3": np.random.normal(0, 1),
                        },
                        source_id="mixed_source",
                        metadata={"type": "normal", "batch": i // 10},
                    )
                    await asyncio.sleep(0.01)

                # Anomalous data
                for i in range(20):
                    yield DataPoint(
                        timestamp=datetime.utcnow(),
                        data={
                            "feature_1": np.random.normal(5, 1),  # Shifted mean
                            "feature_2": np.random.normal(5, 1),
                            "feature_3": np.random.normal(5, 1),
                        },
                        source_id="mixed_source",
                        metadata={"type": "anomalous", "batch": "anomaly"},
                    )
                    await asyncio.sleep(0.01)

        # Create detector
        detector = StreamingAnomalyDetector(
            detector_algorithm="isolation_forest",
            contamination=0.1,
            window_size=30,
            retraining_interval=50,
        )

        # Track results
        alerts_received = []
        anomalies_detected = 0

        def alert_handler(alert: StreamingAlert):
            nonlocal anomalies_detected
            alerts_received.append(alert)
            if alert.alert_type == "anomaly_detected":
                anomalies_detected += 1

        # Create pipeline
        pipeline = RealTimeAnomalyPipeline(
            config=config,
            data_source=MixedDataSource(),
            detector=detector,
            alert_handler=alert_handler,
        )

        # Run pipeline
        await pipeline.start()

        # Let it process all data
        await asyncio.sleep(3.0)

        # Stop pipeline
        await pipeline.stop()

        # Verify results
        assert pipeline.metrics.processed_records > 100
        assert anomalies_detected > 0
        assert len(alerts_received) > 0

        # Check final status
        status = pipeline.get_status()
        assert status["samples_processed"] > 0
        assert status["detector_trained"]

        # Verify metrics make sense
        metrics = pipeline.get_metrics()
        assert metrics.processing_rate > 0
        assert metrics.average_latency > 0
        assert metrics.pipeline_uptime.total_seconds() > 0


@pytest.mark.asyncio
async def test_concurrent_pipelines():
    """Test running multiple pipelines concurrently."""
    # Create multiple pipeline configurations
    configs = []
    for i in range(3):
        config = StreamingConfig(
            pipeline_id=f"concurrent_test_{i}",
            batch_size=5,
            window_size=10,
            max_buffer_size=50,
        )
        configs.append(config)

    # Create simple data source
    class SimpleDataSource(DataSource):
        def __init__(self, source_id):
            self.source_id = source_id

        async def connect(self):
            pass

        async def disconnect(self):
            pass

        async def consume(self):
            for i in range(50):
                yield DataPoint(
                    timestamp=datetime.utcnow(),
                    data={"feature_1": np.random.normal(0, 1)},
                    source_id=self.source_id,
                )
                await asyncio.sleep(0.02)

    # Create pipelines
    pipelines = []
    for i, config in enumerate(configs):
        detector = StreamingAnomalyDetector(window_size=10, retraining_interval=20)
        pipeline = RealTimeAnomalyPipeline(
            config=config,
            data_source=SimpleDataSource(f"source_{i}"),
            detector=detector,
        )
        pipelines.append(pipeline)

    # Start all pipelines concurrently
    await asyncio.gather(*[pipeline.start() for pipeline in pipelines])

    # Let them run
    await asyncio.sleep(1.0)

    # Stop all pipelines
    await asyncio.gather(*[pipeline.stop() for pipeline in pipelines])

    # Verify all pipelines processed data
    for pipeline in pipelines:
        assert pipeline.metrics.processed_records > 0
