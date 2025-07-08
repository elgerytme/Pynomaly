"""Tests for streaming components - MetricsPublisher and WebSocketGateway."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

# Import our streaming components directly
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pynomaly.infrastructure.streaming.real_time_anomaly_pipeline import MetricsPublisher, StreamingMetrics
from pynomaly.infrastructure.streaming.websocket_gateway import WebSocketGateway, RealTimeMetrics


class TestMetricsPublisher:
    """Test MetricsPublisher functionality."""
    
    def test_redis_publisher_init(self):
        """Test Redis publisher initialization."""
        config = {"url": "redis://localhost:6379"}
        publisher = MetricsPublisher("redis", config)
        assert publisher.queue_type == "redis"
        assert publisher.config == config
    
    def test_kafka_publisher_init(self):
        """Test Kafka publisher initialization."""
        config = {"bootstrap_servers": "localhost:9092"}
        publisher = MetricsPublisher("kafka", config)
        assert publisher.queue_type == "kafka"
        assert publisher.config == config
    
    def test_unsupported_queue_type(self):
        """Test that unsupported queue types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported queue type"):
            MetricsPublisher("unsupported", {})
    
    @pytest.mark.asyncio
    @patch('aioredis.from_url')
    async def test_redis_publish(self, mock_redis):
        """Test publishing metrics to Redis."""
        # Mock Redis client
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client
        
        # Create publisher
        publisher = MetricsPublisher("redis", {"url": "redis://localhost:6379"})
        
        # Create test metrics
        metrics = StreamingMetrics(
            processed_records=100,
            anomalies_detected=5,
            processing_rate=10.0,
            average_latency=50.0,
            error_count=0
        )
        
        # Test publish
        await publisher.publish(metrics)
        
        # Verify Redis client was called
        mock_client.xadd.assert_called_once()
        call_args = mock_client.xadd.call_args
        assert call_args[0][0] == "metrics_stream"
        assert "metrics" in call_args[0][1]
    
    @pytest.mark.asyncio
    @patch('aiokafka.AIOKafkaProducer')
    async def test_kafka_publish(self, mock_producer_class):
        """Test publishing metrics to Kafka."""
        # Mock Kafka producer
        mock_producer = AsyncMock()
        mock_producer_class.return_value = mock_producer
        
        # Create publisher
        publisher = MetricsPublisher("kafka", {"bootstrap_servers": "localhost:9092"})
        
        # Create test metrics
        metrics = StreamingMetrics(
            processed_records=100,
            anomalies_detected=5,
            processing_rate=10.0,
            average_latency=50.0,
            error_count=0
        )
        
        # Test publish
        await publisher.publish(metrics)
        
        # Verify Kafka producer was called
        mock_producer.send_and_wait.assert_called_once()
        call_args = mock_producer.send_and_wait.call_args
        assert call_args[0][0] == "metrics"
        assert isinstance(call_args[0][1], bytes)


class TestWebSocketGateway:
    """Test WebSocketGateway functionality."""
    
    @pytest.fixture
    def gateway(self):
        """Create a WebSocketGateway instance for testing."""
        return WebSocketGateway(
            update_interval=5,
            heartbeat_interval=10,
            connection_timeout=30,
            max_connections_per_dashboard=10
        )
    
    def test_gateway_init(self, gateway):
        """Test WebSocketGateway initialization."""
        assert gateway.update_interval == 5
        assert gateway.heartbeat_interval == 10
        assert gateway.connection_timeout == 30
        assert gateway.max_connections_per_dashboard == 10
        assert not gateway.is_running
        assert len(gateway.connections) == 0
    
    @pytest.mark.asyncio
    async def test_gateway_start_stop(self, gateway):
        """Test starting and stopping the gateway."""
        # Start gateway
        await gateway.start()
        assert gateway.is_running
        assert gateway.heartbeat_task is not None
        assert gateway.cleanup_task is not None
        
        # Stop gateway
        await gateway.stop()
        assert not gateway.is_running
        assert len(gateway.connections) == 0
    
    @pytest.mark.asyncio
    async def test_connect_dashboard(self, gateway):
        """Test connecting to a dashboard."""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        # Start gateway
        await gateway.start()
        
        # Connect to dashboard
        dashboard_id = "test_dashboard"
        connection_id = await gateway.connect_dashboard(mock_websocket, dashboard_id)
        
        # Verify connection was established
        assert connection_id in gateway.connections
        assert dashboard_id in gateway.dashboard_connections
        assert connection_id in gateway.dashboard_connections[dashboard_id]
        
        # Verify WebSocket was accepted
        mock_websocket.accept.assert_called_once()
        
        # Clean up
        await gateway.stop()
    
    @pytest.mark.asyncio
    async def test_connection_limits(self, gateway):
        """Test connection limits per dashboard."""
        # Start gateway
        await gateway.start()
        
        # Create max connections
        for i in range(gateway.max_connections_per_dashboard):
            mock_websocket = AsyncMock()
            mock_websocket.accept = AsyncMock()
            await gateway.connect_dashboard(mock_websocket, "test_dashboard")
        
        # Try to exceed limit
        mock_websocket = AsyncMock()
        with pytest.raises(ValueError, match="has reached maximum connections"):
            await gateway.connect_dashboard(mock_websocket, "test_dashboard")
        
        # Clean up
        await gateway.stop()
    
    @pytest.mark.asyncio
    async def test_update_dashboard_metrics(self, gateway):
        """Test updating dashboard metrics."""
        # Start gateway
        await gateway.start()
        
        # Create mock connection
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        dashboard_id = "test_dashboard"
        connection_id = await gateway.connect_dashboard(mock_websocket, dashboard_id)
        
        # Create test metrics
        metrics = RealTimeMetrics(
            dashboard_id=dashboard_id,
            session_id="test_session",
            metrics=StreamingMetrics(
                processed_records=100,
                anomalies_detected=5,
                processing_rate=10.0,
                average_latency=50.0,
                error_count=0
            )
        )
        
        # Update metrics
        await gateway.update_dashboard_metrics(dashboard_id, metrics)
        
        # Verify metrics were stored
        assert dashboard_id in gateway.latest_metrics
        assert gateway.latest_metrics[dashboard_id] == metrics
        
        # Clean up
        await gateway.stop()
    
    @pytest.mark.asyncio
    async def test_heartbeat_functionality(self, gateway):
        """Test heartbeat functionality."""
        # Start gateway
        await gateway.start()
        
        # Create mock connection
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        dashboard_id = "test_dashboard"
        connection_id = await gateway.connect_dashboard(mock_websocket, dashboard_id)
        
        # Wait for heartbeat
        await asyncio.sleep(0.1)
        
        # Handle heartbeat message
        heartbeat_message = {"type": "heartbeat"}
        await gateway.handle_client_message(connection_id, heartbeat_message)
        
        # Verify heartbeat was updated
        connection = gateway.connections[connection_id]
        assert connection.last_heartbeat is not None
        
        # Clean up
        await gateway.stop()
    
    @pytest.mark.asyncio
    async def test_back_pressure_handling(self, gateway):
        """Test back-pressure handling."""
        # Start gateway with small queue size
        gateway.max_message_queue_size = 2
        await gateway.start()
        
        # Create mock connection
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        dashboard_id = "test_dashboard"
        connection_id = await gateway.connect_dashboard(mock_websocket, dashboard_id)
        
        # Fill queue beyond capacity
        for i in range(5):
            await gateway._send_to_connection(connection_id, {"test": f"message_{i}"})
        
        # Verify back-pressure was handled
        assert gateway.stats["messages_dropped"] > 0
        
        # Clean up
        await gateway.stop()
    
    def test_gateway_stats(self, gateway):
        """Test gateway statistics."""
        stats = gateway.get_stats()
        
        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "messages_sent" in stats
        assert "messages_dropped" in stats
        assert "errors" in stats
        assert "dashboard_count" in stats
        assert "connections_per_dashboard" in stats


@pytest.mark.asyncio
async def test_metrics_publisher_integration():
    """Test MetricsPublisher integration with StreamingMetrics."""
    # Create mock metrics
    metrics = StreamingMetrics(
        processed_records=1000,
        anomalies_detected=50,
        processing_rate=100.0,
        average_latency=25.0,
        error_count=2,
        backpressure_events=1,
        last_processed_at=datetime.utcnow(),
        pipeline_uptime=timedelta(hours=1),
        memory_usage_mb=512.0,
        cpu_usage_percent=65.0
    )
    
    # Test JSON serialization
    json_str = metrics.json()
    assert json_str is not None
    
    # Test deserialization
    parsed = json.loads(json_str)
    assert parsed["processed_records"] == 1000
    assert parsed["anomalies_detected"] == 50


@pytest.mark.asyncio
async def test_websocket_gateway_integration():
    """Test WebSocketGateway integration with RealTimeMetrics."""
    gateway = WebSocketGateway(update_interval=1, heartbeat_interval=5)
    
    try:
        await gateway.start()
        
        # Create test metrics
        metrics = RealTimeMetrics(
            dashboard_id="integration_test",
            session_id="test_session",
            metrics=StreamingMetrics(
                processed_records=100,
                anomalies_detected=5,
                processing_rate=10.0,
                average_latency=50.0,
                error_count=0
            )
        )
        
        # Update metrics
        await gateway.update_dashboard_metrics("integration_test", metrics)
        
        # Verify metrics were stored
        assert "integration_test" in gateway.latest_metrics
        
        # Test alert sending
        alert_data = {
            "severity": "high",
            "message": "High error rate detected",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await gateway.send_alert("integration_test", alert_data)
        
        # Verify gateway stats
        stats = gateway.get_stats()
        assert stats["dashboard_count"] == 1
        
    finally:
        await gateway.stop()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
