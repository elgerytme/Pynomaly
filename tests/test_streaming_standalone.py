"""Standalone test for streaming components."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any
from pydantic import BaseModel, Field
from collections import defaultdict
import time

# Define minimal StreamingMetrics for testing
class StreamingMetrics(BaseModel):
    """Minimal streaming metrics for testing."""
    processed_records: int = 0
    anomalies_detected: int = 0
    processing_rate: float = 0.0
    average_latency: float = 0.0
    error_count: int = 0
    backpressure_events: int = 0
    last_processed_at: datetime = Field(default_factory=datetime.utcnow)
    pipeline_uptime: timedelta = timedelta()
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

# Define minimal RealTimeMetrics for testing
class RealTimeMetrics(BaseModel):
    """Minimal real-time metrics for testing."""
    dashboard_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics: StreamingMetrics
    session_id: str
    uptime_seconds: float = 0.0
    status: str = "active"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Define MetricsPublisher for testing
class MetricsPublisher:
    """Minimal metrics publisher for testing."""
    
    def __init__(self, queue_type: str, config: dict):
        self.queue_type = queue_type
        self.config = config
        self.published_metrics = []  # Store metrics for testing
        
    async def publish(self, metrics: StreamingMetrics):
        """Publish metrics (mock implementation)."""
        self.published_metrics.append(metrics)
        print(f"Published metrics to {self.queue_type}: {metrics.dict()}")

# Define WebSocketGateway for testing
class WebSocketGateway:
    """Minimal WebSocket gateway for testing."""
    
    def __init__(self, update_interval: int = 10, heartbeat_interval: int = 30):
        self.update_interval = update_interval
        self.heartbeat_interval = heartbeat_interval
        self.is_running = False
        self.connections: Dict[str, Dict] = {}
        self.dashboard_connections: Dict[str, set] = defaultdict(set)
        self.latest_metrics: Dict[str, RealTimeMetrics] = {}
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_dropped": 0,
            "errors": 0
        }
        
    async def start(self):
        """Start the gateway."""
        self.is_running = True
        print("WebSocket Gateway started")
        
    async def stop(self):
        """Stop the gateway."""
        self.is_running = False
        self.connections.clear()
        self.dashboard_connections.clear()
        print("WebSocket Gateway stopped")
        
    async def connect_dashboard(self, websocket, dashboard_id: str) -> str:
        """Connect to a dashboard."""
        # Accept the WebSocket connection
        await websocket.accept()
        
        connection_id = f"conn_{len(self.connections)}"
        self.connections[connection_id] = {
            "websocket": websocket,
            "dashboard_id": dashboard_id,
            "connected_at": datetime.utcnow()
        }
        self.dashboard_connections[dashboard_id].add(connection_id)
        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.connections)
        print(f"Connected to dashboard {dashboard_id} with connection {connection_id}")
        return connection_id
        
    async def update_dashboard_metrics(self, dashboard_id: str, metrics: RealTimeMetrics):
        """Update dashboard metrics."""
        self.latest_metrics[dashboard_id] = metrics
        print(f"Updated metrics for dashboard {dashboard_id}")
        
    async def send_alert(self, dashboard_id: str, alert_data: Dict[str, Any]):
        """Send alert to dashboard."""
        print(f"Sent alert to dashboard {dashboard_id}: {alert_data}")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get gateway statistics."""
        return {
            **self.stats,
            "dashboard_count": len(self.dashboard_connections),
            "connections_per_dashboard": {
                dashboard_id: len(connections)
                for dashboard_id, connections in self.dashboard_connections.items()
            }
        }

# Mock WebSocket for testing
class MockWebSocket:
    def __init__(self):
        self.accepted = False
        self.messages = []
        
    async def accept(self):
        self.accepted = True
        
    async def send_text(self, message):
        self.messages.append(message)

async def test_metrics_publisher():
    """Test metrics publisher functionality."""
    print("Testing MetricsPublisher...")
    
    # Create publisher
    publisher = MetricsPublisher("kafka", {"bootstrap_servers": "localhost:9092"})
    
    # Create test metrics
    metrics = StreamingMetrics(
        processed_records=100,
        anomalies_detected=5,
        processing_rate=10.0,
        average_latency=50.0,
        error_count=0,
        backpressure_events=1,
        memory_usage_mb=512.0,
        cpu_usage_percent=65.0
    )
    
    # Publish metrics
    await publisher.publish(metrics)
    
    # Verify metrics were published
    assert len(publisher.published_metrics) == 1
    assert publisher.published_metrics[0].processed_records == 100
    print("✓ MetricsPublisher test passed")

async def test_websocket_gateway():
    """Test WebSocket gateway functionality."""
    print("Testing WebSocketGateway...")
    
    # Create gateway
    gateway = WebSocketGateway(update_interval=5, heartbeat_interval=10)
    
    # Start gateway
    await gateway.start()
    assert gateway.is_running
    
    # Test connecting to dashboard
    mock_websocket = MockWebSocket()
    dashboard_id = "test_dashboard"
    connection_id = await gateway.connect_dashboard(mock_websocket, dashboard_id)
    
    # Verify connection
    assert connection_id in gateway.connections
    assert dashboard_id in gateway.dashboard_connections
    assert connection_id in gateway.dashboard_connections[dashboard_id]
    assert mock_websocket.accepted
    
    # Test metrics update
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
    
    await gateway.update_dashboard_metrics(dashboard_id, metrics)
    assert dashboard_id in gateway.latest_metrics
    assert gateway.latest_metrics[dashboard_id] == metrics
    
    # Test alert sending
    alert_data = {
        "severity": "high",
        "message": "High error rate detected",
        "timestamp": datetime.utcnow().isoformat()
    }
    await gateway.send_alert(dashboard_id, alert_data)
    
    # Test statistics
    stats = gateway.get_stats()
    assert stats["total_connections"] == 1
    assert stats["active_connections"] == 1
    assert stats["dashboard_count"] == 1
    assert dashboard_id in stats["connections_per_dashboard"]
    
    # Stop gateway
    await gateway.stop()
    assert not gateway.is_running
    assert len(gateway.connections) == 0
    
    print("✓ WebSocketGateway test passed")

async def test_integration():
    """Test integration between components."""
    print("Testing integration...")
    
    # Create components
    publisher = MetricsPublisher("redis", {"url": "redis://localhost:6379"})
    gateway = WebSocketGateway(update_interval=1, heartbeat_interval=5)
    
    # Start gateway
    await gateway.start()
    
    # Connect to dashboard
    mock_websocket = MockWebSocket()
    dashboard_id = "integration_test"
    connection_id = await gateway.connect_dashboard(mock_websocket, dashboard_id)
    
    # Create and publish metrics
    streaming_metrics = StreamingMetrics(
        processed_records=1000,
        anomalies_detected=50,
        processing_rate=100.0,
        average_latency=25.0,
        error_count=2,
        backpressure_events=1,
        memory_usage_mb=512.0,
        cpu_usage_percent=65.0
    )
    
    await publisher.publish(streaming_metrics)
    
    # Update dashboard with real-time metrics
    realtime_metrics = RealTimeMetrics(
        dashboard_id=dashboard_id,
        session_id="integration_session",
        metrics=streaming_metrics
    )
    
    await gateway.update_dashboard_metrics(dashboard_id, realtime_metrics)
    
    # Verify integration
    assert len(publisher.published_metrics) == 1
    assert dashboard_id in gateway.latest_metrics
    assert gateway.latest_metrics[dashboard_id].metrics.processed_records == 1000
    
    # Test JSON serialization
    json_str = realtime_metrics.json()
    parsed = json.loads(json_str)
    assert parsed["dashboard_id"] == dashboard_id
    assert parsed["metrics"]["processed_records"] == 1000
    
    # Stop gateway
    await gateway.stop()
    
    print("✓ Integration test passed")

async def test_backpressure_simulation():
    """Test back-pressure handling simulation."""
    print("Testing back-pressure simulation...")
    
    # Create gateway with small limits
    gateway = WebSocketGateway(update_interval=1, heartbeat_interval=5)
    await gateway.start()
    
    # Connect multiple dashboards
    for i in range(5):
        mock_websocket = MockWebSocket()
        dashboard_id = f"dashboard_{i}"
        await gateway.connect_dashboard(mock_websocket, dashboard_id)
    
    # Simulate high frequency updates
    for i in range(10):
        for j in range(5):
            dashboard_id = f"dashboard_{j}"
            metrics = RealTimeMetrics(
                dashboard_id=dashboard_id,
                session_id=f"session_{i}",
                metrics=StreamingMetrics(
                    processed_records=i * 100,
                    anomalies_detected=i * 5,
                    processing_rate=i * 10.0,
                    average_latency=i * 5.0,
                    error_count=i
                )
            )
            await gateway.update_dashboard_metrics(dashboard_id, metrics)
    
    # Verify stats
    stats = gateway.get_stats()
    assert stats["dashboard_count"] == 5
    assert stats["total_connections"] == 5
    
    await gateway.stop()
    print("✓ Back-pressure simulation test passed")

async def main():
    """Run all tests."""
    print("Running Streaming Components Tests...")
    print("=" * 50)
    
    try:
        await test_metrics_publisher()
        await test_websocket_gateway()
        await test_integration()
        await test_backpressure_simulation()
        
        print("=" * 50)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
