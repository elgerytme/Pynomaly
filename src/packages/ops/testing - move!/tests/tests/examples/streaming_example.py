"""
Example usage of the enhanced streaming backend and WebSocket gateway.

This example demonstrates:
1. MetricsPublisher sending RealTimeMetrics to Redis Streams/Kafka
2. WebSocket Gateway with dashboard multiplexing
3. Real-time updates with heartbeat and back-pressure handling
4. Reconnection logic and error handling
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock streaming components (in production, these would import from the real modules)
from monorepo.infrastructure.streaming.real_time_anomaly_pipeline import (
    StreamingMetrics,
)
from monorepo.infrastructure.streaming.websocket_gateway import (
    RealTimeMetrics,
    WebSocketGateway,
)


# Mock metrics publisher
class MetricsPublisher:
    """Mock metrics publisher for demonstration."""

    def __init__(self, queue_type: str, config: dict[str, Any]):
        self.queue_type = queue_type
        self.config = config
        self.published_count = 0

    async def start(self):
        """Start the publisher."""
        logger.info(f"Starting {self.queue_type} metrics publisher")

    async def stop(self):
        """Stop the publisher."""
        logger.info(f"Stopping {self.queue_type} metrics publisher")

    async def publish(self, metrics: StreamingMetrics):
        """Publish metrics to the queue."""
        self.published_count += 1
        logger.info(f"Published metrics #{self.published_count} to {self.queue_type}")
        logger.debug(f"Metrics: {metrics.dict()}")


# Mock WebSocket for demonstration
class MockWebSocket:
    """Mock WebSocket for demonstration."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.messages = []
        self.connected = False

    async def accept(self):
        """Accept the WebSocket connection."""
        self.connected = True
        logger.info(f"WebSocket {self.client_id} accepted")

    async def send_text(self, message: str):
        """Send text message."""
        self.messages.append(message)
        data = json.loads(message)
        logger.info(
            f"WebSocket {self.client_id} received: {data.get('type', 'unknown')}"
        )

    async def close(self):
        """Close the WebSocket connection."""
        self.connected = False
        logger.info(f"WebSocket {self.client_id} closed")


class StreamingExample:
    """Comprehensive streaming example."""

    def __init__(self):
        # Initialize components
        self.metrics_publisher = MetricsPublisher(
            "kafka",
            {"bootstrap_servers": ["localhost:9092"], "topic": "realtime_metrics"},
        )

        self.websocket_gateway = WebSocketGateway(
            update_interval=5,  # Update every 5 seconds
            heartbeat_interval=30,  # Heartbeat every 30 seconds
            connection_timeout=60,  # 60 second timeout
            max_connections_per_dashboard=100,
            max_message_queue_size=1000,
        )

        # Mock dashboard connections
        self.dashboard_connections = {}

        # Simulation state
        self.simulation_running = False
        self.metrics_generator_task = None
        self.client_simulator_task = None

    async def start(self):
        """Start the streaming example."""
        logger.info("Starting streaming example...")

        # Start components
        await self.metrics_publisher.start()
        await self.websocket_gateway.start()

        # Start simulation tasks
        self.simulation_running = True
        self.metrics_generator_task = asyncio.create_task(self._generate_metrics())
        self.client_simulator_task = asyncio.create_task(self._simulate_clients())

        logger.info("Streaming example started successfully")

    async def stop(self):
        """Stop the streaming example."""
        logger.info("Stopping streaming example...")

        # Stop simulation
        self.simulation_running = False

        # Cancel tasks
        if self.metrics_generator_task:
            self.metrics_generator_task.cancel()
        if self.client_simulator_task:
            self.client_simulator_task.cancel()

        # Stop components
        await self.websocket_gateway.stop()
        await self.metrics_publisher.stop()

        logger.info("Streaming example stopped")

    async def _generate_metrics(self):
        """Generate mock metrics and publish them."""
        try:
            counter = 0
            while self.simulation_running:
                counter += 1

                # Generate mock metrics for different dashboards
                dashboards = ["dashboard_1", "dashboard_2", "dashboard_3"]

                for dashboard_id in dashboards:
                    # Create streaming metrics
                    streaming_metrics = StreamingMetrics(
                        processed_records=counter * 100,
                        anomalies_detected=counter * 5,
                        processing_rate=50.0 + (counter % 10),
                        average_latency=25.0 + (counter % 5),
                        error_count=counter % 3,
                        backpressure_events=counter % 7,
                        last_processed_at=datetime.utcnow(),
                        pipeline_uptime=timedelta(seconds=counter * 10),
                        memory_usage_mb=512.0 + (counter % 100),
                        cpu_usage_percent=65.0 + (counter % 20),
                    )

                    # Publish to message queue
                    await self.metrics_publisher.publish(streaming_metrics)

                    # Create real-time metrics for WebSocket
                    realtime_metrics = RealTimeMetrics(
                        dashboard_id=dashboard_id,
                        session_id=f"session_{dashboard_id}",
                        metrics=streaming_metrics,
                        uptime_seconds=counter * 10.0,
                        status="active",
                    )

                    # Update dashboard metrics
                    await self.websocket_gateway.update_dashboard_metrics(
                        dashboard_id, realtime_metrics
                    )

                    # Simulate alerts for high error rates
                    if streaming_metrics.error_count > 2:
                        alert_data = {
                            "severity": "high",
                            "message": f"High error count: {streaming_metrics.error_count}",
                            "timestamp": datetime.utcnow().isoformat(),
                            "dashboard_id": dashboard_id,
                        }
                        await self.websocket_gateway.send_alert(
                            dashboard_id, alert_data
                        )

                # Wait before next iteration
                await asyncio.sleep(10)  # Generate metrics every 10 seconds

        except asyncio.CancelledError:
            logger.info("Metrics generation stopped")
        except Exception as e:
            logger.error(f"Error in metrics generation: {e}")

    async def _simulate_clients(self):
        """Simulate WebSocket clients connecting and disconnecting."""
        try:
            client_counter = 0
            while self.simulation_running:
                client_counter += 1

                # Simulate client connecting
                client_id = f"client_{client_counter}"
                dashboard_id = f"dashboard_{(client_counter % 3) + 1}"

                mock_websocket = MockWebSocket(client_id)

                try:
                    # Connect to dashboard
                    connection_id = await self.websocket_gateway.connect_dashboard(
                        mock_websocket, dashboard_id
                    )

                    # Store connection
                    self.dashboard_connections[connection_id] = {
                        "websocket": mock_websocket,
                        "dashboard_id": dashboard_id,
                        "connected_at": datetime.utcnow(),
                    }

                    # Simulate client staying connected for a while
                    await asyncio.sleep(30)

                    # Simulate client disconnecting
                    await self.websocket_gateway.disconnect_dashboard(connection_id)

                    # Clean up
                    if connection_id in self.dashboard_connections:
                        del self.dashboard_connections[connection_id]

                    logger.info(f"Client {client_id} simulation completed")

                except Exception as e:
                    logger.error(f"Error simulating client {client_id}: {e}")

                # Wait before next client
                await asyncio.sleep(15)  # New client every 15 seconds

        except asyncio.CancelledError:
            logger.info("Client simulation stopped")
        except Exception as e:
            logger.error(f"Error in client simulation: {e}")

    async def demonstrate_features(self):
        """Demonstrate various features of the streaming system."""
        logger.info("Demonstrating streaming features...")

        # 1. Show current statistics
        stats = self.websocket_gateway.get_stats()
        logger.info(f"Gateway stats: {stats}")

        # 2. Demonstrate connection limits
        logger.info("Testing connection limits...")
        test_connections = []
        dashboard_id = "test_dashboard"

        # Try to create many connections
        for i in range(5):
            try:
                mock_websocket = MockWebSocket(f"test_client_{i}")
                connection_id = await self.websocket_gateway.connect_dashboard(
                    mock_websocket, dashboard_id
                )
                test_connections.append(connection_id)
                logger.info(f"Created test connection {i}: {connection_id}")
            except Exception as e:
                logger.error(f"Failed to create test connection {i}: {e}")

        # 3. Test metrics update
        logger.info("Testing metrics update...")
        test_metrics = RealTimeMetrics(
            dashboard_id=dashboard_id,
            session_id="test_session",
            metrics=StreamingMetrics(
                processed_records=9999,
                anomalies_detected=999,
                processing_rate=999.0,
                average_latency=99.0,
                error_count=99,
            ),
        )
        await self.websocket_gateway.update_dashboard_metrics(
            dashboard_id, test_metrics
        )

        # 4. Test alert sending
        logger.info("Testing alert sending...")
        alert_data = {
            "severity": "critical",
            "message": "Test alert for demonstration",
            "timestamp": datetime.utcnow().isoformat(),
            "dashboard_id": dashboard_id,
        }
        await self.websocket_gateway.send_alert(dashboard_id, alert_data)

        # 5. Clean up test connections
        logger.info("Cleaning up test connections...")
        for connection_id in test_connections:
            try:
                await self.websocket_gateway.disconnect_dashboard(connection_id)
            except Exception as e:
                logger.error(f"Error disconnecting {connection_id}: {e}")

        # 6. Show final statistics
        final_stats = self.websocket_gateway.get_stats()
        logger.info(f"Final gateway stats: {final_stats}")

        logger.info("Feature demonstration completed")


async def main():
    """Main function to run the streaming example."""
    example = StreamingExample()

    try:
        # Start the example
        await example.start()

        # Let it run for a while
        logger.info("Running streaming example for 60 seconds...")
        await asyncio.sleep(60)

        # Demonstrate features
        await example.demonstrate_features()

        # Let it run a bit more
        logger.info("Continuing for another 30 seconds...")
        await asyncio.sleep(30)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Clean up
        await example.stop()
        logger.info("Example completed")


if __name__ == "__main__":
    asyncio.run(main())
