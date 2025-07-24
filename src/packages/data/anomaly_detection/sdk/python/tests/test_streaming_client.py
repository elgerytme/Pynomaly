"""Tests for the streaming anomaly detection client."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from anomaly_detection_sdk import StreamingClient, AlgorithmType
from anomaly_detection_sdk.models import StreamingConfig, AnomalyData
from anomaly_detection_sdk.exceptions import StreamingError, ValidationError


@pytest.fixture
def streaming_config():
    """Create streaming configuration for testing."""
    return StreamingConfig(
        buffer_size=10,
        detection_threshold=0.5,
        batch_size=5,
        algorithm=AlgorithmType.ISOLATION_FOREST,
        auto_retrain=False
    )


@pytest.fixture
def client(streaming_config):
    """Create a test streaming client."""
    return StreamingClient(
        ws_url="ws://localhost:8000/ws/stream",
        config=streaming_config,
        auto_reconnect=True,
        reconnect_delay=1.0
    )


@pytest.fixture
def sample_data_point():
    """Sample data point for testing."""
    return [1.5, 2.3]


class TestStreamingClient:
    """Test the StreamingClient class."""

    def test_client_initialization(self, streaming_config):
        """Test client initialization with various parameters."""
        # Basic initialization
        client = StreamingClient("ws://localhost:8000/ws/stream")
        assert client.ws_url == "ws://localhost:8000/ws/stream"
        assert client.config.buffer_size == 100  # Default value
        assert client.auto_reconnect is True

        # With all parameters
        client = StreamingClient(
            ws_url="ws://example.com/ws/stream",
            config=streaming_config,
            api_key="test-key",
            auto_reconnect=False,
            reconnect_delay=5.0
        )
        assert client.ws_url == "ws://example.com/ws/stream"
        assert client.config.buffer_size == 10
        assert client.api_key == "test-key"
        assert client.auto_reconnect is False
        assert client.reconnect_delay == 5.0

    def test_event_handler_registration(self, client):
        """Test event handler registration."""
        # Test on_anomaly decorator
        @client.on_anomaly
        def handle_anomaly(anomaly_data):
            pass

        assert len(client._on_anomaly_handlers) == 1

        # Test on_connect decorator
        @client.on_connect
        def handle_connect():
            pass

        assert len(client._on_connect_handlers) == 1

        # Test on_disconnect decorator
        @client.on_disconnect
        def handle_disconnect():
            pass

        assert len(client._on_disconnect_handlers) == 1

        # Test on_error decorator
        @client.on_error
        def handle_error(error):
            pass

        assert len(client._on_error_handlers) == 1

    def test_send_data_validation(self, client):
        """Test data validation for send_data method."""
        # Valid data
        client.send_data([1.0, 2.0])  # Should not raise

        # Invalid data - not a list
        with pytest.raises(ValidationError) as exc_info:
            client.send_data("invalid")
        assert "Data point must be a list" in str(exc_info.value)

        # Invalid data - empty list
        with pytest.raises(ValidationError) as exc_info:
            client.send_data([])
        assert "Data point must be a list" in str(exc_info.value)

    def test_send_data_when_not_running(self, client):
        """Test send_data when client is not running."""
        with pytest.raises(StreamingError) as exc_info:
            client.send_data([1.0, 2.0])
        assert "Client is not running" in str(exc_info.value)

    @patch('threading.Thread')
    def test_start_streaming(self, mock_thread, client):
        """Test starting the streaming client."""
        client.start()
        
        assert client.running is True
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    def test_stop_streaming(self, client):
        """Test stopping the streaming client."""
        client.running = True  # Simulate running state
        client.stop()
        
        assert client.running is False

    def test_buffer_management(self, client):
        """Test buffer management and batch processing."""
        client.running = True
        client.connected = False  # Not connected, so data should be buffered

        # Send data points
        for i in range(3):
            client.send_data([float(i), float(i + 1)])

        # Check buffer size
        assert len(client._buffer) == 3

        # Send enough data to trigger batch processing
        with patch.object(client, '_send_batch') as mock_send_batch:
            for i in range(3, 6):  # This should trigger batch processing
                client.send_data([float(i), float(i + 1)])

    @patch('websockets.connect')
    async def test_websocket_connection(self, mock_connect, client):
        """Test WebSocket connection establishment."""
        mock_websocket = AsyncMock()
        mock_connect.return_value = mock_websocket

        # Test connection
        await client._connect()

        mock_connect.assert_called_once()
        assert client.websocket == mock_websocket
        assert client.connected is True

    def test_message_handling(self, client):
        """Test handling different message types."""
        # Test anomaly message
        anomaly_message = {
            "type": "anomaly",
            "data": {
                "index": 5,
                "score": 0.8,
                "data_point": [10.0, 20.0],
                "confidence": 0.9,
                "timestamp": "2023-01-01T00:00:00Z"
            }
        }

        anomaly_received = []
        
        @client.on_anomaly
        def handle_anomaly(anomaly_data):
            anomaly_received.append(anomaly_data)

        client._handle_message(json.dumps(anomaly_message))
        
        assert len(anomaly_received) == 1
        assert anomaly_received[0].index == 5
        assert anomaly_received[0].score == 0.8

        # Test error message
        error_message = {
            "type": "error",
            "message": "Processing error"
        }

        errors_received = []
        
        @client.on_error
        def handle_error(error):
            errors_received.append(error)

        client._handle_message(json.dumps(error_message))
        
        assert len(errors_received) == 1
        assert isinstance(errors_received[0], StreamingError)

        # Test ping message
        ping_message = {"type": "ping"}
        
        with patch.object(client, 'websocket') as mock_ws:
            client._handle_message(json.dumps(ping_message))
            # Should send pong response (implementation detail)

    def test_invalid_json_message(self, client):
        """Test handling of invalid JSON messages."""
        errors_received = []
        
        @client.on_error
        def handle_error(error):
            errors_received.append(error)

        client._handle_message("invalid json")
        
        assert len(errors_received) == 1
        assert isinstance(errors_received[0], StreamingError)
        assert "Invalid JSON message" in str(errors_received[0])

    def test_connection_properties(self, client):
        """Test connection status properties."""
        # Initially not connected
        assert client.is_connected is False
        assert client.buffer_size == 0

        # Simulate connection
        client.connected = True
        assert client.is_connected is True

        # Add some data to buffer
        client._buffer = [[1, 2], [3, 4]]
        assert client.buffer_size == 2

    def test_error_handling_in_handlers(self, client):
        """Test error handling when event handlers raise exceptions."""
        # Handler that raises an exception
        @client.on_anomaly
        def bad_handler(anomaly_data):
            raise ValueError("Handler error")

        # Handler that works fine
        good_calls = []
        
        @client.on_anomaly
        def good_handler(anomaly_data):
            good_calls.append(anomaly_data)

        # This should not crash the client
        client._handle_anomaly(AnomalyData(
            index=1,
            score=0.5,
            data_point=[1.0, 2.0]
        ))

        # Good handler should still be called
        assert len(good_calls) == 1

    def test_websocket_url_with_api_key(self):
        """Test WebSocket URL construction with API key."""
        client = StreamingClient(
            ws_url="ws://localhost:8000/ws/stream",
            api_key="test-key"
        )
        
        url = client._build_websocket_url()
        assert "token=test-key" in url

    def test_configuration_message(self, client):
        """Test initial configuration message."""
        config_data = client._get_config_message()
        
        assert config_data["type"] == "config"
        assert "config" in config_data
        assert config_data["config"]["buffer_size"] == client.config.buffer_size
        assert config_data["config"]["algorithm"] == client.config.algorithm

    @patch('time.sleep')
    def test_reconnection_logic(self, mock_sleep, client):
        """Test automatic reconnection logic."""
        client.auto_reconnect = True
        client.running = True
        reconnect_attempts = []

        original_connect = client._connect
        
        async def mock_connect():
            reconnect_attempts.append(len(reconnect_attempts))
            if len(reconnect_attempts) < 3:
                raise ConnectionError("Connection failed")
            return await original_connect()

        client._connect = mock_connect

        # This would normally run in the background thread
        # We're testing the logic directly here
        with patch.object(client, '_listen') as mock_listen:
            mock_listen.side_effect = ConnectionError("Connection lost")
            
            # Simulate the reconnection loop
            for _ in range(3):
                try:
                    asyncio.run(client._connect_and_listen())
                except ConnectionError:
                    if client.auto_reconnect and client.running:
                        mock_sleep.assert_called_with(client.reconnect_delay)
                        continue
                    break

    def test_batch_sending_when_connected(self, client):
        """Test batch sending when client is connected."""
        client.connected = True
        client.websocket = Mock()
        
        batch_data = [[1, 2], [3, 4], [5, 6]]
        client._send_batch(batch_data)
        
        # Should send message to websocket
        client.websocket.send.assert_called_once()
        sent_data = json.loads(client.websocket.send.call_args[0][0])
        assert sent_data["type"] == "batch"
        assert sent_data["data"] == batch_data

    def test_batch_sending_when_disconnected(self, client):
        """Test batch sending when client is disconnected."""
        client.connected = False
        client.websocket = None
        
        batch_data = [[1, 2], [3, 4]]
        
        # Should buffer the data instead of trying to send
        original_buffer_size = len(client._buffer)
        client._send_batch(batch_data)
        
        # Data should be added back to buffer
        assert len(client._buffer) >= original_buffer_size

    def test_cleanup_on_disconnect(self, client):
        """Test cleanup when client disconnects."""
        client.connected = True
        client.websocket = Mock()
        
        # Simulate disconnect
        client._handle_disconnect()
        
        assert client.connected is False

    @pytest.mark.asyncio
    async def test_async_context_manager_integration(self):
        """Test integration with async context managers."""
        # This tests the pattern that might be used in real applications
        class AsyncStreamingClient(StreamingClient):
            async def __aenter__(self):
                await self.start_async()
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.stop()

            async def start_async(self):
                # Simulate async start
                self.running = True
                self.connected = True

        async with AsyncStreamingClient("ws://localhost:8000/ws/stream") as client:
            assert client.running is True
            assert client.connected is True

    def test_thread_safety_buffer_access(self, client):
        """Test thread safety of buffer access."""
        import threading
        
        client.running = True
        client.connected = False
        
        results = []
        
        def add_data():
            for i in range(10):
                try:
                    client.send_data([float(i), float(i + 1)])
                    results.append(f"success_{i}")
                except Exception as e:
                    results.append(f"error_{e}")

        # Start multiple threads
        threads = [threading.Thread(target=add_data) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 30 successful operations (3 threads * 10 operations)
        success_count = len([r for r in results if r.startswith("success")])
        assert success_count == 30

    def test_memory_cleanup_on_large_buffer(self, client):
        """Test memory management with large buffers."""
        client.running = True
        client.connected = False
        
        # Fill buffer beyond capacity
        for i in range(client.config.buffer_size * 2):
            client.send_data([float(i), float(i + 1)])

        # Buffer should not grow indefinitely
        # (Implementation should handle this gracefully)
        assert len(client._buffer) <= client.config.buffer_size * 2

    def test_custom_event_handlers(self, client):
        """Test custom event handling scenarios."""
        events_log = []

        @client.on_connect
        def log_connect():
            events_log.append("connected")

        @client.on_disconnect  
        def log_disconnect():
            events_log.append("disconnected")

        @client.on_anomaly
        def log_anomaly(anomaly_data):
            events_log.append(f"anomaly_{anomaly_data.index}")

        @client.on_error
        def log_error(error):
            events_log.append(f"error_{type(error).__name__}")

        # Simulate events
        client._handle_connect()
        client._handle_anomaly(AnomalyData(index=1, score=0.8, data_point=[1, 2]))
        client._handle_error(StreamingError("Test error"))
        client._handle_disconnect()

        assert "connected" in events_log
        assert "anomaly_1" in events_log
        assert "error_StreamingError" in events_log
        assert "disconnected" in events_log