"""Integration tests for Redis message queue adapter."""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from pynomaly.infrastructure.messaging.adapters.redis_queue_adapter import RedisQueueAdapter
from pynomaly.infrastructure.messaging.config.messaging_settings import MessagingSettings
from pynomaly.infrastructure.messaging.models.messages import Message, MessagePriority
from pynomaly.infrastructure.messaging.models.tasks import Task, TaskType


@pytest.fixture
def messaging_settings():
    """Create messaging settings for testing."""
    return MessagingSettings(
        queue_backend="redis",
        redis_queue_db=15,  # Use a test database
        redis_consumer_group="test_workers",
        redis_consumer_name="test_worker",
        worker_concurrency=2,
        task_timeout=30
    )


@pytest.fixture
def redis_url():
    """Redis URL for testing."""
    return "redis://localhost:6379"


@pytest_asyncio.fixture
async def redis_adapter(messaging_settings, redis_url):
    """Create Redis adapter for testing."""
    adapter = RedisQueueAdapter(messaging_settings, redis_url)
    
    # Mock Redis client to avoid actual connections in unit tests
    with patch('redis.asyncio.from_url') as mock_redis:
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client
        
        # Mock successful ping
        mock_client.ping.return_value = True
        
        # Mock stream operations
        mock_client.xadd.return_value = b"1234567890-0"
        mock_client.xgroup_create.side_effect = Exception("BUSYGROUP Consumer Group name already exists")
        mock_client.xreadgroup.return_value = []
        mock_client.xack.return_value = 1
        mock_client.xlen.return_value = 0
        mock_client.xinfo_stream.return_value = {"length": 0, "groups": []}
        mock_client.xpending.return_value = {"pending": 0}
        mock_client.delete.return_value = 1
        mock_client.hset.return_value = 1
        mock_client.hget.return_value = None
        mock_client.expire.return_value = True
        
        adapter._client = mock_client
        
        yield adapter
        
        # Cleanup
        if adapter._connected:
            await adapter.disconnect()


class TestRedisAdapter:
    """Test Redis message queue adapter."""

    @pytest.mark.asyncio
    async def test_connection(self, redis_adapter):
        """Test Redis connection."""
        await redis_adapter.connect()
        assert redis_adapter._connected is True
        
        await redis_adapter.disconnect()
        assert redis_adapter._connected is False

    @pytest.mark.asyncio
    async def test_send_message(self, redis_adapter):
        """Test sending a message."""
        await redis_adapter.connect()
        
        message = Message(
            queue_name="test_queue",
            payload={"data": "test"},
            priority=MessagePriority.HIGH
        )
        
        success = await redis_adapter.send_message("test_queue", message)
        assert success is True
        
        # Verify xadd was called
        redis_adapter._client.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_empty_queue(self, redis_adapter):
        """Test receiving from an empty queue."""
        await redis_adapter.connect()
        
        # Mock empty response
        redis_adapter._client.xreadgroup.return_value = []
        
        message = await redis_adapter.receive_message("test_queue", timeout=1)
        assert message is None

    @pytest.mark.asyncio
    async def test_receive_message_with_data(self, redis_adapter):
        """Test receiving a message with data."""
        await redis_adapter.connect()
        
        # Mock message data
        test_message = Message(
            queue_name="test_queue",
            payload={"data": "test"}
        )
        message_data = test_message.to_dict()
        
        # Convert to Redis stream format
        redis_fields = {}
        for key, value in message_data.items():
            if isinstance(value, dict):
                redis_fields[key.encode()] = str(value).encode()
            else:
                redis_fields[key.encode()] = str(value).encode()
        
        redis_adapter._client.xreadgroup.return_value = [
            [b"stream:test_queue", [(b"1234567890-0", redis_fields)]]
        ]
        
        message = await redis_adapter.receive_message("test_queue")
        assert message is not None
        assert message.queue_name == "test_queue"

    @pytest.mark.asyncio
    async def test_acknowledge_message(self, redis_adapter):
        """Test acknowledging a message."""
        await redis_adapter.connect()
        
        message = Message(
            queue_name="test_queue",
            payload={"data": "test"}
        )
        message.headers["redis_message_id"] = "1234567890-0"
        message.headers["redis_stream"] = "stream:test_queue"
        
        success = await redis_adapter.acknowledge_message(message)
        assert success is True
        
        # Verify xack was called
        redis_adapter._client.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_reject_message_with_requeue(self, redis_adapter):
        """Test rejecting a message with requeue."""
        await redis_adapter.connect()
        
        message = Message(
            queue_name="test_queue",
            payload={"data": "test"},
            max_attempts=3,
            attempts=1
        )
        message.headers["redis_message_id"] = "1234567890-0"
        message.headers["redis_stream"] = "stream:test_queue"
        
        # Mock successful message send for requeue
        redis_adapter._client.xadd.return_value = b"1234567891-0"
        
        success = await redis_adapter.reject_message(message, requeue=True)
        assert success is True

    @pytest.mark.asyncio
    async def test_submit_task(self, redis_adapter):
        """Test submitting a task."""
        await redis_adapter.connect()
        
        task = Task(
            task_type=TaskType.ANOMALY_DETECTION,
            name="Test Task",
            function_name="test_function",
            args=[1, 2, 3],
            kwargs={"param": "value"}
        )
        
        # Mock successful operations
        redis_adapter._client.hset.return_value = 1
        redis_adapter._client.expire.return_value = True
        redis_adapter._client.xadd.return_value = b"1234567890-0"
        
        task_id = await redis_adapter.submit_task(task)
        assert task_id == task.id
        
        # Verify task was stored
        redis_adapter._client.hset.assert_called()

    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, redis_adapter):
        """Test getting task status for non-existent task."""
        await redis_adapter.connect()
        
        # Mock task not found
        redis_adapter._client.hget.return_value = None
        
        task = await redis_adapter.get_task_status("non_existent_task")
        assert task is None

    @pytest.mark.asyncio
    async def test_cancel_task(self, redis_adapter):
        """Test canceling a task."""
        await redis_adapter.connect()
        
        # Mock task not found
        redis_adapter._client.hget.return_value = None
        
        success = await redis_adapter.cancel_task("test_task_id")
        assert success is False

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, redis_adapter):
        """Test getting queue statistics."""
        await redis_adapter.connect()
        
        # Mock stream info
        redis_adapter._client.xinfo_stream.return_value = {
            "length": 5,
            "groups": [{"name": "test_workers"}],
            "first-entry": ["1234567890-0"],
            "last-entry": ["1234567895-0"]
        }
        redis_adapter._client.xpending.return_value = {"pending": 2}
        redis_adapter._client.xlen.return_value = 1
        
        stats = await redis_adapter.get_queue_stats("test_queue")
        
        assert stats["queue_name"] == "test_queue"
        assert stats["total_messages"] == 5
        assert stats["pending_messages"] == 2
        assert stats["dead_letter_messages"] == 1

    @pytest.mark.asyncio
    async def test_purge_queue(self, redis_adapter):
        """Test purging a queue."""
        await redis_adapter.connect()
        
        # Mock current length and operations
        redis_adapter._client.xlen.return_value = 10
        redis_adapter._client.delete.return_value = 1
        redis_adapter._client.xadd.return_value = b"1234567890-0"
        redis_adapter._client.xrange.return_value = [(b"1234567890-0", {})]
        redis_adapter._client.xdel.return_value = 1
        
        count = await redis_adapter.purge_queue("test_queue")
        assert count == 10

    @pytest.mark.asyncio
    async def test_create_queue(self, redis_adapter):
        """Test creating a queue."""
        await redis_adapter.connect()
        
        # Mock operations
        redis_adapter._client.xadd.return_value = b"1234567890-0"
        redis_adapter._client.xrange.return_value = [(b"1234567890-0", {})]
        redis_adapter._client.xdel.return_value = 1
        redis_adapter._client.xgroup_create.return_value = True
        
        success = await redis_adapter.create_queue("new_queue")
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_queue(self, redis_adapter):
        """Test deleting a queue."""
        await redis_adapter.connect()
        
        # Mock deletion
        redis_adapter._client.delete.return_value = 2
        
        success = await redis_adapter.delete_queue("test_queue")
        assert success is True

    @pytest.mark.asyncio
    async def test_health_check(self, redis_adapter):
        """Test health check."""
        await redis_adapter.connect()
        
        # Mock successful ping
        redis_adapter._client.ping.return_value = True
        
        healthy = await redis_adapter.health_check()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, redis_adapter):
        """Test health check failure."""
        await redis_adapter.connect()
        
        # Mock ping failure
        redis_adapter._client.ping.side_effect = Exception("Connection failed")
        
        healthy = await redis_adapter.health_check()
        assert healthy is False

    @pytest.mark.asyncio
    async def test_connection_failure(self, messaging_settings, redis_url):
        """Test connection failure handling."""
        adapter = RedisQueueAdapter(messaging_settings, redis_url)
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            
            # Mock connection failure
            mock_client.ping.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception):
                await adapter.connect()
            
            assert adapter._connected is False

    @pytest.mark.asyncio
    async def test_receive_messages_batch(self, redis_adapter):
        """Test receiving messages in batch."""
        await redis_adapter.connect()
        
        # Mock empty response to stop iteration
        redis_adapter._client.xreadgroup.return_value = []
        
        messages = []
        async for message in redis_adapter.receive_messages("test_queue", batch_size=5):
            messages.append(message)
            break  # Stop after first iteration
        
        # Should be empty since we mocked empty response
        assert len(messages) == 0