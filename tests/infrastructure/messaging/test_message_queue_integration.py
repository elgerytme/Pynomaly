"""
Tests for Message Queue Integration

This module contains comprehensive tests for the message queue integration system.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock

from src.pynomaly.infrastructure.messaging.config import (
    MessageBroker,
    MessageQueueConfig,
    QueueConfig,
    MessageConfig,
    MessageFormat,
    DeliveryMode,
    AcknowledgmentMode,
)
from src.pynomaly.infrastructure.messaging.core import (
    Message,
    MessageHandler,
    AsyncMessageHandler,
    MessageQueue,
    MessageQueueManager,
    init_message_queue_manager,
)
from src.pynomaly.infrastructure.messaging.factory import MessageQueueFactory
from src.pynomaly.infrastructure.messaging.implementations.memory_queue import MemoryMessageQueue


class TestMessage:
    """Test Message class."""
    
    def test_message_creation(self):
        """Test message creation with default values."""
        message = Message(body="test message")
        
        assert message.body == "test message"
        assert message.content_type == "application/json"
        assert message.delivery_count == 0
        assert message.priority == 0
        assert message.id is not None
        assert message.timestamp is not None
    
    def test_message_to_dict(self):
        """Test message serialization to dictionary."""
        message = Message(
            body={"key": "value"},
            queue_name="test_queue",
            routing_key="test.key",
            priority=5
        )
        
        data = message.to_dict()
        
        assert data["body"] == {"key": "value"}
        assert data["queue_name"] == "test_queue"
        assert data["routing_key"] == "test.key"
        assert data["priority"] == 5
        assert data["delivery_count"] == 0
    
    def test_message_from_dict(self):
        """Test message deserialization from dictionary."""
        data = {
            "id": "test-id",
            "body": {"key": "value"},
            "queue_name": "test_queue",
            "priority": 5,
            "delivery_count": 2
        }
        
        message = Message.from_dict(data)
        
        assert message.id == "test-id"
        assert message.body == {"key": "value"}
        assert message.queue_name == "test_queue"
        assert message.priority == 5
        assert message.delivery_count == 2
    
    def test_message_serialization(self):
        """Test message serialization and deserialization."""
        original = Message(
            body="test message",
            queue_name="test_queue",
            priority=3
        )
        
        serialized = original.serialize()
        deserialized = Message.deserialize(serialized)
        
        assert deserialized.body == original.body
        assert deserialized.queue_name == original.queue_name
        assert deserialized.priority == original.priority


class TestMessageHandler:
    """Test MessageHandler classes."""
    
    @pytest.mark.asyncio
    async def test_async_message_handler(self):
        """Test AsyncMessageHandler with async function."""
        
        async def handler_func(message: Message) -> str:
            return f"Processed: {message.body}"
        
        handler = AsyncMessageHandler(handler_func)
        message = Message(body="test")
        
        result = await handler.handle(message)
        assert result == "Processed: test"
    
    @pytest.mark.asyncio
    async def test_async_message_handler_with_sync_function(self):
        """Test AsyncMessageHandler with sync function."""
        
        def handler_func(message: Message) -> str:
            return f"Processed: {message.body}"
        
        handler = AsyncMessageHandler(handler_func)
        message = Message(body="test")
        
        result = await handler.handle(message)
        assert result == "Processed: test"
    
    @pytest.mark.asyncio
    async def test_async_message_handler_error_handling(self):
        """Test AsyncMessageHandler error handling."""
        
        async def failing_handler(message: Message) -> str:
            raise ValueError("Test error")
        
        handler = AsyncMessageHandler(failing_handler)
        message = Message(body="test")
        
        with pytest.raises(ValueError):
            await handler.handle(message)
        
        # Check that delivery count was incremented
        assert message.delivery_count == 1


class TestMemoryMessageQueue:
    """Test MemoryMessageQueue implementation."""
    
    def setup_method(self):
        """Set up test fixture."""
        self.config = MessageQueueConfig(broker=MessageBroker.MEMORY)
        self.queue = MemoryMessageQueue(self.config)
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test queue connection and disconnection."""
        assert not self.queue._is_connected
        
        await self.queue.connect()
        assert self.queue._is_connected
        
        await self.queue.disconnect()
        assert not self.queue._is_connected
    
    @pytest.mark.asyncio
    async def test_send_receive(self):
        """Test sending and receiving messages."""
        await self.queue.connect()
        
        message = Message(body="test message", queue_name="test_queue")
        message_id = await self.queue.send(message)
        
        assert message_id == message.id
        assert self.queue.state.messages_sent == 1
        
        received = await self.queue.receive("test_queue")
        assert received is not None
        assert received.body == "test message"
    
    @pytest.mark.asyncio
    async def test_receive_empty_queue(self):
        """Test receiving from empty queue."""
        await self.queue.connect()
        
        received = await self.queue.receive("nonexistent_queue")
        assert received is None
    
    @pytest.mark.asyncio
    async def test_queue_management(self):
        """Test queue creation, deletion, and purging."""
        await self.queue.connect()
        
        # Create queue
        queue_config = QueueConfig(name="test_queue")
        await self.queue.create_queue(queue_config)
        
        # Send messages
        for i in range(5):
            message = Message(body=f"message {i}", queue_name="test_queue")
            await self.queue.send(message)
        
        # Check queue size
        size = await self.queue.get_queue_size("test_queue")
        assert size == 5
        
        # Purge queue
        purged = await self.queue.purge_queue("test_queue")
        assert purged == 5
        
        size = await self.queue.get_queue_size("test_queue")
        assert size == 0
        
        # Delete queue
        await self.queue.delete_queue("test_queue")
        assert "test_queue" not in self.queue.state.active_queues
    
    @pytest.mark.asyncio
    async def test_message_acknowledgment(self):
        """Test message acknowledgment and rejection."""
        await self.queue.connect()
        
        message = Message(body="test message", queue_name="test_queue")
        await self.queue.send(message)
        
        received = await self.queue.receive("test_queue")
        assert received is not None
        
        # Acknowledge message (no-op for memory queue)
        await self.queue.acknowledge(received)
        
        # Reject message with requeue
        await self.queue.reject(received, requeue=True)
        
        # Should be able to receive it again
        requeued = await self.queue.receive("test_queue")
        assert requeued is not None
        assert requeued.id == received.id
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check."""
        await self.queue.connect()
        
        is_healthy = await self.queue.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_json_and_text_helpers(self):
        """Test JSON and text message helpers."""
        await self.queue.connect()
        
        # Test JSON helper
        data = {"key": "value", "number": 42}
        message_id = await self.queue.send_json(data, queue_name="test_queue")
        
        received = await self.queue.receive("test_queue")
        assert received is not None
        assert received.body == data
        assert received.content_type == "application/json"
        
        # Test text helper
        text = "Hello, world!"
        message_id = await self.queue.send_text(text, queue_name="test_queue")
        
        received = await self.queue.receive("test_queue")
        assert received is not None
        assert received.body == text
        assert received.content_type == "text/plain"


class TestMessageQueueManager:
    """Test MessageQueueManager."""
    
    def setup_method(self):
        """Set up test fixture."""
        self.config = MessageQueueConfig(broker=MessageBroker.MEMORY)
        self.manager = MessageQueueManager(self.config)
    
    @pytest.mark.asyncio
    async def test_add_and_get_queue(self):
        """Test adding and getting queues."""
        queue = MemoryMessageQueue(self.config)
        
        self.manager.add_queue("test_queue", queue)
        retrieved = self.manager.get_queue("test_queue")
        
        assert retrieved is queue
        assert self.manager.get_queue("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_start_stop_manager(self):
        """Test starting and stopping the manager."""
        queue = MemoryMessageQueue(self.config)
        self.manager.add_queue("test_queue", queue)
        
        assert not self.manager._is_running
        
        await self.manager.start()
        assert self.manager._is_running
        
        await self.manager.stop()
        assert not self.manager._is_running
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test manager health check."""
        queue = MemoryMessageQueue(self.config)
        self.manager.add_queue("test_queue", queue)
        
        await self.manager.start()
        
        health = await self.manager.health_check()
        assert health["is_running"] is True
        assert health["overall_healthy"] is True
        assert "test_queue" in health["queues"]
        
        await self.manager.stop()
    
    @pytest.mark.asyncio
    async def test_send_to_queue(self):
        """Test sending messages through manager."""
        queue = MemoryMessageQueue(self.config)
        self.manager.add_queue("test_queue", queue)
        
        await self.manager.start()
        
        message = Message(body="test message")
        message_id = await self.manager.send_to_queue("test_queue", message)
        
        assert message_id == message.id
        
        # Verify message was sent
        received = await queue.receive("test_queue")
        assert received is not None
        assert received.body == "test message"
        
        await self.manager.stop()
    
    def test_global_stats(self):
        """Test global statistics."""
        queue = MemoryMessageQueue(self.config)
        self.manager.add_queue("test_queue", queue)
        
        stats = self.manager.get_global_stats()
        
        assert stats["is_running"] is False
        assert stats["queue_count"] == 1
        assert stats["consumer_count"] == 0
        assert "test_queue" in stats["queues"]


class TestMessageQueueFactory:
    """Test MessageQueueFactory."""
    
    def test_create_memory_queue(self):
        """Test creating memory queue."""
        config = MessageQueueConfig(broker=MessageBroker.MEMORY)
        queue = MessageQueueFactory.create_queue(config)
        
        assert isinstance(queue, MemoryMessageQueue)
    
    def test_create_queue_from_env(self):
        """Test creating queue from environment."""
        with patch.dict('os.environ', {'MESSAGE_BROKER': 'memory'}):
            queue = MessageQueueFactory.create_queue_from_env()
            assert isinstance(queue, MemoryMessageQueue)
    
    def test_get_available_brokers(self):
        """Test getting available brokers."""
        brokers = MessageQueueFactory.get_available_brokers()
        assert MessageBroker.MEMORY in brokers
    
    def test_validate_broker_availability(self):
        """Test broker availability validation."""
        assert MessageQueueFactory.validate_broker_availability(MessageBroker.MEMORY) is True
    
    def test_create_with_fallback(self):
        """Test creating queue with fallback."""
        # Try to create Redis queue, should fall back to memory
        queue = MessageQueueFactory.create_with_fallback(
            MessageBroker.REDIS,
            MessageBroker.MEMORY
        )
        assert isinstance(queue, MemoryMessageQueue)
    
    def test_create_for_testing(self):
        """Test creating queue for testing."""
        queue = MessageQueueFactory.create_for_testing()
        assert isinstance(queue, MemoryMessageQueue)
    
    def test_unsupported_broker(self):
        """Test creating unsupported broker."""
        config = MessageQueueConfig(broker="unsupported")
        
        with pytest.raises(ValueError, match="Unsupported message broker"):
            MessageQueueFactory.create_queue(config)


class TestConfig:
    """Test configuration classes."""
    
    def test_message_queue_config_defaults(self):
        """Test default configuration values."""
        config = MessageQueueConfig()
        
        assert config.broker == MessageBroker.MEMORY
        assert config.enable_metrics is True
        assert config.enable_tracing is False
        assert config.worker_pool_size == 4
        assert config.max_concurrent_messages == 100
    
    def test_message_queue_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict('os.environ', {
            'MESSAGE_BROKER': 'redis',
            'ENABLE_METRICS': 'false',
            'WORKER_POOL_SIZE': '8'
        }):
            config = MessageQueueConfig.from_env()
            
            assert config.broker == MessageBroker.REDIS
            assert config.enable_metrics is False
            assert config.worker_pool_size == 8
    
    def test_connection_config_from_env(self):
        """Test connection configuration from environment."""
        from src.pynomaly.infrastructure.messaging.config import ConnectionConfig
        
        with patch.dict('os.environ', {
            'RABBITMQ_HOST': 'rabbitmq.example.com',
            'RABBITMQ_PORT': '5673',
            'RABBITMQ_USERNAME': 'user',
            'RABBITMQ_PASSWORD': 'pass'
        }):
            config = ConnectionConfig.from_env(MessageBroker.RABBITMQ)
            
            assert config.host == 'rabbitmq.example.com'
            assert config.port == 5673
            assert config.username == 'user'
            assert config.password == 'pass'
    
    def test_queue_config_defaults(self):
        """Test queue configuration defaults."""
        config = QueueConfig(name="test_queue")
        
        assert config.name == "test_queue"
        assert config.durable is True
        assert config.auto_delete is False
        assert config.exclusive is False
        assert config.partitions == 1
        assert config.replication_factor == 1


class TestIntegration:
    """Integration tests for the message queue system."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full message queue workflow."""
        # Initialize manager
        config = MessageQueueConfig(broker=MessageBroker.MEMORY)
        manager = init_message_queue_manager(config)
        
        # Create and add queue
        queue = MessageQueueFactory.create_queue(config)
        manager.add_queue("workflow_queue", queue)
        
        # Register a handler
        processed_messages = []
        
        async def message_handler(message: Message) -> str:
            processed_messages.append(message.body)
            return f"Processed: {message.body}"
        
        queue.register_function_handler("test_queue", message_handler)
        
        # Start manager
        await manager.start()
        
        # Send messages
        for i in range(5):
            message = Message(body=f"message {i}", queue_name="test_queue")
            await manager.send_to_queue("workflow_queue", message)
        
        # Start consuming (simulate)
        await queue.connect()
        for i in range(5):
            message = await queue.receive("test_queue")
            if message:
                await queue.process_message(message)
        
        # Verify processing
        assert len(processed_messages) == 5
        assert "message 0" in processed_messages
        assert "message 4" in processed_messages
        
        # Check statistics
        stats = manager.get_global_stats()
        assert stats["total_messages_sent"] == 5
        
        # Stop manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in message processing."""
        config = MessageQueueConfig(broker=MessageBroker.MEMORY)
        queue = MemoryMessageQueue(config)
        
        # Register failing handler
        async def failing_handler(message: Message) -> str:
            raise ValueError("Processing failed")
        
        queue.register_function_handler("error_queue", failing_handler)
        
        await queue.connect()
        
        # Send message
        message = Message(body="test", queue_name="error_queue")
        await queue.send(message)
        
        # Try to process (should fail)
        received = await queue.receive("error_queue")
        assert received is not None
        
        with pytest.raises(ValueError):
            await queue.process_message(received)
        
        # Verify delivery count was incremented
        assert received.delivery_count == 1


if __name__ == "__main__":
    pytest.main([__file__])
