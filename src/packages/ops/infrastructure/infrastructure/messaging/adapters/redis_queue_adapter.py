"""Redis-based message queue adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from ..config.messaging_settings import MessagingSettings
from ..models.simple_messages import Message, MessageStatus, Task, TaskStatus
from ..protocols.message_queue_protocol import MessageQueueProtocol

# Optional Redis import
try:
    import redis.asyncio as redis
    from redis.exceptions import ConnectionError, RedisError
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    ConnectionError = Exception
    RedisError = Exception
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedisQueueAdapter(MessageQueueProtocol):
    """Redis-based message queue implementation using Redis Streams and Lists."""

    def __init__(self, settings: MessagingSettings, redis_url: str):
        """Initialize Redis queue adapter.

        Args:
            settings: Messaging settings
            redis_url: Redis connection URL
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install redis")
            
        self.settings = settings
        self.redis_url = redis_url
        self._client: redis.Redis | None = None
        self._connected = False
        
        # Queue prefixes
        self._message_prefix = "queue:messages"
        self._task_prefix = "queue:tasks"
        self._stream_prefix = "stream"
        self._dlq_prefix = "dlq"  # Dead letter queue
        
    async def connect(self) -> None:
        """Establish connection to Redis."""
        try:
            self._client = redis.from_url(
                self.redis_url,
                db=self.settings.redis_queue_db,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
                decode_responses=False,  # We handle encoding ourselves
            )
            
            # Test connection
            await self._client.ping()
            self._connected = True
            
            # Initialize consumer group for streams
            await self._ensure_consumer_group()
            
            logger.info("Successfully connected to Redis message queue")
            
        except (ConnectionError, RedisError) as e:
            logger.error(f"Failed to connect to Redis message queue: {e}")
            self._connected = False
            self._client = None
            raise

    async def disconnect(self) -> None:
        """Close connection to Redis."""
        if self._client:
            await self._client.close()
            self._client = None
            self._connected = False
            logger.info("Disconnected from Redis message queue")

    async def _ensure_consumer_group(self) -> None:
        """Ensure consumer group exists for all streams."""
        if not self._client:
            return
            
        try:
            # Create consumer group for the default stream
            stream_key = f"{self._stream_prefix}:default"
            await self._client.xgroup_create(
                stream_key,
                self.settings.redis_consumer_group,
                id="0",
                mkstream=True
            )
        except Exception as e:
            # Group already exists or other error
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Failed to create consumer group: {e}")

    async def send_message(self, queue_name: str, message: Message) -> bool:
        """Send a message to a queue using Redis Stream."""
        if not self._client or not self._connected:
            raise ConnectionError("Not connected to Redis")

        try:
            stream_key = f"{self._stream_prefix}:{queue_name}"
            message_data = {
                "id": message.id,
                "payload": json.dumps(message.payload),
                "headers": json.dumps(message.headers),
                "timestamp": str(message.timestamp),
                "priority": str(message.priority),
                "retries": str(message.retries),
                "status": message.status.value
            }
            
            # Add to stream with auto-generated ID
            message_id = await self._client.xadd(
                stream_key,
                message_data,
                maxlen=self.settings.redis_stream_maxlen
            )
            
            # Also add to a simple list for backup
            list_key = f"{self._message_prefix}:{queue_name}"
            await self._client.lpush(list_key, json.dumps(message_data))
            
            logger.debug(f"Sent message {message.id} to queue {queue_name}")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to send message to queue {queue_name}: {e}")
            return False

    async def receive_message(self, queue_name: str, timeout: int | None = None) -> Message | None:
        """Receive a message from a queue."""
        if not self._client or not self._connected:
            raise ConnectionError("Not connected to Redis")

        try:
            stream_key = f"{self._stream_prefix}:{queue_name}"
            timeout_ms = (timeout * 1000) if timeout else self.settings.redis_block_timeout
            
            # Read from stream using consumer group
            result = await self._client.xreadgroup(
                self.settings.redis_consumer_group,
                self.settings.redis_consumer_name,
                {stream_key: ">"},
                count=1,
                block=timeout_ms
            )
            
            if result:
                stream_data = result[0][1]  # [(stream_name, [(id, fields)])]
                if stream_data:
                    message_id, fields = stream_data[0]
                    return self._decode_message(fields, message_id)
            
            return None
            
        except RedisError as e:
            logger.error(f"Failed to receive message from queue {queue_name}: {e}")
            return None

    async def receive_messages(self, queue_name: str, batch_size: int = 10) -> AsyncIterator[Message]:
        """Receive messages from a queue as an async iterator."""
        if not self._client or not self._connected:
            raise ConnectionError("Not connected to Redis")

        stream_key = f"{self._stream_prefix}:{queue_name}"
        
        try:
            while True:
                result = await self._client.xreadgroup(
                    self.settings.redis_consumer_group,
                    self.settings.redis_consumer_name,
                    {stream_key: ">"},
                    count=batch_size,
                    block=self.settings.redis_block_timeout
                )
                
                if result:
                    stream_data = result[0][1]
                    for message_id, fields in stream_data:
                        message = self._decode_message(fields, message_id)
                        if message:
                            yield message
                else:
                    # No messages, yield control
                    await asyncio.sleep(0.1)
                    
        except RedisError as e:
            logger.error(f"Failed to receive messages from queue {queue_name}: {e}")

    def _decode_message(self, fields: dict, message_id: str) -> Message | None:
        """Decode message from Redis fields."""
        try:
            return Message(
                id=fields.get(b"id", b"").decode("utf-8"),
                payload=json.loads(fields.get(b"payload", b"{}").decode("utf-8")),
                headers=json.loads(fields.get(b"headers", b"{}").decode("utf-8")),
                timestamp=float(fields.get(b"timestamp", b"0").decode("utf-8")),
                priority=int(fields.get(b"priority", b"5").decode("utf-8")),
                retries=int(fields.get(b"retries", b"0").decode("utf-8")),
                status=MessageStatus(fields.get(b"status", b"pending").decode("utf-8")),
                queue_metadata={"redis_message_id": message_id}
            )
        except Exception as e:
            logger.error(f"Failed to decode message: {e}")
            return None

    async def acknowledge_message(self, message: Message) -> bool:
        """Acknowledge that a message has been processed."""
        if not self._client or not self._connected:
            return False

        try:
            redis_message_id = message.queue_metadata.get("redis_message_id")
            if not redis_message_id:
                return False
                
            # Remove from pending list (acknowledge)
            stream_key = f"{self._stream_prefix}:{message.queue_metadata.get('queue_name', 'default')}"
            await self._client.xack(
                stream_key,
                self.settings.redis_consumer_group,
                redis_message_id
            )
            
            return True
            
        except RedisError as e:
            logger.error(f"Failed to acknowledge message {message.id}: {e}")
            return False

    async def reject_message(self, message: Message, requeue: bool = True) -> bool:
        """Reject a message (send to dead letter queue or requeue)."""
        if not self._client or not self._connected:
            return False

        try:
            if requeue and message.retries < self.settings.max_retries:
                # Increment retries and requeue
                message.retries += 1
                message.status = MessageStatus.PENDING
                queue_name = message.queue_metadata.get("queue_name", "default")
                return await self.send_message(queue_name, message)
            else:
                # Send to dead letter queue
                dlq_key = f"{self._dlq_prefix}:messages"
                message_data = {
                    "id": message.id,
                    "payload": json.dumps(message.payload),
                    "headers": json.dumps(message.headers),
                    "timestamp": str(message.timestamp),
                    "failed_at": str(time.time()),
                    "reason": "max_retries_exceeded"
                }
                await self._client.lpush(dlq_key, json.dumps(message_data))
                
                # Acknowledge the original message
                return await self.acknowledge_message(message)
                
        except RedisError as e:
            logger.error(f"Failed to reject message {message.id}: {e}")
            return False

    # Simplified implementations for the remaining abstract methods
    async def submit_task(self, task: Task) -> str:
        """Submit a task for processing."""
        # Convert task to message and send
        message = Message(
            id=task.id,
            payload={"task_type": task.task_type, "data": task.data},
            headers={"task": "true"}
        )
        queue_name = f"tasks:{task.task_type}"
        success = await self.send_message(queue_name, message)
        return task.id if success else ""

    async def get_task_status(self, task_id: str) -> Task | None:
        """Get the current status of a task."""
        # This would require additional tracking in Redis
        # For now, return None (not implemented)
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        # This would require additional tracking in Redis
        # For now, return False (not implemented)
        return False

    async def get_queue_stats(self, queue_name: str) -> dict[str, Any]:
        """Get statistics for a queue."""
        if not self._client:
            return {}

        try:
            stream_key = f"{self._stream_prefix}:{queue_name}"
            list_key = f"{self._message_prefix}:{queue_name}"
            
            stream_info = await self._client.xinfo_stream(stream_key)
            list_length = await self._client.llen(list_key)
            
            return {
                "stream_length": stream_info.get("length", 0),
                "list_length": list_length,
                "last_generated_id": stream_info.get("last-generated-id", "0-0"),
                "consumer_groups": stream_info.get("groups", 0)
            }
        except RedisError:
            return {"error": "Failed to get queue stats"}

    async def purge_queue(self, queue_name: str) -> int:
        """Remove all messages from a queue."""
        if not self._client:
            return 0

        try:
            stream_key = f"{self._stream_prefix}:{queue_name}"
            list_key = f"{self._message_prefix}:{queue_name}"
            
            # Delete stream and list
            stream_deleted = await self._client.delete(stream_key)
            list_deleted = await self._client.delete(list_key)
            
            return stream_deleted + list_deleted
        except RedisError:
            return 0

    async def create_queue(self, queue_name: str, **options) -> bool:
        """Create a new queue."""
        # In Redis, queues are created automatically when first used
        return True

    async def delete_queue(self, queue_name: str) -> bool:
        """Delete a queue."""
        return await self.purge_queue(queue_name) > 0

    async def health_check(self) -> bool:
        """Check if the message queue connection is healthy."""
        if not self._client or not self._connected:
            return False

        try:
            await self._client.ping()
            return True
        except RedisError:
            return False