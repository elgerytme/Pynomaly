"""Redis-based message queue adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

import redis.asyncio as redis
from redis.exceptions import ConnectionError, RedisError

from ..config.messaging_settings import MessagingSettings
from ..models.messages import Message, MessageStatus
from ..models.tasks import Task, TaskStatus
from ..protocols.message_queue_protocol import MessageQueueProtocol

logger = logging.getLogger(__name__)


class RedisQueueAdapter(MessageQueueProtocol):
    """Redis-based message queue implementation using Redis Streams and Lists."""

    def __init__(self, settings: MessagingSettings, redis_url: str):
        """Initialize Redis queue adapter.

        Args:
            settings: Messaging settings
            redis_url: Redis connection URL
        """
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
        except redis.ResponseError as e:
            # Group already exists
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Failed to create consumer group: {e}")

    async def send_message(self, queue_name: str, message: Message) -> bool:
        """Send a message to a queue using Redis Stream.
        
        Args:
            queue_name: Name of the queue
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        if not self._connected or not self._client:
            logger.error("Not connected to Redis")
            return False

        try:
            stream_key = f"{self._stream_prefix}:{queue_name}"
            message_data = message.to_dict()
            
            # Add message to stream
            message_id = await self._client.xadd(
                stream_key,
                message_data,
                maxlen=self.settings.redis_stream_maxlen,
                approximate=True
            )
            
            logger.debug(f"Message {message.id} sent to {queue_name}: {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message {message.id} to {queue_name}: {e}")
            return False

    async def receive_message(self, queue_name: str, timeout: int | None = None) -> Message | None:
        """Receive a message from a queue using Redis Stream.
        
        Args:
            queue_name: Name of the queue
            timeout: Timeout in seconds (None for blocking)
            
        Returns:
            Message if available, None if timeout
        """
        if not self._connected or not self._client:
            logger.error("Not connected to Redis")
            return None

        try:
            stream_key = f"{self._stream_prefix}:{queue_name}"
            block_ms = (timeout * 1000) if timeout else self.settings.redis_block_timeout
            
            # Read from stream with consumer group
            messages = await self._client.xreadgroup(
                self.settings.redis_consumer_group,
                self.settings.redis_consumer_name,
                {stream_key: ">"},
                count=1,
                block=block_ms
            )
            
            if not messages:
                return None
                
            # Extract message data
            stream_messages = messages[0][1]  # [stream_name, [(id, fields), ...]]
            if not stream_messages:
                return None
                
            message_id, fields = stream_messages[0]
            
            # Decode message data
            message_data = {k.decode(): v.decode() for k, v in fields.items()}
            
            # Parse JSON fields
            for key in ["payload", "headers"]:
                if key in message_data:
                    message_data[key] = json.loads(message_data[key])
            
            # Create message object
            message = Message.from_dict(message_data)
            message.mark_processing()
            
            # Store Redis message ID for acknowledgment
            message.headers["redis_message_id"] = message_id.decode()
            message.headers["redis_stream"] = stream_key
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to receive message from {queue_name}: {e}")
            return None

    async def receive_messages(self, queue_name: str, batch_size: int = 10) -> AsyncIterator[Message]:
        """Receive messages from a queue as an async iterator.
        
        Args:
            queue_name: Name of the queue
            batch_size: Number of messages to fetch at once
            
        Yields:
            Messages from the queue
        """
        if not self._connected or not self._client:
            logger.error("Not connected to Redis")
            return

        stream_key = f"{self._stream_prefix}:{queue_name}"
        
        while self._connected:
            try:
                # Read batch of messages
                messages = await self._client.xreadgroup(
                    self.settings.redis_consumer_group,
                    self.settings.redis_consumer_name,
                    {stream_key: ">"},
                    count=batch_size,
                    block=self.settings.redis_block_timeout
                )
                
                if not messages:
                    await asyncio.sleep(0.1)  # Brief pause before next poll
                    continue
                    
                # Process messages
                stream_messages = messages[0][1]
                for message_id, fields in stream_messages:
                    try:
                        # Decode message data
                        message_data = {k.decode(): v.decode() for k, v in fields.items()}
                        
                        # Parse JSON fields
                        for key in ["payload", "headers"]:
                            if key in message_data:
                                message_data[key] = json.loads(message_data[key])
                        
                        # Create message object
                        message = Message.from_dict(message_data)
                        message.mark_processing()
                        
                        # Store Redis message ID for acknowledgment
                        message.headers["redis_message_id"] = message_id.decode()
                        message.headers["redis_stream"] = stream_key
                        
                        yield message
                        
                    except Exception as e:
                        logger.error(f"Failed to process message {message_id}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to receive messages from {queue_name}: {e}")
                await asyncio.sleep(1)  # Wait before retrying

    async def acknowledge_message(self, message: Message) -> bool:
        """Acknowledge that a message has been processed.
        
        Args:
            message: Message to acknowledge
            
        Returns:
            True if acknowledgment was successful
        """
        if not self._connected or not self._client:
            return False

        try:
            redis_message_id = message.headers.get("redis_message_id")
            redis_stream = message.headers.get("redis_stream")
            
            if not redis_message_id or not redis_stream:
                logger.error(f"Missing Redis metadata for message {message.id}")
                return False
            
            # Acknowledge message in consumer group
            await self._client.xack(
                redis_stream,
                self.settings.redis_consumer_group,
                redis_message_id
            )
            
            message.mark_completed()
            logger.debug(f"Message {message.id} acknowledged")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge message {message.id}: {e}")
            return False

    async def reject_message(self, message: Message, requeue: bool = True) -> bool:
        """Reject a message (send to dead letter queue or requeue).
        
        Args:
            message: Message to reject
            requeue: Whether to requeue the message
            
        Returns:
            True if rejection was successful
        """
        if not self._connected or not self._client:
            return False

        try:
            redis_message_id = message.headers.get("redis_message_id")
            redis_stream = message.headers.get("redis_stream")
            
            if not redis_message_id or not redis_stream:
                logger.error(f"Missing Redis metadata for message {message.id}")
                return False
            
            if requeue and message.can_retry():
                # Mark for retry and requeue
                message.mark_retrying()
                return await self.send_message(message.queue_name, message)
            else:
                # Send to dead letter queue
                if self.settings.dead_letter_queue_enabled:
                    dlq_key = f"{self._dlq_prefix}:{message.queue_name}"
                    await self._client.xadd(
                        dlq_key,
                        message.to_dict(),
                        maxlen=1000,  # Limit DLQ size
                        approximate=True
                    )
                
                # Acknowledge original message
                await self._client.xack(
                    redis_stream,
                    self.settings.redis_consumer_group,
                    redis_message_id
                )
                
                message.mark_failed("Message rejected and sent to dead letter queue")
                
            logger.debug(f"Message {message.id} rejected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reject message {message.id}: {e}")
            return False

    async def submit_task(self, task: Task) -> str:
        """Submit a task for processing.
        
        Args:
            task: Task to submit
            
        Returns:
            Task ID
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Redis")

        try:
            # Store task metadata
            task_key = f"{self._task_prefix}:{task.id}"
            task.mark_queued()
            
            await self._client.hset(
                task_key,
                mapping={
                    "data": json.dumps(task.to_dict()),
                    "created_at": str(time.time()),
                    "status": task.status.value
                }
            )
            
            # Set expiration
            if self.settings.queue_default_ttl > 0:
                await self._client.expire(task_key, self.settings.queue_default_ttl)
            
            # Create message for task processing
            message = Message(
                queue_name=task.queue_name,
                payload={
                    "task_id": task.id,
                    "task_type": task.task_type.value,
                    "function_name": task.function_name,
                    "args": task.args,
                    "kwargs": task.kwargs
                },
                message_type="task",
                headers={"task_id": task.id}
            )
            
            # Send to queue
            success = await self.send_message(task.queue_name, message)
            if not success:
                raise RuntimeError(f"Failed to queue task {task.id}")
            
            logger.info(f"Task {task.id} submitted to queue {task.queue_name}")
            return task.id
            
        except Exception as e:
            logger.error(f"Failed to submit task {task.id}: {e}")
            raise

    async def get_task_status(self, task_id: str) -> Task | None:
        """Get the current status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task with current status, None if not found
        """
        if not self._connected or not self._client:
            return None

        try:
            task_key = f"{self._task_prefix}:{task_id}"
            task_data = await self._client.hget(task_key, "data")
            
            if not task_data:
                return None
                
            # Decode and parse task data
            task_dict = json.loads(task_data.decode())
            return Task.from_dict(task_dict)
            
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled successfully
        """
        if not self._connected or not self._client:
            return False

        try:
            task = await self.get_task_status(task_id)
            if not task:
                return False
                
            # Only cancel if not yet running
            if task.status in {TaskStatus.PENDING, TaskStatus.QUEUED}:
                task.mark_cancelled()
                
                # Update task data
                task_key = f"{self._task_prefix}:{task_id}"
                await self._client.hset(
                    task_key,
                    mapping={
                        "data": json.dumps(task.to_dict()),
                        "status": task.status.value
                    }
                )
                
                logger.info(f"Task {task_id} cancelled")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    async def get_queue_stats(self, queue_name: str) -> dict[str, Any]:
        """Get statistics for a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Dictionary with queue statistics
        """
        if not self._connected or not self._client:
            return {}

        try:
            stream_key = f"{self._stream_prefix}:{queue_name}"
            dlq_key = f"{self._dlq_prefix}:{queue_name}"
            
            # Get stream info
            try:
                stream_info = await self._client.xinfo_stream(stream_key)
                pending_info = await self._client.xpending(
                    stream_key, 
                    self.settings.redis_consumer_group
                )
                dlq_length = await self._client.xlen(dlq_key)
            except redis.ResponseError:
                # Stream doesn't exist yet
                return {
                    "queue_name": queue_name,
                    "total_messages": 0,
                    "pending_messages": 0,
                    "dead_letter_messages": 0,
                    "consumers": 0
                }
            
            return {
                "queue_name": queue_name,
                "total_messages": stream_info.get("length", 0),
                "pending_messages": pending_info.get("pending", 0),
                "dead_letter_messages": dlq_length,
                "consumers": len(stream_info.get("groups", [])),
                "first_entry_id": stream_info.get("first-entry", ["0-0"])[0],
                "last_entry_id": stream_info.get("last-entry", ["0-0"])[0]
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats for {queue_name}: {e}")
            return {}

    async def purge_queue(self, queue_name: str) -> int:
        """Remove all messages from a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Number of messages removed
        """
        if not self._connected or not self._client:
            return 0

        try:
            stream_key = f"{self._stream_prefix}:{queue_name}"
            
            # Get current length
            current_length = await self._client.xlen(stream_key)
            
            # Delete the stream (removes all messages)
            await self._client.delete(stream_key)
            
            # Recreate empty stream and consumer group
            await self._client.xadd(stream_key, {"_init": "1"})
            await self._client.xdel(stream_key, await self._client.xrange(stream_key, count=1))
            await self._ensure_consumer_group()
            
            logger.info(f"Purged {current_length} messages from queue {queue_name}")
            return current_length
            
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {e}")
            return 0

    async def create_queue(self, queue_name: str, **options) -> bool:
        """Create a new queue.
        
        Args:
            queue_name: Name of the queue
            **options: Queue-specific options
            
        Returns:
            True if queue was created successfully
        """
        if not self._connected or not self._client:
            return False

        try:
            stream_key = f"{self._stream_prefix}:{queue_name}"
            
            # Create stream with a placeholder message
            await self._client.xadd(stream_key, {"_init": "1"})
            
            # Remove placeholder message
            message_ids = await self._client.xrange(stream_key, count=1)
            if message_ids:
                await self._client.xdel(stream_key, message_ids[0][0])
            
            # Create consumer group
            await self._client.xgroup_create(
                stream_key,
                self.settings.redis_consumer_group,
                id="0",
                mkstream=True
            )
            
            logger.info(f"Created queue {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create queue {queue_name}: {e}")
            return False

    async def delete_queue(self, queue_name: str) -> bool:
        """Delete a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            True if queue was deleted successfully
        """
        if not self._connected or not self._client:
            return False

        try:
            stream_key = f"{self._stream_prefix}:{queue_name}"
            dlq_key = f"{self._dlq_prefix}:{queue_name}"
            
            # Delete stream and dead letter queue
            deleted = await self._client.delete(stream_key, dlq_key)
            
            logger.info(f"Deleted queue {queue_name} ({deleted} keys removed)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete queue {queue_name}: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if the message queue connection is healthy.
        
        Returns:
            True if connection is healthy
        """
        if not self._connected or not self._client:
            return False

        try:
            await self._client.ping()
            return True
        except Exception:
            return False