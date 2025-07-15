"""Redis-based message queue adapter for asynchronous task processing."""

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum

import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError

from packages.infrastructure.infrastructure.messaging.models.tasks import Task, TaskStatus, TaskType
from packages.infrastructure.infrastructure.messaging.models.messages import Message, MessageStatus, MessagePriority


class QueueType(str, Enum):
    """Types of queues supported by the Redis adapter."""
    
    STANDARD = "standard"
    PRIORITY = "priority"
    DELAYED = "delayed"
    DEAD_LETTER = "dead_letter"


@dataclass
class QueueConfig:
    """Configuration for Redis queue."""
    
    name: str
    max_size: int = 10000
    message_ttl: int = 3600  # 1 hour
    max_retries: int = 3
    retry_delay: int = 60    # 1 minute
    enable_dead_letter: bool = True
    compression: bool = True
    priority_levels: int = 10


@dataclass
class WorkerConfig:
    """Configuration for queue workers."""
    
    name: str
    queues: List[str]
    max_concurrent_tasks: int = 5
    poll_interval: float = 1.0
    timeout: int = 300  # 5 minutes
    auto_ack: bool = False
    prefetch_count: int = 1


class RedisQueueAdapter:
    """Redis-based message queue adapter with comprehensive features.
    
    Provides async task queuing, worker management, and reliable message processing
    using Redis as the backend store.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        namespace: str = "pynomaly",
        serializer: str = "json"
    ):
        self.redis_url = redis_url
        self.namespace = namespace
        self.serializer = serializer
        self.logger = logging.getLogger(__name__)
        
        # Redis connection
        self.redis: Optional[redis.Redis] = None
        self.connection_pool = None
        
        # Queue configurations
        self.queue_configs: Dict[str, QueueConfig] = {}
        
        # Worker management
        self.workers: Dict[str, asyncio.Task] = {}
        self.worker_configs: Dict[str, WorkerConfig] = {}
        self.task_handlers: Dict[str, Callable] = {}
        
        # Monitoring
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "workers_active": 0
        }

    async def connect(self) -> None:
        """Initialize Redis connection."""
        try:
            self.connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            self.redis = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=True
            )
            
            # Test connection
            await self.redis.ping()
            self.logger.info("Redis queue adapter connected successfully")
            
        except ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing Redis connection: {e}")
            raise

    async def disconnect(self) -> None:
        """Clean up Redis connection."""
        try:
            # Stop all workers
            await self.stop_all_workers()
            
            # Close Redis connection
            if self.redis:
                await self.redis.close()
                
            if self.connection_pool:
                await self.connection_pool.disconnect()
                
            self.logger.info("Redis queue adapter disconnected")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from Redis: {e}")

    def create_queue(self, config: QueueConfig) -> None:
        """Create a new queue with specified configuration."""
        self.queue_configs[config.name] = config
        self.logger.info(f"Created queue: {config.name}")

    async def send_message(
        self,
        queue_name: str,
        message: Union[Message, Dict[str, Any]],
        priority: MessagePriority = MessagePriority.NORMAL,
        delay: Optional[int] = None
    ) -> str:
        """Send a message to a queue.
        
        Args:
            queue_name: Name of the queue
            message: Message to send
            priority: Message priority level
            delay: Delay in seconds before message becomes available
            
        Returns:
            Message ID
        """
        try:
            if not self.redis:
                raise RuntimeError("Redis connection not initialized")
            
            # Convert message to dict if needed
            if isinstance(message, Message):
                message_dict = asdict(message)
            else:
                message_dict = message
            
            # Generate message ID
            message_id = f"{queue_name}:{int(time.time() * 1000000)}"
            
            # Serialize message
            serialized_message = self._serialize_message(message_dict)
            
            # Add metadata
            envelope = {
                "id": message_id,
                "queue": queue_name,
                "priority": priority.value,
                "created_at": datetime.now().isoformat(),
                "retry_count": 0,
                "payload": serialized_message
            }
            
            # Handle delayed messages
            if delay:
                await self._send_delayed_message(envelope, delay)
            else:
                await self._send_immediate_message(queue_name, envelope, priority)
            
            self.stats["messages_sent"] += 1
            self.logger.debug(f"Message sent to queue {queue_name}: {message_id}")
            
            return message_id
            
        except Exception as e:
            self.logger.error(f"Error sending message to queue {queue_name}: {e}")
            raise

    async def send_task(
        self,
        queue_name: str,
        task: Task,
        priority: MessagePriority = MessagePriority.NORMAL,
        delay: Optional[int] = None
    ) -> str:
        """Send a task to a queue for processing.
        
        Args:
            queue_name: Name of the queue
            task: Task to send
            priority: Task priority level
            delay: Delay in seconds before task becomes available
            
        Returns:
            Task ID
        """
        try:
            # Convert task to message
            message = Message(
                id=task.id,
                type=task.type.value,
                priority=priority,
                payload=asdict(task),
                created_at=datetime.now(),
                status=MessageStatus.PENDING
            )
            
            return await self.send_message(queue_name, message, priority, delay)
            
        except Exception as e:
            self.logger.error(f"Error sending task to queue {queue_name}: {e}")
            raise

    async def receive_message(
        self,
        queue_name: str,
        timeout: int = 30,
        auto_ack: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Receive a message from a queue.
        
        Args:
            queue_name: Name of the queue
            timeout: Timeout in seconds for blocking pop
            auto_ack: Automatically acknowledge message
            
        Returns:
            Message envelope or None if timeout
        """
        try:
            if not self.redis:
                raise RuntimeError("Redis connection not initialized")
            
            # Try priority queue first
            message_data = await self._receive_from_priority_queue(queue_name, timeout)
            
            if not message_data:
                # Fallback to standard queue
                message_data = await self._receive_from_standard_queue(queue_name, timeout)
            
            if message_data:
                envelope = json.loads(message_data)
                
                # Deserialize payload
                envelope["payload"] = self._deserialize_message(envelope["payload"])
                
                # Move to processing set if not auto-ack
                if not auto_ack:
                    await self._move_to_processing(queue_name, envelope)
                
                self.stats["messages_received"] += 1
                return envelope
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error receiving message from queue {queue_name}: {e}")
            return None

    async def acknowledge_message(self, queue_name: str, message_id: str) -> bool:
        """Acknowledge successful message processing.
        
        Args:
            queue_name: Name of the queue
            message_id: ID of the message to acknowledge
            
        Returns:
            True if acknowledged successfully
        """
        try:
            if not self.redis:
                raise RuntimeError("Redis connection not initialized")
            
            processing_key = self._get_processing_key(queue_name)
            
            # Remove from processing set
            result = await self.redis.zrem(processing_key, message_id)
            
            if result:
                self.logger.debug(f"Message acknowledged: {message_id}")
                return True
            else:
                self.logger.warning(f"Message not found in processing set: {message_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error acknowledging message {message_id}: {e}")
            return False

    async def reject_message(
        self,
        queue_name: str,
        message_id: str,
        requeue: bool = True,
        reason: Optional[str] = None
    ) -> bool:
        """Reject a message and optionally requeue it.
        
        Args:
            queue_name: Name of the queue
            message_id: ID of the message to reject
            requeue: Whether to requeue the message
            reason: Reason for rejection
            
        Returns:
            True if rejected successfully
        """
        try:
            if not self.redis:
                raise RuntimeError("Redis connection not initialized")
            
            processing_key = self._get_processing_key(queue_name)
            
            # Get message from processing set
            message_data = await self.redis.zscore(processing_key, message_id)
            
            if message_data is None:
                self.logger.warning(f"Message not found in processing set: {message_id}")
                return False
            
            # Remove from processing set
            await self.redis.zrem(processing_key, message_id)
            
            if requeue:
                # Get full message envelope
                envelope = await self._get_message_envelope(queue_name, message_id)
                
                if envelope:
                    # Increment retry count
                    envelope["retry_count"] += 1
                    config = self.queue_configs.get(queue_name, QueueConfig(name=queue_name))
                    
                    if envelope["retry_count"] < config.max_retries:
                        # Requeue with delay
                        await self._requeue_message(queue_name, envelope, config.retry_delay)
                    else:
                        # Send to dead letter queue
                        await self._send_to_dead_letter(queue_name, envelope, reason)
            
            self.stats["messages_failed"] += 1
            self.logger.debug(f"Message rejected: {message_id}, requeue: {requeue}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rejecting message {message_id}: {e}")
            return False

    async def create_worker(self, config: WorkerConfig) -> str:
        """Create a new worker for processing messages.
        
        Args:
            config: Worker configuration
            
        Returns:
            Worker ID
        """
        try:
            if config.name in self.workers:
                raise ValueError(f"Worker {config.name} already exists")
            
            self.worker_configs[config.name] = config
            
            # Start worker task
            worker_task = asyncio.create_task(self._worker_loop(config))
            self.workers[config.name] = worker_task
            
            self.stats["workers_active"] += 1
            self.logger.info(f"Created worker: {config.name}")
            
            return config.name
            
        except Exception as e:
            self.logger.error(f"Error creating worker {config.name}: {e}")
            raise

    async def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for a specific task type.
        
        Args:
            task_type: Type of task to handle
            handler: Async function to handle the task
        """
        self.task_handlers[task_type] = handler
        self.logger.info(f"Registered handler for task type: {task_type}")

    async def stop_worker(self, worker_name: str) -> bool:
        """Stop a specific worker.
        
        Args:
            worker_name: Name of the worker to stop
            
        Returns:
            True if stopped successfully
        """
        try:
            if worker_name not in self.workers:
                return False
            
            worker_task = self.workers[worker_name]
            worker_task.cancel()
            
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            
            del self.workers[worker_name]
            del self.worker_configs[worker_name]
            
            self.stats["workers_active"] -= 1
            self.logger.info(f"Stopped worker: {worker_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping worker {worker_name}: {e}")
            return False

    async def stop_all_workers(self) -> None:
        """Stop all active workers."""
        worker_names = list(self.workers.keys())
        
        for worker_name in worker_names:
            await self.stop_worker(worker_name)

    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get statistics for a specific queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Dictionary with queue statistics
        """
        try:
            if not self.redis:
                raise RuntimeError("Redis connection not initialized")
            
            stats = {
                "name": queue_name,
                "pending_messages": 0,
                "processing_messages": 0,
                "dead_letter_messages": 0,
                "total_messages": 0
            }
            
            # Count messages in different states
            pending_key = self._get_queue_key(queue_name)
            processing_key = self._get_processing_key(queue_name)
            dead_letter_key = self._get_dead_letter_key(queue_name)
            
            stats["pending_messages"] = await self.redis.llen(pending_key)
            stats["processing_messages"] = await self.redis.zcard(processing_key)
            stats["dead_letter_messages"] = await self.redis.llen(dead_letter_key)
            
            # Priority queue count
            priority_key = self._get_priority_key(queue_name)
            priority_count = await self.redis.zcard(priority_key)
            
            stats["total_messages"] = (
                stats["pending_messages"] +
                stats["processing_messages"] +
                priority_count
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting queue stats for {queue_name}: {e}")
            return {"name": queue_name, "error": str(e)}

    # Private helper methods
    
    def _serialize_message(self, message: Dict[str, Any]) -> str:
        """Serialize message for storage."""
        if self.serializer == "json":
            return json.dumps(message, default=str)
        elif self.serializer == "pickle":
            return pickle.dumps(message).hex()
        else:
            raise ValueError(f"Unsupported serializer: {self.serializer}")

    def _deserialize_message(self, data: str) -> Dict[str, Any]:
        """Deserialize message from storage."""
        if self.serializer == "json":
            return json.loads(data)
        elif self.serializer == "pickle":
            return pickle.loads(bytes.fromhex(data))
        else:
            raise ValueError(f"Unsupported serializer: {self.serializer}")

    def _get_queue_key(self, queue_name: str) -> str:
        """Get Redis key for standard queue."""
        return f"{self.namespace}:queue:{queue_name}"

    def _get_priority_key(self, queue_name: str) -> str:
        """Get Redis key for priority queue."""
        return f"{self.namespace}:priority:{queue_name}"

    def _get_processing_key(self, queue_name: str) -> str:
        """Get Redis key for processing set."""
        return f"{self.namespace}:processing:{queue_name}"

    def _get_dead_letter_key(self, queue_name: str) -> str:
        """Get Redis key for dead letter queue."""
        return f"{self.namespace}:dead_letter:{queue_name}"

    def _get_delayed_key(self, queue_name: str) -> str:
        """Get Redis key for delayed messages."""
        return f"{self.namespace}:delayed:{queue_name}"

    async def _send_immediate_message(self, queue_name: str, envelope: Dict[str, Any], priority: MessagePriority):
        """Send message immediately to queue."""
        if priority == MessagePriority.HIGH:
            # Use priority queue
            priority_key = self._get_priority_key(queue_name)
            score = time.time() + priority.value  # Higher priority = lower score
            await self.redis.zadd(priority_key, {json.dumps(envelope): score})
        else:
            # Use standard queue
            queue_key = self._get_queue_key(queue_name)
            await self.redis.lpush(queue_key, json.dumps(envelope))

    async def _send_delayed_message(self, envelope: Dict[str, Any], delay: int):
        """Send message with delay."""
        delayed_key = self._get_delayed_key(envelope["queue"])
        score = time.time() + delay
        await self.redis.zadd(delayed_key, {json.dumps(envelope): score})

    async def _receive_from_priority_queue(self, queue_name: str, timeout: int) -> Optional[str]:
        """Receive message from priority queue."""
        priority_key = self._get_priority_key(queue_name)
        
        # Get highest priority message
        result = await self.redis.zpopmin(priority_key, 1)
        
        if result:
            return result[0][0]  # Return message data
        
        return None

    async def _receive_from_standard_queue(self, queue_name: str, timeout: int) -> Optional[str]:
        """Receive message from standard queue."""
        queue_key = self._get_queue_key(queue_name)
        
        # Blocking pop with timeout
        result = await self.redis.brpop(queue_key, timeout)
        
        if result:
            return result[1]  # Return message data
        
        return None

    async def _move_to_processing(self, queue_name: str, envelope: Dict[str, Any]):
        """Move message to processing set."""
        processing_key = self._get_processing_key(queue_name)
        score = time.time()
        await self.redis.zadd(processing_key, {envelope["id"]: score})

    async def _get_message_envelope(self, queue_name: str, message_id: str) -> Optional[Dict[str, Any]]:
        """Get message envelope from storage."""
        # This would need to be implemented based on storage strategy
        # For now, return None
        return None

    async def _requeue_message(self, queue_name: str, envelope: Dict[str, Any], delay: int):
        """Requeue message with delay."""
        envelope["retry_at"] = (datetime.now() + timedelta(seconds=delay)).isoformat()
        await self._send_delayed_message(envelope, delay)

    async def _send_to_dead_letter(self, queue_name: str, envelope: Dict[str, Any], reason: Optional[str]):
        """Send message to dead letter queue."""
        dead_letter_key = self._get_dead_letter_key(queue_name)
        envelope["dead_letter_reason"] = reason
        envelope["dead_letter_at"] = datetime.now().isoformat()
        await self.redis.lpush(dead_letter_key, json.dumps(envelope))

    async def _worker_loop(self, config: WorkerConfig):
        """Main worker loop for processing messages."""
        try:
            self.logger.info(f"Starting worker: {config.name}")
            
            while True:
                try:
                    # Process messages from assigned queues
                    for queue_name in config.queues:
                        # Check for available capacity
                        if len(asyncio.current_task().get_name()) >= config.max_concurrent_tasks:
                            break
                        
                        # Receive message
                        envelope = await self.receive_message(
                            queue_name,
                            timeout=int(config.poll_interval),
                            auto_ack=config.auto_ack
                        )
                        
                        if envelope:
                            # Process message
                            await self._process_message(config, queue_name, envelope)
                    
                    # Sleep before next poll
                    await asyncio.sleep(config.poll_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in worker {config.name}: {e}")
                    await asyncio.sleep(1)
            
        except asyncio.CancelledError:
            self.logger.info(f"Worker {config.name} cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in worker {config.name}: {e}")

    async def _process_message(self, config: WorkerConfig, queue_name: str, envelope: Dict[str, Any]):
        """Process a single message."""
        try:
            message_id = envelope["id"]
            payload = envelope["payload"]
            
            # Determine message type
            message_type = payload.get("type", "unknown")
            
            # Find handler
            handler = self.task_handlers.get(message_type)
            
            if handler:
                # Execute handler
                try:
                    await asyncio.wait_for(handler(payload), timeout=config.timeout)
                    
                    # Acknowledge successful processing
                    if not config.auto_ack:
                        await self.acknowledge_message(queue_name, message_id)
                    
                except asyncio.TimeoutError:
                    self.logger.error(f"Handler timeout for message {message_id}")
                    await self.reject_message(queue_name, message_id, requeue=True, reason="timeout")
                
                except Exception as e:
                    self.logger.error(f"Handler error for message {message_id}: {e}")
                    await self.reject_message(queue_name, message_id, requeue=True, reason=str(e))
            else:
                self.logger.warning(f"No handler for message type: {message_type}")
                await self.reject_message(queue_name, message_id, requeue=False, reason="no_handler")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the queue system."""
        try:
            if not self.redis:
                return {"status": "unhealthy", "error": "Redis not connected"}
            
            # Test Redis connection
            await self.redis.ping()
            
            return {
                "status": "healthy",
                "stats": self.stats,
                "active_workers": len(self.workers),
                "total_queues": len(self.queue_configs)
            }
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def purge_queue(self, queue_name: str) -> int:
        """Purge all messages from a queue.
        
        Args:
            queue_name: Name of the queue to purge
            
        Returns:
            Number of messages purged
        """
        try:
            if not self.redis:
                raise RuntimeError("Redis connection not initialized")
            
            total_purged = 0
            
            # Purge standard queue
            queue_key = self._get_queue_key(queue_name)
            purged = await self.redis.delete(queue_key)
            total_purged += purged
            
            # Purge priority queue
            priority_key = self._get_priority_key(queue_name)
            purged = await self.redis.delete(priority_key)
            total_purged += purged
            
            # Purge processing set
            processing_key = self._get_processing_key(queue_name)
            purged = await self.redis.delete(processing_key)
            total_purged += purged
            
            # Purge delayed messages
            delayed_key = self._get_delayed_key(queue_name)
            purged = await self.redis.delete(delayed_key)
            total_purged += purged
            
            self.logger.info(f"Purged {total_purged} messages from queue {queue_name}")
            
            return total_purged
            
        except Exception as e:
            self.logger.error(f"Error purging queue {queue_name}: {e}")
            raise