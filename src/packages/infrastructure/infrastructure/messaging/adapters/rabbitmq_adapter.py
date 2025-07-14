"""RabbitMQ-based message queue adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

try:
    import aio_pika
    from aio_pika import Connection, DeliveryMode, ExchangeType, Message as RabbitMessage
    from aio_pika.abc import AbstractChannel, AbstractConnection, AbstractExchange, AbstractQueue
except ImportError:
    # RabbitMQ support is optional
    aio_pika = None
    Connection = None
    DeliveryMode = None
    ExchangeType = None
    RabbitMessage = None

from ..config.messaging_settings import MessagingSettings
from ..models.messages import Message, MessagePriority, MessageStatus
from ..models.tasks import Task, TaskStatus
from ..protocols.message_queue_protocol import MessageQueueProtocol

logger = logging.getLogger(__name__)


class RabbitMQAdapter(MessageQueueProtocol):
    """RabbitMQ-based message queue implementation using AMQP."""

    def __init__(self, settings: MessagingSettings, rabbitmq_url: str):
        """Initialize RabbitMQ adapter.

        Args:
            settings: Messaging settings
            rabbitmq_url: RabbitMQ connection URL

        Raises:
            ImportError: If aio_pika is not installed
        """
        if aio_pika is None:
            raise ImportError(
                "aio_pika is required for RabbitMQ support. "
                "Install with: pip install aio_pika"
            )

        self.settings = settings
        self.rabbitmq_url = rabbitmq_url
        self._connection: AbstractConnection | None = None
        self._channel: AbstractChannel | None = None
        self._connected = False
        
        # Exchange and queue configurations
        self._exchange_name = "pynomaly"
        self._dlx_name = "pynomaly.dlx"  # Dead letter exchange
        self._task_exchange = "pynomaly.tasks"
        
        # Queue configurations
        self._queues: dict[str, AbstractQueue] = {}
        self._dlq_queues: dict[str, AbstractQueue] = {}

    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            # Establish connection
            self._connection = await aio_pika.connect_robust(
                self.rabbitmq_url,
                heartbeat=30,
                blocked_connection_timeout=30,
            )
            
            # Create channel
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=self.settings.task_batch_size)
            
            # Declare exchanges
            await self._setup_exchanges()
            
            self._connected = True
            logger.info("Successfully connected to RabbitMQ")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Close connection to RabbitMQ."""
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            self._connection = None
            self._channel = None
            self._connected = False
            logger.info("Disconnected from RabbitMQ")

    async def _setup_exchanges(self) -> None:
        """Set up RabbitMQ exchanges."""
        if not self._channel:
            return

        # Main exchange
        self._exchange = await self._channel.declare_exchange(
            self._exchange_name,
            ExchangeType.TOPIC,
            durable=True
        )
        
        # Dead letter exchange
        self._dlx = await self._channel.declare_exchange(
            self._dlx_name,
            ExchangeType.TOPIC,
            durable=True
        )
        
        # Task exchange
        self._task_exchange_obj = await self._channel.declare_exchange(
            self._task_exchange,
            ExchangeType.TOPIC,
            durable=True
        )

    async def _declare_queue(self, queue_name: str) -> AbstractQueue:
        """Declare a queue with dead letter configuration.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Declared queue
        """
        if not self._channel:
            raise RuntimeError("Not connected to RabbitMQ")

        # Declare dead letter queue first
        dlq_name = f"{queue_name}.dlq"
        dlq = await self._channel.declare_queue(
            dlq_name,
            durable=True,
            arguments={
                "x-message-ttl": self.settings.dead_letter_ttl * 1000,  # Convert to ms
            }
        )
        await dlq.bind(self._dlx, routing_key=f"dlq.{queue_name}")
        self._dlq_queues[queue_name] = dlq

        # Declare main queue with DLQ configuration
        queue = await self._channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": self._dlx_name,
                "x-dead-letter-routing-key": f"dlq.{queue_name}",
                "x-message-ttl": self.settings.queue_default_ttl * 1000,  # Convert to ms
            }
        )
        
        # Bind to main exchange
        await queue.bind(self._exchange, routing_key=queue_name)
        self._queues[queue_name] = queue
        
        return queue

    async def send_message(self, queue_name: str, message: Message) -> bool:
        """Send a message to a queue.
        
        Args:
            queue_name: Name of the queue
            message: Message to send
            
        Returns:
            True if message was sent successfully
        """
        if not self._connected or not self._channel:
            logger.error("Not connected to RabbitMQ")
            return False

        try:
            # Ensure queue exists
            if queue_name not in self._queues:
                await self._declare_queue(queue_name)

            # Convert priority
            priority = self._convert_priority(message.priority)
            
            # Create AMQP message
            message_body = json.dumps(message.to_dict()).encode()
            rabbit_message = RabbitMessage(
                message_body,
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=priority,
                correlation_id=message.correlation_id,
                reply_to=message.reply_to,
                headers={
                    "message_id": message.id,
                    "message_type": message.message_type,
                    "created_at": message.created_at.isoformat(),
                    **message.headers,
                }
            )
            
            # Set expiration if specified
            if message.expires_at:
                ttl_ms = int((message.expires_at.timestamp() - time.time()) * 1000)
                if ttl_ms > 0:
                    rabbit_message.expiration = str(ttl_ms)

            # Publish message
            await self._exchange.publish(
                rabbit_message,
                routing_key=queue_name
            )
            
            logger.debug(f"Message {message.id} sent to {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message {message.id} to {queue_name}: {e}")
            return False

    def _convert_priority(self, priority: MessagePriority) -> int:
        """Convert message priority to RabbitMQ priority (0-255).
        
        Args:
            priority: Message priority
            
        Returns:
            RabbitMQ priority value
        """
        priority_map = {
            MessagePriority.LOW: 1,
            MessagePriority.NORMAL: 10,
            MessagePriority.HIGH: 50,
            MessagePriority.CRITICAL: 100,
        }
        return priority_map.get(priority, 10)

    async def receive_message(self, queue_name: str, timeout: int | None = None) -> Message | None:
        """Receive a message from a queue.
        
        Args:
            queue_name: Name of the queue
            timeout: Timeout in seconds (None for blocking)
            
        Returns:
            Message if available, None if timeout
        """
        if not self._connected or not self._channel:
            logger.error("Not connected to RabbitMQ")
            return None

        try:
            # Ensure queue exists
            if queue_name not in self._queues:
                await self._declare_queue(queue_name)

            queue = self._queues[queue_name]
            
            # Get message with timeout
            if timeout:
                try:
                    rabbit_message = await asyncio.wait_for(
                        queue.get(no_ack=False),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    return None
            else:
                rabbit_message = await queue.get(no_ack=False)

            if rabbit_message is None:
                return None

            # Parse message
            message_data = json.loads(rabbit_message.body.decode())
            message = Message.from_dict(message_data)
            message.mark_processing()
            
            # Store RabbitMQ metadata for acknowledgment
            message.headers["rabbitmq_delivery_tag"] = str(rabbit_message.delivery_tag)
            message.headers["rabbitmq_exchange"] = rabbit_message.exchange
            message.headers["rabbitmq_routing_key"] = rabbit_message.routing_key
            
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
        if not self._connected or not self._channel:
            logger.error("Not connected to RabbitMQ")
            return

        # Ensure queue exists
        if queue_name not in self._queues:
            await self._declare_queue(queue_name)

        queue = self._queues[queue_name]
        
        while self._connected:
            try:
                # Get batch of messages
                for _ in range(batch_size):
                    try:
                        rabbit_message = await asyncio.wait_for(
                            queue.get(no_ack=False),
                            timeout=1.0  # Short timeout for batch processing
                        )
                        
                        if rabbit_message is None:
                            break
                            
                        # Parse message
                        message_data = json.loads(rabbit_message.body.decode())
                        message = Message.from_dict(message_data)
                        message.mark_processing()
                        
                        # Store RabbitMQ metadata
                        message.headers["rabbitmq_delivery_tag"] = str(rabbit_message.delivery_tag)
                        message.headers["rabbitmq_exchange"] = rabbit_message.exchange
                        message.headers["rabbitmq_routing_key"] = rabbit_message.routing_key
                        
                        yield message
                        
                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        logger.error(f"Failed to process message: {e}")
                        continue
                        
                # Brief pause before next batch
                await asyncio.sleep(0.1)
                        
            except Exception as e:
                logger.error(f"Failed to receive messages from {queue_name}: {e}")
                await asyncio.sleep(1)

    async def acknowledge_message(self, message: Message) -> bool:
        """Acknowledge that a message has been processed.
        
        Args:
            message: Message to acknowledge
            
        Returns:
            True if acknowledgment was successful
        """
        if not self._connected or not self._channel:
            return False

        try:
            delivery_tag = message.headers.get("rabbitmq_delivery_tag")
            if not delivery_tag:
                logger.error(f"Missing RabbitMQ delivery tag for message {message.id}")
                return False

            # Acknowledge the message
            await self._channel.basic_ack(delivery_tag=int(delivery_tag))
            
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
        if not self._connected or not self._channel:
            return False

        try:
            delivery_tag = message.headers.get("rabbitmq_delivery_tag")
            if not delivery_tag:
                logger.error(f"Missing RabbitMQ delivery tag for message {message.id}")
                return False

            if requeue and message.can_retry():
                # Reject and requeue
                await self._channel.basic_nack(
                    delivery_tag=int(delivery_tag),
                    requeue=True
                )
                message.mark_retrying()
            else:
                # Reject without requeue (will go to DLQ)
                await self._channel.basic_nack(
                    delivery_tag=int(delivery_tag),
                    requeue=False
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
        if not self._connected or not self._channel:
            raise RuntimeError("Not connected to RabbitMQ")

        try:
            task.mark_queued()
            
            # Create message for task processing
            message = Message(
                queue_name=task.queue_name,
                payload={
                    "task_id": task.id,
                    "task_type": task.task_type.value,
                    "task_data": task.to_dict(),
                },
                message_type="task",
                headers={"task_id": task.id}
            )
            
            # Send to task queue
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
        # Note: RabbitMQ doesn't have built-in task storage
        # In a real implementation, you'd use an external store like Redis or database
        logger.warning("Task status retrieval not implemented for RabbitMQ adapter")
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled successfully
        """
        # Note: Task cancellation would require external task storage
        logger.warning("Task cancellation not implemented for RabbitMQ adapter")
        return False

    async def get_queue_stats(self, queue_name: str) -> dict[str, Any]:
        """Get statistics for a queue.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Dictionary with queue statistics
        """
        if not self._connected or not self._channel:
            return {}

        try:
            # Ensure queue exists
            if queue_name not in self._queues:
                await self._declare_queue(queue_name)

            queue = self._queues[queue_name]
            dlq = self._dlq_queues.get(queue_name)
            
            # Get queue info using management API calls
            # Note: This is a simplified implementation
            return {
                "queue_name": queue_name,
                "messages": "N/A",  # Would need management API
                "consumers": "N/A",  # Would need management API
                "dead_letter_messages": "N/A",  # Would need management API
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
        if not self._connected or not self._channel:
            return 0

        try:
            # Ensure queue exists
            if queue_name not in self._queues:
                await self._declare_queue(queue_name)

            queue = self._queues[queue_name]
            
            # Purge the queue
            result = await queue.purge()
            
            logger.info(f"Purged {result} messages from queue {queue_name}")
            return result
            
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
        if not self._connected:
            return False

        try:
            await self._declare_queue(queue_name)
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
        if not self._connected or not self._channel:
            return False

        try:
            # Delete main queue
            if queue_name in self._queues:
                queue = self._queues[queue_name]
                await queue.delete()
                del self._queues[queue_name]
            
            # Delete DLQ
            if queue_name in self._dlq_queues:
                dlq = self._dlq_queues[queue_name]
                await dlq.delete()
                del self._dlq_queues[queue_name]
            
            logger.info(f"Deleted queue {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete queue {queue_name}: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if the message queue connection is healthy.
        
        Returns:
            True if connection is healthy
        """
        if not self._connected or not self._connection:
            return False

        try:
            return not self._connection.is_closed
        except Exception:
            return False