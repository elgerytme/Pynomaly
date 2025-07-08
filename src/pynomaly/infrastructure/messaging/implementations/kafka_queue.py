"""
Kafka Message Queue Implementation

This module provides a Kafka-based message queue implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from ..core import MessageQueue, Message
from ..config import MessageQueueConfig, QueueConfig

logger = logging.getLogger(__name__)


class KafkaMessageQueue(MessageQueue):
    """Kafka-based message queue implementation."""
    
    def __init__(self, config: MessageQueueConfig):
        super().__init__(config)
        self._producer = None
        self._consumer = None
    
    async def connect(self) -> None:
        """Connect to Kafka."""
        try:
            from kafka import KafkaProducer, KafkaConsumer
            
            # Create producer
            self._producer = KafkaProducer(
                bootstrap_servers=self.config.connection.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            self._is_connected = True
            logger.info("Connected to Kafka message queue")
            
        except ImportError:
            raise ImportError("kafka-python package is required for Kafka message queue")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self._producer:
            self._producer.close()
        if self._consumer:
            self._consumer.close()
        self._is_connected = False
        logger.info("Disconnected from Kafka message queue")
    
    async def send(
        self, 
        message: Message, 
        queue_name: Optional[str] = None,
        routing_key: Optional[str] = None
    ) -> str:
        """Send a message to a Kafka topic."""
        topic = queue_name or message.queue_name
        if not topic:
            raise ValueError("Topic name must be provided")
        
        message_data = message.to_dict()
        future = self._producer.send(topic, message_data)
        
        # Wait for message to be sent
        record_metadata = future.get(timeout=10)
        self.state.messages_sent += 1
        
        logger.debug(f"Sent message ID {message.id} to Kafka topic {topic}")
        return message.id
    
    async def receive(
        self, 
        queue_name: str, 
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Receive a message from a Kafka topic."""
        if not self._consumer:
            from kafka import KafkaConsumer
            self._consumer = KafkaConsumer(
                queue_name,
                bootstrap_servers=self.config.connection.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=int(timeout * 1000) if timeout else 1000
            )
        
        try:
            for message_data in self._consumer:
                message = Message.from_dict(message_data.value)
                logger.debug(f"Received message ID {message.id} from Kafka topic {queue_name}")
                return message
        except Exception as e:
            logger.error(f"Error receiving from Kafka topic {queue_name}: {e}")
        
        return None
    
    async def acknowledge(self, message: Message) -> None:
        """Acknowledge message processing."""
        if self._consumer:
            self._consumer.commit()
        logger.debug(f"Acknowledged message ID {message.id}")
    
    async def reject(self, message: Message, requeue: bool = False) -> None:
        """Reject a message (Kafka doesn't support message rejection)."""
        logger.debug(f"Rejected message ID {message.id}, requeue={requeue}")
    
    async def create_queue(self, queue_config: QueueConfig) -> None:
        """Create a Kafka topic (requires admin privileges)."""
        # Note: Topic creation typically requires admin API
        self.state.active_queues[queue_config.name] = queue_config
        logger.info(f"Created Kafka topic {queue_config.name}")
    
    async def delete_queue(self, queue_name: str) -> None:
        """Delete a Kafka topic (requires admin privileges)."""
        # Note: Topic deletion typically requires admin API
        if queue_name in self.state.active_queues:
            del self.state.active_queues[queue_name]
        logger.info(f"Deleted Kafka topic {queue_name}")
    
    async def purge_queue(self, queue_name: str) -> int:
        """Purge messages from a Kafka topic (not supported)."""
        logger.warning("Kafka does not support message purging")
        return 0
    
    async def get_queue_size(self, queue_name: str) -> int:
        """Get the number of messages in a Kafka topic (not easily available)."""
        logger.warning("Kafka message count not easily available")
        return 0
    
    async def health_check(self) -> bool:
        """Check the health of the Kafka connection."""
        try:
            if self._producer:
                # Try to get metadata
                metadata = self._producer.partitions_for('__consumer_offsets')
                return True
        except Exception:
            return False
        return False
