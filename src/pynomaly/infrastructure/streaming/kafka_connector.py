"""Kafka streaming connector for real-time data processing."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Any, Callable
from dataclasses import asdict

from pynomaly.domain.services.streaming_service import StreamRecord, StreamingResult

logger = logging.getLogger(__name__)

# Optional Kafka dependencies
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    AIOKafkaConsumer = None
    AIOKafkaProducer = None
    KafkaError = Exception
    KAFKA_AVAILABLE = False
    logger.warning("Kafka dependencies not available. Install with: pip install aiokafka")


class KafkaStreamConnector:
    """Kafka connector for streaming anomaly detection."""
    
    def __init__(
        self,
        bootstrap_servers: List[str],
        input_topic: str,
        output_topic: Optional[str] = None,
        consumer_group: str = "pynomaly-streaming",
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
        value_deserializer: Optional[Callable] = None,
        value_serializer: Optional[Callable] = None
    ):
        """Initialize Kafka connector.
        
        Args:
            bootstrap_servers: List of Kafka bootstrap servers
            input_topic: Topic to consume messages from
            output_topic: Topic to publish results to (optional)
            consumer_group: Consumer group ID
            auto_offset_reset: Auto offset reset strategy
            enable_auto_commit: Whether to enable auto commit
            value_deserializer: Custom value deserializer
            value_serializer: Custom value serializer
        """
        if not KAFKA_AVAILABLE:
            raise ImportError("Kafka dependencies not available. Install with: pip install aiokafka")
        
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer_group = consumer_group
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        
        self.value_deserializer = value_deserializer or self._default_deserializer
        self.value_serializer = value_serializer or self._default_serializer
        
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.producer: Optional[AIOKafkaProducer] = None
        self._running = False
        
        self._stats = {
            "messages_consumed": 0,
            "messages_produced": 0,
            "errors": 0,
            "last_message_time": None
        }
    
    async def start(self) -> None:
        """Start Kafka consumer and producer."""
        try:
            # Initialize consumer
            self.consumer = AIOKafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=self.enable_auto_commit,
                value_deserializer=self.value_deserializer
            )
            
            # Initialize producer if output topic specified
            if self.output_topic:
                self.producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=self.value_serializer
                )
                await self.producer.start()
            
            await self.consumer.start()
            self._running = True
            logger.info(f"Kafka connector started for topic: {self.input_topic}")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka connector: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop Kafka consumer and producer."""
        self._running = False
        
        try:
            if self.consumer:
                await self.consumer.stop()
            if self.producer:
                await self.producer.stop()
            logger.info("Kafka connector stopped")
        except Exception as e:
            logger.error(f"Error stopping Kafka connector: {e}")
    
    async def consume_stream(self) -> AsyncIterator[StreamRecord]:
        """Consume stream records from Kafka topic.
        
        Yields:
            StreamRecord instances from Kafka messages
        """
        if not self.consumer:
            raise RuntimeError("Kafka consumer not started")
        
        try:
            async for message in self.consumer:
                if not self._running:
                    break
                
                try:
                    # Convert Kafka message to StreamRecord
                    record = self._message_to_record(message)
                    self._stats["messages_consumed"] += 1
                    self._stats["last_message_time"] = datetime.now()
                    yield record
                    
                except Exception as e:
                    logger.error(f"Failed to process Kafka message: {e}")
                    self._stats["errors"] += 1
                    continue
        
        except Exception as e:
            logger.error(f"Kafka consumption error: {e}")
            self._stats["errors"] += 1
            raise
    
    async def publish_result(self, result: StreamingResult) -> None:
        """Publish streaming result to output topic.
        
        Args:
            result: StreamingResult to publish
        """
        if not self.producer or not self.output_topic:
            return
        
        try:
            # Convert result to message
            message_data = self._result_to_message(result)
            
            # Send to Kafka
            await self.producer.send_and_wait(
                self.output_topic,
                value=message_data,
                key=result.record_id.encode('utf-8')
            )
            
            self._stats["messages_produced"] += 1
            
        except Exception as e:
            logger.error(f"Failed to publish result to Kafka: {e}")
            self._stats["errors"] += 1
    
    async def publish_results_batch(self, results: List[StreamingResult]) -> None:
        """Publish batch of results to output topic.
        
        Args:
            results: List of StreamingResults to publish
        """
        if not self.producer or not self.output_topic:
            return
        
        try:
            # Send all results
            tasks = []
            for result in results:
                message_data = self._result_to_message(result)
                task = self.producer.send(
                    self.output_topic,
                    value=message_data,
                    key=result.record_id.encode('utf-8')
                )
                tasks.append(task)
            
            # Wait for all sends to complete
            await asyncio.gather(*tasks)
            self._stats["messages_produced"] += len(results)
            
        except Exception as e:
            logger.error(f"Failed to publish batch results to Kafka: {e}")
            self._stats["errors"] += 1
    
    def _message_to_record(self, message) -> StreamRecord:
        """Convert Kafka message to StreamRecord."""
        # Extract timestamp
        timestamp = datetime.fromtimestamp(message.timestamp / 1000.0)
        
        # Message value should be the data
        data = message.value
        if not isinstance(data, dict):
            data = {"raw_data": data}
        
        # Extract metadata from headers
        metadata = {}
        if message.headers:
            for key, value in message.headers:
                try:
                    metadata[key] = value.decode('utf-8')
                except:
                    metadata[key] = str(value)
        
        # Add Kafka-specific metadata
        metadata.update({
            "kafka_topic": message.topic,
            "kafka_partition": message.partition,
            "kafka_offset": message.offset,
            "kafka_key": message.key.decode('utf-8') if message.key else None
        })
        
        return StreamRecord(
            id=f"kafka_{message.topic}_{message.partition}_{message.offset}",
            timestamp=timestamp,
            data=data,
            metadata=metadata
        )
    
    def _result_to_message(self, result: StreamingResult) -> Dict[str, Any]:
        """Convert StreamingResult to Kafka message data."""
        # Convert result to dictionary
        message_data = {
            "record_id": result.record_id,
            "timestamp": result.timestamp.isoformat(),
            "anomaly_score": result.anomaly_score,
            "is_anomaly": result.is_anomaly,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "metadata": result.metadata
        }
        
        return message_data
    
    def _default_deserializer(self, data: bytes) -> Dict[str, Any]:
        """Default message deserializer."""
        try:
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError:
            # If not JSON, treat as raw data
            return {"raw_data": data.decode('utf-8', errors='ignore')}
        except Exception:
            return {"raw_data": str(data)}
    
    def _default_serializer(self, data: Dict[str, Any]) -> bytes:
        """Default message serializer."""
        try:
            return json.dumps(data, default=str).encode('utf-8')
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return json.dumps({"error": "serialization_failed"}).encode('utf-8')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return self._stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Kafka connection."""
        health = {
            "kafka_connected": False,
            "consumer_ready": False,
            "producer_ready": False,
            "last_error": None
        }
        
        try:
            # Check consumer
            if self.consumer and self._running:
                health["consumer_ready"] = True
            
            # Check producer
            if self.producer and self.output_topic:
                health["producer_ready"] = True
            
            health["kafka_connected"] = health["consumer_ready"]
            
        except Exception as e:
            health["last_error"] = str(e)
        
        return health