#!/usr/bin/env python3
"""
Kafka Stream Processor
High-performance streaming data processor with comprehensive monitoring and error handling.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server


# Metrics
MESSAGES_PROCESSED = Counter('stream_messages_processed_total', 'Total processed messages', ['topic', 'partition', 'status'])
PROCESSING_TIME = Histogram('stream_processing_seconds', 'Time spent processing messages', ['topic', 'processor'])
ACTIVE_PROCESSORS = Gauge('stream_active_processors', 'Number of active stream processors', ['processor_type'])
CONSUMER_LAG = Gauge('stream_consumer_lag', 'Consumer lag in messages', ['topic', 'partition'])
ERROR_RATE = Counter('stream_processing_errors_total', 'Total processing errors', ['topic', 'error_type'])


class StreamingStatus(Enum):
    """Stream processing status."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ProcessingResult(Enum):
    """Processing result status."""
    SUCCESS = "success"
    RETRY = "retry"
    SKIP = "skip"
    FAILED = "failed"


@dataclass
class StreamMessage:
    """Structured stream message."""
    id: str
    topic: str
    partition: int
    offset: int
    timestamp: datetime
    key: Optional[str]
    value: Dict[str, Any]
    headers: Dict[str, str]
    retry_count: int = 0
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None


@dataclass
class ProcessorConfig:
    """Stream processor configuration."""
    name: str
    topics: List[str]
    consumer_group: str
    batch_size: int = 100
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 30000
    enable_auto_commit: bool = False
    auto_offset_reset: str = "latest"
    max_retries: int = 3
    retry_delay_seconds: int = 5
    dead_letter_topic: Optional[str] = None
    processing_timeout_seconds: int = 30
    enable_exactly_once: bool = True
    buffer_size: int = 10000


class KafkaStreamProcessor:
    """High-performance Kafka stream processor with advanced features."""
    
    def __init__(self, config: ProcessorConfig, kafka_config: Dict[str, Any], redis_url: str = "redis://localhost:6379/0"):
        self.config = config
        self.kafka_config = kafka_config
        self.redis_client = redis.Redis.from_url(redis_url)
        
        # Initialize Kafka components
        self.producer = None
        self.consumer = None
        
        # Processing state
        self.status = StreamingStatus.STOPPED
        self.message_buffer: List[StreamMessage] = []
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.error_handlers: Dict[str, Callable] = {}
        self.message_processors: Dict[str, Callable] = {}
        
        # Monitoring
        self.logger = logging.getLogger(f"stream_processor.{config.name}")
        self.metrics_enabled = True
        self.start_time = None
        self.last_commit_time = None
        
        # Circuit breaker
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 10
        self.circuit_breaker_reset_time = None
        
    async def initialize(self) -> None:
        """Initialize the stream processor."""
        try:
            self.logger.info(f"Initializing stream processor: {self.config.name}")
            
            # Initialize Kafka producer
            self.producer = KafkaProducer(
                **self.kafka_config,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=3,
                acks='all',
                enable_idempotence=True
            )
            
            # Initialize Kafka consumer
            self.consumer = KafkaConsumer(
                *self.config.topics,
                group_id=self.config.consumer_group,
                **self.kafka_config,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                max_poll_interval_ms=self.config.max_poll_interval_ms,
                session_timeout_ms=self.config.session_timeout_ms,
                consumer_timeout_ms=1000,
                max_poll_records=self.config.batch_size
            )
            
            # Test connections
            await self._test_kafka_connection()
            await self._test_redis_connection()
            
            self.status = StreamingStatus.STARTING
            self.logger.info("Stream processor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stream processor: {e}")
            self.status = StreamingStatus.ERROR
            raise
    
    async def _test_kafka_connection(self) -> None:
        """Test Kafka connection."""
        try:
            # Test producer
            self.producer.bootstrap_connected()
            
            # Test consumer
            partitions = self.consumer.list_consumer_group_offsets()
            self.logger.info(f"Kafka connection test successful, found {len(partitions)} partitions")
            
        except Exception as e:
            self.logger.error(f"Kafka connection test failed: {e}")
            raise
    
    async def _test_redis_connection(self) -> None:
        """Test Redis connection."""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            self.logger.info("Redis connection test successful")
        except Exception as e:
            self.logger.error(f"Redis connection test failed: {e}")
            raise
    
    def register_message_processor(self, topic: str, processor: Callable[[StreamMessage], ProcessingResult]) -> None:
        """Register a message processor for a specific topic."""
        self.message_processors[topic] = processor
        self.logger.info(f"Registered message processor for topic: {topic}")
    
    def register_error_handler(self, error_type: str, handler: Callable[[Exception, StreamMessage], None]) -> None:
        """Register an error handler for specific error types."""
        self.error_handlers[error_type] = handler
        self.logger.info(f"Registered error handler for: {error_type}")
    
    async def start_processing(self) -> None:
        """Start the stream processing."""
        if self.status in [StreamingStatus.RUNNING, StreamingStatus.STARTING]:
            self.logger.warning("Stream processor is already running")
            return
        
        try:
            self.status = StreamingStatus.STARTING
            self.start_time = datetime.utcnow()
            
            # Start metrics server if enabled
            if self.metrics_enabled:
                start_http_server(8000 + hash(self.config.name) % 1000)
            
            # Start main processing loop
            self.processing_tasks["main"] = asyncio.create_task(self._main_processing_loop())
            
            # Start monitoring task
            self.processing_tasks["monitor"] = asyncio.create_task(self._monitoring_loop())
            
            # Start health check task
            self.processing_tasks["health"] = asyncio.create_task(self._health_check_loop())
            
            self.status = StreamingStatus.RUNNING
            ACTIVE_PROCESSORS.labels(processor_type=self.config.name).inc()
            
            self.logger.info(f"Stream processor {self.config.name} started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start stream processor: {e}")
            self.status = StreamingStatus.ERROR
            raise
    
    async def _main_processing_loop(self) -> None:
        """Main message processing loop."""
        while self.status == StreamingStatus.RUNNING:
            try:
                # Check circuit breaker
                if self._is_circuit_breaker_open():
                    await asyncio.sleep(1)
                    continue
                
                # Poll for messages
                message_batch = await self._poll_messages()
                
                if not message_batch:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process message batch
                await self._process_message_batch(message_batch)
                
                # Commit offsets if auto-commit is disabled
                if not self.config.enable_auto_commit:
                    await self._commit_offsets()
                
            except Exception as e:
                self.logger.error(f"Error in main processing loop: {e}")
                ERROR_RATE.labels(topic="all", error_type="processing_loop").inc()
                await self._handle_processing_error(e)
                await asyncio.sleep(1)
    
    async def _poll_messages(self) -> List[StreamMessage]:
        """Poll messages from Kafka."""
        try:
            raw_messages = self.consumer.poll(timeout_ms=1000)
            messages = []
            
            for topic_partition, msgs in raw_messages.items():
                for msg in msgs:
                    stream_message = StreamMessage(
                        id=str(uuid.uuid4()),
                        topic=msg.topic,
                        partition=msg.partition,
                        offset=msg.offset,
                        timestamp=datetime.fromtimestamp(msg.timestamp / 1000),
                        key=msg.key,
                        value=msg.value,
                        headers={k: v.decode() if isinstance(v, bytes) else str(v) for k, v in msg.headers or []},
                        processing_started=datetime.utcnow()
                    )
                    messages.append(stream_message)
            
            if messages:
                self.logger.debug(f"Polled {len(messages)} messages")
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error polling messages: {e}")
            ERROR_RATE.labels(topic="all", error_type="polling").inc()
            return []
    
    async def _process_message_batch(self, messages: List[StreamMessage]) -> None:
        """Process a batch of messages."""
        processing_tasks = []
        
        for message in messages:
            task = asyncio.create_task(self._process_single_message(message))
            processing_tasks.append(task)
        
        # Wait for all messages to be processed
        results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Handle results
        for i, result in enumerate(results):
            message = messages[i]
            if isinstance(result, Exception):
                await self._handle_message_error(result, message)
            else:
                await self._handle_message_success(message, result)
    
    async def _process_single_message(self, message: StreamMessage) -> ProcessingResult:
        """Process a single message."""
        try:
            # Apply timeout
            return await asyncio.wait_for(
                self._apply_message_processor(message),
                timeout=self.config.processing_timeout_seconds
            )
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Message processing timeout: {message.id}")
            ERROR_RATE.labels(topic=message.topic, error_type="timeout").inc()
            return ProcessingResult.RETRY
        except Exception as e:
            self.logger.error(f"Error processing message {message.id}: {e}")
            ERROR_RATE.labels(topic=message.topic, error_type="processing").inc()
            raise
    
    async def _apply_message_processor(self, message: StreamMessage) -> ProcessingResult:
        """Apply the appropriate message processor."""
        processor = self.message_processors.get(message.topic)
        
        if not processor:
            self.logger.warning(f"No processor registered for topic: {message.topic}")
            return ProcessingResult.SKIP
        
        # Record processing time
        start_time = time.time()
        
        try:
            # Execute processor
            result = await asyncio.get_event_loop().run_in_executor(None, processor, message)
            
            # Record metrics
            processing_time = time.time() - start_time
            PROCESSING_TIME.labels(topic=message.topic, processor=self.config.name).observe(processing_time)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            PROCESSING_TIME.labels(topic=message.topic, processor=self.config.name).observe(processing_time)
            raise
    
    async def _handle_message_success(self, message: StreamMessage, result: ProcessingResult) -> None:
        """Handle successful message processing."""
        message.processing_completed = datetime.utcnow()
        
        if result == ProcessingResult.SUCCESS:
            MESSAGES_PROCESSED.labels(topic=message.topic, partition=message.partition, status="success").inc()
            self.logger.debug(f"Successfully processed message: {message.id}")
        
        elif result == ProcessingResult.RETRY:
            await self._retry_message(message)
        
        elif result == ProcessingResult.SKIP:
            MESSAGES_PROCESSED.labels(topic=message.topic, partition=message.partition, status="skipped").inc()
            self.logger.debug(f"Skipped message: {message.id}")
        
        # Store processing metadata in Redis
        await self._store_processing_metadata(message, str(result.value))
    
    async def _handle_message_error(self, error: Exception, message: StreamMessage) -> None:
        """Handle message processing errors."""
        self.circuit_breaker_failures += 1
        
        # Apply error handler if available
        error_type = type(error).__name__
        error_handler = self.error_handlers.get(error_type)
        
        if error_handler:
            try:
                await asyncio.get_event_loop().run_in_executor(None, error_handler, error, message)
            except Exception as handler_error:
                self.logger.error(f"Error handler failed: {handler_error}")
        
        # Retry or send to dead letter queue
        if message.retry_count < self.config.max_retries:
            await self._retry_message(message)
        else:
            await self._send_to_dead_letter_queue(message, error)
        
        MESSAGES_PROCESSED.labels(topic=message.topic, partition=message.partition, status="failed").inc()
    
    async def _retry_message(self, message: StreamMessage) -> None:
        """Retry message processing."""
        message.retry_count += 1
        
        # Exponential backoff
        delay = self.config.retry_delay_seconds * (2 ** (message.retry_count - 1))
        
        self.logger.info(f"Retrying message {message.id} (attempt {message.retry_count}) after {delay}s")
        
        # Schedule retry
        await asyncio.sleep(delay)
        result = await self._process_single_message(message)
        await self._handle_message_success(message, result)
    
    async def _send_to_dead_letter_queue(self, message: StreamMessage, error: Exception) -> None:
        """Send message to dead letter queue."""
        if not self.config.dead_letter_topic:
            self.logger.warning(f"No dead letter topic configured, dropping message: {message.id}")
            return
        
        try:
            dead_letter_message = {
                "original_message": asdict(message),
                "error": str(error),
                "error_type": type(error).__name__,
                "failed_at": datetime.utcnow().isoformat(),
                "processor": self.config.name
            }
            
            self.producer.send(
                self.config.dead_letter_topic,
                value=dead_letter_message,
                key=message.key
            )
            
            self.logger.info(f"Sent message {message.id} to dead letter queue")
            
        except Exception as e:
            self.logger.error(f"Failed to send message to dead letter queue: {e}")
    
    async def _store_processing_metadata(self, message: StreamMessage, result: str) -> None:
        """Store processing metadata in Redis."""
        try:
            metadata = {
                "message_id": message.id,
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
                "processing_result": result,
                "retry_count": message.retry_count,
                "processing_time": (message.processing_completed - message.processing_started).total_seconds(),
                "processor": self.config.name,
                "timestamp": message.processing_completed.isoformat()
            }
            
            # Store with TTL of 24 hours
            key = f"stream_processing:{self.config.name}:{message.id}"
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.redis_client.setex(key, 86400, json.dumps(metadata))
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store processing metadata: {e}")
    
    async def _commit_offsets(self) -> None:
        """Commit consumer offsets."""
        try:
            self.consumer.commit()
            self.last_commit_time = datetime.utcnow()
            self.logger.debug("Committed consumer offsets")
        except Exception as e:
            self.logger.error(f"Failed to commit offsets: {e}")
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_breaker_failures < self.circuit_breaker_threshold:
            return False
        
        if not self.circuit_breaker_reset_time:
            self.circuit_breaker_reset_time = datetime.utcnow() + timedelta(minutes=5)
            self.logger.warning("Circuit breaker opened")
            return True
        
        if datetime.utcnow() >= self.circuit_breaker_reset_time:
            self.circuit_breaker_failures = 0
            self.circuit_breaker_reset_time = None
            self.logger.info("Circuit breaker reset")
            return False
        
        return True
    
    async def _handle_processing_error(self, error: Exception) -> None:
        """Handle general processing errors."""
        self.circuit_breaker_failures += 1
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.logger.error("Circuit breaker threshold reached, temporary pause")
            await asyncio.sleep(5)
    
    async def _monitoring_loop(self) -> None:
        """Monitor stream processing health and metrics."""
        while self.status == StreamingStatus.RUNNING:
            try:
                # Update consumer lag metrics
                await self._update_consumer_lag_metrics()
                
                # Update processing metrics
                await self._update_processing_metrics()
                
                # Check circuit breaker status
                if self._is_circuit_breaker_open():
                    self.logger.warning("Circuit breaker is open, monitoring processing health")
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_consumer_lag_metrics(self) -> None:
        """Update consumer lag metrics."""
        try:
            partitions = self.consumer.assignment()
            for partition in partitions:
                # Get current offset
                current_offset = self.consumer.position(partition)
                
                # Get high water mark
                high_water_mark = self.consumer.highwater(partition)
                
                if high_water_mark is not None and current_offset is not None:
                    lag = high_water_mark - current_offset
                    CONSUMER_LAG.labels(topic=partition.topic, partition=partition.partition).set(lag)
                    
        except Exception as e:
            self.logger.error(f"Failed to update consumer lag metrics: {e}")
    
    async def _update_processing_metrics(self) -> None:
        """Update general processing metrics."""
        try:
            # Update active processor count
            ACTIVE_PROCESSORS.labels(processor_type=self.config.name).set(1 if self.status == StreamingStatus.RUNNING else 0)
            
            # Store health status in Redis
            health_data = {
                "processor": self.config.name,
                "status": self.status.value,
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
                "circuit_breaker_failures": self.circuit_breaker_failures,
                "last_commit": self.last_commit_time.isoformat() if self.last_commit_time else None,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            key = f"stream_health:{self.config.name}"
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.redis_client.setex(key, 300, json.dumps(health_data))
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update processing metrics: {e}")
    
    async def _health_check_loop(self) -> None:
        """Perform regular health checks."""
        while self.status == StreamingStatus.RUNNING:
            try:
                # Check Kafka connection
                if not await self._check_kafka_health():
                    self.logger.error("Kafka health check failed")
                    ERROR_RATE.labels(topic="system", error_type="kafka_health").inc()
                
                # Check Redis connection
                if not await self._check_redis_health():
                    self.logger.error("Redis health check failed")
                    ERROR_RATE.labels(topic="system", error_type="redis_health").inc()
                
                await asyncio.sleep(60)  # Health check every minute
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_kafka_health(self) -> bool:
        """Check Kafka health."""
        try:
            # Check if consumer is still connected
            if not self.consumer:
                return False
            
            # Try to get metadata
            metadata = self.consumer.list_consumer_group_offsets()
            return True
            
        except Exception:
            return False
    
    async def _check_redis_health(self) -> bool:
        """Check Redis health."""
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            return True
        except Exception:
            return False
    
    async def stop_processing(self) -> None:
        """Stop the stream processing."""
        if self.status == StreamingStatus.STOPPED:
            return
        
        self.logger.info("Stopping stream processor...")
        self.status = StreamingStatus.STOPPING
        
        # Cancel all processing tasks
        for task_name, task in self.processing_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Commit final offsets
        if not self.config.enable_auto_commit:
            await self._commit_offsets()
        
        # Close connections
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        # Update metrics
        ACTIVE_PROCESSORS.labels(processor_type=self.config.name).dec()
        
        self.status = StreamingStatus.STOPPED
        self.logger.info("Stream processor stopped successfully")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "processor_name": self.config.name,
            "status": self.status.value,
            "uptime_seconds": uptime,
            "circuit_breaker_failures": self.circuit_breaker_failures,
            "circuit_breaker_open": self._is_circuit_breaker_open(),
            "last_commit": self.last_commit_time.isoformat() if self.last_commit_time else None,
            "topics": self.config.topics,
            "consumer_group": self.config.consumer_group
        }


# Example usage and message processors
class AnomalyDetectionProcessor:
    """Example anomaly detection message processor."""
    
    def __init__(self):
        self.logger = logging.getLogger("anomaly_processor")
        self.anomaly_threshold = 0.8
    
    def process_log_message(self, message: StreamMessage) -> ProcessingResult:
        """Process log messages for anomaly detection."""
        try:
            log_data = message.value
            
            # Validate required fields
            if not all(key in log_data for key in ['timestamp', 'service', 'level', 'message']):
                self.logger.warning(f"Missing required fields in message: {message.id}")
                return ProcessingResult.SKIP
            
            # Simulate anomaly detection
            anomaly_score = self._calculate_anomaly_score(log_data)
            
            if anomaly_score > self.anomaly_threshold:
                self.logger.warning(f"Anomaly detected in message {message.id}: score={anomaly_score}")
                # In real implementation, trigger alert or store in database
                return ProcessingResult.SUCCESS
            
            return ProcessingResult.SUCCESS
            
        except Exception as e:
            self.logger.error(f"Error processing log message: {e}")
            return ProcessingResult.RETRY
    
    def _calculate_anomaly_score(self, log_data: Dict[str, Any]) -> float:
        """Calculate anomaly score for log data."""
        # Placeholder implementation - replace with actual anomaly detection logic
        import random
        return random.random()


# Example initialization
async def create_stream_processor():
    """Create and configure stream processor."""
    config = ProcessorConfig(
        name="anomaly_detection_processor",
        topics=["application_logs", "system_metrics", "security_events"],
        consumer_group="anomaly_detection_group",
        batch_size=100,
        max_retries=3,
        dead_letter_topic="anomaly_detection_dlq",
        processing_timeout_seconds=30
    )
    
    kafka_config = {
        "bootstrap_servers": ["localhost:9092"],
        "security_protocol": "PLAINTEXT"
    }
    
    processor = KafkaStreamProcessor(config, kafka_config)
    await processor.initialize()
    
    # Register message processors
    anomaly_processor = AnomalyDetectionProcessor()
    processor.register_message_processor("application_logs", anomaly_processor.process_log_message)
    
    return processor