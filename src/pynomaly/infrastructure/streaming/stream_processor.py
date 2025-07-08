"""Real-time streaming anomaly detection processor."""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    import kafka
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ...domain.entities.dataset import Dataset
from ...domain.entities.detection_result import DetectionResult
from ...domain.services.advanced_detection_service import get_detection_service, DetectionAlgorithm
from ...shared.config import Config
from ..monitoring.distributed_tracing import trace_operation
from ..messaging.factory import MessageQueueFactory
from ..messaging.core import Message

logger = logging.getLogger(__name__)


class StreamSource(Enum):
    """Available stream sources."""
    KAFKA = "kafka"
    REDIS = "redis"
    WEBSOCKET = "websocket"
    FILE = "file"
    HTTP = "http"
    MEMORY = "memory"


class StreamFormat(Enum):
    """Stream data formats."""
    JSON = "json"
    CSV = "csv"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    PARQUET = "parquet"


@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    
    # Source configuration
    source_type: StreamSource
    source_config: Dict[str, Any] = field(default_factory=dict)
    
    # Data format
    data_format: StreamFormat = StreamFormat.JSON
    
    # Processing configuration
    batch_size: int = 1000
    batch_timeout_seconds: float = 30.0
    max_memory_mb: int = 512
    
    # Detection configuration
    detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST
    detection_config: Optional[Dict[str, Any]] = None
    
    # Output configuration
    output_topic: Optional[str] = None
    output_format: StreamFormat = StreamFormat.JSON
    
    # Performance tuning
    enable_backpressure: bool = True
    max_queue_size: int = 10000
    worker_threads: int = 4
    enable_compression: bool = True
    
    # Error handling
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    dead_letter_queue: Optional[str] = None


@dataclass
class StreamRecord:
    """Individual stream record."""
    
    id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    partition: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class StreamBatch:
    """Batch of stream records."""
    
    batch_id: str
    records: List[StreamRecord]
    batch_timestamp: datetime
    size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataset(self) -> Dataset:
        """Convert batch to Dataset for anomaly detection."""
        # Extract data from records
        data_list = []
        for record in self.records:
            # Flatten nested data if needed
            flattened = self._flatten_data(record.data)
            data_list.append(flattened)
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        
        # Create dataset
        dataset = Dataset(
            id=self.batch_id,
            name=f"stream_batch_{self.batch_id}",
            description="Stream batch data",
            data=df,
            metadata={
                "batch_timestamp": self.batch_timestamp.isoformat(),
                "batch_size": self.size,
                "source": "stream_processor",
                **self.metadata
            }
        )
        
        return dataset
    
    def _flatten_data(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary data."""
        flattened = {}
        
        for key, value in data.items():
            new_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_data(value, f"{new_key}_"))
            elif isinstance(value, list):
                # Convert lists to string representation for now
                flattened[new_key] = str(value)
            else:
                flattened[new_key] = value
        
        return flattened


class StreamProcessor:
    """Real-time streaming anomaly detection processor."""
    
    def __init__(self, config: StreamConfig):
        """Initialize stream processor."""
        self.config = config
        self.detection_service = get_detection_service()
        
        # Processing state
        self.is_running = False
        self.current_batch = []
        self.batch_count = 0
        self.processed_records = 0
        self.error_count = 0
        
        # Queues and buffers
        self.input_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.output_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        
        # Source and sink connectors
        self.source_connector = None
        self.sink_connector = None
        
        # Background tasks
        self.ingestion_task: Optional[asyncio.Task] = None
        self.processing_task: Optional[asyncio.Task] = None
        self.output_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.start_time = time.time()
        self.last_batch_time = None
        self.processing_times = []
        
        logger.info(f"Stream processor initialized with source: {config.source_type.value}")
    
    async def start(self) -> None:
        """Start stream processing."""
        if self.is_running:
            logger.warning("Stream processor is already running")
            return
        
        try:
            self.is_running = True
            
            # Initialize connectors
            await self._initialize_source_connector()
            await self._initialize_sink_connector()
            
            # Start background tasks
            self.ingestion_task = asyncio.create_task(self._ingestion_loop())
            self.processing_task = asyncio.create_task(self._processing_loop())
            self.output_task = asyncio.create_task(self._output_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Stream processor started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start stream processor: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop stream processing."""
        if not self.is_running:
            return
        
        logger.info("Stopping stream processor...")
        self.is_running = False
        
        # Cancel background tasks
        tasks = [self.ingestion_task, self.processing_task, self.output_task, self.monitoring_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process remaining batches
        await self._process_remaining_data()
        
        # Close connectors
        if self.source_connector:
            await self._close_source_connector()
        if self.sink_connector:
            await self._close_sink_connector()
        
        logger.info("Stream processor stopped")
    
    async def _initialize_source_connector(self) -> None:
        """Initialize source connector based on configuration."""
        if self.config.source_type == StreamSource.KAFKA:
            if not KAFKA_AVAILABLE:
                raise ImportError("Kafka is required for Kafka source")
            self.source_connector = await self._create_kafka_consumer()
        
        elif self.config.source_type == StreamSource.REDIS:
            if not REDIS_AVAILABLE:
                raise ImportError("Redis is required for Redis source")
            self.source_connector = await self._create_redis_consumer()
        
        elif self.config.source_type == StreamSource.MEMORY:
            # In-memory queue for testing
            self.source_connector = asyncio.Queue()
        
        else:
            raise ValueError(f"Unsupported source type: {self.config.source_type}")
    
    async def _create_kafka_consumer(self):
        """Create Kafka consumer."""
        kafka_config = self.config.source_config
        
        consumer = KafkaConsumer(
            kafka_config.get('topic', 'anomaly-detection-input'),
            bootstrap_servers=kafka_config.get('bootstrap_servers', ['localhost:9092']),
            group_id=kafka_config.get('group_id', 'pynomaly-stream-processor'),
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=kafka_config.get('enable_auto_commit', False),
            max_poll_records=self.config.batch_size,
            consumer_timeout_ms=int(self.config.batch_timeout_seconds * 1000)
        )
        
        return consumer
    
    async def _create_redis_consumer(self):
        """Create Redis consumer."""
        redis_config = self.config.source_config
        
        redis_client = Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_config.get('password'),
            decode_responses=True
        )
        
        return redis_client
    
    async def _initialize_sink_connector(self) -> None:
        """Initialize sink connector for output."""
        if self.config.output_topic:
            if self.config.source_type == StreamSource.KAFKA:
                kafka_config = self.config.source_config
                self.sink_connector = KafkaProducer(
                    bootstrap_servers=kafka_config.get('bootstrap_servers', ['localhost:9092']),
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
    
    @trace_operation("stream_ingestion")
    async def _ingestion_loop(self) -> None:
        """Main ingestion loop."""
        while self.is_running:
            try:
                records = await self._fetch_records()
                
                for record in records:
                    if not self.is_running:
                        break
                    
                    # Apply backpressure if queue is full
                    if self.config.enable_backpressure and self.input_queue.full():
                        logger.warning("Input queue full, applying backpressure")
                        await asyncio.sleep(0.1)
                        continue
                    
                    await self.input_queue.put(record)
                
            except Exception as e:
                logger.error(f"Error in ingestion loop: {e}")
                await asyncio.sleep(self.config.retry_delay_seconds)
    
    async def _fetch_records(self) -> List[StreamRecord]:
        """Fetch records from source."""
        records = []
        
        if self.config.source_type == StreamSource.KAFKA:
            try:
                message_pack = self.source_connector.poll(
                    timeout_ms=int(self.config.batch_timeout_seconds * 1000),
                    max_records=self.config.batch_size
                )
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        record = StreamRecord(
                            id=f"{message.topic}_{message.partition}_{message.offset}",
                            timestamp=datetime.fromtimestamp(message.timestamp / 1000),
                            data=message.value,
                            source=message.topic,
                            partition=message.partition,
                            offset=message.offset
                        )
                        records.append(record)
                        
            except Exception as e:
                logger.error(f"Error fetching from Kafka: {e}")
        
        elif self.config.source_type == StreamSource.REDIS:
            try:
                # Use Redis Streams for consumption
                stream_name = self.config.source_config.get('stream', 'anomaly-stream')
                consumer_group = self.config.source_config.get('consumer_group', 'processors')
                consumer_name = self.config.source_config.get('consumer_name', 'processor-1')
                
                messages = await self.source_connector.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {stream_name: '>'},
                    count=self.config.batch_size,
                    block=int(self.config.batch_timeout_seconds * 1000)
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        record = StreamRecord(
                            id=message_id,
                            timestamp=datetime.now(),
                            data=fields,
                            source=stream
                        )
                        records.append(record)
                        
            except Exception as e:
                logger.error(f"Error fetching from Redis: {e}")
        
        elif self.config.source_type == StreamSource.MEMORY:
            # For testing - generate synthetic data
            if not hasattr(self, '_synthetic_counter'):
                self._synthetic_counter = 0
            
            for i in range(min(self.config.batch_size, 10)):
                self._synthetic_counter += 1
                record = StreamRecord(
                    id=f"synthetic_{self._synthetic_counter}",
                    timestamp=datetime.now(),
                    data={
                        "feature_1": np.random.normal(0, 1),
                        "feature_2": np.random.normal(0, 1),
                        "feature_3": np.random.normal(0, 1),
                        "is_anomaly": np.random.random() < 0.1  # 10% anomaly rate
                    },
                    source="synthetic"
                )
                records.append(record)
                
            await asyncio.sleep(0.1)  # Simulate streaming delay
        
        return records
    
    @trace_operation("stream_processing")
    async def _processing_loop(self) -> None:
        """Main processing loop."""
        while self.is_running:
            try:
                # Collect batch
                batch = await self._collect_batch()
                
                if not batch.records:
                    continue
                
                # Process batch
                start_time = time.time()
                result = await self._process_batch(batch)
                processing_time = time.time() - start_time
                
                # Track metrics
                self.processing_times.append(processing_time)
                self.last_batch_time = time.time()
                self.processed_records += len(batch.records)
                
                # Send to output queue
                await self.output_queue.put(result)
                
                logger.debug(f"Processed batch {batch.batch_id} with {len(batch.records)} records in {processing_time:.2f}s")
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(self.config.retry_delay_seconds)
    
    async def _collect_batch(self) -> StreamBatch:
        """Collect records into a batch."""
        records = []
        batch_start = time.time()
        
        # Collect records until batch size or timeout
        while (len(records) < self.config.batch_size and 
               time.time() - batch_start < self.config.batch_timeout_seconds):
            
            try:
                record = await asyncio.wait_for(
                    self.input_queue.get(),
                    timeout=self.config.batch_timeout_seconds
                )
                records.append(record)
                
            except asyncio.TimeoutError:
                break
        
        self.batch_count += 1
        batch = StreamBatch(
            batch_id=f"batch_{self.batch_count}_{int(time.time())}",
            records=records,
            batch_timestamp=datetime.now(),
            size=len(records)
        )
        
        return batch
    
    async def _process_batch(self, batch: StreamBatch) -> DetectionResult:
        """Process a batch of records for anomaly detection."""
        try:
            # Convert batch to dataset
            dataset = batch.to_dataset()
            
            # Run anomaly detection
            result = await self.detection_service.detect_anomalies(
                dataset=dataset,
                algorithm=self.config.detection_algorithm,
                config=self.config.detection_config
            )
            
            # Enhance result with stream metadata
            result.metadata.update({
                "batch_id": batch.batch_id,
                "batch_timestamp": batch.batch_timestamp.isoformat(),
                "stream_source": self.config.source_type.value,
                "processing_mode": "streaming"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {e}")
            raise
    
    async def _output_loop(self) -> None:
        """Output loop for sending results."""
        while self.is_running:
            try:
                result = await self.output_queue.get()
                await self._send_result(result)
                
            except Exception as e:
                logger.error(f"Error in output loop: {e}")
                await asyncio.sleep(self.config.retry_delay_seconds)
    
    async def _send_result(self, result: DetectionResult) -> None:
        """Send detection result to configured output."""
        if self.sink_connector and self.config.output_topic:
            try:
                # Convert result to output format
                output_data = self._format_result(result)
                
                if self.config.source_type == StreamSource.KAFKA:
                    future = self.sink_connector.send(self.config.output_topic, output_data)
                    record_metadata = future.get(timeout=10)
                    logger.debug(f"Sent result to Kafka topic {self.config.output_topic}")
                
            except Exception as e:
                logger.error(f"Error sending result: {e}")
    
    def _format_result(self, result: DetectionResult) -> Dict[str, Any]:
        """Format detection result for output."""
        return {
            "dataset_id": result.dataset_id,
            "algorithm": result.algorithm,
            "timestamp": datetime.now().isoformat(),
            "total_samples": result.total_samples,
            "anomaly_count": result.anomaly_count,
            "contamination_rate": result.contamination_rate.value,
            "execution_time": result.execution_time,
            "anomalies": [
                {
                    "index": anomaly.index,
                    "score": anomaly.score.value,
                    "confidence_lower": anomaly.confidence.lower,
                    "confidence_upper": anomaly.confidence.upper,
                    "explanation": anomaly.explanation
                }
                for anomaly in result.anomalies[:100]  # Limit for output size
            ],
            "metadata": result.metadata
        }
    
    async def _monitoring_loop(self) -> None:
        """Monitoring loop for health checks and metrics."""
        while self.is_running:
            try:
                await self._collect_metrics()
                await self._health_check()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_metrics(self) -> None:
        """Collect processing metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate rates
        processing_rate = self.processed_records / uptime if uptime > 0 else 0
        error_rate = self.error_count / max(1, self.batch_count)
        
        # Calculate latency metrics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        p95_processing_time = np.percentile(self.processing_times, 95) if self.processing_times else 0
        
        metrics = {
            "uptime_seconds": uptime,
            "processed_records": self.processed_records,
            "batch_count": self.batch_count,
            "error_count": self.error_count,
            "processing_rate_per_second": processing_rate,
            "error_rate": error_rate,
            "avg_processing_time": avg_processing_time,
            "p95_processing_time": p95_processing_time,
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize()
        }
        
        logger.debug(f"Stream processor metrics: {metrics}")
    
    async def _health_check(self) -> None:
        """Perform health checks."""
        current_time = time.time()
        
        # Check if processing is stalled
        if (self.last_batch_time and 
            current_time - self.last_batch_time > self.config.batch_timeout_seconds * 2):
            logger.warning("Stream processing appears to be stalled")
        
        # Check queue sizes
        if self.input_queue.qsize() > self.config.max_queue_size * 0.8:
            logger.warning("Input queue is near capacity")
        
        if self.output_queue.qsize() > self.config.max_queue_size * 0.8:
            logger.warning("Output queue is near capacity")
    
    async def _process_remaining_data(self) -> None:
        """Process any remaining data in queues during shutdown."""
        logger.info("Processing remaining data...")
        
        # Process remaining input records
        remaining_records = []
        while not self.input_queue.empty():
            try:
                record = self.input_queue.get_nowait()
                remaining_records.append(record)
            except asyncio.QueueEmpty:
                break
        
        if remaining_records:
            final_batch = StreamBatch(
                batch_id=f"final_batch_{int(time.time())}",
                records=remaining_records,
                batch_timestamp=datetime.now(),
                size=len(remaining_records)
            )
            
            try:
                result = await self._process_batch(final_batch)
                await self._send_result(result)
                logger.info(f"Processed final batch with {len(remaining_records)} records")
            except Exception as e:
                logger.error(f"Error processing final batch: {e}")
    
    async def _close_source_connector(self) -> None:
        """Close source connector."""
        if self.config.source_type == StreamSource.KAFKA:
            if self.source_connector:
                self.source_connector.close()
        
        elif self.config.source_type == StreamSource.REDIS:
            if self.source_connector:
                await self.source_connector.close()
    
    async def _close_sink_connector(self) -> None:
        """Close sink connector."""
        if self.sink_connector:
            if self.config.source_type == StreamSource.KAFKA:
                self.sink_connector.close()
    
    async def add_test_record(self, data: Dict[str, Any]) -> None:
        """Add a test record to the stream (for testing)."""
        if self.config.source_type == StreamSource.MEMORY:
            record = StreamRecord(
                id=f"test_{int(time.time())}",
                timestamp=datetime.now(),
                data=data,
                source="test"
            )
            await self.input_queue.put(record)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processor status."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "processed_records": self.processed_records,
            "batch_count": self.batch_count,
            "error_count": self.error_count,
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "last_batch_time": self.last_batch_time,
            "config": {
                "source_type": self.config.source_type.value,
                "batch_size": self.config.batch_size,
                "batch_timeout": self.config.batch_timeout_seconds,
                "detection_algorithm": self.config.detection_algorithm.value
            }
        }


# Factory function
def create_stream_processor(config: StreamConfig) -> StreamProcessor:
    """Create a stream processor with the given configuration."""
    return StreamProcessor(config)