"""Enhanced real-time streaming pipeline for anomaly detection."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Union
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Optional imports for streaming
try:
    import kafka
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    kafka = None
    KafkaConsumer = None
    KafkaProducer = None
    KAFKA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.exceptions import StreamingError
from pynomaly.shared.protocols import DetectorProtocol

logger = logging.getLogger(__name__)


class StreamingDataPoint(BaseModel):
    """Individual data point in streaming pipeline."""
    
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_id: str = Field(default="unknown")
    batch_id: Optional[str] = None


class StreamingBatch(BaseModel):
    """Batch of streaming data points."""
    
    batch_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    data_points: List[StreamingDataPoint] = Field(default_factory=list)
    size: int = Field(default=0)
    
    def add_data_point(self, data_point: StreamingDataPoint) -> None:
        """Add data point to batch."""
        data_point.batch_id = self.batch_id
        self.data_points.append(data_point)
        self.size = len(self.data_points)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert batch to pandas DataFrame."""
        records = []
        for dp in self.data_points:
            record = {
                "timestamp": dp.timestamp,
                "source_id": dp.source_id,
                "batch_id": dp.batch_id,
                **dp.data
            }
            records.append(record)
        return pd.DataFrame(records)


class StreamingConfig(BaseModel):
    """Configuration for streaming pipeline."""
    
    batch_size: int = Field(default=100, ge=1, le=10000)
    batch_timeout_ms: int = Field(default=1000, ge=100, le=60000)
    max_latency_ms: int = Field(default=100, ge=10, le=5000)
    buffer_size: int = Field(default=10000, ge=100, le=100000)
    parallelism: int = Field(default=1, ge=1, le=32)
    enable_backpressure: bool = Field(default=True)
    checkpoint_interval_ms: int = Field(default=5000, ge=1000, le=60000)
    
    # Anomaly detection specific
    contamination_threshold: float = Field(default=0.1, ge=0.01, le=0.5)
    alert_threshold: float = Field(default=0.8, ge=0.1, le=1.0)
    rolling_window_size: int = Field(default=1000, ge=100, le=100000)
    
    # Performance optimization
    enable_gpu_acceleration: bool = Field(default=False)
    use_adaptive_batching: bool = Field(default=True)
    enable_compression: bool = Field(default=True)


class StreamingMetrics(BaseModel):
    """Metrics for streaming pipeline."""
    
    total_processed: int = Field(default=0)
    total_anomalies: int = Field(default=0)
    total_errors: int = Field(default=0)
    avg_latency_ms: float = Field(default=0.0)
    max_latency_ms: float = Field(default=0.0)
    throughput_per_second: float = Field(default=0.0)
    anomaly_rate: float = Field(default=0.0)
    error_rate: float = Field(default=0.0)
    
    # Real-time metrics
    last_processed_timestamp: Optional[datetime] = None
    processing_lag_ms: float = Field(default=0.0)
    backpressure_events: int = Field(default=0)
    
    def update_latency(self, latency_ms: float) -> None:
        """Update latency metrics."""
        self.avg_latency_ms = (self.avg_latency_ms * 0.9) + (latency_ms * 0.1)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
    
    def update_throughput(self, processed_count: int, time_window_s: float) -> None:
        """Update throughput metrics."""
        self.throughput_per_second = processed_count / time_window_s
    
    def update_rates(self) -> None:
        """Update anomaly and error rates."""
        if self.total_processed > 0:
            self.anomaly_rate = self.total_anomalies / self.total_processed
            self.error_rate = self.total_errors / self.total_processed


class StreamingAlert(BaseModel):
    """Alert for streaming anomalies."""
    
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    anomaly_score: float = Field(ge=0.0, le=1.0)
    severity: str = Field(default="medium")  # low, medium, high, critical
    data_point: StreamingDataPoint
    context: Dict[str, Any] = Field(default_factory=dict)
    resolved: bool = Field(default=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "anomaly_score": self.anomaly_score,
            "severity": self.severity,
            "data": self.data_point.data,
            "context": self.context,
            "resolved": self.resolved
        }


class StreamingSourceProtocol(Protocol):
    """Protocol for streaming data sources."""
    
    async def consume(self) -> AsyncGenerator[StreamingDataPoint, None]:
        """Consume streaming data points."""
        ...
    
    async def close(self) -> None:
        """Close the streaming source."""
        ...


class StreamingSinkProtocol(Protocol):
    """Protocol for streaming data sinks."""
    
    async def emit(self, result: DetectionResult) -> None:
        """Emit detection result."""
        ...
    
    async def emit_alert(self, alert: StreamingAlert) -> None:
        """Emit anomaly alert."""
        ...
    
    async def close(self) -> None:
        """Close the streaming sink."""
        ...


class KafkaStreamingSource:
    """Kafka streaming data source."""
    
    def __init__(self, topic: str, bootstrap_servers: str, **kwargs):
        """Initialize Kafka source."""
        if not KAFKA_AVAILABLE:
            raise StreamingError("Kafka not available. Install with: pip install kafka-python")
        
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.consumer_config = {
            'bootstrap_servers': bootstrap_servers,
            'auto_offset_reset': 'latest',
            'enable_auto_commit': True,
            'group_id': f'pynomaly-{uuid4().hex[:8]}',
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            **kwargs
        }
        self.consumer = None
    
    async def consume(self) -> AsyncGenerator[StreamingDataPoint, None]:
        """Consume data from Kafka."""
        if not self.consumer:
            self.consumer = KafkaConsumer(self.topic, **self.consumer_config)
        
        try:
            for message in self.consumer:
                try:
                    data_point = StreamingDataPoint(
                        timestamp=datetime.fromtimestamp(message.timestamp / 1000),
                        data=message.value,
                        metadata={
                            'partition': message.partition,
                            'offset': message.offset,
                            'topic': message.topic
                        },
                        source_id=f"kafka-{message.partition}-{message.offset}"
                    )
                    yield data_point
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
                    continue
        finally:
            if self.consumer:
                self.consumer.close()
    
    async def close(self) -> None:
        """Close Kafka consumer."""
        if self.consumer:
            self.consumer.close()


class KafkaStreamingSink:
    """Kafka streaming data sink."""
    
    def __init__(self, topic: str, bootstrap_servers: str, **kwargs):
        """Initialize Kafka sink."""
        if not KAFKA_AVAILABLE:
            raise StreamingError("Kafka not available. Install with: pip install kafka-python")
        
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.producer_config = {
            'bootstrap_servers': bootstrap_servers,
            'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
            **kwargs
        }
        self.producer = None
    
    async def emit(self, result: DetectionResult) -> None:
        """Emit detection result to Kafka."""
        if not self.producer:
            self.producer = KafkaProducer(**self.producer_config)
        
        try:
            message = {
                'timestamp': datetime.now().isoformat(),
                'detector_id': str(result.detector_id),
                'anomaly_count': len(result.anomalies),
                'scores': [score.value for score in result.scores],
                'labels': result.labels,
                'metadata': result.metadata or {}
            }
            
            self.producer.send(self.topic, value=message)
            self.producer.flush()
            
        except Exception as e:
            logger.error(f"Error emitting to Kafka: {e}")
            raise StreamingError(f"Failed to emit to Kafka: {e}")
    
    async def emit_alert(self, alert: StreamingAlert) -> None:
        """Emit alert to Kafka."""
        if not self.producer:
            self.producer = KafkaProducer(**self.producer_config)
        
        try:
            self.producer.send(f"{self.topic}-alerts", value=alert.to_dict())
            self.producer.flush()
        except Exception as e:
            logger.error(f"Error emitting alert to Kafka: {e}")
    
    async def close(self) -> None:
        """Close Kafka producer."""
        if self.producer:
            self.producer.close()


class RedisStreamingSource:
    """Redis streaming data source."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, stream_key: str = 'anomaly-stream', **kwargs):
        """Initialize Redis source."""
        if not REDIS_AVAILABLE:
            raise StreamingError("Redis not available. Install with: pip install redis")
        
        self.host = host
        self.port = port
        self.stream_key = stream_key
        self.redis_client = None
        self.consumer_group = f'pynomaly-{uuid4().hex[:8]}'
        self.consumer_name = f'consumer-{uuid4().hex[:8]}'
    
    async def consume(self) -> AsyncGenerator[StreamingDataPoint, None]:
        """Consume data from Redis stream."""
        if not self.redis_client:
            self.redis_client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        
        try:
            # Create consumer group
            try:
                self.redis_client.xgroup_create(self.stream_key, self.consumer_group, id='0', mkstream=True)
            except redis.exceptions.ResponseError:
                pass  # Group already exists
            
            while True:
                try:
                    messages = self.redis_client.xreadgroup(
                        self.consumer_group,
                        self.consumer_name,
                        {self.stream_key: '>'},
                        count=1,
                        block=1000
                    )
                    
                    for stream, msgs in messages:
                        for msg_id, fields in msgs:
                            try:
                                data_point = StreamingDataPoint(
                                    timestamp=datetime.now(),
                                    data=fields,
                                    metadata={'stream_id': msg_id},
                                    source_id=f"redis-{msg_id}"
                                )
                                yield data_point
                                
                                # Acknowledge message
                                self.redis_client.xack(self.stream_key, self.consumer_group, msg_id)
                                
                            except Exception as e:
                                logger.error(f"Error processing Redis message: {e}")
                                continue
                
                except Exception as e:
                    logger.error(f"Error reading from Redis: {e}")
                    await asyncio.sleep(1)
                    continue
                    
        finally:
            if self.redis_client:
                self.redis_client.close()
    
    async def close(self) -> None:
        """Close Redis client."""
        if self.redis_client:
            self.redis_client.close()


class RealTimeStreamingPipeline:
    """Enhanced real-time streaming pipeline for anomaly detection."""
    
    def __init__(
        self,
        detector: DetectorProtocol,
        source: StreamingSourceProtocol,
        sink: StreamingSinkProtocol,
        config: StreamingConfig,
        name: str = "streaming-pipeline"
    ):
        """Initialize streaming pipeline."""
        self.detector = detector
        self.source = source
        self.sink = sink
        self.config = config
        self.name = name
        
        # Pipeline state
        self.is_running = False
        self.metrics = StreamingMetrics()
        self.buffer = deque(maxlen=config.buffer_size)
        self.alerts = deque(maxlen=1000)  # Keep last 1000 alerts
        
        # Performance tracking
        self.latency_samples = deque(maxlen=100)
        self.throughput_window = deque(maxlen=60)  # 1 minute window
        
        # Adaptive batching
        self.adaptive_batch_size = config.batch_size
        self.last_batch_time = time.time()
        
        # Rolling window for trend analysis
        self.rolling_scores = deque(maxlen=config.rolling_window_size)
        
        logger.info(f"Initialized streaming pipeline: {name}")
    
    async def start(self) -> None:
        """Start the streaming pipeline."""
        if self.is_running:
            raise StreamingError("Pipeline is already running")
        
        self.is_running = True
        logger.info(f"Starting streaming pipeline: {self.name}")
        
        try:
            # Start main processing loop
            await self._process_stream()
        except Exception as e:
            logger.error(f"Error in streaming pipeline: {e}")
            raise StreamingError(f"Pipeline failed: {e}")
        finally:
            self.is_running = False
    
    async def stop(self) -> None:
        """Stop the streaming pipeline."""
        logger.info(f"Stopping streaming pipeline: {self.name}")
        self.is_running = False
        
        # Close resources
        await self.source.close()
        await self.sink.close()
    
    async def _process_stream(self) -> None:
        """Main stream processing loop."""
        batch = StreamingBatch()
        last_batch_time = time.time()
        
        async for data_point in self.source.consume():
            if not self.is_running:
                break
            
            start_time = time.time()
            
            try:
                # Add to current batch
                batch.add_data_point(data_point)
                
                # Check if batch should be processed
                should_process = (
                    batch.size >= self.adaptive_batch_size or
                    (time.time() - last_batch_time) * 1000 >= self.config.batch_timeout_ms
                )
                
                if should_process:
                    # Process batch
                    await self._process_batch(batch)
                    
                    # Reset batch
                    batch = StreamingBatch()
                    last_batch_time = time.time()
                    
                    # Update adaptive batch size
                    self._update_adaptive_batch_size()
                
                # Update metrics
                processing_time = (time.time() - start_time) * 1000
                self.metrics.update_latency(processing_time)
                
                # Check for backpressure
                if len(self.buffer) > self.config.buffer_size * 0.8:
                    self.metrics.backpressure_events += 1
                    await asyncio.sleep(0.001)  # Small delay to relieve pressure
                
            except Exception as e:
                logger.error(f"Error processing data point: {e}")
                self.metrics.total_errors += 1
                continue
    
    async def _process_batch(self, batch: StreamingBatch) -> None:
        """Process a batch of data points."""
        if batch.size == 0:
            return
        
        try:
            # Convert to dataset
            df = batch.to_dataframe()
            dataset = Dataset(
                name=f"streaming-batch-{batch.batch_id}",
                data=df,
                metadata={
                    'batch_id': batch.batch_id,
                    'batch_size': batch.size,
                    'timestamp': batch.timestamp.isoformat()
                }
            )
            
            # Detect anomalies
            result = self.detector.detect(dataset)
            
            # Update metrics
            self.metrics.total_processed += batch.size
            anomaly_count = sum(result.labels)
            self.metrics.total_anomalies += anomaly_count
            
            # Update rolling scores
            for score in result.scores:
                self.rolling_scores.append(score.value)
            
            # Generate alerts for high-score anomalies
            await self._generate_alerts(batch, result)
            
            # Emit results
            await self.sink.emit(result)
            
            # Update throughput
            self.throughput_window.append(batch.size)
            
            logger.debug(f"Processed batch {batch.batch_id}: {batch.size} points, {anomaly_count} anomalies")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {e}")
            self.metrics.total_errors += 1
            raise
    
    async def _generate_alerts(self, batch: StreamingBatch, result: DetectionResult) -> None:
        """Generate alerts for high-score anomalies."""
        for i, (score, label) in enumerate(zip(result.scores, result.labels)):
            if score.value >= self.config.alert_threshold:
                severity = self._calculate_severity(score.value)
                
                alert = StreamingAlert(
                    anomaly_score=score.value,
                    severity=severity,
                    data_point=batch.data_points[i],
                    context={
                        'batch_id': batch.batch_id,
                        'batch_index': i,
                        'rolling_avg': np.mean(list(self.rolling_scores)[-100:]) if self.rolling_scores else 0.0,
                        'trend': self._calculate_trend()
                    }
                )
                
                self.alerts.append(alert)
                await self.sink.emit_alert(alert)
                
                logger.warning(f"Anomaly alert generated: {alert.alert_id} (score: {score.value:.3f})")
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate alert severity based on score."""
        if score >= 0.9:
            return "critical"
        elif score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_trend(self) -> str:
        """Calculate trend from rolling scores."""
        if len(self.rolling_scores) < 50:
            return "insufficient_data"
        
        recent_scores = list(self.rolling_scores)[-50:]
        older_scores = list(self.rolling_scores)[-100:-50] if len(self.rolling_scores) >= 100 else []
        
        if not older_scores:
            return "no_trend"
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _update_adaptive_batch_size(self) -> None:
        """Update adaptive batch size based on performance."""
        if not self.config.use_adaptive_batching:
            return
        
        current_time = time.time()
        processing_time = current_time - self.last_batch_time
        
        # Adjust batch size based on processing time
        if processing_time < 0.05:  # Very fast processing
            self.adaptive_batch_size = min(self.adaptive_batch_size * 1.2, self.config.batch_size * 2)
        elif processing_time > 0.2:  # Slow processing
            self.adaptive_batch_size = max(self.adaptive_batch_size * 0.8, self.config.batch_size // 2)
        
        self.last_batch_time = current_time
    
    def get_metrics(self) -> StreamingMetrics:
        """Get current pipeline metrics."""
        # Update rates
        self.metrics.update_rates()
        
        # Update throughput
        if self.throughput_window:
            self.metrics.update_throughput(
                sum(self.throughput_window),
                len(self.throughput_window)
            )
        
        # Update processing lag
        if self.metrics.last_processed_timestamp:
            lag = (datetime.now() - self.metrics.last_processed_timestamp).total_seconds() * 1000
            self.metrics.processing_lag_ms = lag
        
        return self.metrics
    
    def get_recent_alerts(self, limit: int = 10) -> List[StreamingAlert]:
        """Get recent alerts."""
        return list(self.alerts)[-limit:]


def create_kafka_pipeline(
    detector: DetectorProtocol,
    input_topic: str,
    output_topic: str,
    bootstrap_servers: str,
    config: StreamingConfig,
    name: str = "kafka-pipeline"
) -> RealTimeStreamingPipeline:
    """Create a Kafka-based streaming pipeline."""
    source = KafkaStreamingSource(input_topic, bootstrap_servers)
    sink = KafkaStreamingSink(output_topic, bootstrap_servers)
    
    return RealTimeStreamingPipeline(
        detector=detector,
        source=source,
        sink=sink,
        config=config,
        name=name
    )


def create_redis_pipeline(
    detector: DetectorProtocol,
    stream_key: str,
    redis_host: str = 'localhost',
    redis_port: int = 6379,
    config: StreamingConfig = None,
    name: str = "redis-pipeline"
) -> RealTimeStreamingPipeline:
    """Create a Redis-based streaming pipeline."""
    if config is None:
        config = StreamingConfig()
    
    source = RedisStreamingSource(redis_host, redis_port, stream_key)
    
    # For Redis, we'll use a simple console sink for now
    class ConsoleSink:
        async def emit(self, result: DetectionResult) -> None:
            print(f"Detection result: {len(result.anomalies)} anomalies detected")
        
        async def emit_alert(self, alert: StreamingAlert) -> None:
            print(f"ALERT: {alert.severity} - Score: {alert.anomaly_score:.3f}")
        
        async def close(self) -> None:
            pass
    
    sink = ConsoleSink()
    
    return RealTimeStreamingPipeline(
        detector=detector,
        source=source,
        sink=sink,
        config=config,
        name=name
    )
