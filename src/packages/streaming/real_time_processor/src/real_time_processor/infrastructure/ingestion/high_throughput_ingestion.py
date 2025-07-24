#!/usr/bin/env python3
"""
High-Throughput Data Ingestion System
Scalable data ingestion with multiple sources, validation, and routing capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import aiohttp
import asyncpg
from kafka import KafkaProducer
import redis.asyncio as redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import pydantic
from pydantic import BaseModel, validator


# Metrics
INGESTION_RATE = Counter('ingestion_messages_total', 'Total ingested messages', ['source', 'status'])
INGESTION_TIME = Histogram('ingestion_processing_seconds', 'Time spent processing ingestion', ['source'])
INGESTION_QUEUE_SIZE = Gauge('ingestion_queue_size', 'Size of ingestion queue', ['source'])
INGESTION_THROUGHPUT = Gauge('ingestion_throughput_per_second', 'Ingestion throughput per second', ['source'])
VALIDATION_ERRORS = Counter('ingestion_validation_errors_total', 'Validation errors', ['source', 'error_type'])


class IngestionStatus(Enum):
    """Ingestion status types."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    FILTERED = "filtered"


class DataSourceType(Enum):
    """Data source types."""
    HTTP_API = "http_api"
    KAFKA = "kafka"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    WEBHOOK = "webhook"
    STREAM = "stream"
    BATCH = "batch"


class ValidationRule(BaseModel):
    """Data validation rule."""
    field: str
    rule_type: str  # required, type, range, pattern, custom
    parameters: Dict[str, Any] = {}
    error_message: Optional[str] = None


class DataSchema(BaseModel):
    """Data schema definition."""
    name: str
    version: str
    fields: Dict[str, str]  # field_name -> type
    validation_rules: List[ValidationRule] = []
    transformation_rules: List[Dict[str, Any]] = []


class IngestionMessage(BaseModel):
    """Standardized ingestion message."""
    id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    source_type: DataSourceType
    timestamp: datetime = pydantic.Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    schema_name: Optional[str] = None
    priority: int = 5  # 1-10, higher = more priority
    routing_key: Optional[str] = None
    retry_count: int = 0
    status: IngestionStatus = IngestionStatus.PENDING
    
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v


@dataclass
class IngestionConfig:
    """Ingestion system configuration."""
    name: str
    max_workers: int = mp.cpu_count()
    max_queue_size: int = 100000
    batch_size: int = 1000
    flush_interval_seconds: int = 5
    enable_validation: bool = True
    enable_deduplication: bool = True
    deduplication_ttl_seconds: int = 3600
    max_retries: int = 3
    retry_delay_seconds: int = 5
    enable_compression: bool = True
    buffer_size: int = 10000


class DataIngestionEngine:
    """High-throughput data ingestion engine."""
    
    def __init__(self, config: IngestionConfig, kafka_config: Dict[str, Any], 
                 redis_url: str = "redis://localhost:6379/1", 
                 postgres_url: Optional[str] = None):
        self.config = config
        self.kafka_config = kafka_config
        self.redis_url = redis_url
        self.postgres_url = postgres_url
        
        # Initialize components
        self.redis_client = None
        self.postgres_pool = None
        self.kafka_producer = None
        
        # Processing queues
        self.ingestion_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.validation_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.output_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        
        # Processing state
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.schemas: Dict[str, DataSchema] = {}
        self.source_handlers: Dict[str, callable] = {}
        self.routing_rules: Dict[str, str] = {}  # routing_key -> destination_topic
        
        # Metrics and monitoring
        self.logger = logging.getLogger(f"ingestion.{config.name}")
        self.throughput_tracker = {}
        self.last_throughput_update = time.time()
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
    
    async def initialize(self) -> None:
        """Initialize the ingestion engine."""
        try:
            self.logger.info("Initializing data ingestion engine...")
            
            # Initialize Redis
            self.redis_client = redis.Redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize PostgreSQL if configured
            if self.postgres_url:
                self.postgres_pool = await asyncpg.create_pool(self.postgres_url, min_size=5, max_size=20)
            
            # Initialize Kafka producer
            self.kafka_producer = KafkaProducer(
                **self.kafka_config,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                batch_size=16384,
                linger_ms=10,
                compression_type='gzip' if self.config.enable_compression else None,
                retries=3,
                acks='all'
            )
            
            self.logger.info("Data ingestion engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ingestion engine: {e}")
            raise
    
    def register_schema(self, schema: DataSchema) -> None:
        """Register a data schema."""
        self.schemas[schema.name] = schema
        self.logger.info(f"Registered schema: {schema.name} v{schema.version}")
    
    def register_source_handler(self, source: str, handler: callable) -> None:
        """Register a source-specific handler."""
        self.source_handlers[source] = handler
        self.logger.info(f"Registered source handler: {source}")
    
    def add_routing_rule(self, routing_key: str, destination_topic: str) -> None:
        """Add a routing rule."""
        self.routing_rules[routing_key] = destination_topic
        self.logger.info(f"Added routing rule: {routing_key} -> {destination_topic}")
    
    async def start(self) -> None:
        """Start the ingestion engine."""
        if self.is_running:
            self.logger.warning("Ingestion engine is already running")
            return
        
        self.is_running = True
        self.logger.info("Starting data ingestion engine...")
        
        # Start worker tasks
        worker_count = self.config.max_workers
        
        # Ingestion workers
        for i in range(worker_count // 3):
            task = asyncio.create_task(self._ingestion_worker(f"ingestion-{i}"))
            self.worker_tasks.append(task)
        
        # Validation workers
        for i in range(worker_count // 3):
            task = asyncio.create_task(self._validation_worker(f"validation-{i}"))
            self.worker_tasks.append(task)
        
        # Output workers
        for i in range(worker_count // 3):
            task = asyncio.create_task(self._output_worker(f"output-{i}"))
            self.worker_tasks.append(task)
        
        # Monitoring tasks
        self.worker_tasks.append(asyncio.create_task(self._monitoring_worker()))
        self.worker_tasks.append(asyncio.create_task(self._throughput_tracker()))
        
        self.logger.info(f"Started {len(self.worker_tasks)} worker tasks")
    
    async def ingest_message(self, message: Union[IngestionMessage, Dict[str, Any]]) -> str:
        """Ingest a single message."""
        if isinstance(message, dict):
            message = IngestionMessage(**message)
        
        try:
            # Add to ingestion queue
            await self.ingestion_queue.put(message)
            INGESTION_QUEUE_SIZE.labels(source=message.source).set(self.ingestion_queue.qsize())
            
            return message.id
            
        except asyncio.QueueFull:
            self.logger.error(f"Ingestion queue full, dropping message from {message.source}")
            INGESTION_RATE.labels(source=message.source, status="queue_full").inc()
            raise
    
    async def ingest_batch(self, messages: List[Union[IngestionMessage, Dict[str, Any]]]) -> List[str]:
        """Ingest a batch of messages."""
        message_ids = []
        
        for msg in messages:
            try:
                message_id = await self.ingest_message(msg)
                message_ids.append(message_id)
            except Exception as e:
                self.logger.error(f"Failed to ingest message: {e}")
        
        return message_ids
    
    async def _ingestion_worker(self, worker_id: str) -> None:
        """Ingestion worker that processes incoming messages."""
        self.logger.info(f"Started ingestion worker: {worker_id}")
        
        while self.is_running:
            try:
                # Get message from queue
                message = await asyncio.wait_for(self.ingestion_queue.get(), timeout=1.0)
                
                start_time = time.time()
                message.status = IngestionStatus.PROCESSING
                
                # Apply source-specific processing
                if message.source in self.source_handlers:
                    try:
                        message = await self._apply_source_handler(message)
                    except Exception as e:
                        self.logger.error(f"Source handler failed for {message.source}: {e}")
                        message.status = IngestionStatus.FAILED
                        continue
                
                # Deduplication check
                if self.config.enable_deduplication:
                    if await self._is_duplicate(message):
                        message.status = IngestionStatus.FILTERED
                        INGESTION_RATE.labels(source=message.source, status="duplicate").inc()
                        continue
                
                # Add to validation queue
                await self.validation_queue.put(message)
                
                # Record metrics
                processing_time = time.time() - start_time
                INGESTION_TIME.labels(source=message.source).observe(processing_time)
                INGESTION_RATE.labels(source=message.source, status="processed").inc()
                
                # Update throughput tracker
                source = message.source
                if source not in self.throughput_tracker:
                    self.throughput_tracker[source] = []
                self.throughput_tracker[source].append(time.time())
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in ingestion worker {worker_id}: {e}")
    
    async def _validation_worker(self, worker_id: str) -> None:
        """Validation worker that validates message data."""
        self.logger.info(f"Started validation worker: {worker_id}")
        
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.validation_queue.get(), timeout=1.0)
                
                if self.config.enable_validation:
                    validation_result = await self._validate_message(message)
                    if not validation_result.is_valid:
                        message.status = IngestionStatus.FAILED
                        message.metadata['validation_errors'] = validation_result.errors
                        VALIDATION_ERRORS.labels(
                            source=message.source, 
                            error_type="validation_failed"
                        ).inc()
                        continue
                
                # Apply transformations if configured
                if message.schema_name and message.schema_name in self.schemas:
                    schema = self.schemas[message.schema_name]
                    message = await self._apply_transformations(message, schema)
                
                # Add to output queue
                await self.output_queue.put(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in validation worker {worker_id}: {e}")
    
    async def _output_worker(self, worker_id: str) -> None:
        """Output worker that sends messages to destinations."""
        self.logger.info(f"Started output worker: {worker_id}")
        
        message_batch = []
        last_flush = time.time()
        
        while self.is_running:
            try:
                # Collect messages for batching
                try:
                    message = await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
                    message_batch.append(message)
                except asyncio.TimeoutError:
                    pass
                
                # Flush batch if conditions are met
                should_flush = (
                    len(message_batch) >= self.config.batch_size or
                    (message_batch and time.time() - last_flush >= self.config.flush_interval_seconds)
                )
                
                if should_flush and message_batch:
                    await self._flush_message_batch(message_batch)
                    message_batch.clear()
                    last_flush = time.time()
                
            except Exception as e:
                self.logger.error(f"Error in output worker {worker_id}: {e}")
    
    async def _apply_source_handler(self, message: IngestionMessage) -> IngestionMessage:
        """Apply source-specific handler."""
        handler = self.source_handlers[message.source]
        
        # Run handler in thread pool for CPU-intensive operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, handler, message)
    
    async def _is_duplicate(self, message: IngestionMessage) -> bool:
        """Check if message is a duplicate."""
        try:
            # Create hash of message content
            content_hash = hashlib.sha256(
                json.dumps(message.data, sort_keys=True).encode()
            ).hexdigest()
            
            # Check in Redis
            key = f"dedup:{message.source}:{content_hash}"
            exists = await self.redis_client.exists(key)
            
            if not exists:
                # Store for deduplication
                await self.redis_client.setex(key, self.config.deduplication_ttl_seconds, "1")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Deduplication check failed: {e}")
            return False
    
    async def _validate_message(self, message: IngestionMessage) -> 'ValidationResult':
        """Validate message against schema."""
        if not message.schema_name or message.schema_name not in self.schemas:
            return ValidationResult(is_valid=True, errors=[])
        
        schema = self.schemas[message.schema_name]
        errors = []
        
        # Validate fields
        for field_name, field_type in schema.fields.items():
            if field_name not in message.data:
                errors.append(f"Missing required field: {field_name}")
                continue
            
            # Type validation
            value = message.data[field_name]
            if not self._validate_field_type(value, field_type):
                errors.append(f"Invalid type for field {field_name}: expected {field_type}")
        
        # Apply validation rules
        for rule in schema.validation_rules:
            rule_result = await self._apply_validation_rule(message.data, rule)
            if not rule_result.is_valid:
                errors.extend(rule_result.errors)
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type."""
        type_mapping = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, skip validation
    
    async def _apply_validation_rule(self, data: Dict[str, Any], rule: ValidationRule) -> 'ValidationResult':
        """Apply a single validation rule."""
        field_value = data.get(rule.field)
        errors = []
        
        if rule.rule_type == "required" and field_value is None:
            errors.append(rule.error_message or f"Field {rule.field} is required")
        
        elif rule.rule_type == "range" and field_value is not None:
            min_val = rule.parameters.get('min')
            max_val = rule.parameters.get('max')
            
            if min_val is not None and field_value < min_val:
                errors.append(f"Field {rule.field} must be >= {min_val}")
            if max_val is not None and field_value > max_val:
                errors.append(f"Field {rule.field} must be <= {max_val}")
        
        elif rule.rule_type == "pattern" and field_value is not None:
            import re
            pattern = rule.parameters.get('pattern')
            if pattern and not re.match(pattern, str(field_value)):
                errors.append(f"Field {rule.field} does not match required pattern")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    async def _apply_transformations(self, message: IngestionMessage, schema: DataSchema) -> IngestionMessage:
        """Apply schema transformations."""
        for transform in schema.transformation_rules:
            transform_type = transform.get('type')
            
            if transform_type == "field_mapping":
                # Rename fields
                mapping = transform.get('mapping', {})
                new_data = {}
                for old_key, new_key in mapping.items():
                    if old_key in message.data:
                        new_data[new_key] = message.data[old_key]
                    else:
                        new_data[old_key] = message.data[old_key]
                message.data = new_data
            
            elif transform_type == "field_calculation":
                # Calculate new fields
                field_name = transform.get('field')
                expression = transform.get('expression')
                if field_name and expression:
                    # Simple expression evaluation (extend as needed)
                    try:
                        result = eval(expression, {"data": message.data})
                        message.data[field_name] = result
                    except Exception as e:
                        self.logger.warning(f"Field calculation failed: {e}")
        
        return message
    
    async def _flush_message_batch(self, messages: List[IngestionMessage]) -> None:
        """Flush a batch of messages to destinations."""
        try:
            # Group messages by routing key/destination
            destination_groups = {}
            
            for message in messages:
                destination = self._get_message_destination(message)
                if destination not in destination_groups:
                    destination_groups[destination] = []
                destination_groups[destination].append(message)
            
            # Send to each destination
            for destination, message_group in destination_groups.items():
                await self._send_to_destination(destination, message_group)
            
            self.logger.debug(f"Flushed {len(messages)} messages to {len(destination_groups)} destinations")
            
        except Exception as e:
            self.logger.error(f"Failed to flush message batch: {e}")
    
    def _get_message_destination(self, message: IngestionMessage) -> str:
        """Determine message destination."""
        if message.routing_key and message.routing_key in self.routing_rules:
            return self.routing_rules[message.routing_key]
        
        # Default routing based on source
        return f"ingested_{message.source}"
    
    async def _send_to_destination(self, destination: str, messages: List[IngestionMessage]) -> None:
        """Send messages to a specific destination."""
        try:
            # Send to Kafka topic
            for message in messages:
                kafka_message = {
                    "id": message.id,
                    "source": message.source,
                    "timestamp": message.timestamp.isoformat(),
                    "data": message.data,
                    "metadata": message.metadata
                }
                
                self.kafka_producer.send(
                    destination,
                    value=kafka_message,
                    key=message.routing_key
                )
                
                message.status = IngestionStatus.SUCCESS
                INGESTION_RATE.labels(source=message.source, status="success").inc()
            
            # Ensure messages are sent
            self.kafka_producer.flush()
            
        except Exception as e:
            self.logger.error(f"Failed to send to destination {destination}: {e}")
            for message in messages:
                message.status = IngestionStatus.FAILED
                INGESTION_RATE.labels(source=message.source, status="failed").inc()
    
    async def _monitoring_worker(self) -> None:
        """Monitor ingestion system health."""
        while self.is_running:
            try:
                # Update queue size metrics
                INGESTION_QUEUE_SIZE.labels(source="ingestion").set(self.ingestion_queue.qsize())
                INGESTION_QUEUE_SIZE.labels(source="validation").set(self.validation_queue.qsize())
                INGESTION_QUEUE_SIZE.labels(source="output").set(self.output_queue.qsize())
                
                # Store health metrics in Redis
                health_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "ingestion_queue_size": self.ingestion_queue.qsize(),
                    "validation_queue_size": self.validation_queue.qsize(),
                    "output_queue_size": self.output_queue.qsize(),
                    "worker_count": len(self.worker_tasks),
                    "is_running": self.is_running
                }
                
                await self.redis_client.setex(
                    f"ingestion_health:{self.config.name}",
                    300,  # 5 minutes TTL
                    json.dumps(health_data)
                )
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring worker: {e}")
                await asyncio.sleep(60)
    
    async def _throughput_tracker(self) -> None:
        """Track and report throughput metrics."""
        while self.is_running:
            try:
                current_time = time.time()
                window_seconds = 60  # 1-minute window
                
                for source, timestamps in self.throughput_tracker.items():
                    # Remove old timestamps
                    cutoff_time = current_time - window_seconds
                    recent_timestamps = [ts for ts in timestamps if ts >= cutoff_time]
                    self.throughput_tracker[source] = recent_timestamps
                    
                    # Calculate throughput
                    throughput = len(recent_timestamps) / window_seconds
                    INGESTION_THROUGHPUT.labels(source=source).set(throughput)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in throughput tracker: {e}")
                await asyncio.sleep(60)
    
    async def stop(self) -> None:
        """Stop the ingestion engine."""
        self.logger.info("Stopping data ingestion engine...")
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Close connections
        if self.kafka_producer:
            self.kafka_producer.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("Data ingestion engine stopped")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        total_throughput = sum(
            len(timestamps) for timestamps in self.throughput_tracker.values()
        )
        
        return {
            "config": asdict(self.config),
            "is_running": self.is_running,
            "worker_count": len(self.worker_tasks),
            "queue_sizes": {
                "ingestion": self.ingestion_queue.qsize(),
                "validation": self.validation_queue.qsize(),
                "output": self.output_queue.qsize()
            },
            "registered_schemas": list(self.schemas.keys()),
            "source_handlers": list(self.source_handlers.keys()),
            "routing_rules": self.routing_rules,
            "total_throughput": total_throughput
        }


@dataclass
class ValidationResult:
    """Validation result."""
    is_valid: bool
    errors: List[str]


# Example usage and handlers
class LogDataHandler:
    """Example handler for log data."""
    
    def __call__(self, message: IngestionMessage) -> IngestionMessage:
        """Process log message."""
        # Add normalized timestamp
        if 'timestamp' in message.data:
            message.data['normalized_timestamp'] = datetime.utcnow().isoformat()
        
        # Add source metadata
        message.metadata['processed_by'] = 'log_handler'
        message.metadata['processing_time'] = datetime.utcnow().isoformat()
        
        return message


# Example initialization
async def create_ingestion_engine():
    """Create and configure ingestion engine."""
    config = IngestionConfig(
        name="high_throughput_ingestion",
        max_workers=8,
        max_queue_size=100000,
        batch_size=1000,
        enable_validation=True,
        enable_deduplication=True
    )
    
    kafka_config = {
        "bootstrap_servers": ["localhost:9092"],
        "security_protocol": "PLAINTEXT"
    }
    
    engine = DataIngestionEngine(config, kafka_config)
    await engine.initialize()
    
    # Register schema
    log_schema = DataSchema(
        name="application_log",
        version="1.0",
        fields={
            "timestamp": "string",
            "level": "string",
            "message": "string",
            "service": "string"
        },
        validation_rules=[
            ValidationRule(
                field="level",
                rule_type="pattern",
                parameters={"pattern": "^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"}
            )
        ]
    )
    engine.register_schema(log_schema)
    
    # Register handlers
    engine.register_source_handler("application_logs", LogDataHandler())
    engine.add_routing_rule("logs", "processed_logs")
    
    return engine