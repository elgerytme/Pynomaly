"""
Message Queue Configuration

This module defines configuration classes for message queue integration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MessageBroker(str, Enum):
    """Supported message brokers."""
    
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    REDIS = "redis"
    MEMORY = "memory"
    SQS = "sqs"  # Amazon SQS
    PULSAR = "pulsar"  # Apache Pulsar


class MessageFormat(str, Enum):
    """Message serialization formats."""
    
    JSON = "json"
    PICKLE = "pickle"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    MSGPACK = "msgpack"


class DeliveryMode(str, Enum):
    """Message delivery modes."""
    
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class AcknowledgmentMode(str, Enum):
    """Message acknowledgment modes."""
    
    AUTO = "auto"
    MANUAL = "manual"
    NONE = "none"


class CompressionType(str, Enum):
    """Message compression types."""
    
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class MessageConfig(BaseModel):
    """Configuration for individual messages."""
    
    message_format: MessageFormat = Field(
        default=MessageFormat.JSON,
        description="Message serialization format"
    )
    compression: CompressionType = Field(
        default=CompressionType.NONE,
        description="Message compression type"
    )
    encryption_enabled: bool = Field(
        default=False,
        description="Enable message encryption"
    )
    encryption_key: Optional[str] = Field(
        default=None,
        description="Encryption key for messages"
    )
    ttl_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Message time-to-live in seconds"
    )
    priority: int = Field(
        default=0,
        ge=0,
        le=255,
        description="Message priority (0-255)"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Default message headers"
    )


class QueueConfig(BaseModel):
    """Configuration for queues/topics."""
    
    name: str = Field(description="Queue/topic name")
    durable: bool = Field(
        default=True,
        description="Whether the queue survives broker restarts"
    )
    auto_delete: bool = Field(
        default=False,
        description="Whether to delete queue when no consumers"
    )
    exclusive: bool = Field(
        default=False,
        description="Whether queue is exclusive to one connection"
    )
    max_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of messages in queue"
    )
    max_length_bytes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum queue size in bytes"
    )
    message_ttl: Optional[int] = Field(
        default=None,
        ge=1,
        description="Default message TTL in milliseconds"
    )
    
    # Kafka-specific
    partitions: int = Field(
        default=1,
        ge=1,
        description="Number of partitions (Kafka)"
    )
    replication_factor: int = Field(
        default=1,
        ge=1,
        description="Replication factor (Kafka)"
    )
    
    # RabbitMQ-specific
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional queue arguments"
    )


class ConsumerConfig(BaseModel):
    """Configuration for message consumers."""
    
    consumer_group_id: Optional[str] = Field(
        default=None,
        description="Consumer group ID"
    )
    auto_offset_reset: str = Field(
        default="earliest",
        description="Auto offset reset strategy"
    )
    enable_auto_commit: bool = Field(
        default=True,
        description="Enable automatic offset commits"
    )
    auto_commit_interval: int = Field(
        default=5000,
        ge=1,
        description="Auto commit interval in milliseconds"
    )
    max_poll_records: int = Field(
        default=500,
        ge=1,
        description="Maximum records per poll"
    )
    fetch_min_bytes: int = Field(
        default=1,
        ge=1,
        description="Minimum bytes per fetch"
    )
    fetch_max_wait_ms: int = Field(
        default=500,
        ge=0,
        description="Maximum wait time for fetch"
    )
    session_timeout_ms: int = Field(
        default=30000,
        ge=1,
        description="Session timeout in milliseconds"
    )
    heartbeat_interval_ms: int = Field(
        default=3000,
        ge=1,
        description="Heartbeat interval in milliseconds"
    )
    max_poll_interval_ms: int = Field(
        default=300000,
        ge=1,
        description="Maximum poll interval in milliseconds"
    )
    prefetch_count: int = Field(
        default=1,
        ge=0,
        description="Number of messages to prefetch"
    )
    acknowledgment_mode: AcknowledgmentMode = Field(
        default=AcknowledgmentMode.AUTO,
        description="Message acknowledgment mode"
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts"
    )
    retry_delay_ms: int = Field(
        default=1000,
        ge=0,
        description="Retry delay in milliseconds"
    )
    dead_letter_queue: Optional[str] = Field(
        default=None,
        description="Dead letter queue name"
    )


class ProducerConfig(BaseModel):
    """Configuration for message producers."""
    
    batch_size: int = Field(
        default=16384,
        ge=1,
        description="Batch size for sending messages"
    )
    linger_ms: int = Field(
        default=0,
        ge=0,
        description="Time to wait before sending batch"
    )
    buffer_memory: int = Field(
        default=33554432,
        ge=1,
        description="Buffer memory for producer"
    )
    compression_type: CompressionType = Field(
        default=CompressionType.NONE,
        description="Producer compression type"
    )
    retries: int = Field(
        default=3,
        ge=0,
        description="Number of retries"
    )
    retry_backoff_ms: int = Field(
        default=100,
        ge=0,
        description="Retry backoff in milliseconds"
    )
    request_timeout_ms: int = Field(
        default=30000,
        ge=1,
        description="Request timeout in milliseconds"
    )
    delivery_timeout_ms: int = Field(
        default=120000,
        ge=1,
        description="Delivery timeout in milliseconds"
    )
    max_in_flight_requests: int = Field(
        default=5,
        ge=1,
        description="Maximum in-flight requests"
    )
    acks: str = Field(
        default="all",
        description="Acknowledgment strategy"
    )
    enable_idempotence: bool = Field(
        default=True,
        description="Enable producer idempotence"
    )
    delivery_mode: DeliveryMode = Field(
        default=DeliveryMode.AT_LEAST_ONCE,
        description="Message delivery mode"
    )
    mandatory: bool = Field(
        default=False,
        description="Whether message is mandatory (RabbitMQ)"
    )
    immediate: bool = Field(
        default=False,
        description="Whether message is immediate (RabbitMQ)"
    )


class ConnectionConfig(BaseModel):
    """Configuration for broker connections."""
    
    host: str = Field(default="localhost", description="Broker host")
    port: int = Field(default=5672, ge=1, le=65535, description="Broker port")
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password")
    virtual_host: str = Field(default="/", description="Virtual host")
    
    # Connection pool settings
    max_connections: int = Field(
        default=10,
        ge=1,
        description="Maximum connections in pool"
    )
    min_connections: int = Field(
        default=1,
        ge=1,
        description="Minimum connections in pool"
    )
    connection_timeout: int = Field(
        default=30,
        ge=1,
        description="Connection timeout in seconds"
    )
    heartbeat: int = Field(
        default=600,
        ge=0,
        description="Heartbeat interval in seconds"
    )
    blocked_connection_timeout: int = Field(
        default=300,
        ge=0,
        description="Blocked connection timeout"
    )
    
    # SSL/TLS settings
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS")
    ssl_cert_path: Optional[str] = Field(default=None, description="SSL certificate path")
    ssl_key_path: Optional[str] = Field(default=None, description="SSL key path")
    ssl_ca_path: Optional[str] = Field(default=None, description="SSL CA path")
    ssl_verify: bool = Field(default=True, description="Verify SSL certificates")
    
    # Kafka-specific
    bootstrap_servers: List[str] = Field(
        default_factory=lambda: ["localhost:9092"],
        description="Kafka bootstrap servers"
    )
    security_protocol: str = Field(
        default="PLAINTEXT",
        description="Security protocol"
    )
    sasl_mechanism: Optional[str] = Field(
        default=None,
        description="SASL mechanism"
    )
    sasl_plain_username: Optional[str] = Field(
        default=None,
        description="SASL plain username"
    )
    sasl_plain_password: Optional[str] = Field(
        default=None,
        description="SASL plain password"
    )
    
    # Redis-specific
    db: int = Field(default=0, ge=0, description="Redis database number")
    redis_url: Optional[str] = Field(default=None, description="Redis URL")
    
    # AWS SQS-specific
    region: Optional[str] = Field(default=None, description="AWS region")
    access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    
    @classmethod
    def from_env(cls, broker: MessageBroker) -> ConnectionConfig:
        """Create connection config from environment variables."""
        if broker == MessageBroker.RABBITMQ:
            return cls(
                host=os.getenv("RABBITMQ_HOST", "localhost"),
                port=int(os.getenv("RABBITMQ_PORT", "5672")),
                username=os.getenv("RABBITMQ_USERNAME"),
                password=os.getenv("RABBITMQ_PASSWORD"),
                virtual_host=os.getenv("RABBITMQ_VHOST", "/"),
                ssl_enabled=os.getenv("RABBITMQ_SSL", "false").lower() == "true",
            )
        elif broker == MessageBroker.KAFKA:
            return cls(
                bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(","),
                security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
                sasl_mechanism=os.getenv("KAFKA_SASL_MECHANISM"),
                sasl_plain_username=os.getenv("KAFKA_SASL_USERNAME"),
                sasl_plain_password=os.getenv("KAFKA_SASL_PASSWORD"),
            )
        elif broker == MessageBroker.REDIS:
            return cls(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD"),
                db=int(os.getenv("REDIS_DB", "0")),
                redis_url=os.getenv("REDIS_URL"),
                ssl_enabled=os.getenv("REDIS_SSL", "false").lower() == "true",
            )
        elif broker == MessageBroker.SQS:
            return cls(
                region=os.getenv("AWS_REGION", "us-east-1"),
                access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        else:
            return cls()


class MessageQueueConfig(BaseModel):
    """Main configuration for message queue integration."""
    
    broker: MessageBroker = Field(
        default=MessageBroker.MEMORY,
        description="Message broker type"
    )
    connection: ConnectionConfig = Field(
        default_factory=ConnectionConfig,
        description="Connection configuration"
    )
    message: MessageConfig = Field(
        default_factory=MessageConfig,
        description="Message configuration"
    )
    producer: ProducerConfig = Field(
        default_factory=ProducerConfig,
        description="Producer configuration"
    )
    consumer: ConsumerConfig = Field(
        default_factory=ConsumerConfig,
        description="Consumer configuration"
    )
    
    # Global settings
    default_queue_config: QueueConfig = Field(
        default_factory=lambda: QueueConfig(name="default"),
        description="Default queue configuration"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )
    health_check_interval: int = Field(
        default=30,
        ge=1,
        description="Health check interval in seconds"
    )
    
    # Error handling
    enable_dead_letter_queue: bool = Field(
        default=True,
        description="Enable dead letter queue"
    )
    max_retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts"
    )
    retry_delay_seconds: int = Field(
        default=1,
        ge=0,
        description="Retry delay in seconds"
    )
    
    # Performance tuning
    worker_pool_size: int = Field(
        default=4,
        ge=1,
        description="Worker pool size"
    )
    max_concurrent_messages: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent messages"
    )
    
    @classmethod
    def from_env(cls, broker: Optional[MessageBroker] = None) -> MessageQueueConfig:
        """Create configuration from environment variables."""
        if broker is None:
            broker = MessageBroker(os.getenv("MESSAGE_BROKER", "memory"))
        
        return cls(
            broker=broker,
            connection=ConnectionConfig.from_env(broker),
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            enable_tracing=os.getenv("ENABLE_TRACING", "false").lower() == "true",
            worker_pool_size=int(os.getenv("WORKER_POOL_SIZE", "4")),
            max_concurrent_messages=int(os.getenv("MAX_CONCURRENT_MESSAGES", "100")),
        )


@dataclass
class MessageQueueState:
    """Runtime state for message queue system."""
    
    # Connection state
    is_connected: bool = False
    connection_count: int = 0
    last_connection_time: Optional[float] = None
    
    # Message statistics
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    
    # Performance metrics
    average_send_time: float = 0.0
    average_receive_time: float = 0.0
    current_queue_size: int = 0
    
    # Error tracking
    error_count: int = 0
    last_error_time: Optional[float] = None
    last_error_message: Optional[str] = None
    
    # Health status
    is_healthy: bool = True
    health_check_failures: int = 0
    
    # Active queues/topics
    active_queues: Dict[str, QueueConfig] = field(default_factory=dict)
    active_consumers: Dict[str, ConsumerConfig] = field(default_factory=dict)
    active_producers: Dict[str, ProducerConfig] = field(default_factory=dict)
