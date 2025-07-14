"""Messaging configuration settings."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class MessagingSettings(BaseModel):
    """Message queue configuration settings."""

    model_config = {"protected_namespaces": ()}

    # Queue backend configuration
    queue_backend: str = "redis"  # redis, rabbitmq, celery
    queue_url: str | None = None
    queue_default_ttl: int = 3600  # 1 hour default TTL
    max_retries: int = 3
    retry_delay: int = 30  # seconds

    # Redis-specific settings
    redis_queue_db: int = 1  # Separate DB from cache (usually 0)
    redis_stream_maxlen: int = 10000
    redis_consumer_group: str = "pynomaly_workers"
    redis_consumer_name: str = "worker"
    redis_block_timeout: int = 1000  # milliseconds

    # Task processing
    worker_concurrency: int = 4
    task_timeout: int = 300  # 5 minutes
    task_batch_size: int = 10
    heartbeat_interval: int = 30  # seconds

    # Queue monitoring
    enable_metrics: bool = True
    metrics_interval: int = 60  # seconds
    dead_letter_queue_enabled: bool = True
    dead_letter_ttl: int = 86400  # 24 hours

    # Security
    enable_message_encryption: bool = False
    encryption_key: str | None = None

    @field_validator("queue_backend")
    @classmethod
    def validate_queue_backend(cls, v: str) -> str:
        """Validate queue backend type."""
        allowed_backends = {"redis", "rabbitmq", "celery"}
        if v not in allowed_backends:
            raise ValueError(f"Queue backend must be one of: {allowed_backends}")
        return v

    @field_validator("worker_concurrency")
    @classmethod
    def validate_worker_concurrency(cls, v: int) -> int:
        """Validate worker concurrency setting."""
        if v < 1 or v > 100:
            raise ValueError("Worker concurrency must be between 1 and 100")
        return v

    @field_validator("task_timeout")
    @classmethod
    def validate_task_timeout(cls, v: int) -> int:
        """Validate task timeout setting."""
        if v < 1:
            raise ValueError("Task timeout must be positive")
        return v