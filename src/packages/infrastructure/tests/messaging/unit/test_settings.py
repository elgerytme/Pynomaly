"""Unit tests for messaging settings."""

import pytest
from pydantic import ValidationError

from pynomaly.infrastructure.messaging.config.messaging_settings import MessagingSettings


class TestMessagingSettings:
    """Test MessagingSettings model."""

    def test_default_settings(self):
        """Test default messaging settings."""
        settings = MessagingSettings()
        
        assert settings.queue_backend == "redis"
        assert settings.queue_url is None
        assert settings.queue_default_ttl == 3600
        assert settings.max_retries == 3
        assert settings.retry_delay == 30
        
        assert settings.redis_queue_db == 1
        assert settings.redis_stream_maxlen == 10000
        assert settings.redis_consumer_group == "pynomaly_workers"
        assert settings.redis_consumer_name == "worker"
        assert settings.redis_block_timeout == 1000
        
        assert settings.worker_concurrency == 4
        assert settings.task_timeout == 300
        assert settings.task_batch_size == 10
        assert settings.heartbeat_interval == 30
        
        assert settings.enable_metrics is True
        assert settings.metrics_interval == 60
        assert settings.dead_letter_queue_enabled is True
        assert settings.dead_letter_ttl == 86400
        
        assert settings.enable_message_encryption is False
        assert settings.encryption_key is None

    def test_redis_backend_validation(self):
        """Test Redis backend configuration."""
        settings = MessagingSettings(queue_backend="redis")
        assert settings.queue_backend == "redis"

    def test_rabbitmq_backend_validation(self):
        """Test RabbitMQ backend configuration."""
        settings = MessagingSettings(queue_backend="rabbitmq")
        assert settings.queue_backend == "rabbitmq"

    def test_invalid_backend_validation(self):
        """Test invalid backend validation."""
        with pytest.raises(ValidationError) as exc_info:
            MessagingSettings(queue_backend="invalid")
        
        assert "Queue backend must be one of" in str(exc_info.value)

    def test_worker_concurrency_validation(self):
        """Test worker concurrency validation."""
        # Valid values
        settings = MessagingSettings(worker_concurrency=1)
        assert settings.worker_concurrency == 1
        
        settings = MessagingSettings(worker_concurrency=50)
        assert settings.worker_concurrency == 50
        
        # Invalid values
        with pytest.raises(ValidationError):
            MessagingSettings(worker_concurrency=0)
        
        with pytest.raises(ValidationError):
            MessagingSettings(worker_concurrency=101)

    def test_task_timeout_validation(self):
        """Test task timeout validation."""
        # Valid values
        settings = MessagingSettings(task_timeout=1)
        assert settings.task_timeout == 1
        
        settings = MessagingSettings(task_timeout=3600)
        assert settings.task_timeout == 3600
        
        # Invalid values
        with pytest.raises(ValidationError):
            MessagingSettings(task_timeout=0)
        
        with pytest.raises(ValidationError):
            MessagingSettings(task_timeout=-1)

    def test_custom_redis_settings(self):
        """Test custom Redis-specific settings."""
        settings = MessagingSettings(
            queue_backend="redis",
            redis_queue_db=2,
            redis_stream_maxlen=5000,
            redis_consumer_group="custom_workers",
            redis_consumer_name="custom_worker",
            redis_block_timeout=2000
        )
        
        assert settings.redis_queue_db == 2
        assert settings.redis_stream_maxlen == 5000
        assert settings.redis_consumer_group == "custom_workers"
        assert settings.redis_consumer_name == "custom_worker"
        assert settings.redis_block_timeout == 2000

    def test_custom_worker_settings(self):
        """Test custom worker settings."""
        settings = MessagingSettings(
            worker_concurrency=8,
            task_timeout=600,
            task_batch_size=20,
            heartbeat_interval=60
        )
        
        assert settings.worker_concurrency == 8
        assert settings.task_timeout == 600
        assert settings.task_batch_size == 20
        assert settings.heartbeat_interval == 60

    def test_custom_monitoring_settings(self):
        """Test custom monitoring settings."""
        settings = MessagingSettings(
            enable_metrics=False,
            metrics_interval=120,
            dead_letter_queue_enabled=False,
            dead_letter_ttl=172800
        )
        
        assert settings.enable_metrics is False
        assert settings.metrics_interval == 120
        assert settings.dead_letter_queue_enabled is False
        assert settings.dead_letter_ttl == 172800

    def test_encryption_settings(self):
        """Test encryption settings."""
        settings = MessagingSettings(
            enable_message_encryption=True,
            encryption_key="test-key-123"
        )
        
        assert settings.enable_message_encryption is True
        assert settings.encryption_key == "test-key-123"

    def test_url_override(self):
        """Test URL override behavior."""
        settings = MessagingSettings(
            queue_backend="redis",
            queue_url="redis://custom-host:6379/3"
        )
        
        assert settings.queue_url == "redis://custom-host:6379/3"

    def test_retry_settings(self):
        """Test retry configuration."""
        settings = MessagingSettings(
            max_retries=5,
            retry_delay=60
        )
        
        assert settings.max_retries == 5
        assert settings.retry_delay == 60

    def test_ttl_settings(self):
        """Test TTL configuration."""
        settings = MessagingSettings(
            queue_default_ttl=7200,
            dead_letter_ttl=259200  # 3 days
        )
        
        assert settings.queue_default_ttl == 7200
        assert settings.dead_letter_ttl == 259200