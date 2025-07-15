"""Unit tests for messaging models."""

import pytest
from datetime import datetime, timezone, timedelta

from pynomaly.infrastructure.messaging.models.messages import (
    Message, MessagePriority, MessageStatus
)
from pynomaly.infrastructure.messaging.models.tasks import (
    Task, TaskType, TaskStatus
)


class TestMessage:
    """Test Message model."""

    def test_message_creation(self):
        """Test message creation with defaults."""
        message = Message(
            queue_name="test_queue",
            payload={"data": "test"}
        )
        
        assert message.queue_name == "test_queue"
        assert message.payload == {"data": "test"}
        assert message.priority == MessagePriority.NORMAL
        assert message.status == MessageStatus.PENDING
        assert message.attempts == 0
        assert message.max_attempts == 3
        assert isinstance(message.created_at, datetime)

    def test_message_processing_lifecycle(self):
        """Test message processing state transitions."""
        message = Message(
            queue_name="test_queue",
            payload={"data": "test"}
        )
        
        # Mark as processing
        message.mark_processing()
        assert message.status == MessageStatus.PROCESSING
        assert message.processing_started_at is not None

        # Mark as completed
        message.mark_completed()
        assert message.status == MessageStatus.COMPLETED
        assert message.processing_completed_at is not None

    def test_message_failure_and_retry(self):
        """Test message failure and retry logic."""
        message = Message(
            queue_name="test_queue",
            payload={"data": "test"},
            max_attempts=2
        )
        
        # First failure - should retry
        message.mark_retrying("First error")
        assert message.status == MessageStatus.RETRYING
        assert message.attempts == 1
        assert message.last_error == "First error"
        assert message.can_retry() is True

        # Second failure - should go to dead letter
        message.mark_retrying("Second error")
        assert message.status == MessageStatus.DEAD_LETTER
        assert message.attempts == 2
        assert message.can_retry() is False

    def test_message_expiration(self):
        """Test message expiration logic."""
        # Non-expired message
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        message = Message(
            queue_name="test_queue",
            payload={"data": "test"},
            expires_at=future_time
        )
        assert message.is_expired() is False

        # Expired message
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        message.expires_at = past_time
        assert message.is_expired() is True

    def test_message_scheduling(self):
        """Test message scheduling logic."""
        # Future scheduled message
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        message = Message(
            queue_name="test_queue",
            payload={"data": "test"},
            scheduled_at=future_time
        )
        assert message.is_scheduled() is True

        # Past scheduled message
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        message.scheduled_at = past_time
        assert message.is_scheduled() is False

    def test_message_serialization(self):
        """Test message serialization and deserialization."""
        original = Message(
            queue_name="test_queue",
            payload={"data": "test", "number": 42},
            priority=MessagePriority.HIGH,
            headers={"custom": "header"}
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = Message.from_dict(data)
        
        assert restored.queue_name == original.queue_name
        assert restored.payload == original.payload
        assert restored.priority == original.priority
        assert restored.headers == original.headers
        assert restored.id == original.id


class TestTask:
    """Test Task model."""

    def test_task_creation(self):
        """Test task creation with defaults."""
        task = Task(
            task_type=TaskType.ANOMALY_DETECTION,
            name="Test Task",
            function_name="test_function",
            args=[1, 2, 3],
            kwargs={"param": "value"}
        )
        
        assert task.task_type == TaskType.ANOMALY_DETECTION
        assert task.name == "Test Task"
        assert task.function_name == "test_function"
        assert task.args == [1, 2, 3]
        assert task.kwargs == {"param": "value"}
        assert task.status == TaskStatus.PENDING
        assert task.progress == 0.0
        assert task.max_retries == 3
        assert task.retry_count == 0

    def test_task_lifecycle(self):
        """Test task processing lifecycle."""
        task = Task(
            task_type=TaskType.DATA_PROFILING,
            name="Test Task",
            function_name="test_function"
        )
        
        # Queue task
        task.mark_queued()
        assert task.status == TaskStatus.QUEUED

        # Start running
        task.mark_running()
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None

        # Update progress
        task.update_progress(50.0)
        assert task.progress == 50.0

        # Complete task
        result = {"output": "success"}
        task.mark_completed(result)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.progress == 100.0
        assert task.completed_at is not None
        assert task.is_finished() is True

    def test_task_failure_and_retry(self):
        """Test task failure and retry logic."""
        task = Task(
            task_type=TaskType.MODEL_TRAINING,
            name="Test Task",
            function_name="test_function",
            max_retries=2
        )
        
        # First failure - should retry
        task.mark_failed("First error")
        assert task.status == TaskStatus.RETRYING
        assert task.retry_count == 1
        assert task.error == "First error"
        assert task.can_retry() is True

        # Second failure - should retry
        task.mark_failed("Second error")
        assert task.status == TaskStatus.RETRYING
        assert task.retry_count == 2
        assert task.can_retry() is True

        # Third failure - should not retry
        task.mark_failed("Third error")
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 2  # max reached
        assert task.can_retry() is False

    def test_task_timeout(self):
        """Test task timeout handling."""
        task = Task(
            task_type=TaskType.REPORT_GENERATION,
            name="Test Task",
            function_name="test_function",
            timeout=300
        )
        
        task.mark_timeout()
        assert task.status == TaskStatus.TIMEOUT
        assert "timed out after 300 seconds" in task.error
        assert task.is_finished() is True

    def test_task_cancellation(self):
        """Test task cancellation."""
        task = Task(
            task_type=TaskType.DATA_PROCESSING,
            name="Test Task",
            function_name="test_function"
        )
        
        task.mark_cancelled()
        assert task.status == TaskStatus.CANCELLED
        assert task.is_finished() is True

    def test_task_scheduling(self):
        """Test task scheduling logic."""
        # Future scheduled task
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        task = Task(
            task_type=TaskType.CLEANUP,
            name="Test Task",
            function_name="test_function",
            scheduled_at=future_time
        )
        assert task.is_scheduled() is True

        # Past scheduled task
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        task.scheduled_at = past_time
        assert task.is_scheduled() is False

    def test_task_duration(self):
        """Test task duration calculation."""
        task = Task(
            task_type=TaskType.VALIDATION,
            name="Test Task",
            function_name="test_function"
        )
        
        # No duration before starting
        assert task.get_duration() is None

        # Start task
        task.mark_running()
        start_time = task.started_at
        
        # Simulate some processing time
        import time
        time.sleep(0.1)
        
        # Complete task
        task.mark_completed()
        
        duration = task.get_duration()
        assert duration is not None
        assert duration >= 0.1  # At least the sleep time

    def test_task_serialization(self):
        """Test task serialization and deserialization."""
        original = Task(
            task_type=TaskType.ANOMALY_DETECTION,
            name="Test Task",
            function_name="test_function",
            args=[1, "test"],
            kwargs={"param": 42},
            tags=["urgent", "ml"],
            user_id="user123"
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = Task.from_dict(data)
        
        assert restored.task_type == original.task_type
        assert restored.name == original.name
        assert restored.function_name == original.function_name
        assert restored.args == original.args
        assert restored.kwargs == original.kwargs
        assert restored.tags == original.tags
        assert restored.user_id == original.user_id
        assert restored.id == original.id

    def test_task_dependencies(self):
        """Test task dependency handling."""
        parent_task = Task(
            task_type=TaskType.DATA_PROCESSING,
            name="Parent Task",
            function_name="parent_function"
        )
        
        child_task = Task(
            task_type=TaskType.ANOMALY_DETECTION,
            name="Child Task",
            function_name="child_function",
            depends_on=[parent_task.id],
            parent_task_id=parent_task.id
        )
        
        assert parent_task.id in child_task.depends_on
        assert child_task.parent_task_id == parent_task.id


class TestEnums:
    """Test enum values."""

    def test_message_priority_values(self):
        """Test MessagePriority enum values."""
        assert MessagePriority.LOW == "low"
        assert MessagePriority.NORMAL == "normal"
        assert MessagePriority.HIGH == "high"
        assert MessagePriority.CRITICAL == "critical"

    def test_message_status_values(self):
        """Test MessageStatus enum values."""
        assert MessageStatus.PENDING == "pending"
        assert MessageStatus.PROCESSING == "processing"
        assert MessageStatus.COMPLETED == "completed"
        assert MessageStatus.FAILED == "failed"
        assert MessageStatus.RETRYING == "retrying"
        assert MessageStatus.DEAD_LETTER == "dead_letter"

    def test_task_type_values(self):
        """Test TaskType enum values."""
        assert TaskType.ANOMALY_DETECTION == "anomaly_detection"
        assert TaskType.DATA_PROFILING == "data_profiling"
        assert TaskType.MODEL_TRAINING == "model_training"
        assert TaskType.DATA_PROCESSING == "data_processing"
        assert TaskType.REPORT_GENERATION == "report_generation"
        assert TaskType.NOTIFICATION == "notification"
        assert TaskType.CLEANUP == "cleanup"
        assert TaskType.EXPORT == "export"
        assert TaskType.IMPORT == "import"
        assert TaskType.VALIDATION == "validation"

    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.QUEUED == "queued"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"
        assert TaskStatus.RETRYING == "retrying"
        assert TaskStatus.TIMEOUT == "timeout"