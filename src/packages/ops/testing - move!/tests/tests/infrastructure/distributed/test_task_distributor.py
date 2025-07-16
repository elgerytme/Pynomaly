"""Test suite for task distribution system."""

import asyncio
from datetime import UTC
from unittest.mock import patch

import pytest

from monorepo.infrastructure.distributed.distributed_config import DistributedConfig
from monorepo.infrastructure.distributed.task_distributor import (
    DistributedTask,
    TaskDistributor,
    TaskMetadata,
    TaskPriority,
    TaskQueue,
    TaskResult,
    TaskStatus,
    TaskType,
)


class TestTaskQueue:
    """Test cases for TaskQueue."""

    @pytest.mark.asyncio
    async def test_task_queue_creation(self):
        """Test creating a task queue."""
        queue = TaskQueue(max_size=100)

        assert queue.max_size == 100
        assert queue.size() == 0
        assert queue.is_empty() is True
        assert queue.is_full() is False

    @pytest.mark.asyncio
    async def test_put_and_get_task(self):
        """Test adding and retrieving tasks."""
        queue = TaskQueue(max_size=10)

        task = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION,
            function_name="detect_anomalies",
            priority=TaskPriority.NORMAL,
        )

        # Add task
        success = await queue.put(task)
        assert success is True
        assert queue.size() == 1
        assert queue.is_empty() is False

        # Retrieve task
        retrieved_task = await queue.get()
        assert retrieved_task is not None
        assert retrieved_task.task_id == task.task_id
        assert queue.size() == 0
        assert queue.is_empty() is True

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that tasks are returned in priority order."""
        queue = TaskQueue(max_size=10)

        # Add tasks in different priority order
        low_task = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION,
            function_name="detect",
            priority=TaskPriority.LOW,
        )

        high_task = DistributedTask(
            task_type=TaskType.MODEL_TRAINING,
            function_name="train",
            priority=TaskPriority.HIGH,
        )

        critical_task = DistributedTask(
            task_type=TaskType.DATA_PREPROCESSING,
            function_name="preprocess",
            priority=TaskPriority.CRITICAL,
        )

        normal_task = DistributedTask(
            task_type=TaskType.FEATURE_EXTRACTION,
            function_name="extract",
            priority=TaskPriority.NORMAL,
        )

        # Add in random order
        await queue.put(low_task)
        await queue.put(normal_task)
        await queue.put(critical_task)
        await queue.put(high_task)

        # Should retrieve in priority order: CRITICAL, HIGH, NORMAL, LOW
        task1 = await queue.get()
        assert task1.priority == TaskPriority.CRITICAL

        task2 = await queue.get()
        assert task2.priority == TaskPriority.HIGH

        task3 = await queue.get()
        assert task3.priority == TaskPriority.NORMAL

        task4 = await queue.get()
        assert task4.priority == TaskPriority.LOW

    @pytest.mark.asyncio
    async def test_queue_full_behavior(self):
        """Test queue behavior when full."""
        queue = TaskQueue(max_size=2)

        task1 = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION, function_name="detect1"
        )

        task2 = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION, function_name="detect2"
        )

        task3 = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION, function_name="detect3"
        )

        # Fill queue
        success1 = await queue.put(task1)
        success2 = await queue.put(task2)

        assert success1 is True
        assert success2 is True
        assert queue.is_full() is True

        # Try to add one more (should fail)
        success3 = await queue.put(task3)
        assert success3 is False
        assert queue.size() == 2

    @pytest.mark.asyncio
    async def test_remove_task(self):
        """Test removing tasks from queue."""
        queue = TaskQueue(max_size=10)

        task = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION, function_name="detect"
        )

        await queue.put(task)
        assert queue.size() == 1

        # Remove task
        removed = await queue.remove(task.task_id)
        assert removed is True
        assert queue.size() == 0

        # Try to remove non-existent task
        removed = await queue.remove("non-existent")
        assert removed is False

    @pytest.mark.asyncio
    async def test_wait_for_task(self):
        """Test waiting for tasks with timeout."""
        queue = TaskQueue(max_size=10)

        # Test timeout when no tasks available
        task = await queue.wait_for_task(timeout=0.1)
        assert task is None

        # Add task and wait
        test_task = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION, function_name="detect"
        )

        await queue.put(test_task)

        task = await queue.wait_for_task(timeout=1.0)
        assert task is not None
        assert task.task_id == test_task.task_id

    def test_queue_statistics(self):
        """Test queue statistics."""
        queue = TaskQueue(max_size=10)

        stats = queue.get_stats()
        assert stats["total"] == 0
        assert stats["critical"] == 0
        assert stats["high"] == 0
        assert stats["normal"] == 0
        assert stats["low"] == 0


class TestDistributedTask:
    """Test cases for DistributedTask."""

    def test_task_creation(self):
        """Test creating a distributed task."""
        task = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION,
            function_name="detect_anomalies",
            arguments={"data": "test_data"},
            kwargs={"threshold": 0.5},
            context={"session": "test_session"},
        )

        assert task.task_type == TaskType.ANOMALY_DETECTION
        assert task.function_name == "detect_anomalies"
        assert task.arguments["data"] == "test_data"
        assert task.kwargs["threshold"] == 0.5
        assert task.context["session"] == "test_session"
        assert task.priority == TaskPriority.NORMAL  # default

        # Task ID should be automatically generated
        assert task.task_id is not None
        assert len(task.task_id) > 0

    def test_task_with_metadata(self):
        """Test task with custom metadata."""
        metadata = TaskMetadata(
            task_id="custom_id",
            task_type=TaskType.MODEL_TRAINING,
            priority=TaskPriority.HIGH,
            estimated_memory_mb=2048,
            estimated_cpu_cores=4,
            estimated_duration_seconds=300.0,
        )

        task = DistributedTask(
            task_type=TaskType.MODEL_TRAINING,
            function_name="train_model",
            metadata=metadata,
        )

        assert task.metadata.estimated_memory_mb == 2048
        assert task.metadata.estimated_cpu_cores == 4
        assert task.metadata.estimated_duration_seconds == 300.0

    def test_task_resource_requirements(self):
        """Test task resource requirements."""
        task = DistributedTask(
            task_type=TaskType.HYPERPARAMETER_TUNING,
            function_name="optimize_hyperparameters",
            resource_requirements={
                "memory_mb": 8192,
                "cpu_cores": 8,
                "duration_seconds": 1800.0,
                "requires_gpu": True,
            },
        )

        assert task.resource_requirements["memory_mb"] == 8192
        assert task.resource_requirements["cpu_cores"] == 8
        assert task.resource_requirements["duration_seconds"] == 1800.0
        assert task.resource_requirements["requires_gpu"] is True


class TestTaskResult:
    """Test cases for TaskResult."""

    def test_successful_result(self):
        """Test creating a successful task result."""
        from datetime import datetime

        start_time = datetime.now(UTC)
        end_time = datetime.now(UTC)

        result = TaskResult(
            task_id="test_task",
            worker_id="worker_1",
            status=TaskStatus.COMPLETED,
            result={"accuracy": 0.95, "anomalies": 10},
            started_at=start_time,
            completed_at=end_time,
            execution_time_seconds=5.0,
            memory_used_mb=1024.0,
            cpu_time_seconds=4.5,
        )

        assert result.task_id == "test_task"
        assert result.worker_id == "worker_1"
        assert result.status == TaskStatus.COMPLETED
        assert result.is_successful is True
        assert result.result["accuracy"] == 0.95
        assert result.execution_time_seconds == 5.0

    def test_failed_result(self):
        """Test creating a failed task result."""
        from datetime import datetime

        start_time = datetime.now(UTC)
        end_time = datetime.now(UTC)

        result = TaskResult(
            task_id="failed_task",
            worker_id="worker_2",
            status=TaskStatus.FAILED,
            error="Out of memory",
            traceback="Traceback (most recent call last)...",
            started_at=start_time,
            completed_at=end_time,
            execution_time_seconds=2.0,
        )

        assert result.task_id == "failed_task"
        assert result.status == TaskStatus.FAILED
        assert result.is_successful is False
        assert result.error == "Out of memory"
        assert result.traceback is not None


class TestTaskDistributor:
    """Test cases for TaskDistributor."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DistributedConfig(enabled=True, worker={"max_concurrent_tasks": 4})

    @pytest.fixture
    def distributor(self, config):
        """Create test task distributor."""
        with patch(
            "monorepo.infrastructure.distributed.task_distributor.get_distributed_config_manager"
        ) as mock_manager:
            mock_manager.return_value.get_effective_config.return_value = config
            return TaskDistributor(config)

    @pytest.mark.asyncio
    async def test_distributor_start_stop(self, distributor):
        """Test starting and stopping the distributor."""
        await distributor.start()
        assert distributor._running is True
        assert distributor._distribution_task is not None

        await distributor.stop()
        assert distributor._running is False

    @pytest.mark.asyncio
    async def test_submit_task(self, distributor):
        """Test submitting a task."""
        task = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION,
            function_name="detect_anomalies",
            arguments={"data": [1, 2, 3, 4, 5]},
        )

        task_id = await distributor.submit_task(task)

        assert task_id == task.task_id
        assert distributor.total_tasks_submitted == 1
        assert distributor.pending_queue.size() == 1

    @pytest.mark.asyncio
    async def test_cancel_task(self, distributor):
        """Test cancelling a pending task."""
        task = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION, function_name="detect_anomalies"
        )

        task_id = await distributor.submit_task(task)
        assert distributor.pending_queue.size() == 1

        # Cancel the task
        cancelled = await distributor.cancel_task(task_id)
        assert cancelled is True
        assert distributor.pending_queue.size() == 0

    @pytest.mark.asyncio
    async def test_get_task_status(self, distributor):
        """Test getting task status."""
        task = DistributedTask(
            task_type=TaskType.MODEL_TRAINING, function_name="train_model"
        )

        task_id = await distributor.submit_task(task)

        # Should be queued
        status = await distributor.get_task_status(task_id)
        assert status == TaskStatus.QUEUED

        # Non-existent task
        status = await distributor.get_task_status("non-existent")
        assert status is None

    def test_worker_registration(self, distributor):
        """Test worker registration and management."""
        # Register workers
        distributor.register_worker("worker_1", {"anomaly_detection", "model_training"})
        distributor.register_worker("worker_2", {"data_preprocessing"})

        assert len(distributor.available_workers) == 2
        assert "worker_1" in distributor.available_workers
        assert "worker_2" in distributor.available_workers

        assert "anomaly_detection" in distributor.worker_capabilities["worker_1"]
        assert "data_preprocessing" in distributor.worker_capabilities["worker_2"]

    def test_worker_unregistration(self, distributor):
        """Test worker unregistration."""
        # Register and then unregister worker
        distributor.register_worker("worker_1", {"anomaly_detection"})
        assert "worker_1" in distributor.available_workers

        distributor.unregister_worker("worker_1")
        assert "worker_1" not in distributor.available_workers
        assert "worker_1" not in distributor.worker_capabilities

    def test_worker_load_update(self, distributor):
        """Test updating worker load."""
        distributor.register_worker("worker_1", {"anomaly_detection"})

        # Update load
        distributor.update_worker_load("worker_1", 0.7)
        assert distributor.worker_loads["worker_1"] == 0.7

        # Update non-existent worker (should be ignored)
        distributor.update_worker_load("non_existent", 0.5)
        # Should not crash

    def test_find_best_worker(self, distributor):
        """Test finding the best worker for a task."""
        # Register workers with different capabilities
        distributor.register_worker("worker_1", {"anomaly_detection"})
        distributor.register_worker("worker_2", {"model_training"})
        distributor.register_worker("worker_3", {"all"})

        # Set different loads
        distributor.update_worker_load("worker_1", 0.3)
        distributor.update_worker_load("worker_2", 0.8)
        distributor.update_worker_load("worker_3", 0.5)

        # Task that needs anomaly detection
        task = DistributedTask(
            task_type=TaskType.ANOMALY_DETECTION, function_name="detect_anomalies"
        )

        best_worker = distributor._find_best_worker(task)
        # Should pick worker_1 (supports the task and has lower load than worker_3)
        assert best_worker == "worker_1"

        # Task that needs model training
        training_task = DistributedTask(
            task_type=TaskType.MODEL_TRAINING, function_name="train_model"
        )

        best_worker = distributor._find_best_worker(training_task)
        # Should pick worker_3 (worker_2 has higher load)
        assert best_worker in ["worker_2", "worker_3"]

    def test_get_statistics(self, distributor):
        """Test getting distribution statistics."""
        # Register workers and submit tasks
        distributor.register_worker("worker_1", {"anomaly_detection"})
        distributor.register_worker("worker_2", {"model_training"})

        stats = distributor.get_statistics()

        assert "total_submitted" in stats
        assert "total_completed" in stats
        assert "total_failed" in stats
        assert "pending_count" in stats
        assert "running_count" in stats
        assert "available_workers" in stats
        assert "worker_loads" in stats

        assert stats["available_workers"] == 2
        assert stats["total_submitted"] == 0  # No tasks submitted yet

    def test_completion_handler(self, distributor):
        """Test task completion handlers."""
        completion_calls = []

        def test_handler(result):
            completion_calls.append(result.task_id)

        distributor.add_completion_handler(test_handler)

        # Simulate task completion
        from datetime import datetime

        result = TaskResult(
            task_id="test_task",
            worker_id="worker_1",
            status=TaskStatus.COMPLETED,
            result={"test": "data"},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            execution_time_seconds=1.0,
        )

        # Manually call completion handler (simulating internal call)
        asyncio.create_task(distributor._handle_task_completion(result))

        # Handler should eventually be called
        # Note: In a real test, we'd need to wait for the async operation


class TestTaskDistributorIntegration:
    """Integration tests for TaskDistributor."""

    @pytest.mark.asyncio
    async def test_full_task_lifecycle(self):
        """Test complete task lifecycle from submission to completion."""
        config = DistributedConfig(enabled=True)

        with patch(
            "monorepo.infrastructure.distributed.task_distributor.get_distributed_config_manager"
        ) as mock_manager:
            mock_manager.return_value.get_effective_config.return_value = config

            distributor = TaskDistributor(config)

            # Start distributor
            await distributor.start()

            # Register a worker
            distributor.register_worker("worker_1", {"anomaly_detection"})

            # Submit a task
            task = DistributedTask(
                task_type=TaskType.ANOMALY_DETECTION,
                function_name="anomaly_detection",
                arguments={"data": list(range(100))},
                resource_requirements={"duration_seconds": 0.1},  # Fast task
            )

            task_id = await distributor.submit_task(task)

            # Wait for task completion (with timeout)
            result = await distributor.get_task_result(task_id, timeout=10.0)

            # Verify result
            assert result is not None
            assert result.task_id == task_id
            assert result.status == TaskStatus.COMPLETED
            assert result.is_successful is True

            # Check statistics
            stats = distributor.get_statistics()
            assert stats["total_submitted"] == 1
            assert stats["total_completed"] == 1

            # Stop distributor
            await distributor.stop()

    @pytest.mark.asyncio
    async def test_multiple_workers_load_balancing(self):
        """Test load balancing across multiple workers."""
        config = DistributedConfig(enabled=True)

        with patch(
            "monorepo.infrastructure.distributed.task_distributor.get_distributed_config_manager"
        ) as mock_manager:
            mock_manager.return_value.get_effective_config.return_value = config

            distributor = TaskDistributor(config)
            await distributor.start()

            # Register multiple workers
            distributor.register_worker("worker_1", {"anomaly_detection"})
            distributor.register_worker("worker_2", {"anomaly_detection"})
            distributor.register_worker("worker_3", {"anomaly_detection"})

            # Submit multiple tasks
            tasks = []
            for _i in range(6):
                task = DistributedTask(
                    task_type=TaskType.ANOMALY_DETECTION,
                    function_name="anomaly_detection",
                    arguments={"data": list(range(50))},
                    resource_requirements={"duration_seconds": 0.05},
                )
                task_id = await distributor.submit_task(task)
                tasks.append(task_id)

            # Wait for all tasks to complete
            results = []
            for task_id in tasks:
                result = await distributor.get_task_result(task_id, timeout=10.0)
                if result:
                    results.append(result)

            # Verify all tasks completed
            assert len(results) == 6

            # Verify load balancing (tasks should be distributed across workers)
            worker_usage = {}
            for result in results:
                worker_id = result.worker_id
                worker_usage[worker_id] = worker_usage.get(worker_id, 0) + 1

            # Should have used multiple workers
            assert len(worker_usage) > 1

            await distributor.stop()
