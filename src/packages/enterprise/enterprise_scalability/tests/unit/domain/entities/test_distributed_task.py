"""
Unit tests for DistributedTask domain entities.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from enterprise_scalability.domain.entities.distributed_task import (
    DistributedTask, TaskBatch, TaskResult, ResourceRequirements,
    TaskStatus, TaskPriority, TaskType, ProcessingMode
)


class TestResourceRequirements:
    """Test cases for ResourceRequirements entity."""
    
    def test_resource_requirements_creation_defaults(self):
        """Test resource requirements creation with defaults."""
        resources = ResourceRequirements()
        
        assert resources.cpu_cores == 1.0
        assert resources.memory_gb == 1.0
        assert resources.gpu_count == 0
        assert resources.gpu_memory_gb is None
        assert resources.storage_gb == 1.0
        assert resources.network_mbps is None
        assert resources.node_selector == {}
        assert resources.tolerations == []
        assert resources.affinity_rules == []
        
    def test_resource_requirements_creation_comprehensive(self):
        """Test comprehensive resource requirements creation."""
        node_selector = {"gpu": "true", "zone": "us-west-1a"}
        tolerations = [{"key": "gpu", "operator": "Equal", "value": "true", "effect": "NoSchedule"}]
        affinity_rules = [{"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": []}}]
        
        resources = ResourceRequirements(
            cpu_cores=8.0,
            memory_gb=32.0,
            gpu_count=2,
            gpu_memory_gb=16.0,
            storage_gb=100.0,
            network_mbps=1000.0,
            node_selector=node_selector,
            tolerations=tolerations,
            affinity_rules=affinity_rules
        )
        
        assert resources.cpu_cores == 8.0
        assert resources.memory_gb == 32.0
        assert resources.gpu_count == 2
        assert resources.gpu_memory_gb == 16.0
        assert resources.storage_gb == 100.0
        assert resources.network_mbps == 1000.0
        assert resources.node_selector == node_selector
        assert resources.tolerations == tolerations
        assert resources.affinity_rules == affinity_rules


class TestTaskResult:
    """Test cases for TaskResult entity."""
    
    def test_task_result_creation_success(self):
        """Test successful task result creation."""
        return_value = {"prediction": 0.95, "confidence": 0.87}
        artifacts = ["model.pkl", "metrics.json"]
        logs = ["Starting processing", "Processing complete"]
        
        result = TaskResult(
            success=True,
            return_value=return_value,
            execution_time_seconds=120.5,
            cpu_time_seconds=115.2,
            memory_peak_gb=2.5,
            disk_io_gb=0.5,
            network_io_gb=0.1,
            output_size_bytes=1048576,
            artifacts_created=artifacts,
            logs=logs,
            computed_by="worker-node-001"
        )
        
        assert result.success is True
        assert result.return_value == return_value
        assert result.error_message is None
        assert result.error_traceback is None
        assert result.execution_time_seconds == 120.5
        assert result.cpu_time_seconds == 115.2
        assert result.memory_peak_gb == 2.5
        assert result.disk_io_gb == 0.5
        assert result.network_io_gb == 0.1
        assert result.output_size_bytes == 1048576
        assert result.artifacts_created == artifacts
        assert result.logs == logs
        assert result.computed_by == "worker-node-001"
        assert isinstance(result.computed_at, datetime)
        
    def test_task_result_creation_failure(self):
        """Test failed task result creation."""
        error_msg = "Division by zero"
        traceback = "Traceback (most recent call last):\n  File \"script.py\", line 10, in func\n    return x / y\nZeroDivisionError: division by zero"
        
        result = TaskResult(
            success=False,
            error_message=error_msg,
            error_traceback=traceback,
            execution_time_seconds=0.5,
            computed_by="worker-node-002"
        )
        
        assert result.success is False
        assert result.return_value is None
        assert result.error_message == error_msg
        assert result.error_traceback == traceback
        assert result.execution_time_seconds == 0.5
        assert result.computed_by == "worker-node-002"
        
    def test_task_result_defaults(self):
        """Test task result with default values."""
        result = TaskResult(success=True)
        
        assert result.success is True
        assert result.return_value is None
        assert result.error_message is None
        assert result.error_traceback is None
        assert result.execution_time_seconds == 0.0
        assert result.cpu_time_seconds == 0.0
        assert result.memory_peak_gb == 0.0
        assert result.disk_io_gb == 0.0
        assert result.network_io_gb == 0.0
        assert result.output_size_bytes == 0
        assert result.artifacts_created == []
        assert result.logs == []
        assert result.computed_by is None


class TestDistributedTask:
    """Test cases for DistributedTask entity."""
    
    def test_distributed_task_creation_basic(self):
        """Test basic distributed task creation."""
        tenant_id = uuid4()
        user_id = uuid4()
        
        task = DistributedTask(
            name="data-processing-task",
            task_type=TaskType.BATCH_PROCESSING,
            function_name="process_data",
            module_name="data_processor",
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        assert isinstance(task.id, UUID)
        assert task.name == "data-processing-task"
        assert task.task_type == TaskType.BATCH_PROCESSING
        assert task.function_name == "process_data"
        assert task.module_name == "data_processor"
        assert task.description == ""
        assert task.tenant_id == tenant_id
        assert task.user_id == user_id
        assert task.job_id is None
        assert task.workflow_id is None
        assert task.function_args == []
        assert task.function_kwargs == {}
        assert task.environment_vars == {}
        assert task.priority == TaskPriority.NORMAL
        assert task.timeout_seconds is None
        assert task.max_retries == 3
        assert task.retry_delay_seconds == 60
        assert isinstance(task.resources, ResourceRequirements)
        assert task.depends_on == []
        assert task.blocks == []
        assert task.status == TaskStatus.PENDING
        assert task.current_retry == 0
        assert task.assigned_node is None
        assert task.cluster_id is None
        assert isinstance(task.submitted_at, datetime)
        assert task.scheduled_at is None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.result is None
        assert task.progress_percent == 0.0
        assert task.tags == {}
        assert task.labels == {}
        
    def test_distributed_task_creation_comprehensive(self):
        """Test comprehensive distributed task creation."""
        tenant_id = uuid4()
        user_id = uuid4()
        job_id = uuid4()
        workflow_id = uuid4()
        cluster_id = uuid4()
        function_args = ["input.csv", "--batch-size", "1000"]
        function_kwargs = {"output_format": "parquet", "compression": "snappy"}
        environment_vars = {"PYTHONPATH": "/app", "LOG_LEVEL": "INFO"}
        depends_on = [uuid4(), uuid4()]
        blocks = [uuid4()]
        resources = ResourceRequirements(cpu_cores=4.0, memory_gb=16.0, gpu_count=1)
        tags = {"project": "ml-pipeline", "stage": "preprocessing"}
        labels = {"priority": "high", "team": "data-science"}
        
        task = DistributedTask(
            name="feature-engineering-task",
            task_type=TaskType.FEATURE_ENGINEERING,
            function_name="extract_features",
            module_name="feature_extractor",
            description="Extract features from raw data",
            tenant_id=tenant_id,
            user_id=user_id,
            job_id=job_id,
            workflow_id=workflow_id,
            function_args=function_args,
            function_kwargs=function_kwargs,
            environment_vars=environment_vars,
            priority=TaskPriority.HIGH,
            timeout_seconds=3600,
            max_retries=5,
            retry_delay_seconds=120,
            resources=resources,
            depends_on=depends_on,
            blocks=blocks,
            status=TaskStatus.QUEUED,
            current_retry=1,
            assigned_node="worker-node-003",
            cluster_id=cluster_id,
            progress_percent=25.0,
            tags=tags,
            labels=labels
        )
        
        assert task.description == "Extract features from raw data"
        assert task.job_id == job_id
        assert task.workflow_id == workflow_id
        assert task.function_args == function_args
        assert task.function_kwargs == function_kwargs
        assert task.environment_vars == environment_vars
        assert task.priority == TaskPriority.HIGH
        assert task.timeout_seconds == 3600
        assert task.max_retries == 5
        assert task.retry_delay_seconds == 120
        assert task.resources == resources
        assert task.depends_on == depends_on
        assert task.blocks == blocks
        assert task.status == TaskStatus.QUEUED
        assert task.current_retry == 1
        assert task.assigned_node == "worker-node-003"
        assert task.cluster_id == cluster_id
        assert task.progress_percent == 25.0
        assert task.tags == tags
        assert task.labels == labels
        
    def test_max_retries_validation(self):
        """Test max_retries validation."""
        with pytest.raises(ValueError, match="max_retries cannot exceed 10"):
            DistributedTask(
                name="test-task",
                task_type=TaskType.CUSTOM,
                function_name="test_func",
                module_name="test_module",
                tenant_id=uuid4(),
                user_id=uuid4(),
                max_retries=15  # Exceeds limit
            )
            
    def test_is_ready_to_run_true(self):
        """Test is_ready_to_run returns True for pending task with no dependencies."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.PENDING,
            depends_on=[]
        )
        
        assert task.is_ready_to_run() is True
        
    def test_is_ready_to_run_false_has_dependencies(self):
        """Test is_ready_to_run returns False with pending dependencies."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.PENDING,
            depends_on=[uuid4(), uuid4()]
        )
        
        assert task.is_ready_to_run() is False
        
    def test_is_ready_to_run_false_not_pending(self):
        """Test is_ready_to_run returns False for non-pending task."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.RUNNING,
            depends_on=[]
        )
        
        assert task.is_ready_to_run() is False
        
    def test_is_running(self):
        """Test is_running method."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.RUNNING
        )
        
        assert task.is_running() is True
        
        task.status = TaskStatus.PENDING
        assert task.is_running() is False
        
    def test_is_completed(self):
        """Test is_completed method."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.COMPLETED
        )
        
        assert task.is_completed() is True
        
        task.status = TaskStatus.RUNNING
        assert task.is_completed() is False
        
    def test_is_failed(self):
        """Test is_failed method."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        # Test all failed statuses
        for status in [TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]:
            task.status = status
            assert task.is_failed() is True
            
        # Test non-failed status
        task.status = TaskStatus.RUNNING
        assert task.is_failed() is False
        
    def test_can_retry_true(self):
        """Test can_retry returns True when retries available."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.FAILED,
            current_retry=2,
            max_retries=3
        )
        
        assert task.can_retry() is True
        
    def test_can_retry_false_max_retries_reached(self):
        """Test can_retry returns False when max retries reached."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.FAILED,
            current_retry=3,
            max_retries=3
        )
        
        assert task.can_retry() is False
        
    def test_can_retry_false_not_failed(self):
        """Test can_retry returns False for non-failed task."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.RUNNING,
            current_retry=1,
            max_retries=3
        )
        
        assert task.can_retry() is False
        
    def test_get_execution_time_with_times(self):
        """Test get_execution_time with start and completion times."""
        start_time = datetime.utcnow() - timedelta(minutes=5)
        end_time = datetime.utcnow()
        
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            started_at=start_time,
            completed_at=end_time
        )
        
        execution_time = task.get_execution_time()
        expected_time = (end_time - start_time).total_seconds()
        
        assert execution_time == pytest.approx(expected_time, rel=1e-3)
        
    def test_get_execution_time_without_times(self):
        """Test get_execution_time without completion time."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            started_at=datetime.utcnow()
        )
        
        assert task.get_execution_time() is None
        
    def test_get_wait_time_with_scheduled(self):
        """Test get_wait_time with scheduled time."""
        submitted_time = datetime.utcnow() - timedelta(minutes=10)
        scheduled_time = datetime.utcnow() - timedelta(minutes=5)
        
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            submitted_at=submitted_time,
            scheduled_at=scheduled_time
        )
        
        wait_time = task.get_wait_time()
        expected_time = (scheduled_time - submitted_time).total_seconds()
        
        assert wait_time == pytest.approx(expected_time, rel=1e-3)
        
    def test_get_wait_time_without_scheduled(self):
        """Test get_wait_time without scheduled time."""
        submitted_time = datetime.utcnow() - timedelta(minutes=10)
        
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            submitted_at=submitted_time
        )
        
        wait_time = task.get_wait_time()
        expected_time = (datetime.utcnow() - submitted_time).total_seconds()
        
        assert wait_time == pytest.approx(expected_time, rel=1e-1)
        
    def test_schedule(self):
        """Test scheduling task for execution."""
        cluster_id = uuid4()
        node_id = "worker-node-001"
        
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.PENDING
        )
        
        task.schedule(cluster_id, node_id)
        
        assert task.status == TaskStatus.QUEUED
        assert task.cluster_id == cluster_id
        assert task.assigned_node == node_id
        assert task.scheduled_at is not None
        
    def test_start(self):
        """Test starting task execution."""
        node_id = "worker-node-002"
        
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.QUEUED
        )
        
        task.start(node_id)
        
        assert task.status == TaskStatus.RUNNING
        assert task.assigned_node == node_id
        assert task.started_at is not None
        
    def test_complete(self):
        """Test completing task with result."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.RUNNING
        )
        
        result = TaskResult(success=True, return_value={"result": "success"})
        
        task.complete(result)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.result == result
        assert task.progress_percent == 100.0
        
    def test_fail(self):
        """Test failing task with error."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.RUNNING,
            assigned_node="worker-node-001"
        )
        
        error_msg = "Processing failed"
        traceback = "Traceback..."
        
        task.fail(error_msg, traceback)
        
        assert task.status == TaskStatus.FAILED
        assert task.completed_at is not None
        assert task.result is not None
        assert task.result.success is False
        assert task.result.error_message == error_msg
        assert task.result.error_traceback == traceback
        assert task.result.computed_by == "worker-node-001"
        
    def test_retry(self):
        """Test retrying failed task."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.FAILED,
            current_retry=1,
            max_retries=3,
            assigned_node="worker-node-001",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            result=TaskResult(success=False, error_message="Previous error")
        )
        
        task.retry()
        
        assert task.current_retry == 2
        assert task.status == TaskStatus.RETRY
        assert task.assigned_node is None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.result is None
        
    def test_retry_not_allowed(self):
        """Test retry when not allowed."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.FAILED,
            current_retry=3,
            max_retries=3
        )
        
        with pytest.raises(ValueError, match="Task cannot be retried"):
            task.retry()
            
    def test_cancel(self):
        """Test cancelling task."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.RUNNING
        )
        
        task.cancel()
        
        assert task.status == TaskStatus.CANCELLED
        assert task.completed_at is not None
        
    def test_timeout(self):
        """Test timing out task."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.RUNNING,
            assigned_node="worker-node-001"
        )
        
        task.timeout()
        
        assert task.status == TaskStatus.TIMEOUT
        assert task.completed_at is not None
        assert task.result is not None
        assert task.result.success is False
        assert task.result.error_message == "Task execution timed out"
        assert task.result.computed_by == "worker-node-001"
        
    def test_update_progress(self):
        """Test updating task progress."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        task.update_progress(75.5)
        assert task.progress_percent == 75.5
        
        # Test boundary clamping
        task.update_progress(150.0)
        assert task.progress_percent == 100.0
        
        task.update_progress(-10.0)
        assert task.progress_percent == 0.0
        
    def test_add_dependency(self):
        """Test adding task dependency."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        dep_id = uuid4()
        task.add_dependency(dep_id)
        
        assert dep_id in task.depends_on
        assert len(task.depends_on) == 1
        
        # Adding same dependency again should not duplicate
        task.add_dependency(dep_id)
        assert len(task.depends_on) == 1
        
    def test_remove_dependency(self):
        """Test removing task dependency."""
        dep_id = uuid4()
        
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            depends_on=[dep_id]
        )
        
        task.remove_dependency(dep_id)
        
        assert dep_id not in task.depends_on
        assert len(task.depends_on) == 0
        
        # Removing non-existent dependency should not error
        task.remove_dependency(uuid4())
        
    def test_get_task_summary(self):
        """Test getting task summary."""
        submitted_time = datetime.utcnow() - timedelta(minutes=10)
        started_time = datetime.utcnow() - timedelta(minutes=5)
        completed_time = datetime.utcnow()
        
        task = DistributedTask(
            name="ml-training-task",
            task_type=TaskType.MODEL_TRAINING,
            function_name="train_model",
            module_name="ml_trainer",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.COMPLETED,
            priority=TaskPriority.HIGH,
            progress_percent=100.0,
            depends_on=[uuid4(), uuid4()],
            current_retry=1,
            resources=ResourceRequirements(cpu_cores=4.0, memory_gb=16.0, gpu_count=1),
            submitted_at=submitted_time,
            started_at=started_time,
            completed_at=completed_time,
            result=TaskResult(
                success=True,
                execution_time_seconds=300.0,
                memory_peak_gb=12.5,
                cpu_time_seconds=280.0
            )
        )
        
        summary = task.get_task_summary()
        
        assert summary["id"] == str(task.id)
        assert summary["name"] == "ml-training-task"
        assert summary["type"] == TaskType.MODEL_TRAINING
        assert summary["status"] == TaskStatus.COMPLETED
        assert summary["priority"] == TaskPriority.HIGH
        assert summary["progress"] == 100.0
        assert summary["dependencies"] == 2
        assert summary["retries"] == 1
        assert summary["timing"]["submitted_at"] == submitted_time.isoformat()
        assert "wait_time_seconds" in summary["timing"]
        assert "execution_time_seconds" in summary["timing"]
        assert summary["resources"]["cpu_cores"] == 4.0
        assert summary["resources"]["memory_gb"] == 16.0
        assert summary["resources"]["gpu_count"] == 1
        assert summary["result"]["success"] is True
        assert summary["result"]["execution_time"] == 300.0
        assert summary["result"]["memory_peak"] == 12.5
        assert summary["result"]["cpu_time"] == 280.0
        
    def test_get_task_summary_no_result(self):
        """Test task summary without result."""
        task = DistributedTask(
            name="test-task",
            task_type=TaskType.CUSTOM,
            function_name="test_func",
            module_name="test_module",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status=TaskStatus.RUNNING
        )
        
        summary = task.get_task_summary()
        
        assert "result" not in summary


class TestTaskBatch:
    """Test cases for TaskBatch entity."""
    
    def test_task_batch_creation_basic(self):
        """Test basic task batch creation."""
        tenant_id = uuid4()
        user_id = uuid4()
        
        batch = TaskBatch(
            name="data-processing-batch",
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        assert isinstance(batch.id, UUID)
        assert batch.name == "data-processing-batch"
        assert batch.description == ""
        assert batch.batch_type == "general"
        assert batch.tenant_id == tenant_id
        assert batch.user_id == user_id
        assert batch.task_ids == []
        assert batch.total_tasks == 0
        assert batch.max_concurrent_tasks == 10
        assert batch.batch_timeout_seconds is None
        assert batch.stop_on_first_failure is False
        assert batch.tasks_pending == 0
        assert batch.tasks_running == 0
        assert batch.tasks_completed == 0
        assert batch.tasks_failed == 0
        assert batch.tasks_cancelled == 0
        assert batch.status == "pending"
        assert batch.progress_percent == 0.0
        assert isinstance(batch.created_at, datetime)
        assert batch.started_at is None
        assert batch.completed_at is None
        assert batch.tags == {}
        
    def test_task_batch_creation_comprehensive(self):
        """Test comprehensive task batch creation."""
        tenant_id = uuid4()
        user_id = uuid4()
        task_ids = [uuid4(), uuid4(), uuid4()]
        tags = {"project": "ml-pipeline", "environment": "production"}
        
        batch = TaskBatch(
            name="ml-training-batch",
            description="Batch training of multiple models",
            batch_type="ml_training",
            tenant_id=tenant_id,
            user_id=user_id,
            task_ids=task_ids,
            total_tasks=3,
            max_concurrent_tasks=2,
            batch_timeout_seconds=7200,
            stop_on_first_failure=True,
            tasks_pending=1,
            tasks_running=1,
            tasks_completed=1,
            tasks_failed=0,
            tasks_cancelled=0,
            status="running",
            progress_percent=66.7,
            tags=tags
        )
        
        assert batch.description == "Batch training of multiple models"
        assert batch.batch_type == "ml_training"
        assert batch.task_ids == task_ids
        assert batch.total_tasks == 3
        assert batch.max_concurrent_tasks == 2
        assert batch.batch_timeout_seconds == 7200
        assert batch.stop_on_first_failure is True
        assert batch.tasks_pending == 1
        assert batch.tasks_running == 1
        assert batch.tasks_completed == 1
        assert batch.tasks_failed == 0
        assert batch.tasks_cancelled == 0
        assert batch.status == "running"
        assert batch.progress_percent == 66.7
        assert batch.tags == tags
        
    def test_update_task_counts(self):
        """Test updating task counts from task list."""
        batch = TaskBatch(
            name="test-batch",
            tenant_id=uuid4(),
            user_id=uuid4(),
            total_tasks=5
        )
        
        tasks = [
            DistributedTask(
                name="task-1",
                task_type=TaskType.CUSTOM,
                function_name="func1",
                module_name="mod1",
                tenant_id=uuid4(),
                user_id=uuid4(),
                status=TaskStatus.PENDING
            ),
            DistributedTask(
                name="task-2",
                task_type=TaskType.CUSTOM,
                function_name="func2",
                module_name="mod2",
                tenant_id=uuid4(),
                user_id=uuid4(),
                status=TaskStatus.RUNNING
            ),
            DistributedTask(
                name="task-3",
                task_type=TaskType.CUSTOM,
                function_name="func3",
                module_name="mod3",
                tenant_id=uuid4(),
                user_id=uuid4(),
                status=TaskStatus.COMPLETED
            ),
            DistributedTask(
                name="task-4",
                task_type=TaskType.CUSTOM,
                function_name="func4",
                module_name="mod4",
                tenant_id=uuid4(),
                user_id=uuid4(),
                status=TaskStatus.FAILED
            ),
            DistributedTask(
                name="task-5",
                task_type=TaskType.CUSTOM,
                function_name="func5",
                module_name="mod5",
                tenant_id=uuid4(),
                user_id=uuid4(),
                status=TaskStatus.CANCELLED
            )
        ]
        
        batch.update_task_counts(tasks)
        
        assert batch.tasks_pending == 1
        assert batch.tasks_running == 1
        assert batch.tasks_completed == 1
        assert batch.tasks_failed == 2  # FAILED + CANCELLED
        assert batch.progress_percent == 60.0  # (1+2)/5 * 100
        assert batch.status == "running"  # Has running tasks
        
    def test_update_task_counts_all_completed(self):
        """Test task count update with all tasks completed."""
        batch = TaskBatch(
            name="test-batch",
            tenant_id=uuid4(),
            user_id=uuid4(),
            total_tasks=2
        )
        
        tasks = [
            DistributedTask(
                name="task-1",
                task_type=TaskType.CUSTOM,
                function_name="func1",
                module_name="mod1",
                tenant_id=uuid4(),
                user_id=uuid4(),
                status=TaskStatus.COMPLETED
            ),
            DistributedTask(
                name="task-2",
                task_type=TaskType.CUSTOM,
                function_name="func2",
                module_name="mod2",
                tenant_id=uuid4(),
                user_id=uuid4(),
                status=TaskStatus.COMPLETED
            )
        ]
        
        batch.update_task_counts(tasks)
        
        assert batch.tasks_completed == 2
        assert batch.progress_percent == 100.0
        assert batch.status == "completed"
        assert batch.completed_at is not None
        
    def test_update_task_counts_stop_on_failure(self):
        """Test task count update with stop on first failure."""
        batch = TaskBatch(
            name="test-batch",
            tenant_id=uuid4(),
            user_id=uuid4(),
            total_tasks=3,
            stop_on_first_failure=True
        )
        
        tasks = [
            DistributedTask(
                name="task-1",
                task_type=TaskType.CUSTOM,
                function_name="func1",
                module_name="mod1",
                tenant_id=uuid4(),
                user_id=uuid4(),
                status=TaskStatus.COMPLETED
            ),
            DistributedTask(
                name="task-2",
                task_type=TaskType.CUSTOM,
                function_name="func2",
                module_name="mod2",
                tenant_id=uuid4(),
                user_id=uuid4(),
                status=TaskStatus.FAILED
            ),
            DistributedTask(
                name="task-3",
                task_type=TaskType.CUSTOM,
                function_name="func3",
                module_name="mod3",
                tenant_id=uuid4(),
                user_id=uuid4(),
                status=TaskStatus.PENDING
            )
        ]
        
        batch.update_task_counts(tasks)
        
        assert batch.tasks_failed == 1
        assert batch.status == "failed"
        assert batch.completed_at is not None
        
    def test_get_success_rate(self):
        """Test calculating batch success rate."""
        batch = TaskBatch(
            name="test-batch",
            tenant_id=uuid4(),
            user_id=uuid4(),
            tasks_completed=8,
            tasks_failed=2
        )
        
        success_rate = batch.get_success_rate()
        assert success_rate == 80.0  # 8/(8+2) * 100
        
    def test_get_success_rate_no_completed_tasks(self):
        """Test success rate with no completed tasks."""
        batch = TaskBatch(
            name="test-batch",
            tenant_id=uuid4(),
            user_id=uuid4(),
            tasks_completed=0,
            tasks_failed=0
        )
        
        success_rate = batch.get_success_rate()
        assert success_rate == 0.0
        
    def test_is_completed(self):
        """Test is_completed method."""
        batch = TaskBatch(
            name="test-batch",
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        batch.status = "completed"
        assert batch.is_completed() is True
        
        batch.status = "failed"
        assert batch.is_completed() is True
        
        batch.status = "running"
        assert batch.is_completed() is False
        
    def test_get_batch_summary(self):
        """Test getting batch summary."""
        created_time = datetime.utcnow() - timedelta(hours=2)
        started_time = datetime.utcnow() - timedelta(hours=1, minutes=30)
        completed_time = datetime.utcnow()
        
        batch = TaskBatch(
            name="data-processing-batch",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status="completed",
            progress_percent=100.0,
            total_tasks=10,
            tasks_pending=0,
            tasks_running=0,
            tasks_completed=8,
            tasks_failed=2,
            max_concurrent_tasks=3,
            stop_on_first_failure=False,
            created_at=created_time,
            started_at=started_time,
            completed_at=completed_time
        )
        
        summary = batch.get_batch_summary()
        
        assert summary["id"] == str(batch.id)
        assert summary["name"] == "data-processing-batch"
        assert summary["status"] == "completed"
        assert summary["progress"] == 100.0
        assert summary["tasks"]["total"] == 10
        assert summary["tasks"]["pending"] == 0
        assert summary["tasks"]["running"] == 0
        assert summary["tasks"]["completed"] == 8
        assert summary["tasks"]["failed"] == 2
        assert summary["tasks"]["success_rate"] == 80.0
        assert summary["timing"]["created_at"] == created_time.isoformat()
        assert summary["timing"]["started_at"] == started_time.isoformat()
        assert summary["timing"]["completed_at"] == completed_time.isoformat()
        assert "execution_time_seconds" in summary["timing"]
        assert summary["configuration"]["max_concurrent"] == 3
        assert summary["configuration"]["stop_on_failure"] is False
        
    def test_get_batch_summary_not_started(self):
        """Test batch summary when not started."""
        batch = TaskBatch(
            name="test-batch",
            tenant_id=uuid4(),
            user_id=uuid4(),
            status="pending"
        )
        
        summary = batch.get_batch_summary()
        
        assert summary["timing"]["started_at"] is None
        assert summary["timing"]["completed_at"] is None
        assert summary["timing"]["execution_time_seconds"] is None