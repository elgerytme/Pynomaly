"""Comprehensive unit tests for PipelineOrchestrator domain entity."""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import patch

from data_pipelines.domain.entities.pipeline_orchestrator import (
    PipelineOrchestrator, OrchestrationStatus, ExecutionStrategy, ExecutionMetrics,
    PipelineExecutionMode
)


class TestOrchestrationStatus:
    """Test cases for OrchestrationStatus enum."""

    def test_orchestration_status_values(self):
        """Test all OrchestrationStatus enum values."""
        assert OrchestrationStatus.IDLE.value == "idle"
        assert OrchestrationStatus.SCHEDULING.value == "scheduling"
        assert OrchestrationStatus.RUNNING.value == "running"
        assert OrchestrationStatus.PAUSED.value == "paused"
        assert OrchestrationStatus.STOPPED.value == "stopped"
        assert OrchestrationStatus.ERROR.value == "error"
        assert OrchestrationStatus.MAINTENANCE.value == "maintenance"


class TestExecutionStrategy:
    """Test cases for ExecutionStrategy enum."""

    def test_execution_strategy_values(self):
        """Test all ExecutionStrategy enum values."""
        assert ExecutionStrategy.SEQUENTIAL.value == "sequential"
        assert ExecutionStrategy.PARALLEL.value == "parallel"
        assert ExecutionStrategy.CONDITIONAL.value == "conditional"
        assert ExecutionStrategy.BATCH.value == "batch"
        assert ExecutionStrategy.STREAMING.value == "streaming"
        assert ExecutionStrategy.HYBRID.value == "hybrid"


class TestPipelineExecutionMode:
    """Test cases for PipelineExecutionMode enum."""

    def test_pipeline_execution_mode_values(self):
        """Test all PipelineExecutionMode enum values."""
        assert PipelineExecutionMode.IMMEDIATE.value == "immediate"
        assert PipelineExecutionMode.SCHEDULED.value == "scheduled"
        assert PipelineExecutionMode.TRIGGERED.value == "triggered"
        assert PipelineExecutionMode.MANUAL.value == "manual"


class TestExecutionMetrics:
    """Test cases for ExecutionMetrics entity."""

    def test_initialization_defaults(self):
        """Test metrics initialization with defaults."""
        metrics = ExecutionMetrics()
        
        assert metrics.total_executions == 0
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 0
        assert metrics.cancelled_executions == 0
        assert metrics.average_duration_seconds == 0.0
        assert metrics.min_duration_seconds == 0.0
        assert metrics.max_duration_seconds == 0.0
        assert metrics.average_cpu_usage == 0.0
        assert metrics.average_memory_usage_mb == 0.0
        assert metrics.peak_memory_usage_mb == 0.0
        assert metrics.average_queue_time_seconds == 0.0
        assert metrics.max_queue_time_seconds == 0.0

    def test_initialization_with_data(self):
        """Test metrics initialization with provided data."""
        metrics = ExecutionMetrics(
            total_executions=10,
            successful_executions=8,
            failed_executions=2,
            cancelled_executions=1,
            average_duration_seconds=120.5,
            min_duration_seconds=45.0,
            max_duration_seconds=300.0,
            average_cpu_usage=0.75,
            average_memory_usage_mb=512.0,
            peak_memory_usage_mb=1024.0
        )
        
        assert metrics.total_executions == 10
        assert metrics.successful_executions == 8
        assert metrics.failed_executions == 2
        assert metrics.cancelled_executions == 1
        assert metrics.average_duration_seconds == 120.5
        assert metrics.min_duration_seconds == 45.0
        assert metrics.max_duration_seconds == 300.0
        assert metrics.average_cpu_usage == 0.75
        assert metrics.average_memory_usage_mb == 512.0
        assert metrics.peak_memory_usage_mb == 1024.0

    def test_post_init_validation_negative_total(self):
        """Test validation fails for negative total executions."""
        with pytest.raises(ValueError, match="Total executions cannot be negative"):
            ExecutionMetrics(total_executions=-1)

    def test_post_init_validation_negative_successful(self):
        """Test validation fails for negative successful executions."""
        with pytest.raises(ValueError, match="Successful executions cannot be negative"):
            ExecutionMetrics(successful_executions=-1)

    def test_post_init_validation_negative_failed(self):
        """Test validation fails for negative failed executions."""
        with pytest.raises(ValueError, match="Failed executions cannot be negative"):
            ExecutionMetrics(failed_executions=-1)

    def test_success_rate_property_zero_executions(self):
        """Test success rate property with zero executions."""
        metrics = ExecutionMetrics()
        assert metrics.success_rate == 0.0

    def test_success_rate_property_calculated(self):
        """Test success rate property calculation."""
        metrics = ExecutionMetrics(total_executions=10, successful_executions=8)
        assert metrics.success_rate == 80.0

    def test_failure_rate_property_zero_executions(self):
        """Test failure rate property with zero executions."""
        metrics = ExecutionMetrics()
        assert metrics.failure_rate == 0.0

    def test_failure_rate_property_calculated(self):
        """Test failure rate property calculation."""
        metrics = ExecutionMetrics(total_executions=10, failed_executions=3)
        assert metrics.failure_rate == 30.0

    def test_update_execution_first_execution_success(self):
        """Test updating metrics with first successful execution."""
        metrics = ExecutionMetrics()
        metrics.update_execution(True, 120.0, 0.8, 512.0)
        
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.failed_executions == 0
        assert metrics.average_duration_seconds == 120.0
        assert metrics.min_duration_seconds == 120.0
        assert metrics.max_duration_seconds == 120.0
        assert metrics.average_cpu_usage == 0.8
        assert metrics.average_memory_usage_mb == 512.0
        assert metrics.peak_memory_usage_mb == 512.0

    def test_update_execution_first_execution_failure(self):
        """Test updating metrics with first failed execution."""
        metrics = ExecutionMetrics()
        metrics.update_execution(False, 60.0)
        
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 1
        assert metrics.average_duration_seconds == 60.0

    def test_update_execution_multiple_executions(self):
        """Test updating metrics with multiple executions."""
        metrics = ExecutionMetrics()
        
        # First execution
        metrics.update_execution(True, 100.0, 0.7, 400.0)
        # Second execution
        metrics.update_execution(False, 200.0, 0.9, 600.0)
        # Third execution
        metrics.update_execution(True, 150.0, 0.8, 500.0)
        
        assert metrics.total_executions == 3
        assert metrics.successful_executions == 2
        assert metrics.failed_executions == 1
        assert metrics.success_rate == 2/3 * 100
        assert metrics.failure_rate == 1/3 * 100
        
        # Check averages
        expected_avg_duration = (100.0 + 200.0 + 150.0) / 3
        assert metrics.average_duration_seconds == expected_avg_duration
        assert metrics.min_duration_seconds == 100.0
        assert metrics.max_duration_seconds == 200.0
        
        expected_avg_cpu = (0.7 + 0.9 + 0.8) / 3
        assert abs(metrics.average_cpu_usage - expected_avg_cpu) < 0.01
        
        expected_avg_memory = (400.0 + 600.0 + 500.0) / 3
        assert abs(metrics.average_memory_usage_mb - expected_avg_memory) < 0.01
        assert metrics.peak_memory_usage_mb == 600.0

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ExecutionMetrics(
            total_executions=5,
            successful_executions=4,
            failed_executions=1,
            average_duration_seconds=125.5
        )
        
        result = metrics.to_dict()
        
        assert result["total_executions"] == 5
        assert result["successful_executions"] == 4
        assert result["failed_executions"] == 1
        assert result["success_rate"] == 80.0
        assert result["failure_rate"] == 20.0
        assert result["average_duration_seconds"] == 125.5


class TestPipelineOrchestrator:
    """Test cases for PipelineOrchestrator entity."""

    def test_initialization_defaults(self):
        """Test orchestrator initialization with defaults."""
        orchestrator = PipelineOrchestrator(name="test_orchestrator")
        
        assert isinstance(orchestrator.id, UUID)
        assert orchestrator.name == "test_orchestrator"
        assert orchestrator.description == ""
        assert orchestrator.version == "1.0.0"
        assert orchestrator.status == OrchestrationStatus.IDLE
        assert orchestrator.execution_strategy == ExecutionStrategy.SEQUENTIAL
        assert orchestrator.execution_mode == PipelineExecutionMode.SCHEDULED
        assert orchestrator.registered_pipelines == []
        assert orchestrator.active_executions == {}
        assert orchestrator.execution_queue == []
        assert orchestrator.max_concurrent_executions == 5
        assert orchestrator.max_queue_size == 100
        assert orchestrator.worker_pool_size == 10
        assert orchestrator.memory_limit_mb is None
        assert orchestrator.cpu_limit_cores is None
        assert orchestrator.schedule_enabled is True
        assert orchestrator.schedule_interval_minutes == 60
        assert orchestrator.monitoring_enabled is True
        assert isinstance(orchestrator.metrics, ExecutionMetrics)
        assert orchestrator.last_execution_at is None
        assert orchestrator.next_scheduled_at is None
        assert isinstance(orchestrator.created_at, datetime)
        assert orchestrator.created_by == ""
        assert isinstance(orchestrator.updated_at, datetime)
        assert orchestrator.updated_by == ""
        assert orchestrator.config == {}
        assert orchestrator.environment_vars == {}
        assert orchestrator.tags == []

    def test_initialization_with_data(self):
        """Test orchestrator initialization with provided data."""
        orchestrator_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 0, 0)
        config = {"log_level": "INFO"}
        env_vars = {"ENV": "production"}
        tags = ["production", "critical"]
        
        orchestrator = PipelineOrchestrator(
            id=orchestrator_id,
            name="Production Orchestrator",
            description="Main production orchestrator",
            version="2.0.0",
            status=OrchestrationStatus.RUNNING,
            execution_strategy=ExecutionStrategy.PARALLEL,
            execution_mode=PipelineExecutionMode.TRIGGERED,
            max_concurrent_executions=20,
            max_queue_size=200,
            worker_pool_size=50,
            memory_limit_mb=8192,
            cpu_limit_cores=16.0,
            schedule_enabled=False,
            schedule_interval_minutes=30,
            monitoring_enabled=True,
            created_at=created_at,
            created_by="admin",
            updated_at=created_at,
            updated_by="admin",
            config=config,
            environment_vars=env_vars,
            tags=tags
        )
        
        assert orchestrator.id == orchestrator_id
        assert orchestrator.name == "Production Orchestrator"
        assert orchestrator.description == "Main production orchestrator"
        assert orchestrator.version == "2.0.0"
        assert orchestrator.status == OrchestrationStatus.RUNNING
        assert orchestrator.execution_strategy == ExecutionStrategy.PARALLEL
        assert orchestrator.execution_mode == PipelineExecutionMode.TRIGGERED
        assert orchestrator.max_concurrent_executions == 20
        assert orchestrator.max_queue_size == 200
        assert orchestrator.worker_pool_size == 50
        assert orchestrator.memory_limit_mb == 8192
        assert orchestrator.cpu_limit_cores == 16.0
        assert orchestrator.schedule_enabled is False
        assert orchestrator.schedule_interval_minutes == 30
        assert orchestrator.created_at == created_at
        assert orchestrator.created_by == "admin"
        assert orchestrator.config == config
        assert orchestrator.environment_vars == env_vars
        assert orchestrator.tags == tags

    def test_post_init_validation_empty_name(self):
        """Test validation fails for empty name."""
        with pytest.raises(ValueError, match="Orchestrator name cannot be empty"):
            PipelineOrchestrator(name="")

    def test_post_init_validation_name_too_long(self):
        """Test validation fails for name too long."""
        long_name = "a" * 101
        with pytest.raises(ValueError, match="Orchestrator name cannot exceed 100 characters"):
            PipelineOrchestrator(name=long_name)

    def test_post_init_validation_invalid_max_concurrent(self):
        """Test validation fails for invalid max concurrent executions."""
        with pytest.raises(ValueError, match="Max concurrent executions must be positive"):
            PipelineOrchestrator(name="test", max_concurrent_executions=0)

    def test_post_init_validation_invalid_max_queue_size(self):
        """Test validation fails for invalid max queue size."""
        with pytest.raises(ValueError, match="Max queue size must be positive"):
            PipelineOrchestrator(name="test", max_queue_size=-1)

    def test_post_init_validation_invalid_worker_pool_size(self):
        """Test validation fails for invalid worker pool size."""
        with pytest.raises(ValueError, match="Worker pool size must be positive"):
            PipelineOrchestrator(name="test", worker_pool_size=0)

    def test_post_init_validation_invalid_schedule_interval(self):
        """Test validation fails for invalid schedule interval."""
        with pytest.raises(ValueError, match="Schedule interval must be positive"):
            PipelineOrchestrator(name="test", schedule_interval_minutes=0)

    def test_is_running_property(self):
        """Test is_running property."""
        orchestrator = PipelineOrchestrator(name="test")
        assert orchestrator.is_running is False
        
        orchestrator.status = OrchestrationStatus.RUNNING
        assert orchestrator.is_running is True

    def test_is_available_property(self):
        """Test is_available property."""
        orchestrator = PipelineOrchestrator(name="test")
        
        # Test available statuses
        available_statuses = [
            OrchestrationStatus.IDLE,
            OrchestrationStatus.RUNNING,
            OrchestrationStatus.SCHEDULING
        ]
        
        for status in available_statuses:
            orchestrator.status = status
            assert orchestrator.is_available is True
        
        # Test unavailable statuses
        orchestrator.status = OrchestrationStatus.STOPPED
        assert orchestrator.is_available is False

    def test_current_queue_size_property(self):
        """Test current_queue_size property."""
        orchestrator = PipelineOrchestrator(name="test")
        assert orchestrator.current_queue_size == 0
        
        orchestrator.execution_queue = [{"id": "1"}, {"id": "2"}]
        assert orchestrator.current_queue_size == 2

    def test_current_active_executions_property(self):
        """Test current_active_executions property."""
        orchestrator = PipelineOrchestrator(name="test")
        assert orchestrator.current_active_executions == 0
        
        orchestrator.active_executions = {"exec1": {}, "exec2": {}}
        assert orchestrator.current_active_executions == 2

    def test_is_queue_full_property(self):
        """Test is_queue_full property."""
        orchestrator = PipelineOrchestrator(name="test", max_queue_size=2)
        assert orchestrator.is_queue_full is False
        
        orchestrator.execution_queue = [{"id": "1"}, {"id": "2"}]
        assert orchestrator.is_queue_full is True

    def test_can_accept_execution_property(self):
        """Test can_accept_execution property."""
        orchestrator = PipelineOrchestrator(name="test", max_concurrent_executions=2, max_queue_size=5)
        orchestrator.status = OrchestrationStatus.RUNNING
        
        # Should accept when available and not at capacity
        assert orchestrator.can_accept_execution is True
        
        # Fill active executions
        orchestrator.active_executions = {"exec1": {}, "exec2": {}}
        assert orchestrator.can_accept_execution is True  # Still has queue space
        
        # Fill queue
        orchestrator.execution_queue = [{"id": str(i)} for i in range(5)]
        assert orchestrator.can_accept_execution is False

    def test_resource_utilization_property(self):
        """Test resource_utilization property."""
        orchestrator = PipelineOrchestrator(name="test", max_concurrent_executions=10, max_queue_size=20)
        
        orchestrator.active_executions = {"exec1": {}, "exec2": {}}
        orchestrator.execution_queue = [{"id": str(i)} for i in range(5)]
        
        utilization = orchestrator.resource_utilization
        
        assert utilization["execution_utilization"] == 20.0  # 2/10 * 100
        assert utilization["queue_utilization"] == 25.0      # 5/20 * 100

    def test_register_pipeline(self):
        """Test registering a pipeline."""
        orchestrator = PipelineOrchestrator(name="test")
        pipeline_id = uuid4()
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.register_pipeline(pipeline_id)
            
            assert pipeline_id in orchestrator.registered_pipelines
            assert len(orchestrator.registered_pipelines) == 1
            assert orchestrator.updated_at == mock_now

    def test_register_pipeline_duplicate(self):
        """Test registering duplicate pipeline fails."""
        orchestrator = PipelineOrchestrator(name="test")
        pipeline_id = uuid4()
        
        orchestrator.register_pipeline(pipeline_id)
        
        with pytest.raises(ValueError, match=f"Pipeline {pipeline_id} is already registered"):
            orchestrator.register_pipeline(pipeline_id)

    def test_unregister_pipeline_success(self):
        """Test unregistering pipeline successfully."""
        orchestrator = PipelineOrchestrator(name="test")
        pipeline_id = uuid4()
        
        orchestrator.register_pipeline(pipeline_id)
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 13, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            result = orchestrator.unregister_pipeline(pipeline_id)
            
            assert result is True
            assert pipeline_id not in orchestrator.registered_pipelines
            assert orchestrator.updated_at == mock_now

    def test_unregister_pipeline_not_found(self):
        """Test unregistering non-existent pipeline."""
        orchestrator = PipelineOrchestrator(name="test")
        pipeline_id = uuid4()
        
        result = orchestrator.unregister_pipeline(pipeline_id)
        
        assert result is False

    def test_unregister_pipeline_currently_executing(self):
        """Test unregistering pipeline that is currently executing fails."""
        orchestrator = PipelineOrchestrator(name="test")
        pipeline_id = uuid4()
        
        orchestrator.register_pipeline(pipeline_id)
        orchestrator.active_executions["exec1"] = {"pipeline_id": pipeline_id}
        
        with pytest.raises(ValueError, match=f"Cannot unregister pipeline {pipeline_id} - currently executing"):
            orchestrator.unregister_pipeline(pipeline_id)

    def test_queue_execution_success(self):
        """Test queuing execution successfully."""
        orchestrator = PipelineOrchestrator(name="test")
        pipeline_id = uuid4()
        
        orchestrator.register_pipeline(pipeline_id)
        orchestrator.status = OrchestrationStatus.RUNNING
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 14, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            execution_config = {"batch_size": 1000, "priority": 3}
            execution_id = orchestrator.queue_execution(pipeline_id, execution_config)
            
            assert len(orchestrator.execution_queue) == 1
            
            queued_request = orchestrator.execution_queue[0]
            assert queued_request["execution_id"] == execution_id
            assert queued_request["pipeline_id"] == pipeline_id
            assert queued_request["queued_at"] == mock_now
            assert queued_request["config"] == execution_config
            assert queued_request["priority"] == 3
            assert queued_request["retry_count"] == 0
            assert orchestrator.updated_at == mock_now

    def test_queue_execution_with_priority_sorting(self):
        """Test queuing executions are sorted by priority."""
        orchestrator = PipelineOrchestrator(name="test")
        pipeline_id = uuid4()
        
        orchestrator.register_pipeline(pipeline_id)
        orchestrator.status = OrchestrationStatus.RUNNING
        
        # Queue executions with different priorities
        exec_id_1 = orchestrator.queue_execution(pipeline_id, {"priority": 5})
        exec_id_2 = orchestrator.queue_execution(pipeline_id, {"priority": 1})  # Higher priority
        exec_id_3 = orchestrator.queue_execution(pipeline_id, {"priority": 3})
        
        # Check they are sorted by priority (lower number = higher priority)
        priorities = [req["priority"] for req in orchestrator.execution_queue]
        assert priorities == [1, 3, 5]

    def test_queue_execution_unavailable_orchestrator(self):
        """Test queuing execution when orchestrator is unavailable."""
        orchestrator = PipelineOrchestrator(name="test")
        pipeline_id = uuid4()
        
        orchestrator.register_pipeline(pipeline_id)
        orchestrator.status = OrchestrationStatus.STOPPED  # Unavailable
        
        with pytest.raises(ValueError, match="Cannot accept new execution - orchestrator unavailable or at capacity"):
            orchestrator.queue_execution(pipeline_id)

    def test_queue_execution_unregistered_pipeline(self):
        """Test queuing execution for unregistered pipeline."""
        orchestrator = PipelineOrchestrator(name="test")
        pipeline_id = uuid4()
        
        orchestrator.status = OrchestrationStatus.RUNNING
        
        with pytest.raises(ValueError, match=f"Pipeline {pipeline_id} is not registered"):
            orchestrator.queue_execution(pipeline_id)

    def test_start_execution_success(self):
        """Test starting execution successfully."""
        orchestrator = PipelineOrchestrator(name="test")
        pipeline_id = uuid4()
        
        orchestrator.register_pipeline(pipeline_id)
        orchestrator.status = OrchestrationStatus.RUNNING
        
        execution_id = orchestrator.queue_execution(pipeline_id)
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 15, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            result = orchestrator.start_execution(execution_id)
            
            assert result is True
            assert len(orchestrator.execution_queue) == 0
            assert execution_id in orchestrator.active_executions
            
            execution_info = orchestrator.active_executions[execution_id]
            assert execution_info["started_at"] == mock_now
            assert execution_info["status"] == "running"
            assert orchestrator.last_execution_at == mock_now
            assert orchestrator.updated_at == mock_now

    def test_start_execution_not_in_queue(self):
        """Test starting execution not in queue."""
        orchestrator = PipelineOrchestrator(name="test")
        execution_id = "nonexistent"
        
        result = orchestrator.start_execution(execution_id)
        
        assert result is False

    def test_start_execution_at_capacity(self):
        """Test starting execution when at capacity."""
        orchestrator = PipelineOrchestrator(name="test", max_concurrent_executions=1)
        pipeline_id = uuid4()
        
        orchestrator.register_pipeline(pipeline_id)
        orchestrator.status = OrchestrationStatus.RUNNING
        
        # Fill capacity
        orchestrator.active_executions["existing"] = {}
        
        execution_id = orchestrator.queue_execution(pipeline_id)
        result = orchestrator.start_execution(execution_id)
        
        # Should put back in queue
        assert result is False
        assert len(orchestrator.execution_queue) == 1

    def test_complete_execution_success(self):
        """Test completing execution successfully."""
        orchestrator = PipelineOrchestrator(name="test")
        execution_id = "test_exec"
        
        # Set up active execution
        orchestrator.active_executions[execution_id] = {
            "execution_id": execution_id,
            "pipeline_id": uuid4(),
            "started_at": datetime(2024, 1, 15, 10, 0, 0),
            "status": "running"
        }
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 10, 5, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            result_data = {"records_processed": 1000}
            orchestrator.complete_execution(execution_id, True, 300.0, result_data)
            
            assert execution_id not in orchestrator.active_executions
            assert orchestrator.metrics.total_executions == 1
            assert orchestrator.metrics.successful_executions == 1
            assert orchestrator.updated_at == mock_now
            
            # Check completion record
            assert "completed_executions" in orchestrator.config
            completion_record = orchestrator.config["completed_executions"][0]
            assert completion_record["execution_id"] == execution_id
            assert completion_record["success"] is True
            assert completion_record["duration_seconds"] == 300.0
            assert completion_record["result_data"] == result_data

    def test_complete_execution_not_active(self):
        """Test completing execution that is not active."""
        orchestrator = PipelineOrchestrator(name="test")
        execution_id = "nonexistent"
        
        with pytest.raises(ValueError, match=f"Execution {execution_id} is not active"):
            orchestrator.complete_execution(execution_id, True, 100.0)

    def test_cancel_execution_success(self):
        """Test cancelling execution successfully."""
        orchestrator = PipelineOrchestrator(name="test")
        execution_id = "test_exec"
        
        # Set up active execution
        orchestrator.active_executions[execution_id] = {
            "execution_id": execution_id,
            "pipeline_id": uuid4(),
            "started_at": datetime(2024, 1, 15, 10, 0, 0),
            "status": "running"
        }
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 10, 2, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            result = orchestrator.cancel_execution(execution_id)
            
            assert result is True
            assert execution_id not in orchestrator.active_executions
            assert orchestrator.metrics.cancelled_executions == 1
            assert orchestrator.updated_at == mock_now
            
            # Check cancellation record
            assert "cancelled_executions" in orchestrator.config
            cancellation_record = orchestrator.config["cancelled_executions"][0]
            assert cancellation_record["execution_id"] == execution_id
            assert cancellation_record["cancelled_at"] == mock_now.isoformat()

    def test_cancel_execution_not_active(self):
        """Test cancelling execution that is not active."""
        orchestrator = PipelineOrchestrator(name="test")
        execution_id = "nonexistent"
        
        result = orchestrator.cancel_execution(execution_id)
        
        assert result is False

    def test_start_orchestration(self):
        """Test starting orchestration."""
        orchestrator = PipelineOrchestrator(name="test", status=OrchestrationStatus.IDLE)
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 16, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.start_orchestration()
            
            assert orchestrator.status == OrchestrationStatus.RUNNING
            assert orchestrator.updated_at == mock_now

    def test_start_orchestration_already_running(self):
        """Test starting orchestration when already running."""
        orchestrator = PipelineOrchestrator(name="test", status=OrchestrationStatus.RUNNING)
        
        with pytest.raises(ValueError, match="Orchestrator is already running"):
            orchestrator.start_orchestration()

    def test_stop_orchestration(self):
        """Test stopping orchestration."""
        orchestrator = PipelineOrchestrator(name="test", status=OrchestrationStatus.RUNNING)
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 17, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.stop_orchestration()
            
            assert orchestrator.status == OrchestrationStatus.STOPPED
            assert orchestrator.updated_at == mock_now

    def test_stop_orchestration_already_stopped(self):
        """Test stopping orchestration when already stopped."""
        orchestrator = PipelineOrchestrator(name="test", status=OrchestrationStatus.STOPPED)
        
        with pytest.raises(ValueError, match="Orchestrator is already stopped"):
            orchestrator.stop_orchestration()

    def test_pause_orchestration(self):
        """Test pausing orchestration."""
        orchestrator = PipelineOrchestrator(name="test", status=OrchestrationStatus.RUNNING)
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 18, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.pause_orchestration()
            
            assert orchestrator.status == OrchestrationStatus.PAUSED
            assert orchestrator.updated_at == mock_now

    def test_pause_orchestration_not_running(self):
        """Test pausing orchestration when not running."""
        orchestrator = PipelineOrchestrator(name="test", status=OrchestrationStatus.IDLE)
        
        with pytest.raises(ValueError, match="Can only pause a running orchestrator"):
            orchestrator.pause_orchestration()

    def test_resume_orchestration(self):
        """Test resuming orchestration."""
        orchestrator = PipelineOrchestrator(name="test", status=OrchestrationStatus.PAUSED)
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 19, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.resume_orchestration()
            
            assert orchestrator.status == OrchestrationStatus.RUNNING
            assert orchestrator.updated_at == mock_now

    def test_resume_orchestration_not_paused(self):
        """Test resuming orchestration when not paused."""
        orchestrator = PipelineOrchestrator(name="test", status=OrchestrationStatus.RUNNING)
        
        with pytest.raises(ValueError, match="Can only resume a paused orchestrator"):
            orchestrator.resume_orchestration()

    def test_enter_maintenance_mode(self):
        """Test entering maintenance mode."""
        orchestrator = PipelineOrchestrator(name="test")
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 20, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.enter_maintenance_mode()
            
            assert orchestrator.status == OrchestrationStatus.MAINTENANCE
            assert orchestrator.updated_at == mock_now

    def test_exit_maintenance_mode(self):
        """Test exiting maintenance mode."""
        orchestrator = PipelineOrchestrator(name="test", status=OrchestrationStatus.MAINTENANCE)
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 21, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.exit_maintenance_mode()
            
            assert orchestrator.status == OrchestrationStatus.IDLE
            assert orchestrator.updated_at == mock_now

    def test_exit_maintenance_mode_not_in_maintenance(self):
        """Test exiting maintenance mode when not in maintenance."""
        orchestrator = PipelineOrchestrator(name="test", status=OrchestrationStatus.RUNNING)
        
        with pytest.raises(ValueError, match="Not in maintenance mode"):
            orchestrator.exit_maintenance_mode()

    def test_update_resource_limits(self):
        """Test updating resource limits."""
        orchestrator = PipelineOrchestrator(name="test")
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 22, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.update_resource_limits(
                max_concurrent=15,
                max_queue_size=150,
                memory_limit_mb=4096,
                cpu_limit_cores=8.0
            )
            
            assert orchestrator.max_concurrent_executions == 15
            assert orchestrator.max_queue_size == 150
            assert orchestrator.memory_limit_mb == 4096
            assert orchestrator.cpu_limit_cores == 8.0
            assert orchestrator.updated_at == mock_now

    def test_update_resource_limits_invalid_values(self):
        """Test updating resource limits with invalid values."""
        orchestrator = PipelineOrchestrator(name="test")
        
        with pytest.raises(ValueError, match="Max concurrent executions must be positive"):
            orchestrator.update_resource_limits(max_concurrent=0)
        
        with pytest.raises(ValueError, match="Max queue size must be positive"):
            orchestrator.update_resource_limits(max_queue_size=-1)
        
        with pytest.raises(ValueError, match="Memory limit must be positive"):
            orchestrator.update_resource_limits(memory_limit_mb=0)
        
        with pytest.raises(ValueError, match="CPU limit must be positive"):
            orchestrator.update_resource_limits(cpu_limit_cores=-1.0)

    def test_add_tag(self):
        """Test adding tags."""
        orchestrator = PipelineOrchestrator(name="test")
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 23, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.add_tag("production")
            orchestrator.add_tag("critical")
            
            assert "production" in orchestrator.tags
            assert "critical" in orchestrator.tags
            assert len(orchestrator.tags) == 2
            assert orchestrator.updated_at == mock_now

    def test_add_tag_duplicate(self):
        """Test adding duplicate tag."""
        orchestrator = PipelineOrchestrator(name="test")
        orchestrator.add_tag("production")
        orchestrator.add_tag("production")  # Duplicate
        
        assert orchestrator.tags.count("production") == 1

    def test_add_tag_empty(self):
        """Test adding empty tag does nothing."""
        orchestrator = PipelineOrchestrator(name="test")
        original_updated_at = orchestrator.updated_at
        
        orchestrator.add_tag("")
        
        assert len(orchestrator.tags) == 0
        assert orchestrator.updated_at == original_updated_at

    def test_remove_tag(self):
        """Test removing tags."""
        orchestrator = PipelineOrchestrator(name="test")
        orchestrator.tags = ["production", "critical", "legacy"]
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 16, 0, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.remove_tag("critical")
            
            assert "critical" not in orchestrator.tags
            assert len(orchestrator.tags) == 2
            assert orchestrator.updated_at == mock_now

    def test_remove_tag_nonexistent(self):
        """Test removing non-existent tag."""
        orchestrator = PipelineOrchestrator(name="test")
        orchestrator.tags = ["production"]
        original_updated_at = orchestrator.updated_at
        
        orchestrator.remove_tag("nonexistent")
        
        assert len(orchestrator.tags) == 1
        assert orchestrator.updated_at == original_updated_at

    def test_has_tag(self):
        """Test checking for tags."""
        orchestrator = PipelineOrchestrator(name="test")
        orchestrator.tags = ["production", "critical"]
        
        assert orchestrator.has_tag("production") is True
        assert orchestrator.has_tag("critical") is True
        assert orchestrator.has_tag("nonexistent") is False

    def test_update_config(self):
        """Test updating configuration."""
        orchestrator = PipelineOrchestrator(name="test")
        
        with patch('data_pipelines.domain.entities.pipeline_orchestrator.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 16, 1, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            orchestrator.update_config("log_level", "DEBUG")
            orchestrator.update_config("timeout", 3600)
            
            assert orchestrator.config["log_level"] == "DEBUG"
            assert orchestrator.config["timeout"] == 3600
            assert orchestrator.updated_at == mock_now

    def test_get_health_status_healthy(self):
        """Test getting health status when healthy."""
        orchestrator = PipelineOrchestrator(name="test")
        orchestrator.status = OrchestrationStatus.RUNNING
        
        health = orchestrator.get_health_status()
        
        assert health["status"] == "running"
        assert health["health_score"] == 100
        assert health["health_issues"] == []
        assert health["is_healthy"] is True
        assert "resource_utilization" in health
        assert "metrics" in health

    def test_get_health_status_with_issues(self):
        """Test getting health status with issues."""
        orchestrator = PipelineOrchestrator(name="test", max_concurrent_executions=10, max_queue_size=20)
        orchestrator.status = OrchestrationStatus.ERROR
        
        # Create high utilization
        orchestrator.active_executions = {f"exec{i}": {} for i in range(10)}  # 100% execution utilization
        orchestrator.execution_queue = [{"id": str(i)} for i in range(17)]     # 85% queue utilization
        
        # Set high failure rate
        orchestrator.metrics.total_executions = 100
        orchestrator.metrics.failed_executions = 15  # 15% failure rate (above 10% threshold)
        
        health = orchestrator.get_health_status()
        
        assert health["health_score"] < 100
        assert len(health["health_issues"]) > 0
        assert health["is_healthy"] is False
        assert "High failure rate" in health["health_issues"]
        assert "Orchestrator in error state" in health["health_issues"]

    def test_to_dict(self):
        """Test converting orchestrator to dictionary."""
        orchestrator = PipelineOrchestrator(
            name="Test Orchestrator",
            description="Test orchestrator",
            execution_strategy=ExecutionStrategy.PARALLEL,
            max_concurrent_executions=10,
            created_by="admin",
            config={"log_level": "INFO"},
            tags=["test", "orchestrator"]
        )
        
        result = orchestrator.to_dict()
        
        assert result["id"] == str(orchestrator.id)
        assert result["name"] == "Test Orchestrator"
        assert result["description"] == "Test orchestrator"
        assert result["execution_strategy"] == "parallel"
        assert result["max_concurrent_executions"] == 10
        assert result["created_by"] == "admin"
        assert result["config"] == {"log_level": "INFO"}
        assert result["tags"] == ["test", "orchestrator"]
        assert result["is_running"] is False
        assert result["is_available"] is True
        assert "resource_utilization" in result

    def test_str_representation(self):
        """Test string representation."""
        orchestrator = PipelineOrchestrator(name="Test Orchestrator")
        orchestrator.registered_pipelines = [uuid4(), uuid4()]
        
        str_repr = str(orchestrator)
        
        assert "PipelineOrchestrator('Test Orchestrator'" in str_repr
        assert "idle" in str_repr
        assert "pipelines=2" in str_repr

    def test_repr_representation(self):
        """Test detailed representation."""
        orchestrator = PipelineOrchestrator(name="Test Orchestrator")
        orchestrator.active_executions = {"exec1": {}}
        
        repr_str = repr(orchestrator)
        
        assert f"id={orchestrator.id}" in repr_str
        assert "name='Test Orchestrator'" in repr_str
        assert "status=idle" in repr_str
        assert "executions=1" in repr_str