"""Pytest configuration for data pipelines package testing."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock

# Import shared test utilities
from test_utilities.fixtures import *
from test_utilities.factories import *
from test_utilities.helpers import *
from test_utilities.markers import *

from data_pipelines.domain.entities.pipeline_orchestrator import (
    PipelineOrchestrator, OrchestrationStatus, ExecutionStrategy, ExecutionMetrics,
    PipelineExecutionMode
)
from data_pipelines.domain.entities.pipeline_schedule import (
    PipelineSchedule, ScheduleType, ScheduleStatus, ScheduleTrigger, TriggerType
)
from data_pipelines.domain.entities.pipeline_workflow import (
    PipelineWorkflow, WorkflowStatus, WorkflowStep, StepType, StepStatus, 
    StepCondition, ExecutionMode
)


@pytest.fixture
def sample_execution_metrics():
    """Sample execution metrics for testing."""
    metrics = ExecutionMetrics()
    metrics.update_execution(True, 120.0, 0.8, 512.0)
    metrics.update_execution(False, 90.0, 0.6, 256.0)
    metrics.update_execution(True, 150.0, 0.9, 768.0)
    return metrics


@pytest.fixture
def sample_pipeline_orchestrator():
    """Sample pipeline orchestrator for testing."""
    return PipelineOrchestrator(
        name="Production Orchestrator",
        description="Main production pipeline orchestrator",
        execution_strategy=ExecutionStrategy.PARALLEL,
        max_concurrent_executions=10,
        max_queue_size=50,
        created_by="admin"
    )


@pytest.fixture
def sample_schedule_trigger():
    """Sample schedule trigger for testing."""
    return ScheduleTrigger(
        name="Daily ETL Trigger",
        trigger_type=TriggerType.TIME_BASED,
        cron_expression="0 2 * * *",
        timezone="UTC"
    )


@pytest.fixture
def sample_pipeline_schedule(sample_schedule_trigger):
    """Sample pipeline schedule for testing."""
    schedule = PipelineSchedule(
        name="Daily Data Pipeline",
        description="Daily ETL processing pipeline",
        pipeline_id=uuid4(),
        schedule_type=ScheduleType.CRON,
        max_concurrent_runs=2,
        created_by="data_engineer"
    )
    schedule.add_trigger(sample_schedule_trigger)
    return schedule


@pytest.fixture
def sample_step_condition():
    """Sample step condition for testing."""
    return StepCondition(
        expression="data_count > 1000",
        operator=">=",
        value=1000,
        source_field="record_count"
    )


@pytest.fixture
def sample_workflow_step(sample_step_condition):
    """Sample workflow step for testing."""
    step = WorkflowStep(
        name="Data Validation",
        description="Validate incoming data quality",
        step_type=StepType.VALIDATE,
        timeout_seconds=600,
        max_retries=2
    )
    step.add_condition(sample_step_condition)
    return step


@pytest.fixture
def sample_pipeline_workflow(sample_workflow_step):
    """Sample pipeline workflow for testing."""
    workflow = PipelineWorkflow(
        name="Customer Data Processing",
        description="Complete customer data ETL workflow",
        max_parallel_steps=3,
        created_by="data_engineer"
    )
    workflow.add_step(sample_workflow_step)
    
    # Add additional steps to create a workflow
    extract_step = WorkflowStep(
        name="Extract Customer Data",
        step_type=StepType.EXTRACT,
        config={"source": "customer_db", "table": "customers"}
    )
    
    transform_step = WorkflowStep(
        name="Transform Data",
        step_type=StepType.TRANSFORM,
        depends_on=[extract_step.id],
        config={"operations": ["clean", "normalize", "enrich"]}
    )
    
    load_step = WorkflowStep(
        name="Load to Warehouse",
        step_type=StepType.LOAD,
        depends_on=[transform_step.id],
        config={"destination": "data_warehouse", "table": "dim_customers"}
    )
    
    workflow.add_step(extract_step)
    workflow.add_step(transform_step)
    workflow.add_step(load_step)
    
    return workflow


@pytest.fixture
def orchestrator_with_pipelines():
    """Orchestrator with registered pipelines and active executions."""
    orchestrator = PipelineOrchestrator(
        name="Test Orchestrator",
        description="Orchestrator for testing",
        max_concurrent_executions=5,
        created_by="tester"
    )
    
    # Register some pipelines
    pipeline_ids = [uuid4() for _ in range(3)]
    for pipeline_id in pipeline_ids:
        orchestrator.register_pipeline(pipeline_id)
    
    # Queue some executions
    for i, pipeline_id in enumerate(pipeline_ids[:2]):
        execution_id = orchestrator.queue_execution(
            pipeline_id, 
            {"priority": i + 1, "batch_size": 1000}
        )
        if i == 0:  # Start first execution
            orchestrator.start_execution(execution_id)
    
    return orchestrator


@pytest.fixture
def active_schedule_with_runs():
    """Active schedule with execution history."""
    schedule = PipelineSchedule(
        name="Hourly Processing",
        description="Process data every hour",
        pipeline_id=uuid4(),
        schedule_type=ScheduleType.INTERVAL,
        status=ScheduleStatus.ACTIVE,
        created_by="scheduler"
    )
    
    # Add interval trigger
    trigger = ScheduleTrigger(
        name="Hourly Trigger",
        trigger_type=TriggerType.TIME_BASED,
        interval_seconds=3600
    )
    schedule.add_trigger(trigger)
    
    # Simulate some runs
    schedule.total_runs = 10
    schedule.successful_runs = 8
    schedule.failed_runs = 2
    schedule.last_run_time = datetime.utcnow() - timedelta(hours=1)
    schedule.last_successful_run = schedule.last_run_time
    schedule.next_run_time = datetime.utcnow() + timedelta(hours=1)
    
    return schedule


@pytest.fixture
def complex_workflow():
    """Complex workflow with multiple parallel groups and conditions."""
    workflow = PipelineWorkflow(
        name="Complex ETL Workflow",
        description="Multi-stage ETL with parallel processing",
        execution_mode=ExecutionMode.PARALLEL,
        max_parallel_steps=5,
        created_by="architect"
    )
    
    # Stage 1: Parallel extraction
    extract_customers = WorkflowStep(
        name="Extract Customers",
        step_type=StepType.EXTRACT,
        parallel_group="extraction"
    )
    
    extract_orders = WorkflowStep(
        name="Extract Orders", 
        step_type=StepType.EXTRACT,
        parallel_group="extraction"
    )
    
    extract_products = WorkflowStep(
        name="Extract Products",
        step_type=StepType.EXTRACT,
        parallel_group="extraction"
    )
    
    # Stage 2: Data validation (depends on extraction)
    validate_data = WorkflowStep(
        name="Validate Data Quality",
        step_type=StepType.VALIDATE,
        depends_on=[extract_customers.id, extract_orders.id, extract_products.id]
    )
    
    # Stage 3: Parallel transformation
    transform_customers = WorkflowStep(
        name="Transform Customers",
        step_type=StepType.TRANSFORM,
        depends_on=[validate_data.id],
        parallel_group="transformation"
    )
    
    transform_orders = WorkflowStep(
        name="Transform Orders",
        step_type=StepType.TRANSFORM,
        depends_on=[validate_data.id],
        parallel_group="transformation"
    )
    
    # Stage 4: Join and aggregate
    join_data = WorkflowStep(
        name="Join Customer Orders",
        step_type=StepType.JOIN,
        depends_on=[transform_customers.id, transform_orders.id]
    )
    
    # Stage 5: Load to warehouse
    load_warehouse = WorkflowStep(
        name="Load to Data Warehouse",
        step_type=StepType.LOAD,
        depends_on=[join_data.id]
    )
    
    # Add all steps
    steps = [
        extract_customers, extract_orders, extract_products,
        validate_data, transform_customers, transform_orders,
        join_data, load_warehouse
    ]
    
    for step in steps:
        workflow.add_step(step)
    
    return workflow


@pytest.fixture
def running_workflow():
    """Workflow in running state with some completed steps."""
    workflow = PipelineWorkflow(
        name="Running Workflow",
        description="Workflow currently executing",
        status=WorkflowStatus.RUNNING,
        created_by="executor"
    )
    
    # Add steps with different statuses
    completed_step = WorkflowStep(
        name="Completed Step",
        step_type=StepType.EXTRACT,
        status=StepStatus.COMPLETED
    )
    completed_step.started_at = datetime.utcnow() - timedelta(minutes=30)
    completed_step.completed_at = datetime.utcnow() - timedelta(minutes=25)
    completed_step.output_data = {"records_processed": 5000}
    
    running_step = WorkflowStep(
        name="Running Step",
        step_type=StepType.TRANSFORM,
        status=StepStatus.RUNNING,
        depends_on=[completed_step.id]
    )
    running_step.started_at = datetime.utcnow() - timedelta(minutes=10)
    
    pending_step = WorkflowStep(
        name="Pending Step",
        step_type=StepType.LOAD,
        status=StepStatus.PENDING,
        depends_on=[running_step.id]
    )
    
    workflow.add_step(completed_step)
    workflow.add_step(running_step)
    workflow.add_step(pending_step)
    
    # Set workflow state
    workflow.started_at = datetime.utcnow() - timedelta(minutes=30)
    workflow.completed_steps = 1
    workflow.current_step_ids.add(running_step.id)
    
    return workflow