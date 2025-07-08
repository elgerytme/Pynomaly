"""
Advanced Scheduler Example

This example demonstrates the advanced scheduler system with:
1. DAG parsing and circular dependency detection
2. Resource management with CPU/RAM/Workers quotas
3. Cron and interval triggers
4. Schedule persistence
5. Integration with ProcessingOrchestrator
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import scheduler components
from src.pynomaly.infrastructure.scheduler import (
    # Core scheduler
    AdvancedScheduler,
    get_scheduler,
    
    # Entities
    Schedule,
    JobDefinition,
    ResourceRequirement,
    
    # DAG parsing
    DAGParser,
    CircularDependencyError,
    
    # Resource management
    ResourceManager,
    
    # Triggers
    TriggerManager,
    TriggerType,
    
    # Repository
    InMemoryScheduleRepository,
    FileSystemScheduleRepository,
)

# Import processing orchestrator
from src.pynomaly.domain.services.processing_orchestrator import ProcessingOrchestrator
from src.pynomaly.shared.config import Config


def create_sample_dag_schedule() -> Schedule:
    """Create a sample schedule with a DAG of jobs."""
    
    # Create job definitions with dependencies
    job_a = JobDefinition(
        job_id="extract_data",
        name="Extract Data",
        description="Extract data from source systems",
        processing_config={
            "input_path": "/data/source",
            "output_path": "/data/raw",
            "engine": "multiprocessing"
        },
        resource_requirements=ResourceRequirement(
            cpu_cores=2.0,
            memory_gb=4.0,
            workers=2
        ),
        depends_on=set()  # No dependencies (root job)
    )
    
    job_b = JobDefinition(
        job_id="clean_data",
        name="Clean Data",
        description="Clean and validate extracted data",
        processing_config={
            "input_path": "/data/raw",
            "output_path": "/data/clean",
            "engine": "multiprocessing"
        },
        resource_requirements=ResourceRequirement(
            cpu_cores=1.5,
            memory_gb=3.0,
            workers=1
        ),
        depends_on={"extract_data"}  # Depends on extract_data
    )
    
    job_c = JobDefinition(
        job_id="feature_engineering",
        name="Feature Engineering",
        description="Create features for anomaly detection",
        processing_config={
            "input_path": "/data/clean",
            "output_path": "/data/features",
            "engine": "dask"
        },
        resource_requirements=ResourceRequirement(
            cpu_cores=2.0,
            memory_gb=6.0,
            workers=3
        ),
        depends_on={"clean_data"}  # Depends on clean_data
    )
    
    job_d = JobDefinition(
        job_id="train_model",
        name="Train Model",
        description="Train anomaly detection model",
        processing_config={
            "input_path": "/data/features",
            "output_path": "/models/trained",
            "engine": "dask"
        },
        resource_requirements=ResourceRequirement(
            cpu_cores=4.0,
            memory_gb=8.0,
            workers=2,
            gpu_count=1
        ),
        depends_on={"feature_engineering"}  # Depends on feature_engineering
    )
    
    job_e = JobDefinition(
        job_id="validate_model",
        name="Validate Model",
        description="Validate trained model performance",
        processing_config={
            "input_path": "/models/trained",
            "output_path": "/models/validated",
            "engine": "multiprocessing"
        },
        resource_requirements=ResourceRequirement(
            cpu_cores=1.0,
            memory_gb=2.0,
            workers=1
        ),
        depends_on={"train_model"}  # Depends on train_model
    )
    
    job_f = JobDefinition(
        job_id="deploy_model",
        name="Deploy Model",
        description="Deploy validated model to production",
        processing_config={
            "input_path": "/models/validated",
            "output_path": "/models/production",
            "engine": "multiprocessing"
        },
        resource_requirements=ResourceRequirement(
            cpu_cores=1.0,
            memory_gb=1.0,
            workers=1
        ),
        depends_on={"validate_model"}  # Depends on validate_model
    )
    
    # Create schedule
    schedule = Schedule(
        schedule_id="ml_pipeline_daily",
        name="ML Pipeline - Daily",
        description="Daily ML pipeline for anomaly detection",
        cron_expression="0 2 * * *",  # Run at 2 AM every day
        max_concurrent_jobs=5,
        global_timeout_minutes=360,  # 6 hours
        tags={"team": "ml", "environment": "production"}
    )
    
    # Add jobs to schedule
    schedule.add_job(job_a)
    schedule.add_job(job_b)
    schedule.add_job(job_c)
    schedule.add_job(job_d)
    schedule.add_job(job_e)
    schedule.add_job(job_f)
    
    return schedule


def create_circular_dependency_schedule() -> Schedule:
    """Create a schedule with circular dependencies to test detection."""
    
    job_a = JobDefinition(
        job_id="job_a",
        name="Job A",
        description="First job",
        depends_on={"job_c"}  # This will create a circular dependency
    )
    
    job_b = JobDefinition(
        job_id="job_b",
        name="Job B",
        description="Second job",
        depends_on={"job_a"}
    )
    
    job_c = JobDefinition(
        job_id="job_c",
        name="Job C",
        description="Third job",
        depends_on={"job_b"}
    )
    
    schedule = Schedule(
        schedule_id="circular_test",
        name="Circular Dependency Test",
        description="Test circular dependency detection",
        interval_seconds=300  # 5 minutes
    )
    
    schedule.add_job(job_a)
    schedule.add_job(job_b)
    schedule.add_job(job_c)
    
    return schedule


def test_dag_parsing():
    """Test DAG parsing and circular dependency detection."""
    logger.info("Testing DAG parsing and circular dependency detection...")
    
    # Test valid DAG
    valid_schedule = create_sample_dag_schedule()
    dag_parser = DAGParser()
    
    try:
        dag_parser.parse_schedule(valid_schedule)
        logger.info("✓ Valid DAG parsed successfully")
        
        # Display DAG information
        logger.info(f"DAG Statistics: {dag_parser.get_dag_statistics()}")
        logger.info(f"Execution Order: {dag_parser.get_execution_order()}")
        logger.info(f"DAG Visualization:\n{dag_parser.visualize_dag()}")
        
    except Exception as e:
        logger.error(f"✗ Failed to parse valid DAG: {e}")
    
    # Test circular dependency detection
    circular_schedule = create_circular_dependency_schedule()
    
    try:
        dag_parser.parse_schedule(circular_schedule)
        logger.error("✗ Circular dependency not detected!")
    except CircularDependencyError as e:
        logger.info(f"✓ Circular dependency detected: {e}")
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")


def test_resource_management():
    """Test resource management and quota enforcement."""
    logger.info("Testing resource management...")
    
    # Create resource manager with limits
    resource_manager = ResourceManager(
        cpu_limit=10.0,
        memory_limit=20.0,
        worker_limit=10,
        gpu_limit=2
    )
    
    # Test resource allocation
    logger.info(f"Initial available resources: {resource_manager.available_resources()}")
    
    # Try to allocate resources
    success = resource_manager.allocate_resources(
        cpu=4.0,
        memory=8.0,
        workers=3,
        gpu=1
    )
    
    if success:
        logger.info("✓ Resources allocated successfully")
        logger.info(f"Available after allocation: {resource_manager.available_resources()}")
    else:
        logger.error("✗ Resource allocation failed")
    
    # Try to allocate more resources than available
    success = resource_manager.allocate_resources(
        cpu=8.0,
        memory=15.0,
        workers=8,
        gpu=2
    )
    
    if not success:
        logger.info("✓ Resource quota enforcement working")
    else:
        logger.error("✗ Resource quota enforcement failed")
    
    # Release resources
    resource_manager.release_resources(
        cpu=4.0,
        memory=8.0,
        workers=3,
        gpu=1
    )
    
    logger.info(f"Available after release: {resource_manager.available_resources()}")


def test_trigger_management():
    """Test trigger management with cron and interval triggers."""
    logger.info("Testing trigger management...")
    
    trigger_manager = TriggerManager()
    
    # Test cron trigger (every 5 minutes)
    try:
        cron_trigger = trigger_manager.create_cron_trigger("test_cron", "*/5 * * * *")
        logger.info("✓ Cron trigger created successfully")
        
        current_time = datetime.now()
        next_run = cron_trigger.get_next_run_time(current_time)
        logger.info(f"Next cron run: {next_run}")
        
    except Exception as e:
        logger.error(f"✗ Cron trigger creation failed: {e}")
    
    # Test interval trigger (every 10 minutes)
    try:
        interval_trigger = trigger_manager.create_interval_trigger("test_interval", 600)
        logger.info("✓ Interval trigger created successfully")
        
        current_time = datetime.now()
        next_run = interval_trigger.get_next_run_time(current_time)
        logger.info(f"Next interval run: {next_run}")
        
    except Exception as e:
        logger.error(f"✗ Interval trigger creation failed: {e}")
    
    # Test trigger evaluation
    current_time = datetime.now()
    last_run = current_time - timedelta(minutes=15)
    
    should_run = trigger_manager.should_run("test_cron", current_time, last_run)
    logger.info(f"Cron trigger should run: {should_run}")
    
    should_run = trigger_manager.should_run("test_interval", current_time, last_run)
    logger.info(f"Interval trigger should run: {should_run}")


def test_schedule_persistence():
    """Test schedule persistence with repository."""
    logger.info("Testing schedule persistence...")
    
    # Test in-memory repository
    memory_repo = InMemoryScheduleRepository()
    
    # Create and save schedule
    schedule = create_sample_dag_schedule()
    memory_repo.save_schedule(schedule)
    
    # Retrieve schedule
    retrieved = memory_repo.get_schedule(schedule.schedule_id)
    if retrieved and retrieved.schedule_id == schedule.schedule_id:
        logger.info("✓ In-memory persistence working")
    else:
        logger.error("✗ In-memory persistence failed")
    
    # Test file system repository
    try:
        file_repo = FileSystemScheduleRepository(base_path="./test_data")
        
        # Save schedule
        file_repo.save_schedule(schedule)
        
        # Retrieve schedule
        retrieved = file_repo.get_schedule(schedule.schedule_id)
        if retrieved and retrieved.schedule_id == schedule.schedule_id:
            logger.info("✓ File system persistence working")
        else:
            logger.error("✗ File system persistence failed")
        
        # Clean up
        file_repo.clear_all()
        
    except Exception as e:
        logger.error(f"✗ File system persistence failed: {e}")


async def test_scheduler_integration():
    """Test full scheduler integration."""
    logger.info("Testing scheduler integration...")
    
    # Set up components
    repository = InMemoryScheduleRepository()
    orchestrator = ProcessingOrchestrator(Config())
    resource_manager = ResourceManager(
        cpu_limit=16.0,
        memory_limit=32.0,
        worker_limit=20,
        gpu_limit=4
    )
    trigger_manager = TriggerManager()
    
    # Create scheduler
    scheduler = get_scheduler(
        repository=repository,
        orchestrator=orchestrator,
        resource_manager=resource_manager,
        trigger_manager=trigger_manager,
        max_concurrent_jobs=5
    )
    
    # Create and save a schedule
    schedule = create_sample_dag_schedule()
    # Change to interval trigger for testing
    schedule.cron_expression = None
    schedule.interval_seconds = 30  # 30 seconds for testing
    
    repository.save_schedule(schedule)
    
    # Set up trigger
    trigger_manager.create_interval_trigger(schedule.schedule_id, 30)
    
    logger.info("✓ Scheduler integration test setup complete")
    logger.info("Note: Full scheduler execution would require actual ProcessingOrchestrator backend")


def main():
    """Run all tests and examples."""
    logger.info("Advanced Scheduler Example - Starting...")
    
    # Test individual components
    test_dag_parsing()
    test_resource_management()
    test_trigger_management()
    test_schedule_persistence()
    
    # Test integration
    asyncio.run(test_scheduler_integration())
    
    logger.info("Advanced Scheduler Example - Complete!")


if __name__ == "__main__":
    main()
