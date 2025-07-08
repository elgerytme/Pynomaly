"""Advanced scheduler class to coordinate job execution based on schedule definitions."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict

from .entities import JobInstance, Schedule, ScheduleExecution, ExecutionResult, JobStatus, ScheduleStatus
from .dag_parser import DAGParser
from .dependency_resolver import DependencyResolver
from .resource_manager import ResourceManager
from .trigger_manager import TriggerManager, TriggerType
from .schedule_repository import ScheduleRepository
from ...domain.services.processing_orchestrator import ProcessingOrchestrator


logger = logging.getLogger(__name__)


class SchedulerStatus:
    """Status of the scheduler service."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"


class AdvancedScheduler:
    """Coordinator for job execution based on schedule definitions."""
    def __init__(
        self,
        repository: ScheduleRepository,
        orchestrator: ProcessingOrchestrator,
        resource_manager: ResourceManager,
        trigger_manager: TriggerManager,
        max_concurrent_jobs: int = 10,
    ) -> None:
        """Initialize the scheduler with required components."""
        self.repository = repository
        self.orchestrator = orchestrator
        self.resource_manager = resource_manager
        self.trigger_manager = trigger_manager
        self.max_concurrent_jobs = max_concurrent_jobs
        self.status = SchedulerStatus.INITIALIZING

    async def start(self) -> None:
        """Start the scheduler service."""
        self.status = SchedulerStatus.RUNNING
        logger.info("Scheduler service started")
        
        await self._schedule_next_runs()

    async def _schedule_next_runs(self) -> None:
        """Schedule the next runs based on their trigger configurations."""
        while self.status == SchedulerStatus.RUNNING:
            # Get current time
            current_time = datetime.now()

            # Iterate over all schedules
            for schedule in self.repository.list_schedules():
                # Check if the schedule should run
                if not schedule.is_active():
                    continue
                
                last_run_time = schedule.last_execution_at
                if self.trigger_manager.should_run(schedule.schedule_id, current_time, last_run_time):
                    await self._execute_schedule(schedule)

            # Wait before next scheduling cycle
            await asyncio.sleep(60)

    async def _execute_schedule(self, schedule: Schedule) -> None:
        """Execute the jobs in a schedule based on the DAG."""
        logger.info(f"Executing schedule {schedule.schedule_id}")

        # Create dependency resolver and resolve execution plan
        dependency_resolver = DependencyResolver(schedule)
        execution_plan = dependency_resolver.validate_and_plan_execution()

        # Prepare execution
        execution_id = str(uuid.uuid4())
        execution_result = ExecutionResult(
            execution_id=execution_id,
            schedule_id=schedule.schedule_id,
            status=JobStatus.PENDING,
        )

        # Stage-wise execution
        completed_jobs = set()
        for stage_jobs in execution_plan.stages:
            stage_instances = []
            for job_id in stage_jobs:
                # Get job definition
                job = schedule.get_job(job_id)
                if job is None:
                    logger.error(f"Job {job_id} not found in schedule {schedule.schedule_id}")
                    continue

                # Check and allocate resources
                resources = job.resource_requirements
                if not self.resource_manager.allocate_resources(
                    resources.cpu_cores, resources.memory_gb, resources.workers, resources.gpu_count
                ):
                    logger.warning(f"Insufficient resources for job {job_id}")
                    continue

                # Create job instance
                instance_id = str(uuid.uuid4())
                job_instance = JobInstance(
                    instance_id=instance_id,
                    job_id=job_id,
                    execution_id=execution_id,
                    status=JobStatus.PENDING,
                )
                stage_instances.append(job_instance)

            # Execute stage
            tasks = [self._run_job_instance(instance, schedule) for instance in stage_instances]
            await asyncio.gather(*tasks)

            # Update results
            for job_instance in stage_instances:
                execution_result.add_job_instance(job_instance)
                logger.info(f"Job instance {job_instance.instance_id} status: {job_instance.status}")

                if job_instance.status == JobStatus.COMPLETED:
                    completed_jobs.add(job_instance.job_id)

            if not dependency_resolver.execution_plan_complete(completed_jobs):
                execution_result.partial()
                schedule.update_execution_stats(execution_result)
                return

        execution_result.complete()
        schedule.update_execution_stats(execution_result)
        logger.info(f"Schedule {schedule.schedule_id} execution completed")

        # Record schedule execution
        schedule_execution = ScheduleExecution(
            execution_id=execution_id,
            schedule_id=schedule.schedule_id,
            trigger_type=TriggerType.MANUAL.value,
            trigger_info={},
            result=execution_result,
        )
        self.repository.save_execution(schedule_execution)

    async def _run_job_instance(self, instance: JobInstance, schedule: Schedule) -> None:
        """Run a specific job instance."""
        job = schedule.get_job(instance.job_id)
        if job is None:
            instance.fail("Job definition not found")
            return

        config = job.processing_config

        # Start processing session
        try:
            session_id = await self.orchestrator.start_batch_session(
                name=f"{instance.job_id}_exec_{instance.execution_id}",
                input_path=config.get("input_path"),
                output_path=config.get("output_path"),
                engine=config.get("engine"),
                detection_algorithm=config.get("detection_algorithm"),
            )
            instance.start(session_id)
            logger.info(f"Started processing session {session_id} for job {instance.job_id}")

            # Simulate processing (in real scenario, await the completion)
            await asyncio.sleep(2)

            # Complete instance
            instance.complete(output_data={})

        except Exception as e:
            instance.fail(str(e))
            logger.error(f"Failed to execute job {instance.job_id}: {e}")

        finally:
            # Release allocated resources
            resources = job.resource_requirements
            self.resource_manager.release_resources(
                resources.cpu_cores, resources.memory_gb, resources.workers, resources.gpu_count
            )



def get_scheduler(
    repository: ScheduleRepository,
    orchestrator: ProcessingOrchestrator,
    resource_manager: ResourceManager,
    trigger_manager: TriggerManager,
    max_concurrent_jobs: int = 10,
) -> AdvancedScheduler:
    """Get an instance of the advanced scheduler."""
    return AdvancedScheduler(
        repository=repository,
        orchestrator=orchestrator,
        resource_manager=resource_manager,
        trigger_manager=trigger_manager,
        max_concurrent_jobs=max_concurrent_jobs,
    )
