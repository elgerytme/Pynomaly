"""Main scheduler implementation that orchestrates job execution."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from .entities import (
    Schedule, JobDefinition, JobStatus, ExecutionStatus, 
    ScheduleStatus, ResourceRequirement
)
from .dag_parser import DAGParser
from .dependency_resolver import DependencyResolver
from .resource_manager import ResourceManager, ResourceQuota
from .trigger_manager import TriggerManager
from .schedule_repository import ScheduleRepository


@dataclass
class JobExecution:
    """Represents a running job execution."""
    execution_id: str
    job_id: str
    schedule_id: str
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ScheduleExecution:
    """Represents a running schedule execution."""
    execution_id: str
    schedule_id: str
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    job_executions: Dict[str, JobExecution]
    completed_jobs: Set[str]
    failed_jobs: Set[str]


class Scheduler:
    """Advanced scheduler for executing DAGs of batch jobs."""
    
    def __init__(
        self,
        repository: ScheduleRepository,
        resource_quota: Optional[ResourceQuota] = None,
        processing_orchestrator=None
    ):
        """Initialize the scheduler."""
        self.repository = repository
        self.resource_manager = ResourceManager(resource_quota)
        self.processing_orchestrator = processing_orchestrator
        self.logger = logging.getLogger(__name__)
        
        # Execution tracking
        self.active_executions: Dict[str, ScheduleExecution] = {}
        self.running_jobs: Dict[str, JobExecution] = {}
        
        # Scheduler state
        self.is_running = False
        
    async def start(self) -> None:
        """Start the scheduler."""
        self.is_running = True
        self.logger.info("Scheduler started")
        
        # Start the main scheduling loop
        await self._scheduling_loop()
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        self.is_running = False
        self.logger.info("Scheduler stopped")
        
        # Cancel any running executions
        for execution in self.active_executions.values():
            await self._cancel_execution(execution.execution_id)
    
    async def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while self.is_running:
            try:
                # Check for schedules that need to be executed
                schedules = self.repository.get_all_active()
                
                for schedule in schedules:
                    if await self._should_execute_schedule(schedule):
                        await self._execute_schedule(schedule)
                
                # Check for completed jobs and update executions
                await self._check_running_jobs()
                
                # Sleep for a short interval before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in scheduling loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _should_execute_schedule(self, schedule: Schedule) -> bool:
        """Check if a schedule should be executed now."""
        if not schedule.is_active():
            return False
        
        # Check if already running
        if schedule.schedule_id in self.active_executions:
            return False
        
        # Check trigger conditions
        now = datetime.now()
        
        if schedule.next_execution_at and now >= schedule.next_execution_at:
            return True
        
        # If no next execution time set, compute it
        if not schedule.next_execution_at:
            next_exec = TriggerManager.compute_next_execution(
                schedule.cron_expression,
                schedule.interval_seconds,
                now
            )
            if next_exec:
                schedule.next_execution_at = next_exec
                self.repository.save(schedule)
        
        return False
    
    async def _execute_schedule(self, schedule: Schedule) -> str:
        """Execute a schedule."""
        execution_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting execution {execution_id} for schedule {schedule.name}")
        
        # Validate DAG before execution
        errors = DependencyResolver.validate_dependencies(schedule.jobs)
        if errors:
            self.logger.error(f"Schedule {schedule.name} has dependency errors: {errors}")
            return execution_id
        
        # Check for circular dependencies
        cycles = DAGParser.detect_cycles(schedule.jobs)
        if cycles:
            self.logger.error(f"Schedule {schedule.name} has circular dependencies: {cycles}")
            return execution_id
        
        # Create execution record
        execution = ScheduleExecution(
            execution_id=execution_id,
            schedule_id=schedule.schedule_id,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.now(),
            job_executions={},
            completed_jobs=set(),
            failed_jobs=set()
        )
        
        self.active_executions[execution_id] = execution
        
        # Update schedule
        schedule.last_execution_id = execution_id
        schedule.last_execution_at = datetime.now()
        schedule.last_execution_status = ExecutionStatus.RUNNING
        schedule.total_executions += 1
        
        # Compute next execution time
        schedule.next_execution_at = TriggerManager.compute_next_execution(
            schedule.cron_expression,
            schedule.interval_seconds,
            datetime.now()
        )
        
        self.repository.save(schedule)
        
        # Start executing jobs
        await self._execute_ready_jobs(execution, schedule)
        
        return execution_id
    
    async def _execute_ready_jobs(self, execution: ScheduleExecution, schedule: Schedule) -> None:
        """Execute jobs that are ready to run."""
        # Get jobs that are ready to execute
        ready_jobs = DependencyResolver.get_ready_jobs(
            schedule.jobs, 
            execution.completed_jobs
        )
        
        # Filter by resource availability and concurrency limits
        allocatable_jobs = self.resource_manager.get_allocatable_jobs(
            {job_id: schedule.jobs[job_id] for job_id in ready_jobs}
        )
        
        # Respect max concurrent jobs limit
        current_running = len([
            job for job in execution.job_executions.values() 
            if job.status == JobStatus.RUNNING
        ])
        
        max_new_jobs = schedule.max_concurrent_jobs - current_running
        jobs_to_start = allocatable_jobs[:max_new_jobs]
        
        # Start the jobs
        for job_id in jobs_to_start:
            if job_id not in execution.job_executions:
                await self._start_job(execution, schedule.jobs[job_id])
    
    async def _start_job(self, execution: ScheduleExecution, job: JobDefinition) -> None:
        """Start executing a single job."""
        job_execution_id = str(uuid.uuid4())
        
        # Allocate resources
        if not self.resource_manager.allocate(job.job_id, job.resource_requirements):
            self.logger.warning(f"Could not allocate resources for job {job.job_id}")
            return
        
        job_execution = JobExecution(
            execution_id=job_execution_id,
            job_id=job.job_id,
            schedule_id=execution.schedule_id,
            status=JobStatus.RUNNING,
            start_time=datetime.now()
        )
        
        execution.job_executions[job.job_id] = job_execution
        self.running_jobs[job_execution_id] = job_execution
        
        self.logger.info(f"Started job {job.name} (ID: {job.job_id})")
        
        # Here you would integrate with ProcessingOrchestrator
        # For now, we'll simulate job execution
        asyncio.create_task(self._simulate_job_execution(job_execution, job))
    
    async def _simulate_job_execution(self, job_execution: JobExecution, job: JobDefinition) -> None:
        """Simulate job execution (replace with ProcessingOrchestrator integration)."""
        try:
            # Simulate some work
            await asyncio.sleep(5)  # Simulated job duration
            
            # Mark as completed
            job_execution.status = JobStatus.COMPLETED
            job_execution.end_time = datetime.now()
            
            self.logger.info(f"Job {job.name} completed successfully")
            
        except Exception as e:
            job_execution.status = JobStatus.FAILED
            job_execution.end_time = datetime.now()
            job_execution.error_message = str(e)
            
            self.logger.error(f"Job {job.name} failed: {e}")
    
    async def _check_running_jobs(self) -> None:
        """Check status of running jobs and update executions."""
        completed_job_ids = []
        
        for job_execution_id, job_execution in self.running_jobs.items():
            if job_execution.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                completed_job_ids.append(job_execution_id)
                
                # Deallocate resources
                schedule = self.repository.get_by_id(job_execution.schedule_id)
                if schedule:
                    job = schedule.get_job(job_execution.job_id)
                    if job:
                        self.resource_manager.deallocate(job.job_id)
                
                # Update execution
                execution = self.active_executions.get(job_execution.schedule_id)
                if execution:
                    if job_execution.status == JobStatus.COMPLETED:
                        execution.completed_jobs.add(job_execution.job_id)
                    else:
                        execution.failed_jobs.add(job_execution.job_id)
                    
                    # Check if we can start more jobs
                    await self._execute_ready_jobs(execution, schedule)
                    
                    # Check if execution is complete
                    await self._check_execution_completion(execution, schedule)
        
        # Remove completed jobs from running list
        for job_id in completed_job_ids:
            del self.running_jobs[job_id]
    
    async def _check_execution_completion(self, execution: ScheduleExecution, schedule: Schedule) -> None:
        """Check if a schedule execution is complete."""
        total_jobs = len(schedule.jobs)
        completed_or_failed = len(execution.completed_jobs) + len(execution.failed_jobs)
        
        if completed_or_failed >= total_jobs:
            # Execution is complete
            execution.end_time = datetime.now()
            
            if execution.failed_jobs:
                execution.status = ExecutionStatus.PARTIAL if execution.completed_jobs else ExecutionStatus.FAILED
                schedule.last_execution_status = execution.status
            else:
                execution.status = ExecutionStatus.COMPLETED
                schedule.last_execution_status = ExecutionStatus.COMPLETED
                schedule.successful_executions += 1
            
            if execution.failed_jobs:
                schedule.failed_executions += 1
            
            self.repository.save(schedule)
            
            # Remove from active executions
            del self.active_executions[execution.execution_id]
            
            self.logger.info(
                f"Execution {execution.execution_id} completed. "
                f"Status: {execution.status}, "
                f"Completed: {len(execution.completed_jobs)}, "
                f"Failed: {len(execution.failed_jobs)}"
            )
    
    async def _cancel_execution(self, execution_id: str) -> None:
        """Cancel a running execution."""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return
        
        # Cancel all running jobs
        for job_execution in execution.job_executions.values():
            if job_execution.status == JobStatus.RUNNING:
                job_execution.status = JobStatus.CANCELLED
                job_execution.end_time = datetime.now()
                
                # Deallocate resources
                schedule = self.repository.get_by_id(execution.schedule_id)
                if schedule:
                    job = schedule.get_job(job_execution.job_id)
                    if job:
                        self.resource_manager.deallocate(job.job_id)
        
        execution.status = ExecutionStatus.CANCELLED
        execution.end_time = datetime.now()
        
        # Update schedule
        schedule = self.repository.get_by_id(execution.schedule_id)
        if schedule:
            schedule.last_execution_status = ExecutionStatus.CANCELLED
            self.repository.save(schedule)
        
        del self.active_executions[execution_id]
        
        self.logger.info(f"Cancelled execution {execution_id}")
    
    def get_execution_status(self, execution_id: str) -> Optional[ScheduleExecution]:
        """Get the status of a running execution."""
        return self.active_executions.get(execution_id)
    
    def get_active_executions(self) -> List[ScheduleExecution]:
        """Get all active executions."""
        return list(self.active_executions.values())
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        return self.resource_manager.get_resource_utilization()
