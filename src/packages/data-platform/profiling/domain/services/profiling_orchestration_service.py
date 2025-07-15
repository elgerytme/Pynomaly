"""Domain service for orchestrating complex profiling operations."""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from dataclasses import dataclass

from ..entities.data_profile import DataProfile, ProfilingJob, ProfilingStatus
from ..value_objects.profiling_metadata import (
    ProfilingConfiguration, ProfilingResult, ExecutionTimeline, ExecutionPhase,
    ResourceMetrics, ResourceType, ProfilingStrategy
)
from ..value_objects.quality_metrics import QualityConfiguration, QualityReport


class ProfilingPriority(str, Enum):
    """Priority levels for profiling operations."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal" 
    LOW = "low"
    BACKGROUND = "background"


class ProfilingMode(str, Enum):
    """Profiling execution modes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"
    SCHEDULED = "scheduled"


@dataclass
class ProfilingRequest:
    """Request for profiling operation."""
    request_id: str
    dataset_id: str
    dataset_source: Dict[str, Any]
    configuration: ProfilingConfiguration
    priority: ProfilingPriority = ProfilingPriority.NORMAL
    mode: ProfilingMode = ProfilingMode.ASYNCHRONOUS
    schedule: Optional[str] = None  # Cron expression for scheduled profiling
    dependencies: List[str] = None  # List of dependent request IDs
    quality_config: Optional[QualityConfiguration] = None
    callback_url: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProfilingContext:
    """Context for profiling execution."""
    request: ProfilingRequest
    job: ProfilingJob
    result: ProfilingResult
    timeline: List[ExecutionTimeline]
    resource_metrics: List[ResourceMetrics]
    warnings: List[str]
    errors: List[str]
    
    def __post_init__(self):
        if self.timeline is None:
            self.timeline = []
        if self.resource_metrics is None:
            self.resource_metrics = []
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class ProfilingOrchestrationService:
    """Domain service for orchestrating complex profiling workflows."""

    def __init__(self):
        self._active_jobs: Dict[str, ProfilingContext] = {}
        self._job_queue: List[ProfilingRequest] = []
        self._completed_jobs: Dict[str, ProfilingResult] = {}
        self._max_concurrent_jobs = 5
        self._resource_monitor_enabled = True
        self._auto_retry_enabled = True
        self._max_retry_attempts = 3

    async def submit_profiling_request(
        self,
        request: ProfilingRequest
    ) -> ProfilingJob:
        """Submit a profiling request for execution.
        
        Args:
            request: Profiling request to submit
            
        Returns:
            ProfilingJob that was created
        """
        # Validate request
        self._validate_request(request)
        
        # Create profiling job
        job = ProfilingJob(
            job_id=f"job_{request.request_id}",
            profile_id=request.dataset_id,
            dataset_source=request.dataset_source,
            profiling_config=self._convert_config_to_dict(request.configuration)
        )
        
        # Create profiling result
        result = ProfilingResult.create_initial(
            result_id=f"result_{request.request_id}",
            profile_id=request.dataset_id,
            dataset_id=request.dataset_id,
            configuration=request.configuration
        )
        
        # Create context
        context = ProfilingContext(
            request=request,
            job=job,
            result=result,
            timeline=[],
            resource_metrics=[],
            warnings=[],
            errors=[]
        )
        
        # Handle different execution modes
        if request.mode == ProfilingMode.SYNCHRONOUS:
            await self._execute_synchronous(context)
        elif request.mode == ProfilingMode.ASYNCHRONOUS:
            await self._queue_for_async_execution(context)
        elif request.mode == ProfilingMode.BATCH:
            await self._queue_for_batch_execution(context)
        elif request.mode == ProfilingMode.STREAMING:
            await self._start_streaming_profiling(context)
        elif request.mode == ProfilingMode.SCHEDULED:
            await self._schedule_profiling(context)
        
        return job

    async def cancel_profiling_job(self, job_id: str, reason: str = "") -> bool:
        """Cancel a running profiling job.
        
        Args:
            job_id: Job ID to cancel
            reason: Reason for cancellation
            
        Returns:
            True if job was cancelled, False otherwise
        """
        if job_id not in self._active_jobs:
            return False
        
        context = self._active_jobs[job_id]
        
        # Update job status
        context.job = dataclasses.replace(
            context.job,
            status=ProfilingStatus.CANCELLED,
            completed_at=datetime.now()
        )
        
        # Add cancellation to timeline
        cancellation_timeline = ExecutionTimeline(
            phase=ExecutionPhase.FINALIZATION,
            start_time=datetime.now(),
            end_time=datetime.now(),
            status=ProfilingStatus.CANCELLED,
            metadata={"cancellation_reason": reason}
        )
        context.timeline.append(cancellation_timeline)
        
        # Move to completed jobs
        self._completed_jobs[job_id] = context.result
        del self._active_jobs[job_id]
        
        return True

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a profiling job.
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Dictionary with job status information
        """
        # Check active jobs
        if job_id in self._active_jobs:
            context = self._active_jobs[job_id]
            return self._create_status_dict(context)
        
        # Check completed jobs
        if job_id in self._completed_jobs:
            result = self._completed_jobs[job_id]
            return {
                "job_id": job_id,
                "status": result.status.value,
                "completed_at": result.end_time,
                "duration_seconds": result.total_duration_seconds,
                "progress_percentage": 100.0,
                "result_id": result.result_id
            }
        
        # Check job queue
        for request in self._job_queue:
            if request.request_id == job_id:
                return {
                    "job_id": job_id,
                    "status": "queued",
                    "queue_position": self._job_queue.index(request) + 1,
                    "estimated_start_time": self._estimate_queue_wait_time(request)
                }
        
        return None

    async def list_active_jobs(self) -> List[Dict[str, Any]]:
        """List all active profiling jobs.
        
        Returns:
            List of active job information
        """
        return [
            self._create_status_dict(context)
            for context in self._active_jobs.values()
        ]

    async def get_job_metrics(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a profiling job.
        
        Args:
            job_id: Job ID to get metrics for
            
        Returns:
            Dictionary with detailed job metrics
        """
        if job_id not in self._active_jobs and job_id not in self._completed_jobs:
            return None
        
        if job_id in self._active_jobs:
            context = self._active_jobs[job_id]
            result = context.result
        else:
            result = self._completed_jobs[job_id]
        
        return {
            "job_id": job_id,
            "execution_metrics": {
                "total_duration_seconds": result.total_duration_seconds,
                "phases": [
                    {
                        "phase": timeline.phase.value,
                        "duration_seconds": timeline.duration_seconds,
                        "status": timeline.status.value,
                        "items_processed": timeline.items_processed,
                        "throughput_per_second": timeline.throughput_per_second
                    }
                    for timeline in result.execution_timeline
                ]
            },
            "resource_metrics": [
                {
                    "resource_type": metric.resource_type.value,
                    "peak_usage": metric.peak_usage,
                    "average_usage": metric.average_usage,
                    "unit": metric.unit,
                    "efficiency": metric.usage_efficiency
                }
                for metric in result.resource_metrics
            ],
            "data_metrics": {
                "rows_processed": result.total_rows_processed,
                "columns_analyzed": result.total_columns_analyzed,
                "processing_rate": result.processing_rate_rows_per_second,
                "sample_size": result.sample_size,
                "sampling_applied": result.sampling_applied
            },
            "quality_metrics": {
                "schema_elements": result.schema_elements_discovered,
                "patterns_found": result.patterns_discovered,
                "quality_issues": result.quality_issues_found,
                "relationships": result.relationships_identified
            },
            "error_metrics": {
                "warnings_count": len(result.warnings),
                "errors_count": len(result.errors),
                "success_rate": result.success_rate
            }
        }

    async def retry_failed_job(self, job_id: str) -> Optional[ProfilingJob]:
        """Retry a failed profiling job.
        
        Args:
            job_id: Job ID to retry
            
        Returns:
            New ProfilingJob if retry was successful, None otherwise
        """
        if not self._auto_retry_enabled:
            return None
        
        # Find the failed job in completed jobs
        if job_id not in self._completed_jobs:
            return None
        
        result = self._completed_jobs[job_id]
        if result.status != ProfilingStatus.FAILED:
            return None
        
        # Check retry count
        retry_count = result.metadata.get("retry_count", 0)
        if retry_count >= self._max_retry_attempts:
            return None
        
        # Create new request with retry metadata
        original_request = result.metadata.get("original_request")
        if not original_request:
            return None
        
        # Update retry metadata
        new_request_id = f"{original_request.request_id}_retry_{retry_count + 1}"
        retry_request = dataclasses.replace(
            original_request,
            request_id=new_request_id,
            metadata={
                **original_request.metadata,
                "retry_count": retry_count + 1,
                "original_job_id": job_id,
                "retry_reason": "automatic_retry_after_failure"
            }
        )
        
        return await self.submit_profiling_request(retry_request)

    async def optimize_job_queue(self) -> Dict[str, Any]:
        """Optimize the job queue based on priorities and dependencies.
        
        Returns:
            Dictionary with optimization results
        """
        if not self._job_queue:
            return {"message": "Queue is empty", "optimizations_applied": 0}
        
        original_order = [req.request_id for req in self._job_queue]
        
        # Sort by priority first
        priority_order = {
            ProfilingPriority.CRITICAL: 0,
            ProfilingPriority.HIGH: 1,
            ProfilingPriority.NORMAL: 2,
            ProfilingPriority.LOW: 3,
            ProfilingPriority.BACKGROUND: 4
        }
        
        # Create dependency graph
        dependency_graph = self._build_dependency_graph()
        
        # Topological sort considering priorities
        optimized_queue = self._topological_sort_with_priorities(
            self._job_queue, dependency_graph, priority_order
        )
        
        self._job_queue = optimized_queue
        new_order = [req.request_id for req in self._job_queue]
        
        optimizations_applied = sum(
            1 for i, req_id in enumerate(original_order)
            if i >= len(new_order) or new_order[i] != req_id
        )
        
        return {
            "message": "Queue optimization completed",
            "optimizations_applied": optimizations_applied,
            "original_order": original_order,
            "optimized_order": new_order,
            "dependency_graph": dependency_graph
        }

    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics.
        
        Returns:
            Dictionary with system health information
        """
        active_jobs_count = len(self._active_jobs)
        queued_jobs_count = len(self._job_queue)
        completed_jobs_count = len(self._completed_jobs)
        
        # Calculate resource utilization
        current_memory_usage = 0.0
        current_cpu_usage = 0.0
        
        for context in self._active_jobs.values():
            if context.resource_metrics:
                latest_memory = next(
                    (m for m in reversed(context.resource_metrics) 
                     if m.resource_type == ResourceType.MEMORY), None
                )
                latest_cpu = next(
                    (m for m in reversed(context.resource_metrics) 
                     if m.resource_type == ResourceType.CPU), None
                )
                
                if latest_memory:
                    current_memory_usage += latest_memory.current_usage
                if latest_cpu:
                    current_cpu_usage += latest_cpu.current_usage
        
        # Calculate success rate
        failed_jobs = sum(
            1 for result in self._completed_jobs.values()
            if result.status == ProfilingStatus.FAILED
        )
        success_rate = ((completed_jobs_count - failed_jobs) / max(1, completed_jobs_count)) * 100
        
        return {
            "system_status": "healthy" if active_jobs_count < self._max_concurrent_jobs else "busy",
            "job_statistics": {
                "active_jobs": active_jobs_count,
                "queued_jobs": queued_jobs_count,
                "completed_jobs": completed_jobs_count,
                "failed_jobs": failed_jobs,
                "success_rate_percentage": success_rate
            },
            "resource_utilization": {
                "memory_usage_mb": current_memory_usage,
                "cpu_usage_percentage": current_cpu_usage / max(1, active_jobs_count),
                "capacity_utilization_percentage": (active_jobs_count / self._max_concurrent_jobs) * 100
            },
            "queue_health": {
                "average_wait_time_minutes": self._calculate_average_wait_time(),
                "longest_waiting_job": self._get_longest_waiting_job(),
                "queue_efficiency_score": self._calculate_queue_efficiency()
            },
            "performance_metrics": {
                "average_job_duration_minutes": self._calculate_average_job_duration(),
                "throughput_jobs_per_hour": self._calculate_throughput(),
                "error_rate_percentage": (failed_jobs / max(1, completed_jobs_count)) * 100
            }
        }

    def _validate_request(self, request: ProfilingRequest) -> None:
        """Validate profiling request."""
        if not request.request_id:
            raise ValueError("Request ID is required")
        
        if not request.dataset_id:
            raise ValueError("Dataset ID is required")
        
        if not request.dataset_source:
            raise ValueError("Dataset source is required")
        
        if not request.configuration:
            raise ValueError("Profiling configuration is required")
        
        # Validate dependencies
        if request.dependencies:
            for dep_id in request.dependencies:
                if dep_id not in self._completed_jobs and dep_id not in self._active_jobs:
                    # Check if dependency is in queue
                    if not any(req.request_id == dep_id for req in self._job_queue):
                        raise ValueError(f"Dependency job {dep_id} not found")

    def _convert_config_to_dict(self, config: ProfilingConfiguration) -> Dict[str, Any]:
        """Convert profiling configuration to dictionary."""
        return {
            "strategy": config.strategy.value,
            "parallel_processing": config.parallel_processing,
            "max_workers": config.max_workers,
            "memory_limit_mb": config.memory_limit_mb,
            "timeout_minutes": config.timeout_minutes,
            "analysis_modules": config.analysis_modules,
            "complexity_score": config.estimated_complexity_score
        }

    def _create_status_dict(self, context: ProfilingContext) -> Dict[str, Any]:
        """Create status dictionary from context."""
        return {
            "job_id": context.job.job_id,
            "request_id": context.request.request_id,
            "dataset_id": context.request.dataset_id,
            "status": context.job.status.value,
            "priority": context.request.priority.value,
            "mode": context.request.mode.value,
            "progress_percentage": context.job.progress_percentage,
            "current_step": context.job.current_step,
            "estimated_completion": context.job.estimated_completion,
            "started_at": context.job.started_at,
            "warnings_count": len(context.warnings),
            "errors_count": len(context.errors),
            "memory_usage_mb": context.job.memory_usage_mb,
            "cpu_usage_percentage": context.job.cpu_usage_percentage
        }

    async def _execute_synchronous(self, context: ProfilingContext) -> None:
        """Execute profiling synchronously."""
        # This would integrate with actual profiling engines
        # For now, we'll simulate the execution flow
        context.job.status = ProfilingStatus.IN_PROGRESS
        context.job.started_at = datetime.now()
        
        # Add to active jobs for monitoring
        self._active_jobs[context.job.job_id] = context

    async def _queue_for_async_execution(self, context: ProfilingContext) -> None:
        """Queue profiling for asynchronous execution."""
        self._job_queue.append(context.request)
        
        # Start processing if capacity allows
        if len(self._active_jobs) < self._max_concurrent_jobs:
            await self._process_next_job()

    async def _queue_for_batch_execution(self, context: ProfilingContext) -> None:
        """Queue profiling for batch execution."""
        # Add batch-specific metadata
        context.request.metadata["execution_mode"] = "batch"
        self._job_queue.append(context.request)

    async def _start_streaming_profiling(self, context: ProfilingContext) -> None:
        """Start streaming profiling."""
        # Add streaming-specific configuration
        context.request.metadata["execution_mode"] = "streaming"
        self._active_jobs[context.job.job_id] = context

    async def _schedule_profiling(self, context: ProfilingContext) -> None:
        """Schedule profiling for later execution."""
        # Add scheduling metadata
        context.request.metadata["execution_mode"] = "scheduled"
        context.request.metadata["scheduled_time"] = datetime.now()
        self._job_queue.append(context.request)

    async def _process_next_job(self) -> None:
        """Process the next job in the queue."""
        if not self._job_queue or len(self._active_jobs) >= self._max_concurrent_jobs:
            return
        
        # Get next job based on priority and dependencies
        next_request = self._get_next_executable_request()
        if not next_request:
            return
        
        # Remove from queue
        self._job_queue.remove(next_request)
        
        # Create context and start execution
        # This would integrate with actual profiling execution logic
        pass

    def _get_next_executable_request(self) -> Optional[ProfilingRequest]:
        """Get the next executable request from the queue."""
        for request in self._job_queue:
            # Check if all dependencies are completed
            if self._are_dependencies_completed(request):
                return request
        return None

    def _are_dependencies_completed(self, request: ProfilingRequest) -> bool:
        """Check if all dependencies for a request are completed."""
        for dep_id in request.dependencies:
            if dep_id not in self._completed_jobs:
                return False
        return True

    def _estimate_queue_wait_time(self, request: ProfilingRequest) -> Optional[datetime]:
        """Estimate when a queued request will start execution."""
        position = self._job_queue.index(request)
        avg_job_duration = self._calculate_average_job_duration()
        
        if avg_job_duration > 0:
            estimated_wait_minutes = position * avg_job_duration
            return datetime.now() + timedelta(minutes=estimated_wait_minutes)
        
        return None

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph for job queue optimization."""
        graph = {}
        for request in self._job_queue:
            graph[request.request_id] = request.dependencies
        return graph

    def _topological_sort_with_priorities(
        self,
        requests: List[ProfilingRequest],
        dependency_graph: Dict[str, List[str]],
        priority_order: Dict[ProfilingPriority, int]
    ) -> List[ProfilingRequest]:
        """Perform topological sort considering priorities."""
        # Simple implementation - in practice, this would be more sophisticated
        return sorted(
            requests,
            key=lambda req: (
                priority_order.get(req.priority, 999),
                len(req.dependencies),
                req.request_id
            )
        )

    def _calculate_average_wait_time(self) -> float:
        """Calculate average wait time for jobs in queue."""
        if not self._job_queue:
            return 0.0
        
        total_wait_time = 0.0
        for request in self._job_queue:
            created_time = request.metadata.get("created_at", datetime.now())
            if isinstance(created_time, datetime):
                wait_time = (datetime.now() - created_time).total_seconds() / 60  # minutes
                total_wait_time += wait_time
        
        return total_wait_time / len(self._job_queue)

    def _get_longest_waiting_job(self) -> Optional[str]:
        """Get the job that has been waiting the longest."""
        if not self._job_queue:
            return None
        
        longest_wait_request = None
        longest_wait_time = 0.0
        
        for request in self._job_queue:
            created_time = request.metadata.get("created_at", datetime.now())
            if isinstance(created_time, datetime):
                wait_time = (datetime.now() - created_time).total_seconds()
                if wait_time > longest_wait_time:
                    longest_wait_time = wait_time
                    longest_wait_request = request
        
        return longest_wait_request.request_id if longest_wait_request else None

    def _calculate_queue_efficiency(self) -> float:
        """Calculate queue efficiency score."""
        if not self._job_queue:
            return 100.0
        
        # Simple efficiency calculation based on priority distribution
        high_priority_count = sum(
            1 for req in self._job_queue 
            if req.priority in [ProfilingPriority.CRITICAL, ProfilingPriority.HIGH]
        )
        
        return max(0.0, 100.0 - (high_priority_count / len(self._job_queue)) * 50)

    def _calculate_average_job_duration(self) -> float:
        """Calculate average job duration in minutes."""
        if not self._completed_jobs:
            return 30.0  # Default estimate
        
        durations = [
            result.total_duration_seconds / 60
            for result in self._completed_jobs.values()
            if result.total_duration_seconds
        ]
        
        return sum(durations) / len(durations) if durations else 30.0

    def _calculate_throughput(self) -> float:
        """Calculate jobs per hour throughput."""
        if not self._completed_jobs:
            return 0.0
        
        # Calculate based on jobs completed in the last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_jobs = [
            result for result in self._completed_jobs.values()
            if result.end_time and result.end_time >= one_hour_ago
        ]
        
        return len(recent_jobs)