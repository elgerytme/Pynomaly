"""Resource manager for the scheduler system."""

from __future__ import annotations

from dataclasses import dataclass

from .entities import JobDefinition, ResourceRequirement


@dataclass
class ResourceUsage:
    """Current resource usage."""
    cpu_cores: float = 0.0
    memory_gb: float = 0.0
    workers: int = 0
    gpu_count: int = 0


@dataclass
class ResourceQuota:
    """Resource quota/limits."""
    max_cpu_cores: float = 8.0
    max_memory_gb: float = 16.0
    max_workers: int = 10
    max_gpu_count: int = 0


class ResourceManager:
    """Manages resource allocation and limits for job execution."""

    def __init__(self, quota: ResourceQuota | None = None):
        """Initialize resource manager with optional quota."""
        self.quota = quota or ResourceQuota()
        self.current_usage = ResourceUsage()
        self.allocated_jobs: dict[str, ResourceRequirement] = {}

    def can_allocate(self, job_id: str, requirement: ResourceRequirement) -> bool:
        """Check if resources can be allocated for a job."""
        if job_id in self.allocated_jobs:
            return True  # Already allocated

        # Calculate what usage would be after allocation
        projected_cpu = self.current_usage.cpu_cores + requirement.cpu_cores
        projected_memory = self.current_usage.memory_gb + requirement.memory_gb
        projected_workers = self.current_usage.workers + requirement.workers
        projected_gpu = self.current_usage.gpu_count + requirement.gpu_count

        # Check against quotas
        if projected_cpu > self.quota.max_cpu_cores:
            return False
        if projected_memory > self.quota.max_memory_gb:
            return False
        if projected_workers > self.quota.max_workers:
            return False
        if projected_gpu > self.quota.max_gpu_count:
            return False

        return True

    def allocate(self, job_id: str, requirement: ResourceRequirement) -> bool:
        """Allocate resources for a job."""
        if not self.can_allocate(job_id, requirement):
            return False

        if job_id not in self.allocated_jobs:
            self.current_usage.cpu_cores += requirement.cpu_cores
            self.current_usage.memory_gb += requirement.memory_gb
            self.current_usage.workers += requirement.workers
            self.current_usage.gpu_count += requirement.gpu_count
            self.allocated_jobs[job_id] = requirement

        return True

    def deallocate(self, job_id: str) -> bool:
        """Deallocate resources for a job."""
        if job_id not in self.allocated_jobs:
            return False

        requirement = self.allocated_jobs[job_id]
        self.current_usage.cpu_cores -= requirement.cpu_cores
        self.current_usage.memory_gb -= requirement.memory_gb
        self.current_usage.workers -= requirement.workers
        self.current_usage.gpu_count -= requirement.gpu_count

        del self.allocated_jobs[job_id]
        return True

    def get_available_resources(self) -> ResourceUsage:
        """Get currently available resources."""
        return ResourceUsage(
            cpu_cores=self.quota.max_cpu_cores - self.current_usage.cpu_cores,
            memory_gb=self.quota.max_memory_gb - self.current_usage.memory_gb,
            workers=self.quota.max_workers - self.current_usage.workers,
            gpu_count=self.quota.max_gpu_count - self.current_usage.gpu_count
        )

    def get_allocatable_jobs(self, jobs: dict[str, JobDefinition]) -> list[str]:
        """Get list of job IDs that can be allocated resources."""
        allocatable = []
        for job_id, job in jobs.items():
            if self.can_allocate(job_id, job.resource_requirements):
                allocatable.append(job_id)
        return allocatable

    def get_resource_utilization(self) -> dict[str, float]:
        """Get current resource utilization as percentages."""
        return {
            "cpu": (self.current_usage.cpu_cores / self.quota.max_cpu_cores) * 100 if self.quota.max_cpu_cores > 0 else 0,
            "memory": (self.current_usage.memory_gb / self.quota.max_memory_gb) * 100 if self.quota.max_memory_gb > 0 else 0,
            "workers": (self.current_usage.workers / self.quota.max_workers) * 100 if self.quota.max_workers > 0 else 0,
            "gpu": (self.current_usage.gpu_count / self.quota.max_gpu_count) * 100 if self.quota.max_gpu_count > 0 else 0
        }

    def reset(self) -> None:
        """Reset all resource allocations."""
        self.current_usage = ResourceUsage()
        self.allocated_jobs.clear()
