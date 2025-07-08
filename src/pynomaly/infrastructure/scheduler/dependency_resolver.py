"""Dependency resolver for the scheduler system."""

from __future__ import annotations

from typing import List, Dict, Set

from .entities import JobDefinition


class DependencyResolver:
    """Resolves dependencies between jobs in a DAG."""

    @staticmethod
    def topological_sort(jobs: Dict[str, JobDefinition]) -> List[str]:
        """
        Perform topological sort on the DAG of jobs.

        Returns a list of job ids in the order they should be executed.
        Raises ValueError if a circular dependency is detected.
        """
        in_degree = {job_id: 0 for job_id in jobs}
        
        # Calculate in-degrees (how many jobs depend on each job)
        for job in jobs.values():
            for dep in job.depends_on:
                if dep in in_degree:
                    in_degree[job.job_id] += 1
                else:
                    raise ValueError(f"Job {job.job_id} depends on non-existent job {dep}")
        
        # Find nodes with no incoming edges
        queue = [job_id for job_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Remove edge from current to jobs that depend on it
            current_job = jobs[current]
            for dependent_job_id, dependent_job in jobs.items():
                if current in dependent_job.depends_on:
                    in_degree[dependent_job_id] -= 1
                    if in_degree[dependent_job_id] == 0:
                        queue.append(dependent_job_id)
        
        # Check for cycles
        if len(result) != len(jobs):
            raise ValueError("Circular dependency detected in DAG")
        
        return result

    @staticmethod
    def get_ready_jobs(jobs: Dict[str, JobDefinition], completed_jobs: Set[str]) -> List[str]:
        """
        Get jobs that are ready to be executed (all dependencies completed).

        Returns a list of job ids that can be started now.
        """
        ready = []
        for job_id, job in jobs.items():
            if job_id not in completed_jobs:
                if job.depends_on.issubset(completed_jobs):
                    ready.append(job_id)
        return ready

    @staticmethod
    def validate_dependencies(jobs: Dict[str, JobDefinition]) -> List[str]:
        """
        Validate all dependencies exist.

        Returns a list of error messages for invalid dependencies.
        """
        errors = []
        all_job_ids = set(jobs.keys())
        
        for job_id, job in jobs.items():
            for dep in job.depends_on:
                if dep not in all_job_ids:
                    errors.append(f"Job '{job_id}' depends on non-existent job '{dep}'")
                if dep == job_id:
                    errors.append(f"Job '{job_id}' cannot depend on itself")
        
        return errors
