"""Dependency resolution for job scheduling."""

from __future__ import annotations

from collections import deque

from .entities import JobDefinition


class DependencyResolver:
    """Resolves job dependencies and provides execution order."""

    @staticmethod
    def topological_sort(jobs: dict[str, JobDefinition]) -> list[str]:
        """Perform topological sort to get execution order."""
        # Build adjacency list (reverse of dependencies)
        graph: dict[str, set[str]] = {job_id: set() for job_id in jobs}
        in_degree: dict[str, int] = dict.fromkeys(jobs, 0)

        # Build the graph: if A depends on B, then B -> A
        for job_id, job in jobs.items():
            for dep in job.depends_on:
                if dep in graph:  # Only consider valid dependencies
                    graph[dep].add(job_id)
                    in_degree[job_id] += 1

        # Kahn's algorithm for topological sorting
        queue = deque([job_id for job_id in jobs if in_degree[job_id] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(jobs):
            raise ValueError("Circular dependency detected - cannot sort")

        return result

    @staticmethod
    def get_ready_jobs(
        jobs: dict[str, JobDefinition],
        completed_jobs: set[str]
    ) -> list[str]:
        """Get jobs that are ready to run (all dependencies completed)."""
        ready = []

        for job_id, job in jobs.items():
            if job_id in completed_jobs:
                continue  # Already completed

            # Check if all dependencies are completed
            if job.depends_on.issubset(completed_jobs):
                ready.append(job_id)

        return ready

    @staticmethod
    def get_execution_levels(jobs: dict[str, JobDefinition]) -> list[list[str]]:
        """Group jobs into execution levels (jobs in same level can run in parallel)."""
        sorted_jobs = DependencyResolver.topological_sort(jobs)
        levels = []
        completed = set()

        while len(completed) < len(jobs):
            current_level = []

            for job_id in sorted_jobs:
                if job_id in completed:
                    continue

                # Check if all dependencies are completed
                job = jobs[job_id]
                if job.depends_on.issubset(completed):
                    current_level.append(job_id)

            if not current_level:
                raise ValueError("Unable to resolve dependencies - possible circular dependency")

            levels.append(current_level)
            completed.update(current_level)

        return levels
