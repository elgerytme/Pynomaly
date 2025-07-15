"""DAG parser for dependency analysis."""

from __future__ import annotations

from .entities import JobDefinition


class DAGParser:
    """Parser for analyzing job dependency DAGs."""

    @staticmethod
    def parse(jobs: dict[str, JobDefinition]) -> dict[str, set[str]]:
        """Parse jobs into a dependency graph."""
        dag = {}
        for job_id, job in jobs.items():
            dag[job_id] = job.depends_on.copy()
        return dag

    @staticmethod
    def detect_cycles(jobs: dict[str, JobDefinition]) -> list[list[str]]:
        """Detect circular dependencies in the job graph."""
        dag = DAGParser.parse(jobs)
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)

            for neighbor in dag.get(node, set()):
                if neighbor in dag:  # Only follow valid job dependencies
                    dfs(neighbor, path + [node])

            rec_stack.remove(node)

        for job_id in jobs:
            if job_id not in visited:
                dfs(job_id, [])

        return cycles

    @staticmethod
    def validate_dependencies(jobs: dict[str, JobDefinition]) -> list[str]:
        """Validate that all dependencies reference existing jobs."""
        errors = []
        job_ids = set(jobs.keys())

        for job_id, job in jobs.items():
            for dep in job.depends_on:
                if dep not in job_ids:
                    errors.append(f"Job '{job_id}' depends on non-existent job '{dep}'")

        return errors
