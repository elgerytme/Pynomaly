"""DAG parser for the scheduler system."""

from __future__ import annotations

from typing import List, Dict, Tuple, Set

from .entities import JobDefinition


class DAGParser:
    """Parser for Directed Acyclic Graph of jobs."""

    @staticmethod
    def parse(jobs: Dict[str, JobDefinition]) -> List[Tuple[str, List[str]]]:
        """
        Parse job definitions and return a list of tuples with each job id and its dependencies.
        """
        dag = [(job_id, list(job.depends_on)) for job_id, job in jobs.items()]
        return dag

    @staticmethod
    def detect_cycles(jobs: Dict[str, JobDefinition]) -> Set[str]:
        """
        Detect circular dependencies in the DAG of jobs.

        Returns a set of job ids that are part of a cycle.
        """
        edges = {job_id: set(job.depends_on) for job_id, job in jobs.items()}
        visited = set()
        stack = set()
        cycle_nodes = set()

        def visit(node):
            if node in stack:
                cycle_nodes.add(node)
                return
            if node in visited:
                return
            visited.add(node)
            stack.add(node)
            for neighbor in edges.get(node, []):
                visit(neighbor)
            stack.remove(node)

        for job_id in jobs:
            visit(job_id)

        return cycle_nodes
