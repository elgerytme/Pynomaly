"""Dependency resolver module for job execution based on DAG."""

from __future__ import annotations

import logging
from typing import List, Dict, Set, Tuple, Optional

from .entities import JobDefinition, Schedule, JobStatus
from .dag_parser import DAGParser


logger = logging.getLogger(__name__)


class ExecutionStage:
    """Represents a stage in the execution plan."""
    
    def __init__(self, stage_number: int, job_ids: List[str]) -> None:
        """Initialize execution stage."""
        self.stage_number = stage_number
        self.job_ids = job_ids
    
    def __str__(self) -> str:
        """String representation."""
        return f"Stage {self.stage_number}: {', '.join(self.job_ids)}"


class ExecutionPlan:
    """Defines an execution plan detailing stages of job execution."""

    def __init__(self, stages: List[List[str]]) -> None:
        """Initialize execution plan with stages."""
        if not stages:
            raise ValueError("Stages must not be empty")
        self.stages = stages
        self.execution_stages = [
            ExecutionStage(i + 1, stage) for i, stage in enumerate(stages)
        ]

    def get_next_stage(self, completed_jobs: Set[str]) -> List[str]:
        """Get the next stage to execute (jobs whose dependencies are met)."""
        for stage in self.stages:
            if all(job_id in completed_jobs for job_id in stage):
                continue
            return [job_id for job_id in stage if job_id not in completed_jobs]
        return []

    def is_complete(self, completed_jobs: Set[str]) -> bool:
        """Check if the execution plan is complete."""
        return all(job_id in completed_jobs for stage in self.stages for job_id in stage)

    def get_stage_for_job(self, job_id: str) -> Optional[int]:
        """Get the stage number for a specific job."""
        for stage_num, stage in enumerate(self.stages):
            if job_id in stage:
                return stage_num
        return None


class DependencyGraph:
    """Represents the dependency graph for visualization and analysis."""
    
    def __init__(self, dag_parser: DAGParser) -> None:
        """Initialize dependency graph from DAG parser."""
        self.dag_parser = dag_parser
        self.nodes = dag_parser.nodes
        self.edges = dag_parser.edges
    
    def get_critical_path(self) -> List[str]:
        """Get the critical path through the DAG."""
        # Find the path with maximum depth
        max_depth = max(node.depth for node in self.nodes.values())
        
        # Build critical path from leaf to root
        critical_path = []
        current_depth = max_depth
        
        while current_depth >= 0:
            # Find nodes at this depth
            depth_nodes = [
                node for node in self.nodes.values() 
                if node.depth == current_depth
            ]
            
            if depth_nodes:
                # For simplicity, take the first node at this depth
                # In practice, you might want to consider resource requirements
                critical_path.append(depth_nodes[0].job_id)
            
            current_depth -= 1
        
        return list(reversed(critical_path))
    
    def get_parallelizable_jobs(self) -> Dict[int, List[str]]:
        """Get jobs that can be executed in parallel by depth."""
        return {
            depth: [node.job_id for node in nodes]
            for depth, nodes in self.dag_parser.get_nodes_by_depth().items()
        }


class DependencyResolver:
    """Resolves job dependencies and creates execution plans based on DAG."""

    def __init__(self, schedule: Schedule) -> None:
        """Initialize dependency resolver with a schedule."""
        self.schedule = schedule
        self.dag_parser = DAGParser()
        self.execution_plan: Optional[ExecutionPlan] = None
        self.dependency_graph: Optional[DependencyGraph] = None

    def resolve_dependencies(self) -> ExecutionPlan:
        """Resolve dependencies and create an execution plan."""
        logger.info(f"Resolving dependencies for schedule {self.schedule.schedule_id}")

        # Parse DAG graph
        self.dag_parser.parse_schedule(self.schedule)

        # Create dependency graph
        self.dependency_graph = DependencyGraph(self.dag_parser)

        # Get execution order
        stages = self.dag_parser.get_execution_order()

        # Create execution plan
        self.execution_plan = ExecutionPlan(stages=stages)

        logger.info(f"Created execution plan with {len(stages)} stages")
        return self.execution_plan

    def validate_and_plan_execution(self) -> ExecutionPlan:
        """Validate DAG and plan execution if valid."""
        if not self.execution_plan:
            self.resolve_dependencies()
        return self.execution_plan

    def can_execute_job(self, job_id: str, completed_jobs: Set[str]) -> bool:
        """Check if a specific job can be executed given completed jobs."""
        return self.dag_parser.can_run_job(job_id, completed_jobs)

    def get_ready_jobs(self, completed_jobs: Set[str]) -> List[str]:
        """Get list of jobs that are ready to be executed."""
        return self.dag_parser.get_ready_jobs(completed_jobs)

    def execution_plan_complete(self, completed_jobs: Set[str]) -> bool:
        """Check if the entire execution plan is complete."""
        if not self.execution_plan:
            return False
        return self.execution_plan.is_complete(completed_jobs)

    def get_job_dependencies(self, job_id: str) -> Set[str]:
        """Get direct dependencies for a job."""
        return self.dag_parser.get_dependencies_for_job(job_id)

    def get_job_dependents(self, job_id: str) -> Set[str]:
        """Get jobs that depend on this job."""
        return self.dag_parser.get_dependents_for_job(job_id)

    def get_critical_path(self) -> List[str]:
        """Get the critical path through the execution plan."""
        if not self.dependency_graph:
            raise ValueError("Dependencies not resolved yet")
        return self.dependency_graph.get_critical_path()

    def get_parallelizable_jobs(self) -> Dict[int, List[str]]:
        """Get jobs that can be executed in parallel."""
        if not self.dependency_graph:
            raise ValueError("Dependencies not resolved yet")
        return self.dependency_graph.get_parallelizable_jobs()

    def visualize_execution_plan(self) -> str:
        """Visualize the execution plan."""
        if not self.execution_plan:
            return "Execution plan not created"
        
        lines = []
        lines.append("Execution Plan:")
        lines.append("=" * 50)
        
        for stage_num, stage_jobs in enumerate(self.execution_plan.stages, 1):
            lines.append(f"Stage {stage_num}: {', '.join(stage_jobs)}")
        
        lines.append("")
        lines.append("DAG Visualization:")
        lines.append(self.dag_parser.visualize_dag())
        
        return "\n".join(lines)

    def get_execution_statistics(self) -> Dict[str, any]:
        """Get statistics about the execution plan."""
        if not self.execution_plan:
            return {}
        
        dag_stats = self.dag_parser.get_dag_statistics()
        
        return {
            "total_jobs": dag_stats["total_nodes"],
            "total_stages": len(self.execution_plan.stages),
            "max_parallelism": max(len(stage) for stage in self.execution_plan.stages),
            "dag_statistics": dag_stats,
        }
