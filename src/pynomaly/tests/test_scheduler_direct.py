# Test direct imports of advanced scheduler components

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from datetime import datetime
from pynomaly.infrastructure.scheduler.entities import Schedule, JobDefinition
from pynomaly.infrastructure.scheduler.dag_parser import DAGParser
from pynomaly.infrastructure.scheduler.dependency_resolver import DependencyResolver
from pynomaly.infrastructure.scheduler.resource_manager import ResourceManager
from pynomaly.infrastructure.scheduler.trigger_manager import TriggerManager
from pynomaly.infrastructure.scheduler.schedule_repository import InMemoryScheduleRepository

# Example test case

def test_scheduler_components():
    # Initialize components
    job1 = JobDefinition(job_id="1", name="Job 1")
    job2 = JobDefinition(job_id="2", name="Job 2", depends_on={"1"})
    jobs = {job1.job_id: job1, job2.job_id: job2}

    # Parse DAG
    dag = DAGParser.parse(jobs)
    assert len(dag) == 2

    # Detect cycles
    cycles = DAGParser.detect_cycles(jobs)
    assert len(cycles) == 0

    # Resolve dependencies
    ordered_jobs = DependencyResolver.topological_sort(jobs)
    assert ordered_jobs == ["1", "2"]

    # Allocate resources
    resource_manager = ResourceManager()
    can_allocate = resource_manager.can_allocate("1", job1.resource_requirements)
    assert can_allocate

    # Compute trigger
    next_execution = TriggerManager.compute_next_execution(cron_expression="* * * * *")
    assert isinstance(next_execution, datetime)

    print("All scheduler components imported and tested successfully.")

if __name__ == "__main__":
    test_scheduler_components()
