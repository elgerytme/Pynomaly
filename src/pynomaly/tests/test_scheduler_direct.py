#!/usr/bin/env python3
"""Test scheduler components functionality."""

from datetime import datetime

import pytest

from pynomaly.infrastructure.scheduler.dag_parser import DAGParser
from pynomaly.infrastructure.scheduler.dependency_resolver import DependencyResolver
from pynomaly.infrastructure.scheduler.entities import (
    JobDefinition,
    ResourceRequirement,
)
from pynomaly.infrastructure.scheduler.resource_manager import ResourceManager
from pynomaly.infrastructure.scheduler.trigger_manager import TriggerManager


def test_job_definition_creation():
    """Test creating job definitions."""
    job = JobDefinition(job_id="test-job", name="Test Job")
    assert job.job_id == "test-job"
    assert job.name == "Test Job"
    assert job.depends_on == set()
    assert isinstance(job.resource_requirements, ResourceRequirement)


def test_job_definition_with_dependencies():
    """Test job definition with dependencies."""
    job = JobDefinition(
        job_id="job2",
        name="Job 2",
        depends_on={"job1"}
    )
    assert job.depends_on == {"job1"}


def test_dag_parser():
    """Test DAG parsing functionality."""
    job1 = JobDefinition(job_id="1", name="Job 1")
    job2 = JobDefinition(job_id="2", name="Job 2", depends_on={"1"})
    jobs = {job1.job_id: job1, job2.job_id: job2}

    # Parse DAG
    dag = DAGParser.parse(jobs)
    assert len(dag) == 2
    assert dag["1"] == set()
    assert dag["2"] == {"1"}


def test_cycle_detection():
    """Test cycle detection in job dependencies."""
    job1 = JobDefinition(job_id="1", name="Job 1", depends_on={"2"})
    job2 = JobDefinition(job_id="2", name="Job 2", depends_on={"1"})
    jobs = {job1.job_id: job1, job2.job_id: job2}

    cycles = DAGParser.detect_cycles(jobs)
    assert len(cycles) > 0  # Should detect the cycle


def test_dependency_resolution():
    """Test dependency resolution and topological sorting."""
    job1 = JobDefinition(job_id="1", name="Job 1")
    job2 = JobDefinition(job_id="2", name="Job 2", depends_on={"1"})
    job3 = JobDefinition(job_id="3", name="Job 3", depends_on={"1", "2"})
    jobs = {job1.job_id: job1, job2.job_id: job2, job3.job_id: job3}

    ordered_jobs = DependencyResolver.topological_sort(jobs)

    # Job 1 should come before job 2, and both should come before job 3
    assert ordered_jobs.index("1") < ordered_jobs.index("2")
    assert ordered_jobs.index("1") < ordered_jobs.index("3")
    assert ordered_jobs.index("2") < ordered_jobs.index("3")


def test_resource_manager():
    """Test resource allocation and management."""
    resource_manager = ResourceManager()

    # Test basic allocation
    req = ResourceRequirement(cpu_cores=2.0, memory_gb=4.0)
    can_allocate = resource_manager.can_allocate("job1", req)
    assert can_allocate

    # Test allocation
    allocated = resource_manager.allocate("job1", req)
    assert allocated

    # Test deallocation
    deallocated = resource_manager.deallocate("job1")
    assert deallocated


def test_trigger_manager():
    """Test trigger computation."""
    # Test with cron expression (every minute)
    next_execution = TriggerManager.compute_next_execution(cron_expression="* * * * *")
    assert isinstance(next_execution, datetime)

    # Test with interval
    next_execution = TriggerManager.compute_next_execution(interval_seconds=3600)
    assert isinstance(next_execution, datetime)

    # Test cron validation
    assert TriggerManager.validate_cron_expression("0 9 * * *")  # Valid
    assert not TriggerManager.validate_cron_expression("invalid")  # Invalid


def test_ready_jobs():
    """Test getting ready jobs based on dependencies."""
    job1 = JobDefinition(job_id="1", name="Job 1")
    job2 = JobDefinition(job_id="2", name="Job 2", depends_on={"1"})
    job3 = JobDefinition(job_id="3", name="Job 3")
    jobs = {job1.job_id: job1, job2.job_id: job2, job3.job_id: job3}

    # Initially, jobs 1 and 3 should be ready (no dependencies)
    ready = DependencyResolver.get_ready_jobs(jobs, set())
    assert "1" in ready
    assert "3" in ready
    assert "2" not in ready

    # After job 1 completes, job 2 should be ready
    ready = DependencyResolver.get_ready_jobs(jobs, {"1"})
    assert "2" in ready
    assert "3" in ready


def test_execution_levels():
    """Test grouping jobs into execution levels."""
    job1 = JobDefinition(job_id="1", name="Job 1")
    job2 = JobDefinition(job_id="2", name="Job 2")
    job3 = JobDefinition(job_id="3", name="Job 3", depends_on={"1", "2"})
    jobs = {job1.job_id: job1, job2.job_id: job2, job3.job_id: job3}

    levels = DependencyResolver.get_execution_levels(jobs)

    # First level should have jobs 1 and 2 (can run in parallel)
    assert len(levels) == 2
    assert set(levels[0]) == {"1", "2"}
    assert levels[1] == ["3"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
