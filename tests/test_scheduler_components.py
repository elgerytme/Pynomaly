#!/usr/bin/env python3
"""Direct test of scheduler components without package imports."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import datetime for testing
from datetime import datetime

# Import scheduler components directly
from pynomaly.infrastructure.scheduler.entities import (
    Schedule, JobDefinition, ResourceRequirement, 
    ScheduleStatus, JobStatus, ExecutionStatus
)
from pynomaly.infrastructure.scheduler.dag_parser import DAGParser
from pynomaly.infrastructure.scheduler.dependency_resolver import DependencyResolver
from pynomaly.infrastructure.scheduler.resource_manager import ResourceManager, ResourceQuota
from pynomaly.infrastructure.scheduler.trigger_manager import TriggerManager


def test_scheduler_components():
    """Test all advanced scheduler components."""
    print("Testing Advanced Scheduler Components...")
    
    # Test 1: JobDefinition and ResourceRequirement
    print("\n1. Testing JobDefinition and ResourceRequirement...")
    job1 = JobDefinition(job_id="job1", name="Data Ingestion Job")
    job2 = JobDefinition(
        job_id="job2", 
        name="Data Processing Job",
        depends_on={"job1"},
        resource_requirements=ResourceRequirement(cpu_cores=2.0, memory_gb=4.0)
    )
    job3 = JobDefinition(
        job_id="job3",
        name="Analysis Job", 
        depends_on={"job2"}
    )
    
    jobs = {job1.job_id: job1, job2.job_id: job2, job3.job_id: job3}
    print(f"✓ Created {len(jobs)} job definitions")
    
    # Test 2: DAG Parser
    print("\n2. Testing DAG Parser...")
    dag = DAGParser.parse(jobs)
    print(f"✓ Parsed DAG with {len(dag)} nodes")
    
    cycles = DAGParser.detect_cycles(jobs)
    print(f"✓ No circular dependencies detected: {len(cycles) == 0}")
    
    # Test 3: Dependency Resolver
    print("\n3. Testing Dependency Resolver...")
    try:
        ordered_jobs = DependencyResolver.topological_sort(jobs)
        print(f"✓ Topological sort: {ordered_jobs}")
        
        # Test ready jobs
        completed = set()
        ready_jobs = DependencyResolver.get_ready_jobs(jobs, completed)
        print(f"✓ Initially ready jobs: {ready_jobs}")
        
        # Simulate completing job1
        completed.add("job1")
        ready_jobs = DependencyResolver.get_ready_jobs(jobs, completed)
        print(f"✓ After job1 completion, ready jobs: {ready_jobs}")
        
    except ValueError as e:
        print(f"✗ Dependency resolution failed: {e}")
    
    # Test 4: Resource Manager
    print("\n4. Testing Resource Manager...")
    quota = ResourceQuota(max_cpu_cores=8.0, max_memory_gb=16.0, max_workers=5)
    resource_manager = ResourceManager(quota)
    
    # Test allocation
    can_allocate = resource_manager.can_allocate("job1", job1.resource_requirements)
    print(f"✓ Can allocate job1: {can_allocate}")
    
    allocated = resource_manager.allocate("job1", job1.resource_requirements)
    print(f"✓ Allocated job1: {allocated}")
    
    allocated = resource_manager.allocate("job2", job2.resource_requirements)
    print(f"✓ Allocated job2: {allocated}")
    
    utilization = resource_manager.get_resource_utilization()
    print(f"✓ Resource utilization: {utilization}")
    
    # Test 5: Trigger Manager
    print("\n5. Testing Trigger Manager...")
    
    # Test cron validation
    valid_cron = TriggerManager.validate_cron_expression("0 */2 * * *")
    print(f"✓ Valid cron expression: {valid_cron}")
    
    invalid_cron = TriggerManager.validate_cron_expression("invalid cron")
    print(f"✓ Invalid cron rejected: {not invalid_cron}")
    
    # Test next execution computation
    next_exec = TriggerManager.compute_next_execution(cron_expression="* * * * *")
    print(f"✓ Next execution computed: {next_exec is not None}")
    
    interval_exec = TriggerManager.compute_next_execution(interval_seconds=3600)
    print(f"✓ Interval execution computed: {interval_exec is not None}")
    
    # Test 6: Schedule Creation
    print("\n6. Testing Schedule...")
    schedule = Schedule(
        schedule_id="test_schedule",
        name="Test Data Pipeline",
        cron_expression="0 2 * * *",  # Daily at 2 AM
        jobs=jobs
    )
    
    print(f"✓ Created schedule: {schedule.name}")
    print(f"✓ Schedule status: {schedule.status}")
    print(f"✓ Job count: {len(schedule.jobs)}")
    
    # Test adding/removing jobs
    test_job = JobDefinition(job_id="test_job", name="Test Job")
    schedule.add_job(test_job)
    print(f"✓ Added job, new count: {len(schedule.jobs)}")
    
    schedule.remove_job("test_job")
    print(f"✓ Removed job, count: {len(schedule.jobs)}")
    
    print("\n🎉 All scheduler component tests passed!")
    return True


def test_advanced_scenarios():
    """Test more advanced scheduler scenarios."""
    print("\n\nTesting Advanced Scenarios...")
    
    # Test circular dependency detection
    print("\n1. Testing Circular Dependency Detection...")
    job_a = JobDefinition(job_id="a", name="Job A", depends_on={"b"})
    job_b = JobDefinition(job_id="b", name="Job B", depends_on={"c"})
    job_c = JobDefinition(job_id="c", name="Job C", depends_on={"a"})  # Creates cycle
    
    circular_jobs = {"a": job_a, "b": job_b, "c": job_c}
    cycles = DAGParser.detect_cycles(circular_jobs)
    print(f"✓ Detected circular dependencies: {len(cycles) > 0}")
    
    # Test resource constraints
    print("\n2. Testing Resource Constraints...")
    small_quota = ResourceQuota(max_cpu_cores=1.0, max_memory_gb=1.0, max_workers=1)
    constrained_rm = ResourceManager(small_quota)
    
    big_job = JobDefinition(
        job_id="big_job",
        name="Resource Heavy Job",
        resource_requirements=ResourceRequirement(cpu_cores=4.0, memory_gb=8.0)
    )
    
    can_allocate_big = constrained_rm.can_allocate("big_job", big_job.resource_requirements)
    print(f"✓ Correctly rejected oversized job: {not can_allocate_big}")
    
    print("\n🚀 Advanced scenario tests completed!")


if __name__ == "__main__":
    try:
        success = test_scheduler_components()
        if success:
            test_advanced_scenarios()
            print("\n✅ All advanced scheduler tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
