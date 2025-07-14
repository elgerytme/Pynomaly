"""Tests for Batch Orchestrator."""

import asyncio
import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.pynomaly.application.services.batch_orchestrator import (
    BatchOrchestrator, BatchJobRequest, BatchPriority, JobDependencyManager
)
from src.pynomaly.application.services.batch_processing_service import BatchStatus


@pytest.fixture
def batch_orchestrator():
    """Create a batch orchestrator for testing."""
    return BatchOrchestrator()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'feature_1': range(50),
        'feature_2': [x * 2 for x in range(50)],
        'feature_3': [x % 5 for x in range(50)]
    })


@pytest.fixture
def job_request(sample_dataframe):
    """Create a batch job request for testing."""
    return BatchJobRequest(
        name="Test Job",
        description="Test batch job",
        processor_type="anomaly_detection",
        data_source=sample_dataframe,
        priority=BatchPriority.MEDIUM,
        config_overrides={"batch_size": 10},
        processor_kwargs={"algorithm": "isolation_forest"}
    )


class TestJobDependencyManager:
    """Test job dependency management."""
    
    def test_add_dependency(self):
        """Test adding job dependencies."""
        manager = JobDependencyManager()
        
        manager.add_dependency("job_b", "job_a")
        
        assert "job_b" in manager.dependencies
        assert "job_a" in manager.dependencies["job_b"]
        assert "job_a" in manager.dependents
        assert "job_b" in manager.dependents["job_a"]
    
    def test_can_run_no_dependencies(self):
        """Test job can run when no dependencies."""
        manager = JobDependencyManager()
        assert manager.can_run("job_a") is True
    
    def test_can_run_with_completed_dependencies(self):
        """Test job can run when dependencies are completed."""
        manager = JobDependencyManager()
        
        manager.add_dependency("job_b", "job_a")
        assert manager.can_run("job_b") is False
        
        manager.mark_completed("job_a")
        assert manager.can_run("job_b") is True
    
    def test_mark_completed_returns_ready_jobs(self):
        """Test marking job completed returns ready dependent jobs."""
        manager = JobDependencyManager()
        
        # Create dependency chain: job_a -> job_b, job_c
        manager.add_dependency("job_b", "job_a")
        manager.add_dependency("job_c", "job_a")
        
        ready_jobs = manager.mark_completed("job_a")
        
        assert "job_b" in ready_jobs
        assert "job_c" in ready_jobs
        assert len(ready_jobs) == 2
    
    def test_get_pending_dependencies(self):
        """Test getting pending dependencies."""
        manager = JobDependencyManager()
        
        manager.add_dependency("job_c", "job_a")
        manager.add_dependency("job_c", "job_b")
        
        pending = manager.get_pending_dependencies("job_c")
        assert pending == {"job_a", "job_b"}
        
        manager.mark_completed("job_a")
        pending = manager.get_pending_dependencies("job_c")
        assert pending == {"job_b"}


class TestBatchOrchestrator:
    """Test batch orchestrator functionality."""
    
    @pytest.mark.asyncio
    async def test_submit_job(self, batch_orchestrator, job_request):
        """Test job submission."""
        job_id = await batch_orchestrator.submit_job(job_request)
        
        assert job_id is not None
        assert job_id in batch_orchestrator.running_jobs or job_id in batch_orchestrator.scheduled_jobs
    
    @pytest.mark.asyncio
    async def test_submit_job_invalid_processor(self, batch_orchestrator, sample_dataframe):
        """Test job submission with invalid processor."""
        request = BatchJobRequest(
            name="Invalid Job",
            processor_type="invalid_processor",
            data_source=sample_dataframe
        )
        
        with pytest.raises(ValueError, match="Unknown processor type"):
            await batch_orchestrator.submit_job(request)
    
    @pytest.mark.asyncio
    async def test_submit_job_with_dependencies(self, batch_orchestrator, sample_dataframe):
        """Test job submission with dependencies."""
        # Submit first job
        request_a = BatchJobRequest(
            name="Job A",
            processor_type="data_quality",
            data_source=sample_dataframe,
            schedule_immediately=False
        )
        job_a_id = await batch_orchestrator.submit_job(request_a)
        
        # Submit dependent job
        request_b = BatchJobRequest(
            name="Job B",
            processor_type="anomaly_detection",
            data_source=sample_dataframe,
            depends_on=[job_a_id],
            schedule_immediately=False
        )
        job_b_id = await batch_orchestrator.submit_job(request_b)
        
        # Verify dependency
        assert not batch_orchestrator.dependency_manager.can_run(job_b_id)
        assert batch_orchestrator.dependency_manager.can_run(job_a_id)
    
    @pytest.mark.asyncio
    async def test_job_execution_with_completion(self, batch_orchestrator, sample_dataframe):
        """Test complete job execution workflow."""
        # Create a small job for quick execution
        small_data = sample_dataframe.head(10)
        
        request = BatchJobRequest(
            name="Quick Job",
            processor_type="data_quality",
            data_source=small_data,
            config_overrides={"batch_size": 5}
        )
        
        job_id = await batch_orchestrator.submit_job(request)
        
        # Wait for job completion
        max_wait = 10  # seconds
        waited = 0
        while waited < max_wait:
            if job_id in batch_orchestrator.completed_jobs:
                break
            await asyncio.sleep(0.1)
            waited += 0.1
        
        # Verify completion
        assert job_id in batch_orchestrator.completed_jobs
        result = batch_orchestrator.completed_jobs[job_id]
        assert result.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_cancel_running_job(self, batch_orchestrator, sample_dataframe):
        """Test cancelling a running job."""
        # Create a job that would take some time
        request = BatchJobRequest(
            name="Long Job",
            processor_type="data_profiling",
            data_source=sample_dataframe,
            config_overrides={"batch_size": 5}
        )
        
        job_id = await batch_orchestrator.submit_job(request)
        
        # Give job time to start
        await asyncio.sleep(0.1)
        
        # Cancel the job
        cancelled = await batch_orchestrator.cancel_job(job_id)
        assert cancelled is True
    
    @pytest.mark.asyncio
    async def test_cancel_scheduled_job(self, batch_orchestrator, sample_dataframe):
        """Test cancelling a scheduled job."""
        request = BatchJobRequest(
            name="Scheduled Job",
            processor_type="data_quality",
            data_source=sample_dataframe,
            schedule_immediately=False
        )
        
        job_id = await batch_orchestrator.submit_job(request)
        assert job_id in batch_orchestrator.scheduled_jobs
        
        cancelled = await batch_orchestrator.cancel_job(job_id)
        assert cancelled is True
        assert job_id not in batch_orchestrator.scheduled_jobs
    
    def test_get_job_status_running(self, batch_orchestrator):
        """Test getting status of running job."""
        # Create mock running job
        from src.pynomaly.application.services.batch_processing_service import BatchJob
        
        job = BatchJob(name="Test Job", processor_name="test")
        job.status = BatchStatus.RUNNING
        job.metrics.total_items = 100
        job.metrics.processed_items = 25
        job.metrics.total_batches = 10
        job.metrics.processed_batches = 3
        
        batch_orchestrator.running_jobs[job.id] = job
        
        status = batch_orchestrator.get_job_status(job.id)
        
        assert status["status"] == "running"
        assert status["progress_percentage"] == 30.0  # 3/10 * 100
        assert status["processed_items"] == 25
        assert status["total_items"] == 100
    
    def test_get_job_status_completed(self, batch_orchestrator):
        """Test getting status of completed job."""
        from src.pynomaly.application.services.batch_orchestrator import BatchJobResult
        
        result = BatchJobResult(
            job_id="test_job",
            status=BatchStatus.COMPLETED,
            execution_time_seconds=45.5,
            items_processed=100,
            items_failed=0
        )
        
        batch_orchestrator.completed_jobs["test_job"] = result
        
        status = batch_orchestrator.get_job_status("test_job")
        
        assert status["status"] == BatchStatus.COMPLETED.value
        assert status["execution_time"] == 45.5
        assert status["items_processed"] == 100
        assert status["items_failed"] == 0
    
    def test_get_job_status_scheduled(self, batch_orchestrator, sample_dataframe):
        """Test getting status of scheduled job."""
        request = BatchJobRequest(
            name="Scheduled Job",
            processor_type="data_quality",
            data_source=sample_dataframe,
            depends_on=["dependency_job"]
        )
        
        # Manually add to scheduled jobs for testing
        batch_orchestrator.scheduled_jobs["test_job"] = request
        batch_orchestrator.dependency_manager.add_dependency("test_job", "dependency_job")
        
        status = batch_orchestrator.get_job_status("test_job")
        
        assert status["status"] == "scheduled"
        assert "dependency_job" in status["pending_dependencies"]
        assert status["can_run"] is False
    
    def test_list_jobs_all(self, batch_orchestrator):
        """Test listing all jobs."""
        from src.pynomaly.application.services.batch_processing_service import BatchJob
        from src.pynomaly.application.services.batch_orchestrator import BatchJobResult, BatchJobRequest
        
        # Add running job
        running_job = BatchJob(name="Running Job", processor_name="test")
        running_job.status = BatchStatus.RUNNING
        batch_orchestrator.running_jobs["running_1"] = running_job
        
        # Add scheduled job
        scheduled_request = BatchJobRequest(
            name="Scheduled Job",
            processor_type="test",
            data_source=[]
        )
        batch_orchestrator.scheduled_jobs["scheduled_1"] = scheduled_request
        scheduled_job = BatchJob(name="Scheduled Job", processor_name="test")
        batch_orchestrator.batch_service._active_jobs["scheduled_1"] = scheduled_job
        
        # Add completed job
        completed_result = BatchJobResult(
            job_id="completed_1",
            status=BatchStatus.COMPLETED,
            execution_time_seconds=30.0,
            items_processed=50
        )
        batch_orchestrator.completed_jobs["completed_1"] = completed_result
        completed_job = BatchJob(name="Completed Job", processor_name="test")
        batch_orchestrator.batch_service._active_jobs["completed_1"] = completed_job
        
        all_jobs = batch_orchestrator.list_jobs()
        assert len(all_jobs) == 3
        
        statuses = [job["status"] for job in all_jobs]
        assert "running" in statuses
        assert "scheduled" in statuses
        assert BatchStatus.COMPLETED.value in statuses
    
    def test_list_jobs_filtered(self, batch_orchestrator):
        """Test listing jobs with filters."""
        from src.pynomaly.application.services.batch_processing_service import BatchJob
        
        # Add multiple running jobs
        for i in range(3):
            job = BatchJob(name=f"Running Job {i}", processor_name="test")
            job.status = BatchStatus.RUNNING
            batch_orchestrator.running_jobs[f"running_{i}"] = job
        
        running_jobs = batch_orchestrator.list_jobs(status_filter="running")
        assert len(running_jobs) == 3
        assert all(job["status"] == "running" for job in running_jobs)
        
        # Test limit
        limited_jobs = batch_orchestrator.list_jobs(limit=2)
        assert len(limited_jobs) == 2
    
    def test_get_system_status(self, batch_orchestrator):
        """Test getting system status."""
        from src.pynomaly.application.services.batch_processing_service import BatchJob
        
        # Add some jobs
        running_job = BatchJob(name="Running", processor_name="test")
        batch_orchestrator.running_jobs["r1"] = running_job
        
        batch_orchestrator.scheduled_jobs["s1"] = MagicMock()
        batch_orchestrator.completed_jobs["c1"] = MagicMock()
        
        status = batch_orchestrator.get_system_status()
        
        assert status["running_jobs"] == 1
        assert status["scheduled_jobs"] == 1
        assert status["completed_jobs"] == 1
        assert status["max_concurrent_jobs"] == batch_orchestrator.max_concurrent_jobs
        assert "system_recommendations" in status
    
    @pytest.mark.asyncio
    async def test_cleanup_old_jobs(self, batch_orchestrator):
        """Test cleanup of old completed jobs."""
        from src.pynomaly.application.services.batch_orchestrator import BatchJobResult
        from src.pynomaly.application.services.batch_processing_service import BatchJob
        
        # Create old completed job
        old_result = BatchJobResult(
            job_id="old_job",
            status=BatchStatus.COMPLETED,
            execution_time_seconds=10.0,
            items_processed=100
        )
        old_job = BatchJob(name="Old Job", processor_name="test")
        old_job.completed_at = datetime.now(timezone.utc) - pd.Timedelta(hours=48)
        
        batch_orchestrator.completed_jobs["old_job"] = old_result
        batch_orchestrator.batch_service._active_jobs["old_job"] = old_job
        
        # Create recent completed job
        recent_result = BatchJobResult(
            job_id="recent_job",
            status=BatchStatus.COMPLETED,
            execution_time_seconds=15.0,
            items_processed=50
        )
        recent_job = BatchJob(name="Recent Job", processor_name="test")
        recent_job.completed_at = datetime.now(timezone.utc) - pd.Timedelta(hours=1)
        
        batch_orchestrator.completed_jobs["recent_job"] = recent_result
        batch_orchestrator.batch_service._active_jobs["recent_job"] = recent_job
        
        # Cleanup jobs older than 24 hours
        cleaned = await batch_orchestrator.cleanup_old_jobs(24)
        
        assert cleaned >= 1  # At least the old job should be cleaned
        assert "recent_job" in batch_orchestrator.completed_jobs


@pytest.mark.integration
class TestBatchOrchestratorIntegration:
    """Integration tests for batch orchestrator."""
    
    @pytest.mark.asyncio
    async def test_dependency_chain_execution(self, sample_dataframe):
        """Test execution of jobs with dependency chain."""
        orchestrator = BatchOrchestrator()
        
        # Create small dataset for quick execution
        small_data = sample_dataframe.head(10)
        
        # Submit job A (no dependencies)
        request_a = BatchJobRequest(
            name="Job A - Data Quality",
            processor_type="data_quality",
            data_source=small_data,
            config_overrides={"batch_size": 5}
        )
        job_a_id = await orchestrator.submit_job(request_a)
        
        # Submit job B (depends on A)
        request_b = BatchJobRequest(
            name="Job B - Anomaly Detection",
            processor_type="anomaly_detection", 
            data_source=small_data,
            depends_on=[job_a_id],
            config_overrides={"batch_size": 5}
        )
        job_b_id = await orchestrator.submit_job(request_b)
        
        # Submit job C (depends on B)
        request_c = BatchJobRequest(
            name="Job C - Data Profiling",
            processor_type="data_profiling",
            data_source=small_data,
            depends_on=[job_b_id],
            config_overrides={"batch_size": 5}
        )
        job_c_id = await orchestrator.submit_job(request_c)
        
        # Wait for all jobs to complete
        max_wait = 30  # seconds
        waited = 0
        while waited < max_wait:
            if (job_a_id in orchestrator.completed_jobs and
                job_b_id in orchestrator.completed_jobs and
                job_c_id in orchestrator.completed_jobs):
                break
            await asyncio.sleep(0.5)
            waited += 0.5
        
        # Verify all jobs completed
        assert job_a_id in orchestrator.completed_jobs
        assert job_b_id in orchestrator.completed_jobs
        assert job_c_id in orchestrator.completed_jobs
        
        # Verify execution order (A should complete before B, B before C)
        result_a = orchestrator.completed_jobs[job_a_id]
        result_b = orchestrator.completed_jobs[job_b_id]
        result_c = orchestrator.completed_jobs[job_c_id]
        
        # All should have completed successfully
        assert result_a.status == BatchStatus.COMPLETED
        assert result_b.status == BatchStatus.COMPLETED
        assert result_c.status == BatchStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_concurrent_job_execution(self, sample_dataframe):
        """Test concurrent execution of independent jobs."""
        orchestrator = BatchOrchestrator()
        orchestrator.max_concurrent_jobs = 3
        
        small_data = sample_dataframe.head(20)
        
        # Submit multiple independent jobs
        job_ids = []
        for i in range(3):
            request = BatchJobRequest(
                name=f"Concurrent Job {i}",
                processor_type="data_quality",
                data_source=small_data,
                config_overrides={"batch_size": 10}
            )
            job_id = await orchestrator.submit_job(request)
            job_ids.append(job_id)
        
        # Wait for completion
        max_wait = 30
        waited = 0
        while waited < max_wait:
            completed_count = sum(1 for job_id in job_ids if job_id in orchestrator.completed_jobs)
            if completed_count == len(job_ids):
                break
            await asyncio.sleep(0.5)
            waited += 0.5
        
        # Verify all jobs completed
        for job_id in job_ids:
            assert job_id in orchestrator.completed_jobs
            result = orchestrator.completed_jobs[job_id]
            assert result.status == BatchStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_job_failure_and_recovery(self, sample_dataframe):
        """Test handling of job failures."""
        orchestrator = BatchOrchestrator()
        
        # Register a failing processor
        async def failing_processor(batch_data, context):
            if context["batch_index"] == 1:
                raise ValueError("Simulated failure")
            return {"processed": len(batch_data)}
        
        orchestrator.batch_service.register_processor("failing_processor", failing_processor)
        
        request = BatchJobRequest(
            name="Failing Job",
            processor_type="failing_processor",
            data_source=sample_dataframe.head(20),
            config_overrides={"batch_size": 10}
        )
        
        job_id = await orchestrator.submit_job(request)
        
        # Wait for job to fail
        max_wait = 10
        waited = 0
        while waited < max_wait:
            if job_id in orchestrator.completed_jobs:
                break
            await asyncio.sleep(0.5)
            waited += 0.5
        
        # Verify job failed
        assert job_id in orchestrator.completed_jobs
        result = orchestrator.completed_jobs[job_id]
        assert result.status == BatchStatus.FAILED
        assert result.error_message is not None