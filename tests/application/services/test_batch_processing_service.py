"""Tests for Batch Processing Service."""

import asyncio
import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.pynomaly.application.services.batch_processing_service import (
    BatchProcessingService, BatchJob, BatchStatus, BatchPriority, BatchConfig,
    anomaly_detection_batch_processor, data_quality_batch_processor
)


@pytest.fixture
def batch_service():
    """Create a batch processing service for testing."""
    return BatchProcessingService()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'feature_1': range(100),
        'feature_2': [x * 2 for x in range(100)],
        'feature_3': [x % 10 for x in range(100)]
    })


@pytest.fixture
def batch_config():
    """Create a batch configuration for testing."""
    return BatchConfig(
        batch_size=10,
        max_concurrent_batches=2,
        memory_limit_mb=100.0,
        timeout_seconds=60,
        retry_attempts=2
    )


class TestBatchProcessingService:
    """Test batch processing service functionality."""
    
    def test_register_processor(self, batch_service):
        """Test processor registration."""
        async def test_processor(data, context):
            return {"processed": len(data)}
        
        batch_service.register_processor("test_processor", test_processor)
        assert "test_processor" in batch_service._processors
    
    @pytest.mark.asyncio
    async def test_create_batch_job(self, batch_service, sample_dataframe, batch_config):
        """Test batch job creation."""
        # Register a test processor
        async def test_processor(data, context):
            return {"processed": len(data)}
        
        batch_service.register_processor("test_processor", test_processor)
        
        # Create job
        job = await batch_service.create_batch_job(
            name="Test Job",
            data=sample_dataframe,
            processor_name="test_processor",
            config=batch_config,
            description="Test batch job"
        )
        
        assert job.name == "Test Job"
        assert job.status == BatchStatus.PENDING
        assert job.metrics.total_items == 100
        assert job.metrics.total_batches == 10  # 100 items / 10 batch_size
        assert job.id in batch_service._active_jobs
    
    @pytest.mark.asyncio
    async def test_create_batch_job_invalid_processor(self, batch_service, sample_dataframe):
        """Test batch job creation with invalid processor."""
        with pytest.raises(ValueError, match="Processor 'invalid' not registered"):
            await batch_service.create_batch_job(
                name="Test Job",
                data=sample_dataframe,
                processor_name="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_start_batch_job(self, batch_service, batch_config):
        """Test starting a batch job."""
        # Create simple test data
        test_data = list(range(20))
        
        # Mock processor that processes data
        async def test_processor(batch_data, context):
            await asyncio.sleep(0.01)  # Simulate processing
            return {
                "batch_index": context["batch_index"],
                "processed_count": len(batch_data)
            }
        
        batch_service.register_processor("test_processor", test_processor)
        
        # Create job
        job = await batch_service.create_batch_job(
            name="Test Job",
            data=test_data,
            processor_name="test_processor",
            config=batch_config
        )
        
        # Start job
        await batch_service.start_batch_job(job.id)
        
        # Verify job completion
        assert job.status == BatchStatus.COMPLETED
        assert job.metrics.processed_batches == job.metrics.total_batches
        assert job.metrics.processed_items == 20
        assert job.started_at is not None
        assert job.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_batch_job_with_failure(self, batch_service, batch_config):
        """Test batch job handling with failures."""
        test_data = list(range(20))
        
        # Mock processor that fails on specific batches
        async def failing_processor(batch_data, context):
            if context["batch_index"] == 1:  # Fail on second batch
                raise ValueError("Test failure")
            return {"processed": len(batch_data)}
        
        batch_service.register_processor("failing_processor", failing_processor)
        
        job = await batch_service.create_batch_job(
            name="Failing Job",
            data=test_data,
            processor_name="failing_processor",
            config=batch_config
        )
        
        # Start job and expect failure
        with pytest.raises(ValueError, match="Test failure"):
            await batch_service.start_batch_job(job.id)
        
        assert job.status == BatchStatus.FAILED
        assert job.last_error == "Test failure"
    
    def test_create_batches_dataframe(self, batch_service):
        """Test batch creation from DataFrame."""
        df = pd.DataFrame({'col': range(25)})
        batches = batch_service._create_batches(df, 10)
        
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5
    
    def test_create_batches_list(self, batch_service):
        """Test batch creation from list."""
        data = list(range(25))
        batches = batch_service._create_batches(data, 10)
        
        assert len(batches) == 3
        assert batches[0] == list(range(10))
        assert batches[1] == list(range(10, 20))
        assert batches[2] == list(range(20, 25))
    
    @pytest.mark.asyncio
    async def test_pause_and_resume_job(self, batch_service):
        """Test pausing and resuming a job."""
        # Create a long-running job
        test_data = list(range(100))
        
        async def slow_processor(batch_data, context):
            await asyncio.sleep(0.1)
            return {"processed": len(batch_data)}
        
        batch_service.register_processor("slow_processor", slow_processor)
        
        job = await batch_service.create_batch_job(
            name="Slow Job",
            data=test_data,
            processor_name="slow_processor",
            config=BatchConfig(batch_size=10, max_concurrent_batches=1)
        )
        
        # Start job in background
        job_task = asyncio.create_task(batch_service.start_batch_job(job.id))
        
        # Give it time to start
        await asyncio.sleep(0.05)
        
        # Pause job
        await batch_service.pause_job(job.id)
        assert job.status == BatchStatus.PAUSED
        
        # Cancel the running task
        job_task.cancel()
        try:
            await job_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_cancel_job(self, batch_service, sample_dataframe):
        """Test job cancellation."""
        async def test_processor(batch_data, context):
            return {"processed": len(batch_data)}
        
        batch_service.register_processor("test_processor", test_processor)
        
        job = await batch_service.create_batch_job(
            name="Test Job",
            data=sample_dataframe,
            processor_name="test_processor"
        )
        
        await batch_service.cancel_job(job.id)
        assert job.status == BatchStatus.CANCELLED
        assert job.completed_at is not None
    
    def test_get_job_status(self, batch_service):
        """Test getting job status."""
        # Create a job manually for testing
        job = BatchJob(
            name="Test Job",
            processor_name="test",
            input_data=[1, 2, 3]
        )
        batch_service._active_jobs[job.id] = job
        
        status = batch_service.get_job_status(job.id)
        assert status is not None
        assert status.id == job.id
        
        # Test non-existent job
        assert batch_service.get_job_status("non-existent") is None
    
    def test_list_jobs(self, batch_service):
        """Test listing jobs with filters."""
        # Create test jobs
        job1 = BatchJob(name="Job 1", processor_name="test", status=BatchStatus.RUNNING)
        job2 = BatchJob(name="Job 2", processor_name="test", status=BatchStatus.COMPLETED)
        job3 = BatchJob(name="Job 3", processor_name="test", status=BatchStatus.FAILED, priority=BatchPriority.HIGH)
        
        batch_service._active_jobs[job1.id] = job1
        batch_service._active_jobs[job2.id] = job2
        batch_service._active_jobs[job3.id] = job3
        
        # Test no filter
        all_jobs = batch_service.list_jobs()
        assert len(all_jobs) == 3
        
        # Test status filter
        running_jobs = batch_service.list_jobs(status_filter=BatchStatus.RUNNING)
        assert len(running_jobs) == 1
        assert running_jobs[0].status == BatchStatus.RUNNING
        
        # Test priority filter
        high_priority_jobs = batch_service.list_jobs(priority_filter=BatchPriority.HIGH)
        assert len(high_priority_jobs) == 1
        assert high_priority_jobs[0].priority == BatchPriority.HIGH
    
    def test_add_progress_callback(self, batch_service):
        """Test adding progress callbacks."""
        callback_called = []
        
        def test_callback(job_id, current, total, message=""):
            callback_called.append((job_id, current, total, message))
        
        batch_service.add_progress_callback("test_job", test_callback)
        
        # Trigger callback
        batch_service._notify_progress("test_job", 5, 10, "Test message")
        
        assert len(callback_called) == 1
        assert callback_called[0] == ("test_job", 5, 10, "Test message")
    
    @pytest.mark.asyncio
    async def test_cleanup_completed_jobs(self, batch_service):
        """Test cleanup of completed jobs."""
        # Create old completed job
        old_job = BatchJob(
            name="Old Job",
            processor_name="test",
            status=BatchStatus.COMPLETED
        )
        old_job.completed_at = datetime.now(timezone.utc) - pd.Timedelta(hours=48)
        
        # Create recent completed job
        recent_job = BatchJob(
            name="Recent Job",
            processor_name="test",
            status=BatchStatus.COMPLETED
        )
        recent_job.completed_at = datetime.now(timezone.utc) - pd.Timedelta(hours=1)
        
        batch_service._active_jobs[old_job.id] = old_job
        batch_service._active_jobs[recent_job.id] = recent_job
        
        # Cleanup jobs older than 24 hours
        cleaned_count = await batch_service.cleanup_completed_jobs(24)
        
        assert cleaned_count == 1
        assert old_job.id not in batch_service._active_jobs
        assert recent_job.id in batch_service._active_jobs


class TestBatchProcessors:
    """Test standard batch processors."""
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_processor(self):
        """Test anomaly detection batch processor."""
        # Create test data
        test_data = pd.DataFrame({
            'feature_1': range(100),
            'feature_2': [x * 2 for x in range(100)]
        })
        
        context = {
            'batch_index': 0,
            'algorithm': 'isolation_forest'
        }
        
        result = await anomaly_detection_batch_processor(test_data, context)
        
        assert 'batch_index' in result
        assert 'processed_rows' in result
        assert 'anomalies_detected' in result
        assert 'algorithm' in result
        assert result['batch_index'] == 0
        assert result['processed_rows'] == 100
        assert result['algorithm'] == 'isolation_forest'
    
    @pytest.mark.asyncio
    async def test_data_quality_processor(self):
        """Test data quality batch processor."""
        # Create test data with some quality issues
        test_data = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['a', 'b', 'c', 'b', 'e'],  # 'b' is duplicate
            'col3': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        context = {'batch_index': 0}
        
        result = await data_quality_batch_processor(test_data, context)
        
        assert 'batch_index' in result
        assert 'processed_rows' in result
        assert 'null_values' in result
        assert 'duplicates' in result
        assert 'quality_score' in result
        assert result['batch_index'] == 0
        assert result['processed_rows'] == 5
        assert result['null_values'] == 1  # One null in col1
        assert 0 <= result['quality_score'] <= 1


class TestBatchJobModel:
    """Test BatchJob model functionality."""
    
    def test_calculate_progress_percentage(self):
        """Test progress percentage calculation."""
        job = BatchJob(name="Test", processor_name="test")
        job.metrics.total_batches = 10
        job.metrics.processed_batches = 3
        
        assert job.calculate_progress_percentage() == 30.0
        
        # Test zero total batches
        job.metrics.total_batches = 0
        assert job.calculate_progress_percentage() == 0.0
    
    def test_estimate_remaining_time(self):
        """Test remaining time estimation."""
        job = BatchJob(name="Test", processor_name="test")
        job.metrics.total_items = 100
        job.metrics.processed_items = 25
        job.metrics.processing_rate_items_per_second = 5.0
        
        remaining_time = job.estimate_remaining_time()
        assert remaining_time == 15.0  # (100-25) / 5
        
        # Test zero processing rate
        job.metrics.processing_rate_items_per_second = 0
        assert job.estimate_remaining_time() is None
    
    def test_update_metrics(self):
        """Test metrics update functionality."""
        job = BatchJob(name="Test", processor_name="test")
        job.metrics.start_time = datetime.now(timezone.utc) - pd.Timedelta(seconds=10)
        job.metrics.processed_items = 50
        job.metrics.total_batches = 10
        job.metrics.processed_batches = 3
        job.metrics.failed_batches = 1
        
        job.update_metrics()
        
        assert job.metrics.processing_rate_items_per_second > 0
        assert job.metrics.error_rate == 0.1  # 1/10
        assert job.metrics.success_rate == 0.3  # 3/10
        assert job.metrics.estimated_completion_time is not None


@pytest.mark.integration
class TestBatchProcessingIntegration:
    """Integration tests for batch processing."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_batch_processing(self):
        """Test complete batch processing workflow."""
        service = BatchProcessingService()
        
        # Create test data
        data = pd.DataFrame({
            'feature_1': range(50),
            'feature_2': [x ** 2 for x in range(50)],
            'feature_3': [x % 7 for x in range(50)]
        })
        
        # Register processors
        service.register_processor("anomaly_detection", anomaly_detection_batch_processor)
        service.register_processor("data_quality", data_quality_batch_processor)
        
        # Track progress
        progress_events = []
        def track_progress(job_id, current, total, message=""):
            progress_events.append((current, total, message))
        
        # Create and run anomaly detection job
        job1 = await service.create_batch_job(
            name="Anomaly Detection Job",
            data=data,
            processor_name="anomaly_detection",
            config=BatchConfig(batch_size=10, enable_progress_tracking=True),
            priority=BatchPriority.HIGH
        )
        
        service.add_progress_callback(job1.id, track_progress)
        await service.start_batch_job(job1.id)
        
        # Verify job completion
        assert job1.status == BatchStatus.COMPLETED
        assert job1.metrics.processed_items == 50
        assert job1.metrics.total_batches == 5  # 50/10
        assert len(progress_events) > 0
        
        # Create and run data quality job
        job2 = await service.create_batch_job(
            name="Data Quality Job",
            data=data,
            processor_name="data_quality",
            config=BatchConfig(batch_size=25),
            priority=BatchPriority.MEDIUM
        )
        
        await service.start_batch_job(job2.id)
        
        # Verify both jobs
        assert job2.status == BatchStatus.COMPLETED
        assert job2.metrics.processed_items == 50
        
        # List all jobs
        all_jobs = service.list_jobs()
        assert len(all_jobs) == 2
        
        completed_jobs = service.list_jobs(status_filter=BatchStatus.COMPLETED)
        assert len(completed_jobs) == 2