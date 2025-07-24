"""Unit tests for the background worker system."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import pandas as pd

from anomaly_detection.worker import (
    AnomalyDetectionWorker, JobQueue, WorkerJob,
    JobType, JobStatus, JobPriority
)


class TestJobQueue:
    """Test the job queue implementation."""
    
    @pytest.fixture
    def job_queue(self):
        """Create a job queue for testing."""
        return JobQueue()
    
    @pytest.fixture
    def sample_job(self):
        """Create a sample job for testing."""
        return WorkerJob(
            job_id="test-job-123",
            job_type=JobType.DETECTION,
            priority=JobPriority.NORMAL,
            status=JobStatus.PENDING,
            payload={"algorithm": "isolation_forest"},
            created_at=datetime.utcnow()
        )
    
    async def test_enqueue_and_dequeue(self, job_queue, sample_job):
        """Test basic enqueue and dequeue operations."""
        # Initially empty
        job = await job_queue.dequeue()
        assert job is None
        
        # Enqueue job
        await job_queue.enqueue(sample_job)
        
        # Dequeue job
        job = await job_queue.dequeue()
        assert job is not None
        assert job.job_id == "test-job-123"
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
    
    async def test_priority_ordering(self, job_queue):
        """Test that jobs are dequeued in priority order."""
        # Create jobs with different priorities
        low_job = WorkerJob(
            job_id="low",
            job_type=JobType.DETECTION,
            priority=JobPriority.LOW,
            status=JobStatus.PENDING,
            payload={},
            created_at=datetime.utcnow()
        )
        
        high_job = WorkerJob(
            job_id="high",
            job_type=JobType.DETECTION,
            priority=JobPriority.HIGH,
            status=JobStatus.PENDING,
            payload={},
            created_at=datetime.utcnow()
        )
        
        critical_job = WorkerJob(
            job_id="critical",
            job_type=JobType.DETECTION,
            priority=JobPriority.CRITICAL,
            status=JobStatus.PENDING,
            payload={},
            created_at=datetime.utcnow()
        )
        
        # Enqueue in random order
        await job_queue.enqueue(low_job)
        await job_queue.enqueue(high_job)
        await job_queue.enqueue(critical_job)
        
        # Should dequeue in priority order
        first = await job_queue.dequeue()
        assert first.job_id == "critical"
        
        second = await job_queue.dequeue()
        assert second.job_id == "high"
        
        third = await job_queue.dequeue()
        assert third.job_id == "low"
    
    async def test_get_job(self, job_queue, sample_job):
        """Test getting job by ID."""
        await job_queue.enqueue(sample_job)
        
        retrieved_job = await job_queue.get_job("test-job-123")
        assert retrieved_job is not None
        assert retrieved_job.job_id == "test-job-123"
        
        # Non-existent job
        missing_job = await job_queue.get_job("missing")
        assert missing_job is None
    
    async def test_update_job(self, job_queue, sample_job):
        """Test updating job in history."""
        await job_queue.enqueue(sample_job)
        
        # Update job status
        sample_job.status = JobStatus.COMPLETED
        sample_job.result = {"success": True}
        await job_queue.update_job(sample_job)
        
        # Retrieve and verify update
        updated_job = await job_queue.get_job("test-job-123")
        assert updated_job.status == JobStatus.COMPLETED
        assert updated_job.result == {"success": True}
    
    async def test_queue_status(self, job_queue):
        """Test queue status reporting."""
        # Initially empty
        status = await job_queue.get_queue_status()
        assert status["pending_jobs"] == 0
        assert status["total_jobs"] == 0
        
        # Add some jobs
        job1 = WorkerJob(
            job_id="job1",
            job_type=JobType.DETECTION,
            priority=JobPriority.NORMAL,
            status=JobStatus.PENDING,
            payload={},
            created_at=datetime.utcnow()
        )
        
        job2 = WorkerJob(
            job_id="job2",
            job_type=JobType.ENSEMBLE,
            priority=JobPriority.HIGH,
            status=JobStatus.PENDING,
            payload={},
            created_at=datetime.utcnow()
        )
        
        await job_queue.enqueue(job1)
        await job_queue.enqueue(job2)
        
        status = await job_queue.get_queue_status()
        assert status["pending_jobs"] == 2
        assert status["total_jobs"] == 2
        assert status["status_counts"]["pending"] == 2


class TestWorkerJob:
    """Test the WorkerJob class."""
    
    def test_job_creation(self):
        """Test creating a worker job."""
        job = WorkerJob(
            job_id="test-123",
            job_type=JobType.DETECTION,
            priority=JobPriority.HIGH,
            status=JobStatus.PENDING,
            payload={"algorithm": "isolation_forest"},
            created_at=datetime.utcnow()
        )
        
        assert job.job_id == "test-123"
        assert job.job_type == JobType.DETECTION
        assert job.priority == JobPriority.HIGH
        assert job.status == JobStatus.PENDING
        assert job.retry_count == 0
        assert job.max_retries == 3
        assert job.progress == 0.0
    
    def test_job_to_dict(self):
        """Test converting job to dictionary."""
        created_at = datetime.utcnow()
        job = WorkerJob(
            job_id="test-123",
            job_type=JobType.ENSEMBLE,
            priority=JobPriority.LOW,
            status=JobStatus.COMPLETED,
            payload={"algorithms": ["iforest", "lof"]},
            created_at=created_at,
            result={"anomalies": 5}
        )
        
        job_dict = job.to_dict()
        
        assert job_dict["job_id"] == "test-123"
        assert job_dict["job_type"] == "ensemble"
        assert job_dict["priority"] == 1  # LOW priority value
        assert job_dict["status"] == "completed"
        assert job_dict["payload"] == {"algorithms": ["iforest", "lof"]}
        assert job_dict["result"] == {"anomalies": 5}
        assert job_dict["created_at"] == created_at.isoformat()


class TestAnomalyDetectionWorker:
    """Test the main worker class."""
    
    @pytest.fixture
    def worker(self, temp_dir):
        """Create a worker for testing."""
        return AnomalyDetectionWorker(
            models_dir=str(temp_dir / "models"),
            max_concurrent_jobs=2,
            enable_monitoring=False  # Disable for testing
        )
    
    @pytest.fixture
    def sample_csv_data(self, temp_dir):
        """Create sample CSV data for testing."""
        # Generate synthetic data
        np.random.seed(42)
        normal_data = np.random.randn(100, 3)
        anomaly_data = np.random.randn(10, 3) * 3 + 5  # Shifted anomalies
        
        data = np.vstack([normal_data, anomaly_data])
        df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
        
        csv_path = temp_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    async def test_submit_job(self, worker):
        """Test submitting a job to the worker."""
        job_id = await worker.submit_job(
            JobType.DETECTION,
            {"algorithm": "isolation_forest", "data_source": [[1, 2], [3, 4]]},
            priority=JobPriority.HIGH
        )
        
        assert job_id is not None
        assert len(job_id) > 0
        
        # Check job was added to queue
        job = await worker.job_queue.get_job(job_id)
        assert job is not None
        assert job.job_type == JobType.DETECTION
        assert job.priority == JobPriority.HIGH
        assert job.status == JobStatus.PENDING
    
    async def test_get_job_status(self, worker):
        """Test getting job status."""
        job_id = await worker.submit_job(
            JobType.DETECTION,
            {"algorithm": "lof", "data_source": [[1, 2], [3, 4]]}
        )
        
        status = await worker.get_job_status(job_id)
        assert status is not None
        assert status["job_id"] == job_id
        assert status["status"] == "pending"
        assert status["job_type"] == "detection"
        
        # Non-existent job
        missing_status = await worker.get_job_status("missing-job")
        assert missing_status is None
    
    async def test_cancel_pending_job(self, worker):
        """Test cancelling a pending job."""
        job_id = await worker.submit_job(
            JobType.DETECTION,
            {"algorithm": "isolation_forest", "data_source": [[1, 2], [3, 4]]}
        )
        
        # Cancel the job
        success = await worker.cancel_job(job_id)
        assert success is True
        
        # Check status
        status = await worker.get_job_status(job_id)
        assert status["status"] == "cancelled"
    
    async def test_process_detection_job_with_direct_data(self, worker):
        """Test processing detection job with direct data."""
        job = WorkerJob(
            job_id="test-detection",
            job_type=JobType.DETECTION,
            priority=JobPriority.NORMAL,
            status=JobStatus.RUNNING,
            payload={
                "data_source": [[0, 0], [1, 1], [0.5, 0.5], [10, 10]],  # Last point is anomaly
                "algorithm": "isolation_forest",
                "contamination": 0.25
            },
            created_at=datetime.utcnow()
        )
        
        result = await worker._process_detection_job(job)
        
        assert result["job_id"] == "test-detection"
        assert result["algorithm"] == "isolation_forest"
        assert result["total_samples"] == 4
        assert result["anomalies_detected"] >= 0
        assert 0 <= result["anomaly_rate"] <= 1
    
    async def test_process_detection_job_with_csv(self, worker, sample_csv_data):
        """Test processing detection job with CSV file."""
        job = WorkerJob(
            job_id="test-csv-detection",
            job_type=JobType.DETECTION,
            priority=JobPriority.NORMAL,
            status=JobStatus.RUNNING,
            payload={
                "data_source": str(sample_csv_data),
                "algorithm": "lof",
                "contamination": 0.1
            },
            created_at=datetime.utcnow()
        )
        
        result = await worker._process_detection_job(job)
        
        assert result["job_id"] == "test-csv-detection"
        assert result["algorithm"] == "lof"
        assert result["total_samples"] == 110  # 100 normal + 10 anomalies
        assert result["anomalies_detected"] >= 0
    
    async def test_process_ensemble_job(self, worker):
        """Test processing ensemble job."""
        job = WorkerJob(
            job_id="test-ensemble",
            job_type=JobType.ENSEMBLE,
            priority=JobPriority.NORMAL,
            status=JobStatus.RUNNING,
            payload={
                "data_source": [[0, 0], [1, 1], [0.5, 0.5], [10, 10], [11, 11]],
                "algorithms": ["isolation_forest", "lof"],
                "method": "majority",
                "contamination": 0.2
            },
            created_at=datetime.utcnow()
        )
        
        result = await worker._process_ensemble_job(job)
        
        assert result["job_id"] == "test-ensemble"
        assert result["algorithms"] == ["isolation_forest", "lof"]
        assert result["method"] == "majority"
        assert result["total_samples"] == 5
        assert result["anomalies_detected"] >= 0
    
    async def test_process_explanation_job(self, worker):
        """Test processing explanation generation job."""
        job = WorkerJob(
            job_id="test-explanation",
            job_type=JobType.EXPLANATION_GENERATION,
            priority=JobPriority.NORMAL,
            status=JobStatus.RUNNING,
            payload={
                "anomaly_indices": [5, 10, 15],
                "method": "feature_importance"
            },
            created_at=datetime.utcnow()
        )
        
        result = await worker._process_explanation_job(job)
        
        assert result["job_id"] == "test-explanation"
        assert result["explanation_method"] == "feature_importance"
        assert result["explanations_generated"] == 3
        assert len(result["explanations"]) == 3
        
        # Check explanation structure
        explanation = result["explanations"][0]
        assert "sample_index" in explanation
        assert "feature_importance" in explanation
        assert "confidence" in explanation
    
    async def test_process_scheduled_analysis_job(self, worker):
        """Test processing scheduled analysis job."""
        job = WorkerJob(
            job_id="test-scheduled",
            job_type=JobType.SCHEDULED_ANALYSIS,
            priority=JobPriority.NORMAL,
            status=JobStatus.RUNNING,
            payload={
                "analysis_type": "weekly_report",
                "data_sources": ["source1", "source2", "source3"]
            },
            created_at=datetime.utcnow()
        )
        
        result = await worker._process_scheduled_analysis_job(job)
        
        assert result["job_id"] == "test-scheduled"
        assert result["analysis_type"] == "weekly_report"
        assert result["sources_analyzed"] == 3
        assert result["total_anomalies"] >= 0
        assert 0 <= result["average_quality_score"] <= 1
        assert len(result["analysis_results"]) == 3
    
    async def test_worker_status(self, worker):
        """Test getting worker status."""
        status = await worker.get_worker_status()
        
        assert "is_running" in status
        assert "max_concurrent_jobs" in status
        assert "currently_running_jobs" in status
        assert "queue_status" in status
        assert "monitoring_enabled" in status
        
        assert status["max_concurrent_jobs"] == 2
        assert status["monitoring_enabled"] is False
        assert status["currently_running_jobs"] == 0
    
    async def test_invalid_job_type_handling(self, worker):
        """Test handling of invalid job types."""
        # Create job with invalid type (simulate corruption)
        job = WorkerJob(
            job_id="test-invalid",
            job_type=JobType.DETECTION,  # We'll change this after creation
            priority=JobPriority.NORMAL,
            status=JobStatus.RUNNING,
            payload={},
            created_at=datetime.utcnow()
        )
        
        # Manually set an invalid job type
        job.job_type = "invalid_type"
        
        # This should raise an exception
        with pytest.raises(Exception):
            await worker._execute_job(job)
    
    async def test_data_preprocessing_job(self, worker, temp_dir):
        """Test data preprocessing job."""
        # Create input data
        input_path = temp_dir / "input.csv"
        output_path = temp_dir / "output.csv"
        
        # Create test data with outliers
        np.random.seed(42)
        normal_data = np.random.randn(100, 2)
        outliers = np.array([[10, 10], [-10, -10], [15, -15]])  # Clear outliers
        
        data = np.vstack([normal_data, outliers])
        df = pd.DataFrame(data, columns=['x', 'y'])
        df.to_csv(input_path, index=False)
        
        job = WorkerJob(
            job_id="test-preprocessing",
            job_type=JobType.DATA_PREPROCESSING,
            priority=JobPriority.NORMAL,
            status=JobStatus.RUNNING,
            payload={
                "input_path": str(input_path),
                "output_path": str(output_path),
                "steps": ["normalize", "remove_outliers"],
                "original_samples": 103
            },
            created_at=datetime.utcnow()
        )
        
        result = await worker._process_data_preprocessing_job(job)
        
        assert result["job_id"] == "test-preprocessing"
        assert result["preprocessing_steps"] == ["normalize", "remove_outliers"]
        assert result["original_samples"] == 103
        assert result["processed_samples"] < 103  # Should remove outliers
        assert output_path.exists()
        
        # Verify output file
        output_df = pd.read_csv(output_path)
        assert len(output_df) < 103  # Outliers removed
        
        # Data should be normalized (mean close to 0, std close to 1)
        assert abs(output_df['x'].mean()) < 0.1
        assert abs(output_df['x'].std() - 1.0) < 0.1
    
    async def test_stream_monitoring_job(self, worker):
        """Test stream monitoring job."""
        job = WorkerJob(
            job_id="test-stream",
            job_type=JobType.STREAM_MONITORING,
            priority=JobPriority.NORMAL,
            status=JobStatus.RUNNING,
            payload={
                "stream_config": {
                    "window_size": 50,
                    "source": "test_stream"
                },
                "duration_seconds": 3  # Short duration for testing
            },
            created_at=datetime.utcnow()
        )
        
        start_time = asyncio.get_event_loop().time()
        result = await worker._process_stream_monitoring_job(job)
        end_time = asyncio.get_event_loop().time()
        
        # Should take approximately 3 seconds
        duration = end_time - start_time
        assert 2.5 <= duration <= 4.0  # Allow some tolerance
        
        assert result["job_id"] == "test-stream"
        assert result["windows_processed"] >= 2  # At least 2-3 windows in 3 seconds
        assert result["monitoring_duration"] == 3
        assert 0 <= result["anomaly_rate"] <= 1


class TestWorkerIntegration:
    """Integration tests for the worker system."""
    
    @pytest.fixture
    def worker(self, temp_dir):
        """Create a worker for integration testing."""
        return AnomalyDetectionWorker(
            models_dir=str(temp_dir / "models"),
            max_concurrent_jobs=1,  # Single job for controlled testing
            enable_monitoring=False
        )
    
    async def test_end_to_end_detection_workflow(self, worker, temp_dir):
        """Test complete detection workflow from submission to completion."""
        # Create test data
        test_data = [[0, 0], [1, 1], [0.5, 0.5], [10, 10]]  # Last is anomaly
        output_path = temp_dir / "detection_results.json"
        
        # Submit job
        job_id = await worker.submit_job(
            JobType.DETECTION,
            {
                "data_source": test_data,
                "algorithm": "isolation_forest",
                "contamination": 0.25,
                "output_path": str(output_path)
            },
            priority=JobPriority.HIGH
        )
        
        # Start worker in background
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Wait for job completion
            max_wait = 30  # 30 seconds max
            wait_time = 0
            
            while wait_time < max_wait:
                status = await worker.get_job_status(job_id)
                if status and status["status"] in ["completed", "failed"]:
                    break
                
                await asyncio.sleep(1)
                wait_time += 1
            
            # Check final status
            final_status = await worker.get_job_status(job_id)
            assert final_status is not None
            assert final_status["status"] == "completed"
            assert final_status["result"] is not None
            
            # Check output file was created
            assert output_path.exists()
            
            # Verify output file content
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            
            assert "anomalies_detected" in output_data
            assert "total_samples" in output_data
            
        finally:
            await worker.stop()
            worker_task.cancel()
            
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
    
    async def test_multiple_job_processing(self, worker):
        """Test processing multiple jobs concurrently."""
        # Submit multiple jobs
        job_ids = []
        
        for i in range(3):
            job_id = await worker.submit_job(
                JobType.EXPLANATION_GENERATION,
                {
                    "anomaly_indices": [i, i+1, i+2],
                    "method": f"method_{i}"
                },
                priority=JobPriority.NORMAL
            )
            job_ids.append(job_id)
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Wait for all jobs to complete
            max_wait = 60
            wait_time = 0
            
            while wait_time < max_wait:
                all_complete = True
                
                for job_id in job_ids:
                    status = await worker.get_job_status(job_id)
                    if not status or status["status"] not in ["completed", "failed"]:
                        all_complete = False
                        break
                
                if all_complete:
                    break
                
                await asyncio.sleep(1)
                wait_time += 1
            
            # Verify all jobs completed
            for job_id in job_ids:
                final_status = await worker.get_job_status(job_id)
                assert final_status is not None
                assert final_status["status"] == "completed"
                assert final_status["result"] is not None
            
            # Check worker status
            worker_status = await worker.get_worker_status()
            assert worker_status["currently_running_jobs"] == 0
            
        finally:
            await worker.stop()
            worker_task.cancel()
            
            try:
                await worker_task
            except asyncio.CancelledError:
                pass