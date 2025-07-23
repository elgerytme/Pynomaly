"""Unit tests for worker entry point."""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from enum import Enum

from anomaly_detection.worker import (
    AnomalyDetectionWorker, JobQueue, Job, JobType, JobStatus, JobPriority,
    run_worker_demo, main
)


class TestJobQueue:
    """Test cases for JobQueue class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.queue = JobQueue()
    
    def test_job_queue_initialization(self):
        """Test job queue initialization."""
        assert isinstance(self.queue._queues, dict)
        assert len(self.queue._queues) == len(JobPriority)
        assert self.queue._lock is not None
        
        # Check that all priority levels have empty deques
        for priority in JobPriority:
            assert priority in self.queue._queues
            assert isinstance(self.queue._queues[priority], deque)
            assert len(self.queue._queues[priority]) == 0
    
    def test_add_job_single_priority(self):
        """Test adding jobs with single priority."""
        job = Job(
            job_id="job_123",
            job_type=JobType.DETECTION,
            data={"algorithm": "isolation_forest"},
            priority=JobPriority.NORMAL
        )
        
        self.queue.add_job(job)
        
        assert self.queue.size() == 1
        assert len(self.queue._queues[JobPriority.NORMAL]) == 1
        assert self.queue._queues[JobPriority.NORMAL][0] == job
    
    def test_add_job_multiple_priorities(self):
        """Test adding jobs with different priorities."""
        jobs = [
            Job("job_low", JobType.DETECTION, {}, JobPriority.LOW),
            Job("job_high", JobType.DETECTION, {}, JobPriority.HIGH),
            Job("job_normal", JobType.DETECTION, {}, JobPriority.NORMAL),
            Job("job_critical", JobType.DETECTION, {}, JobPriority.CRITICAL)
        ]
        
        for job in jobs:
            self.queue.add_job(job)
        
        assert self.queue.size() == 4
        assert len(self.queue._queues[JobPriority.LOW]) == 1
        assert len(self.queue._queues[JobPriority.NORMAL]) == 1
        assert len(self.queue._queues[JobPriority.HIGH]) == 1
        assert len(self.queue._queues[JobPriority.CRITICAL]) == 1
    
    def test_get_next_job_priority_order(self):
        """Test that jobs are retrieved in priority order."""
        jobs = [
            Job("job_low", JobType.DETECTION, {}, JobPriority.LOW),
            Job("job_normal", JobType.DETECTION, {}, JobPriority.NORMAL),
            Job("job_high", JobType.DETECTION, {}, JobPriority.HIGH),
            Job("job_critical", JobType.DETECTION, {}, JobPriority.CRITICAL)
        ]
        
        # Add jobs in random order
        for job in [jobs[1], jobs[3], jobs[0], jobs[2]]:
            self.queue.add_job(job)
        
        # Should retrieve in priority order (CRITICAL, HIGH, NORMAL, LOW)
        retrieved_jobs = []
        while not self.queue.is_empty():
            retrieved_jobs.append(self.queue.get_next_job())
        
        assert len(retrieved_jobs) == 4
        assert retrieved_jobs[0].job_id == "job_critical"
        assert retrieved_jobs[1].job_id == "job_high"
        assert retrieved_jobs[2].job_id == "job_normal"
        assert retrieved_jobs[3].job_id == "job_low"
    
    def test_get_next_job_fifo_within_priority(self):
        """Test FIFO order within same priority level."""
        jobs = [
            Job("job_1", JobType.DETECTION, {}, JobPriority.NORMAL),
            Job("job_2", JobType.ENSEMBLE, {}, JobPriority.NORMAL),
            Job("job_3", JobType.TRAINING, {}, JobPriority.NORMAL)
        ]
        
        for job in jobs:
            self.queue.add_job(job)
        
        # Should retrieve in FIFO order within same priority
        retrieved_jobs = []
        while not self.queue.is_empty():
            retrieved_jobs.append(self.queue.get_next_job())
        
        assert retrieved_jobs[0].job_id == "job_1"
        assert retrieved_jobs[1].job_id == "job_2"
        assert retrieved_jobs[2].job_id == "job_3"
    
    def test_get_next_job_empty_queue(self):
        """Test getting next job from empty queue."""
        job = self.queue.get_next_job()
        assert job is None
    
    def test_is_empty(self):
        """Test queue empty check."""
        assert self.queue.is_empty() is True
        
        self.queue.add_job(Job("job_1", JobType.DETECTION, {}, JobPriority.NORMAL))
        assert self.queue.is_empty() is False
        
        self.queue.get_next_job()
        assert self.queue.is_empty() is True
    
    def test_size(self):
        """Test queue size calculation."""
        assert self.queue.size() == 0
        
        # Add jobs with different priorities
        self.queue.add_job(Job("job_1", JobType.DETECTION, {}, JobPriority.LOW))
        self.queue.add_job(Job("job_2", JobType.DETECTION, {}, JobPriority.HIGH))
        self.queue.add_job(Job("job_3", JobType.DETECTION, {}, JobPriority.HIGH))
        
        assert self.queue.size() == 3
        
        self.queue.get_next_job()
        assert self.queue.size() == 2
    
    def test_clear(self):
        """Test clearing the queue."""
        # Add some jobs
        for i in range(5):
            self.queue.add_job(Job(f"job_{i}", JobType.DETECTION, {}, JobPriority.NORMAL))
        
        assert self.queue.size() == 5
        
        self.queue.clear()
        
        assert self.queue.size() == 0
        assert self.queue.is_empty() is True
    
    def test_thread_safety(self):
        """Test thread safety of job queue operations."""
        import threading
        
        def add_jobs():
            for i in range(100):
                job = Job(f"job_{i}", JobType.DETECTION, {}, JobPriority.NORMAL)
                self.queue.add_job(job)
        
        def get_jobs():
            jobs = []
            for _ in range(50):
                job = self.queue.get_next_job()
                if job:
                    jobs.append(job)
            return jobs
        
        # Start multiple threads
        add_thread1 = threading.Thread(target=add_jobs)
        add_thread2 = threading.Thread(target=add_jobs)
        get_thread = threading.Thread(target=get_jobs)
        
        add_thread1.start()
        add_thread2.start()
        add_thread1.join()
        add_thread2.join()
        
        # Should have 200 jobs
        assert self.queue.size() == 200
        
        get_thread.start()
        get_thread.join()
        
        # Should have 150 jobs remaining
        assert self.queue.size() == 150


class TestJob:
    """Test cases for Job class."""
    
    def test_job_initialization(self):
        """Test job initialization with all parameters."""
        job_data = {"algorithm": "isolation_forest", "data": [[1, 2], [3, 4]]}
        job = Job(
            job_id="test_job_123",
            job_type=JobType.DETECTION,
            data=job_data,
            priority=JobPriority.HIGH,
            created_at=datetime.now()
        )
        
        assert job.job_id == "test_job_123"
        assert job.job_type == JobType.DETECTION
        assert job.data == job_data
        assert job.priority == JobPriority.HIGH
        assert job.status == JobStatus.QUEUED
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.result is None
        assert job.error is None
        assert job.retry_count == 0
    
    def test_job_default_values(self):
        """Test job initialization with default values."""
        job = Job("test_job", JobType.ENSEMBLE, {})
        
        assert job.priority == JobPriority.NORMAL
        assert job.status == JobStatus.QUEUED
        assert job.created_at is not None
        assert job.retry_count == 0
    
    def test_job_status_transitions(self):
        """Test job status transitions."""
        job = Job("test_job", JobType.DETECTION, {})
        
        # Initial status
        assert job.status == JobStatus.QUEUED
        
        # Start processing
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now()
        assert job.status == JobStatus.PROCESSING
        assert job.started_at is not None
        
        # Complete successfully
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.result = {"anomalies": [0, 1, 0]}
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        
        # Test failed status
        failed_job = Job("failed_job", JobType.DETECTION, {})
        failed_job.status = JobStatus.FAILED
        failed_job.error = "Processing failed"
        assert failed_job.status == JobStatus.FAILED
        assert failed_job.error == "Processing failed"
    
    def test_job_duration_calculation(self):
        """Test job duration calculation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=5)
        
        job = Job("test_job", JobType.DETECTION, {})
        job.started_at = start_time
        job.completed_at = end_time
        
        duration = (job.completed_at - job.started_at).total_seconds()
        assert duration == 5.0
    
    def test_job_retry_increment(self):
        """Test job retry count increment."""
        job = Job("test_job", JobType.DETECTION, {})
        
        assert job.retry_count == 0
        
        job.retry_count += 1
        assert job.retry_count == 1
        
        job.retry_count += 1
        assert job.retry_count == 2


class TestAnomalyDetectionWorker:
    """Test cases for AnomalyDetectionWorker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.worker = AnomalyDetectionWorker(max_workers=2)
        
        # Mock services
        self.mock_detection_service = Mock()
        self.mock_ensemble_service = Mock()
        self.mock_model_repository = Mock()
        self.mock_streaming_service = Mock()
        self.mock_explainability_service = Mock()
        self.mock_health_service = Mock()
        
        # Inject mocked services
        self.worker.detection_service = self.mock_detection_service
        self.worker.ensemble_service = self.mock_ensemble_service
        self.worker.model_repository = self.mock_model_repository
        self.worker.streaming_service = self.mock_streaming_service
        self.worker.explainability_service = self.mock_explainability_service
        self.worker.health_service = self.mock_health_service
    
    def test_worker_initialization(self):
        """Test worker initialization."""
        worker = AnomalyDetectionWorker(max_workers=4)
        
        assert worker.max_workers == 4
        assert isinstance(worker.job_queue, JobQueue)
        assert isinstance(worker.executor, ThreadPoolExecutor)
        assert worker.is_running is False
        assert worker.active_jobs == {}
        assert worker.completed_jobs == []
        assert worker.failed_jobs == []
    
    def test_add_job(self):
        """Test adding job to worker."""
        job_data = {"algorithm": "isolation_forest", "data": [[1, 2]]}
        
        job_id = self.worker.add_job(
            job_type=JobType.DETECTION,
            data=job_data,
            priority=JobPriority.HIGH
        )
        
        assert job_id is not None
        assert self.worker.job_queue.size() == 1
        
        # Get the job and verify
        job = self.worker.job_queue.get_next_job()
        assert job.job_id == job_id
        assert job.job_type == JobType.DETECTION
        assert job.data == job_data
        assert job.priority == JobPriority.HIGH
    
    async def test_process_detection_job(self):
        """Test processing detection job."""
        # Mock detection service response
        self.mock_detection_service.detect_anomalies = AsyncMock(return_value={
            "anomalies": [0, 1, 0],
            "scores": [0.1, 0.8, 0.2],
            "algorithm": "isolation_forest"
        })
        
        job = Job(
            job_id="detection_job_123",
            job_type=JobType.DETECTION,
            data={
                "algorithm": "isolation_forest",
                "data": [[1, 2], [3, 4], [5, 6]],
                "contamination": 0.1
            }
        )
        
        result = await self.worker._process_job(job)
        
        assert result is not None
        assert result["anomalies"] == [0, 1, 0]
        assert result["algorithm"] == "isolation_forest"
        self.mock_detection_service.detect_anomalies.assert_called_once()
    
    async def test_process_ensemble_job(self):
        """Test processing ensemble job."""
        # Mock ensemble service response
        self.mock_ensemble_service.detect_with_ensemble = AsyncMock(return_value={
            "anomalies": [0, 1, 0],
            "ensemble_scores": [0.1, 0.8, 0.2],
            "individual_results": {"isolation_forest": {"scores": [0.1, 0.7, 0.2]}},
            "ensemble_method": "majority"
        })
        
        job = Job(
            job_id="ensemble_job_123",
            job_type=JobType.ENSEMBLE,
            data={
                "algorithms": ["isolation_forest", "one_class_svm"],
                "data": [[1, 2], [3, 4], [5, 6]],
                "ensemble_method": "majority",
                "contamination": 0.1
            }
        )
        
        result = await self.worker._process_job(job)
        
        assert result is not None
        assert result["anomalies"] == [0, 1, 0]
        assert result["ensemble_method"] == "majority"
        self.mock_ensemble_service.detect_with_ensemble.assert_called_once()
    
    async def test_process_training_job(self):
        """Test processing model training job."""
        # Mock model repository save
        self.mock_model_repository.save = Mock(return_value="trained_model_123")
        
        job = Job(
            job_id="training_job_123",
            job_type=JobType.TRAINING,
            data={
                "algorithm": "isolation_forest",
                "training_data": [[1, 2], [3, 4], [5, 6]],
                "model_name": "test_model",
                "parameters": {"n_estimators": 100}
            }
        )
        
        with patch('anomaly_detection.worker.Model') as mock_model_class:
            mock_model = Mock()
            mock_model.model_id = "trained_model_123"
            mock_model_class.return_value = mock_model
            
            result = await self.worker._process_job(job)
            
            assert result is not None
            assert result["model_id"] == "trained_model_123"
            assert result["status"] == "trained"
            self.mock_model_repository.save.assert_called_once()
    
    async def test_process_streaming_job(self):
        """Test processing streaming job."""
        # Mock streaming service
        self.mock_streaming_service.start_streaming = AsyncMock(return_value={
            "stream_id": "stream_123",
            "status": "started",
            "algorithm": "isolation_forest"
        })
        
        job = Job(
            job_id="streaming_job_123",
            job_type=JobType.STREAMING,
            data={
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "buffer_size": 1000
            }
        )
        
        result = await self.worker._process_job(job)
        
        assert result is not None
        assert result["stream_id"] == "stream_123"
        assert result["status"] == "started"
        self.mock_streaming_service.start_streaming.assert_called_once()
    
    async def test_process_explanation_job(self):
        """Test processing explanation job."""
        # Mock explainability service
        self.mock_explainability_service.explain_prediction = AsyncMock(return_value={
            "explanation_type": "feature_importance",
            "feature_importance": [0.3, 0.7],
            "sample_index": 0,
            "prediction": 1
        })
        
        job = Job(
            job_id="explanation_job_123",
            job_type=JobType.EXPLANATION,
            data={
                "data": [[1, 2], [3, 4]],
                "sample_index": 0,
                "explainer_type": "feature_importance",
                "algorithm": "isolation_forest"
            }
        )
        
        result = await self.worker._process_job(job)
        
        assert result is not None
        assert result["explanation_type"] == "feature_importance"
        assert result["feature_importance"] == [0.3, 0.7]
        self.mock_explainability_service.explain_prediction.assert_called_once()
    
    async def test_process_health_check_job(self):
        """Test processing health check job."""
        # Mock health service
        self.mock_health_service.get_health_summary = AsyncMock(return_value={
            "overall_status": "healthy",
            "services": {"detection": "healthy", "models": "healthy"},
            "timestamp": datetime.now().isoformat()
        })
        
        job = Job(
            job_id="health_job_123",
            job_type=JobType.HEALTH_CHECK,
            data={}
        )
        
        result = await self.worker._process_job(job)
        
        assert result is not None
        assert result["overall_status"] == "healthy"
        self.mock_health_service.get_health_summary.assert_called_once()
    
    async def test_process_job_with_error(self):
        """Test processing job that raises an error."""
        # Mock service to raise an error
        self.mock_detection_service.detect_anomalies = AsyncMock(
            side_effect=Exception("Detection algorithm failed")
        )
        
        job = Job(
            job_id="failing_job_123",
            job_type=JobType.DETECTION,
            data={"algorithm": "isolation_forest", "data": [[1, 2]]}
        )
        
        with pytest.raises(Exception, match="Detection algorithm failed"):
            await self.worker._process_job(job)
    
    async def test_process_unknown_job_type(self):
        """Test processing job with unknown type."""
        # Create job with invalid type (mock enum extension)
        job = Job(
            job_id="unknown_job_123",
            job_type="UNKNOWN_TYPE",  # Invalid type
            data={}
        )
        
        with pytest.raises(ValueError, match="Unknown job type"):
            await self.worker._process_job(job)
    
    def test_get_job_status(self):
        """Test getting job status."""
        # Add a job
        job_id = self.worker.add_job(JobType.DETECTION, {"data": [[1, 2]]})
        
        # Get job status
        status = self.worker.get_job_status(job_id)
        
        assert status is not None
        assert status["job_id"] == job_id
        assert status["status"] == JobStatus.QUEUED.value
        assert "created_at" in status
    
    def test_get_job_status_nonexistent(self):
        """Test getting status of nonexistent job."""
        status = self.worker.get_job_status("nonexistent_job")
        assert status is None
    
    def test_cancel_job(self):
        """Test canceling a queued job."""
        # Add a job
        job_id = self.worker.add_job(JobType.DETECTION, {"data": [[1, 2]]})
        
        # Cancel the job
        result = self.worker.cancel_job(job_id)
        
        assert result is True
        # Job should be removed from queue
        assert self.worker.job_queue.size() == 0
    
    def test_cancel_nonexistent_job(self):
        """Test canceling a nonexistent job."""
        result = self.worker.cancel_job("nonexistent_job")
        assert result is False
    
    def test_get_worker_stats(self):
        """Test getting worker statistics."""
        # Add some jobs
        self.worker.add_job(JobType.DETECTION, {"data": [[1, 2]]})
        self.worker.add_job(JobType.ENSEMBLE, {"data": [[1, 2]]})
        
        # Add some completed jobs
        completed_job = Job("completed_job", JobType.DETECTION, {})
        completed_job.status = JobStatus.COMPLETED
        self.worker.completed_jobs.append(completed_job)
        
        # Add some failed jobs
        failed_job = Job("failed_job", JobType.DETECTION, {})
        failed_job.status = JobStatus.FAILED
        self.worker.failed_jobs.append(failed_job)
        
        stats = self.worker.get_worker_stats()
        
        assert stats["queued_jobs"] == 2
        assert stats["active_jobs"] == 0
        assert stats["completed_jobs"] == 1
        assert stats["failed_jobs"] == 1
        assert stats["total_jobs"] == 4
        assert stats["max_workers"] == 2
        assert stats["is_running"] is False
    
    async def test_start_stop_worker(self):
        """Test starting and stopping worker."""
        # Start worker
        await self.worker.start()
        assert self.worker.is_running is True
        
        # Stop worker
        await self.worker.stop()
        assert self.worker.is_running is False
    
    def test_clear_completed_jobs(self):
        """Test clearing completed jobs."""
        # Add some completed jobs
        for i in range(5):
            job = Job(f"job_{i}", JobType.DETECTION, {})
            job.status = JobStatus.COMPLETED
            self.worker.completed_jobs.append(job)
        
        assert len(self.worker.completed_jobs) == 5
        
        self.worker.clear_completed_jobs()
        
        assert len(self.worker.completed_jobs) == 0
    
    def test_clear_failed_jobs(self):
        """Test clearing failed jobs."""
        # Add some failed jobs
        for i in range(3):
            job = Job(f"job_{i}", JobType.DETECTION, {})
            job.status = JobStatus.FAILED
            self.worker.failed_jobs.append(job)
        
        assert len(self.worker.failed_jobs) == 3
        
        self.worker.clear_failed_jobs()
        
        assert len(self.worker.failed_jobs) == 0


class TestWorkerDemo:
    """Test cases for worker demo functionality."""
    
    @patch('anomaly_detection.worker.AnomalyDetectionWorker')
    async def test_run_worker_demo(self, mock_worker_class):
        """Test running worker demo."""
        # Mock worker instance
        mock_worker = Mock()
        mock_worker.add_job = Mock(side_effect=["job_1", "job_2", "job_3", "job_4", "job_5"])
        mock_worker.start = AsyncMock()
        mock_worker.stop = AsyncMock()
        mock_worker.get_worker_stats = Mock(return_value={
            "queued_jobs": 0,
            "active_jobs": 0,
            "completed_jobs": 5,
            "failed_jobs": 0,
            "total_jobs": 5
        })
        mock_worker.get_job_status = Mock(return_value={
            "job_id": "job_1",
            "status": "completed",
            "result": {"anomalies": [0, 1, 0]}
        })
        mock_worker_class.return_value = mock_worker
        
        # Run demo
        await run_worker_demo()
        
        # Verify worker operations
        mock_worker.start.assert_called_once()
        mock_worker.stop.assert_called_once()
        
        # Should have added 5 different types of jobs
        assert mock_worker.add_job.call_count == 5
        
        # Check job types were added
        job_calls = mock_worker.add_job.call_args_list
        job_types = [call[1]["job_type"] for call in job_calls]
        assert JobType.DETECTION in job_types
        assert JobType.ENSEMBLE in job_types
        assert JobType.TRAINING in job_types
        assert JobType.STREAMING in job_types
        assert JobType.HEALTH_CHECK in job_types
    
    @patch('anomaly_detection.worker.AnomalyDetectionWorker')
    @patch('asyncio.sleep')
    async def test_run_worker_demo_with_delays(self, mock_sleep, mock_worker_class):
        """Test worker demo includes proper delays."""
        # Mock worker
        mock_worker = Mock()
        mock_worker.add_job = Mock(return_value="job_123")
        mock_worker.start = AsyncMock()
        mock_worker.stop = AsyncMock()
        mock_worker.get_worker_stats = Mock(return_value={
            "queued_jobs": 0,
            "completed_jobs": 5,
            "failed_jobs": 0,
            "total_jobs": 5
        })
        mock_worker.get_job_status = Mock(return_value={
            "status": "completed"
        })
        mock_worker_class.return_value = mock_worker
        
        await run_worker_demo()
        
        # Should have sleep calls for delays
        assert mock_sleep.call_count >= 2  # At least some delays
    
    @patch('anomaly_detection.worker.get_logger')
    @patch('anomaly_detection.worker.AnomalyDetectionWorker')
    async def test_run_worker_demo_logging(self, mock_worker_class, mock_get_logger):
        """Test worker demo includes proper logging."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Mock worker
        mock_worker = Mock()
        mock_worker.add_job = Mock(return_value="job_123")
        mock_worker.start = AsyncMock()
        mock_worker.stop = AsyncMock()
        mock_worker.get_worker_stats = Mock(return_value={
            "queued_jobs": 0,
            "completed_jobs": 1,
            "failed_jobs": 0,
            "total_jobs": 1
        })
        mock_worker.get_job_status = Mock(return_value={
            "status": "completed"
        })
        mock_worker_class.return_value = mock_worker
        
        await run_worker_demo()
        
        # Should have logged demo progress
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Starting Anomaly Detection Worker Demo" in call for call in log_calls)
        assert any("Worker Demo completed" in call for call in log_calls)


class TestWorkerMain:
    """Test cases for worker main function."""
    
    @patch('anomaly_detection.worker.run_worker_demo')
    @patch('anomaly_detection.worker.get_logger')
    def test_main_function(self, mock_get_logger, mock_run_demo):
        """Test main function runs worker demo."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Mock run_worker_demo as coroutine
        mock_run_demo.return_value = AsyncMock()
        
        main()
        
        # Should have called run_worker_demo
        mock_run_demo.assert_called_once()
    
    @patch('anomaly_detection.worker.run_worker_demo')
    @patch('anomaly_detection.worker.get_logger')
    def test_main_function_with_exception(self, mock_get_logger, mock_run_demo):
        """Test main function handles exceptions."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Mock run_worker_demo to raise exception
        mock_run_demo.side_effect = Exception("Demo failed")
        
        # Should not raise exception, but log it
        main()
        
        mock_logger.error.assert_called()
    
    @patch('asyncio.run')
    @patch('anomaly_detection.worker.get_logger')
    def test_main_function_asyncio_run(self, mock_get_logger, mock_asyncio_run):
        """Test main function uses asyncio.run."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        main()
        
        # Should have called asyncio.run
        mock_asyncio_run.assert_called_once()


class TestWorkerPerformance:
    """Test cases for worker performance and resource management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.worker = AnomalyDetectionWorker(max_workers=4)
    
    def test_concurrent_job_processing(self):
        """Test that worker can handle concurrent jobs."""
        # Add multiple jobs
        job_ids = []
        for i in range(10):
            job_id = self.worker.add_job(
                JobType.DETECTION,
                {"data": [[i, i+1]], "algorithm": "isolation_forest"},
                JobPriority.NORMAL
            )
            job_ids.append(job_id)
        
        assert len(job_ids) == 10
        assert self.worker.job_queue.size() == 10
        
        # All jobs should have unique IDs
        assert len(set(job_ids)) == 10
    
    def test_job_priority_handling(self):
        """Test that high priority jobs are processed first."""
        # Add jobs with different priorities
        low_priority_job = self.worker.add_job(
            JobType.DETECTION, {"data": [[1, 2]]}, JobPriority.LOW
        )
        critical_priority_job = self.worker.add_job(
            JobType.DETECTION, {"data": [[3, 4]]}, JobPriority.CRITICAL
        )
        normal_priority_job = self.worker.add_job(
            JobType.DETECTION, {"data": [[5, 6]]}, JobPriority.NORMAL
        )
        
        # Get jobs in processing order
        first_job = self.worker.job_queue.get_next_job()
        second_job = self.worker.job_queue.get_next_job()
        third_job = self.worker.job_queue.get_next_job()
        
        # Should be processed in priority order
        assert first_job.job_id == critical_priority_job
        assert second_job.job_id == normal_priority_job
        assert third_job.job_id == low_priority_job
    
    def test_worker_resource_cleanup(self):
        """Test that worker properly cleans up resources."""
        worker = AnomalyDetectionWorker(max_workers=2)
        
        # Add some jobs
        worker.add_job(JobType.DETECTION, {"data": [[1, 2]]})
        worker.add_job(JobType.ENSEMBLE, {"data": [[3, 4]]})
        
        # Simulate completed jobs
        completed_job = Job("completed", JobType.DETECTION, {})
        completed_job.status = JobStatus.COMPLETED
        worker.completed_jobs.append(completed_job)
        
        # Clear resources
        worker.clear_completed_jobs()
        worker.job_queue.clear()
        
        assert len(worker.completed_jobs) == 0
        assert worker.job_queue.size() == 0
    
    async def test_worker_graceful_shutdown(self):
        """Test worker graceful shutdown."""
        worker = AnomalyDetectionWorker(max_workers=2)
        
        # Start worker
        await worker.start()
        assert worker.is_running is True
        
        # Add a job
        worker.add_job(JobType.DETECTION, {"data": [[1, 2]]})
        
        # Stop worker gracefully
        await worker.stop()
        assert worker.is_running is False
        
        # Executor should be shut down
        assert worker.executor._shutdown is True


class TestWorkerErrorHandling:
    """Test cases for worker error handling and recovery."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.worker = AnomalyDetectionWorker(max_workers=2)
        
        # Mock services that can fail
        self.mock_detection_service = Mock()
        self.worker.detection_service = self.mock_detection_service
    
    async def test_job_retry_mechanism(self):
        """Test job retry mechanism on failure."""
        # Mock service to fail first time, succeed second time
        self.mock_detection_service.detect_anomalies = AsyncMock(
            side_effect=[Exception("Temporary failure"), {"anomalies": [0, 1]}]
        )
        
        job = Job(
            job_id="retry_job_123",
            job_type=JobType.DETECTION,
            data={"algorithm": "isolation_forest", "data": [[1, 2]]}
        )
        
        # First attempt should fail
        with pytest.raises(Exception, match="Temporary failure"):
            await self.worker._process_job(job)
        
        # Increment retry count
        job.retry_count += 1
        
        # Second attempt should succeed
        result = await self.worker._process_job(job)
        assert result is not None
        assert result["anomalies"] == [0, 1]
    
    async def test_job_max_retries(self):
        """Test job failure after max retries."""
        # Mock service to always fail
        self.mock_detection_service.detect_anomalies = AsyncMock(
            side_effect=Exception("Persistent failure")
        )
        
        job = Job(
            job_id="failing_job_123",
            job_type=JobType.DETECTION,
            data={"algorithm": "isolation_forest", "data": [[1, 2]]}
        )
        
        # Simulate max retries
        max_retries = 3
        for retry in range(max_retries + 1):
            with pytest.raises(Exception, match="Persistent failure"):
                await self.worker._process_job(job)
            job.retry_count += 1
        
        # Job should be marked as failed after max retries
        assert job.retry_count > max_retries
    
    def test_invalid_job_data_handling(self):
        """Test handling of jobs with invalid data."""
        # Job with missing required data
        job_id = self.worker.add_job(
            JobType.DETECTION,
            {},  # Empty data
            JobPriority.NORMAL
        )
        
        # Job should still be added (validation happens during processing)
        assert job_id is not None
        assert self.worker.job_queue.size() == 1
    
    def test_worker_state_consistency(self):
        """Test worker maintains consistent state during errors."""
        worker = AnomalyDetectionWorker(max_workers=2)
        
        # Add jobs
        job_ids = []
        for i in range(5):
            job_id = worker.add_job(JobType.DETECTION, {"data": [[i, i+1]]})
            job_ids.append(job_id)
        
        # Simulate some jobs completing and some failing
        for i in range(2):
            completed_job = Job(f"completed_{i}", JobType.DETECTION, {})
            completed_job.status = JobStatus.COMPLETED
            worker.completed_jobs.append(completed_job)
        
        for i in range(1):
            failed_job = Job(f"failed_{i}", JobType.DETECTION, {})
            failed_job.status = JobStatus.FAILED
            worker.failed_jobs.append(failed_job)
        
        # Check worker stats consistency
        stats = worker.get_worker_stats()
        assert stats["queued_jobs"] == 5
        assert stats["completed_jobs"] == 2
        assert stats["failed_jobs"] == 1
        assert stats["total_jobs"] == 8  # 5 queued + 2 completed + 1 failed


class TestWorkerIntegration:
    """Integration test cases for worker components."""
    
    @patch('anomaly_detection.worker.DetectionService')
    @patch('anomaly_detection.worker.EnsembleService')
    def test_worker_service_integration(self, mock_ensemble_class, mock_detection_class):
        """Test worker integration with actual services."""
        # Mock service instances
        mock_detection = Mock()
        mock_ensemble = Mock()
        mock_detection_class.return_value = mock_detection
        mock_ensemble_class.return_value = mock_ensemble
        
        worker = AnomalyDetectionWorker(max_workers=2)
        
        # Services should be initialized
        assert worker.detection_service is not None
        assert worker.ensemble_service is not None
        
        # Service classes should be instantiated
        mock_detection_class.assert_called_once()
        mock_ensemble_class.assert_called_once()
    
    def test_end_to_end_job_processing(self):
        """Test end-to-end job processing workflow."""
        worker = AnomalyDetectionWorker(max_workers=1)
        
        # Add a detection job
        job_id = worker.add_job(
            JobType.DETECTION,
            {
                "algorithm": "isolation_forest",
                "data": [[1, 2], [3, 4], [5, 6]],
                "contamination": 0.1
            },
            JobPriority.HIGH
        )
        
        # Job should be queued
        assert worker.job_queue.size() == 1
        
        # Get job status
        status = worker.get_job_status(job_id)
        assert status["status"] == JobStatus.QUEUED.value
        assert status["job_id"] == job_id
        
        # Job should exist in queue
        job = worker.job_queue.get_next_job()
        assert job is not None
        assert job.job_id == job_id
        assert job.job_type == JobType.DETECTION
        
        # Queue should be empty after getting job
        assert worker.job_queue.size() == 0
    
    def test_multiple_job_types_integration(self):
        """Test processing multiple different job types."""
        worker = AnomalyDetectionWorker(max_workers=2)
        
        # Add different types of jobs
        detection_job = worker.add_job(JobType.DETECTION, {"data": [[1, 2]]})
        ensemble_job = worker.add_job(JobType.ENSEMBLE, {"data": [[3, 4]]})
        training_job = worker.add_job(JobType.TRAINING, {"data": [[5, 6]]})
        health_job = worker.add_job(JobType.HEALTH_CHECK, {})
        
        assert worker.job_queue.size() == 4
        
        # All jobs should have different IDs
        job_ids = [detection_job, ensemble_job, training_job, health_job]
        assert len(set(job_ids)) == 4
        
        # Get worker stats
        stats = worker.get_worker_stats()
        assert stats["queued_jobs"] == 4
        assert stats["total_jobs"] == 4