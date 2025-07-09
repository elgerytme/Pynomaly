"""
Unit tests for the anomaly detection orchestrator module.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from pynomaly.infrastructure.anomaly_detection.orchestrator import (
    AnomalyDetectionOrchestrator,
    DetectionTask,
    DetectionTaskStatus,
    DetectionMode,
    DetectionTaskMetrics,
    ModelMetrics
)
from pynomaly.infrastructure.config.enhanced_config_loader import AnomalyDetectionConfig
from pynomaly.shared.exceptions import (
    AnomalyDetectionError,
    TaskExecutionError,
    ConfigurationError
)


class TestDetectionMode:
    """Test DetectionMode enum."""
    
    def test_all_mode_values(self):
        """Test all mode values are present."""
        assert DetectionMode.BATCH.value == "batch"
        assert DetectionMode.STREAM.value == "stream"
        assert DetectionMode.ENSEMBLE.value == "ensemble"


class TestDetectionTaskStatus:
    """Test DetectionTaskStatus enum."""
    
    def test_all_status_values(self):
        """Test all status values are present."""
        assert DetectionTaskStatus.PENDING.value == "pending"
        assert DetectionTaskStatus.RUNNING.value == "running"
        assert DetectionTaskStatus.COMPLETED.value == "completed"
        assert DetectionTaskStatus.FAILED.value == "failed"
        assert DetectionTaskStatus.RETRYING.value == "retrying"


class TestDetectionTask:
    """Test DetectionTask dataclass."""
    
    def test_default_values(self):
        """Test default values of DetectionTask."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3], [4, 5, 6]]}
        )
        
        assert task.task_id == "test-task"
        assert task.model_id == "test-model"
        assert task.data == {"features": [[1, 2, 3], [4, 5, 6]]}
        assert task.status == DetectionTaskStatus.PENDING
        assert task.mode == DetectionMode.BATCH
        assert task.config == {}
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.priority == 5
        assert task.error_message is None
        assert task.results is None
        assert isinstance(task.created_at, datetime)
        assert task.started_at is None
        assert task.completed_at is None
    
    def test_custom_values(self):
        """Test custom values of DetectionTask."""
        custom_time = datetime.now() - timedelta(hours=1)
        config = {"threshold": 0.8}
        results = {"anomalies": [0, 1, 0]}
        
        task = DetectionTask(
            task_id="custom-task",
            model_id="custom-model",
            data={"features": [[1, 2], [3, 4]]},
            status=DetectionTaskStatus.RUNNING,
            mode=DetectionMode.STREAM,
            config=config,
            retry_count=2,
            max_retries=5,
            priority=10,
            error_message="Custom error",
            results=results,
            created_at=custom_time,
            started_at=custom_time,
            completed_at=custom_time
        )
        
        assert task.task_id == "custom-task"
        assert task.model_id == "custom-model"
        assert task.data == {"features": [[1, 2], [3, 4]]}
        assert task.status == DetectionTaskStatus.RUNNING
        assert task.mode == DetectionMode.STREAM
        assert task.config == config
        assert task.retry_count == 2
        assert task.max_retries == 5
        assert task.priority == 10
        assert task.error_message == "Custom error"
        assert task.results == results
        assert task.created_at == custom_time
        assert task.started_at == custom_time
        assert task.completed_at == custom_time


class TestDetectionTaskMetrics:
    """Test DetectionTaskMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values of DetectionTaskMetrics."""
        metrics = DetectionTaskMetrics()
        
        assert metrics.total_tasks == 0
        assert metrics.completed_tasks == 0
        assert metrics.failed_tasks == 0
        assert metrics.retrying_tasks == 0
        assert metrics.active_tasks == 0
        assert metrics.avg_execution_time == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.total_samples_processed == 0
        assert metrics.total_anomalies_detected == 0
        assert metrics.avg_confidence_score == 0.0
        assert isinstance(metrics.last_updated, datetime)
    
    def test_custom_values(self):
        """Test custom values of DetectionTaskMetrics."""
        custom_time = datetime.now() - timedelta(hours=1)
        
        metrics = DetectionTaskMetrics(
            total_tasks=1000,
            completed_tasks=950,
            failed_tasks=30,
            retrying_tasks=10,
            active_tasks=10,
            avg_execution_time=2.5,
            success_rate=95.0,
            total_samples_processed=100000,
            total_anomalies_detected=5000,
            avg_confidence_score=0.85,
            last_updated=custom_time
        )
        
        assert metrics.total_tasks == 1000
        assert metrics.completed_tasks == 950
        assert metrics.failed_tasks == 30
        assert metrics.retrying_tasks == 10
        assert metrics.active_tasks == 10
        assert metrics.avg_execution_time == 2.5
        assert metrics.success_rate == 95.0
        assert metrics.total_samples_processed == 100000
        assert metrics.total_anomalies_detected == 5000
        assert metrics.avg_confidence_score == 0.85
        assert metrics.last_updated == custom_time
    
    def test_get_success_rate(self):
        """Test success rate calculation."""
        metrics = DetectionTaskMetrics(
            total_tasks=100,
            completed_tasks=95,
            failed_tasks=5
        )
        
        assert metrics.get_success_rate() == 95.0
        
        # Test with no tasks
        empty_metrics = DetectionTaskMetrics()
        assert empty_metrics.get_success_rate() == 0.0


class TestModelMetrics:
    """Test ModelMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values of ModelMetrics."""
        metrics = ModelMetrics()
        
        assert metrics.models_loaded == 0
        assert metrics.avg_model_load_time == 0.0
        assert metrics.avg_prediction_time == 0.0
        assert metrics.model_accuracy == 0.0
        assert metrics.model_precision == 0.0
        assert metrics.model_recall == 0.0
        assert metrics.model_f1_score == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.cpu_usage_percent == 0.0
        assert isinstance(metrics.last_updated, datetime)
    
    def test_custom_values(self):
        """Test custom values of ModelMetrics."""
        custom_time = datetime.now() - timedelta(hours=1)
        
        metrics = ModelMetrics(
            models_loaded=5,
            avg_model_load_time=3.2,
            avg_prediction_time=0.15,
            model_accuracy=0.92,
            model_precision=0.88,
            model_recall=0.85,
            model_f1_score=0.86,
            memory_usage_mb=1024.0,
            cpu_usage_percent=65.0,
            last_updated=custom_time
        )
        
        assert metrics.models_loaded == 5
        assert metrics.avg_model_load_time == 3.2
        assert metrics.avg_prediction_time == 0.15
        assert metrics.model_accuracy == 0.92
        assert metrics.model_precision == 0.88
        assert metrics.model_recall == 0.85
        assert metrics.model_f1_score == 0.86
        assert metrics.memory_usage_mb == 1024.0
        assert metrics.cpu_usage_percent == 65.0
        assert metrics.last_updated == custom_time


class TestAnomalyDetectionOrchestrator:
    """Test AnomalyDetectionOrchestrator class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock anomaly detection configuration."""
        config = Mock(spec=AnomalyDetectionConfig)
        config.mode = DetectionMode.BATCH
        config.max_workers = 8
        config.max_retries = 3
        config.retry_delay = 5.0
        config.timeout = 60.0
        config.queue_size = 200
        config.model_cache_size = 10
        config.batch_size = 100
        return config
    
    @pytest.fixture
    def orchestrator(self, mock_config):
        """Create an anomaly detection orchestrator instance."""
        with patch('pynomaly.infrastructure.anomaly_detection.orchestrator.StructuredLogger'):
            with patch('pynomaly.infrastructure.anomaly_detection.orchestrator.MetricsService'):
                return AnomalyDetectionOrchestrator(config=mock_config)
    
    def test_init(self, orchestrator, mock_config):
        """Test orchestrator initialization."""
        assert orchestrator.config == mock_config
        assert orchestrator.orchestrator_id is not None
        assert len(orchestrator.orchestrator_id) > 0
        assert orchestrator.running is False
        assert orchestrator.shutdown_event is not None
        assert orchestrator.task_queue is not None
        assert orchestrator.active_tasks == {}
        assert orchestrator.completed_tasks == []
        assert orchestrator.failed_tasks == []
        assert orchestrator.workers == []
        assert orchestrator.models == {}
        assert isinstance(orchestrator.metrics, DetectionTaskMetrics)
        assert isinstance(orchestrator.model_metrics, ModelMetrics)
    
    def test_init_default_config(self):
        """Test orchestrator initialization with default config."""
        with patch('pynomaly.infrastructure.anomaly_detection.orchestrator.StructuredLogger'):
            with patch('pynomaly.infrastructure.anomaly_detection.orchestrator.MetricsService'):
                orchestrator = AnomalyDetectionOrchestrator()
                
                assert orchestrator.config is not None
                assert orchestrator.config.mode == DetectionMode.BATCH
                assert orchestrator.config.max_workers == 4
                assert orchestrator.config.max_retries == 3
                assert orchestrator.config.timeout == 60.0
    
    @pytest.mark.asyncio
    async def test_start_success(self, orchestrator):
        """Test successful orchestrator start."""
        with patch.object(orchestrator, '_start_workers', new_callable=AsyncMock) as mock_start_workers:
            with patch.object(orchestrator, '_start_metrics_collector', new_callable=AsyncMock) as mock_start_metrics:
                await orchestrator.start()
                
                assert orchestrator.running is True
                mock_start_workers.assert_called_once()
                mock_start_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, orchestrator):
        """Test starting orchestrator when already running."""
        orchestrator.running = True
        
        with pytest.raises(RuntimeError, match="Anomaly detection orchestrator is already running"):
            await orchestrator.start()
    
    @pytest.mark.asyncio
    async def test_stop_success(self, orchestrator):
        """Test successful orchestrator stop."""
        # Set up running state
        orchestrator.running = True
        
        with patch.object(orchestrator, '_stop_workers', new_callable=AsyncMock) as mock_stop_workers:
            with patch.object(orchestrator, '_stop_metrics_collector', new_callable=AsyncMock) as mock_stop_metrics:
                await orchestrator.stop()
                
                assert orchestrator.running is False
                mock_stop_workers.assert_called_once()
                mock_stop_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_not_running(self, orchestrator):
        """Test stopping orchestrator when not running."""
        assert orchestrator.running is False
        
        # Should not raise an exception
        await orchestrator.stop()
        
        assert orchestrator.running is False
    
    @pytest.mark.asyncio
    async def test_submit_task_success(self, orchestrator):
        """Test successful task submission."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3]]}
        )
        
        with patch.object(orchestrator.task_queue, 'put', new_callable=AsyncMock) as mock_put:
            await orchestrator.submit_task(task)
            
            mock_put.assert_called_once_with(task)
    
    @pytest.mark.asyncio
    async def test_submit_task_not_running(self, orchestrator):
        """Test task submission when orchestrator is not running."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3]]}
        )
        
        assert orchestrator.running is False
        
        with pytest.raises(RuntimeError, match="Anomaly detection orchestrator is not running"):
            await orchestrator.submit_task(task)
    
    @pytest.mark.asyncio
    async def test_create_task_success(self, orchestrator):
        """Test successful task creation."""
        task = await orchestrator.create_task(
            model_id="test-model",
            data={"features": [[1, 2, 3]]},
            mode=DetectionMode.STREAM,
            config={"threshold": 0.8},
            priority=10
        )
        
        assert task.model_id == "test-model"
        assert task.data == {"features": [[1, 2, 3]]}
        assert task.mode == DetectionMode.STREAM
        assert task.config == {"threshold": 0.8}
        assert task.priority == 10
        assert task.status == DetectionTaskStatus.PENDING
        assert task.task_id is not None
        assert len(task.task_id) > 0
    
    @pytest.mark.asyncio
    async def test_create_and_submit_task_success(self, orchestrator):
        """Test successful task creation and submission."""
        orchestrator.running = True
        
        with patch.object(orchestrator, 'submit_task', new_callable=AsyncMock) as mock_submit:
            task = await orchestrator.create_and_submit_task(
                model_id="test-model",
                data={"features": [[1, 2, 3]]},
                mode=DetectionMode.STREAM
            )
            
            assert task.model_id == "test-model"
            assert task.data == {"features": [[1, 2, 3]]}
            assert task.mode == DetectionMode.STREAM
            mock_submit.assert_called_once_with(task)
    
    @pytest.mark.asyncio
    async def test_get_task_status_found(self, orchestrator):
        """Test getting task status when task exists."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3]]},
            status=DetectionTaskStatus.RUNNING
        )
        
        orchestrator.active_tasks["test-task"] = task
        
        status = await orchestrator.get_task_status("test-task")
        
        assert status == DetectionTaskStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, orchestrator):
        """Test getting task status when task doesn't exist."""
        status = await orchestrator.get_task_status("non-existent-task")
        
        assert status is None
    
    @pytest.mark.asyncio
    async def test_get_task_results_found(self, orchestrator):
        """Test getting task results when task exists."""
        results = {"anomalies": [0, 1, 0], "scores": [0.1, 0.9, 0.2]}
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3]]},
            status=DetectionTaskStatus.COMPLETED,
            results=results
        )
        
        orchestrator.active_tasks["test-task"] = task
        
        task_results = await orchestrator.get_task_results("test-task")
        
        assert task_results == results
    
    @pytest.mark.asyncio
    async def test_get_task_results_not_found(self, orchestrator):
        """Test getting task results when task doesn't exist."""
        task_results = await orchestrator.get_task_results("non-existent-task")
        
        assert task_results is None
    
    @pytest.mark.asyncio
    async def test_cancel_task_success(self, orchestrator):
        """Test successful task cancellation."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3]]},
            status=DetectionTaskStatus.RUNNING
        )
        
        orchestrator.active_tasks["test-task"] = task
        
        success = await orchestrator.cancel_task("test-task")
        
        assert success is True
        assert task.status == DetectionTaskStatus.FAILED
        assert task.error_message == "Task cancelled"
    
    @pytest.mark.asyncio
    async def test_cancel_task_not_found(self, orchestrator):
        """Test task cancellation when task doesn't exist."""
        success = await orchestrator.cancel_task("non-existent-task")
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_cancel_task_already_completed(self, orchestrator):
        """Test task cancellation when task is already completed."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3]]},
            status=DetectionTaskStatus.COMPLETED
        )
        
        orchestrator.active_tasks["test-task"] = task
        
        success = await orchestrator.cancel_task("test-task")
        
        assert success is False
        assert task.status == DetectionTaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, orchestrator):
        """Test successful model loading."""
        mock_model = Mock()
        
        with patch.object(orchestrator, '_load_model_from_storage', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_model
            
            await orchestrator.load_model("test-model", {"model_path": "/path/to/model"})
            
            # Verify model was loaded and cached
            assert "test-model" in orchestrator.models
            assert orchestrator.models["test-model"] == mock_model
            mock_load.assert_called_once_with("test-model", {"model_path": "/path/to/model"})
    
    @pytest.mark.asyncio
    async def test_load_model_already_loaded(self, orchestrator):
        """Test loading model when already loaded."""
        mock_model = Mock()
        orchestrator.models["test-model"] = mock_model
        
        with patch.object(orchestrator, '_load_model_from_storage', new_callable=AsyncMock) as mock_load:
            await orchestrator.load_model("test-model", {"model_path": "/path/to/model"})
            
            # Verify model was not loaded again
            assert orchestrator.models["test-model"] == mock_model
            mock_load.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_unload_model_success(self, orchestrator):
        """Test successful model unloading."""
        mock_model = Mock()
        orchestrator.models["test-model"] = mock_model
        
        success = await orchestrator.unload_model("test-model")
        
        assert success is True
        assert "test-model" not in orchestrator.models
    
    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self, orchestrator):
        """Test unloading model when not loaded."""
        success = await orchestrator.unload_model("non-existent-model")
        
        assert success is False
    
    def test_get_metrics(self, orchestrator):
        """Test getting orchestrator metrics."""
        metrics = orchestrator.get_metrics()
        
        assert isinstance(metrics, DetectionTaskMetrics)
        assert metrics == orchestrator.metrics
    
    def test_get_model_metrics(self, orchestrator):
        """Test getting model metrics."""
        metrics = orchestrator.get_model_metrics()
        
        assert isinstance(metrics, ModelMetrics)
        assert metrics == orchestrator.model_metrics
    
    def test_get_loaded_models(self, orchestrator):
        """Test getting loaded models."""
        mock_model1 = Mock()
        mock_model2 = Mock()
        orchestrator.models["model-1"] = mock_model1
        orchestrator.models["model-2"] = mock_model2
        
        loaded_models = orchestrator.get_loaded_models()
        
        assert len(loaded_models) == 2
        assert "model-1" in loaded_models
        assert "model-2" in loaded_models
    
    def test_get_active_tasks(self, orchestrator):
        """Test getting active tasks."""
        task1 = DetectionTask(
            task_id="task-1",
            model_id="model-1",
            data={"features": [[1, 2, 3]]}
        )
        task2 = DetectionTask(
            task_id="task-2",
            model_id="model-2",
            data={"features": [[4, 5, 6]]}
        )
        
        orchestrator.active_tasks["task-1"] = task1
        orchestrator.active_tasks["task-2"] = task2
        
        active_tasks = orchestrator.get_active_tasks()
        
        assert len(active_tasks) == 2
        assert task1 in active_tasks
        assert task2 in active_tasks
    
    def test_get_completed_tasks(self, orchestrator):
        """Test getting completed tasks."""
        task1 = DetectionTask(
            task_id="task-1",
            model_id="model-1",
            data={"features": [[1, 2, 3]]},
            status=DetectionTaskStatus.COMPLETED
        )
        task2 = DetectionTask(
            task_id="task-2",
            model_id="model-2",
            data={"features": [[4, 5, 6]]},
            status=DetectionTaskStatus.COMPLETED
        )
        
        orchestrator.completed_tasks = [task1, task2]
        
        completed_tasks = orchestrator.get_completed_tasks()
        
        assert len(completed_tasks) == 2
        assert task1 in completed_tasks
        assert task2 in completed_tasks
    
    def test_get_failed_tasks(self, orchestrator):
        """Test getting failed tasks."""
        task1 = DetectionTask(
            task_id="task-1",
            model_id="model-1",
            data={"features": [[1, 2, 3]]},
            status=DetectionTaskStatus.FAILED
        )
        task2 = DetectionTask(
            task_id="task-2",
            model_id="model-2",
            data={"features": [[4, 5, 6]]},
            status=DetectionTaskStatus.FAILED
        )
        
        orchestrator.failed_tasks = [task1, task2]
        
        failed_tasks = orchestrator.get_failed_tasks()
        
        assert len(failed_tasks) == 2
        assert task1 in failed_tasks
        assert task2 in failed_tasks
    
    @pytest.mark.asyncio
    async def test_process_task_success(self, orchestrator):
        """Test successful task processing."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3]]}
        )
        
        mock_model = Mock()
        orchestrator.models["test-model"] = mock_model
        
        # Mock successful processing
        with patch.object(orchestrator, '_execute_detection', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"anomalies": [0, 1, 0], "scores": [0.1, 0.9, 0.2]}
            
            await orchestrator._process_task(task)
            
            # Verify task was executed
            mock_execute.assert_called_once_with(task, mock_model)
            
            # Verify task status was updated
            assert task.status == DetectionTaskStatus.COMPLETED
            assert task.results == {"anomalies": [0, 1, 0], "scores": [0.1, 0.9, 0.2]}
            assert task.started_at is not None
            assert task.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_process_task_failure(self, orchestrator):
        """Test task processing with failure."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3]]}
        )
        
        mock_model = Mock()
        orchestrator.models["test-model"] = mock_model
        
        # Mock task execution failure
        with patch.object(orchestrator, '_execute_detection', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("Detection failed")
            
            await orchestrator._process_task(task)
            
            # Verify task status was updated
            assert task.status == DetectionTaskStatus.FAILED
            assert task.error_message == "Detection failed"
            assert task.started_at is not None
            assert task.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_process_task_retry(self, orchestrator):
        """Test task processing with retry."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3]]},
            retry_count=0,
            max_retries=3
        )
        
        mock_model = Mock()
        orchestrator.models["test-model"] = mock_model
        
        # Mock task execution failure
        with patch.object(orchestrator, '_execute_detection', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("Detection failed")
            
            with patch.object(orchestrator, '_retry_task', new_callable=AsyncMock) as mock_retry:
                await orchestrator._process_task(task)
                
                # Verify retry was attempted
                mock_retry.assert_called_once_with(task)
                
                # Verify task status was updated for retry
                assert task.status == DetectionTaskStatus.RETRYING
    
    @pytest.mark.asyncio
    async def test_execute_detection_batch_mode(self, orchestrator):
        """Test detection execution in batch mode."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3], [4, 5, 6]]},
            mode=DetectionMode.BATCH
        )
        
        mock_model = Mock()
        mock_model.predict.return_value = [0, 1]
        
        # Mock the actual detection logic
        with patch.object(orchestrator, '_detect_batch', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = {"anomalies": [0, 1], "scores": [0.1, 0.9]}
            
            results = await orchestrator._execute_detection(task, mock_model)
            
            # Verify batch detection was called
            mock_detect.assert_called_once_with(task, mock_model)
            assert results == {"anomalies": [0, 1], "scores": [0.1, 0.9]}
    
    @pytest.mark.asyncio
    async def test_execute_detection_stream_mode(self, orchestrator):
        """Test detection execution in stream mode."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3]]},
            mode=DetectionMode.STREAM
        )
        
        mock_model = Mock()
        
        # Mock the actual detection logic
        with patch.object(orchestrator, '_detect_stream', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = {"anomalies": [0], "scores": [0.1]}
            
            results = await orchestrator._execute_detection(task, mock_model)
            
            # Verify stream detection was called
            mock_detect.assert_called_once_with(task, mock_model)
            assert results == {"anomalies": [0], "scores": [0.1]}
    
    @pytest.mark.asyncio
    async def test_execute_detection_ensemble_mode(self, orchestrator):
        """Test detection execution in ensemble mode."""
        task = DetectionTask(
            task_id="test-task",
            model_id="test-model",
            data={"features": [[1, 2, 3], [4, 5, 6]]},
            mode=DetectionMode.ENSEMBLE
        )
        
        mock_model = Mock()
        
        # Mock the actual detection logic
        with patch.object(orchestrator, '_detect_ensemble', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = {"anomalies": [0, 1], "scores": [0.1, 0.9]}
            
            results = await orchestrator._execute_detection(task, mock_model)
            
            # Verify ensemble detection was called
            mock_detect.assert_called_once_with(task, mock_model)
            assert results == {"anomalies": [0, 1], "scores": [0.1, 0.9]}


class TestAnomalyDetectionOrchestratorIntegration:
    """Integration tests for anomaly detection orchestrator."""
    
    @pytest.mark.asyncio
    async def test_full_detection_lifecycle(self):
        """Test the full lifecycle of a detection task."""
        config = Mock(spec=AnomalyDetectionConfig)
        config.mode = DetectionMode.BATCH
        config.max_workers = 2
        config.max_retries = 3
        config.retry_delay = 0.1  # Short delay for testing
        
        with patch('pynomaly.infrastructure.anomaly_detection.orchestrator.StructuredLogger'):
            with patch('pynomaly.infrastructure.anomaly_detection.orchestrator.MetricsService'):
                orchestrator = AnomalyDetectionOrchestrator(config=config)
                
                # Start orchestrator
                await orchestrator.start()
                assert orchestrator.running is True
                
                # Load a model
                await orchestrator.load_model("test-model", {"model_path": "/path/to/model"})
                assert "test-model" in orchestrator.models
                
                # Create and submit task
                task = await orchestrator.create_and_submit_task(
                    model_id="test-model",
                    data={"features": [[1, 2, 3], [4, 5, 6]]},
                    mode=DetectionMode.BATCH
                )
                
                # Wait for task to be processed
                await asyncio.sleep(0.2)
                
                # Check task status
                status = await orchestrator.get_task_status(task.task_id)
                assert status in [DetectionTaskStatus.COMPLETED, DetectionTaskStatus.FAILED]
                
                # Get metrics
                metrics = orchestrator.get_metrics()
                assert metrics.total_tasks >= 1
                
                # Unload model
                success = await orchestrator.unload_model("test-model")
                assert success is True
                assert "test-model" not in orchestrator.models
                
                # Stop orchestrator
                await orchestrator.stop()
                assert orchestrator.running is False
