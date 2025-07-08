"""Tests for automated retraining use case."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock
from uuid import uuid4

from pynomaly.application.use_cases.automated_retraining_use_case import (
    AutomatedRetrainingUseCase,
    RetrainingPolicy,
    RetrainingDecision,
    RetrainingConfiguration,
    RetrainingRequest,
    RetrainingResponse,
    PerformanceDegradationTrigger,
)
from pynomaly.application.services.automated_training_service import (
    AutomatedTrainingService,
    TrainingConfig,
    TrainingResult,
    TrainingStatus,
    TriggerType,
)
from pynomaly.application.use_cases.drift_monitoring_use_case import DriftMonitoringUseCase
from pynomaly.application.use_cases.train_detector import TrainDetectorUseCase
from pynomaly.domain.entities import Detector
from pynomaly.domain.entities.drift_detection import DriftDetectionResult
from pynomaly.shared.protocols import DetectorRepositoryProtocol


class TestAutomatedRetrainingUseCase:
    """Test AutomatedRetrainingUseCase."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector_id = uuid4()
        
        # Create mocks
        self.detector_repository = Mock(spec=DetectorRepositoryProtocol)
        self.training_service = Mock(spec=AutomatedTrainingService)
        self.drift_monitoring_use_case = Mock(spec=DriftMonitoringUseCase)
        self.train_detector_use_case = Mock(spec=TrainDetectorUseCase)
        self.performance_monitor = Mock()
        self.notification_service = Mock()
        
        # Create mock detector
        self.mock_detector = Mock(spec=Detector)
        self.mock_detector.id = self.detector_id
        self.mock_detector.name = "Test Detector"
        self.mock_detector.created_at = datetime.utcnow() - timedelta(days=10)
        
        # Configure repository mock
        self.detector_repository.find_by_id.return_value = self.mock_detector
        
        # Configure training service mocks
        self.training_service.start_training = AsyncMock(return_value="training_123")
        self.training_service.cancel_training = AsyncMock(return_value=True)
        self.training_service.get_training_result = AsyncMock()
        self.training_service.get_training_history = AsyncMock(return_value=[])
        
        # Configure drift monitoring mock
        self.drift_monitoring_use_case.perform_drift_check = AsyncMock()
        
        # Configure performance monitor mocks
        self.performance_monitor.get_performance_metrics = AsyncMock(return_value={
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88
        })
        
        # Configure notification service mock
        self.notification_service.send_notification = AsyncMock()
        
        # Create use case instance
        self.use_case = AutomatedRetrainingUseCase(
            detector_repository=self.detector_repository,
            training_service=self.training_service,
            drift_monitoring_use_case=self.drift_monitoring_use_case,
            train_detector_use_case=self.train_detector_use_case,
            performance_monitor=self.performance_monitor,
            notification_service=self.notification_service
        )
    
    def test_init(self):
        """Test use case initialization."""
        assert self.use_case.detector_repository == self.detector_repository
        assert self.use_case.training_service == self.training_service
        assert self.use_case.drift_monitoring_use_case == self.drift_monitoring_use_case
        assert self.use_case.train_detector_use_case == self.train_detector_use_case
        assert self.use_case.performance_monitor == self.performance_monitor
        assert self.use_case.notification_service == self.notification_service
        assert isinstance(self.use_case.retraining_configs, dict)
        assert isinstance(self.use_case.active_retrainings, dict)
        assert isinstance(self.use_case.monitoring_tasks, dict)
        assert isinstance(self.use_case.performance_history, dict)
    
    @pytest.mark.asyncio
    async def test_configure_retraining_valid(self):
        """Test configuring retraining with valid configuration."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=True,
            performance_triggers=[
                PerformanceDegradationTrigger(
                    metric_name="accuracy",
                    threshold=0.8,
                    evaluation_window_days=7
                )
            ]
        )
        
        result = await self.use_case.configure_retraining(config)
        
        assert result == config
        assert self.detector_id in self.use_case.retraining_configs
        assert self.use_case.retraining_configs[self.detector_id] == config
        self.detector_repository.find_by_id.assert_called_once_with(self.detector_id)
    
    @pytest.mark.asyncio
    async def test_configure_retraining_invalid_detector(self):
        """Test configuring retraining with invalid detector ID."""
        self.detector_repository.find_by_id.return_value = None
        
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=True,
            performance_triggers=[
                PerformanceDegradationTrigger(
                    metric_name="accuracy",
                    threshold=0.8,
                    evaluation_window_days=7
                )
            ]
        )
        
        with pytest.raises(ValueError, match="Detector .* not found"):
            await self.use_case.configure_retraining(config)
    
    @pytest.mark.asyncio
    async def test_configure_retraining_missing_performance_triggers(self):
        """Test configuring retraining without required performance triggers."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=True,
            performance_triggers=[]  # Missing required triggers
        )
        
        with pytest.raises(ValueError, match="Performance triggers required"):
            await self.use_case.configure_retraining(config)
    
    @pytest.mark.asyncio
    async def test_configure_retraining_missing_schedule(self):
        """Test configuring retraining without required schedule configuration."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.SCHEDULED,
            enabled=True
            # Missing schedule_cron and max_model_age_days
        )
        
        with pytest.raises(ValueError, match="Schedule configuration required"):
            await self.use_case.configure_retraining(config)
    
    @pytest.mark.asyncio
    async def test_evaluate_retraining_necessity_no_config(self):
        """Test evaluating retraining necessity with no configuration."""
        decision = await self.use_case.evaluate_retraining_necessity(self.detector_id)
        assert decision == RetrainingDecision.NO_RETRAINING_NEEDED
    
    @pytest.mark.asyncio
    async def test_evaluate_retraining_necessity_disabled(self):
        """Test evaluating retraining necessity with disabled configuration."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=False
        )
        self.use_case.retraining_configs[self.detector_id] = config
        
        decision = await self.use_case.evaluate_retraining_necessity(self.detector_id)
        assert decision == RetrainingDecision.NO_RETRAINING_NEEDED
    
    @pytest.mark.asyncio
    async def test_evaluate_retraining_necessity_performance_degradation(self):
        """Test evaluating retraining necessity with performance degradation."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=True,
            performance_triggers=[
                PerformanceDegradationTrigger(
                    metric_name="accuracy",
                    threshold=0.8,
                    evaluation_window_days=7
                )
            ]
        )
        self.use_case.retraining_configs[self.detector_id] = config
        
        # Mock performance degradation
        self.performance_monitor.get_performance_metrics.return_value = {
            "accuracy": 0.75  # Below threshold
        }
        
        decision = await self.use_case.evaluate_retraining_necessity(self.detector_id)
        assert decision == RetrainingDecision.PERFORMANCE_RETRAINING
    
    @pytest.mark.asyncio
    async def test_evaluate_retraining_necessity_data_drift(self):
        """Test evaluating retraining necessity with data drift."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.DATA_DRIFT,
            enabled=True,
            drift_threshold=0.1
        )
        self.use_case.retraining_configs[self.detector_id] = config
        
        # Mock data drift detection
        drift_result = Mock(spec=DriftDetectionResult)
        drift_result.drift_detected = True
        drift_result.drift_score = 0.15  # Above threshold
        self.drift_monitoring_use_case.perform_drift_check.return_value = drift_result
        
        decision = await self.use_case.evaluate_retraining_necessity(self.detector_id)
        assert decision == RetrainingDecision.DRIFT_RETRAINING
    
    @pytest.mark.asyncio
    async def test_evaluate_retraining_necessity_scheduled(self):
        """Test evaluating retraining necessity with scheduled retraining."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.SCHEDULED,
            enabled=True,
            max_model_age_days=5  # Model is older than this
        )
        self.use_case.retraining_configs[self.detector_id] = config
        
        # Mock old detector
        self.mock_detector.created_at = datetime.utcnow() - timedelta(days=10)
        
        decision = await self.use_case.evaluate_retraining_necessity(self.detector_id)
        assert decision == RetrainingDecision.SCHEDULED_RETRAINING
    
    @pytest.mark.asyncio
    async def test_request_retraining_success(self):
        """Test successful retraining request."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=True,
            performance_triggers=[
                PerformanceDegradationTrigger(
                    metric_name="accuracy",
                    threshold=0.8,
                    evaluation_window_days=7
                )
            ]
        )
        self.use_case.retraining_configs[self.detector_id] = config
        
        # Mock successful training
        training_result = Mock(spec=TrainingResult)
        training_result.status = TrainingStatus.COMPLETED
        training_result.error_message = None
        self.training_service.get_training_result.return_value = training_result
        
        request = RetrainingRequest(
            detector_id=self.detector_id,
            trigger_type=TriggerType.PERFORMANCE_THRESHOLD,
            trigger_reason="Performance degradation detected",
            performance_metrics={"accuracy": 0.75}
        )
        
        response = await self.use_case.request_retraining(request)
        
        assert response.detector_id == self.detector_id
        assert response.decision == RetrainingDecision.PERFORMANCE_RETRAINING
        assert response.retraining_id in self.use_case.active_retrainings
        
        # Verify training was started
        self.training_service.start_training.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_request_retraining_detector_not_found(self):
        """Test retraining request with non-existent detector."""
        self.detector_repository.find_by_id.return_value = None
        
        request = RetrainingRequest(
            detector_id=self.detector_id,
            trigger_type=TriggerType.PERFORMANCE_THRESHOLD,
            trigger_reason="Performance degradation detected"
        )
        
        with pytest.raises(ValueError, match="Detector .* not found"):
            await self.use_case.request_retraining(request)
    
    @pytest.mark.asyncio
    async def test_request_retraining_no_config(self):
        """Test retraining request with no configuration."""
        request = RetrainingRequest(
            detector_id=self.detector_id,
            trigger_type=TriggerType.PERFORMANCE_THRESHOLD,
            trigger_reason="Performance degradation detected"
        )
        
        with pytest.raises(ValueError, match="No retraining configuration"):
            await self.use_case.request_retraining(request)
    
    @pytest.mark.asyncio
    async def test_request_retraining_constraints_violated(self):
        """Test retraining request with constraints violated."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=True,
            min_retraining_interval_hours=24,
            performance_triggers=[
                PerformanceDegradationTrigger(
                    metric_name="accuracy",
                    threshold=0.8,
                    evaluation_window_days=7
                )
            ]
        )
        self.use_case.retraining_configs[self.detector_id] = config
        
        # Add recent completed retraining
        recent_response = RetrainingResponse(
            retraining_id="recent_123",
            detector_id=self.detector_id,
            decision=RetrainingDecision.PERFORMANCE_RETRAINING,
            status="completed",
            completed_at=datetime.utcnow() - timedelta(hours=1)  # Too recent
        )
        self.use_case.active_retrainings["recent_123"] = recent_response
        
        request = RetrainingRequest(
            detector_id=self.detector_id,
            trigger_type=TriggerType.PERFORMANCE_THRESHOLD,
            trigger_reason="Performance degradation detected"
        )
        
        response = await self.use_case.request_retraining(request)
        
        assert response.status == "rejected"
        assert response.message == "Minimum retraining interval not met"
    
    @pytest.mark.asyncio
    async def test_get_retraining_status(self):
        """Test getting retraining status."""
        response = RetrainingResponse(
            retraining_id="test_123",
            detector_id=self.detector_id,
            decision=RetrainingDecision.PERFORMANCE_RETRAINING
        )
        self.use_case.active_retrainings["test_123"] = response
        
        result = await self.use_case.get_retraining_status("test_123")
        assert result == response
        
        # Test non-existent retraining
        result = await self.use_case.get_retraining_status("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cancel_retraining(self):
        """Test canceling retraining."""
        response = RetrainingResponse(
            retraining_id="test_123",
            detector_id=self.detector_id,
            decision=RetrainingDecision.PERFORMANCE_RETRAINING,
            training_id="training_456"
        )
        self.use_case.active_retrainings["test_123"] = response
        
        result = await self.use_case.cancel_retraining("test_123")
        
        assert result is True
        assert response.status == "cancelled"
        assert response.completed_at is not None
        self.training_service.cancel_training.assert_called_once_with("training_456")
        
        # Test non-existent retraining
        result = await self.use_case.cancel_retraining("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_retraining_history(self):
        """Test getting retraining history."""
        # Add some completed retraining responses
        response1 = RetrainingResponse(
            retraining_id="test_123",
            detector_id=self.detector_id,
            decision=RetrainingDecision.PERFORMANCE_RETRAINING,
            completed_at=datetime.utcnow() - timedelta(hours=2)
        )
        
        response2 = RetrainingResponse(
            retraining_id="test_456",
            detector_id=self.detector_id,
            decision=RetrainingDecision.DRIFT_RETRAINING,
            completed_at=datetime.utcnow() - timedelta(hours=1)
        )
        
        response3 = RetrainingResponse(
            retraining_id="test_789",
            detector_id=uuid4(),  # Different detector
            decision=RetrainingDecision.PERFORMANCE_RETRAINING,
            completed_at=datetime.utcnow() - timedelta(hours=3)
        )
        
        self.use_case.active_retrainings["test_123"] = response1
        self.use_case.active_retrainings["test_456"] = response2
        self.use_case.active_retrainings["test_789"] = response3
        
        history = await self.use_case.get_retraining_history(self.detector_id)
        
        assert len(history) == 2
        assert history[0] == response2  # Most recent first
        assert history[1] == response1
    
    @pytest.mark.asyncio
    async def test_can_retrain_minimum_interval(self):
        """Test checking if retraining can be done based on minimum interval."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=True,
            min_retraining_interval_hours=6,
            performance_triggers=[
                PerformanceDegradationTrigger(
                    metric_name="accuracy",
                    threshold=0.8,
                    evaluation_window_days=7
                )
            ]
        )
        self.use_case.retraining_configs[self.detector_id] = config
        
        # No previous retraining - should be able to retrain
        result = await self.use_case._can_retrain(self.detector_id)
        assert result is True
        
        # Add recent completed retraining
        recent_response = RetrainingResponse(
            retraining_id="recent_123",
            detector_id=self.detector_id,
            decision=RetrainingDecision.PERFORMANCE_RETRAINING,
            status="completed",
            completed_at=datetime.utcnow() - timedelta(hours=1)  # Too recent
        )
        self.use_case.active_retrainings["recent_123"] = recent_response
        
        result = await self.use_case._can_retrain(self.detector_id)
        assert result is False
        
        # Older completed retraining - should be able to retrain
        recent_response.completed_at = datetime.utcnow() - timedelta(hours=8)
        result = await self.use_case._can_retrain(self.detector_id)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_can_retrain_concurrent_limit(self):
        """Test checking if retraining can be done based on concurrent limit."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=True,
            max_concurrent_retrainings=2,
            performance_triggers=[
                PerformanceDegradationTrigger(
                    metric_name="accuracy",
                    threshold=0.8,
                    evaluation_window_days=7
                )
            ]
        )
        self.use_case.retraining_configs[self.detector_id] = config
        
        # Add active retraining (within limit)
        active_response1 = RetrainingResponse(
            retraining_id="active_123",
            detector_id=self.detector_id,
            decision=RetrainingDecision.PERFORMANCE_RETRAINING,
            status="running"
        )
        self.use_case.active_retrainings["active_123"] = active_response1
        
        result = await self.use_case._can_retrain(self.detector_id)
        assert result is True
        
        # Add another active retraining (at limit)
        active_response2 = RetrainingResponse(
            retraining_id="active_456",
            detector_id=self.detector_id,
            decision=RetrainingDecision.PERFORMANCE_RETRAINING,
            status="initiated"
        )
        self.use_case.active_retrainings["active_456"] = active_response2
        
        result = await self.use_case._can_retrain(self.detector_id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_performance_degradation_no_monitor(self):
        """Test checking performance degradation without performance monitor."""
        use_case = AutomatedRetrainingUseCase(
            detector_repository=self.detector_repository,
            training_service=self.training_service,
            drift_monitoring_use_case=self.drift_monitoring_use_case,
            train_detector_use_case=self.train_detector_use_case,
            performance_monitor=None  # No monitor
        )
        
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=True,
            performance_triggers=[
                PerformanceDegradationTrigger(
                    metric_name="accuracy",
                    threshold=0.8,
                    evaluation_window_days=7
                )
            ]
        )
        
        result = await use_case._check_performance_degradation(self.detector_id, config)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_performance_degradation_success(self):
        """Test successful performance degradation check."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.PERFORMANCE_DEGRADATION,
            enabled=True,
            performance_triggers=[
                PerformanceDegradationTrigger(
                    metric_name="accuracy",
                    threshold=0.8,
                    evaluation_window_days=7
                )
            ]
        )
        
        # Mock performance metrics below threshold
        self.performance_monitor.get_performance_metrics.return_value = {
            "accuracy": 0.75  # Below threshold
        }
        
        result = await self.use_case._check_performance_degradation(self.detector_id, config)
        assert result is True
        
        # Mock performance metrics above threshold
        self.performance_monitor.get_performance_metrics.return_value = {
            "accuracy": 0.85  # Above threshold
        }
        
        result = await self.use_case._check_performance_degradation(self.detector_id, config)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_data_drift_detected(self):
        """Test data drift detection check."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.DATA_DRIFT,
            enabled=True,
            drift_threshold=0.1
        )
        
        # Mock drift detection result
        drift_result = Mock(spec=DriftDetectionResult)
        drift_result.drift_detected = True
        drift_result.drift_score = 0.15  # Above threshold
        self.drift_monitoring_use_case.perform_drift_check.return_value = drift_result
        
        result = await self.use_case._check_data_drift(self.detector_id, config)
        assert result is True
        
        # Mock no drift detected
        drift_result.drift_detected = False
        result = await self.use_case._check_data_drift(self.detector_id, config)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_scheduled_retraining_model_age(self):
        """Test scheduled retraining check based on model age."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.SCHEDULED,
            enabled=True,
            max_model_age_days=5
        )
        
        # Mock old detector (10 days old)
        self.mock_detector.created_at = datetime.utcnow() - timedelta(days=10)
        
        result = await self.use_case._check_scheduled_retraining(self.detector_id, config)
        assert result is True
        
        # Mock recent detector (2 days old)
        self.mock_detector.created_at = datetime.utcnow() - timedelta(days=2)
        
        result = await self.use_case._check_scheduled_retraining(self.detector_id, config)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_scheduled_retraining_cron_schedule(self):
        """Test scheduled retraining check based on cron schedule."""
        config = RetrainingConfiguration(
            detector_id=self.detector_id,
            policy=RetrainingPolicy.SCHEDULED,
            enabled=True,
            schedule_cron="0 0 * * *"  # Daily at midnight
        )
        
        # Mock old last training (2 days ago)
        training_result = Mock(spec=TrainingResult)
        training_result.completion_time = datetime.utcnow() - timedelta(days=2)
        self.training_service.get_training_history.return_value = [training_result]
        
        result = await self.use_case._check_scheduled_retraining(self.detector_id, config)
        assert result is True
        
        # Mock recent last training (1 hour ago)
        training_result.completion_time = datetime.utcnow() - timedelta(hours=1)
        
        result = await self.use_case._check_scheduled_retraining(self.detector_id, config)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_last_training_time(self):
        """Test getting last training time."""
        # Mock training history
        training_result = Mock(spec=TrainingResult)
        training_result.completion_time = datetime.utcnow() - timedelta(hours=2)
        self.training_service.get_training_history.return_value = [training_result]
        
        result = await self.use_case._get_last_training_time(self.detector_id)
        assert result == training_result.completion_time
        
        # Mock no training history
        self.training_service.get_training_history.return_value = []
        result = await self.use_case._get_last_training_time(self.detector_id)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_wait_for_training_completion(self):
        """Test waiting for training completion."""
        # Mock completed training result
        training_result = Mock(spec=TrainingResult)
        training_result.status = TrainingStatus.COMPLETED
        self.training_service.get_training_result.return_value = training_result
        
        result = await self.use_case._wait_for_training_completion("training_123")
        assert result == training_result
        
        # Mock failed training result
        training_result.status = TrainingStatus.FAILED
        result = await self.use_case._wait_for_training_completion("training_123")
        assert result == training_result
    
    def test_calculate_improvement(self):
        """Test calculating performance improvement."""
        baseline = {"accuracy": 0.8, "precision": 0.75, "recall": 0.85}
        new_metrics = {"accuracy": 0.85, "precision": 0.8, "recall": 0.9}
        
        improvement = self.use_case._calculate_improvement(baseline, new_metrics)
        
        # Expected: ((0.85-0.8)/0.8 + (0.8-0.75)/0.75 + (0.9-0.85)/0.85) / 3
        expected = ((0.85-0.8)/0.8 + (0.8-0.75)/0.75 + (0.9-0.85)/0.85) / 3
        assert abs(improvement - expected) < 0.001
    
    def test_calculate_improvement_no_common_metrics(self):
        """Test calculating improvement with no common metrics."""
        baseline = {"accuracy": 0.8}
        new_metrics = {"precision": 0.75}
        
        improvement = self.use_case._calculate_improvement(baseline, new_metrics)
        assert improvement == 0.0
    
    @pytest.mark.asyncio
    async def test_send_notification(self):
        """Test sending notifications."""
        channels = ["email", "slack"]
        
        await self.use_case._send_notification(
            "retraining_completed",
            "Retraining completed successfully",
            channels
        )
        
        # Verify notification sent to all channels
        assert self.notification_service.send_notification.call_count == 2
        
        # Test without notification service
        use_case = AutomatedRetrainingUseCase(
            detector_repository=self.detector_repository,
            training_service=self.training_service,
            drift_monitoring_use_case=self.drift_monitoring_use_case,
            train_detector_use_case=self.train_detector_use_case,
            notification_service=None
        )
        
        # Should not raise exception
        await use_case._send_notification(
            "retraining_completed",
            "Retraining completed successfully",
            channels
        )
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self):
        """Test stopping monitoring tasks."""
        # Create mock tasks
        task1 = Mock()
        task2 = Mock()
        task1.cancel = Mock()
        task2.cancel = Mock()
        
        # Add tasks to monitoring
        self.use_case.monitoring_tasks[self.detector_id] = task1
        self.use_case.monitoring_tasks[uuid4()] = task2
        
        await self.use_case.stop_monitoring()
        
        # Verify tasks were cancelled
        task1.cancel.assert_called_once()
        task2.cancel.assert_called_once()
        
        # Verify monitoring tasks cleared
        assert len(self.use_case.monitoring_tasks) == 0
