"""Tests for PerformanceDegradationMonitoringService."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from packages.data_science.application.services.performance_degradation_monitoring_service import (
    PerformanceDegradationMonitoringService,
)
from packages.data_science.domain.entities.model_performance_degradation import (
    ModelPerformanceDegradation,
    DegradationStatus,
    RecoveryAction,
)
from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask,
)
from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    PerformanceDegradationMetrics,
    DegradationMetricType,
    DegradationSeverity,
)
from packages.data_science.domain.services.performance_baseline_service import (
    PerformanceBaselineService,
)
from packages.data_science.domain.services.performance_history_service import (
    PerformanceHistoryService,
)
from packages.core.domain.entities.alert import (
    Alert,
    AlertType,
    AlertSeverity,
    AlertCondition,
)


class TestPerformanceDegradationMonitoringService:
    """Test suite for PerformanceDegradationMonitoringService."""
    
    @pytest.fixture
    def mock_degradation_repository(self):
        """Create mock degradation repository."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_metrics_repository(self):
        """Create mock metrics repository."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_baseline_service(self):
        """Create mock baseline service."""
        return MagicMock(spec=PerformanceBaselineService)
    
    @pytest.fixture
    def mock_alert_service(self):
        """Create mock alert service."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_notification_service(self):
        """Create mock notification service."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_history_service(self):
        """Create mock history service."""
        return AsyncMock(spec=PerformanceHistoryService)
    
    @pytest.fixture
    def monitoring_service(
        self, 
        mock_degradation_repository,
        mock_metrics_repository,
        mock_baseline_service,
        mock_alert_service,
        mock_notification_service,
        mock_history_service
    ):
        """Create monitoring service instance."""
        return PerformanceDegradationMonitoringService(
            degradation_repository=mock_degradation_repository,
            metrics_repository=mock_metrics_repository,
            baseline_service=mock_baseline_service,
            alert_service=mock_alert_service,
            notification_service=mock_notification_service,
            history_service=mock_history_service,
            monitoring_interval_minutes=1,  # Fast for testing
        )
    
    @pytest.fixture
    def sample_baseline_metrics(self):
        """Create sample baseline metrics."""
        return ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            roc_auc=0.95,
            prediction_time_seconds=0.05,
        )
    
    @pytest.fixture
    def sample_historical_metrics(self, sample_baseline_metrics):
        """Create sample historical metrics."""
        return [sample_baseline_metrics] * 10
    
    @pytest.fixture
    def sample_degradation_entity(self, sample_baseline_metrics):
        """Create sample degradation entity."""
        degradation_metrics = [
            PerformanceDegradationMetrics(
                metric_type=DegradationMetricType.ACCURACY_DROP,
                threshold_value=0.85,
                baseline_value=0.90,
            ),
        ]
        
        return ModelPerformanceDegradation(
            model_id="test-model-123",
            model_name="Test Model",
            model_version="1.0.0",
            task_type=ModelTask.BINARY_CLASSIFICATION,
            baseline_metrics=sample_baseline_metrics,
            degradation_metrics=degradation_metrics,
        )
    
    @pytest.mark.asyncio
    async def test_start_monitoring(
        self, 
        monitoring_service, 
        mock_degradation_repository,
        sample_degradation_entity
    ):
        """Test starting monitoring service."""
        mock_degradation_repository.get_all_active.return_value = [sample_degradation_entity]
        
        await monitoring_service.start_monitoring()
        
        assert monitoring_service._is_running is True
        mock_degradation_repository.get_all_active.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitoring_service):
        """Test stopping monitoring service."""
        # Start monitoring first
        monitoring_service._is_running = True
        monitoring_service._monitoring_tasks = {"test-model": MagicMock()}
        monitoring_service._monitoring_tasks["test-model"].cancel = MagicMock()
        
        await monitoring_service.stop_monitoring()
        
        assert monitoring_service._is_running is False
        assert len(monitoring_service._monitoring_tasks) == 0
        monitoring_service._monitoring_tasks["test-model"].cancel.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_setup_degradation_monitoring(
        self, 
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository,
        mock_baseline_service,
        sample_historical_metrics,
        sample_baseline_metrics
    ):
        """Test setting up degradation monitoring."""
        # Mock repository responses
        mock_degradation_repository.get_by_model_id.return_value = None
        mock_metrics_repository.get_historical_metrics.return_value = sample_historical_metrics
        mock_baseline_service.establish_baseline.return_value = sample_baseline_metrics
        mock_baseline_service._extract_metric_value.return_value = 0.90
        
        degradation_thresholds = {
            "accuracy_drop": 10.0,
            "prediction_time_increase": 20.0,
        }
        
        result = await monitoring_service.setup_degradation_monitoring(
            model_id="test-model-123",
            model_name="Test Model",
            model_version="1.0.0",
            task_type="binary_classification",
            degradation_thresholds=degradation_thresholds,
            auto_recovery_enabled=True,
            notification_recipients=["admin@example.com"],
        )
        
        assert result.model_id == "test-model-123"
        assert result.model_name == "Test Model"
        assert result.auto_recovery_enabled is True
        
        mock_degradation_repository.save.assert_called_once()
        mock_baseline_service.establish_baseline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_setup_degradation_monitoring_already_exists(
        self, 
        monitoring_service,
        mock_degradation_repository,
        sample_degradation_entity
    ):
        """Test setting up monitoring when it already exists."""
        mock_degradation_repository.get_by_model_id.return_value = sample_degradation_entity
        
        with pytest.raises(ValueError, match="Monitoring already exists"):
            await monitoring_service.setup_degradation_monitoring(
                model_id="test-model-123",
                model_name="Test Model",
                model_version="1.0.0",
                task_type="binary_classification",
                degradation_thresholds={"accuracy_drop": 10.0},
            )
    
    @pytest.mark.asyncio
    async def test_setup_degradation_monitoring_no_historical_metrics(
        self, 
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository
    ):
        """Test setting up monitoring with no historical metrics."""
        mock_degradation_repository.get_by_model_id.return_value = None
        mock_metrics_repository.get_historical_metrics.return_value = []
        
        with pytest.raises(ValueError, match="No historical metrics found"):
            await monitoring_service.setup_degradation_monitoring(
                model_id="test-model-123",
                model_name="Test Model",
                model_version="1.0.0",
                task_type="binary_classification",
                degradation_thresholds={"accuracy_drop": 10.0},
            )
    
    @pytest.mark.asyncio
    async def test_evaluate_model_performance(
        self, 
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository,
        mock_history_service,
        sample_degradation_entity
    ):
        """Test evaluating model performance."""
        # Mock current metrics (degraded)
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.80,  # Below threshold
            prediction_time_seconds=0.06,
        )
        
        mock_degradation_repository.get_by_model_id.return_value = sample_degradation_entity
        mock_metrics_repository.get_latest_metrics.return_value = current_metrics
        
        result = await monitoring_service.evaluate_model_performance("test-model-123")
        
        assert result["status"] == DegradationStatus.DEGRADED.value
        assert len(result["degradations"]) > 0
        assert result["should_alert"] is True
        
        mock_degradation_repository.update.assert_called_once()
        mock_history_service.record_degradation_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_evaluate_model_performance_no_monitoring(
        self, 
        monitoring_service,
        mock_degradation_repository
    ):
        """Test evaluating performance when no monitoring exists."""
        mock_degradation_repository.get_by_model_id.return_value = None
        
        with pytest.raises(ValueError, match="No degradation monitoring found"):
            await monitoring_service.evaluate_model_performance("test-model-123")
    
    @pytest.mark.asyncio
    async def test_evaluate_model_performance_no_metrics(
        self, 
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository,
        sample_degradation_entity
    ):
        """Test evaluating performance when no latest metrics exist."""
        mock_degradation_repository.get_by_model_id.return_value = sample_degradation_entity
        mock_metrics_repository.get_latest_metrics.return_value = None
        
        result = await monitoring_service.evaluate_model_performance("test-model-123")
        
        assert result["error"] == "No latest metrics found"
        assert result["model_id"] == "test-model-123"
    
    @pytest.mark.asyncio
    async def test_update_baseline(
        self, 
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository,
        mock_baseline_service,
        sample_degradation_entity,
        sample_historical_metrics,
        sample_baseline_metrics
    ):
        """Test updating baseline."""
        mock_degradation_repository.get_by_model_id.return_value = sample_degradation_entity
        mock_metrics_repository.get_historical_metrics.return_value = sample_historical_metrics
        mock_baseline_service.establish_baseline.return_value = sample_baseline_metrics
        
        await monitoring_service.update_baseline("test-model-123", "recent_average")
        
        mock_baseline_service.establish_baseline.assert_called_once()
        mock_degradation_repository.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enable_monitoring(
        self, 
        monitoring_service,
        mock_degradation_repository,
        sample_degradation_entity
    ):
        """Test enabling monitoring."""
        sample_degradation_entity.monitoring_enabled = False
        mock_degradation_repository.get_by_model_id.return_value = sample_degradation_entity
        
        await monitoring_service.enable_monitoring("test-model-123")
        
        assert sample_degradation_entity.monitoring_enabled is True
        mock_degradation_repository.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disable_monitoring(
        self, 
        monitoring_service,
        mock_degradation_repository,
        sample_degradation_entity
    ):
        """Test disabling monitoring."""
        sample_degradation_entity.monitoring_enabled = True
        mock_degradation_repository.get_by_model_id.return_value = sample_degradation_entity
        
        # Add a mock task
        mock_task = MagicMock()
        monitoring_service._monitoring_tasks["test-model-123"] = mock_task
        
        await monitoring_service.disable_monitoring("test-model-123")
        
        assert sample_degradation_entity.monitoring_enabled is False
        mock_degradation_repository.update.assert_called_once()
        mock_task.cancel.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trigger_recovery_action(
        self, 
        monitoring_service,
        mock_degradation_repository,
        mock_history_service,
        sample_degradation_entity
    ):
        """Test triggering recovery action."""
        sample_degradation_entity.auto_recovery_enabled = True
        mock_degradation_repository.get_by_model_id.return_value = sample_degradation_entity
        
        await monitoring_service.trigger_recovery_action(
            "test-model-123",
            RecoveryAction.ADJUST_THRESHOLD,
            "test_user",
            {"reason": "test"}
        )
        
        mock_degradation_repository.update.assert_called()
        mock_history_service.record_recovery_action.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_monitoring_status(
        self, 
        monitoring_service,
        mock_degradation_repository,
        sample_degradation_entity
    ):
        """Test getting monitoring status."""
        mock_degradation_repository.get_by_model_id.return_value = sample_degradation_entity
        
        status = await monitoring_service.get_monitoring_status("test-model-123")
        
        assert status["model_id"] == "test-model-123"
        assert status["model_name"] == "Test Model"
        assert status["status"] == DegradationStatus.HEALTHY.value
    
    @pytest.mark.asyncio
    async def test_get_monitoring_status_not_found(
        self, 
        monitoring_service,
        mock_degradation_repository
    ):
        """Test getting monitoring status when not found."""
        mock_degradation_repository.get_by_model_id.return_value = None
        
        status = await monitoring_service.get_monitoring_status("test-model-123")
        
        assert status["error"] == "No monitoring found"
        assert status["model_id"] == "test-model-123"
    
    @pytest.mark.asyncio
    async def test_list_all_monitoring(
        self, 
        monitoring_service,
        mock_degradation_repository,
        sample_degradation_entity
    ):
        """Test listing all monitoring."""
        mock_degradation_repository.get_all_active.return_value = [sample_degradation_entity]
        
        monitoring_list = await monitoring_service.list_all_monitoring()
        
        assert len(monitoring_list) == 1
        assert monitoring_list[0]["model_id"] == "test-model-123"
    
    @pytest.mark.asyncio
    async def test_get_performance_history(
        self, 
        monitoring_service,
        mock_history_service
    ):
        """Test getting performance history."""
        mock_timeline = [{"timestamp": "2023-01-01", "status": "healthy"}]
        mock_patterns = {"frequent_degradations": False}
        mock_stability = {"stability_score": 0.9}
        
        mock_history_service.get_degradation_timeline.return_value = mock_timeline
        mock_history_service.analyze_degradation_patterns.return_value = mock_patterns
        mock_history_service.get_performance_stability_score.return_value = mock_stability
        
        history = await monitoring_service.get_performance_history("test-model-123", 30)
        
        assert history["model_id"] == "test-model-123"
        assert history["timeline"] == mock_timeline
        assert history["patterns"] == mock_patterns
        assert history["stability"] == mock_stability
        assert history["analysis_period_days"] == 30
    
    @pytest.mark.asyncio
    async def test_get_performance_history_no_service(self, monitoring_service):
        """Test getting performance history when history service is not available."""
        monitoring_service.history_service = None
        
        history = await monitoring_service.get_performance_history("test-model-123", 30)
        
        assert history["error"] == "History service not available"
    
    @pytest.mark.asyncio
    async def test_get_stability_report(
        self, 
        monitoring_service,
        mock_history_service
    ):
        """Test getting stability report."""
        mock_stability = {"stability_score": 0.85, "stability_grade": "B"}
        mock_history_service.get_performance_stability_score.return_value = mock_stability
        
        report = await monitoring_service.get_stability_report("test-model-123", 30)
        
        assert report == mock_stability
        mock_history_service.get_performance_stability_score.assert_called_once_with("test-model-123", 30)
    
    @pytest.mark.asyncio
    async def test_compare_models(
        self, 
        monitoring_service,
        mock_history_service
    ):
        """Test comparing models."""
        mock_comparison = {
            "model_analyses": {"model-1": {}, "model-2": {}},
            "summary_stats": {"total_models_analyzed": 2},
        }
        mock_history_service.compare_model_performance_history.return_value = mock_comparison
        
        comparison = await monitoring_service.compare_models(["model-1", "model-2"], 30)
        
        assert comparison == mock_comparison
        mock_history_service.compare_model_performance_history.assert_called_once_with(["model-1", "model-2"], 30)
    
    @pytest.mark.asyncio
    async def test_cleanup_history(
        self, 
        monitoring_service,
        mock_history_service
    ):
        """Test cleaning up history."""
        mock_history_service.cleanup_old_history.return_value = 100
        
        cleaned_count = await monitoring_service.cleanup_history()
        
        assert cleaned_count == 100
        mock_history_service.cleanup_old_history.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_history_no_service(self, monitoring_service):
        """Test cleaning up history when history service is not available."""
        monitoring_service.history_service = None
        
        cleaned_count = await monitoring_service.cleanup_history()
        
        assert cleaned_count == 0
    
    @pytest.mark.asyncio
    async def test_handle_degradation_alert(
        self, 
        monitoring_service,
        mock_alert_service,
        mock_notification_service,
        mock_degradation_repository,
        sample_degradation_entity
    ):
        """Test handling degradation alert."""
        # Set up entity to trigger alert
        sample_degradation_entity.status = DegradationStatus.DEGRADED
        sample_degradation_entity.notification_settings = {
            "enabled": True,
            "recipients": ["admin@example.com"],
        }
        
        evaluation_result = {
            "status": "degraded",
            "degradations": [
                {
                    "metric_type": "accuracy_drop",
                    "severity": "moderate",
                    "degradation_percentage": 15.0,
                }
            ],
            "should_alert": True,
        }
        
        await monitoring_service._handle_degradation_alert(sample_degradation_entity, evaluation_result)
        
        mock_alert_service.create_alert.assert_called_once()
        mock_notification_service.send_degradation_notification.assert_called_once()
        mock_degradation_repository.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_auto_recovery(
        self, 
        monitoring_service,
        mock_degradation_repository,
        sample_degradation_entity
    ):
        """Test handling auto recovery."""
        sample_degradation_entity.auto_recovery_enabled = True
        
        evaluation_result = {
            "recovery_actions_recommended": ["adjust_threshold", "data_quality_check"],
        }
        
        await monitoring_service._handle_auto_recovery(sample_degradation_entity, evaluation_result)
        
        # Should trigger recovery actions
        assert len(sample_degradation_entity.recovery_actions) > 0
        mock_degradation_repository.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_adjust_thresholds(
        self, 
        monitoring_service,
        mock_metrics_repository,
        mock_baseline_service,
        sample_degradation_entity,
        sample_historical_metrics
    ):
        """Test adjusting thresholds."""
        mock_metrics_repository.get_historical_metrics.return_value = sample_historical_metrics
        mock_baseline_service.suggest_degradation_thresholds.return_value = {
            DegradationMetricType.ACCURACY_DROP: 10.0,
        }
        
        await monitoring_service._adjust_thresholds(sample_degradation_entity, {})
        
        mock_baseline_service.suggest_degradation_thresholds.assert_called_once()
        
        # Check that thresholds were adjusted
        accuracy_metric = next(
            m for m in sample_degradation_entity.degradation_metrics 
            if m.metric_type == DegradationMetricType.ACCURACY_DROP
        )
        # Should be adjusted to be more lenient
        assert accuracy_metric.threshold_value != 0.85
    
    @pytest.mark.asyncio
    async def test_adjust_thresholds_insufficient_data(
        self, 
        monitoring_service,
        mock_metrics_repository,
        sample_degradation_entity
    ):
        """Test adjusting thresholds with insufficient data."""
        mock_metrics_repository.get_historical_metrics.return_value = []
        
        # Should not raise error, just return without adjustment
        await monitoring_service._adjust_thresholds(sample_degradation_entity, {})
        
        # Thresholds should remain unchanged
        accuracy_metric = next(
            m for m in sample_degradation_entity.degradation_metrics 
            if m.metric_type == DegradationMetricType.ACCURACY_DROP
        )
        assert accuracy_metric.threshold_value == 0.85
    
    @pytest.mark.asyncio
    async def test_trigger_data_quality_check(
        self, 
        monitoring_service,
        sample_degradation_entity
    ):
        """Test triggering data quality check."""
        # Should not raise error
        await monitoring_service._trigger_data_quality_check(sample_degradation_entity, {})
    
    @pytest.mark.asyncio
    async def test_trigger_infrastructure_scaling(
        self, 
        monitoring_service,
        sample_degradation_entity
    ):
        """Test triggering infrastructure scaling."""
        # Should not raise error
        await monitoring_service._trigger_infrastructure_scaling(sample_degradation_entity, {})
    
    @pytest.mark.asyncio
    async def test_execute_recovery_action_success(
        self, 
        monitoring_service,
        mock_metrics_repository,
        sample_degradation_entity,
        sample_historical_metrics
    ):
        """Test executing recovery action successfully."""
        mock_metrics_repository.get_historical_metrics.return_value = sample_historical_metrics
        
        await monitoring_service._execute_recovery_action(
            sample_degradation_entity,
            RecoveryAction.ADJUST_THRESHOLD,
            {}
        )
        
        # Should mark action as completed successfully
        action = sample_degradation_entity.recovery_actions[-1]
        assert action["success"] is True
        assert "completed successfully" in action["notes"]
    
    @pytest.mark.asyncio
    async def test_execute_recovery_action_failure(
        self, 
        monitoring_service,
        sample_degradation_entity
    ):
        """Test executing recovery action with failure."""
        with patch.object(monitoring_service, '_adjust_thresholds', side_effect=Exception("Test error")):
            with pytest.raises(Exception):
                await monitoring_service._execute_recovery_action(
                    sample_degradation_entity,
                    RecoveryAction.ADJUST_THRESHOLD,
                    {}
                )
            
            # Should mark action as failed
            action = sample_degradation_entity.recovery_actions[-1]
            assert action["success"] is False
            assert "Test error" in action["notes"]
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(
        self, 
        monitoring_service,
        mock_degradation_repository,
        sample_degradation_entity
    ):
        """Test main monitoring loop."""
        # Set up entity that needs evaluation
        sample_degradation_entity.last_evaluation_at = datetime.utcnow() - timedelta(hours=1)
        mock_degradation_repository.get_all_active.return_value = [sample_degradation_entity]
        
        # Mock the evaluation method
        monitoring_service.evaluate_model_performance = AsyncMock()
        
        # Start monitoring loop
        monitoring_service._is_running = True
        
        # Create task and let it run briefly
        task = asyncio.create_task(monitoring_service._monitoring_loop())
        await asyncio.sleep(0.1)  # Let loop run once
        
        # Stop monitoring
        monitoring_service._is_running = False
        
        # Wait for task completion
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
        
        # Should have called evaluation
        monitoring_service.evaluate_model_performance.assert_called()
    
    @pytest.mark.asyncio
    async def test_model_monitoring_task(
        self, 
        monitoring_service,
        mock_degradation_repository,
        sample_degradation_entity
    ):
        """Test individual model monitoring task."""
        # Set up entity that needs evaluation
        sample_degradation_entity.last_evaluation_at = datetime.utcnow() - timedelta(hours=1)
        mock_degradation_repository.get_by_model_id.return_value = sample_degradation_entity
        
        # Mock the evaluation method
        monitoring_service.evaluate_model_performance = AsyncMock()
        
        # Start monitoring task
        monitoring_service._is_running = True
        
        # Create task and let it run briefly
        task = asyncio.create_task(monitoring_service._model_monitoring_task(sample_degradation_entity))
        await asyncio.sleep(0.1)  # Let loop run once
        
        # Stop monitoring
        monitoring_service._is_running = False
        
        # Wait for task completion
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
        
        # Should have called evaluation
        monitoring_service.evaluate_model_performance.assert_called()
    
    @pytest.mark.asyncio
    async def test_start_model_monitoring(
        self, 
        monitoring_service,
        sample_degradation_entity
    ):
        """Test starting model monitoring."""
        await monitoring_service._start_model_monitoring(sample_degradation_entity)
        
        # Should have created a monitoring task
        assert "test-model-123" in monitoring_service._monitoring_tasks
        
        # Clean up
        monitoring_service._monitoring_tasks["test-model-123"].cancel()
    
    @pytest.mark.asyncio
    async def test_start_model_monitoring_cancels_existing(
        self, 
        monitoring_service,
        sample_degradation_entity
    ):
        """Test starting model monitoring cancels existing task."""
        # Add existing task
        mock_existing_task = MagicMock()
        monitoring_service._monitoring_tasks["test-model-123"] = mock_existing_task
        
        await monitoring_service._start_model_monitoring(sample_degradation_entity)
        
        # Should cancel existing task
        mock_existing_task.cancel.assert_called_once()
        
        # Should have new task
        assert "test-model-123" in monitoring_service._monitoring_tasks
        
        # Clean up
        monitoring_service._monitoring_tasks["test-model-123"].cancel()