"""Integration tests for performance degradation detection system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from packages.data_science.application.services.performance_degradation_monitoring_service import (
    PerformanceDegradationMonitoringService,
)
from packages.data_science.domain.services.performance_baseline_service import (
    PerformanceBaselineService,
)
from packages.data_science.domain.services.performance_history_service import (
    PerformanceHistoryService,
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


class TestPerformanceDegradationIntegration:
    """Integration test suite for performance degradation detection system."""
    
    @pytest.fixture
    def baseline_service(self):
        """Create real baseline service."""
        return PerformanceBaselineService(confidence_level=0.95, min_samples=5)
    
    @pytest.fixture
    def mock_history_repository(self):
        """Create mock history repository."""
        repo = AsyncMock()
        repo.get_degradation_history.return_value = []
        repo.get_degradation_summary.return_value = {
            "degradation_events": 0,
            "recovery_events": 0,
            "critical_events": 0,
        }
        repo.get_degradation_trends.return_value = []
        repo.get_recovery_history.return_value = []
        repo.cleanup_old_history.return_value = 0
        return repo
    
    @pytest.fixture
    def history_service(self, mock_history_repository):
        """Create real history service with mocked repository."""
        return PerformanceHistoryService(
            history_repository=mock_history_repository,
            default_retention_days=90
        )
    
    @pytest.fixture
    def mock_degradation_repository(self):
        """Create mock degradation repository."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_metrics_repository(self):
        """Create mock metrics repository."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_alert_service(self):
        """Create mock alert service."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_notification_service(self):
        """Create mock notification service."""
        return AsyncMock()
    
    @pytest.fixture
    def monitoring_service(
        self,
        mock_degradation_repository,
        mock_metrics_repository,
        baseline_service,
        mock_alert_service,
        mock_notification_service,
        history_service
    ):
        """Create monitoring service with real baseline and history services."""
        return PerformanceDegradationMonitoringService(
            degradation_repository=mock_degradation_repository,
            metrics_repository=mock_metrics_repository,
            baseline_service=baseline_service,
            alert_service=mock_alert_service,
            notification_service=mock_notification_service,
            history_service=history_service,
            monitoring_interval_minutes=1,
        )
    
    @pytest.fixture
    def historical_metrics(self):
        """Create historical metrics for baseline establishment."""
        base_date = datetime.utcnow() - timedelta(days=30)
        metrics = []
        
        for i in range(20):
            metrics.append(ModelPerformanceMetrics(
                task_type=ModelTask.BINARY_CLASSIFICATION,
                sample_size=1000,
                accuracy=0.90 + (i * 0.001),  # Slightly improving trend
                precision=0.88 + (i * 0.001),
                recall=0.92 - (i * 0.0005),
                f1_score=0.90 + (i * 0.0005),
                roc_auc=0.95 + (i * 0.0002),
                prediction_time_seconds=0.05 + (i * 0.0005),
                evaluation_date=base_date + timedelta(days=i),
            ))
        
        return metrics
    
    @pytest.mark.asyncio
    async def test_complete_degradation_detection_flow(
        self,
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository,
        mock_alert_service,
        mock_notification_service,
        historical_metrics
    ):
        """Test complete flow from setup to degradation detection and alerting."""
        # Step 1: Setup monitoring
        mock_degradation_repository.get_by_model_id.return_value = None
        mock_metrics_repository.get_historical_metrics.return_value = historical_metrics
        
        degradation_entity = await monitoring_service.setup_degradation_monitoring(
            model_id="test-model-123",
            model_name="Test Model",
            model_version="1.0.0",
            task_type="binary_classification",
            degradation_thresholds={
                "accuracy_drop": 5.0,
                "precision_drop": 5.0,
                "prediction_time_increase": 20.0,
            },
            auto_recovery_enabled=True,
            notification_recipients=["admin@example.com"],
        )
        
        # Verify setup
        assert degradation_entity.model_id == "test-model-123"
        assert degradation_entity.monitoring_enabled is True
        assert degradation_entity.auto_recovery_enabled is True
        assert len(degradation_entity.degradation_metrics) == 3
        mock_degradation_repository.save.assert_called_once()
        
        # Step 2: Evaluate with healthy metrics
        mock_degradation_repository.get_by_model_id.return_value = degradation_entity
        
        healthy_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.905,  # Above threshold
            precision=0.885,
            recall=0.915,
            f1_score=0.900,
            roc_auc=0.955,
            prediction_time_seconds=0.055,
        )
        
        mock_metrics_repository.get_latest_metrics.return_value = healthy_metrics
        
        healthy_result = await monitoring_service.evaluate_model_performance("test-model-123")
        
        # Should be healthy
        assert healthy_result["status"] == DegradationStatus.HEALTHY.value
        assert len(healthy_result["degradations"]) == 0
        assert healthy_result["should_alert"] is False
        
        # Step 3: Evaluate with degraded metrics
        degraded_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.82,  # Below threshold (should trigger)
            precision=0.80,  # Below threshold (should trigger)
            recall=0.85,
            f1_score=0.825,
            roc_auc=0.88,
            prediction_time_seconds=0.075,  # Above threshold (should trigger)
        )
        
        mock_metrics_repository.get_latest_metrics.return_value = degraded_metrics
        
        degraded_result = await monitoring_service.evaluate_model_performance("test-model-123")
        
        # Should be degraded
        assert degraded_result["status"] == DegradationStatus.DEGRADED.value
        assert len(degraded_result["degradations"]) == 3  # All three metrics degraded
        assert degraded_result["should_alert"] is True
        
        # Verify alert was created
        mock_alert_service.create_alert.assert_called_once()
        mock_notification_service.send_degradation_notification.assert_called_once()
        
        # Step 4: Verify recovery actions were triggered
        assert len(degradation_entity.recovery_actions) > 0
        
        # Step 5: Simulate recovery
        recovery_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.895,  # Back above threshold
            precision=0.875,
            recall=0.905,
            f1_score=0.890,
            roc_auc=0.945,
            prediction_time_seconds=0.058,
        )
        
        mock_metrics_repository.get_latest_metrics.return_value = recovery_metrics
        
        # First recovery evaluation
        recovery_result1 = await monitoring_service.evaluate_model_performance("test-model-123")
        assert recovery_result1["status"] == DegradationStatus.RECOVERING.value
        
        # Second recovery evaluation
        recovery_result2 = await monitoring_service.evaluate_model_performance("test-model-123")
        assert recovery_result2["status"] == DegradationStatus.RECOVERING.value
        
        # Third recovery evaluation - should be fully recovered
        recovery_result3 = await monitoring_service.evaluate_model_performance("test-model-123")
        assert recovery_result3["status"] == DegradationStatus.HEALTHY.value
        
        # Verify repository updates
        assert mock_degradation_repository.update.call_count >= 6  # Multiple updates
    
    @pytest.mark.asyncio
    async def test_baseline_recalculation_integration(
        self,
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository,
        historical_metrics
    ):
        """Test baseline recalculation integration."""
        # Setup existing degradation entity
        baseline_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            roc_auc=0.95,
            prediction_time_seconds=0.05,
        )
        
        degradation_metrics = [
            PerformanceDegradationMetrics(
                metric_type=DegradationMetricType.ACCURACY_DROP,
                threshold_value=0.85,
                baseline_value=0.90,
            ),
        ]
        
        degradation_entity = ModelPerformanceDegradation(
            model_id="test-model-123",
            model_name="Test Model",
            model_version="1.0.0",
            task_type=ModelTask.BINARY_CLASSIFICATION,
            baseline_metrics=baseline_metrics,
            degradation_metrics=degradation_metrics,
        )
        
        mock_degradation_repository.get_by_model_id.return_value = degradation_entity
        mock_metrics_repository.get_historical_metrics.return_value = historical_metrics
        
        # Update baseline
        await monitoring_service.update_baseline("test-model-123", "recent_average")
        
        # Verify baseline was updated
        updated_baseline = degradation_entity.baseline_metrics
        assert updated_baseline is not None
        assert updated_baseline.accuracy > 0.90  # Should be higher due to improving trend
        
        # Verify degradation metrics were updated
        accuracy_metric = next(
            m for m in degradation_entity.degradation_metrics 
            if m.metric_type == DegradationMetricType.ACCURACY_DROP
        )
        assert accuracy_metric.baseline_value == updated_baseline.accuracy
        
        mock_degradation_repository.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_history_tracking_integration(
        self,
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository,
        mock_history_repository,
        historical_metrics
    ):
        """Test history tracking integration."""
        # Setup monitoring
        mock_degradation_repository.get_by_model_id.return_value = None
        mock_metrics_repository.get_historical_metrics.return_value = historical_metrics
        
        degradation_entity = await monitoring_service.setup_degradation_monitoring(
            model_id="test-model-123",
            model_name="Test Model",
            model_version="1.0.0",
            task_type="binary_classification",
            degradation_thresholds={"accuracy_drop": 5.0},
        )
        
        # Simulate degradation event
        mock_degradation_repository.get_by_model_id.return_value = degradation_entity
        
        degraded_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.82,  # Below threshold
        )
        
        mock_metrics_repository.get_latest_metrics.return_value = degraded_metrics
        
        # Evaluate - should trigger history recording
        result = await monitoring_service.evaluate_model_performance("test-model-123")
        
        # Verify history was recorded
        mock_history_repository.save_degradation_event.assert_called()
        
        # Verify call arguments
        call_args = mock_history_repository.save_degradation_event.call_args
        assert call_args[0][0] == "test-model-123"  # model_id
        assert call_args[0][1]["status"] == DegradationStatus.DEGRADED.value
        assert call_args[0][2] == degraded_metrics  # metrics
        
        # Simulate recovery action
        await monitoring_service.trigger_recovery_action(
            "test-model-123",
            RecoveryAction.RETRAIN_MODEL,
            "test_user",
            {"reason": "performance degradation"}
        )
        
        # Verify recovery action was recorded
        assert mock_history_repository.save_degradation_event.call_count == 2
    
    @pytest.mark.asyncio
    async def test_performance_stability_analysis_integration(
        self,
        monitoring_service,
        mock_history_repository
    ):
        """Test performance stability analysis integration."""
        # Mock history data showing degradation pattern
        mock_history_data = [
            {"degradations": []},  # healthy
            {"degradations": [{"severity": "minor"}]},  # degraded
            {"degradations": []},  # healthy
            {"degradations": [{"severity": "major"}]},  # degraded
            {"degradations": [{"severity": "critical"}], "overall_severity": "critical"},  # critical
            {"degradations": []},  # healthy
        ]
        
        mock_history_repository.get_degradation_history.return_value = mock_history_data
        
        # Get stability report
        stability_report = await monitoring_service.get_stability_report("test-model-123", 30)
        
        # Verify stability analysis
        assert stability_report["stability_score"] < 1.0  # Should be less than perfect
        assert stability_report["metrics"]["total_evaluations"] == 6
        assert stability_report["metrics"]["degraded_evaluations"] == 3
        assert stability_report["metrics"]["critical_evaluations"] == 1
        assert stability_report["metrics"]["degradation_rate"] == 0.5
        
        # Verify grading
        assert stability_report["stability_grade"] in ["A", "B", "C", "D", "F"]
    
    @pytest.mark.asyncio
    async def test_model_comparison_integration(
        self,
        monitoring_service,
        mock_history_repository
    ):
        """Test model comparison integration."""
        # Mock different stability data for different models
        def mock_get_history(model_id, start_date=None):
            if model_id == "model-1":
                return [{"degradations": []} for _ in range(10)]  # Very stable
            elif model_id == "model-2":
                return [{"degradations": []} if i % 2 == 0 else {"degradations": [{"severity": "minor"}]} 
                       for i in range(10)]  # Moderately stable
            else:
                return [{"degradations": [{"severity": "major"}]} for _ in range(10)]  # Unstable
        
        mock_history_repository.get_degradation_history.side_effect = mock_get_history
        mock_history_repository.get_degradation_summary.return_value = {
            "degradation_events": 5,
            "recovery_events": 2,
            "critical_events": 1,
        }
        
        # Compare models
        comparison = await monitoring_service.compare_models(
            ["model-1", "model-2", "model-3"], 30
        )
        
        # Verify comparison results
        assert comparison["summary_stats"]["total_models_analyzed"] == 3
        assert comparison["summary_stats"]["valid_models"] == 3
        assert len(comparison["model_analyses"]) == 3
        assert len(comparison["ranked_models"]) == 3
        
        # Verify ranking (model-1 should be best)
        assert comparison["ranked_models"][0] == "model-1"
        assert comparison["ranked_models"][-1] == "model-3"
        
        # Verify insights
        assert len(comparison["comparison_insights"]) > 0
    
    @pytest.mark.asyncio
    async def test_auto_recovery_integration(
        self,
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository,
        historical_metrics
    ):
        """Test auto recovery integration."""
        # Setup monitoring with auto recovery enabled
        mock_degradation_repository.get_by_model_id.return_value = None
        mock_metrics_repository.get_historical_metrics.return_value = historical_metrics
        
        degradation_entity = await monitoring_service.setup_degradation_monitoring(
            model_id="test-model-123",
            model_name="Test Model",
            model_version="1.0.0",
            task_type="binary_classification",
            degradation_thresholds={"accuracy_drop": 5.0},
            auto_recovery_enabled=True,
        )
        
        # Trigger critical degradation
        mock_degradation_repository.get_by_model_id.return_value = degradation_entity
        
        critical_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.50,  # Critically low
        )
        
        mock_metrics_repository.get_latest_metrics.return_value = critical_metrics
        
        # Evaluate - should trigger auto recovery
        result = await monitoring_service.evaluate_model_performance("test-model-123")
        
        # Verify critical status
        assert result["status"] == DegradationStatus.CRITICAL.value
        assert result["overall_severity"] == DegradationSeverity.CRITICAL.value
        
        # Verify recovery actions were triggered
        assert len(degradation_entity.recovery_actions) > 0
        
        # Verify recovery action types
        recovery_actions = [action["action"] for action in degradation_entity.recovery_actions]
        assert any(action in recovery_actions for action in [
            RecoveryAction.ADJUST_THRESHOLD.value,
            RecoveryAction.DATA_QUALITY_CHECK.value,
            RecoveryAction.INFRASTRUCTURE_SCALING.value,
        ])
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle_integration(
        self,
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository,
        historical_metrics
    ):
        """Test complete monitoring lifecycle integration."""
        # Step 1: Setup monitoring
        mock_degradation_repository.get_by_model_id.return_value = None
        mock_metrics_repository.get_historical_metrics.return_value = historical_metrics
        
        degradation_entity = await monitoring_service.setup_degradation_monitoring(
            model_id="test-model-123",
            model_name="Test Model",
            model_version="1.0.0",
            task_type="binary_classification",
            degradation_thresholds={"accuracy_drop": 5.0},
        )
        
        # Step 2: Start monitoring
        mock_degradation_repository.get_all_active.return_value = [degradation_entity]
        await monitoring_service.start_monitoring()
        
        assert monitoring_service._is_running is True
        
        # Step 3: Get monitoring status
        mock_degradation_repository.get_by_model_id.return_value = degradation_entity
        status = await monitoring_service.get_monitoring_status("test-model-123")
        
        assert status["model_id"] == "test-model-123"
        assert status["status"] == DegradationStatus.HEALTHY.value
        assert status["monitoring_enabled"] is True
        
        # Step 4: List all monitoring
        monitoring_list = await monitoring_service.list_all_monitoring()
        assert len(monitoring_list) == 1
        assert monitoring_list[0]["model_id"] == "test-model-123"
        
        # Step 5: Disable monitoring
        await monitoring_service.disable_monitoring("test-model-123")
        assert degradation_entity.monitoring_enabled is False
        
        # Step 6: Re-enable monitoring
        await monitoring_service.enable_monitoring("test-model-123")
        assert degradation_entity.monitoring_enabled is True
        
        # Step 7: Stop monitoring
        await monitoring_service.stop_monitoring()
        assert monitoring_service._is_running is False
    
    @pytest.mark.asyncio
    async def test_threshold_adjustment_integration(
        self,
        monitoring_service,
        mock_degradation_repository,
        mock_metrics_repository,
        historical_metrics
    ):
        """Test threshold adjustment integration."""
        # Setup monitoring
        mock_degradation_repository.get_by_model_id.return_value = None
        mock_metrics_repository.get_historical_metrics.return_value = historical_metrics
        
        degradation_entity = await monitoring_service.setup_degradation_monitoring(
            model_id="test-model-123",
            model_name="Test Model",
            model_version="1.0.0",
            task_type="binary_classification",
            degradation_thresholds={"accuracy_drop": 5.0},
            auto_recovery_enabled=True,
        )
        
        # Store original threshold
        original_threshold = degradation_entity.degradation_metrics[0].threshold_value
        
        # Trigger threshold adjustment
        mock_degradation_repository.get_by_model_id.return_value = degradation_entity
        mock_metrics_repository.get_historical_metrics.return_value = historical_metrics
        
        await monitoring_service.trigger_recovery_action(
            "test-model-123",
            RecoveryAction.ADJUST_THRESHOLD,
            "test_user",
            {"reason": "frequent false positives"}
        )
        
        # Verify threshold was adjusted
        adjusted_threshold = degradation_entity.degradation_metrics[0].threshold_value
        assert adjusted_threshold != original_threshold
        
        # Verify recovery action was recorded
        assert len(degradation_entity.recovery_actions) > 0
        assert degradation_entity.recovery_actions[0]["action"] == RecoveryAction.ADJUST_THRESHOLD.value
        assert degradation_entity.recovery_actions[0]["success"] is True