"""Tests for ModelPerformanceDegradation domain entity."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from packages.data_science.domain.entities.model_performance_degradation import (
    ModelPerformanceDegradation,
    DegradationStatus,
    RecoveryAction,
)
from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    PerformanceDegradationMetrics,
    DegradationMetricType,
    DegradationSeverity,
)
from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask,
)


class TestModelPerformanceDegradation:
    """Test suite for ModelPerformanceDegradation entity."""
    
    @pytest.fixture
    def baseline_metrics(self):
        """Create baseline performance metrics."""
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
    def degradation_metrics(self):
        """Create degradation metrics list."""
        return [
            PerformanceDegradationMetrics(
                metric_type=DegradationMetricType.ACCURACY_DROP,
                threshold_value=0.85,
                baseline_value=0.90,
            ),
            PerformanceDegradationMetrics(
                metric_type=DegradationMetricType.PREDICTION_TIME_INCREASE,
                threshold_value=0.075,
                baseline_value=0.05,
            ),
        ]
    
    @pytest.fixture
    def degradation_entity(self, baseline_metrics, degradation_metrics):
        """Create a ModelPerformanceDegradation entity."""
        return ModelPerformanceDegradation(
            model_id="test-model-123",
            model_name="Test Model",
            model_version="1.0.0",
            task_type=ModelTask.BINARY_CLASSIFICATION,
            baseline_metrics=baseline_metrics,
            degradation_metrics=degradation_metrics,
        )
    
    def test_create_degradation_entity(self, degradation_entity):
        """Test creating a degradation entity."""
        assert degradation_entity.model_id == "test-model-123"
        assert degradation_entity.model_name == "Test Model"
        assert degradation_entity.model_version == "1.0.0"
        assert degradation_entity.task_type == ModelTask.BINARY_CLASSIFICATION
        assert degradation_entity.status == DegradationStatus.HEALTHY
        assert degradation_entity.monitoring_enabled is True
        assert degradation_entity.auto_recovery_enabled is False
        assert degradation_entity.consecutive_healthy_evaluations == 0
    
    def test_add_degradation_metric(self, degradation_entity):
        """Test adding a degradation metric."""
        new_metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.PRECISION_DROP,
            threshold_value=0.80,
            baseline_value=0.88,
        )
        
        initial_count = len(degradation_entity.degradation_metrics)
        degradation_entity.add_degradation_metric(new_metric)
        
        assert len(degradation_entity.degradation_metrics) == initial_count + 1
        assert any(m.metric_type == DegradationMetricType.PRECISION_DROP 
                  for m in degradation_entity.degradation_metrics)
    
    def test_add_duplicate_degradation_metric(self, degradation_entity):
        """Test adding a duplicate degradation metric replaces existing."""
        new_metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.80,  # Different threshold
            baseline_value=0.90,
        )
        
        initial_count = len(degradation_entity.degradation_metrics)
        degradation_entity.add_degradation_metric(new_metric)
        
        # Should replace, not add
        assert len(degradation_entity.degradation_metrics) == initial_count
        
        # Should have new threshold value
        accuracy_metric = next(m for m in degradation_entity.degradation_metrics 
                             if m.metric_type == DegradationMetricType.ACCURACY_DROP)
        assert accuracy_metric.threshold_value == 0.80
    
    def test_remove_degradation_metric(self, degradation_entity):
        """Test removing a degradation metric."""
        initial_count = len(degradation_entity.degradation_metrics)
        degradation_entity.remove_degradation_metric(DegradationMetricType.ACCURACY_DROP)
        
        assert len(degradation_entity.degradation_metrics) == initial_count - 1
        assert not any(m.metric_type == DegradationMetricType.ACCURACY_DROP 
                      for m in degradation_entity.degradation_metrics)
    
    def test_update_baseline_metrics(self, degradation_entity):
        """Test updating baseline metrics."""
        new_baseline = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1500,
            accuracy=0.92,
            precision=0.90,
            recall=0.94,
            f1_score=0.92,
            roc_auc=0.97,
            prediction_time_seconds=0.04,
        )
        
        degradation_entity.update_baseline_metrics(new_baseline)
        
        assert degradation_entity.baseline_metrics.accuracy == 0.92
        assert degradation_entity.baseline_metrics.prediction_time_seconds == 0.04
        
        # Check that degradation metrics were updated
        accuracy_metric = next(m for m in degradation_entity.degradation_metrics 
                             if m.metric_type == DegradationMetricType.ACCURACY_DROP)
        assert accuracy_metric.baseline_value == 0.92
    
    def test_update_baseline_metrics_incompatible_task_type(self, degradation_entity):
        """Test updating baseline metrics with incompatible task type."""
        incompatible_baseline = ModelPerformanceMetrics(
            task_type=ModelTask.REGRESSION,  # Different task type
            sample_size=1000,
            mse=0.1,
            rmse=0.316,
            mae=0.2,
            r2_score=0.8,
        )
        
        with pytest.raises(ValueError, match="task type must match"):
            degradation_entity.update_baseline_metrics(incompatible_baseline)
    
    def test_evaluate_performance_healthy(self, degradation_entity, baseline_metrics):
        """Test performance evaluation with healthy metrics."""
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.89,  # Above threshold (0.85)
            precision=0.87,
            recall=0.91,
            f1_score=0.89,
            roc_auc=0.94,
            prediction_time_seconds=0.06,  # Below threshold (0.075)
        )
        
        result = degradation_entity.evaluate_performance(current_metrics)
        
        assert result["status"] == DegradationStatus.HEALTHY.value
        assert len(result["degradations"]) == 0
        assert result["should_alert"] is False
        assert degradation_entity.status == DegradationStatus.HEALTHY
        assert degradation_entity.consecutive_healthy_evaluations == 1
    
    def test_evaluate_performance_degraded(self, degradation_entity):
        """Test performance evaluation with degraded metrics."""
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.80,  # Below threshold (0.85)
            precision=0.82,
            recall=0.83,
            f1_score=0.82,
            roc_auc=0.88,
            prediction_time_seconds=0.08,  # Above threshold (0.075)
        )
        
        result = degradation_entity.evaluate_performance(current_metrics)
        
        assert result["status"] == DegradationStatus.DEGRADED.value
        assert len(result["degradations"]) == 2  # Both accuracy and prediction time
        assert result["should_alert"] is True
        assert degradation_entity.status == DegradationStatus.DEGRADED
        assert degradation_entity.consecutive_healthy_evaluations == 0
        assert degradation_entity.degradation_detected_at is not None
    
    def test_evaluate_performance_critical(self, degradation_entity):
        """Test performance evaluation with critical degradation."""
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.40,  # Severely below threshold (critical degradation)
            precision=0.45,
            recall=0.50,
            f1_score=0.47,
            roc_auc=0.60,
            prediction_time_seconds=0.20,  # Very high latency
        )
        
        result = degradation_entity.evaluate_performance(current_metrics)
        
        assert result["status"] == DegradationStatus.CRITICAL.value
        assert result["overall_severity"] == DegradationSeverity.CRITICAL.value
        assert degradation_entity.status == DegradationStatus.CRITICAL
    
    def test_evaluate_performance_recovery(self, degradation_entity):
        """Test performance recovery detection."""
        # First, degrade the model
        degraded_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.80,
            prediction_time_seconds=0.08,
        )
        degradation_entity.evaluate_performance(degraded_metrics)
        assert degradation_entity.status == DegradationStatus.DEGRADED
        
        # Now recover
        healthy_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.89,
            prediction_time_seconds=0.06,
        )
        
        # First healthy evaluation - should be recovering
        result1 = degradation_entity.evaluate_performance(healthy_metrics)
        assert result1["status"] == DegradationStatus.RECOVERING.value
        assert degradation_entity.consecutive_healthy_evaluations == 1
        
        # Second healthy evaluation - still recovering
        result2 = degradation_entity.evaluate_performance(healthy_metrics)
        assert result2["status"] == DegradationStatus.RECOVERING.value
        assert degradation_entity.consecutive_healthy_evaluations == 2
        
        # Third healthy evaluation - fully recovered
        result3 = degradation_entity.evaluate_performance(healthy_metrics)
        assert result3["status"] == DegradationStatus.HEALTHY.value
        assert degradation_entity.consecutive_healthy_evaluations == 3
        assert degradation_entity.recovery_completed_at is not None
        assert degradation_entity.degradation_detected_at is None
    
    def test_evaluate_performance_monitoring_disabled(self, degradation_entity):
        """Test evaluation when monitoring is disabled."""
        degradation_entity.disable_monitoring()
        
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.50,  # Very poor performance
        )
        
        result = degradation_entity.evaluate_performance(current_metrics)
        
        assert result["status"] == "monitoring_disabled"
        assert len(result["degradations"]) == 0
    
    def test_evaluate_performance_no_baseline(self, degradation_entity):
        """Test evaluation without baseline metrics."""
        degradation_entity.baseline_metrics = None
        
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.80,
        )
        
        result = degradation_entity.evaluate_performance(current_metrics)
        
        assert result["status"] == "no_baseline"
        assert len(result["degradations"]) == 0
    
    def test_trigger_recovery_action(self, degradation_entity):
        """Test triggering recovery actions."""
        degradation_entity.auto_recovery_enabled = True
        
        degradation_entity.trigger_recovery_action(
            RecoveryAction.ADJUST_THRESHOLD,
            "test_user",
            {"reason": "high false positive rate"}
        )
        
        assert len(degradation_entity.recovery_actions) == 1
        action = degradation_entity.recovery_actions[0]
        assert action["action"] == RecoveryAction.ADJUST_THRESHOLD.value
        assert action["initiated_by"] == "test_user"
        assert action["status"] == "initiated"
        assert action["context"]["reason"] == "high false positive rate"
        assert degradation_entity.recovery_started_at is not None
    
    def test_trigger_recovery_action_auto_recovery_disabled(self, degradation_entity):
        """Test triggering recovery action when auto-recovery is disabled."""
        degradation_entity.auto_recovery_enabled = False
        
        with pytest.raises(ValueError, match="Auto recovery is disabled"):
            degradation_entity.trigger_recovery_action(
                RecoveryAction.ADJUST_THRESHOLD,
                "test_user"
            )
    
    def test_trigger_manual_recovery_action(self, degradation_entity):
        """Test triggering manual recovery action (always allowed)."""
        degradation_entity.auto_recovery_enabled = False
        
        # Manual intervention should always be allowed
        degradation_entity.trigger_recovery_action(
            RecoveryAction.MANUAL_INTERVENTION,
            "test_user"
        )
        
        assert len(degradation_entity.recovery_actions) == 1
    
    def test_complete_recovery_action(self, degradation_entity):
        """Test completing recovery actions."""
        degradation_entity.auto_recovery_enabled = True
        
        degradation_entity.trigger_recovery_action(
            RecoveryAction.ADJUST_THRESHOLD,
            "test_user"
        )
        
        degradation_entity.complete_recovery_action(
            RecoveryAction.ADJUST_THRESHOLD,
            success=True,
            notes="Threshold adjusted successfully"
        )
        
        action = degradation_entity.recovery_actions[0]
        assert action["status"] == "completed"
        assert action["success"] is True
        assert action["notes"] == "Threshold adjusted successfully"
        assert "completed_at" in action
    
    def test_complete_recovery_action_failure(self, degradation_entity):
        """Test completing recovery action with failure."""
        degradation_entity.auto_recovery_enabled = True
        
        degradation_entity.trigger_recovery_action(
            RecoveryAction.RETRAIN_MODEL,
            "system"
        )
        
        degradation_entity.complete_recovery_action(
            RecoveryAction.RETRAIN_MODEL,
            success=False,
            notes="Training failed due to insufficient data"
        )
        
        action = degradation_entity.recovery_actions[0]
        assert action["status"] == "failed"
        assert action["success"] is False
        assert action["notes"] == "Training failed due to insufficient data"
    
    def test_enable_disable_monitoring(self, degradation_entity):
        """Test enabling and disabling monitoring."""
        # Test disable
        degradation_entity.disable_monitoring()
        assert degradation_entity.monitoring_enabled is False
        assert degradation_entity.status == DegradationStatus.MONITORING_DISABLED
        
        # Test enable
        degradation_entity.enable_monitoring()
        assert degradation_entity.monitoring_enabled is True
        assert degradation_entity.status == DegradationStatus.HEALTHY
    
    def test_is_due_for_evaluation(self, degradation_entity):
        """Test evaluation timing logic."""
        # New entity should be due for evaluation
        assert degradation_entity.is_due_for_evaluation() is True
        
        # After evaluation, should not be due yet
        degradation_entity.last_evaluation_at = datetime.utcnow()
        assert degradation_entity.is_due_for_evaluation() is False
        
        # After interval passes, should be due again
        degradation_entity.last_evaluation_at = datetime.utcnow() - timedelta(
            minutes=degradation_entity.evaluation_interval_minutes + 1
        )
        assert degradation_entity.is_due_for_evaluation() is True
        
        # Disabled monitoring should never be due
        degradation_entity.disable_monitoring()
        assert degradation_entity.is_due_for_evaluation() is False
    
    def test_should_send_alert(self, degradation_entity):
        """Test alert sending logic."""
        # Healthy status should not alert
        assert degradation_entity.should_send_alert() is False
        
        # Degrade the model first
        degraded_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.80,
        )
        degradation_entity.evaluate_performance(degraded_metrics)
        
        # Should alert after degradation
        assert degradation_entity.should_send_alert() is True
        
        # Mark alert as sent
        degradation_entity.mark_alert_sent()
        
        # Should not alert again immediately
        assert degradation_entity.should_send_alert() is False
        
        # Should alert again after cooldown period
        degradation_entity.alert_sent_at = datetime.utcnow() - timedelta(hours=2)
        assert degradation_entity.should_send_alert() is True
        
        # Disabled monitoring should not alert
        degradation_entity.disable_monitoring()
        assert degradation_entity.should_send_alert() is False
    
    def test_get_current_degradation_summary(self, degradation_entity):
        """Test degradation summary generation."""
        # Degrade the model
        degraded_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.80,
            prediction_time_seconds=0.08,
        )
        degradation_entity.evaluate_performance(degraded_metrics)
        
        summary = degradation_entity.get_current_degradation_summary()
        
        assert summary["model_id"] == "test-model-123"
        assert summary["model_name"] == "Test Model"
        assert summary["status"] == DegradationStatus.DEGRADED.value
        assert summary["degradation_count"] == 2
        assert "accuracy_drop" in summary["degraded_metrics"]
        assert "prediction_time_increase" in summary["degraded_metrics"]
        assert summary["should_alert"] is True
    
    def test_get_performance_history(self, degradation_entity):
        """Test performance history retrieval."""
        # Add some detection history
        degradation_entity.detection_history = [
            {
                "timestamp": "2023-01-01T00:00:00",
                "status": "degraded",
                "degradations": [],
            },
            {
                "timestamp": "2023-01-02T00:00:00", 
                "status": "healthy",
                "degradations": [],
            },
        ]
        
        history = degradation_entity.get_performance_history(limit=10)
        assert len(history) == 2
        assert history[0]["timestamp"] == "2023-01-01T00:00:00"
        assert history[1]["timestamp"] == "2023-01-02T00:00:00"
        
        # Test limit
        history_limited = degradation_entity.get_performance_history(limit=1)
        assert len(history_limited) == 1
        assert history_limited[0]["timestamp"] == "2023-01-02T00:00:00"  # Most recent
    
    def test_validate_invariants(self, degradation_entity):
        """Test domain invariant validation."""
        # Valid state should not raise
        degradation_entity.validate_invariants()
        
        # Current metrics without baseline should raise
        degradation_entity.baseline_metrics = None
        degradation_entity.current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.80,
        )
        
        with pytest.raises(ValueError, match="Cannot have current metrics without baseline"):
            degradation_entity.validate_invariants()
    
    def test_metric_compatibility_with_task_type(self, degradation_entity):
        """Test metric compatibility validation."""
        # Add incompatible metric (regression metric to classification task)
        incompatible_metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.MSE_INCREASE,
            threshold_value=0.2,
            baseline_value=0.1,
        )
        
        degradation_entity.add_degradation_metric(incompatible_metric)
        
        with pytest.raises(ValueError, match="not compatible with task"):
            degradation_entity.validate_invariants()
    
    def test_recovery_actions_recommendation(self, degradation_entity):
        """Test recovery action recommendations."""
        # Create degradation with critical severity
        critical_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.40,  # Critical degradation
            prediction_time_seconds=0.20,  # High latency
        )
        
        result = degradation_entity.evaluate_performance(critical_metrics)
        
        recommended_actions = result["recovery_actions_recommended"]
        assert RecoveryAction.RETRAIN_MODEL.value in recommended_actions
        assert RecoveryAction.DATA_QUALITY_CHECK.value in recommended_actions
        assert RecoveryAction.INFRASTRUCTURE_SCALING.value in recommended_actions
    
    def test_detection_history_tracking(self, degradation_entity):
        """Test that detection events are properly tracked in history."""
        initial_history_length = len(degradation_entity.detection_history)
        
        # Trigger degradation
        degraded_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.80,
        )
        
        degradation_entity.evaluate_performance(degraded_metrics)
        
        # Check that history was updated
        assert len(degradation_entity.detection_history) == initial_history_length + 1
        
        latest_entry = degradation_entity.detection_history[-1]
        assert latest_entry["status"] == DegradationStatus.DEGRADED.value
        assert len(latest_entry["degradations"]) > 0
        assert "timestamp" in latest_entry
        assert "evaluation_id" in latest_entry