"""Comprehensive tests for model performance degradation detection functionality."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from packages.data_science.domain.entities.model_performance_degradation import (
    ModelPerformanceDegradation,
    DegradationStatus,
    RecoveryAction,
)
from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    PerformanceDegradationMetrics,
    DegradationSeverity,
    DegradationMetricType,
)
from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask,
)


class TestPerformanceDegradationMetrics:
    """Test performance degradation metrics value object."""
    
    def test_create_accuracy_degradation_metric(self):
        """Test creating accuracy degradation metric."""
        metric = PerformanceDegradationMetrics.create_accuracy_degradation_metric(
            baseline_accuracy=0.85,
            threshold_percentage=5.0
        )
        
        assert metric.metric_type == DegradationMetricType.ACCURACY_DROP
        assert metric.baseline_value == 0.85
        assert metric.threshold_value == 0.8075  # 0.85 * (1 - 0.05)
        assert metric.alert_enabled is True
    
    def test_create_mse_degradation_metric(self):
        """Test creating MSE degradation metric."""
        metric = PerformanceDegradationMetrics.create_mse_degradation_metric(
            baseline_mse=0.05,
            threshold_percentage=20.0
        )
        
        assert metric.metric_type == DegradationMetricType.MSE_INCREASE
        assert metric.baseline_value == 0.05
        assert metric.threshold_value == 0.06  # 0.05 * (1 + 0.20)
        assert metric.alert_enabled is True
    
    def test_degradation_percentage_calculation(self):
        """Test degradation percentage calculation."""
        # Test accuracy drop (higher is better)
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.80,
            baseline_value=0.85,
            current_value=0.80
        )
        
        degradation_pct = metric.calculate_degradation_percentage()
        assert abs(degradation_pct - 5.88) < 0.1  # (0.85 - 0.80) / 0.85 * 100
        
        # Test MSE increase (lower is better)
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.MSE_INCREASE,
            threshold_value=0.06,
            baseline_value=0.05,
            current_value=0.065
        )
        
        degradation_pct = metric.calculate_degradation_percentage()
        assert abs(degradation_pct - 30.0) < 0.1  # (0.065 - 0.05) / 0.05 * 100
    
    def test_is_degraded_accuracy(self):
        """Test degradation detection for accuracy metric."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.80,
            baseline_value=0.85,
            current_value=0.75  # Below threshold
        )
        
        assert metric.is_degraded() is True
        
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.80,
            baseline_value=0.85,
            current_value=0.82  # Above threshold
        )
        
        assert metric.is_degraded() is False
    
    def test_is_degraded_mse(self):
        """Test degradation detection for MSE metric."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.MSE_INCREASE,
            threshold_value=0.06,
            baseline_value=0.05,
            current_value=0.07  # Above threshold (worse)
        )
        
        assert metric.is_degraded() is True
        
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.MSE_INCREASE,
            threshold_value=0.06,
            baseline_value=0.05,
            current_value=0.055  # Below threshold (better)
        )
        
        assert metric.is_degraded() is False
    
    def test_severity_threshold_calculation(self):
        """Test severity threshold calculation."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.70,
            baseline_value=0.85,
            current_value=0.60  # ~29% degradation
        )
        
        severity = metric.get_severity_threshold()
        assert severity == DegradationSeverity.MAJOR
        
        metric.current_value = 0.40  # ~53% degradation
        severity = metric.get_severity_threshold()
        assert severity == DegradationSeverity.CRITICAL
    
    def test_should_alert(self):
        """Test alert triggering logic."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.80,
            baseline_value=0.85,
            current_value=0.75,
            consecutive_breaches=1,
            alert_enabled=True
        )
        
        assert metric.should_alert() is True
        
        # Disable alerts
        metric.alert_enabled = False
        assert metric.should_alert() is False
        
        # No consecutive breaches
        metric.alert_enabled = True
        metric.consecutive_breaches = 0
        assert metric.should_alert() is False
    
    def test_update_current_value(self):
        """Test updating current value and recalculating metrics."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.80,
            baseline_value=0.85,
            current_value=0.83,
            consecutive_breaches=0
        )
        
        # Update with degraded value
        updated_metric = metric.update_current_value(0.75)
        
        assert updated_metric.current_value == 0.75
        assert updated_metric.is_degraded() is True
        assert updated_metric.consecutive_breaches == 1
        assert updated_metric.degradation_percentage > 0
    
    def test_alert_message_generation(self):
        """Test alert message generation."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.80,
            baseline_value=0.85,
            current_value=0.75,
            consecutive_breaches=1,
            alert_enabled=True,
            severity=DegradationSeverity.MAJOR
        )
        
        message = metric.get_alert_message()
        assert "Performance degradation detected" in message
        assert "accuracy_drop" in message
        assert "MAJOR" in message
        assert "0.850" in message  # baseline
        assert "0.750" in message  # current
    
    def test_recovery_recommendation(self):
        """Test recovery recommendation generation."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.80,
            baseline_value=0.85,
            current_value=0.75
        )
        
        recommendation = metric.get_recovery_recommendation()
        assert "retraining with recent data" in recommendation.lower()
    
    def test_to_monitoring_dict(self):
        """Test conversion to monitoring dictionary."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.80,
            baseline_value=0.85,
            current_value=0.75
        )
        
        monitoring_dict = metric.to_monitoring_dict()
        
        required_keys = {
            "metric_type", "threshold_value", "baseline_value", 
            "current_value", "degradation_percentage", "severity",
            "is_degraded", "should_alert", "recovery_recommendation"
        }
        
        assert all(key in monitoring_dict for key in required_keys)
        assert monitoring_dict["metric_type"] == "accuracy_drop"
        assert monitoring_dict["is_degraded"] is True


class TestModelPerformanceDegradation:
    """Test model performance degradation entity."""
    
    @pytest.fixture
    def baseline_metrics(self):
        """Create baseline performance metrics."""
        return ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            roc_auc=0.90,
            prediction_time_seconds=0.05,
            prediction_confidence=0.88
        )
    
    @pytest.fixture
    def degradation_metrics(self):
        """Create degradation metrics for monitoring."""
        return [
            PerformanceDegradationMetrics.create_accuracy_degradation_metric(
                baseline_accuracy=0.85,
                threshold_percentage=5.0
            ),
            PerformanceDegradationMetrics.create_latency_degradation_metric(
                baseline_latency=0.05,
                threshold_percentage=50.0
            )
        ]
    
    @pytest.fixture
    def degradation_entity(self, baseline_metrics, degradation_metrics):
        """Create model performance degradation entity."""
        entity = ModelPerformanceDegradation(
            model_id="test_model_123",
            model_name="Test Classification Model",
            model_version="v1.0.0",
            task_type=ModelTask.BINARY_CLASSIFICATION,
            baseline_metrics=baseline_metrics,
            degradation_metrics=degradation_metrics,
            monitoring_enabled=True
        )
        return entity
    
    def test_entity_creation(self, degradation_entity):
        """Test entity creation with valid data."""
        assert degradation_entity.model_id == "test_model_123"
        assert degradation_entity.model_name == "Test Classification Model"
        assert degradation_entity.task_type == ModelTask.BINARY_CLASSIFICATION
        assert degradation_entity.status == DegradationStatus.HEALTHY
        assert degradation_entity.monitoring_enabled is True
        assert len(degradation_entity.degradation_metrics) == 2
    
    def test_add_degradation_metric(self, degradation_entity):
        """Test adding new degradation metric."""
        precision_metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.PRECISION_DROP,
            threshold_value=0.78,
            baseline_value=0.82
        )
        
        initial_count = len(degradation_entity.degradation_metrics)
        degradation_entity.add_degradation_metric(precision_metric)
        
        assert len(degradation_entity.degradation_metrics) == initial_count + 1
        assert any(
            m.metric_type == DegradationMetricType.PRECISION_DROP 
            for m in degradation_entity.degradation_metrics
        )
    
    def test_remove_degradation_metric(self, degradation_entity):
        """Test removing degradation metric."""
        initial_count = len(degradation_entity.degradation_metrics)
        degradation_entity.remove_degradation_metric(DegradationMetricType.ACCURACY_DROP)
        
        assert len(degradation_entity.degradation_metrics) == initial_count - 1
        assert not any(
            m.metric_type == DegradationMetricType.ACCURACY_DROP 
            for m in degradation_entity.degradation_metrics
        )
    
    def test_update_baseline_metrics(self, degradation_entity, baseline_metrics):
        """Test updating baseline metrics."""
        new_baseline = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            accuracy=0.90,  # Improved baseline
            precision=0.87,
            recall=0.92,
            f1_score=0.89,
            roc_auc=0.93,
            prediction_time_seconds=0.04,
            prediction_confidence=0.92
        )
        
        degradation_entity.update_baseline_metrics(new_baseline)
        
        assert degradation_entity.baseline_metrics.accuracy == 0.90
        
        # Check that degradation metrics were updated
        accuracy_metric = next(
            (m for m in degradation_entity.degradation_metrics 
             if m.metric_type == DegradationMetricType.ACCURACY_DROP),
            None
        )
        assert accuracy_metric is not None
        assert accuracy_metric.baseline_value == 0.90
    
    def test_evaluate_performance_healthy(self, degradation_entity):
        """Test performance evaluation with healthy metrics."""
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            accuracy=0.86,  # Above threshold
            precision=0.83,
            recall=0.89,
            f1_score=0.86,
            roc_auc=0.91,
            prediction_time_seconds=0.05,  # Within threshold
            prediction_confidence=0.89
        )
        
        result = degradation_entity.evaluate_performance(current_metrics)
        
        assert result["status"] == DegradationStatus.HEALTHY.value
        assert len(result["degradations"]) == 0
        assert result["should_alert"] is False
        assert degradation_entity.status == DegradationStatus.HEALTHY
    
    def test_evaluate_performance_degraded(self, degradation_entity):
        """Test performance evaluation with degraded metrics."""
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            accuracy=0.75,  # Below threshold (0.8075)
            precision=0.72,
            recall=0.78,
            f1_score=0.75,
            roc_auc=0.82,
            prediction_time_seconds=0.05,
            prediction_confidence=0.80
        )
        
        result = degradation_entity.evaluate_performance(current_metrics)
        
        assert result["status"] == DegradationStatus.DEGRADED.value
        assert len(result["degradations"]) > 0
        assert result["should_alert"] is True
        assert degradation_entity.status == DegradationStatus.DEGRADED
        assert degradation_entity.degradation_detected_at is not None
    
    def test_evaluate_performance_critical(self, degradation_entity):
        """Test performance evaluation with critical degradation."""
        current_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            accuracy=0.50,  # Severely degraded
            precision=0.45,
            recall=0.55,
            f1_score=0.50,
            roc_auc=0.60,
            prediction_time_seconds=0.15,  # Also degraded
            prediction_confidence=0.60
        )
        
        result = degradation_entity.evaluate_performance(current_metrics)
        
        assert result["status"] == DegradationStatus.CRITICAL.value
        assert len(result["degradations"]) > 0
        assert degradation_entity.status == DegradationStatus.CRITICAL
        
        # Check that we have multiple degradations
        degradation_types = [d["metric_type"] for d in result["degradations"]]
        assert "accuracy_drop" in degradation_types
        assert "prediction_time_increase" in degradation_types
    
    def test_recovery_process(self, degradation_entity):
        """Test recovery process from degraded to healthy state."""
        # First, degrade the model
        degraded_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            accuracy=0.75,
            precision=0.72,
            recall=0.78,
            f1_score=0.75,
            roc_auc=0.82,
            prediction_time_seconds=0.05,
            prediction_confidence=0.80
        )
        
        result = degradation_entity.evaluate_performance(degraded_metrics)
        assert result["status"] == DegradationStatus.DEGRADED.value
        
        # Then recover with healthy metrics
        healthy_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            accuracy=0.86,
            precision=0.83,
            recall=0.89,
            f1_score=0.86,
            roc_auc=0.91,
            prediction_time_seconds=0.05,
            prediction_confidence=0.89
        )
        
        # First healthy evaluation - should be "recovering"
        result = degradation_entity.evaluate_performance(healthy_metrics)
        assert result["status"] == DegradationStatus.RECOVERING.value
        assert degradation_entity.consecutive_healthy_evaluations == 1
        
        # Second healthy evaluation - still recovering
        result = degradation_entity.evaluate_performance(healthy_metrics)
        assert result["status"] == DegradationStatus.RECOVERING.value
        assert degradation_entity.consecutive_healthy_evaluations == 2
        
        # Third healthy evaluation - fully recovered
        result = degradation_entity.evaluate_performance(healthy_metrics)
        assert result["status"] == DegradationStatus.HEALTHY.value
        assert degradation_entity.consecutive_healthy_evaluations == 3
        assert degradation_entity.recovery_completed_at is not None
    
    def test_trigger_recovery_action(self, degradation_entity):
        """Test triggering recovery actions."""
        degradation_entity.auto_recovery_enabled = True
        
        degradation_entity.trigger_recovery_action(
            action=RecoveryAction.RETRAIN_MODEL,
            initiated_by="system",
            context={"reason": "accuracy_degradation"}
        )
        
        assert len(degradation_entity.recovery_actions) == 1
        action_record = degradation_entity.recovery_actions[0]
        assert action_record["action"] == RecoveryAction.RETRAIN_MODEL.value
        assert action_record["initiated_by"] == "system"
        assert action_record["status"] == "initiated"
        assert degradation_entity.recovery_started_at is not None
    
    def test_complete_recovery_action(self, degradation_entity):
        """Test completing recovery actions."""
        degradation_entity.auto_recovery_enabled = True
        
        # First trigger an action
        degradation_entity.trigger_recovery_action(
            action=RecoveryAction.RETRAIN_MODEL,
            initiated_by="system"
        )
        
        # Then complete it
        degradation_entity.complete_recovery_action(
            action=RecoveryAction.RETRAIN_MODEL,
            success=True,
            notes="Retraining completed successfully"
        )
        
        action_record = degradation_entity.recovery_actions[0]
        assert action_record["status"] == "completed"
        assert action_record["success"] is True
        assert action_record["notes"] == "Retraining completed successfully"
    
    def test_enable_disable_monitoring(self, degradation_entity):
        """Test enabling and disabling monitoring."""
        assert degradation_entity.monitoring_enabled is True
        assert degradation_entity.status == DegradationStatus.HEALTHY
        
        degradation_entity.disable_monitoring()
        assert degradation_entity.monitoring_enabled is False
        assert degradation_entity.status == DegradationStatus.MONITORING_DISABLED
        
        degradation_entity.enable_monitoring()
        assert degradation_entity.monitoring_enabled is True
        assert degradation_entity.status == DegradationStatus.HEALTHY
    
    def test_is_due_for_evaluation(self, degradation_entity):
        """Test evaluation timing logic."""
        # New entity should be due for evaluation
        assert degradation_entity.is_due_for_evaluation() is True
        
        # After evaluation, should not be due immediately
        degradation_entity.last_evaluation_at = datetime.utcnow()
        assert degradation_entity.is_due_for_evaluation() is False
        
        # After interval passes, should be due again
        degradation_entity.last_evaluation_at = datetime.utcnow() - timedelta(minutes=65)
        assert degradation_entity.is_due_for_evaluation() is True
        
        # Disabled monitoring should never be due
        degradation_entity.disable_monitoring()
        assert degradation_entity.is_due_for_evaluation() is False
    
    def test_should_send_alert(self, degradation_entity):
        """Test alert sending logic."""
        # Healthy state should not send alerts
        assert degradation_entity.should_send_alert() is False
        
        # Degrade the model
        degraded_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            accuracy=0.75,
            prediction_time_seconds=0.05
        )
        
        degradation_entity.evaluate_performance(degraded_metrics)
        assert degradation_entity.should_send_alert() is True
        
        # Mark alert as sent - should not send again immediately
        degradation_entity.mark_alert_sent()
        assert degradation_entity.should_send_alert() is False
        
        # After time passes, should send again
        degradation_entity.alert_sent_at = datetime.utcnow() - timedelta(minutes=65)
        assert degradation_entity.should_send_alert() is True
    
    def test_get_current_degradation_summary(self, degradation_entity):
        """Test getting degradation summary."""
        summary = degradation_entity.get_current_degradation_summary()
        
        required_keys = {
            "model_id", "model_name", "model_version", "status",
            "monitoring_enabled", "degradation_count", "degraded_metrics",
            "max_severity", "should_alert"
        }
        
        assert all(key in summary for key in required_keys)
        assert summary["model_id"] == "test_model_123"
        assert summary["status"] == DegradationStatus.HEALTHY.value
        assert summary["degradation_count"] == 0
    
    def test_performance_history_tracking(self, degradation_entity):
        """Test performance history tracking."""
        # Initial history should be empty
        history = degradation_entity.get_performance_history()
        assert len(history) == 0
        
        # Degrade the model to create history
        degraded_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            accuracy=0.75,
            prediction_time_seconds=0.05
        )
        
        degradation_entity.evaluate_performance(degraded_metrics)
        
        # Should now have history
        history = degradation_entity.get_performance_history()
        assert len(history) == 1
        assert history[0]["status"] == DegradationStatus.DEGRADED.value
        assert len(history[0]["degradations"]) > 0
    
    def test_validation_errors(self):
        """Test entity validation errors."""
        # Test invalid task type compatibility
        baseline_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.REGRESSION,  # Different from entity task type
            mse=0.05,
            rmse=0.22,
            mae=0.15,
            r2_score=0.85
        )
        
        with pytest.raises(ValueError, match="Task type must match"):
            ModelPerformanceDegradation(
                model_id="test_model",
                model_name="Test Model",
                model_version="v1.0.0",
                task_type=ModelTask.BINARY_CLASSIFICATION,  # Different from baseline
                baseline_metrics=baseline_metrics
            )
    
    def test_duplicate_degradation_metrics(self):
        """Test duplicate degradation metrics validation."""
        accuracy_metric1 = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.80,
            baseline_value=0.85
        )
        
        accuracy_metric2 = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.75,
            baseline_value=0.85
        )
        
        with pytest.raises(ValueError, match="Duplicate degradation metric types"):
            ModelPerformanceDegradation(
                model_id="test_model",
                model_name="Test Model",
                model_version="v1.0.0",
                task_type=ModelTask.BINARY_CLASSIFICATION,
                degradation_metrics=[accuracy_metric1, accuracy_metric2]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])