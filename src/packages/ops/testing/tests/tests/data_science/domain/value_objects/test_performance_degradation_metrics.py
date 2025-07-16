"""Tests for performance degradation metrics value object."""

import pytest
from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    PerformanceDegradationMetrics,
    DegradationMetricType,
    DegradationSeverity,
)


class TestPerformanceDegradationMetrics:
    """Test suite for PerformanceDegradationMetrics value object."""
    
    def test_create_basic_metric(self):
        """Test creating a basic degradation metric."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
        )
        
        assert metric.metric_type == DegradationMetricType.ACCURACY_DROP
        assert metric.threshold_value == 0.85
        assert metric.baseline_value == 0.90
        assert metric.current_value is None
        assert metric.severity == DegradationSeverity.MINOR
    
    def test_calculate_degradation_percentage_accuracy_drop(self):
        """Test degradation percentage calculation for accuracy drop."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.80,
        )
        
        # (0.90 - 0.80) / 0.90 * 100 = 11.11%
        degradation_pct = metric.calculate_degradation_percentage()
        assert abs(degradation_pct - 11.11) < 0.01
    
    def test_calculate_degradation_percentage_mse_increase(self):
        """Test degradation percentage calculation for MSE increase."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.MSE_INCREASE,
            threshold_value=0.12,
            baseline_value=0.10,
            current_value=0.15,
        )
        
        # (0.15 - 0.10) / 0.10 * 100 = 50%
        degradation_pct = metric.calculate_degradation_percentage()
        assert degradation_pct == 50.0
    
    def test_is_degraded_accuracy_drop(self):
        """Test degradation detection for accuracy drop."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.80,  # Below threshold
        )
        
        assert metric.is_degraded() is True
        
        # Test not degraded
        metric_not_degraded = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.87,  # Above threshold
        )
        
        assert metric_not_degraded.is_degraded() is False
    
    def test_is_degraded_mse_increase(self):
        """Test degradation detection for MSE increase."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.MSE_INCREASE,
            threshold_value=0.12,
            baseline_value=0.10,
            current_value=0.15,  # Above threshold
        )
        
        assert metric.is_degraded() is True
        
        # Test not degraded
        metric_not_degraded = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.MSE_INCREASE,
            threshold_value=0.12,
            baseline_value=0.10,
            current_value=0.08,  # Below threshold
        )
        
        assert metric_not_degraded.is_degraded() is False
    
    def test_get_severity_threshold(self):
        """Test severity threshold calculation."""
        # Test critical severity (>=50% degradation)
        metric_critical = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.40,
            baseline_value=0.90,
            current_value=0.40,  # 55.56% degradation
        )
        assert metric_critical.get_severity_threshold() == DegradationSeverity.CRITICAL
        
        # Test major severity (>=30% degradation)
        metric_major = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.60,
            baseline_value=0.90,
            current_value=0.60,  # 33.33% degradation
        )
        assert metric_major.get_severity_threshold() == DegradationSeverity.MAJOR
        
        # Test moderate severity (>=15% degradation)
        metric_moderate = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.75,
            baseline_value=0.90,
            current_value=0.75,  # 16.67% degradation
        )
        assert metric_moderate.get_severity_threshold() == DegradationSeverity.MODERATE
        
        # Test minor severity (<15% degradation)
        metric_minor = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.85,  # 5.56% degradation
        )
        assert metric_minor.get_severity_threshold() == DegradationSeverity.MINOR
    
    def test_should_alert(self):
        """Test alert triggering logic."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.80,
            consecutive_breaches=2,
            alert_enabled=True,
        )
        
        assert metric.should_alert() is True
        
        # Test alert disabled
        metric_no_alert = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.80,
            consecutive_breaches=2,
            alert_enabled=False,
        )
        
        assert metric_no_alert.should_alert() is False
        
        # Test not degraded
        metric_not_degraded = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.87,
            consecutive_breaches=2,
            alert_enabled=True,
        )
        
        assert metric_not_degraded.should_alert() is False
    
    def test_get_alert_message(self):
        """Test alert message generation."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.80,
            consecutive_breaches=1,
            alert_enabled=True,
        )
        
        message = metric.get_alert_message()
        assert "accuracy_drop" in message
        assert "11.1%" in message
        assert "0.900" in message
        assert "0.800" in message
        assert "MINOR" in message
    
    def test_get_recovery_recommendation(self):
        """Test recovery recommendation generation."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
        )
        
        recommendation = metric.get_recovery_recommendation()
        assert "retraining" in recommendation.lower()
        assert "feature quality" in recommendation.lower()
    
    def test_update_current_value(self):
        """Test updating current value and recalculating metrics."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            consecutive_breaches=0,
        )
        
        # Update with degraded value
        updated_metric = metric.update_current_value(0.80)
        
        assert updated_metric.current_value == 0.80
        assert updated_metric.consecutive_breaches == 1
        assert updated_metric.is_degraded() is True
        
        # Update with non-degraded value
        updated_metric2 = updated_metric.update_current_value(0.87)
        
        assert updated_metric2.current_value == 0.87
        assert updated_metric2.consecutive_breaches == 0
        assert updated_metric2.is_degraded() is False
    
    def test_to_monitoring_dict(self):
        """Test conversion to monitoring dictionary."""
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.80,
            consecutive_breaches=1,
        )
        
        monitoring_dict = metric.to_monitoring_dict()
        
        assert monitoring_dict["metric_type"] == "accuracy_drop"
        assert monitoring_dict["threshold_value"] == 0.85
        assert monitoring_dict["baseline_value"] == 0.90
        assert monitoring_dict["current_value"] == 0.80
        assert monitoring_dict["is_degraded"] is True
        assert monitoring_dict["consecutive_breaches"] == 1
        assert "recovery_recommendation" in monitoring_dict
    
    def test_factory_methods(self):
        """Test factory methods for creating common metrics."""
        # Test accuracy degradation metric
        accuracy_metric = PerformanceDegradationMetrics.create_accuracy_degradation_metric(
            baseline_accuracy=0.90,
            threshold_percentage=10.0,
        )
        
        assert accuracy_metric.metric_type == DegradationMetricType.ACCURACY_DROP
        assert accuracy_metric.baseline_value == 0.90
        assert accuracy_metric.threshold_value == 0.81  # 90% * (1 - 10/100)
        
        # Test MSE degradation metric
        mse_metric = PerformanceDegradationMetrics.create_mse_degradation_metric(
            baseline_mse=0.10,
            threshold_percentage=20.0,
        )
        
        assert mse_metric.metric_type == DegradationMetricType.MSE_INCREASE
        assert mse_metric.baseline_value == 0.10
        assert mse_metric.threshold_value == 0.12  # 10% * (1 + 20/100)
        
        # Test latency degradation metric
        latency_metric = PerformanceDegradationMetrics.create_latency_degradation_metric(
            baseline_latency=0.05,
            threshold_percentage=50.0,
        )
        
        assert latency_metric.metric_type == DegradationMetricType.PREDICTION_TIME_INCREASE
        assert latency_metric.baseline_value == 0.05
        assert latency_metric.threshold_value == 0.075  # 5% * (1 + 50/100)
    
    def test_validation_errors(self):
        """Test validation error handling."""
        # Test negative threshold value
        with pytest.raises(ValueError):
            PerformanceDegradationMetrics(
                metric_type=DegradationMetricType.ACCURACY_DROP,
                threshold_value=-0.1,
                baseline_value=0.90,
            )
        
        # Test negative baseline value
        with pytest.raises(ValueError):
            PerformanceDegradationMetrics(
                metric_type=DegradationMetricType.ACCURACY_DROP,
                threshold_value=0.85,
                baseline_value=-0.1,
            )
        
        # Test invalid degradation percentage
        with pytest.raises(ValueError):
            PerformanceDegradationMetrics(
                metric_type=DegradationMetricType.ACCURACY_DROP,
                threshold_value=0.85,
                baseline_value=0.90,
                degradation_percentage=150.0,
            )
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test zero baseline value
        metric = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.0,
            baseline_value=0.0,
            current_value=0.1,
        )
        
        assert metric.calculate_degradation_percentage() == 0.0
        
        # Test current value equals baseline
        metric_equal = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.90,
        )
        
        assert metric_equal.calculate_degradation_percentage() == 0.0
        assert metric_equal.is_degraded() is False
        
        # Test current value equals threshold
        metric_threshold = PerformanceDegradationMetrics(
            metric_type=DegradationMetricType.ACCURACY_DROP,
            threshold_value=0.85,
            baseline_value=0.90,
            current_value=0.85,
        )
        
        assert metric_threshold.is_degraded() is False  # Should not be degraded at threshold