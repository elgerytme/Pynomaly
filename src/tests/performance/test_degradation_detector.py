"""Unit tests for performance degradation detector."""

import pytest
from hypothesis import given, strategies as st
from datetime import datetime, timedelta
from uuid import uuid4

from pynomaly.domain.entities.model_performance import ModelPerformanceBaseline
from pynomaly.infrastructure.monitoring.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    PerformanceAlert,
    PerformanceTracker,
)


class TestDegradationDetector:
    """Test suite for performance degradation detection."""

    @pytest.fixture
    def baseline(self):
        """Create a baseline for testing."""
        return ModelPerformanceBaseline(
            model_id="test_model",
            mean=10.0,
            std=2.0,
            min_value=5.0,
            max_value=15.0,
            pct_thresholds={"90": 14.0, "95": 14.8, "99": 15.0},
            description="Test baseline"
        )

    @pytest.fixture
    def monitor(self):
        """Create a performance monitor."""
        return PerformanceMonitor(
            alert_thresholds={
                "execution_time": 30.0,
                "memory_usage": 1000.0,
                "cpu_usage": 80.0,
                "samples_per_second": 100.0
            }
        )

    def test_baseline_no_degradation(self, baseline):
        """Test normal performance within baseline."""
        # Values within normal range
        assert not baseline.is_degraded(10.0)  # Mean value
        assert not baseline.is_degraded(9.0)   # Slightly below mean
        assert not baseline.is_degraded(11.0)  # Slightly above mean
        assert not baseline.is_degraded(8.0)   # One std below mean

    def test_baseline_degradation_detected(self, baseline):
        """Test degradation detection scenarios."""
        # Values significantly below mean (degraded performance)
        assert baseline.is_degraded(5.0)  # 2.5 std below mean
        assert baseline.is_degraded(4.0)  # 3 std below mean
        assert baseline.is_degraded(2.0)  # 4 std below mean

    def test_baseline_with_zero_std(self):
        """Test baseline with zero standard deviation."""
        baseline = ModelPerformanceBaseline(
            model_id="test_model",
            mean=10.0,
            std=0.0,  # Zero std
            min_value=10.0,
            max_value=10.0
        )
        
        # Should not detect degradation when std is zero
        assert not baseline.is_degraded(5.0)
        assert not baseline.is_degraded(15.0)

    def test_baseline_threshold_factors(self, baseline):
        """Test different threshold factors."""
        # With threshold factor of 1.0
        assert baseline.is_degraded(7.0, threshold_factor=1.0)  # 1.5 std below
        
        # With threshold factor of 3.0
        assert not baseline.is_degraded(5.0, threshold_factor=3.0)  # 2.5 std below
        assert baseline.is_degraded(3.0, threshold_factor=3.0)     # 3.5 std below

    def test_percentile_thresholds(self, baseline):
        """Test percentile threshold retrieval."""
        assert baseline.get_threshold("90") == 14.0
        assert baseline.get_threshold("95") == 14.8
        assert baseline.get_threshold("99") == 15.0
        assert baseline.get_threshold("nonexistent") is None

    @given(st.floats(min_value=0.1, max_value=100.0))
    def test_baseline_with_random_values(self, random_value):
        """Property-based test with random values."""
        baseline = ModelPerformanceBaseline(
            model_id="test_model",
            mean=random_value,
            std=random_value / 10,  # Small std
            min_value=random_value - random_value / 2,
            max_value=random_value + random_value / 2
        )
        
        # Values close to mean should not be degraded
        assert not baseline.is_degraded(random_value)
        assert not baseline.is_degraded(random_value * 0.95)

    def test_performance_metrics_alert_generation(self, monitor):
        """Test alert generation based on performance metrics."""
        # Create metrics that exceed thresholds
        metrics = PerformanceMetrics(
            execution_time=35.0,  # Exceeds threshold of 30.0
            memory_usage=1200.0,  # Exceeds threshold of 1000.0
            cpu_usage=85.0,       # Exceeds threshold of 80.0
            samples_per_second=50.0,  # Below threshold of 100.0
            operation_name="test_operation"
        )
        
        # Manually trigger alert check
        monitor._check_alerts(metrics)
        
        # Should have generated multiple alerts
        alerts = monitor.get_active_alerts()
        assert len(alerts) = 1
        
        # Check alert properties
        alert_types = [alert.alert_type for alert in alerts]
        assert "threshold_exceeded" in alert_types

    def test_alert_properties(self, monitor):
        """Test alert object properties."""
        # Create an alert manually
        alert = PerformanceAlert(
            alert_type="threshold_exceeded",
            severity="high",
            message="Test alert",
            metric_name="execution_time",
            current_value=35.0,
            threshold_value=30.0,
            operation_name="test_operation"
        )
        
        # Test alert properties
        assert alert.alert_type == "threshold_exceeded"
        assert alert.severity == "high"
        assert alert.current_value == 35.0
        assert alert.threshold_value == 30.0
        
        # Test serialization
        alert_dict = alert.to_dict()
        assert alert_dict["alert_type"] == "threshold_exceeded"
        assert alert_dict["severity"] == "high"
        assert alert_dict["current_value"] == 35.0

    def test_performance_tracker_context_manager(self, monitor):
        """Test PerformanceTracker context manager."""
        samples_processed = 1000
        quality_metrics = {"accuracy": 0.95, "f1_score": 0.88}
        
        with PerformanceTracker(
            monitor,
            operation_name="test_operation",
            algorithm_name="test_algo",
            dataset_size=samples_processed
        ) as tracker:
            # Set metrics during operation
            tracker.set_samples_processed(samples_processed)
            tracker.set_quality_metrics(quality_metrics)
            
            # Simulate some work
            import time
            time.sleep(0.01)
        
        # Check that metrics were recorded
        assert len(monitor.metrics_history) = 0
        latest_metric = monitor.metrics_history[-1]
        assert latest_metric.operation_name == "test_operation"
        assert latest_metric.algorithm_name == "test_algo"
        assert latest_metric.samples_processed == samples_processed

    def test_multiple_degradation_scenarios(self, baseline):
        """Test multiple degradation scenarios."""
        # Default threshold factor is 2.0, so degradation occurs when z-score  -2.0
        # With mean=10, std=2: z-score = (value - 10) / 2
        # So degradation when value  10 - 2*2 = 6.0
        scenarios = [
            # (value, expected_degraded, description)
            (10.0, False, "Mean value"),
            (12.0, False, "One std above mean"),
            (8.0, False, "One std below mean"),
            (6.0, False, "Two std below mean (at threshold)"),
            (5.9, True, "Just below threshold"),
            (4.0, True, "Three std below mean"),
            (2.0, True, "Four std below mean"),
            (14.0, False, "Two std above mean"),
            (16.0, False, "Three std above mean"),
        ]
        
        for value, expected_degraded, description in scenarios:
            actual_degraded = baseline.is_degraded(value)
            assert actual_degraded == expected_degraded, f"Failed for {description}: value={value}, expected={expected_degraded}, actual={actual_degraded}"

    @given(
        st.floats(min_value=1.0, max_value=100.0),
        st.floats(min_value=0.1, max_value=10.0),
        st.floats(min_value=0.5, max_value=5.0)
    )
    def test_degradation_with_hypothesis(self, mean, std, threshold_factor):
        """Property-based test for degradation detection."""
        baseline = ModelPerformanceBaseline(
            model_id="test_model",
            mean=mean,
            std=std,
            min_value=mean - 3 * std,
            max_value=mean + 3 * std
        )
        
        # Values very close to mean should not be degraded
        assert not baseline.is_degraded(mean, threshold_factor=threshold_factor)
        
        # Values significantly below mean should be degraded
        very_low_value = mean - (threshold_factor + 1) * std
        assert baseline.is_degraded(very_low_value, threshold_factor=threshold_factor)

    def test_alert_callback_mechanism(self, monitor):
        """Test alert callback mechanism."""
        callback_called = []
        
        def test_callback(alert):
            callback_called.append(alert)
        
        # Add callback
        monitor.add_alert_callback(test_callback)
        
        # Create metrics that trigger alert
        metrics = PerformanceMetrics(
            execution_time=35.0,  # Exceeds threshold
            operation_name="test_operation"
        )
        
        # Trigger alert
        monitor._check_alerts(metrics)
        
        # Check callback was called
        assert len(callback_called) = 1
        assert callback_called[0].metric_name == "execution_time"

    def test_alert_clearing(self, monitor):
        """Test alert clearing functionality."""
        # Generate some alerts
        metrics = PerformanceMetrics(
            execution_time=35.0,
            memory_usage=1200.0,
            operation_name="test_operation"
        )
        
        monitor._check_alerts(metrics)
        initial_alerts = len(monitor.get_active_alerts())
        assert initial_alerts = 0
        
        # Clear all alerts
        monitor.clear_alerts()
        assert len(monitor.get_active_alerts()) == 0
        
        # Generate alerts again
        monitor._check_alerts(metrics)
        assert len(monitor.get_active_alerts()) = 0
        
        # Clear specific alert type
        monitor.clear_alerts(alert_type="threshold_exceeded")
        remaining_alerts = monitor.get_active_alerts()
        for alert in remaining_alerts:
            assert alert.alert_type != "threshold_exceeded"

    def test_baseline_serialization(self, baseline):
        """Test baseline serialization to dictionary."""
        baseline_dict = baseline.to_dict()
        
        required_keys = [
            "id", "model_id", "version", "mean", "std", 
            "min_value", "max_value", "pct_thresholds",
            "created_at", "updated_at", "description"
        ]
        
        for key in required_keys:
            assert key in baseline_dict
        
        assert baseline_dict["model_id"] == "test_model"
        assert baseline_dict["mean"] == 10.0
        assert baseline_dict["std"] == 2.0
        assert baseline_dict["pct_thresholds"] == {"90": 14.0, "95": 14.8, "99": 15.0}
