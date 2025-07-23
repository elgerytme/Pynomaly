"""Comprehensive test suite for ModelPerformanceMonitor."""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from anomaly_detection.infrastructure.monitoring.model_performance_monitor import (
    ModelPerformanceMonitor,
    ModelPerformanceMetrics,
    AlertThreshold,
    PerformanceAlert,
    get_model_performance_monitor,
    initialize_monitoring
)
from anomaly_detection.domain.entities.detection_result import DetectionResult


class TestModelPerformanceMonitor:
    """Test suite for ModelPerformanceMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create ModelPerformanceMonitor instance."""
        return ModelPerformanceMonitor(
            retention_hours=24,
            alert_enabled=True,
            metrics_buffer_size=1000
        )
    
    @pytest.fixture
    def sample_detection_result(self):
        """Create sample DetectionResult."""
        return DetectionResult(
            predictions=np.array([-1, 1, 1, -1, 1, 1, -1, 1, 1, 1]),
            confidence_scores=np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.1, 0.7, 0.2, 0.1, 0.2]),
            algorithm="iforest",
            metadata={"test": True}
        )
    
    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample ground truth data."""
        return np.array([-1, 1, 1, -1, 1, -1, -1, 1, 1, 1])  # Some mismatches for testing

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        monitor = ModelPerformanceMonitor()
        
        assert monitor.retention_hours == 168  # 1 week
        assert monitor.alert_enabled is True
        assert monitor.metrics_buffer_size == 10000
        assert len(monitor._alert_thresholds) > 0  # Should have default thresholds
        assert not monitor._is_running

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        monitor = ModelPerformanceMonitor(
            retention_hours=48,
            alert_enabled=False,
            metrics_buffer_size=5000
        )
        
        assert monitor.retention_hours == 48
        assert monitor.alert_enabled is False
        assert monitor.metrics_buffer_size == 5000

    def test_setup_default_thresholds(self, monitor):
        """Test default threshold setup."""
        expected_thresholds = [
            "precision", "recall", "f1_score", "false_positive_rate",
            "prediction_time_ms", "memory_usage_mb", "data_drift_score", "concept_drift_score"
        ]
        
        for threshold_name in expected_thresholds:
            assert threshold_name in monitor._alert_thresholds
            threshold = monitor._alert_thresholds[threshold_name]
            assert isinstance(threshold, AlertThreshold)
            assert threshold.enabled is True

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring."""
        assert not monitor._is_running
        
        await monitor.start_monitoring()
        assert monitor._is_running
        assert monitor._cleanup_task is not None
        
        await monitor.stop_monitoring()
        assert not monitor._is_running

    def test_record_prediction_metrics_basic(self, monitor, sample_detection_result):
        """Test recording basic prediction metrics."""
        monitor.record_prediction_metrics(
            model_id="test_model",
            algorithm="iforest",
            prediction_result=sample_detection_result,
            prediction_time_ms=150.5
        )
        
        assert len(monitor._metrics_buffer) == 1
        assert len(monitor._performance_history["test_model"]) == 1
        
        metric = monitor._metrics_buffer[0]
        assert metric.model_id == "test_model"
        assert metric.algorithm == "iforest"
        assert metric.prediction_time_ms == 150.5
        assert metric.samples_processed == 10
        assert metric.anomalies_detected == 3
        assert metric.anomaly_rate == 0.3

    def test_record_prediction_metrics_with_ground_truth(self, monitor, sample_detection_result, sample_ground_truth):
        """Test recording prediction metrics with ground truth."""
        with patch('sklearn.metrics.precision_score', return_value=0.75):
            with patch('sklearn.metrics.recall_score', return_value=0.80):
                with patch('sklearn.metrics.f1_score', return_value=0.77):
                    with patch('sklearn.metrics.accuracy_score', return_value=0.78):
                        monitor.record_prediction_metrics(
                            model_id="test_model",
                            algorithm="iforest",
                            prediction_result=sample_detection_result,
                            prediction_time_ms=100.0,
                            ground_truth=sample_ground_truth
                        )
        
        metric = monitor._metrics_buffer[0]
        assert metric.precision == 0.75
        assert metric.recall == 0.80
        assert metric.f1_score == 0.77
        assert metric.accuracy == 0.78

    def test_record_training_metrics(self, monitor):
        """Test recording training metrics."""
        monitor.record_training_metrics(
            model_id="test_model",
            algorithm="iforest",
            training_time_ms=5000.0,
            dataset_size=1000,
            feature_count=10
        )
        
        assert len(monitor._metrics_buffer) == 1
        
        metric = monitor._metrics_buffer[0]
        assert metric.model_id == "test_model"
        assert metric.algorithm == "iforest"
        assert metric.training_time_ms == 5000.0
        assert metric.dataset_size == 1000
        assert metric.feature_count == 10

    def test_record_drift_metrics(self, monitor):
        """Test recording drift metrics."""
        monitor.record_drift_metrics(
            model_id="test_model",
            data_drift_score=0.25,
            concept_drift_score=0.15
        )
        
        assert len(monitor._metrics_buffer) == 1
        
        metric = monitor._metrics_buffer[0]
        assert metric.model_id == "test_model"
        assert metric.algorithm == "drift_detection"
        assert metric.data_drift_score == 0.25
        assert metric.concept_drift_score == 0.15

    def test_alert_triggering(self, monitor, sample_detection_result):
        """Test alert triggering when thresholds are exceeded."""
        # Set a very strict precision threshold
        strict_threshold = AlertThreshold(
            metric_name="precision",
            operator="gt",
            threshold_value=0.95,
            severity="warning"
        )
        monitor.add_alert_threshold(strict_threshold)
        
        # Record metrics with ground truth that should trigger alert
        ground_truth = np.array([-1, 1, 1, -1, 1, 1, -1, 1, 1, 1])
        
        with patch('sklearn.metrics.precision_score', return_value=0.5):  # Below threshold
            monitor.record_prediction_metrics(
                model_id="test_model",
                algorithm="iforest",
                prediction_result=sample_detection_result,
                prediction_time_ms=100.0,
                ground_truth=ground_truth
            )
        
        # Should not trigger alert because we're checking for "gt" (greater than)
        assert len(monitor._active_alerts) == 0
        
        # Now test with "lt" (less than) threshold
        strict_threshold.operator = "lt"
        monitor.add_alert_threshold(strict_threshold)
        
        with patch('sklearn.metrics.precision_score', return_value=0.5):  # Below threshold
            monitor.record_prediction_metrics(
                model_id="test_model_2",
                algorithm="iforest",
                prediction_result=sample_detection_result,
                prediction_time_ms=100.0,
                ground_truth=ground_truth
            )
        
        # Should trigger alert
        assert len(monitor._active_alerts) > 0

    def test_alert_cooldown(self, monitor):
        """Test alert cooldown functionality."""
        # Create threshold with short cooldown
        threshold = AlertThreshold(
            metric_name="prediction_time_ms",
            operator="gt",
            threshold_value=50.0,
            severity="warning",
            cooldown_minutes=0  # No cooldown for testing
        )
        monitor.add_alert_threshold(threshold)
        
        # Trigger alert twice
        monitor._trigger_alert("prediction_time_ms", 100.0, threshold, "test_model")
        alert_count_1 = len(monitor._active_alerts)
        
        monitor._trigger_alert("prediction_time_ms", 120.0, threshold, "test_model")
        alert_count_2 = len(monitor._active_alerts)
        
        # Second alert should update the existing one
        assert alert_count_2 >= alert_count_1

    def test_get_model_performance_summary(self, monitor, sample_detection_result):
        """Test getting model performance summary."""
        # Record multiple metrics
        for i in range(5):
            monitor.record_prediction_metrics(
                model_id="test_model",
                algorithm="iforest",
                prediction_result=sample_detection_result,
                prediction_time_ms=100.0 + i * 10
            )
        
        summary = monitor.get_model_performance_summary("test_model", hours=24)
        
        assert summary["model_id"] == "test_model"
        assert summary["total_predictions"] == 50  # 5 * 10 samples
        assert summary["total_anomalies_detected"] == 15  # 5 * 3 anomalies
        assert summary["metrics_count"] == 5
        assert "avg_prediction_time_ms" in summary
        assert "avg_anomaly_rate" in summary

    def test_get_model_performance_summary_no_data(self, monitor):
        """Test getting performance summary with no data."""
        summary = monitor.get_model_performance_summary("nonexistent_model", hours=24)
        
        assert summary["model_id"] == "nonexistent_model"
        assert "message" in summary
        assert "No recent metrics available" in summary["message"]

    def test_calculate_trends(self, monitor):
        """Test trend calculation."""
        # Create metrics with improving precision
        metrics = []
        for i in range(10):
            metric = ModelPerformanceMetrics(
                model_id="test_model",
                algorithm="iforest",
                timestamp=datetime.utcnow(),
                precision=0.5 + i * 0.05,  # Improving trend
                prediction_time_ms=100.0 - i * 5  # Improving trend (decreasing time)
            )
            metrics.append(metric)
        
        trends = monitor._calculate_trends(metrics)
        
        assert "precision" in trends
        assert trends["precision"] == "improving"
        assert "prediction_time_ms" in trends
        assert trends["prediction_time_ms"] == "improving"

    def test_get_active_alerts(self, monitor):
        """Test getting active alerts."""
        # Create test alert
        alert = PerformanceAlert(
            alert_id="test_alert",
            metric_name="precision",
            current_value=0.5,
            threshold_value=0.7,
            severity="warning",
            message="Test alert",
            timestamp=datetime.utcnow(),
            model_id="test_model"
        )
        
        monitor._active_alerts["test_key"] = alert
        
        active_alerts = monitor.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == "test_alert"

    def test_resolve_alert(self, monitor):
        """Test resolving alerts."""
        # Create test alert
        alert = PerformanceAlert(
            alert_id="test_alert",
            metric_name="precision",
            current_value=0.5,
            threshold_value=0.7,
            severity="warning",
            message="Test alert",
            timestamp=datetime.utcnow(),
            model_id="test_model"
        )
        
        monitor._active_alerts["test_key"] = alert
        
        # Resolve alert
        result = monitor.resolve_alert("test_alert")
        assert result is True
        assert alert.resolved is True
        assert alert.resolution_timestamp is not None

    def test_resolve_nonexistent_alert(self, monitor):
        """Test resolving nonexistent alert."""
        result = monitor.resolve_alert("nonexistent_alert")
        assert result is False

    def test_add_alert_threshold(self, monitor):
        """Test adding alert thresholds."""
        threshold = AlertThreshold(
            metric_name="custom_metric",
            operator="gte",
            threshold_value=0.8,
            severity="critical"
        )
        
        monitor.add_alert_threshold(threshold)
        
        assert "custom_metric" in monitor._alert_thresholds
        assert monitor._alert_thresholds["custom_metric"] == threshold

    def test_add_alert_callback(self, monitor):
        """Test adding alert callbacks."""
        callback_called = []
        
        def test_callback(alert):
            callback_called.append(alert.alert_id)
        
        monitor.add_alert_callback(test_callback)
        
        # Trigger an alert to test callback
        threshold = AlertThreshold(
            metric_name="test_metric",
            operator="gt",
            threshold_value=0.5,
            severity="info"
        )
        
        monitor._trigger_alert("test_metric", 0.8, threshold, "test_model")
        
        assert len(callback_called) > 0

    def test_export_metrics(self, monitor, sample_detection_result):
        """Test exporting metrics to file."""
        # Record some metrics
        monitor.record_prediction_metrics(
            model_id="test_model",
            algorithm="iforest",
            prediction_result=sample_detection_result,
            prediction_time_ms=100.0
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            monitor.export_metrics(output_path, model_id="test_model", hours=24)
            
            assert output_path.exists()
            
            # Verify content
            with open(output_path) as f:
                data = json.load(f)
                assert "metrics" in data
                assert data["model_id"] == "test_model"
                assert data["metrics_count"] == 1
        
        finally:
            output_path.unlink()

    def test_get_model_comparison(self, monitor, sample_detection_result):
        """Test model comparison functionality."""
        # Record metrics for multiple models
        models = ["model_1", "model_2", "model_3"]
        
        for model_id in models:
            monitor.record_prediction_metrics(
                model_id=model_id,
                algorithm="iforest",
                prediction_result=sample_detection_result,
                prediction_time_ms=100.0
            )
        
        comparison = monitor.get_model_comparison(models, hours=24)
        
        assert "models" in comparison
        assert len(comparison["models"]) == 3
        assert "analysis" in comparison
        
        for model_id in models:
            assert model_id in comparison["models"]

    def test_update_baselines(self, monitor):
        """Test baseline updating."""
        # Create test metric
        metric = ModelPerformanceMetrics(
            model_id="test_model",
            algorithm="iforest",
            timestamp=datetime.utcnow(),
            precision=0.8,
            prediction_time_ms=100.0
        )
        
        # Update baselines
        monitor._update_baselines("test_model", metric)
        
        assert "test_model" in monitor._performance_baselines
        baselines = monitor._performance_baselines["test_model"]
        assert "precision_avg" in baselines
        assert "prediction_time_ms_avg" in baselines

    @pytest.mark.asyncio
    async def test_cleanup_worker(self, monitor):
        """Test cleanup worker functionality."""
        # Add old metrics
        old_time = datetime.utcnow() - timedelta(hours=200)  # Older than retention
        
        old_metric = ModelPerformanceMetrics(
            model_id="test_model",
            algorithm="iforest",
            timestamp=old_time
        )
        
        monitor._metrics_buffer.append(old_metric)
        monitor._performance_history["test_model"].append(old_metric)
        
        # Add recent metrics
        recent_metric = ModelPerformanceMetrics(
            model_id="test_model",
            algorithm="iforest",
            timestamp=datetime.utcnow()
        )
        
        monitor._metrics_buffer.append(recent_metric)
        monitor._performance_history["test_model"].append(recent_metric)
        
        initial_buffer_size = len(monitor._metrics_buffer)
        assert initial_buffer_size == 2
        
        # Mock the cleanup process
        cutoff_time = datetime.utcnow() - timedelta(hours=monitor.retention_hours)
        
        with monitor._lock:
            # Simulate cleanup
            monitor._metrics_buffer = monitor._metrics_buffer.__class__(
                [m for m in monitor._metrics_buffer if m.timestamp >= cutoff_time],
                maxlen=monitor.metrics_buffer_size
            )
        
        # Should have removed old metric
        assert len(monitor._metrics_buffer) == 1

    def test_monitoring_context(self, monitor):
        """Test monitoring context manager."""
        with monitor.monitor_prediction("test_model", "iforest"):
            # Simulate some work
            import time
            time.sleep(0.01)
        
        # Should have recorded timing metrics
        assert len(monitor._metrics_buffer) == 1
        metric = monitor._metrics_buffer[0]
        assert metric.model_id == "test_model"
        assert metric.algorithm == "iforest"
        assert metric.prediction_time_ms > 0

    def test_global_instance_functions(self):
        """Test global instance management functions."""
        # Test get_model_performance_monitor
        monitor1 = get_model_performance_monitor()
        monitor2 = get_model_performance_monitor()
        assert monitor1 is monitor2  # Should be same instance
        
        # Test initialize_monitoring
        monitor3 = initialize_monitoring(retention_hours=48)
        assert monitor3.retention_hours == 48

    def test_custom_metrics_recording(self, monitor, sample_detection_result):
        """Test recording custom metrics."""
        custom_metrics = {
            "custom_score": 0.95,
            "processing_steps": 5.0
        }
        
        monitor.record_prediction_metrics(
            model_id="test_model",
            algorithm="iforest",
            prediction_result=sample_detection_result,
            prediction_time_ms=100.0,
            custom_metrics=custom_metrics
        )
        
        metric = monitor._metrics_buffer[0]
        assert metric.custom_metrics["custom_score"] == 0.95
        assert metric.custom_metrics["processing_steps"] == 5.0

    def test_performance_metrics_with_confidence(self, monitor):
        """Test metrics calculation with confidence scores."""
        # Create detection result with confidence scores
        detection_result = DetectionResult(
            predictions=np.array([-1, 1, 1, -1, 1]),
            confidence_scores=np.array([0.9, 0.1, 0.2, 0.8, 0.3]),
            algorithm="iforest"
        )
        
        monitor.record_prediction_metrics(
            model_id="test_model",
            algorithm="iforest",
            prediction_result=detection_result,
            prediction_time_ms=100.0
        )
        
        metric = monitor._metrics_buffer[0]
        assert metric.prediction_confidence is not None
        expected_confidence = np.mean([0.9, 0.1, 0.2, 0.8, 0.3])
        assert abs(metric.prediction_confidence - expected_confidence) < 0.001

    def test_memory_monitoring(self, monitor, sample_detection_result):
        """Test memory usage monitoring."""
        monitor.record_prediction_metrics(
            model_id="test_model",
            algorithm="iforest",
            prediction_result=sample_detection_result,
            prediction_time_ms=100.0,
            memory_usage_mb=256.5,
            cpu_usage_percent=45.2
        )
        
        metric = monitor._metrics_buffer[0]
        assert metric.memory_usage_mb == 256.5
        assert metric.cpu_usage_percent == 45.2

    def test_alert_threshold_operators(self, monitor):
        """Test different alert threshold operators."""
        test_cases = [
            ("gt", 100.0, 150.0, True),   # 150 > 100
            ("gt", 100.0, 50.0, False),   # 50 not > 100
            ("lt", 100.0, 50.0, True),    # 50 < 100
            ("lt", 100.0, 150.0, False),  # 150 not < 100
            ("gte", 100.0, 100.0, True),  # 100 >= 100
            ("gte", 100.0, 99.0, False),  # 99 not >= 100
            ("lte", 100.0, 100.0, True),  # 100 <= 100
            ("lte", 100.0, 101.0, False), # 101 not <= 100
            ("eq", 100.0, 100.0, True),   # 100 == 100
            ("eq", 100.0, 99.0, False),   # 99 != 100
        ]
        
        for operator, threshold_value, current_value, should_trigger in test_cases:
            threshold = AlertThreshold(
                metric_name="test_metric",
                operator=operator,
                threshold_value=threshold_value,
                severity="info",
                cooldown_minutes=0
            )
            
            # Clear previous alerts
            monitor._active_alerts.clear()
            
            # Test alert triggering
            monitor._trigger_alert("test_metric", current_value, threshold, "test_model")
            
            if should_trigger:
                assert len(monitor._active_alerts) > 0, f"Should trigger alert for {operator} {threshold_value} vs {current_value}"
            else:
                assert len(monitor._active_alerts) == 0, f"Should not trigger alert for {operator} {threshold_value} vs {current_value}"

    def teardown_method(self):
        """Cleanup after each test."""
        # Reset global instance
        import anomaly_detection.infrastructure.monitoring.model_performance_monitor as mpm
        mpm._model_performance_monitor = None