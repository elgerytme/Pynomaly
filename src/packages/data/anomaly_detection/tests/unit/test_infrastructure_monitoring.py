"""Unit tests for infrastructure monitoring components."""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from collections import deque
from threading import RLock
from typing import Dict, List, Any, Optional

from anomaly_detection.infrastructure.monitoring.metrics_collector import (
    MetricsCollector, MetricPoint, get_metrics_collector
)
from anomaly_detection.infrastructure.monitoring.health_checker import (
    HealthChecker, HealthCheck, HealthStatus, get_health_checker
)
from anomaly_detection.infrastructure.monitoring.performance_monitor import (
    PerformanceMonitor, PerformanceContext, get_performance_monitor
)
from anomaly_detection.infrastructure.monitoring.model_performance_monitor import (
    ModelPerformanceMonitor, ModelMetrics, AlertThreshold, get_model_performance_monitor
)
from anomaly_detection.infrastructure.monitoring.alerting_system import (
    AlertingSystem, Alert, AlertSeverity, AlertChannel, get_alerting_system
)
from anomaly_detection.infrastructure.monitoring.dashboard import (
    MonitoringDashboard, get_monitoring_dashboard
)


class TestMetricsCollector:
    """Test cases for MetricsCollector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector()
        self.collector._metrics = deque(maxlen=1000)
        self.collector._operation_timers = {}
        self.collector._lock = RLock()
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        assert isinstance(self.collector._metrics, deque)
        assert isinstance(self.collector._operation_timers, dict)
        assert isinstance(self.collector._lock, RLock)
        assert self.collector._metrics.maxlen == 1000
    
    def test_record_metric(self):
        """Test recording a basic metric."""
        timestamp = datetime.now()
        
        self.collector.record_metric(
            name="test_metric",
            value=10.5,
            tags={"component": "test"},
            unit="seconds",
            timestamp=timestamp
        )
        
        assert len(self.collector._metrics) == 1
        metric = self.collector._metrics[0]
        assert metric.name == "test_metric"
        assert metric.value == 10.5
        assert metric.tags == {"component": "test"}
        assert metric.unit == "seconds"
        assert metric.timestamp == timestamp
    
    def test_increment_counter(self):
        """Test incrementing counter metrics."""
        self.collector.increment_counter("api_requests", {"endpoint": "/health"})
        self.collector.increment_counter("api_requests", {"endpoint": "/health"})
        
        assert len(self.collector._metrics) == 2
        
        # Both metrics should have value 1 (increment by 1 each time)
        for metric in self.collector._metrics:
            assert metric.name == "api_requests"
            assert metric.value == 1
            assert metric.tags == {"endpoint": "/health"}
    
    def test_set_gauge(self):
        """Test setting gauge metrics."""
        self.collector.set_gauge("memory_usage", 85.5, {"type": "heap"})
        
        assert len(self.collector._metrics) == 1
        metric = self.collector._metrics[0]
        assert metric.name == "memory_usage"
        assert metric.value == 85.5
        assert metric.tags == {"type": "heap"}
    
    def test_record_timing(self):
        """Test recording timing metrics."""
        self.collector.record_timing("api_response_time", 250.0, {"endpoint": "/detect"})
        
        assert len(self.collector._metrics) == 1
        metric = self.collector._metrics[0]
        assert metric.name == "api_response_time"
        assert metric.value == 250.0
        assert metric.unit == "ms"
        assert metric.tags == {"endpoint": "/detect"}
    
    @patch('time.time')
    def test_start_operation(self, mock_time):
        """Test starting operation timing."""
        mock_time.return_value = 1000.0
        
        operation_id = self.collector.start_operation("model_training", {"model": "test"})
        
        assert operation_id in self.collector._operation_timers
        timer_info = self.collector._operation_timers[operation_id]
        assert timer_info["operation"] == "model_training"
        assert timer_info["start_time"] == 1000.0
        assert timer_info["tags"] == {"model": "test"}
    
    @patch('time.time')
    def test_end_operation(self, mock_time):
        """Test ending operation timing."""
        mock_time.side_effect = [1000.0, 1002.5]  # 2.5 second duration
        
        # Start operation
        operation_id = self.collector.start_operation("model_training")
        
        # End operation
        duration = self.collector.end_operation(operation_id)
        
        assert duration == 2500.0  # 2.5 seconds in milliseconds
        assert operation_id not in self.collector._operation_timers
        
        # Should have recorded timing metric
        assert len(self.collector._metrics) == 1
        metric = self.collector._metrics[0]
        assert metric.name == "operation_duration"
        assert metric.value == 2500.0
        assert metric.tags["operation"] == "model_training"
    
    def test_end_operation_invalid_id(self):
        """Test ending operation with invalid ID."""
        duration = self.collector.end_operation("invalid_id")
        
        assert duration is None
        assert len(self.collector._metrics) == 0
    
    def test_record_model_metric(self):
        """Test recording model-specific metrics."""
        self.collector.record_model_metric(
            model_id="model_123",
            metric_name="accuracy",
            value=0.95,
            additional_tags={"dataset": "test"}
        )
        
        assert len(self.collector._metrics) == 1
        metric = self.collector._metrics[0]
        assert metric.name == "model_accuracy"
        assert metric.value == 0.95
        assert metric.tags["model_id"] == "model_123"
        assert metric.tags["dataset"] == "test"
    
    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        # Record various metrics
        self.collector.record_metric("latency", 100.0, {"service": "api"})
        self.collector.record_metric("latency", 200.0, {"service": "api"})
        self.collector.record_metric("latency", 150.0, {"service": "api"})
        self.collector.increment_counter("requests", {"service": "api"})
        self.collector.increment_counter("requests", {"service": "api"})
        
        stats = self.collector.get_summary_stats()
        
        assert stats["total_metrics"] == 5
        assert stats["unique_metric_names"] == {"latency", "requests"}
        assert "latency" in stats["metrics_by_name"]
        assert "requests" in stats["metrics_by_name"]
        assert stats["metrics_by_name"]["latency"]["count"] == 3
        assert stats["metrics_by_name"]["latency"]["avg"] == 150.0
        assert stats["metrics_by_name"]["latency"]["min"] == 100.0
        assert stats["metrics_by_name"]["latency"]["max"] == 200.0
    
    def test_get_metrics_by_name(self):
        """Test filtering metrics by name."""
        self.collector.record_metric("cpu_usage", 50.0)
        self.collector.record_metric("memory_usage", 75.0)
        self.collector.record_metric("cpu_usage", 60.0)
        
        cpu_metrics = self.collector.get_metrics_by_name("cpu_usage")
        
        assert len(cpu_metrics) == 2
        assert all(m.name == "cpu_usage" for m in cpu_metrics)
        assert [m.value for m in cpu_metrics] == [50.0, 60.0]
    
    def test_get_metrics_by_tags(self):
        """Test filtering metrics by tags."""
        self.collector.record_metric("latency", 100.0, {"service": "api"})
        self.collector.record_metric("latency", 200.0, {"service": "worker"})
        self.collector.record_metric("latency", 150.0, {"service": "api"})
        
        api_metrics = self.collector.get_metrics_by_tags({"service": "api"})
        
        assert len(api_metrics) == 2
        assert all(m.tags.get("service") == "api" for m in api_metrics)
    
    def test_cleanup_old_metrics(self):
        """Test cleaning up old metrics."""
        old_time = datetime.now() - timedelta(hours=2)
        recent_time = datetime.now() - timedelta(minutes=5)
        
        # Add old and recent metrics
        self.collector.record_metric("old_metric", 1.0, timestamp=old_time)
        self.collector.record_metric("recent_metric", 2.0, timestamp=recent_time)
        
        removed_count = self.collector.cleanup_old_metrics(max_age_hours=1)
        
        assert removed_count == 1
        assert len(self.collector._metrics) == 1
        assert self.collector._metrics[0].name == "recent_metric"
    
    def test_thread_safety(self):
        """Test thread safety of metrics collection."""
        import threading
        
        def record_metrics():
            for i in range(100):
                self.collector.record_metric("thread_test", i)
        
        threads = [threading.Thread(target=record_metrics) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should have all 500 metrics without corruption
        assert len(self.collector._metrics) == 500
    
    def test_metrics_deque_size_limit(self):
        """Test that metrics deque respects size limit."""
        # Record more metrics than the limit
        for i in range(1200):
            self.collector.record_metric("test", i)
        
        # Should only keep the last 1000 metrics
        assert len(self.collector._metrics) == 1000
        assert self.collector._metrics[0].value == 200  # First kept metric
        assert self.collector._metrics[-1].value == 1199  # Last metric


class TestHealthChecker:
    """Test cases for HealthChecker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.health_checker = HealthChecker()
        self.health_checker._checks = {}
        self.health_checker._lock = RLock()
    
    def test_health_checker_initialization(self):
        """Test health checker initialization."""
        assert isinstance(self.health_checker._checks, dict)
        assert isinstance(self.health_checker._lock, RLock)
    
    def test_register_check(self):
        """Test registering a health check."""
        def test_check():
            return HealthStatus.HEALTHY, {"status": "ok"}
        
        self.health_checker.register_check("test_service", test_check, timeout=5.0)
        
        assert "test_service" in self.health_checker._checks
        check_info = self.health_checker._checks["test_service"]
        assert check_info["check_function"] == test_check
        assert check_info["timeout"] == 5.0
    
    def test_unregister_check(self):
        """Test unregistering a health check."""
        self.health_checker.register_check("test_service", lambda: (HealthStatus.HEALTHY, {}))
        
        result = self.health_checker.unregister_check("test_service")
        
        assert result is True
        assert "test_service" not in self.health_checker._checks
    
    def test_unregister_nonexistent_check(self):
        """Test unregistering a non-existent check."""
        result = self.health_checker.unregister_check("nonexistent")
        
        assert result is False
    
    async def test_run_check_healthy(self):
        """Test running a healthy check."""
        def healthy_check():
            return HealthStatus.HEALTHY, {"latency": "10ms"}
        
        self.health_checker.register_check("test", healthy_check)
        
        result = await self.health_checker.run_check("test")
        
        assert result["name"] == "test"
        assert result["status"] == HealthStatus.HEALTHY
        assert result["details"]["latency"] == "10ms"
        assert "timestamp" in result
        assert "duration_ms" in result
    
    async def test_run_check_unhealthy(self):
        """Test running an unhealthy check."""
        def unhealthy_check():
            return HealthStatus.UNHEALTHY, {"error": "Connection failed"}
        
        self.health_checker.register_check("test", unhealthy_check)
        
        result = await self.health_checker.run_check("test")
        
        assert result["status"] == HealthStatus.UNHEALTHY
        assert result["details"]["error"] == "Connection failed"
    
    async def test_run_check_with_exception(self):
        """Test running a check that raises an exception."""
        def failing_check():
            raise ConnectionError("Database connection failed")
        
        self.health_checker.register_check("test", failing_check)
        
        result = await self.health_checker.run_check("test")
        
        assert result["status"] == HealthStatus.UNHEALTHY
        assert "ConnectionError" in result["details"]["error"]
        assert "Database connection failed" in result["details"]["error"]
    
    async def test_run_check_timeout(self):
        """Test running a check that times out."""
        async def slow_check():
            await asyncio.sleep(0.1)
            return HealthStatus.HEALTHY, {}
        
        self.health_checker.register_check("test", slow_check, timeout=0.05)
        
        result = await self.health_checker.run_check("test")
        
        assert result["status"] == HealthStatus.UNHEALTHY
        assert "timeout" in result["details"]["error"].lower()
    
    async def test_run_nonexistent_check(self):
        """Test running a non-existent check."""
        result = await self.health_checker.run_check("nonexistent")
        
        assert result["name"] == "nonexistent"
        assert result["status"] == HealthStatus.UNHEALTHY
        assert "not registered" in result["details"]["error"]
    
    async def test_run_all_checks(self):
        """Test running all registered checks."""
        def healthy_check():
            return HealthStatus.HEALTHY, {}
        
        def unhealthy_check():
            return HealthStatus.UNHEALTHY, {"error": "Failed"}
        
        self.health_checker.register_check("service1", healthy_check)
        self.health_checker.register_check("service2", unhealthy_check)
        
        results = await self.health_checker.run_all_checks()
        
        assert len(results) == 2
        assert results["service1"]["status"] == HealthStatus.HEALTHY
        assert results["service2"]["status"] == HealthStatus.UNHEALTHY
    
    async def test_get_health_summary(self):
        """Test getting overall health summary."""
        def healthy_check():
            return HealthStatus.HEALTHY, {}
        
        def degraded_check():
            return HealthStatus.DEGRADED, {"warning": "High latency"}
        
        def unhealthy_check():
            return HealthStatus.UNHEALTHY, {"error": "Failed"}
        
        self.health_checker.register_check("service1", healthy_check)
        self.health_checker.register_check("service2", degraded_check)
        self.health_checker.register_check("service3", unhealthy_check)
        
        summary = await self.health_checker.get_health_summary()
        
        assert summary["overall_status"] == HealthStatus.UNHEALTHY  # Worst status
        assert summary["total_checks"] == 3
        assert summary["healthy_count"] == 1
        assert summary["degraded_count"] == 1
        assert summary["unhealthy_count"] == 1
        assert len(summary["check_results"]) == 3
    
    async def test_built_in_database_check(self):
        """Test built-in database health check."""
        with patch('anomaly_detection.infrastructure.repositories.model_repository.ModelRepository') as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance
            mock_repo_instance.list_models.return_value = []
            
            self.health_checker.register_database_check()
            
            result = await self.health_checker.run_check("database")
            
            assert result["status"] == HealthStatus.HEALTHY
            mock_repo_instance.list_models.assert_called_once()
    
    async def test_built_in_memory_check(self):
        """Test built-in memory health check."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(percent=75.0, available=1024*1024*1024)
            
            self.health_checker.register_memory_check()
            
            result = await self.health_checker.run_check("memory")
            
            assert result["status"] == HealthStatus.HEALTHY
            assert result["details"]["usage_percent"] == 75.0
    
    async def test_built_in_disk_check(self):
        """Test built-in disk health check."""
        with patch('psutil.disk_usage') as mock_disk:
            mock_disk.return_value = Mock(percent=60.0, free=1024*1024*1024)
            
            self.health_checker.register_disk_check()
            
            result = await self.health_checker.run_check("disk")
            
            assert result["status"] == HealthStatus.HEALTHY
            assert result["details"]["usage_percent"] == 60.0


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
        self.monitor._profiles = deque(maxlen=1000)
        self.monitor._lock = RLock()
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        assert isinstance(self.monitor._profiles, deque)
        assert isinstance(self.monitor._lock, RLock)
    
    @patch('time.time')
    @patch('psutil.Process')
    def test_create_context(self, mock_process, mock_time):
        """Test creating performance monitoring context."""
        mock_time.side_effect = [1000.0, 1002.0]
        mock_proc = Mock()
        mock_proc.memory_info.return_value = Mock(rss=1024*1024)
        mock_proc.cpu_percent.return_value = 25.0
        mock_process.return_value = mock_proc
        
        with self.monitor.create_context("test_operation", {"param": "value"}):
            pass
        
        assert len(self.monitor._profiles) == 1
        profile = self.monitor._profiles[0]
        assert profile["operation"] == "test_operation"
        assert profile["duration_ms"] == 2000.0
        assert profile["tags"]["param"] == "value"
        assert "memory_rss" in profile
        assert "cpu_percent" in profile
    
    def test_record_profile(self):
        """Test recording performance profile."""
        profile_data = {
            "operation": "model_training",
            "duration_ms": 5000.0,
            "memory_peak": 512*1024*1024,
            "cpu_avg": 80.0
        }
        
        self.monitor.record_profile(**profile_data)
        
        assert len(self.monitor._profiles) == 1
        profile = self.monitor._profiles[0]
        assert profile["operation"] == "model_training"
        assert profile["duration_ms"] == 5000.0
        assert profile["memory_peak"] == 512*1024*1024
        assert profile["cpu_avg"] == 80.0
        assert "timestamp" in profile
    
    def test_get_operation_stats(self):
        """Test getting operation statistics."""
        # Record multiple profiles for same operation
        for duration in [1000.0, 2000.0, 3000.0]:
            self.monitor.record_profile(
                operation="api_request",
                duration_ms=duration,
                memory_rss=1024*1024
            )
        
        stats = self.monitor.get_operation_stats("api_request")
        
        assert stats["operation"] == "api_request"
        assert stats["count"] == 3
        assert stats["avg_duration_ms"] == 2000.0
        assert stats["min_duration_ms"] == 1000.0
        assert stats["max_duration_ms"] == 3000.0
    
    def test_get_recent_profiles(self):
        """Test getting recent performance profiles."""
        old_time = datetime.now() - timedelta(hours=2)
        recent_time = datetime.now() - timedelta(minutes=5)
        
        self.monitor.record_profile(
            operation="old_op",
            duration_ms=1000.0,
            timestamp=old_time
        )
        self.monitor.record_profile(
            operation="recent_op",
            duration_ms=2000.0,
            timestamp=recent_time
        )
        
        recent_profiles = self.monitor.get_recent_profiles(max_age_hours=1)
        
        assert len(recent_profiles) == 1
        assert recent_profiles[0]["operation"] == "recent_op"
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_get_resource_usage(self, mock_disk, mock_cpu, mock_memory):
        """Test getting current resource usage."""
        mock_memory.return_value = Mock(percent=75.0, available=1024*1024*1024)
        mock_cpu.return_value = 60.0
        mock_disk.return_value = Mock(percent=45.0, free=2*1024*1024*1024)
        
        usage = self.monitor.get_resource_usage()
        
        assert usage["memory_percent"] == 75.0
        assert usage["cpu_percent"] == 60.0
        assert usage["disk_percent"] == 45.0
        assert "timestamp" in usage
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Record various operations
        operations = ["api_request", "model_training", "data_processing"]
        for i, op in enumerate(operations):
            for j in range(3):
                self.monitor.record_profile(
                    operation=op,
                    duration_ms=(i+1) * 1000.0 + j * 100.0,
                    memory_rss=(i+1) * 1024 * 1024
                )
        
        summary = self.monitor.get_performance_summary()
        
        assert summary["total_profiles"] == 9
        assert len(summary["operation_stats"]) == 3
        assert "api_request" in summary["operation_stats"]
        assert "model_training" in summary["operation_stats"]
        assert "data_processing" in summary["operation_stats"]
    
    def test_performance_context_exception_handling(self):
        """Test performance context with exception."""
        with pytest.raises(ValueError):
            with self.monitor.create_context("failing_operation"):
                raise ValueError("Test error")
        
        # Should still record profile even with exception
        assert len(self.monitor._profiles) == 1
        profile = self.monitor._profiles[0]
        assert profile["operation"] == "failing_operation"
        assert profile.get("exception") == "ValueError: Test error"


class TestModelPerformanceMonitor:
    """Test cases for ModelPerformanceMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = ModelPerformanceMonitor()
        self.monitor._model_metrics = {}
        self.monitor._alert_thresholds = {}
        self.monitor._active_alerts = {}
        self.monitor._lock = RLock()
    
    def test_model_performance_monitor_initialization(self):
        """Test model performance monitor initialization."""
        assert isinstance(self.monitor._model_metrics, dict)
        assert isinstance(self.monitor._alert_thresholds, dict)
        assert isinstance(self.monitor._active_alerts, dict)
    
    def test_record_prediction_metrics(self):
        """Test recording prediction metrics."""
        metrics = {
            "accuracy": 0.95,
            "precision": 0.87,
            "recall": 0.92,
            "f1_score": 0.89
        }
        
        self.monitor.record_prediction_metrics(
            model_id="model_123",
            metrics=metrics,
            sample_count=1000
        )
        
        assert "model_123" in self.monitor._model_metrics
        model_data = self.monitor._model_metrics["model_123"]
        assert len(model_data["prediction_metrics"]) == 1
        
        recorded_metric = model_data["prediction_metrics"][0]
        assert recorded_metric["accuracy"] == 0.95
        assert recorded_metric["sample_count"] == 1000
        assert "timestamp" in recorded_metric
    
    def test_record_training_metrics(self):
        """Test recording training metrics."""
        training_metrics = {
            "training_time_ms": 30000.0,
            "convergence_epochs": 50,
            "final_loss": 0.02
        }
        
        self.monitor.record_training_metrics(
            model_id="model_123",
            metrics=training_metrics,
            dataset_size=5000
        )
        
        model_data = self.monitor._model_metrics["model_123"]
        assert len(model_data["training_metrics"]) == 1
        
        recorded_metric = model_data["training_metrics"][0]
        assert recorded_metric["training_time_ms"] == 30000.0
        assert recorded_metric["dataset_size"] == 5000
    
    def test_record_drift_metrics(self):
        """Test recording drift detection metrics."""
        drift_metrics = {
            "drift_score": 0.15,
            "drift_detected": True,
            "affected_features": ["feature1", "feature2"]
        }
        
        self.monitor.record_drift_metrics(
            model_id="model_123",
            drift_type="concept_drift",
            metrics=drift_metrics
        )
        
        model_data = self.monitor._model_metrics["model_123"]
        assert len(model_data["drift_metrics"]) == 1
        
        recorded_metric = model_data["drift_metrics"][0]
        assert recorded_metric["drift_type"] == "concept_drift"
        assert recorded_metric["drift_score"] == 0.15
        assert recorded_metric["drift_detected"] is True
    
    def test_add_alert_threshold(self):
        """Test adding alert thresholds."""
        threshold = AlertThreshold(
            metric_name="accuracy",
            min_value=0.8,
            max_value=None,
            severity=AlertSeverity.WARNING
        )
        
        self.monitor.add_alert_threshold("model_123", threshold)
        
        assert "model_123" in self.monitor._alert_thresholds
        thresholds = self.monitor._alert_thresholds["model_123"]
        assert "accuracy" in thresholds
        assert thresholds["accuracy"] == threshold
    
    def test_check_alert_thresholds(self):
        """Test checking metrics against alert thresholds."""
        # Set up threshold
        threshold = AlertThreshold(
            metric_name="accuracy",
            min_value=0.85,
            severity=AlertSeverity.CRITICAL
        )
        self.monitor.add_alert_threshold("model_123", threshold)
        
        # Record metric that violates threshold
        metrics = {"accuracy": 0.75}  # Below minimum
        
        with patch.object(self.monitor, '_trigger_alert') as mock_trigger:
            self.monitor.record_prediction_metrics("model_123", metrics, 1000)
            
            mock_trigger.assert_called_once()
            alert_call = mock_trigger.call_args[0][0]
            assert alert_call.model_id == "model_123"
            assert alert_call.metric_name == "accuracy"
            assert alert_call.severity == AlertSeverity.CRITICAL
    
    def test_get_model_performance_summary(self):
        """Test getting model performance summary."""
        # Record various metrics
        self.monitor.record_prediction_metrics(
            "model_123",
            {"accuracy": 0.9, "precision": 0.85},
            1000
        )
        self.monitor.record_training_metrics(
            "model_123",
            {"training_time_ms": 25000.0},
            5000
        )
        
        summary = self.monitor.get_model_performance_summary("model_123")
        
        assert summary["model_id"] == "model_123"
        assert summary["prediction_metrics_count"] == 1
        assert summary["training_metrics_count"] == 1
        assert "latest_prediction_metrics" in summary
        assert "latest_training_metrics" in summary
        assert summary["latest_prediction_metrics"]["accuracy"] == 0.9
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        # Create mock alert
        alert = Alert(
            model_id="model_123",
            metric_name="accuracy",
            current_value=0.75,
            threshold_value=0.85,
            severity=AlertSeverity.WARNING,
            message="Accuracy below threshold"
        )
        
        self.monitor._active_alerts["alert_123"] = alert
        
        active_alerts = self.monitor.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0].model_id == "model_123"
        assert active_alerts[0].metric_name == "accuracy"
    
    def test_export_metrics(self):
        """Test exporting metrics to file."""
        self.monitor.record_prediction_metrics(
            "model_123",
            {"accuracy": 0.9},
            1000
        )
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                self.monitor.export_metrics("metrics.json")
                
                mock_file.assert_called_once_with("metrics.json", 'w')
                mock_json_dump.assert_called_once()
                
                # Check exported data structure
                exported_data = mock_json_dump.call_args[0][0]
                assert "model_123" in exported_data
    
    def test_get_model_comparison(self):
        """Test comparing multiple models."""
        # Record metrics for multiple models
        models = ["model_1", "model_2", "model_3"]
        for i, model in enumerate(models):
            self.monitor.record_prediction_metrics(
                model,
                {"accuracy": 0.8 + i * 0.05, "precision": 0.75 + i * 0.05},
                1000
            )
        
        comparison = self.monitor.get_model_comparison(
            models,
            metric_name="accuracy"
        )
        
        assert len(comparison) == 3
        assert comparison[0]["model_id"] == "model_1"
        assert comparison[0]["accuracy"] == 0.8
        assert comparison[2]["model_id"] == "model_3"
        assert comparison[2]["accuracy"] == 0.9


class TestAlertingSystem:
    """Test cases for AlertingSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.alerting = AlertingSystem()
        self.alerting._channels = {}
        self.alerting._alert_rules = []
        self.alerting._alert_history = deque(maxlen=1000)
        self.alerting._lock = RLock()
    
    def test_alerting_system_initialization(self):
        """Test alerting system initialization."""
        assert isinstance(self.alerting._channels, dict)
        assert isinstance(self.alerting._alert_rules, list)
        assert isinstance(self.alerting._alert_history, deque)
    
    def test_add_slack_channel(self):
        """Test adding Slack alert channel."""
        self.alerting.add_slack_channel(
            name="alerts",
            webhook_url="https://hooks.slack.com/test",
            channel="#alerts"
        )
        
        assert "alerts" in self.alerting._channels
        channel = self.alerting._channels["alerts"]
        assert channel["type"] == "slack"
        assert channel["webhook_url"] == "https://hooks.slack.com/test"
        assert channel["channel"] == "#alerts"
    
    def test_add_email_channel(self):
        """Test adding email alert channel."""
        self.alerting.add_email_channel(
            name="admin_email",
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            username="alerts@company.com",
            password="password123",
            recipients=["admin@company.com"]
        )
        
        channel = self.alerting._channels["admin_email"]
        assert channel["type"] == "email"
        assert channel["smtp_server"] == "smtp.gmail.com"
        assert channel["recipients"] == ["admin@company.com"]
    
    def test_add_webhook_channel(self):
        """Test adding webhook alert channel."""
        self.alerting.add_webhook_channel(
            name="webhook_alert",
            url="https://api.company.com/alerts",
            headers={"Authorization": "Bearer token123"}
        )
        
        channel = self.alerting._channels["webhook_alert"]
        assert channel["type"] == "webhook"
        assert channel["url"] == "https://api.company.com/alerts"
        assert channel["headers"]["Authorization"] == "Bearer token123"
    
    def test_add_alert_rule(self):
        """Test adding alert routing rule."""
        self.alerting.add_alert_rule(
            severity=AlertSeverity.CRITICAL,
            channels=["slack", "email"],
            conditions={"model_id": "production_model"}
        )
        
        assert len(self.alerting._alert_rules) == 1
        rule = self.alerting._alert_rules[0]
        assert rule["severity"] == AlertSeverity.CRITICAL
        assert rule["channels"] == ["slack", "email"]
        assert rule["conditions"]["model_id"] == "production_model"
    
    @patch('aiohttp.ClientSession.post')
    async def test_send_slack_alert(self, mock_post):
        """Test sending Slack alert."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_response
        
        self.alerting.add_slack_channel("test", "https://hooks.slack.com/test", "#alerts")
        
        alert = Alert(
            model_id="model_123",
            metric_name="accuracy",
            current_value=0.75,
            threshold_value=0.85,
            severity=AlertSeverity.WARNING,
            message="Accuracy below threshold"
        )
        
        result = await self.alerting._send_slack_alert(alert, self.alerting._channels["test"])
        
        assert result is True
        mock_post.assert_called_once()
    
    @patch('smtplib.SMTP')
    async def test_send_email_alert(self, mock_smtp_class):
        """Test sending email alert."""
        mock_smtp = Mock()
        mock_smtp_class.return_value = mock_smtp
        
        self.alerting.add_email_channel(
            "test", "smtp.test.com", 587, "user", "pass", ["admin@test.com"]
        )
        
        alert = Alert(
            model_id="model_123",
            metric_name="accuracy",
            current_value=0.75,
            threshold_value=0.85,
            severity=AlertSeverity.CRITICAL,
            message="Critical accuracy drop"
        )
        
        result = await self.alerting._send_email_alert(alert, self.alerting._channels["test"])
        
        assert result is True
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once_with("user", "pass")
        mock_smtp.send_message.assert_called_once()
    
    async def test_process_alert_with_routing(self):
        """Test processing alert with routing rules."""
        # Set up channels and rules
        self.alerting.add_slack_channel("slack", "https://hooks.slack.com/test", "#alerts")
        self.alerting.add_alert_rule(
            severity=AlertSeverity.CRITICAL,
            channels=["slack"],
            conditions={"model_id": "production_model"}
        )
        
        alert = Alert(
            model_id="production_model",
            metric_name="accuracy",
            current_value=0.60,
            threshold_value=0.80,
            severity=AlertSeverity.CRITICAL,
            message="Critical accuracy drop"
        )
        
        with patch.object(self.alerting, '_send_slack_alert', return_value=True) as mock_send:
            await self.alerting.process_alert(alert)
            
            mock_send.assert_called_once_with(alert, self.alerting._channels["slack"])
        
        # Check alert history
        assert len(self.alerting._alert_history) == 1
        assert self.alerting._alert_history[0]["alert_id"] == alert.alert_id
    
    def test_get_alert_statistics(self):
        """Test getting alert statistics."""
        # Add some alerts to history
        for i in range(5):
            alert_data = {
                "alert_id": f"alert_{i}",
                "severity": AlertSeverity.WARNING.value,
                "model_id": f"model_{i % 2}",
                "timestamp": datetime.now() - timedelta(hours=i)
            }
            self.alerting._alert_history.append(alert_data)
        
        stats = self.alerting.get_alert_statistics()
        
        assert stats["total_alerts"] == 5
        assert stats["by_severity"][AlertSeverity.WARNING.value] == 5
        assert len(stats["by_model"]) == 2
        assert "model_0" in stats["by_model"]
        assert "model_1" in stats["by_model"]
    
    async def test_test_channels(self):
        """Test testing alert channels."""
        self.alerting.add_slack_channel("slack", "https://hooks.slack.com/test", "#alerts")
        
        with patch.object(self.alerting, '_send_slack_alert', return_value=True) as mock_send:
            results = await self.alerting.test_channels()
            
            assert len(results) == 1
            assert results["slack"]["success"] is True
            mock_send.assert_called_once()


class TestMonitoringDashboard:
    """Test cases for MonitoringDashboard class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dashboard = MonitoringDashboard()
    
    @patch('anomaly_detection.infrastructure.monitoring.dashboard.get_metrics_collector')
    @patch('anomaly_detection.infrastructure.monitoring.dashboard.get_health_checker')
    @patch('anomaly_detection.infrastructure.monitoring.dashboard.get_performance_monitor')
    def test_get_dashboard_summary(self, mock_perf_monitor, mock_health_checker, mock_metrics):
        """Test getting dashboard summary."""
        # Mock metrics collector
        mock_metrics_instance = Mock()
        mock_metrics_instance.get_summary_stats.return_value = {
            "total_metrics": 100,
            "unique_metric_names": {"latency", "accuracy"}
        }
        mock_metrics.return_value = mock_metrics_instance
        
        # Mock health checker
        mock_health_instance = Mock()
        mock_health_instance.get_health_summary = AsyncMock(return_value={
            "overall_status": HealthStatus.HEALTHY,
            "healthy_count": 3,
            "total_checks": 3
        })
        mock_health_checker.return_value = mock_health_instance
        
        # Mock performance monitor
        mock_perf_instance = Mock()
        mock_perf_instance.get_performance_summary.return_value = {
            "total_profiles": 50,
            "operation_stats": {"api_request": {"avg_duration_ms": 150.0}}
        }
        mock_perf_monitor.return_value = mock_perf_instance
        
        summary = asyncio.run(self.dashboard.get_dashboard_summary())
        
        assert summary["metrics_summary"]["total_metrics"] == 100
        assert summary["health_summary"]["overall_status"] == HealthStatus.HEALTHY
        assert summary["performance_summary"]["total_profiles"] == 50
        assert "timestamp" in summary
    
    def test_get_performance_trends(self):
        """Test getting performance trends."""
        with patch('anomaly_detection.infrastructure.monitoring.dashboard.get_performance_monitor') as mock_monitor:
            mock_instance = Mock()
            mock_instance.get_recent_profiles.return_value = [
                {"operation": "api_request", "duration_ms": 100.0, "timestamp": datetime.now()},
                {"operation": "api_request", "duration_ms": 150.0, "timestamp": datetime.now()},
            ]
            mock_monitor.return_value = mock_instance
            
            trends = self.dashboard.get_performance_trends(hours=24)
            
            assert len(trends) == 2
            assert trends[0]["operation"] == "api_request"
            mock_instance.get_recent_profiles.assert_called_once_with(max_age_hours=24)
    
    def test_get_alert_summary(self):
        """Test getting alert summary."""
        with patch('anomaly_detection.infrastructure.monitoring.dashboard.get_alerting_system') as mock_alerting:
            mock_instance = Mock()
            mock_instance.get_alert_statistics.return_value = {
                "total_alerts": 10,
                "by_severity": {"WARNING": 8, "CRITICAL": 2}
            }
            mock_instance.get_active_alerts.return_value = []
            mock_alerting.return_value = mock_instance
            
            summary = self.dashboard.get_alert_summary()
            
            assert summary["statistics"]["total_alerts"] == 10
            assert summary["active_alerts"] == []
    
    def test_get_operation_breakdown(self):
        """Test getting operation performance breakdown."""
        with patch('anomaly_detection.infrastructure.monitoring.dashboard.get_performance_monitor') as mock_monitor:
            mock_instance = Mock()
            mock_instance.get_performance_summary.return_value = {
                "operation_stats": {
                    "api_request": {"count": 100, "avg_duration_ms": 150.0},
                    "model_training": {"count": 5, "avg_duration_ms": 30000.0}
                }
            }
            mock_monitor.return_value = mock_instance
            
            breakdown = self.dashboard.get_operation_breakdown()
            
            assert len(breakdown) == 2
            assert breakdown["api_request"]["count"] == 100
            assert breakdown["model_training"]["avg_duration_ms"] == 30000.0


class TestMonitoringIntegration:
    """Test integration between monitoring components."""
    
    def test_singleton_instances(self):
        """Test that monitoring components return singleton instances."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        assert collector1 is collector2
        
        health1 = get_health_checker()
        health2 = get_health_checker()
        assert health1 is health2
        
        perf1 = get_performance_monitor()
        perf2 = get_performance_monitor()
        assert perf1 is perf2
    
    @patch('anomaly_detection.infrastructure.monitoring.metrics_collector.get_metrics_collector')
    def test_performance_monitor_metrics_integration(self, mock_get_collector):
        """Test integration between performance monitor and metrics collector."""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        
        monitor = PerformanceMonitor()
        monitor.record_profile(
            operation="test_op",
            duration_ms=1000.0,
            memory_rss=1024*1024
        )
        
        # Performance monitor should record metrics
        assert len(monitor._profiles) == 1
    
    async def test_model_performance_alerting_integration(self):
        """Test integration between model performance monitor and alerting."""
        model_monitor = ModelPerformanceMonitor()
        
        # Add threshold that will be violated
        threshold = AlertThreshold(
            metric_name="accuracy",
            min_value=0.8,
            severity=AlertSeverity.WARNING
        )
        model_monitor.add_alert_threshold("model_123", threshold)
        
        with patch.object(model_monitor, '_trigger_alert') as mock_trigger:
            # Record metric that violates threshold
            model_monitor.record_prediction_metrics(
                "model_123",
                {"accuracy": 0.75},  # Below threshold
                1000
            )
            
            mock_trigger.assert_called_once()
            alert = mock_trigger.call_args[0][0]
            assert alert.model_id == "model_123"
            assert alert.current_value == 0.75


class TestMonitoringConfiguration:
    """Test monitoring configuration and settings integration."""
    
    def test_metrics_collector_with_custom_size(self):
        """Test metrics collector with custom deque size."""
        collector = MetricsCollector(max_metrics=500)
        
        # Record more metrics than limit
        for i in range(600):
            collector.record_metric("test", i)
        
        assert len(collector._metrics) == 500
    
    def test_health_checker_with_default_checks(self):
        """Test health checker with default system checks."""
        health_checker = HealthChecker()
        
        # Register default checks
        health_checker.register_memory_check()
        health_checker.register_disk_check()
        
        assert "memory" in health_checker._checks
        assert "disk" in health_checker._checks
    
    @patch('anomaly_detection.infrastructure.config.settings.get_settings')
    def test_monitoring_settings_integration(self, mock_get_settings):
        """Test monitoring components with settings integration."""
        mock_settings = Mock()
        mock_settings.monitoring.enable_metrics = True
        mock_settings.monitoring.metrics_port = 9090
        mock_get_settings.return_value = mock_settings
        
        # Test that components can access monitoring settings
        collector = get_metrics_collector()
        assert collector is not None  # Settings don't prevent creation