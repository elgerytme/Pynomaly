"""Integration tests for monitoring and observability components."""

import pytest
import asyncio
import time
from typing import Dict, Any
from datetime import datetime, timedelta

from anomaly_detection.infrastructure.monitoring import (
    get_metrics_collector,
    get_health_checker,
    get_performance_monitor,
    get_monitoring_dashboard
)
from anomaly_detection.infrastructure.monitoring.health_checker import HealthStatus
from anomaly_detection.domain.services.detection_service import DetectionService


class TestMonitoringIntegration:
    """Integration tests for monitoring components working together."""
    
    @pytest.fixture(autouse=True)
    def setup_monitoring(self, metrics_collector, health_checker, performance_monitor):
        """Set up monitoring for each test."""
        # Clear any existing data
        metrics_collector.clear_all_metrics()
        performance_monitor.clear_profiles()
        yield
        # Cleanup after each test
        metrics_collector.clear_all_metrics()
        performance_monitor.clear_profiles()
    
    def test_metrics_collector_basic_operations(self, metrics_collector):
        """Test basic metrics collection operations."""
        # Record various types of metrics
        metrics_collector.record_metric("test_counter", 1.0, {"type": "test"}, "count")
        metrics_collector.increment_counter("api_requests", 5, {"endpoint": "/detect"})
        metrics_collector.set_gauge("memory_usage", 1024.5, {"unit": "mb"})
        metrics_collector.record_timing("operation_duration", 150.0, {"operation": "detection"})
        
        # Get summary
        summary = metrics_collector.get_summary_stats()
        
        assert summary["total_metrics"] == 4
        assert "counters" in summary
        assert "gauges" in summary
        assert "timing_stats" in summary
        
        # Check specific metrics were recorded
        assert summary["counters"]["api_requests_{'endpoint': '/detect'}"] == 5
        assert summary["gauges"]["memory_usage_{'unit': 'mb'}"] == 1024.5
        assert "test_counter" in summary["timing_stats"]
        assert "operation_duration" in summary["timing_stats"]
    
    def test_performance_monitor_context_manager(self, performance_monitor):
        """Test performance monitoring with context manager."""
        # Monitor an operation
        with performance_monitor.create_context("test_operation", track_memory=True) as ctx:
            # Simulate some work
            time.sleep(0.1)
            
            # Increment some counters
            ctx.increment_counter("io_operations", 3)
            ctx.increment_counter("cache_hits", 2)
            
            # Simulate more work
            time.sleep(0.05)
        
        # Check that profile was recorded
        profiles = performance_monitor.get_recent_profiles(operation="test_operation", limit=1)
        assert len(profiles) == 1
        
        profile = profiles[0]
        assert profile.operation == "test_operation"
        assert profile.total_duration_ms > 100  # At least 150ms
        assert profile.success is True
        assert profile.io_operations == 3
        assert profile.cache_hits == 2
        
        # Check operation stats
        stats = performance_monitor.get_operation_stats("test_operation")
        assert "test_operation" in stats
        
        op_stats = stats["test_operation"]
        assert op_stats["count"] == 1
        assert op_stats["success_count"] == 1
        assert op_stats["error_count"] == 0
        assert op_stats["avg_duration_ms"] > 100
    
    def test_performance_monitor_error_handling(self, performance_monitor):
        """Test performance monitoring with errors."""
        # Monitor an operation that fails
        try:
            with performance_monitor.create_context("failing_operation") as ctx:
                time.sleep(0.05)
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected
        
        # Check that error was recorded
        profiles = performance_monitor.get_recent_profiles(operation="failing_operation", limit=1)
        assert len(profiles) == 1
        
        profile = profiles[0]
        assert profile.operation == "failing_operation"
        assert profile.success is False
        assert profile.error_message == "Simulated error"
        
        # Check operation stats
        stats = performance_monitor.get_operation_stats("failing_operation")
        op_stats = stats["failing_operation"]
        assert op_stats["error_count"] == 1
        assert op_stats["success_count"] == 0
    
    async def test_health_checker_all_checks(self, health_checker):
        """Test running all health checks."""
        # Run all checks
        results = await health_checker.run_all_checks(force=True)
        
        assert len(results) > 0
        
        # Should have basic checks
        expected_checks = ["algorithms", "model_repository", "memory", "disk"]
        
        for check_name in expected_checks:
            if check_name in results:
                result = results[check_name]
                assert result.name == check_name
                assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
                assert isinstance(result.duration_ms, float)
                assert result.duration_ms >= 0
                assert isinstance(result.message, str)
                assert len(result.message) > 0
        
        # Get health summary
        summary = health_checker.get_health_summary()
        
        assert "overall_status" in summary
        assert "checks" in summary
        assert "status_counts" in summary
        assert "total_checks" in summary
        
        # Overall status should be one of the valid statuses
        valid_statuses = ["healthy", "degraded", "unhealthy", "unknown"]
        assert summary["overall_status"] in valid_statuses
    
    async def test_health_checker_individual_checks(self, health_checker):
        """Test individual health checks."""
        # Test algorithm health check (should be healthy)
        result = await health_checker.run_check("algorithms")
        
        assert result is not None
        assert result.name == "algorithms"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]  # Should not be unhealthy
        assert "algorithms" in result.message or "functional" in result.message
        
        # Test model repository check
        result = await health_checker.run_check("model_repository")
        
        assert result is not None
        assert result.name == "model_repository"
        # This might be degraded if directory doesn't exist, but shouldn't crash
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
    
    def test_monitoring_dashboard_integration(self, monitoring_dashboard, metrics_collector, performance_monitor):
        """Test monitoring dashboard data aggregation."""
        # Generate some test data
        metrics_collector.record_metric("detection_count", 5, {"algorithm": "iforest"})
        metrics_collector.record_metric("anomalies_found", 2, {"algorithm": "iforest"})
        
        # Add performance data
        with performance_monitor.create_context("test_detection") as ctx:
            time.sleep(0.05)
            ctx.increment_counter("network_requests", 1)
        
        # Get dashboard data
        trends = monitoring_dashboard.get_performance_trends(hours=1)
        assert "trends" in trends
        assert "period_hours" in trends
        assert trends["period_hours"] == 1
        
        alerts = monitoring_dashboard.get_alert_summary()
        assert "total_alerts" in alerts
        assert "alerts" in alerts
        assert isinstance(alerts["total_alerts"], int)
        
        operations = monitoring_dashboard.get_operation_breakdown()
        assert "total_operations_monitored" in operations
        assert "top_operations" in operations
    
    async def test_monitoring_dashboard_summary(self, monitoring_dashboard):
        """Test dashboard summary generation."""
        summary = await monitoring_dashboard.get_dashboard_summary()
        
        # Check all expected fields are present
        expected_fields = [
            "overall_health_status", "healthy_checks", "degraded_checks",
            "unhealthy_checks", "total_operations", "operations_last_hour",
            "avg_response_time_ms", "success_rate", "active_alerts",
            "recent_errors", "slow_operations", "generated_at"
        ]
        
        for field in expected_fields:
            assert hasattr(summary, field), f"Missing field: {field}"
        
        assert summary.overall_health_status in ["healthy", "degraded", "unhealthy", "unknown"]
        assert isinstance(summary.total_operations, int)
        assert isinstance(summary.success_rate, float)
        assert 0.0 <= summary.success_rate <= 1.0
        assert isinstance(summary.generated_at, datetime)
    
    def test_monitoring_with_detection_service(self, detection_service: DetectionService, 
                                             test_data: Dict[str, Any], 
                                             metrics_collector, performance_monitor):
        """Test monitoring integration with actual detection service."""
        data = test_data['data_only']
        
        # Perform detection (this should trigger monitoring)
        result = detection_service.detect_anomalies(
            data=data,
            algorithm="iforest",
            contamination=0.1
        )
        
        assert result.success is True
        
        # Check that metrics were recorded
        summary = metrics_collector.get_summary_stats()
        assert summary["total_metrics"] > 0
        
        # Check that performance was monitored
        perf_summary = performance_monitor.get_performance_summary()
        # May or may not have profiles depending on implementation,
        # but should not crash
        assert "total_profiles" in perf_summary
    
    def test_monitoring_metrics_integration(self, metrics_collector, performance_monitor):
        """Test integration between metrics collector and performance monitor."""
        # Start an operation
        operation_id = metrics_collector.start_operation("integration_test")
        
        # Simulate work
        time.sleep(0.1)
        
        # End operation
        duration = metrics_collector.end_operation(operation_id, success=True)
        
        assert duration > 90  # Should be at least 100ms
        
        # Check metrics were recorded
        summary = metrics_collector.get_summary_stats()
        assert summary["total_metrics"] > 0
        
        # Check performance metrics
        perf_metrics = metrics_collector.get_performance_metrics(
            operation="integration_test",
            limit=1
        )
        assert len(perf_metrics) == 1
        assert perf_metrics[0].operation == "integration_test"
        assert perf_metrics[0].success is True
    
    def test_model_metrics_recording(self, metrics_collector):
        """Test model-specific metrics recording."""
        # Record model training metrics
        metrics_collector.record_model_metrics(
            model_id="test-model-001",
            algorithm="isolation_forest",
            operation="train",
            duration_ms=5000.0,
            success=True,
            samples_processed=1000,
            anomalies_detected=100,
            accuracy=0.92,
            precision=0.88,
            recall=0.94,
            f1_score=0.91
        )
        
        # Get model metrics
        model_metrics = metrics_collector.get_model_metrics(
            model_id="test-model-001",
            limit=1
        )
        
        assert len(model_metrics) == 1
        metric = model_metrics[0]
        
        assert metric.model_id == "test-model-001"
        assert metric.algorithm == "isolation_forest"
        assert metric.operation == "train"
        assert metric.duration_ms == 5000.0
        assert metric.success is True
        assert metric.samples_processed == 1000
        assert metric.anomalies_detected == 100
        assert metric.accuracy == 0.92
        assert metric.precision == 0.88
        assert metric.recall == 0.94
        assert metric.f1_score == 0.91
        
        # Check that individual metrics were also recorded
        summary = metrics_collector.get_summary_stats()
        assert summary["total_metrics"] > 5  # Should have recorded multiple metrics
    
    def test_cleanup_operations(self, metrics_collector, performance_monitor):
        """Test cleanup of old monitoring data."""
        # Add some test data
        metrics_collector.record_metric("test_metric_1", 1.0)
        metrics_collector.record_metric("test_metric_2", 2.0)
        
        with performance_monitor.create_context("cleanup_test"):
            time.sleep(0.01)
        
        # Check data exists
        assert metrics_collector.get_summary_stats()["total_metrics"] == 2
        assert len(performance_monitor.get_recent_profiles(limit=100)) == 1
        
        # Cleanup metrics (won't remove recent ones in normal cleanup)
        removed = metrics_collector.cleanup_old_metrics()
        # Recent metrics won't be removed, so count should be 0
        assert removed == 0
        
        # Clear all for testing
        metrics_collector.clear_all_metrics()
        performance_monitor.clear_profiles()
        
        # Verify cleanup
        assert metrics_collector.get_summary_stats()["total_metrics"] == 0
        assert len(performance_monitor.get_recent_profiles(limit=100)) == 0
    
    async def test_concurrent_monitoring_operations(self, metrics_collector, performance_monitor):
        """Test concurrent monitoring operations."""
        import threading
        import concurrent.futures
        
        results = []
        errors = []
        
        def monitoring_operation(operation_id):
            try:
                # Record metrics
                metrics_collector.record_metric(f"concurrent_metric_{operation_id}", float(operation_id))
                
                # Monitor performance
                with performance_monitor.create_context(f"concurrent_op_{operation_id}"):
                    time.sleep(0.01 * operation_id)  # Variable sleep
                
                results.append(operation_id)
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(monitoring_operation, i) for i in range(1, 6)]
            concurrent.futures.wait(futures)
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert set(results) == {1, 2, 3, 4, 5}
        
        # Verify all metrics were recorded
        summary = metrics_collector.get_summary_stats()
        assert summary["total_metrics"] == 5
        
        # Verify all performance profiles were recorded
        profiles = performance_monitor.get_recent_profiles(limit=10)
        assert len(profiles) == 5
        
        # Each operation should have unique name
        operation_names = {p.operation for p in profiles}
        expected_names = {f"concurrent_op_{i}" for i in range(1, 6)}
        assert operation_names == expected_names