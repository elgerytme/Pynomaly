"""Tests for production monitoring infrastructure."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.pynomaly.infrastructure.monitoring.health_service import HealthStatus
from src.pynomaly.infrastructure.monitoring.production_monitor import (
    LogLevel,
    ProductionMonitor,
    get_monitor,
    init_monitor,
    log_info,
)


class TestProductionMonitor:
    """Test production monitoring functionality."""

    def test_monitor_initialization(self):
        """Test monitor initialization with different settings."""
        monitor = ProductionMonitor(
            log_level=LogLevel.DEBUG,
            enable_performance_monitoring=True,
            enable_health_checks=True,
        )

        assert monitor.log_level == LogLevel.DEBUG
        assert monitor.enable_performance_monitoring is True
        assert monitor.enable_health_checks is True
        assert monitor.health_service is not None
        assert monitor.performance_monitor is not None

    def test_basic_logging(self):
        """Test basic logging functionality."""
        monitor = ProductionMonitor(log_level=LogLevel.INFO)

        # Test different log levels
        monitor.info("Test info message", component="test", operation="logging")
        monitor.warning("Test warning", component="test")
        monitor.error("Test error", component="test")
        monitor.debug("Test debug", component="test")  # Should be filtered out

        # Check log entries
        assert len(monitor.log_entries) == 3  # Debug should be filtered
        assert monitor.log_entries[0].level == LogLevel.INFO
        assert monitor.log_entries[1].level == LogLevel.WARNING
        assert monitor.log_entries[2].level == LogLevel.ERROR

    def test_error_reporting(self):
        """Test error reporting and tracking."""
        monitor = ProductionMonitor()

        # Create test exception
        try:
            raise ValueError("Test error")
        except ValueError as e:
            error_id = monitor.report_error(
                e,
                "test_component",
                "test_operation",
                context={"data": "test"},
                severity="high",
            )

        assert error_id
        assert len(monitor.error_reports) == 1

        error_report = monitor.error_reports[0]
        assert error_report.error_type == "ValueError"
        assert error_report.error_message == "Test error"
        assert error_report.component == "test_component"
        assert error_report.operation == "test_operation"
        assert error_report.severity == "high"
        assert not error_report.resolved

        # Test error resolution
        resolved = monitor.resolve_error(error_id, "Issue fixed")
        assert resolved
        assert error_report.resolved
        assert error_report.resolution_notes == "Issue fixed"

    def test_audit_logging(self):
        """Test audit event logging."""
        monitor = ProductionMonitor()

        event_id = monitor.audit_event(
            event_type="access",
            operation="data_access",
            resource="dataset_123",
            action="read",
            user_id="user_456",
            outcome="success",
        )

        assert event_id
        assert len(monitor.audit_events) == 1

        audit_event = monitor.audit_events[0]
        assert audit_event.event_type == "access"
        assert audit_event.operation == "data_access"
        assert audit_event.resource == "dataset_123"
        assert audit_event.action == "read"
        assert audit_event.user_id == "user_456"
        assert audit_event.outcome == "success"

    def test_operation_monitoring(self):
        """Test operation monitoring context manager."""
        monitor = ProductionMonitor()

        # Test successful operation
        with monitor.monitor_operation(
            "test_operation", "test_component"
        ) as request_id:
            assert request_id
            time.sleep(0.01)  # Small delay for timing

        # Check logs were created
        assert len(monitor.log_entries) >= 2  # Start and completion
        start_log = monitor.log_entries[0]
        end_log = monitor.log_entries[1]

        assert start_log.operation == "test_operation"
        assert start_log.component == "test_component"
        assert "Starting" in start_log.message

        assert end_log.operation == "test_operation"
        assert end_log.status == "success"
        assert "Completed" in end_log.message
        assert end_log.duration_ms > 0

    def test_operation_monitoring_with_error(self):
        """Test operation monitoring when error occurs."""
        monitor = ProductionMonitor()

        # Test operation with error
        with pytest.raises(ValueError):
            with monitor.monitor_operation("test_operation", "test_component"):
                raise ValueError("Test error")

        # Check error was reported and logged
        assert len(monitor.error_reports) == 1
        error_log = [log for log in monitor.log_entries if log.level == LogLevel.ERROR][
            0
        ]
        assert error_log.status == "error"
        assert "Failed" in error_log.message

    @pytest.mark.asyncio
    async def test_async_operation_monitoring(self):
        """Test async operation monitoring."""
        monitor = ProductionMonitor(enable_health_checks=False)  # Disable to avoid deps

        async with monitor.monitor_async_operation(
            "async_test", "test_component"
        ) as request_id:
            assert request_id
            await asyncio.sleep(0.01)

        # Verify logs
        assert len(monitor.log_entries) >= 2
        assert "async" in monitor.log_entries[0].message.lower()

    def test_metrics_summary(self):
        """Test metrics summary generation."""
        monitor = ProductionMonitor()

        # Generate some test data
        monitor.info("Test message 1", component="comp1")
        monitor.warning("Test warning", component="comp1")
        monitor.error("Test error", component="comp2")

        # Create test error
        try:
            raise RuntimeError("Test")
        except RuntimeError as e:
            monitor.report_error(e, "comp1", "test_op")

        # Create audit event
        monitor.audit_event("access", "read", "resource1", "view")

        summary = monitor.get_metrics_summary()

        assert summary["logs"]["total"] == 3
        assert summary["logs"]["by_level"]["info"] == 1
        assert summary["logs"]["by_level"]["warning"] == 1
        assert summary["logs"]["by_level"]["error"] == 1

        assert summary["errors"]["total"] == 1
        assert summary["errors"]["by_type"]["RuntimeError"] == 1
        assert summary["errors"]["unresolved"] == 1

        assert summary["audit_events"]["total"] == 1

    def test_log_export(self):
        """Test log export functionality."""
        monitor = ProductionMonitor()

        # Generate test logs
        monitor.info("Test 1", component="comp1", operation="op1")
        monitor.warning("Test 2", component="comp2", operation="op2")

        # Export all logs
        exported = monitor.export_logs()
        assert len(exported) == 2
        assert exported[0]["message"] == "Test 1"
        assert exported[1]["message"] == "Test 2"

        # Export filtered logs
        filtered = monitor.export_logs(level=LogLevel.WARNING)
        assert len(filtered) == 1
        assert filtered[0]["message"] == "Test 2"

        component_filtered = monitor.export_logs(component="comp1")
        assert len(component_filtered) == 1
        assert component_filtered[0]["message"] == "Test 1"

    @pytest.mark.asyncio
    async def test_health_check_integration(self):
        """Test health check integration."""
        # Mock health service to avoid external dependencies
        monitor = ProductionMonitor(enable_health_checks=True)

        # Mock the health service
        mock_checks = {"test_check": Mock()}
        mock_checks["test_check"].status = HealthStatus.HEALTHY
        mock_checks["test_check"].message = "All good"
        mock_checks["test_check"].duration_ms = 5.0
        mock_checks["test_check"].details = {}

        with patch.object(
            monitor.health_service,
            "perform_comprehensive_health_check",
            return_value=mock_checks,
        ):
            result = await monitor.health_check()

        assert result["status"] == "healthy"
        assert "test_check" in result["checks"]
        assert result["checks"]["test_check"]["status"] == "healthy"

    def test_global_monitor_functions(self):
        """Test global monitor convenience functions."""
        # Initialize global monitor
        init_monitor(log_level=LogLevel.DEBUG)

        # Test convenience functions
        log_info("Global info test", component="global")

        monitor = get_monitor()
        assert len(monitor.log_entries) >= 1
        assert monitor.log_entries[-1].message == "Global info test"

    def test_memory_limits(self):
        """Test memory limits for stored data."""
        monitor = ProductionMonitor(
            max_log_entries=2, max_error_reports=2, max_audit_events=2
        )

        # Test log entry limits
        for i in range(5):
            monitor.info(f"Message {i}")

        assert len(monitor.log_entries) == 2
        assert monitor.log_entries[0].message == "Message 3"  # Should keep latest
        assert monitor.log_entries[1].message == "Message 4"

        # Test error report limits
        for i in range(5):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                monitor.report_error(e, "comp", "op")

        assert len(monitor.error_reports) == 2

        # Test audit event limits
        for i in range(5):
            monitor.audit_event("test", "op", f"resource{i}", "action")

        assert len(monitor.audit_events) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
