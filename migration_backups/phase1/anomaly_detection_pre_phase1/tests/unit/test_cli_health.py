"""Unit tests for CLI health monitoring commands."""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock, AsyncMock
from typer.testing import CliRunner

from anomaly_detection.cli_new.commands.health import app


class TestHealthCommands:
    """Test cases for health monitoring CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        
        # Sample health report data
        self.sample_health_report = Mock()
        self.sample_health_report.overall_status = Mock()
        self.sample_health_report.overall_status.value = 'healthy'
        self.sample_health_report.overall_score = 95.5
        self.sample_health_report.uptime_seconds = 3600
        self.sample_health_report.timestamp = Mock()
        self.sample_health_report.timestamp.strftime.return_value = '2023-01-01 12:00:00'
        self.sample_health_report.to_dict.return_value = {
            'overall_status': 'healthy',
            'overall_score': 95.5,
            'uptime_seconds': 3600,
            'metrics': [],
            'active_alerts': [],
            'recommendations': []
        }
        
        # Sample metrics
        self.sample_metric = Mock()
        self.sample_metric.name = 'cpu_usage'
        self.sample_metric.value = 45.2
        self.sample_metric.unit = '%'
        self.sample_metric.status = Mock()
        self.sample_metric.status.value = 'healthy'
        self.sample_metric.threshold_warning = 80.0
        self.sample_metric.threshold_critical = 95.0
        
        self.sample_health_report.metrics = [self.sample_metric]
        self.sample_health_report.active_alerts = []
        self.sample_health_report.recommendations = ['Monitor CPU usage closely']
        self.sample_health_report.performance_summary = {'avg_response_time': 150}
        
        # Sample alert
        self.sample_alert = Mock()
        self.sample_alert.severity = Mock()
        self.sample_alert.severity.value = 'warning'
        self.sample_alert.title = 'High Memory Usage'
        self.sample_alert.message = 'Memory usage is above threshold'
        self.sample_alert.metric_name = 'memory_usage'
        self.sample_alert.current_value = 85.0
        self.sample_alert.threshold_value = 80.0
        self.sample_alert.timestamp = Mock()
        self.sample_alert.timestamp.strftime.return_value = '12:00:00'
        self.sample_alert.resolved = False
        self.sample_alert.to_dict.return_value = {
            'severity': 'warning',
            'title': 'High Memory Usage',
            'message': 'Memory usage is above threshold',
            'timestamp': '2023-01-01T12:00:00'
        }
        
        # Sample performance data
        self.sample_performance_summary = {
            'response_time_stats': {
                'avg_ms': 150.5,
                'median_ms': 120.0,
                'min_ms': 50.0,
                'max_ms': 500.0,
                'p95_ms': 350.0,
                'p99_ms': 450.0
            },
            'error_stats': {
                'total_errors': 5,
                'error_rate_percent': 2.5,
                'recent_errors': 2
            },
            'throughput_stats': {
                'avg_rps': 100.5,
                'max_rps': 200.0,
                'min_rps': 50.0
            },
            'data_points': 1000
        }
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_status_command_success(self, mock_asyncio_run, mock_health_service_class):
        """Test status command with successful health check."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        async def mock_health_check():
            mock_health_service.get_health_report.return_value = self.sample_health_report
            return self.sample_health_report
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_health_check())
        
        # Run command
        result = self.runner.invoke(app, ['status'])
        
        # Assertions
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()
        assert "System Health: HEALTHY" in result.stdout
        assert "95.5/100" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_status_command_detailed(self, mock_asyncio_run, mock_health_service_class):
        """Test status command with detailed output."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        # Add active alerts to health report
        self.sample_health_report.active_alerts = [self.sample_alert]
        
        async def mock_health_check():
            mock_health_service.get_health_report.return_value = self.sample_health_report
            return self.sample_health_report
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_health_check())
        
        # Run command with detailed flag
        result = self.runner.invoke(app, ['status', '--detailed'])
        
        # Assertions
        assert result.exit_code == 0
        assert "Health Metrics" in result.stdout
        assert "cpu_usage" in result.stdout
        assert "Active Alerts" in result.stdout
        assert "High Memory Usage" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_status_command_json_output(self, mock_asyncio_run, mock_health_service_class):
        """Test status command with JSON output."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        async def mock_health_check():
            mock_health_service.get_health_report.return_value = self.sample_health_report
            return self.sample_health_report
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_health_check())
        
        # Run command with JSON output
        result = self.runner.invoke(app, ['status', '--json'])
        
        # Assertions
        assert result.exit_code == 0
        # Should contain JSON output
        try:
            json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    def test_status_command_import_error(self):
        """Test status command when health monitoring components are not available."""
        with patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService', side_effect=ImportError):
            result = self.runner.invoke(app, ['status'])
            
            assert result.exit_code == 1
            assert "Health monitoring components not available" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_status_command_health_check_failure(self, mock_asyncio_run, mock_health_service_class):
        """Test status command when health check fails."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        mock_asyncio_run.side_effect = Exception("Health check failed")
        
        # Run command
        result = self.runner.invoke(app, ['status'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Health check failed" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_monitor_command_success(self, mock_asyncio_run, mock_health_service_class):
        """Test monitor command with successful monitoring."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        async def mock_monitor():
            # Simulate monitoring loop with KeyboardInterrupt to stop
            raise KeyboardInterrupt()
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_monitor())
        
        # Run command
        result = self.runner.invoke(app, ['monitor', '--interval', '5', '--duration', '10'])
        
        # Assertions
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_monitor_command_keyboard_interrupt(self, mock_asyncio_run, mock_health_service_class):
        """Test monitor command with keyboard interrupt."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        mock_asyncio_run.side_effect = KeyboardInterrupt()
        
        # Run command
        result = self.runner.invoke(app, ['monitor'])
        
        # Assertions
        assert result.exit_code == 0
        assert "Health monitoring stopped" in result.stdout
    
    def test_monitor_command_import_error(self):
        """Test monitor command when health monitoring components are not available."""
        with patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService', side_effect=ImportError):
            result = self.runner.invoke(app, ['monitor'])
            
            assert result.exit_code == 1
            assert "Health monitoring components not available" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_monitor_command_failure(self, mock_asyncio_run, mock_health_service_class):
        """Test monitor command when monitoring fails."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        mock_asyncio_run.side_effect = Exception("Monitoring failed")
        
        # Run command
        result = self.runner.invoke(app, ['monitor'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Health monitoring failed" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('anomaly_detection.cli_new.commands.health.AlertSeverity')
    @patch('asyncio.run')
    def test_alerts_command_success(self, mock_asyncio_run, mock_alert_severity, mock_health_service_class):
        """Test alerts command with successful alert listing."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.alert_manager = Mock()
        
        async def mock_get_alerts():
            mock_health_service.alert_manager.get_active_alerts.return_value = [self.sample_alert]
            return [self.sample_alert]
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_get_alerts())
        
        # Run command
        result = self.runner.invoke(app, ['alerts'])
        
        # Assertions
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()
        assert "Active Alerts" in result.stdout
        assert "High Memory Usage" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('anomaly_detection.cli_new.commands.health.AlertSeverity')
    @patch('asyncio.run')
    def test_alerts_command_with_severity_filter(self, mock_asyncio_run, mock_alert_severity, mock_health_service_class):
        """Test alerts command with severity filter."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.alert_manager = Mock()
        
        mock_alert_severity.return_value = Mock()
        
        async def mock_get_alerts():
            mock_health_service.alert_manager.get_active_alerts.return_value = [self.sample_alert]
            return [self.sample_alert]
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_get_alerts())
        
        # Run command with severity filter
        result = self.runner.invoke(app, ['alerts', '--severity', 'warning'])
        
        # Assertions
        assert result.exit_code == 0
        mock_alert_severity.assert_called_once_with('warning')
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_alerts_command_history(self, mock_asyncio_run, mock_health_service_class):
        """Test alerts command with history flag."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        async def mock_get_alert_history():
            mock_health_service.get_alert_history.return_value = [self.sample_alert]
            return [self.sample_alert]
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_get_alert_history())
        
        # Run command with history flag
        result = self.runner.invoke(app, ['alerts', '--history', '--hours', '12'])
        
        # Assertions
        assert result.exit_code == 0
        assert "Alert History" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_alerts_command_json_output(self, mock_asyncio_run, mock_health_service_class):
        """Test alerts command with JSON output."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.alert_manager = Mock()
        
        async def mock_get_alerts():
            mock_health_service.alert_manager.get_active_alerts.return_value = [self.sample_alert]
            return [self.sample_alert]
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_get_alerts())
        
        # Run command with JSON output
        result = self.runner.invoke(app, ['alerts', '--json'])
        
        # Assertions
        assert result.exit_code == 0
        # Should contain JSON output
        try:
            json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_alerts_command_no_alerts(self, mock_asyncio_run, mock_health_service_class):
        """Test alerts command when no alerts are found."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.alert_manager = Mock()
        
        async def mock_get_alerts():
            mock_health_service.alert_manager.get_active_alerts.return_value = []
            return []
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_get_alerts())
        
        # Run command
        result = self.runner.invoke(app, ['alerts'])
        
        # Assertions
        assert result.exit_code == 0
        assert "No active alerts found" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('anomaly_detection.cli_new.commands.health.AlertSeverity')
    def test_alerts_command_invalid_severity(self, mock_alert_severity, mock_health_service_class):
        """Test alerts command with invalid severity."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        mock_alert_severity.side_effect = ValueError("Invalid severity")
        
        # Run command with invalid severity
        result = self.runner.invoke(app, ['alerts', '--severity', 'invalid'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Invalid severity: invalid" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_performance_command_success(self, mock_asyncio_run, mock_health_service_class):
        """Test performance command with successful metrics retrieval."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.performance_tracker = Mock()
        
        async def mock_get_performance():
            mock_health_service.performance_tracker.get_performance_summary.return_value = self.sample_performance_summary
            return self.sample_performance_summary
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_get_performance())
        
        # Run command
        result = self.runner.invoke(app, ['performance'])
        
        # Assertions
        assert result.exit_code == 0
        assert "Performance Metrics Summary" in result.stdout
        assert "Response Time Statistics" in result.stdout
        assert "150.5 ms" in result.stdout  # avg_ms
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_performance_command_json_output(self, mock_asyncio_run, mock_health_service_class):
        """Test performance command with JSON output."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.performance_tracker = Mock()
        
        async def mock_get_performance():
            mock_health_service.performance_tracker.get_performance_summary.return_value = self.sample_performance_summary
            return self.sample_performance_summary
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_get_performance())
        
        # Run command with JSON output
        result = self.runner.invoke(app, ['performance', '--json'])
        
        # Assertions
        assert result.exit_code == 0
        # Should contain JSON output
        try:
            json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_performance_command_no_data(self, mock_asyncio_run, mock_health_service_class):
        """Test performance command when no performance data is available."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.performance_tracker = Mock()
        
        async def mock_get_performance():
            mock_health_service.performance_tracker.get_performance_summary.return_value = {}
            return {}
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_get_performance())
        
        # Run command
        result = self.runner.invoke(app, ['performance'])
        
        # Assertions
        assert result.exit_code == 0
        assert "No performance data available" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_start_monitoring_command_success(self, mock_asyncio_run, mock_health_service_class):
        """Test start_monitoring command with successful service start."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.monitoring_enabled = False  # Will exit loop immediately
        
        async def mock_start_monitoring():
            mock_health_service.start_monitoring.return_value = None
            # Simulate KeyboardInterrupt to stop
            raise KeyboardInterrupt()
        
        mock_asyncio_run.side_effect = lambda coro: asyncio.run(mock_start_monitoring())
        
        # Run command
        result = self.runner.invoke(app, ['start-monitoring', '--interval', '30'])
        
        # Assertions
        assert result.exit_code == 0
        mock_health_service_class.assert_called_once_with(check_interval=30)
        mock_asyncio_run.assert_called_once()
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_start_monitoring_command_keyboard_interrupt(self, mock_asyncio_run, mock_health_service_class):
        """Test start_monitoring command with keyboard interrupt."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        mock_asyncio_run.side_effect = KeyboardInterrupt()
        
        # Run command
        result = self.runner.invoke(app, ['start-monitoring'])
        
        # Assertions
        assert result.exit_code == 0
        assert "Health monitoring stopped" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    def test_thresholds_command_show_all(self, mock_health_service_class):
        """Test thresholds command showing all thresholds."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.thresholds = {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 85.0, 'critical': 98.0}
        }
        
        # Run command
        result = self.runner.invoke(app, ['thresholds'])
        
        # Assertions
        assert result.exit_code == 0
        assert "Health Monitoring Thresholds" in result.stdout
        assert "Cpu Usage" in result.stdout
        assert "80.0" in result.stdout
        assert "95.0" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    def test_thresholds_command_show_specific(self, mock_health_service_class):
        """Test thresholds command showing specific metric thresholds."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.thresholds = {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0}
        }
        
        # Run command
        result = self.runner.invoke(app, ['thresholds', '--metric', 'cpu_usage'])
        
        # Assertions
        assert result.exit_code == 0
        assert "Thresholds for cpu_usage" in result.stdout
        assert "Warning: 80.0" in result.stdout
        assert "Critical: 95.0" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    def test_thresholds_command_metric_not_found(self, mock_health_service_class):
        """Test thresholds command with non-existent metric."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.thresholds = {}
        
        # Run command
        result = self.runner.invoke(app, ['thresholds', '--metric', 'nonexistent'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Metric 'nonexistent' not found" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    def test_thresholds_command_set_thresholds_success(self, mock_health_service_class):
        """Test thresholds command setting new thresholds."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        # Run command
        result = self.runner.invoke(app, [
            'thresholds',
            '--metric', 'cpu_usage',
            '--set-warning', '75.0',
            '--set-critical', '90.0'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_health_service.set_threshold.assert_called_once_with('cpu_usage', 75.0, 90.0)
        assert "Updated thresholds for cpu_usage" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    def test_thresholds_command_invalid_thresholds(self, mock_health_service_class):
        """Test thresholds command with invalid threshold values."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        # Run command with warning >= critical
        result = self.runner.invoke(app, [
            'thresholds',
            '--metric', 'cpu_usage',
            '--set-warning', '95.0',
            '--set-critical', '90.0'
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Warning threshold must be less than critical threshold" in result.stdout
    
    def test_thresholds_command_import_error(self):
        """Test thresholds command when health monitoring components are not available."""
        with patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService', side_effect=ImportError):
            result = self.runner.invoke(app, ['thresholds'])
            
            assert result.exit_code == 1
            assert "Health monitoring components not available" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    def test_thresholds_command_error(self, mock_health_service_class):
        """Test thresholds command when service raises error."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        mock_health_service.thresholds = Mock()
        mock_health_service.thresholds.__getitem__.side_effect = Exception("Service error")
        
        # Run command
        result = self.runner.invoke(app, ['thresholds'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Failed to manage thresholds" in result.stdout
    
    def test_all_commands_import_error_handling(self):
        """Test that all commands handle ImportError correctly."""
        commands = ['status', 'monitor', 'alerts', 'performance', 'start-monitoring', 'thresholds']
        
        for command in commands:
            with patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService', side_effect=ImportError):
                result = self.runner.invoke(app, [command])
                
                assert result.exit_code == 1
                assert "Health monitoring components not available" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_alerts_command_error(self, mock_asyncio_run, mock_health_service_class):
        """Test alerts command when service raises error."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        mock_asyncio_run.side_effect = Exception("Service error")
        
        # Run command
        result = self.runner.invoke(app, ['alerts'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Failed to get alerts" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_performance_command_error(self, mock_asyncio_run, mock_health_service_class):
        """Test performance command when service raises error."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        mock_asyncio_run.side_effect = Exception("Service error")
        
        # Run command
        result = self.runner.invoke(app, ['performance'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Failed to get performance metrics" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.health.HealthMonitoringService')
    @patch('asyncio.run')
    def test_start_monitoring_command_error(self, mock_asyncio_run, mock_health_service_class):
        """Test start_monitoring command when service raises error."""
        # Setup mocks
        mock_health_service = Mock()
        mock_health_service_class.return_value = mock_health_service
        
        mock_asyncio_run.side_effect = Exception("Service error")
        
        # Run command
        result = self.runner.invoke(app, ['start-monitoring'])
        
        # Assertions
        assert result.exit_code == 1
        assert "Failed to start health monitoring" in result.stdout