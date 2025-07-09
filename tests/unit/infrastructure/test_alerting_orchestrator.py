"""
Unit tests for the alerting orchestrator module.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from pynomaly.infrastructure.alerting.orchestrator import (
    AlertingOrchestrator,
    AlertTask,
    AlertTaskStatus,
    AlertSeverity,
    AlertChannel,
    AlertTaskMetrics,
    ChannelMetrics
)
from pynomaly.infrastructure.config.enhanced_config_loader import AlertingConfig
from pynomaly.shared.exceptions import (
    AlertingError,
    TaskExecutionError,
    ConfigurationError
)


class TestAlertSeverity:
    """Test AlertSeverity enum."""
    
    def test_all_severity_values(self):
        """Test all severity values are present."""
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlertChannel:
    """Test AlertChannel enum."""
    
    def test_all_channel_values(self):
        """Test all channel values are present."""
        assert AlertChannel.EMAIL.value == "email"
        assert AlertChannel.SMS.value == "sms"
        assert AlertChannel.SLACK.value == "slack"
        assert AlertChannel.WEBHOOK.value == "webhook"
        assert AlertChannel.DASHBOARD.value == "dashboard"


class TestAlertTaskStatus:
    """Test AlertTaskStatus enum."""
    
    def test_all_status_values(self):
        """Test all status values are present."""
        assert AlertTaskStatus.PENDING.value == "pending"
        assert AlertTaskStatus.PROCESSING.value == "processing"
        assert AlertTaskStatus.SENT.value == "sent"
        assert AlertTaskStatus.FAILED.value == "failed"
        assert AlertTaskStatus.RETRYING.value == "retrying"


class TestAlertTask:
    """Test AlertTask dataclass."""
    
    def test_default_values(self):
        """Test default values of AlertTask."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert message",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL]
        )
        
        assert task.task_id == "test-task"
        assert task.rule_id == "test-rule"
        assert task.message == "Test alert message"
        assert task.severity == AlertSeverity.HIGH
        assert task.channels == [AlertChannel.EMAIL]
        assert task.status == AlertTaskStatus.PENDING
        assert task.context == {}
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.error_message is None
        assert task.sent_channels == []
        assert task.failed_channels == []
        assert isinstance(task.created_at, datetime)
        assert task.started_at is None
        assert task.completed_at is None
    
    def test_custom_values(self):
        """Test custom values of AlertTask."""
        custom_time = datetime.now() - timedelta(hours=1)
        context = {"anomaly_count": 5, "source": "sensor_1"}
        
        task = AlertTask(
            task_id="custom-task",
            rule_id="custom-rule",
            message="Custom alert message",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            status=AlertTaskStatus.PROCESSING,
            context=context,
            retry_count=2,
            max_retries=5,
            error_message="Custom error",
            sent_channels=[AlertChannel.EMAIL],
            failed_channels=[AlertChannel.SLACK],
            created_at=custom_time,
            started_at=custom_time,
            completed_at=custom_time
        )
        
        assert task.task_id == "custom-task"
        assert task.rule_id == "custom-rule"
        assert task.message == "Custom alert message"
        assert task.severity == AlertSeverity.CRITICAL
        assert task.channels == [AlertChannel.EMAIL, AlertChannel.SLACK]
        assert task.status == AlertTaskStatus.PROCESSING
        assert task.context == context
        assert task.retry_count == 2
        assert task.max_retries == 5
        assert task.error_message == "Custom error"
        assert task.sent_channels == [AlertChannel.EMAIL]
        assert task.failed_channels == [AlertChannel.SLACK]
        assert task.created_at == custom_time
        assert task.started_at == custom_time
        assert task.completed_at == custom_time


class TestAlertTaskMetrics:
    """Test AlertTaskMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values of AlertTaskMetrics."""
        metrics = AlertTaskMetrics()
        
        assert metrics.total_alerts == 0
        assert metrics.sent_alerts == 0
        assert metrics.failed_alerts == 0
        assert metrics.retrying_alerts == 0
        assert metrics.active_alerts == 0
        assert metrics.avg_processing_time == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.alerts_by_severity == {}
        assert metrics.alerts_by_channel == {}
        assert isinstance(metrics.last_updated, datetime)
    
    def test_custom_values(self):
        """Test custom values of AlertTaskMetrics."""
        custom_time = datetime.now() - timedelta(hours=1)
        alerts_by_severity = {
            AlertSeverity.LOW: 100,
            AlertSeverity.MEDIUM: 50,
            AlertSeverity.HIGH: 20,
            AlertSeverity.CRITICAL: 5
        }
        alerts_by_channel = {
            AlertChannel.EMAIL: 120,
            AlertChannel.SLACK: 80,
            AlertChannel.SMS: 30
        }
        
        metrics = AlertTaskMetrics(
            total_alerts=1000,
            sent_alerts=950,
            failed_alerts=30,
            retrying_alerts=10,
            active_alerts=10,
            avg_processing_time=1.5,
            success_rate=95.0,
            alerts_by_severity=alerts_by_severity,
            alerts_by_channel=alerts_by_channel,
            last_updated=custom_time
        )
        
        assert metrics.total_alerts == 1000
        assert metrics.sent_alerts == 950
        assert metrics.failed_alerts == 30
        assert metrics.retrying_alerts == 10
        assert metrics.active_alerts == 10
        assert metrics.avg_processing_time == 1.5
        assert metrics.success_rate == 95.0
        assert metrics.alerts_by_severity == alerts_by_severity
        assert metrics.alerts_by_channel == alerts_by_channel
        assert metrics.last_updated == custom_time
    
    def test_get_success_rate(self):
        """Test success rate calculation."""
        metrics = AlertTaskMetrics(
            total_alerts=100,
            sent_alerts=95,
            failed_alerts=5
        )
        
        assert metrics.get_success_rate() == 95.0
        
        # Test with no alerts
        empty_metrics = AlertTaskMetrics()
        assert empty_metrics.get_success_rate() == 0.0


class TestChannelMetrics:
    """Test ChannelMetrics dataclass."""
    
    def test_default_values(self):
        """Test default values of ChannelMetrics."""
        metrics = ChannelMetrics()
        
        assert metrics.channels_configured == 0
        assert metrics.avg_send_time == 0.0
        assert metrics.channel_success_rates == {}
        assert metrics.channel_error_rates == {}
        assert metrics.rate_limits_hit == 0
        assert metrics.retry_attempts == 0
        assert isinstance(metrics.last_updated, datetime)
    
    def test_custom_values(self):
        """Test custom values of ChannelMetrics."""
        custom_time = datetime.now() - timedelta(hours=1)
        success_rates = {
            AlertChannel.EMAIL: 98.5,
            AlertChannel.SLACK: 95.0,
            AlertChannel.SMS: 92.0
        }
        error_rates = {
            AlertChannel.EMAIL: 1.5,
            AlertChannel.SLACK: 5.0,
            AlertChannel.SMS: 8.0
        }
        
        metrics = ChannelMetrics(
            channels_configured=3,
            avg_send_time=0.8,
            channel_success_rates=success_rates,
            channel_error_rates=error_rates,
            rate_limits_hit=5,
            retry_attempts=20,
            last_updated=custom_time
        )
        
        assert metrics.channels_configured == 3
        assert metrics.avg_send_time == 0.8
        assert metrics.channel_success_rates == success_rates
        assert metrics.channel_error_rates == error_rates
        assert metrics.rate_limits_hit == 5
        assert metrics.retry_attempts == 20
        assert metrics.last_updated == custom_time


class TestAlertingOrchestrator:
    """Test AlertingOrchestrator class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock alerting configuration."""
        config = Mock(spec=AlertingConfig)
        config.max_workers = 6
        config.max_retries = 3
        config.retry_delay = 2.0
        config.timeout = 30.0
        config.queue_size = 500
        config.rate_limit_per_minute = 100
        config.batch_size = 50
        return config
    
    @pytest.fixture
    def orchestrator(self, mock_config):
        """Create an alerting orchestrator instance."""
        with patch('pynomaly.infrastructure.alerting.orchestrator.StructuredLogger'):
            with patch('pynomaly.infrastructure.alerting.orchestrator.MetricsService'):
                return AlertingOrchestrator(config=mock_config)
    
    def test_init(self, orchestrator, mock_config):
        """Test orchestrator initialization."""
        assert orchestrator.config == mock_config
        assert orchestrator.orchestrator_id is not None
        assert len(orchestrator.orchestrator_id) > 0
        assert orchestrator.running is False
        assert orchestrator.shutdown_event is not None
        assert orchestrator.task_queue is not None
        assert orchestrator.active_tasks == {}
        assert orchestrator.sent_tasks == []
        assert orchestrator.failed_tasks == []
        assert orchestrator.workers == []
        assert orchestrator.alert_rules == {}
        assert orchestrator.channels == {}
        assert isinstance(orchestrator.metrics, AlertTaskMetrics)
        assert isinstance(orchestrator.channel_metrics, ChannelMetrics)
    
    def test_init_default_config(self):
        """Test orchestrator initialization with default config."""
        with patch('pynomaly.infrastructure.alerting.orchestrator.StructuredLogger'):
            with patch('pynomaly.infrastructure.alerting.orchestrator.MetricsService'):
                orchestrator = AlertingOrchestrator()
                
                assert orchestrator.config is not None
                assert orchestrator.config.max_workers == 3
                assert orchestrator.config.max_retries == 3
                assert orchestrator.config.timeout == 30.0
                assert orchestrator.config.rate_limit_per_minute == 60
    
    @pytest.mark.asyncio
    async def test_start_success(self, orchestrator):
        """Test successful orchestrator start."""
        with patch.object(orchestrator, '_start_workers', new_callable=AsyncMock) as mock_start_workers:
            with patch.object(orchestrator, '_start_metrics_collector', new_callable=AsyncMock) as mock_start_metrics:
                await orchestrator.start()
                
                assert orchestrator.running is True
                mock_start_workers.assert_called_once()
                mock_start_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, orchestrator):
        """Test starting orchestrator when already running."""
        orchestrator.running = True
        
        with pytest.raises(RuntimeError, match="Alerting orchestrator is already running"):
            await orchestrator.start()
    
    @pytest.mark.asyncio
    async def test_stop_success(self, orchestrator):
        """Test successful orchestrator stop."""
        # Set up running state
        orchestrator.running = True
        
        with patch.object(orchestrator, '_stop_workers', new_callable=AsyncMock) as mock_stop_workers:
            with patch.object(orchestrator, '_stop_metrics_collector', new_callable=AsyncMock) as mock_stop_metrics:
                await orchestrator.stop()
                
                assert orchestrator.running is False
                mock_stop_workers.assert_called_once()
                mock_stop_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_not_running(self, orchestrator):
        """Test stopping orchestrator when not running."""
        assert orchestrator.running is False
        
        # Should not raise an exception
        await orchestrator.stop()
        
        assert orchestrator.running is False
    
    @pytest.mark.asyncio
    async def test_submit_alert_success(self, orchestrator):
        """Test successful alert submission."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL]
        )
        
        with patch.object(orchestrator.task_queue, 'put', new_callable=AsyncMock) as mock_put:
            await orchestrator.submit_alert(task)
            
            mock_put.assert_called_once_with(task)
    
    @pytest.mark.asyncio
    async def test_submit_alert_not_running(self, orchestrator):
        """Test alert submission when orchestrator is not running."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL]
        )
        
        assert orchestrator.running is False
        
        with pytest.raises(RuntimeError, match="Alerting orchestrator is not running"):
            await orchestrator.submit_alert(task)
    
    @pytest.mark.asyncio
    async def test_create_alert_success(self, orchestrator):
        """Test successful alert creation."""
        task = await orchestrator.create_alert(
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            context={"anomaly_count": 10}
        )
        
        assert task.rule_id == "test-rule"
        assert task.message == "Test alert"
        assert task.severity == AlertSeverity.CRITICAL
        assert task.channels == [AlertChannel.EMAIL, AlertChannel.SLACK]
        assert task.context == {"anomaly_count": 10}
        assert task.status == AlertTaskStatus.PENDING
        assert task.task_id is not None
        assert len(task.task_id) > 0
    
    @pytest.mark.asyncio
    async def test_create_and_submit_alert_success(self, orchestrator):
        """Test successful alert creation and submission."""
        orchestrator.running = True
        
        with patch.object(orchestrator, 'submit_alert', new_callable=AsyncMock) as mock_submit:
            task = await orchestrator.create_and_submit_alert(
                rule_id="test-rule",
                message="Test alert",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL]
            )
            
            assert task.rule_id == "test-rule"
            assert task.message == "Test alert"
            assert task.severity == AlertSeverity.HIGH
            assert task.channels == [AlertChannel.EMAIL]
            mock_submit.assert_called_once_with(task)
    
    @pytest.mark.asyncio
    async def test_get_alert_status_found(self, orchestrator):
        """Test getting alert status when alert exists."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL],
            status=AlertTaskStatus.PROCESSING
        )
        
        orchestrator.active_tasks["test-task"] = task
        
        status = await orchestrator.get_alert_status("test-task")
        
        assert status == AlertTaskStatus.PROCESSING
    
    @pytest.mark.asyncio
    async def test_get_alert_status_not_found(self, orchestrator):
        """Test getting alert status when alert doesn't exist."""
        status = await orchestrator.get_alert_status("non-existent-task")
        
        assert status is None
    
    @pytest.mark.asyncio
    async def test_cancel_alert_success(self, orchestrator):
        """Test successful alert cancellation."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL],
            status=AlertTaskStatus.PROCESSING
        )
        
        orchestrator.active_tasks["test-task"] = task
        
        success = await orchestrator.cancel_alert("test-task")
        
        assert success is True
        assert task.status == AlertTaskStatus.FAILED
        assert task.error_message == "Alert cancelled"
    
    @pytest.mark.asyncio
    async def test_cancel_alert_not_found(self, orchestrator):
        """Test alert cancellation when alert doesn't exist."""
        success = await orchestrator.cancel_alert("non-existent-task")
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_cancel_alert_already_sent(self, orchestrator):
        """Test alert cancellation when alert is already sent."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL],
            status=AlertTaskStatus.SENT
        )
        
        orchestrator.active_tasks["test-task"] = task
        
        success = await orchestrator.cancel_alert("test-task")
        
        assert success is False
        assert task.status == AlertTaskStatus.SENT
    
    @pytest.mark.asyncio
    async def test_add_alert_rule_success(self, orchestrator):
        """Test successful alert rule addition."""
        rule = {
            "id": "test-rule",
            "name": "Test Rule",
            "condition": "anomaly_count > 5",
            "severity": AlertSeverity.HIGH,
            "channels": [AlertChannel.EMAIL],
            "enabled": True
        }
        
        await orchestrator.add_alert_rule(rule)
        
        assert "test-rule" in orchestrator.alert_rules
        assert orchestrator.alert_rules["test-rule"] == rule
    
    @pytest.mark.asyncio
    async def test_remove_alert_rule_success(self, orchestrator):
        """Test successful alert rule removal."""
        rule = {
            "id": "test-rule",
            "name": "Test Rule",
            "condition": "anomaly_count > 5",
            "severity": AlertSeverity.HIGH,
            "channels": [AlertChannel.EMAIL],
            "enabled": True
        }
        
        orchestrator.alert_rules["test-rule"] = rule
        
        success = await orchestrator.remove_alert_rule("test-rule")
        
        assert success is True
        assert "test-rule" not in orchestrator.alert_rules
    
    @pytest.mark.asyncio
    async def test_remove_alert_rule_not_found(self, orchestrator):
        """Test alert rule removal when rule doesn't exist."""
        success = await orchestrator.remove_alert_rule("non-existent-rule")
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_configure_channel_success(self, orchestrator):
        """Test successful channel configuration."""
        channel_config = {
            "type": AlertChannel.EMAIL,
            "settings": {
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "username": "alerts@example.com",
                "password": "password123"
            }
        }
        
        await orchestrator.configure_channel(AlertChannel.EMAIL, channel_config)
        
        assert AlertChannel.EMAIL in orchestrator.channels
        assert orchestrator.channels[AlertChannel.EMAIL] == channel_config
    
    @pytest.mark.asyncio
    async def test_remove_channel_success(self, orchestrator):
        """Test successful channel removal."""
        channel_config = {
            "type": AlertChannel.EMAIL,
            "settings": {"smtp_server": "smtp.example.com"}
        }
        
        orchestrator.channels[AlertChannel.EMAIL] = channel_config
        
        success = await orchestrator.remove_channel(AlertChannel.EMAIL)
        
        assert success is True
        assert AlertChannel.EMAIL not in orchestrator.channels
    
    @pytest.mark.asyncio
    async def test_remove_channel_not_found(self, orchestrator):
        """Test channel removal when channel doesn't exist."""
        success = await orchestrator.remove_channel(AlertChannel.SLACK)
        
        assert success is False
    
    def test_get_metrics(self, orchestrator):
        """Test getting orchestrator metrics."""
        metrics = orchestrator.get_metrics()
        
        assert isinstance(metrics, AlertTaskMetrics)
        assert metrics == orchestrator.metrics
    
    def test_get_channel_metrics(self, orchestrator):
        """Test getting channel metrics."""
        metrics = orchestrator.get_channel_metrics()
        
        assert isinstance(metrics, ChannelMetrics)
        assert metrics == orchestrator.channel_metrics
    
    def test_get_alert_rules(self, orchestrator):
        """Test getting alert rules."""
        rule1 = {"id": "rule-1", "name": "Rule 1"}
        rule2 = {"id": "rule-2", "name": "Rule 2"}
        
        orchestrator.alert_rules["rule-1"] = rule1
        orchestrator.alert_rules["rule-2"] = rule2
        
        rules = orchestrator.get_alert_rules()
        
        assert len(rules) == 2
        assert rule1 in rules
        assert rule2 in rules
    
    def test_get_configured_channels(self, orchestrator):
        """Test getting configured channels."""
        email_config = {"type": AlertChannel.EMAIL, "settings": {}}
        slack_config = {"type": AlertChannel.SLACK, "settings": {}}
        
        orchestrator.channels[AlertChannel.EMAIL] = email_config
        orchestrator.channels[AlertChannel.SLACK] = slack_config
        
        channels = orchestrator.get_configured_channels()
        
        assert len(channels) == 2
        assert AlertChannel.EMAIL in channels
        assert AlertChannel.SLACK in channels
    
    def test_get_active_alerts(self, orchestrator):
        """Test getting active alerts."""
        task1 = AlertTask(
            task_id="task-1",
            rule_id="rule-1",
            message="Alert 1",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL]
        )
        task2 = AlertTask(
            task_id="task-2",
            rule_id="rule-2",
            message="Alert 2",
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.SLACK]
        )
        
        orchestrator.active_tasks["task-1"] = task1
        orchestrator.active_tasks["task-2"] = task2
        
        active_alerts = orchestrator.get_active_alerts()
        
        assert len(active_alerts) == 2
        assert task1 in active_alerts
        assert task2 in active_alerts
    
    def test_get_sent_alerts(self, orchestrator):
        """Test getting sent alerts."""
        task1 = AlertTask(
            task_id="task-1",
            rule_id="rule-1",
            message="Alert 1",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL],
            status=AlertTaskStatus.SENT
        )
        task2 = AlertTask(
            task_id="task-2",
            rule_id="rule-2",
            message="Alert 2",
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.SLACK],
            status=AlertTaskStatus.SENT
        )
        
        orchestrator.sent_tasks = [task1, task2]
        
        sent_alerts = orchestrator.get_sent_alerts()
        
        assert len(sent_alerts) == 2
        assert task1 in sent_alerts
        assert task2 in sent_alerts
    
    def test_get_failed_alerts(self, orchestrator):
        """Test getting failed alerts."""
        task1 = AlertTask(
            task_id="task-1",
            rule_id="rule-1",
            message="Alert 1",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL],
            status=AlertTaskStatus.FAILED
        )
        task2 = AlertTask(
            task_id="task-2",
            rule_id="rule-2",
            message="Alert 2",
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.SLACK],
            status=AlertTaskStatus.FAILED
        )
        
        orchestrator.failed_tasks = [task1, task2]
        
        failed_alerts = orchestrator.get_failed_alerts()
        
        assert len(failed_alerts) == 2
        assert task1 in failed_alerts
        assert task2 in failed_alerts
    
    @pytest.mark.asyncio
    async def test_process_alert_success(self, orchestrator):
        """Test successful alert processing."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL]
        )
        
        # Mock successful processing
        with patch.object(orchestrator, '_send_alert', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {"sent": [AlertChannel.EMAIL], "failed": []}
            
            await orchestrator._process_alert(task)
            
            # Verify alert was sent
            mock_send.assert_called_once_with(task)
            
            # Verify task status was updated
            assert task.status == AlertTaskStatus.SENT
            assert task.sent_channels == [AlertChannel.EMAIL]
            assert task.failed_channels == []
            assert task.started_at is not None
            assert task.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_process_alert_failure(self, orchestrator):
        """Test alert processing with failure."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL]
        )
        
        # Mock alert sending failure
        with patch.object(orchestrator, '_send_alert', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("Send failed")
            
            await orchestrator._process_alert(task)
            
            # Verify task status was updated
            assert task.status == AlertTaskStatus.FAILED
            assert task.error_message == "Send failed"
            assert task.started_at is not None
            assert task.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_process_alert_retry(self, orchestrator):
        """Test alert processing with retry."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL],
            retry_count=0,
            max_retries=3
        )
        
        # Mock alert sending failure
        with patch.object(orchestrator, '_send_alert', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("Send failed")
            
            with patch.object(orchestrator, '_retry_alert', new_callable=AsyncMock) as mock_retry:
                await orchestrator._process_alert(task)
                
                # Verify retry was attempted
                mock_retry.assert_called_once_with(task)
                
                # Verify task status was updated for retry
                assert task.status == AlertTaskStatus.RETRYING
    
    @pytest.mark.asyncio
    async def test_send_alert_email_success(self, orchestrator):
        """Test successful email alert sending."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL]
        )
        
        # Configure email channel
        orchestrator.channels[AlertChannel.EMAIL] = {
            "type": AlertChannel.EMAIL,
            "settings": {"smtp_server": "smtp.example.com"}
        }
        
        # Mock the actual sending logic
        with patch.object(orchestrator, '_send_email', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True
            
            result = await orchestrator._send_alert(task)
            
            # Verify email sending was called
            mock_send.assert_called_once_with(task)
            assert result["sent"] == [AlertChannel.EMAIL]
            assert result["failed"] == []
    
    @pytest.mark.asyncio
    async def test_send_alert_multiple_channels(self, orchestrator):
        """Test sending alert to multiple channels."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        
        # Configure channels
        orchestrator.channels[AlertChannel.EMAIL] = {
            "type": AlertChannel.EMAIL,
            "settings": {"smtp_server": "smtp.example.com"}
        }
        orchestrator.channels[AlertChannel.SLACK] = {
            "type": AlertChannel.SLACK,
            "settings": {"webhook_url": "https://hooks.slack.com/..."}
        }
        
        # Mock the actual sending logic
        with patch.object(orchestrator, '_send_email', new_callable=AsyncMock) as mock_send_email:
            with patch.object(orchestrator, '_send_slack', new_callable=AsyncMock) as mock_send_slack:
                mock_send_email.return_value = True
                mock_send_slack.return_value = True
                
                result = await orchestrator._send_alert(task)
                
                # Verify both channels were used
                mock_send_email.assert_called_once_with(task)
                mock_send_slack.assert_called_once_with(task)
                assert set(result["sent"]) == {AlertChannel.EMAIL, AlertChannel.SLACK}
                assert result["failed"] == []
    
    @pytest.mark.asyncio
    async def test_send_alert_partial_failure(self, orchestrator):
        """Test sending alert with partial failure."""
        task = AlertTask(
            task_id="test-task",
            rule_id="test-rule",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        
        # Configure channels
        orchestrator.channels[AlertChannel.EMAIL] = {
            "type": AlertChannel.EMAIL,
            "settings": {"smtp_server": "smtp.example.com"}
        }
        orchestrator.channels[AlertChannel.SLACK] = {
            "type": AlertChannel.SLACK,
            "settings": {"webhook_url": "https://hooks.slack.com/..."}
        }
        
        # Mock the actual sending logic with partial failure
        with patch.object(orchestrator, '_send_email', new_callable=AsyncMock) as mock_send_email:
            with patch.object(orchestrator, '_send_slack', new_callable=AsyncMock) as mock_send_slack:
                mock_send_email.return_value = True
                mock_send_slack.side_effect = Exception("Slack failed")
                
                result = await orchestrator._send_alert(task)
                
                # Verify results
                assert result["sent"] == [AlertChannel.EMAIL]
                assert result["failed"] == [AlertChannel.SLACK]


class TestAlertingOrchestratorIntegration:
    """Integration tests for alerting orchestrator."""
    
    @pytest.mark.asyncio
    async def test_full_alert_lifecycle(self):
        """Test the full lifecycle of an alert."""
        config = Mock(spec=AlertingConfig)
        config.max_workers = 2
        config.max_retries = 3
        config.retry_delay = 0.1  # Short delay for testing
        
        with patch('pynomaly.infrastructure.alerting.orchestrator.StructuredLogger'):
            with patch('pynomaly.infrastructure.alerting.orchestrator.MetricsService'):
                orchestrator = AlertingOrchestrator(config=config)
                
                # Start orchestrator
                await orchestrator.start()
                assert orchestrator.running is True
                
                # Configure channel
                await orchestrator.configure_channel(AlertChannel.EMAIL, {
                    "type": AlertChannel.EMAIL,
                    "settings": {"smtp_server": "smtp.example.com"}
                })
                assert AlertChannel.EMAIL in orchestrator.channels
                
                # Add alert rule
                await orchestrator.add_alert_rule({
                    "id": "test-rule",
                    "name": "Test Rule",
                    "condition": "anomaly_count > 5",
                    "severity": AlertSeverity.HIGH,
                    "channels": [AlertChannel.EMAIL],
                    "enabled": True
                })
                assert "test-rule" in orchestrator.alert_rules
                
                # Create and submit alert
                task = await orchestrator.create_and_submit_alert(
                    rule_id="test-rule",
                    message="Test alert message",
                    severity=AlertSeverity.HIGH,
                    channels=[AlertChannel.EMAIL]
                )
                
                # Wait for alert to be processed
                await asyncio.sleep(0.2)
                
                # Check alert status
                status = await orchestrator.get_alert_status(task.task_id)
                assert status in [AlertTaskStatus.SENT, AlertTaskStatus.FAILED]
                
                # Get metrics
                metrics = orchestrator.get_metrics()
                assert metrics.total_alerts >= 1
                
                # Remove alert rule
                success = await orchestrator.remove_alert_rule("test-rule")
                assert success is True
                assert "test-rule" not in orchestrator.alert_rules
                
                # Remove channel
                success = await orchestrator.remove_channel(AlertChannel.EMAIL)
                assert success is True
                assert AlertChannel.EMAIL not in orchestrator.channels
                
                # Stop orchestrator
                await orchestrator.stop()
                assert orchestrator.running is False
