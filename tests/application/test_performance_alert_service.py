"""Tests for performance alert service."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from pynomaly.application.services.performance_alert_service import (
    PerformanceAlertService,
    ConsoleAlertChannel,
    EmailAlertChannel,
    SlackAlertChannel,
    WebhookAlertChannel
)
from pynomaly.domain.value_objects.performance_degradation_metrics import (
    DegradationAlert,
    DegradationSeverity,
    DegradationType,
    PerformanceDegradation,
)
from pynomaly.infrastructure.repositories.performance_degradation_repository import (
    PerformanceDegradationRepository,
)


class TestAlertChannels:
    """Test suite for alert channels."""
    
    @pytest.fixture
    def sample_degradation(self):
        """Create sample performance degradation."""
        return PerformanceDegradation(
            degradation_type=DegradationType.ACCURACY_DROP,
            severity=DegradationSeverity.HIGH,
            metric_name="accuracy",
            current_value=0.75,
            baseline_value=0.85,
            degradation_amount=-0.10,
            degradation_percentage=-11.76,
            threshold_violated="warning",
            confidence_level=0.85,
            detection_method="baseline_comparison",
            samples_used=25
        )
    
    @pytest.fixture
    def sample_alert(self, sample_degradation):
        """Create sample alert."""
        return DegradationAlert(
            alert_id="alert_123",
            model_id="model_456",
            degradation=sample_degradation,
            alert_level=DegradationSeverity.HIGH,
            message="Performance degradation detected in accuracy metric",
            recommended_actions=[
                "Review training data quality",
                "Check for feature drift",
                "Consider model retraining"
            ]
        )
    
    def test_console_channel_creation(self):
        """Test console alert channel creation."""
        channel = ConsoleAlertChannel("console", {"enabled": True})
        
        assert channel.name == "console"
        assert channel.enabled is True
    
    def test_console_channel_should_send(self, sample_alert):
        """Test console channel send filtering."""
        # Test with minimum severity
        channel = ConsoleAlertChannel("console", {"enabled": True, "min_severity": "medium"})
        
        assert channel.should_send(sample_alert) is True  # High severity should pass
        
        # Change alert to low severity
        sample_alert.alert_level = DegradationSeverity.LOW
        assert channel.should_send(sample_alert) is False  # Low severity should not pass
    
    @pytest.mark.asyncio
    async def test_console_channel_send_alert(self, sample_alert):
        """Test console channel alert sending."""
        channel = ConsoleAlertChannel("console", {"enabled": True})
        
        # Should succeed (logs to console)
        result = await channel.send_alert(sample_alert)
        assert result is True
    
    def test_email_channel_creation(self):
        """Test email alert channel creation."""
        config = {
            "enabled": True,
            "recipients": ["admin@example.com", "ops@example.com"]
        }
        channel = EmailAlertChannel("email", config)
        
        assert channel.name == "email"
        assert channel.enabled is True
        assert channel.config["recipients"] == ["admin@example.com", "ops@example.com"]
    
    @pytest.mark.asyncio
    async def test_email_channel_send_alert(self, sample_alert):
        """Test email channel alert sending."""
        config = {
            "enabled": True,
            "recipients": ["admin@example.com"]
        }
        channel = EmailAlertChannel("email", config)
        
        # Should succeed (mocked email sending)
        result = await channel.send_alert(sample_alert)
        assert result is True
    
    def test_slack_channel_creation(self):
        """Test Slack alert channel creation."""
        config = {
            "enabled": True,
            "webhook_url": "https://hooks.slack.com/test",
            "channel": "#alerts"
        }
        channel = SlackAlertChannel("slack", config)
        
        assert channel.name == "slack"
        assert channel.enabled is True
        assert channel.config["webhook_url"] == "https://hooks.slack.com/test"
    
    @pytest.mark.asyncio
    async def test_slack_channel_send_alert(self, sample_alert):
        """Test Slack channel alert sending."""
        config = {
            "enabled": True,
            "webhook_url": "https://hooks.slack.com/test",
            "channel": "#alerts"
        }
        channel = SlackAlertChannel("slack", config)
        
        # Should succeed (mocked Slack sending)
        result = await channel.send_alert(sample_alert)
        assert result is True
    
    def test_webhook_channel_creation(self):
        """Test webhook alert channel creation."""
        config = {
            "enabled": True,
            "url": "https://api.example.com/alerts"
        }
        channel = WebhookAlertChannel("webhook", config)
        
        assert channel.name == "webhook"
        assert channel.enabled is True
        assert channel.config["url"] == "https://api.example.com/alerts"
    
    @pytest.mark.asyncio
    async def test_webhook_channel_send_alert(self, sample_alert):
        """Test webhook channel alert sending."""
        config = {
            "enabled": True,
            "url": "https://api.example.com/alerts"
        }
        channel = WebhookAlertChannel("webhook", config)
        
        # Should succeed (mocked webhook sending)
        result = await channel.send_alert(sample_alert)
        assert result is True


class TestPerformanceAlertService:
    """Test suite for PerformanceAlertService."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repository = MagicMock(spec=PerformanceDegradationRepository)
        repository.store_alert = AsyncMock()
        repository.get_alert = AsyncMock()
        repository.update_alert = AsyncMock()
        repository.get_active_alerts = AsyncMock(return_value=[])
        return repository
    
    @pytest.fixture
    def alert_config(self):
        """Create alert configuration."""
        return {
            "console": {"enabled": True, "min_severity": "low"},
            "email": {
                "enabled": True,
                "recipients": ["admin@example.com"],
                "min_severity": "medium"
            },
            "slack": {
                "enabled": True,
                "webhook_url": "https://hooks.slack.com/test",
                "channel": "#alerts",
                "min_severity": "high"
            },
            "throttling": {
                "enabled": True,
                "minutes": 30,
                "max_per_hour": 10
            }
        }
    
    @pytest.fixture
    def alert_service(self, mock_repository, alert_config):
        """Create alert service with mocked dependencies."""
        return PerformanceAlertService(mock_repository, alert_config)
    
    @pytest.fixture
    def sample_degradation(self):
        """Create sample performance degradation."""
        return PerformanceDegradation(
            degradation_type=DegradationType.ACCURACY_DROP,
            severity=DegradationSeverity.HIGH,
            metric_name="accuracy",
            current_value=0.75,
            baseline_value=0.85,
            degradation_amount=-0.10,
            degradation_percentage=-11.76,
            threshold_violated="warning",
            confidence_level=0.85,
            detection_method="baseline_comparison",
            samples_used=25
        )
    
    def test_alert_service_initialization(self, alert_service):
        """Test alert service initialization."""
        assert len(alert_service.channels) >= 3  # console, email, slack
        assert "console" in alert_service.channels
        assert "email" in alert_service.channels
        assert "slack" in alert_service.channels
    
    @pytest.mark.asyncio
    async def test_create_alert(self, alert_service, sample_degradation):
        """Test alert creation."""
        model_id = uuid4()
        
        alert = await alert_service.create_alert(
            model_id=model_id,
            degradation=sample_degradation,
            custom_message="Custom alert message",
            tags=["test", "urgent"]
        )
        
        assert alert.model_id == str(model_id)
        assert alert.degradation == sample_degradation
        assert alert.alert_level == sample_degradation.severity
        assert alert.message == "Custom alert message"
        assert "test" in alert.tags
        assert "urgent" in alert.tags
        assert len(alert.recommended_actions) > 0
        
        # Verify repository was called
        alert_service.repository.store_alert.assert_called_once_with(alert)
    
    @pytest.mark.asyncio
    async def test_create_alert_default_message(self, alert_service, sample_degradation):
        """Test alert creation with default message."""
        model_id = uuid4()
        
        alert = await alert_service.create_alert(
            model_id=model_id,
            degradation=sample_degradation
        )
        
        # Should generate default message
        assert "accuracy" in alert.message.lower()
        assert "degraded" in alert.message.lower()
        assert alert.alert_level == DegradationSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_send_alert_all_channels(self, alert_service, sample_degradation):
        """Test sending alert through all channels."""
        model_id = uuid4()
        
        alert = await alert_service.create_alert(
            model_id=model_id,
            degradation=sample_degradation
        )
        
        results = await alert_service.send_alert(alert)
        
        # Should have results for enabled channels
        assert "console" in results
        assert "email" in results  # High severity should pass medium threshold
        assert "slack" in results  # High severity should pass high threshold
        
        # All should succeed (mocked)
        assert all(results.values())
    
    @pytest.mark.asyncio
    async def test_send_alert_severity_filtering(self, alert_service):
        """Test alert filtering by severity."""
        model_id = uuid4()
        
        # Create low severity degradation
        low_degradation = PerformanceDegradation(
            degradation_type=DegradationType.ACCURACY_DROP,
            severity=DegradationSeverity.LOW,
            metric_name="accuracy",
            current_value=0.82,
            baseline_value=0.85,
            degradation_amount=-0.03,
            degradation_percentage=-3.53,
            threshold_violated="warning",
            confidence_level=0.65,
            detection_method="baseline_comparison",
            samples_used=20
        )
        
        alert = await alert_service.create_alert(
            model_id=model_id,
            degradation=low_degradation
        )
        
        results = await alert_service.send_alert(alert)
        
        # Only console should receive low severity (email requires medium, slack requires high)
        assert "console" in results
        assert results["console"] is True
        
        # Email and Slack should be filtered out or return False
        if "email" in results:
            assert results["email"] is False or "email" not in results
        if "slack" in results:
            assert results["slack"] is False or "slack" not in results
    
    @pytest.mark.asyncio
    async def test_process_degradations(self, alert_service):
        """Test processing multiple degradations."""
        model_id = uuid4()
        
        degradations = [
            PerformanceDegradation(
                degradation_type=DegradationType.ACCURACY_DROP,
                severity=DegradationSeverity.HIGH,
                metric_name="accuracy",
                current_value=0.75,
                baseline_value=0.85,
                degradation_amount=-0.10,
                degradation_percentage=-11.76,
                threshold_violated="warning",
                confidence_level=0.85,
                detection_method="baseline_comparison",
                samples_used=25
            ),
            PerformanceDegradation(
                degradation_type=DegradationType.PRECISION_DROP,
                severity=DegradationSeverity.CRITICAL,
                metric_name="precision",
                current_value=0.65,
                baseline_value=0.85,
                degradation_amount=-0.20,
                degradation_percentage=-23.53,
                threshold_violated="critical",
                confidence_level=0.92,
                detection_method="baseline_comparison",
                samples_used=25
            )
        ]
        
        alerts = await alert_service.process_degradations(
            model_id=model_id,
            degradations=degradations,
            send_alerts=True
        )
        
        assert len(alerts) == 2
        assert alerts[0].degradation.metric_name == "accuracy"
        assert alerts[1].degradation.metric_name == "precision"
        
        # Verify repository calls
        assert alert_service.repository.store_alert.call_count == 2
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_service):
        """Test alert acknowledgment."""
        # Mock existing alert
        sample_alert = DegradationAlert(
            alert_id="alert_123",
            model_id="model_456",
            degradation=PerformanceDegradation(
                degradation_type=DegradationType.ACCURACY_DROP,
                severity=DegradationSeverity.HIGH,
                metric_name="accuracy",
                current_value=0.75,
                baseline_value=0.85,
                degradation_amount=-0.10,
                degradation_percentage=-11.76,
                threshold_violated="warning",
                confidence_level=0.85,
                detection_method="baseline_comparison",
                samples_used=25
            ),
            alert_level=DegradationSeverity.HIGH,
            message="Test alert"
        )
        
        alert_service.repository.get_alert.return_value = sample_alert
        
        result = await alert_service.acknowledge_alert(
            alert_id="alert_123",
            acknowledged_by="admin"
        )
        
        assert result is True
        
        # Verify repository calls
        alert_service.repository.get_alert.assert_called_once_with("alert_123")
        alert_service.repository.update_alert.assert_called_once()
        
        # Check the updated alert
        updated_alert = alert_service.repository.update_alert.call_args[0][0]
        assert updated_alert.acknowledged is True
        assert updated_alert.acknowledged_by == "admin"
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_service):
        """Test alert resolution."""
        # Mock existing alert
        sample_alert = DegradationAlert(
            alert_id="alert_123",
            model_id="model_456",
            degradation=PerformanceDegradation(
                degradation_type=DegradationType.ACCURACY_DROP,
                severity=DegradationSeverity.HIGH,
                metric_name="accuracy",
                current_value=0.75,
                baseline_value=0.85,
                degradation_amount=-0.10,
                degradation_percentage=-11.76,
                threshold_violated="warning",
                confidence_level=0.85,
                detection_method="baseline_comparison",
                samples_used=25
            ),
            alert_level=DegradationSeverity.HIGH,
            message="Test alert"
        )
        
        alert_service.repository.get_alert.return_value = sample_alert
        
        result = await alert_service.resolve_alert(
            alert_id="alert_123",
            resolved_by="ops_team"
        )
        
        assert result is True
        
        # Verify repository calls
        alert_service.repository.get_alert.assert_called_once_with("alert_123")
        alert_service.repository.update_alert.assert_called_once()
        
        # Check the updated alert
        updated_alert = alert_service.repository.update_alert.call_args[0][0]
        assert updated_alert.resolved is True
        assert updated_alert.resolved_by == "ops_team"
    
    @pytest.mark.asyncio
    async def test_get_active_alerts(self, alert_service):
        """Test getting active alerts."""
        model_id = uuid4()
        
        # Mock active alerts
        active_alerts = [
            DegradationAlert(
                alert_id="alert_1",
                model_id=str(model_id),
                degradation=PerformanceDegradation(
                    degradation_type=DegradationType.ACCURACY_DROP,
                    severity=DegradationSeverity.HIGH,
                    metric_name="accuracy",
                    current_value=0.75,
                    baseline_value=0.85,
                    degradation_amount=-0.10,
                    degradation_percentage=-11.76,
                    threshold_violated="warning",
                    confidence_level=0.85,
                    detection_method="baseline_comparison",
                    samples_used=25
                ),
                alert_level=DegradationSeverity.HIGH,
                message="Test alert 1"
            )
        ]
        
        alert_service.repository.get_active_alerts.return_value = active_alerts
        
        alerts = await alert_service.get_active_alerts(model_id)
        
        assert len(alerts) == 1
        assert alerts[0].alert_id == "alert_1"
        
        # Verify repository call
        alert_service.repository.get_active_alerts.assert_called_once_with(model_id)
    
    def test_generate_default_message(self, alert_service):
        """Test default message generation."""
        degradation = PerformanceDegradation(
            degradation_type=DegradationType.ACCURACY_DROP,
            severity=DegradationSeverity.HIGH,
            metric_name="accuracy",
            current_value=0.75,
            baseline_value=0.85,
            degradation_amount=-0.10,
            degradation_percentage=-11.76,
            threshold_violated="warning",
            confidence_level=0.85,
            detection_method="baseline_comparison",
            samples_used=25
        )
        
        message = alert_service._generate_default_message(degradation)
        
        assert "accuracy" in message
        assert "degradation" in message.lower()
        assert "0.75" in message  # Current value
        assert "0.85" in message  # Baseline value
        assert "11.76%" in message  # Degradation percentage
    
    def test_generate_recommendations_critical(self, alert_service):
        """Test recommendation generation for critical degradation."""
        degradation = PerformanceDegradation(
            degradation_type=DegradationType.ACCURACY_DROP,
            severity=DegradationSeverity.CRITICAL,
            metric_name="accuracy",
            current_value=0.65,
            baseline_value=0.85,
            degradation_amount=-0.20,
            degradation_percentage=-23.53,
            threshold_violated="critical",
            confidence_level=0.95,
            detection_method="baseline_comparison",
            samples_used=30
        )
        
        recommendations = alert_service._generate_recommendations(degradation)
        
        assert len(recommendations) > 0
        assert any("critical" in rec.lower() for rec in recommendations)
        assert any("rollback" in rec.lower() or "retraining" in rec.lower() for rec in recommendations)
        assert any("accuracy" in rec.lower() for rec in recommendations)
    
    def test_should_throttle_alert_disabled(self, alert_service):
        """Test throttling when disabled."""
        # Disable throttling
        alert_service.config['throttling']['enabled'] = False
        
        alert = DegradationAlert(
            alert_id="alert_123",
            model_id="model_456",
            degradation=PerformanceDegradation(
                degradation_type=DegradationType.ACCURACY_DROP,
                severity=DegradationSeverity.HIGH,
                metric_name="accuracy",
                current_value=0.75,
                baseline_value=0.85,
                degradation_amount=-0.10,
                degradation_percentage=-11.76,
                threshold_violated="warning",
                confidence_level=0.85,
                detection_method="baseline_comparison",
                samples_used=25
            ),
            alert_level=DegradationSeverity.HIGH,
            message="Test alert"
        )
        
        should_throttle = alert_service._should_throttle_alert(alert)
        assert should_throttle is False
    
    def test_should_throttle_alert_time_based(self, alert_service):
        """Test time-based alert throttling."""
        alert = DegradationAlert(
            alert_id="alert_123",
            model_id="model_456",
            degradation=PerformanceDegradation(
                degradation_type=DegradationType.ACCURACY_DROP,
                severity=DegradationSeverity.HIGH,
                metric_name="accuracy",
                current_value=0.75,
                baseline_value=0.85,
                degradation_amount=-0.10,
                degradation_percentage=-11.76,
                threshold_violated="warning",
                confidence_level=0.85,
                detection_method="baseline_comparison",
                samples_used=25
            ),
            alert_level=DegradationSeverity.HIGH,
            message="Test alert"
        )
        
        # Set up recent alert
        throttle_key = f"{alert.model_id}_{alert.degradation.metric_name}"
        recent_time = datetime.utcnow() - timedelta(minutes=10)  # 10 minutes ago
        alert_service._alert_throttle[throttle_key] = (recent_time, 1)
        
        should_throttle = alert_service._should_throttle_alert(alert)
        assert should_throttle is True  # Should be throttled (within 30 minutes)
    
    def test_update_throttle_tracking(self, alert_service):
        """Test throttle tracking updates."""
        alert = DegradationAlert(
            alert_id="alert_123",
            model_id="model_456",
            degradation=PerformanceDegradation(
                degradation_type=DegradationType.ACCURACY_DROP,
                severity=DegradationSeverity.HIGH,
                metric_name="accuracy",
                current_value=0.75,
                baseline_value=0.85,
                degradation_amount=-0.10,
                degradation_percentage=-11.76,
                threshold_violated="warning",
                confidence_level=0.85,
                detection_method="baseline_comparison",
                samples_used=25
            ),
            alert_level=DegradationSeverity.HIGH,
            message="Test alert"
        )
        
        # Update throttle tracking
        alert_service._update_throttle_tracking(alert)
        
        throttle_key = f"{alert.model_id}_{alert.degradation.metric_name}"
        assert throttle_key in alert_service._alert_throttle
        
        timestamp, count = alert_service._alert_throttle[throttle_key]
        assert count == 1
        assert isinstance(timestamp, datetime)