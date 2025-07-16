"""Comprehensive tests for Alert domain entity and related components."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from monorepo.domain.entities.alert import (
    Alert,
    AlertCondition,
    AlertCorrelation,
    AlertNotification,
    AlertSeverity,
    AlertStatus,
    AlertType,
    MLNoiseFeatures,
    NotificationChannel,
)


class TestAlertCondition:
    """Test AlertCondition class."""

    def test_alert_condition_initialization(self):
        """Test alert condition initialization."""
        condition = AlertCondition(
            metric_name="cpu_usage",
            operator="gt",
            threshold=80.0,
            time_window_minutes=5,
            consecutive_breaches=2,
            description="CPU usage is too high",
        )

        assert condition.metric_name == "cpu_usage"
        assert condition.operator == "gt"
        assert condition.threshold == 80.0
        assert condition.time_window_minutes == 5
        assert condition.consecutive_breaches == 2
        assert condition.description == "CPU usage is too high"

    def test_alert_condition_validation_invalid_operator(self):
        """Test alert condition validation with invalid operator."""
        with pytest.raises(ValueError, match="Operator must be one of"):
            AlertCondition(metric_name="cpu_usage", operator="invalid", threshold=80.0)

    def test_alert_condition_validation_invalid_time_window(self):
        """Test alert condition validation with invalid time window."""
        with pytest.raises(ValueError, match="Time window must be positive"):
            AlertCondition(
                metric_name="cpu_usage",
                operator="gt",
                threshold=80.0,
                time_window_minutes=0,
            )

    def test_alert_condition_validation_invalid_consecutive_breaches(self):
        """Test alert condition validation with invalid consecutive breaches."""
        with pytest.raises(ValueError, match="Consecutive breaches must be positive"):
            AlertCondition(
                metric_name="cpu_usage",
                operator="gt",
                threshold=80.0,
                consecutive_breaches=0,
            )

    def test_alert_condition_evaluate_greater_than(self):
        """Test alert condition evaluation with greater than operator."""
        condition = AlertCondition(
            metric_name="cpu_usage", operator="gt", threshold=80.0
        )

        assert condition.evaluate(85.0) is True
        assert condition.evaluate(80.0) is False
        assert condition.evaluate(75.0) is False

    def test_alert_condition_evaluate_less_than(self):
        """Test alert condition evaluation with less than operator."""
        condition = AlertCondition(
            metric_name="cpu_usage", operator="lt", threshold=20.0
        )

        assert condition.evaluate(15.0) is True
        assert condition.evaluate(20.0) is False
        assert condition.evaluate(25.0) is False

    def test_alert_condition_evaluate_equal_to(self):
        """Test alert condition evaluation with equal to operator."""
        condition = AlertCondition(
            metric_name="cpu_usage", operator="eq", threshold=50.0
        )

        assert condition.evaluate(50.0) is True
        assert condition.evaluate(49.0) is False
        assert condition.evaluate(51.0) is False

    def test_alert_condition_evaluate_greater_than_equal(self):
        """Test alert condition evaluation with greater than or equal operator."""
        condition = AlertCondition(
            metric_name="cpu_usage", operator="gte", threshold=80.0
        )

        assert condition.evaluate(85.0) is True
        assert condition.evaluate(80.0) is True
        assert condition.evaluate(75.0) is False

    def test_alert_condition_evaluate_less_than_equal(self):
        """Test alert condition evaluation with less than or equal operator."""
        condition = AlertCondition(
            metric_name="cpu_usage", operator="lte", threshold=20.0
        )

        assert condition.evaluate(15.0) is True
        assert condition.evaluate(20.0) is True
        assert condition.evaluate(25.0) is False

    def test_alert_condition_evaluate_not_equal(self):
        """Test alert condition evaluation with not equal operator."""
        condition = AlertCondition(
            metric_name="cpu_usage", operator="ne", threshold=50.0
        )

        assert condition.evaluate(60.0) is True
        assert condition.evaluate(40.0) is True
        assert condition.evaluate(50.0) is False

    def test_alert_condition_get_description_custom(self):
        """Test getting custom description."""
        condition = AlertCondition(
            metric_name="cpu_usage",
            operator="gt",
            threshold=80.0,
            description="Custom CPU alert",
        )

        assert condition.get_description() == "Custom CPU alert"

    def test_alert_condition_get_description_generated(self):
        """Test getting generated description."""
        condition = AlertCondition(
            metric_name="memory_usage", operator="gte", threshold=90.0
        )

        expected = "memory_usage greater than or equal to 90.0"
        assert condition.get_description() == expected

    def test_alert_condition_all_operators_description(self):
        """Test description generation for all operators."""
        operators = {
            "gt": "greater than",
            "lt": "less than",
            "eq": "equal to",
            "gte": "greater than or equal to",
            "lte": "less than or equal to",
            "ne": "not equal to",
        }

        for op, text in operators.items():
            condition = AlertCondition(
                metric_name="test_metric", operator=op, threshold=50.0
            )
            expected = f"test_metric {text} 50.0"
            assert condition.get_description() == expected


class TestAlertNotification:
    """Test AlertNotification class."""

    def test_alert_notification_initialization(self):
        """Test alert notification initialization."""
        alert_id = uuid4()
        notification = AlertNotification(
            alert_id=alert_id,
            channel=NotificationChannel.EMAIL,
            recipient="user@example.com",
        )

        assert notification.alert_id == alert_id
        assert notification.channel == NotificationChannel.EMAIL
        assert notification.recipient == "user@example.com"
        assert notification.sent_at is None
        assert notification.delivered_at is None
        assert notification.status == "pending"
        assert notification.error_message is None
        assert notification.metadata == {}

    def test_alert_notification_properties(self):
        """Test alert notification properties."""
        notification = AlertNotification(
            channel=NotificationChannel.SLACK, recipient="@channel", status="delivered"
        )

        assert notification.is_delivered is True
        assert notification.is_failed is False

        notification.status = "failed"
        assert notification.is_delivered is False
        assert notification.is_failed is True

    def test_alert_notification_mark_sent(self):
        """Test marking notification as sent."""
        notification = AlertNotification(
            channel=NotificationChannel.EMAIL, recipient="user@example.com"
        )

        notification.mark_sent()

        assert notification.status == "sent"
        assert notification.sent_at is not None
        assert isinstance(notification.sent_at, datetime)

    def test_alert_notification_mark_delivered(self):
        """Test marking notification as delivered."""
        notification = AlertNotification(
            channel=NotificationChannel.EMAIL, recipient="user@example.com"
        )

        notification.mark_delivered()

        assert notification.status == "delivered"
        assert notification.delivered_at is not None
        assert isinstance(notification.delivered_at, datetime)

    def test_alert_notification_mark_failed(self):
        """Test marking notification as failed."""
        notification = AlertNotification(
            channel=NotificationChannel.EMAIL, recipient="user@example.com"
        )

        error_msg = "SMTP server unavailable"
        notification.mark_failed(error_msg)

        assert notification.status == "failed"
        assert notification.error_message == error_msg


class TestAlertCorrelation:
    """Test AlertCorrelation class."""

    def test_alert_correlation_initialization(self):
        """Test alert correlation initialization."""
        correlation = AlertCorrelation(
            primary_alert_id="alert1",
            related_alert_ids=["alert2", "alert3"],
            correlation_score=0.85,
            correlation_type="temporal",
        )

        assert correlation.primary_alert_id == "alert1"
        assert correlation.related_alert_ids == ["alert2", "alert3"]
        assert correlation.correlation_score == 0.85
        assert correlation.correlation_type == "temporal"
        assert isinstance(correlation.correlation_id, str)
        assert isinstance(correlation.timestamp, datetime)

    def test_alert_correlation_validation_invalid_score(self):
        """Test alert correlation validation with invalid score."""
        with pytest.raises(
            ValueError, match="Correlation score must be between 0.0 and 1.0"
        ):
            AlertCorrelation(primary_alert_id="alert1", correlation_score=1.5)

        with pytest.raises(
            ValueError, match="Correlation score must be between 0.0 and 1.0"
        ):
            AlertCorrelation(primary_alert_id="alert1", correlation_score=-0.1)


class TestMLNoiseFeatures:
    """Test MLNoiseFeatures class."""

    def test_ml_noise_features_initialization(self):
        """Test ML noise features initialization."""
        features = MLNoiseFeatures(
            alert_frequency=5.0,
            alert_duration=30.0,
            time_between_alerts=120.0,
            correlation_with_other_alerts=0.7,
            system_load=0.8,
            data_quality_score=0.9,
            false_positive_rate=0.1,
            user_feedback_score=0.8,
            alert_resolution_time=45.0,
            seasonal_factor=0.3,
        )

        assert features.alert_frequency == 5.0
        assert features.alert_duration == 30.0
        assert features.time_between_alerts == 120.0
        assert features.correlation_with_other_alerts == 0.7
        assert features.system_load == 0.8
        assert features.data_quality_score == 0.9
        assert features.false_positive_rate == 0.1
        assert features.user_feedback_score == 0.8
        assert features.alert_resolution_time == 45.0
        assert features.seasonal_factor == 0.3

    def test_ml_noise_features_default_values(self):
        """Test ML noise features with default values."""
        features = MLNoiseFeatures()

        assert features.alert_frequency == 0.0
        assert features.alert_duration == 0.0
        assert features.time_between_alerts == 0.0
        assert features.correlation_with_other_alerts == 0.0
        assert features.system_load == 0.0
        assert features.data_quality_score == 0.0
        assert features.false_positive_rate == 0.0
        assert features.user_feedback_score == 0.0
        assert features.alert_resolution_time == 0.0
        assert features.seasonal_factor == 0.0

    def test_ml_noise_features_to_dict(self):
        """Test converting ML noise features to dictionary."""
        features = MLNoiseFeatures(
            alert_frequency=5.0, alert_duration=30.0, time_between_alerts=120.0
        )

        result = features.to_dict()

        assert isinstance(result, dict)
        assert result["alert_frequency"] == 5.0
        assert result["alert_duration"] == 30.0
        assert result["time_between_alerts"] == 120.0
        assert len(result) == 10  # All features included


class TestAlertInitialization:
    """Test Alert initialization and validation."""

    def test_alert_initialization_with_required_fields(self):
        """Test alert initialization with required fields."""
        condition = AlertCondition(
            metric_name="cpu_usage", operator="gt", threshold=80.0
        )

        alert = Alert(
            name="High CPU Usage",
            description="CPU usage is above threshold",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        assert alert.name == "High CPU Usage"
        assert alert.description == "CPU usage is above threshold"
        assert alert.alert_type == AlertType.SYSTEM_HEALTH
        assert alert.severity == AlertSeverity.HIGH
        assert alert.condition == condition
        assert alert.created_by == "admin@example.com"
        assert isinstance(alert.id, type(uuid4()))
        assert isinstance(alert.created_at, datetime)
        assert alert.status == AlertStatus.ACTIVE
        assert alert.triggered_at is None
        assert alert.acknowledged_at is None
        assert alert.resolved_at is None
        assert alert.acknowledged_by is None
        assert alert.resolved_by is None
        assert alert.source == ""
        assert alert.tags == []
        assert alert.metadata == {}
        assert alert.notifications == []
        assert alert.suppression_rules == {}
        assert alert.escalation_rules == {}

    def test_alert_initialization_with_all_fields(self):
        """Test alert initialization with all fields."""
        condition = AlertCondition(
            metric_name="memory_usage", operator="gte", threshold=90.0
        )

        notification = AlertNotification(
            channel=NotificationChannel.EMAIL, recipient="user@example.com"
        )

        alert = Alert(
            name="Memory Alert",
            description="Memory usage is critical",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.CRITICAL,
            condition=condition,
            created_by="admin@example.com",
            source="monitoring_system",
            tags=["production", "critical"],
            metadata={"environment": "prod"},
            notifications=[notification],
            suppression_rules={"enabled": True},
            escalation_rules={"escalation_time_minutes": 30},
        )

        assert alert.source == "monitoring_system"
        assert alert.tags == ["production", "critical"]
        assert alert.metadata == {"environment": "prod"}
        assert len(alert.notifications) == 1
        assert alert.suppression_rules == {"enabled": True}
        assert alert.escalation_rules == {"escalation_time_minutes": 30}

    def test_alert_validation_empty_name(self):
        """Test alert validation with empty name."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        with pytest.raises(ValueError, match="Alert name cannot be empty"):
            Alert(
                name="",
                description="Test description",
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.HIGH,
                condition=condition,
                created_by="admin@example.com",
            )

    def test_alert_validation_empty_description(self):
        """Test alert validation with empty description."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        with pytest.raises(ValueError, match="Alert description cannot be empty"):
            Alert(
                name="Test Alert",
                description="",
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.HIGH,
                condition=condition,
                created_by="admin@example.com",
            )

    def test_alert_validation_invalid_alert_type(self):
        """Test alert validation with invalid alert type."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        with pytest.raises(TypeError, match="Alert type must be AlertType"):
            Alert(
                name="Test Alert",
                description="Test description",
                alert_type="invalid_type",
                severity=AlertSeverity.HIGH,
                condition=condition,
                created_by="admin@example.com",
            )

    def test_alert_validation_invalid_severity(self):
        """Test alert validation with invalid severity."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        with pytest.raises(TypeError, match="Severity must be AlertSeverity"):
            Alert(
                name="Test Alert",
                description="Test description",
                alert_type=AlertType.SYSTEM_HEALTH,
                severity="invalid_severity",
                condition=condition,
                created_by="admin@example.com",
            )

    def test_alert_validation_invalid_condition(self):
        """Test alert validation with invalid condition."""
        with pytest.raises(TypeError, match="Condition must be AlertCondition"):
            Alert(
                name="Test Alert",
                description="Test description",
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.HIGH,
                condition="invalid_condition",
                created_by="admin@example.com",
            )

    def test_alert_validation_empty_created_by(self):
        """Test alert validation with empty created_by."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        with pytest.raises(ValueError, match="Created by cannot be empty"):
            Alert(
                name="Test Alert",
                description="Test description",
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.HIGH,
                condition=condition,
                created_by="",
            )


class TestAlertProperties:
    """Test Alert properties."""

    def test_alert_status_properties(self):
        """Test alert status properties."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
            status=AlertStatus.ACTIVE,
        )

        assert alert.is_active is True
        assert alert.is_acknowledged is False
        assert alert.is_resolved is False

        alert.status = AlertStatus.ACKNOWLEDGED
        assert alert.is_active is False
        assert alert.is_acknowledged is True
        assert alert.is_resolved is False

        alert.status = AlertStatus.RESOLVED
        assert alert.is_active is False
        assert alert.is_acknowledged is False
        assert alert.is_resolved is True

    def test_alert_is_critical_property(self):
        """Test is_critical property."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.CRITICAL,
            condition=condition,
            created_by="admin@example.com",
        )

        assert alert.is_critical is True

        alert.severity = AlertSeverity.HIGH
        assert alert.is_critical is False

    def test_alert_duration_minutes_property(self):
        """Test duration_minutes property."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        # No triggered_at set
        assert alert.duration_minutes is None

        # Set triggered_at
        alert.triggered_at = datetime.utcnow() - timedelta(minutes=30)
        duration = alert.duration_minutes
        assert duration is not None
        assert duration >= 29  # Allow for small time differences
        assert duration <= 31

        # Set resolved_at
        alert.resolved_at = alert.triggered_at + timedelta(minutes=20)
        assert alert.duration_minutes == 20.0

    def test_alert_response_time_minutes_property(self):
        """Test response_time_minutes property."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        # No times set
        assert alert.response_time_minutes is None

        # Set triggered_at but not acknowledged_at
        alert.triggered_at = datetime.utcnow()
        assert alert.response_time_minutes is None

        # Set acknowledged_at
        alert.acknowledged_at = alert.triggered_at + timedelta(minutes=10)
        assert alert.response_time_minutes == 10.0

    def test_alert_resolution_time_minutes_property(self):
        """Test resolution_time_minutes property."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        # No times set
        assert alert.resolution_time_minutes is None

        # Set triggered_at but not resolved_at
        alert.triggered_at = datetime.utcnow()
        assert alert.resolution_time_minutes is None

        # Set resolved_at
        alert.resolved_at = alert.triggered_at + timedelta(minutes=45)
        assert alert.resolution_time_minutes == 45.0


class TestAlertLifecycle:
    """Test alert lifecycle methods."""

    def test_alert_trigger(self):
        """Test triggering an alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.trigger("monitoring_system", {"cpu_value": 85.0})

        assert alert.status == AlertStatus.ACTIVE
        assert alert.triggered_at is not None
        assert isinstance(alert.triggered_at, datetime)
        assert alert.metadata["triggered_by"] == "monitoring_system"
        assert alert.metadata["trigger_context"] == {"cpu_value": 85.0}
        assert alert.metadata["trigger_count"] == 1

    def test_alert_trigger_multiple_times(self):
        """Test triggering an alert multiple times."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.trigger("system1")
        alert.trigger("system2")

        assert alert.metadata["trigger_count"] == 2
        assert alert.metadata["triggered_by"] == "system2"

    def test_alert_trigger_after_resolution(self):
        """Test triggering an alert after it was resolved."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        # Trigger and resolve
        alert.trigger("system")
        alert.resolve("admin", "Fixed the issue")

        # Trigger again
        alert.trigger("system")

        assert alert.status == AlertStatus.ACTIVE
        assert alert.acknowledged_at is None
        assert alert.resolved_at is None
        assert alert.acknowledged_by is None
        assert alert.resolved_by is None

    def test_alert_acknowledge(self):
        """Test acknowledging an alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.trigger("system")
        alert.acknowledge("engineer@example.com", "Investigating the issue")

        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_at is not None
        assert alert.acknowledged_by == "engineer@example.com"
        assert alert.metadata["acknowledgment_notes"] == "Investigating the issue"

    def test_alert_acknowledge_non_active(self):
        """Test acknowledging a non-active alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
            status=AlertStatus.RESOLVED,
        )

        with pytest.raises(ValueError, match="Can only acknowledge active alerts"):
            alert.acknowledge("engineer@example.com")

    def test_alert_resolve(self):
        """Test resolving an alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.trigger("system")
        alert.acknowledge("engineer@example.com")
        alert.resolve("engineer@example.com", "Issue resolved by restarting service")

        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None
        assert alert.resolved_by == "engineer@example.com"
        assert (
            alert.metadata["resolution_notes"] == "Issue resolved by restarting service"
        )
        assert "total_duration_minutes" in alert.metadata
        assert "response_time_minutes" in alert.metadata
        assert "resolution_time_minutes" in alert.metadata

    def test_alert_resolve_already_resolved(self):
        """Test resolving an already resolved alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
            status=AlertStatus.RESOLVED,
        )

        with pytest.raises(ValueError, match="Alert is already resolved"):
            alert.resolve("engineer@example.com")

    def test_alert_suppress(self):
        """Test suppressing an alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.suppress("admin@example.com", "Maintenance window", 60)

        assert alert.status == AlertStatus.SUPPRESSED
        assert alert.metadata["suppressed_by"] == "admin@example.com"
        assert alert.metadata["suppression_reason"] == "Maintenance window"
        assert "suppressed_at" in alert.metadata
        assert "suppression_expires_at" in alert.metadata

    def test_alert_unsuppress(self):
        """Test unsuppressing an alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.suppress("admin@example.com", "Maintenance window")
        alert.unsuppress()

        assert alert.status == AlertStatus.ACTIVE
        assert "suppressed_by" not in alert.metadata
        assert "suppression_reason" not in alert.metadata

    def test_alert_expire(self):
        """Test expiring an alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.expire()

        assert alert.status == AlertStatus.EXPIRED
        assert "expired_at" in alert.metadata


class TestAlertNotificationManagement:
    """Test alert notification management."""

    def test_add_notification(self):
        """Test adding a notification to an alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        notification = AlertNotification(
            channel=NotificationChannel.EMAIL, recipient="user@example.com"
        )

        alert.add_notification(notification)

        assert len(alert.notifications) == 1
        assert alert.notifications[0].alert_id == alert.id
        assert alert.notifications[0].channel == NotificationChannel.EMAIL
        assert alert.notifications[0].recipient == "user@example.com"

    def test_get_notification_summary(self):
        """Test getting notification summary."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        # Add notifications with different statuses
        notifications = [
            AlertNotification(
                channel=NotificationChannel.EMAIL,
                recipient="user1@example.com",
                status="pending",
            ),
            AlertNotification(
                channel=NotificationChannel.EMAIL,
                recipient="user2@example.com",
                status="sent",
            ),
            AlertNotification(
                channel=NotificationChannel.SLACK,
                recipient="@channel",
                status="delivered",
            ),
            AlertNotification(
                channel=NotificationChannel.SMS,
                recipient="+1234567890",
                status="failed",
            ),
        ]

        for notification in notifications:
            alert.add_notification(notification)

        summary = alert.get_notification_summary()

        assert summary["pending"] == 1
        assert summary["sent"] == 1
        assert summary["delivered"] == 1
        assert summary["failed"] == 1

    def test_get_notification_summary_empty(self):
        """Test getting notification summary with no notifications."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        summary = alert.get_notification_summary()

        assert summary["pending"] == 0
        assert summary["sent"] == 0
        assert summary["delivered"] == 0
        assert summary["failed"] == 0


class TestAlertEscalationAndSupression:
    """Test alert escalation and suppression logic."""

    def test_should_escalate_true(self):
        """Test should_escalate returns True when conditions are met."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        # Set escalation rules
        alert.set_escalation_rules(30, ["manager@example.com"])

        # Trigger alert 35 minutes ago
        alert.triggered_at = datetime.utcnow() - timedelta(minutes=35)
        alert.status = AlertStatus.ACTIVE

        assert alert.should_escalate() is True

    def test_should_escalate_false_already_acknowledged(self):
        """Test should_escalate returns False when alert is acknowledged."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.set_escalation_rules(30, ["manager@example.com"])
        alert.triggered_at = datetime.utcnow() - timedelta(minutes=35)
        alert.acknowledged_at = datetime.utcnow() - timedelta(minutes=5)
        alert.status = AlertStatus.ACTIVE

        assert alert.should_escalate() is False

    def test_should_escalate_false_not_enough_time(self):
        """Test should_escalate returns False when not enough time has passed."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.set_escalation_rules(30, ["manager@example.com"])
        alert.triggered_at = datetime.utcnow() - timedelta(minutes=15)
        alert.status = AlertStatus.ACTIVE

        assert alert.should_escalate() is False

    def test_should_auto_resolve_true(self):
        """Test should_auto_resolve returns True when conditions are met."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.metadata["auto_resolve_time_minutes"] = 60
        alert.triggered_at = datetime.utcnow() - timedelta(minutes=65)

        assert alert.should_auto_resolve() is True

    def test_should_auto_resolve_false(self):
        """Test should_auto_resolve returns False when conditions are not met."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.metadata["auto_resolve_time_minutes"] = 60
        alert.triggered_at = datetime.utcnow() - timedelta(minutes=30)

        assert alert.should_auto_resolve() is False

    def test_is_suppression_expired_true(self):
        """Test is_suppression_expired returns True when suppression has expired."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        # Suppress with expiry time in the past
        past_time = datetime.utcnow() - timedelta(minutes=10)
        alert.status = AlertStatus.SUPPRESSED
        alert.metadata["suppression_expires_at"] = past_time.isoformat()

        assert alert.is_suppression_expired() is True

    def test_is_suppression_expired_false(self):
        """Test is_suppression_expired returns False when suppression is still valid."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        # Suppress with expiry time in the future
        future_time = datetime.utcnow() + timedelta(minutes=10)
        alert.status = AlertStatus.SUPPRESSED
        alert.metadata["suppression_expires_at"] = future_time.isoformat()

        assert alert.is_suppression_expired() is False

    def test_set_escalation_rules(self):
        """Test setting escalation rules."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.set_escalation_rules(30, ["manager@example.com", "director@example.com"])

        assert alert.escalation_rules["escalation_time_minutes"] == 30
        assert alert.escalation_rules["escalation_contacts"] == [
            "manager@example.com",
            "director@example.com",
        ]
        assert alert.escalation_rules["escalation_enabled"] is True

    def test_set_suppression_rules(self):
        """Test setting suppression rules."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.set_suppression_rules(10, 5)

        assert alert.suppression_rules["duplicate_window_minutes"] == 10
        assert alert.suppression_rules["max_notifications_per_hour"] == 5
        assert alert.suppression_rules["suppression_enabled"] is True


class TestAlertUtilityMethods:
    """Test alert utility methods."""

    def test_add_tag(self):
        """Test adding tags to alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        alert.add_tag("production")
        alert.add_tag("critical")

        assert "production" in alert.tags
        assert "critical" in alert.tags
        assert len(alert.tags) == 2

    def test_add_duplicate_tag(self):
        """Test adding duplicate tags."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
            tags=["production"],
        )

        alert.add_tag("production")  # Duplicate

        assert alert.tags == ["production"]  # No duplicate added

    def test_remove_tag(self):
        """Test removing tags from alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
            tags=["production", "critical", "hardware"],
        )

        alert.remove_tag("critical")

        assert "critical" not in alert.tags
        assert "production" in alert.tags
        assert "hardware" in alert.tags
        assert len(alert.tags) == 2

    def test_update_condition(self):
        """Test updating alert condition."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        new_condition = AlertCondition(metric_name="cpu", operator="gt", threshold=90.0)
        alert.update_condition(new_condition)

        assert alert.condition.threshold == 90.0
        assert "condition_updated_at" in alert.metadata

    def test_get_info(self):
        """Test getting comprehensive alert information."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.CRITICAL,
            condition=condition,
            created_by="admin@example.com",
            source="monitoring",
            tags=["production"],
        )

        alert.trigger("system")

        info = alert.get_info()

        assert info["name"] == "Test Alert"
        assert info["description"] == "Test description"
        assert info["alert_type"] == "system_health"
        assert info["severity"] == "critical"
        assert info["status"] == "active"
        assert info["created_by"] == "admin@example.com"
        assert info["source"] == "monitoring"
        assert info["tags"] == ["production"]
        assert "id" in info
        assert "created_at" in info
        assert "triggered_at" in info
        assert "condition" in info
        assert "notification_summary" in info
        assert "should_escalate" in info

    def test_get_timeline(self):
        """Test getting alert timeline."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        # Add some events
        alert.trigger("system")
        alert.acknowledge("engineer@example.com", "Looking into it")
        alert.resolve("engineer@example.com", "Fixed the issue")

        # Add notification
        notification = AlertNotification(
            channel=NotificationChannel.EMAIL, recipient="user@example.com"
        )
        notification.mark_sent()
        alert.add_notification(notification)

        timeline = alert.get_timeline()

        assert (
            len(timeline) >= 4
        )  # created, triggered, acknowledged, resolved, notification
        assert timeline[0]["event"] == "created"
        assert timeline[1]["event"] == "triggered"
        assert timeline[2]["event"] == "acknowledged"
        assert timeline[3]["event"] == "resolved"

        # Check that timeline is sorted by timestamp
        timestamps = [event["timestamp"] for event in timeline]
        assert timestamps == sorted(timestamps)

    def test_str_representation(self):
        """Test string representation of alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        str_repr = str(alert)
        assert "Test Alert" in str_repr
        assert "high" in str_repr
        assert "active" in str_repr
        assert "system_health" in str_repr

    def test_repr_representation(self):
        """Test repr representation of alert."""
        condition = AlertCondition(metric_name="cpu", operator="gt", threshold=80.0)

        alert = Alert(
            name="Test Alert",
            description="Test description",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.HIGH,
            condition=condition,
            created_by="admin@example.com",
        )

        repr_str = repr(alert)
        assert "Alert(" in repr_str
        assert "name='Test Alert'" in repr_str
        assert "severity=high" in repr_str
        assert "status=active" in repr_str
