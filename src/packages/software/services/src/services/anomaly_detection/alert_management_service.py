"""Application service for managing alerts and notifications."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from monorepo.domain.entities import (
    Alert,
    AlertCondition,
    AlertNotification,
    AlertSeverity,
    AlertStatus,
    AlertType,
    NotificationChannel,
)
from monorepo.domain.exceptions import AlertNotFoundError, InvalidAlertStateError
from monorepo.shared.protocols import (
    AlertNotificationRepositoryProtocol,
    AlertRepositoryProtocol,
)


class AlertManagementService:
    """Service for managing alerts and notifications."""

    def __init__(
        self,
        alert_repository: AlertRepositoryProtocol,
        notification_repository: AlertNotificationRepositoryProtocol,
    ):
        """Initialize the service.

        Args:
            alert_repository: Repository for alerts
            notification_repository: Repository for alert notifications
        """
        self.alert_repository = alert_repository
        self.notification_repository = notification_repository

    async def create_alert(
        self,
        name: str,
        description: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        condition: AlertCondition,
        created_by: str,
        source: str = "",
        tags: list[str] | None = None,
        escalation_time_minutes: int | None = None,
        escalation_contacts: list[str] | None = None,
    ) -> Alert:
        """Create a new alert.

        Args:
            name: Name of the alert
            description: Description of the alert
            alert_type: Type of alert
            severity: Severity level
            condition: Alert condition
            created_by: User creating the alert
            source: Source system or component
            tags: Tags for the alert
            escalation_time_minutes: Time before escalation
            escalation_contacts: Contacts for escalation

        Returns:
            Created alert
        """
        # Check if alert name already exists
        existing_alerts = await self.alert_repository.find_by_name(name)
        if existing_alerts:
            raise ValueError(f"Alert with name '{name}' already exists")

        alert = Alert(
            name=name,
            description=description,
            alert_type=alert_type,
            severity=severity,
            condition=condition,
            created_by=created_by,
            source=source,
            tags=tags or [],
        )

        # Set escalation rules if provided
        if escalation_time_minutes and escalation_contacts:
            alert.set_escalation_rules(escalation_time_minutes, escalation_contacts)

        # Set default suppression rules
        alert.set_suppression_rules()

        await self.alert_repository.save(alert)
        return alert

    async def trigger_alert(
        self,
        alert_id: UUID,
        triggered_by: str = "system",
        context: dict[str, Any] | None = None,
        send_notifications: bool = True,
    ) -> Alert:
        """Trigger an alert.

        Args:
            alert_id: ID of the alert to trigger
            triggered_by: Who/what triggered the alert
            context: Additional context about the trigger
            send_notifications: Whether to send notifications

        Returns:
            Updated alert
        """
        alert = await self.alert_repository.find_by_id(alert_id)
        if not alert:
            raise AlertNotFoundError(alert_id=alert_id)

        # Check if alert should be suppressed
        if await self._should_suppress_alert(alert):
            return alert

        # Trigger the alert
        alert.trigger(triggered_by, context)
        await self.alert_repository.save(alert)

        # Send notifications if requested
        if send_notifications:
            await self._send_alert_notifications(alert)

        return alert

    async def acknowledge_alert(
        self, alert_id: UUID, acknowledged_by: str, notes: str = ""
    ) -> Alert:
        """Acknowledge an alert.

        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: User acknowledging the alert
            notes: Acknowledgment notes

        Returns:
            Updated alert
        """
        alert = await self.alert_repository.find_by_id(alert_id)
        if not alert:
            raise AlertNotFoundError(alert_id=alert_id)

        if not alert.is_active:
            raise InvalidAlertStateError(
                alert_id=alert_id,
                operation="acknowledge",
                reason=f"Cannot acknowledge {alert.status.value} alert",
            )

        alert.acknowledge(acknowledged_by, notes)
        await self.alert_repository.save(alert)

        return alert

    async def resolve_alert(
        self, alert_id: UUID, resolved_by: str, resolution_notes: str = ""
    ) -> Alert:
        """Resolve an alert.

        Args:
            alert_id: ID of the alert to resolve
            resolved_by: User resolving the alert
            resolution_notes: Resolution notes

        Returns:
            Updated alert
        """
        alert = await self.alert_repository.find_by_id(alert_id)
        if not alert:
            raise AlertNotFoundError(alert_id=alert_id)

        if alert.is_resolved:
            raise InvalidAlertStateError(
                alert_id=alert_id,
                operation="resolve",
                reason="Alert is already resolved",
            )

        alert.resolve(resolved_by, resolution_notes)
        await self.alert_repository.save(alert)

        return alert

    async def suppress_alert(
        self,
        alert_id: UUID,
        suppressed_by: str,
        reason: str = "",
        duration_minutes: int | None = None,
    ) -> Alert:
        """Suppress an alert.

        Args:
            alert_id: ID of the alert to suppress
            suppressed_by: User suppressing the alert
            reason: Reason for suppression
            duration_minutes: How long to suppress (None for indefinite)

        Returns:
            Updated alert
        """
        alert = await self.alert_repository.find_by_id(alert_id)
        if not alert:
            raise AlertNotFoundError(alert_id=alert_id)

        alert.suppress(suppressed_by, reason, duration_minutes)
        await self.alert_repository.save(alert)

        return alert

    async def update_alert_condition(
        self, alert_id: UUID, new_condition: AlertCondition
    ) -> Alert:
        """Update an alert's condition.

        Args:
            alert_id: ID of the alert to update
            new_condition: New alert condition

        Returns:
            Updated alert
        """
        alert = await self.alert_repository.find_by_id(alert_id)
        if not alert:
            raise AlertNotFoundError(alert_id=alert_id)

        alert.update_condition(new_condition)
        await self.alert_repository.save(alert)

        return alert

    async def evaluate_alert_condition(
        self, alert_id: UUID, metric_value: float
    ) -> dict[str, Any]:
        """Evaluate an alert condition against a metric value.

        Args:
            alert_id: ID of the alert
            metric_value: Current metric value

        Returns:
            Evaluation result
        """
        alert = await self.alert_repository.find_by_id(alert_id)
        if not alert:
            raise AlertNotFoundError(alert_id=alert_id)

        condition_met = alert.condition.evaluate(metric_value)

        result = {
            "alert_id": str(alert_id),
            "alert_name": alert.name,
            "metric_value": metric_value,
            "threshold": alert.condition.threshold,
            "operator": alert.condition.operator,
            "condition_met": condition_met,
            "should_trigger": condition_met
            and alert.status not in [AlertStatus.SUPPRESSED],
            "current_status": alert.status.value,
        }

        # If condition is met and alert should trigger, trigger it
        if result["should_trigger"]:
            await self.trigger_alert(
                alert_id,
                context={
                    "metric_value": metric_value,
                    "threshold": alert.condition.threshold,
                },
            )
            result["action_taken"] = "alert_triggered"
        else:
            result["action_taken"] = "no_action"

        return result

    async def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
        alert_type: AlertType | None = None,
    ) -> list[Alert]:
        """Get all active alerts.

        Args:
            severity: Filter by severity (optional)
            alert_type: Filter by alert type (optional)

        Returns:
            List of active alerts
        """
        all_alerts = await self.alert_repository.find_by_status(AlertStatus.ACTIVE)

        filtered_alerts = all_alerts

        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]

        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a.alert_type == alert_type]

        return filtered_alerts

    async def get_alerts_requiring_escalation(self) -> list[Alert]:
        """Get alerts that need escalation.

        Returns:
            List of alerts requiring escalation
        """
        active_alerts = await self.alert_repository.find_by_status(AlertStatus.ACTIVE)
        return [alert for alert in active_alerts if alert.should_escalate()]

    async def escalate_alerts(self) -> dict[str, Any]:
        """Escalate alerts that meet escalation criteria.

        Returns:
            Escalation summary
        """
        alerts_to_escalate = await self.get_alerts_requiring_escalation()

        escalated_count = 0
        failed_escalations = []

        for alert in alerts_to_escalate:
            try:
                await self._escalate_alert(alert)
                escalated_count += 1
            except Exception as e:
                failed_escalations.append(
                    {
                        "alert_id": str(alert.id),
                        "alert_name": alert.name,
                        "error": str(e),
                    }
                )

        return {
            "total_alerts_evaluated": len(alerts_to_escalate),
            "successfully_escalated": escalated_count,
            "failed_escalations": failed_escalations,
            "escalation_timestamp": datetime.utcnow().isoformat(),
        }

    async def process_alert_conditions(
        self, metrics: dict[str, float]
    ) -> dict[str, Any]:
        """Process multiple metric values against all alert conditions.

        Args:
            metrics: Dictionary of metric_name -> value

        Returns:
            Processing summary
        """
        all_alerts = await self.alert_repository.find_all()

        processing_results = []
        triggered_alerts = []

        for alert in all_alerts:
            metric_name = alert.condition.metric_name

            if metric_name in metrics:
                metric_value = metrics[metric_name]

                try:
                    result = await self.evaluate_alert_condition(alert.id, metric_value)
                    processing_results.append(result)

                    if result["action_taken"] == "alert_triggered":
                        triggered_alerts.append(alert.name)

                except Exception as e:
                    processing_results.append(
                        {
                            "alert_id": str(alert.id),
                            "alert_name": alert.name,
                            "error": str(e),
                            "action_taken": "error",
                        }
                    )

        return {
            "processed_alerts": len(processing_results),
            "triggered_alerts": len(triggered_alerts),
            "triggered_alert_names": triggered_alerts,
            "processing_timestamp": datetime.utcnow().isoformat(),
            "detailed_results": processing_results,
        }

    async def get_alert_analytics(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> dict[str, Any]:
        """Get analytics data for alerts.

        Args:
            start_date: Start date for analytics (optional)
            end_date: End date for analytics (optional)

        Returns:
            Analytics data
        """
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)

        if not end_date:
            end_date = datetime.utcnow()

        all_alerts = await self.alert_repository.find_all()

        # Filter alerts by date range
        filtered_alerts = [
            alert for alert in all_alerts if start_date <= alert.created_at <= end_date
        ]

        # Calculate analytics
        total_alerts = len(filtered_alerts)
        active_alerts = len([a for a in filtered_alerts if a.is_active])
        resolved_alerts = len([a for a in filtered_alerts if a.is_resolved])

        # Severity breakdown
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len(
                [a for a in filtered_alerts if a.severity == severity]
            )

        # Type breakdown
        type_counts = {}
        for alert_type in AlertType:
            type_counts[alert_type.value] = len(
                [a for a in filtered_alerts if a.alert_type == alert_type]
            )

        # Response time analytics
        acknowledged_alerts = [
            a for a in filtered_alerts if a.response_time_minutes is not None
        ]
        avg_response_time = None
        if acknowledged_alerts:
            avg_response_time = sum(
                a.response_time_minutes for a in acknowledged_alerts
            ) / len(acknowledged_alerts)

        # Resolution time analytics
        resolved_with_time = [
            a for a in filtered_alerts if a.resolution_time_minutes is not None
        ]
        avg_resolution_time = None
        if resolved_with_time:
            avg_resolution_time = sum(
                a.resolution_time_minutes for a in resolved_with_time
            ) / len(resolved_with_time)

        return {
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "summary": {
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "resolved_alerts": resolved_alerts,
                "resolution_rate": (
                    resolved_alerts / total_alerts if total_alerts > 0 else 0
                ),
            },
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "response_metrics": {
                "average_response_time_minutes": avg_response_time,
                "average_resolution_time_minutes": avg_resolution_time,
                "alerts_with_response_time": len(acknowledged_alerts),
                "alerts_with_resolution_time": len(resolved_with_time),
            },
            "top_alert_sources": self._get_top_alert_sources(filtered_alerts),
            "alert_frequency": self._calculate_alert_frequency(
                filtered_alerts, start_date, end_date
            ),
        }

    async def cleanup_expired_alerts(self) -> dict[str, Any]:
        """Clean up expired alerts and suppressions.

        Returns:
            Cleanup summary
        """
        all_alerts = await self.alert_repository.find_all()

        expired_suppressions = 0
        auto_resolved = 0

        for alert in all_alerts:
            # Check for expired suppressions
            if alert.is_suppression_expired():
                alert.unsuppress()
                expired_suppressions += 1
                await self.alert_repository.save(alert)

            # Check for auto-resolution
            if alert.should_auto_resolve():
                alert.resolve("system", "Auto-resolved due to timeout")
                auto_resolved += 1
                await self.alert_repository.save(alert)

        return {
            "expired_suppressions_removed": expired_suppressions,
            "auto_resolved_alerts": auto_resolved,
            "cleanup_timestamp": datetime.utcnow().isoformat(),
        }

    async def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if an alert should be suppressed to prevent spam."""
        if not alert.suppression_rules.get("suppression_enabled", False):
            return False

        # Check for recent duplicate alerts
        duplicate_window = alert.suppression_rules.get("duplicate_window_minutes", 5)
        datetime.utcnow() - timedelta(minutes=duplicate_window)

        # For now, assume no recent duplicates (would need to check repository)
        return False

    async def _send_alert_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        # For now, create a basic email notification
        # In a real implementation, this would integrate with notification services

        notification = AlertNotification(
            alert_id=alert.id,
            channel=NotificationChannel.EMAIL,
            recipient="alerts@company.com",
            status="pending",
        )

        alert.add_notification(notification)
        await self.notification_repository.save(notification)
        await self.alert_repository.save(alert)

    async def _escalate_alert(self, alert: Alert) -> None:
        """Escalate an alert to higher-level contacts."""
        escalation_contacts = alert.escalation_rules.get("escalation_contacts", [])

        for contact in escalation_contacts:
            escalation_notification = AlertNotification(
                alert_id=alert.id,
                channel=NotificationChannel.EMAIL,
                recipient=contact,
                status="pending",
            )
            escalation_notification.metadata["escalation"] = True

            alert.add_notification(escalation_notification)
            await self.notification_repository.save(escalation_notification)

        # Mark alert metadata to track escalation
        alert.metadata["escalated_at"] = datetime.utcnow().isoformat()
        alert.metadata["escalation_level"] = (
            alert.metadata.get("escalation_level", 0) + 1
        )

        await self.alert_repository.save(alert)

    def _get_top_alert_sources(
        self, alerts: list[Alert], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Get top alert sources by frequency."""
        source_counts = {}

        for alert in alerts:
            source = alert.source or "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1

        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {"source": source, "count": count}
            for source, count in sorted_sources[:limit]
        ]

    def _calculate_alert_frequency(
        self, alerts: list[Alert], start_date: datetime, end_date: datetime
    ) -> dict[str, float]:
        """Calculate alert frequency statistics."""
        time_span = end_date - start_date
        days = time_span.total_seconds() / (24 * 3600)

        if days == 0:
            return {"daily_average": len(alerts), "weekly_average": len(alerts) * 7}

        daily_avg = len(alerts) / days
        weekly_avg = daily_avg * 7

        return {
            "daily_average": daily_avg,
            "weekly_average": weekly_avg,
            "total_in_period": len(alerts),
        }
