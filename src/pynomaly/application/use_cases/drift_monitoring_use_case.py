"""Use case for comprehensive drift monitoring and alerting.

This module orchestrates drift detection activities, manages monitoring
configurations, and handles drift alerts and notifications.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from pynomaly.application.services.drift_detection_service import DriftDetectionService
from pynomaly.domain.entities.drift_detection import (
    DriftAlert,
    DriftDetectionResult,
    DriftMonitoringStatus,
    DriftReport,
    DriftSeverity,
    DriftType,
    ModelMonitoringConfig,
    MonitoringStatus,
)
from pynomaly.infrastructure.monitoring.prometheus_metrics import get_metrics_service

logger = logging.getLogger(__name__)


class DriftMonitoringUseCase:
    """Use case for orchestrating drift monitoring and alerting."""

    def __init__(
        self,
        drift_detection_service: DriftDetectionService,
        detector_repository=None,
        alert_repository=None,
        notification_service=None,
        metrics_service=None,
    ):
        """Initialize drift monitoring use case.

        Args:
            drift_detection_service: Core drift detection service
            detector_repository: Repository for detector data
            alert_repository: Repository for drift alerts
            notification_service: Service for sending notifications
            metrics_service: Metrics collection service
        """
        self.drift_service = drift_detection_service
        self.detector_repository = detector_repository
        self.alert_repository = alert_repository
        self.notification_service = notification_service
        self.metrics_service = metrics_service or get_metrics_service()

        # Active monitoring tasks
        self.monitoring_tasks: dict[str, asyncio.Task] = {}

        logger.info("Drift monitoring use case initialized")

    async def configure_monitoring(
        self, detector_id: str, config: ModelMonitoringConfig
    ) -> DriftMonitoringStatus:
        """Configure drift monitoring for a detector.

        Args:
            detector_id: ID of the detector to monitor
            config: Monitoring configuration

        Returns:
            Initial monitoring status
        """
        try:
            # Validate configuration
            if config.check_interval_hours < 1:
                raise ValueError("Check interval must be at least 1 hour")

            if config.min_sample_size < 50:
                raise ValueError("Minimum sample size must be at least 50")

            # Set up monitoring in drift service
            status = await self.drift_service.setup_monitoring(detector_id, config)

            # Start monitoring task if enabled
            if config.enabled:
                await self._start_monitoring_task(detector_id)

            # Record metrics
            if self.metrics_service:
                self.metrics_service.record_error(
                    error_type="monitoring_configured",
                    component="drift_monitoring",
                    severity="info",
                )

            logger.info(f"Drift monitoring configured for detector {detector_id}")
            return status

        except Exception as e:
            logger.error(f"Failed to configure monitoring for {detector_id}: {e}")
            raise

    async def perform_drift_check(
        self,
        detector_id: str,
        reference_data: np.ndarray | None = None,
        current_data: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> DriftDetectionResult:
        """Perform immediate drift check for a detector.

        Args:
            detector_id: ID of the detector
            reference_data: Reference dataset (optional)
            current_data: Current dataset (optional)
            feature_names: Feature names (optional)

        Returns:
            Drift detection result
        """
        try:
            # Get monitoring configuration
            config = self.drift_service.monitoring_configs.get(detector_id)
            if not config:
                raise ValueError(
                    f"No monitoring configuration for detector {detector_id}"
                )

            # Get data if not provided
            if reference_data is None or current_data is None:
                reference_data, current_data = await self._get_monitoring_data(
                    detector_id, config
                )

            # Perform drift detection
            detection_methods = config.enabled_methods
            result = await self.drift_service.detect_data_drift(
                detector_id=detector_id,
                reference_data=reference_data,
                current_data=current_data,
                feature_names=feature_names,
                detection_methods=detection_methods,
            )

            # Update monitoring status
            await self._update_monitoring_status(detector_id, result)

            # Handle alerts if drift detected
            if result.drift_detected and config.should_alert(result.severity):
                await self._handle_drift_alert(result, config)

            # Record metrics
            if self.metrics_service:
                self.metrics_service.record_error(
                    error_type="drift_check_performed",
                    component="drift_monitoring",
                    severity=(
                        "info" if not result.drift_detected else result.severity.value
                    ),
                )

            logger.info(
                f"Drift check completed for {detector_id}: "
                f"drift={result.drift_detected}, severity={result.severity.value}"
            )

            return result

        except Exception as e:
            logger.error(f"Drift check failed for {detector_id}: {e}")

            # Update monitoring status with error
            status = self.drift_service.monitoring_status.get(detector_id)
            if status:
                status.update_check_status(success=False, error=str(e))

            raise

    async def check_performance_drift(
        self,
        detector_id: str,
        reference_metrics: dict[str, float],
        current_metrics: dict[str, float],
        threshold: float = 0.05,
    ) -> DriftDetectionResult:
        """Check for performance drift in a detector.

        Args:
            detector_id: ID of the detector
            reference_metrics: Reference performance metrics
            current_metrics: Current performance metrics
            threshold: Minimum change to consider drift

        Returns:
            Performance drift detection result
        """
        try:
            result = await self.drift_service.detect_performance_drift(
                detector_id=detector_id,
                reference_metrics=reference_metrics,
                current_metrics=current_metrics,
                threshold=threshold,
            )

            # Get monitoring configuration
            config = self.drift_service.monitoring_configs.get(detector_id)

            # Handle alerts if drift detected and monitoring is configured
            if (
                result.drift_detected
                and config
                and config.should_alert(result.severity)
            ):
                await self._handle_drift_alert(result, config)

            logger.info(
                f"Performance drift check completed for {detector_id}: "
                f"drift={result.drift_detected}"
            )

            return result

        except Exception as e:
            logger.error(f"Performance drift check failed for {detector_id}: {e}")
            raise

    async def get_monitoring_status(
        self, detector_id: str
    ) -> DriftMonitoringStatus | None:
        """Get current monitoring status for a detector.

        Args:
            detector_id: ID of the detector

        Returns:
            Monitoring status or None if not found
        """
        return await self.drift_service.get_monitoring_status(detector_id)

    async def pause_monitoring(self, detector_id: str) -> bool:
        """Pause drift monitoring for a detector.

        Args:
            detector_id: ID of the detector

        Returns:
            True if successfully paused
        """
        try:
            # Pause in drift service
            success = await self.drift_service.pause_monitoring(detector_id)

            # Stop monitoring task
            if detector_id in self.monitoring_tasks:
                self.monitoring_tasks[detector_id].cancel()
                del self.monitoring_tasks[detector_id]

            if success:
                logger.info(f"Monitoring paused for detector {detector_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to pause monitoring for {detector_id}: {e}")
            return False

    async def resume_monitoring(self, detector_id: str) -> bool:
        """Resume drift monitoring for a detector.

        Args:
            detector_id: ID of the detector

        Returns:
            True if successfully resumed
        """
        try:
            # Resume in drift service
            success = await self.drift_service.resume_monitoring(detector_id)

            # Start monitoring task
            if success:
                await self._start_monitoring_task(detector_id)
                logger.info(f"Monitoring resumed for detector {detector_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to resume monitoring for {detector_id}: {e}")
            return False

    async def stop_monitoring(self, detector_id: str) -> bool:
        """Stop drift monitoring for a detector.

        Args:
            detector_id: ID of the detector

        Returns:
            True if successfully stopped
        """
        try:
            # Stop monitoring task
            if detector_id in self.monitoring_tasks:
                self.monitoring_tasks[detector_id].cancel()
                del self.monitoring_tasks[detector_id]

            # Update status
            status = self.drift_service.monitoring_status.get(detector_id)
            if status:
                status.status = MonitoringStatus.STOPPED

            logger.info(f"Monitoring stopped for detector {detector_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop monitoring for {detector_id}: {e}")
            return False

    async def list_active_monitors(self) -> list[str]:
        """List all actively monitored detectors.

        Returns:
            List of detector IDs being monitored
        """
        active_monitors = []

        for detector_id, status in self.drift_service.monitoring_status.items():
            if status.status == MonitoringStatus.ACTIVE:
                active_monitors.append(detector_id)

        return active_monitors

    async def get_drift_alerts(
        self,
        detector_id: str | None = None,
        severity: DriftSeverity | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[DriftAlert]:
        """Get drift alerts with optional filtering.

        Args:
            detector_id: Filter by detector ID (optional)
            severity: Filter by severity (optional)
            active_only: Only return active alerts
            limit: Maximum number of alerts to return

        Returns:
            List of drift alerts
        """
        # Mock implementation - in real system would query alert repository
        alerts = []

        # For demonstration, create a sample alert
        if detector_id:
            sample_alert = DriftAlert(
                detector_id=detector_id,
                alert_type=DriftType.DATA_DRIFT,
                severity=severity or DriftSeverity.MEDIUM,
                title="Sample Drift Alert",
                message="This is a sample drift alert for demonstration",
            )
            alerts.append(sample_alert)

        logger.info(f"Retrieved {len(alerts)} drift alerts")
        return alerts[:limit]

    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge a drift alert.

        Args:
            alert_id: ID of the alert to acknowledge
            user: User acknowledging the alert

        Returns:
            True if successfully acknowledged
        """
        try:
            # In real implementation, would retrieve alert from repository
            # For now, just log the acknowledgment
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False

    async def resolve_alert(
        self, alert_id: str, user: str, action: str | None = None
    ) -> bool:
        """Resolve a drift alert.

        Args:
            alert_id: ID of the alert to resolve
            user: User resolving the alert
            action: Action taken to resolve (optional)

        Returns:
            True if successfully resolved
        """
        try:
            # In real implementation, would update alert in repository
            logger.info(f"Alert {alert_id} resolved by {user}: {action}")
            return True

        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False

    async def generate_drift_report(
        self, detector_id: str, period_days: int = 30
    ) -> DriftReport:
        """Generate comprehensive drift monitoring report.

        Args:
            detector_id: ID of the detector
            period_days: Report period in days

        Returns:
            Drift monitoring report
        """
        try:
            period_end = datetime.now()
            period_start = period_end - timedelta(days=period_days)

            report = await self.drift_service.generate_drift_report(
                detector_id=detector_id,
                period_start=period_start,
                period_end=period_end,
            )

            logger.info(f"Drift report generated for detector {detector_id}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate drift report for {detector_id}: {e}")
            raise

    async def get_system_health(self) -> dict[str, Any]:
        """Get overall drift monitoring system health.

        Returns:
            System health summary
        """
        try:
            total_monitors = len(self.drift_service.monitoring_status)
            active_monitors = len(
                [
                    s
                    for s in self.drift_service.monitoring_status.values()
                    if s.status == MonitoringStatus.ACTIVE
                ]
            )
            error_monitors = len(
                [
                    s
                    for s in self.drift_service.monitoring_status.values()
                    if s.status == MonitoringStatus.ERROR
                ]
            )

            # Calculate average health score
            health_scores = [
                s.overall_health_score
                for s in self.drift_service.monitoring_status.values()
            ]
            avg_health_score = np.mean(health_scores) if health_scores else 1.0

            # Get recent drift detections
            recent_drift_count = len(
                [
                    s
                    for s in self.drift_service.monitoring_status.values()
                    if s.last_drift_detected
                    and (datetime.now() - s.last_drift_detected).days <= 7
                ]
            )

            system_health = {
                "total_monitors": total_monitors,
                "active_monitors": active_monitors,
                "error_monitors": error_monitors,
                "average_health_score": float(avg_health_score),
                "recent_drift_detections": recent_drift_count,
                "monitoring_tasks_running": len(self.monitoring_tasks),
                "system_status": "healthy" if error_monitors == 0 else "degraded",
                "last_updated": datetime.now().isoformat(),
            }

            return system_health

        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "error": str(e),
                "system_status": "error",
                "last_updated": datetime.now().isoformat(),
            }

    # Private helper methods

    async def _start_monitoring_task(self, detector_id: str):
        """Start background monitoring task for a detector."""
        if detector_id in self.monitoring_tasks:
            # Cancel existing task
            self.monitoring_tasks[detector_id].cancel()

        # Create new monitoring task
        task = asyncio.create_task(self._monitoring_loop(detector_id))
        self.monitoring_tasks[detector_id] = task

        logger.info(f"Started monitoring task for detector {detector_id}")

    async def _monitoring_loop(self, detector_id: str):
        """Background monitoring loop for a detector."""
        logger.info(f"Starting monitoring loop for detector {detector_id}")

        try:
            while True:
                # Check if monitoring is due
                result = await self.drift_service.check_drift_monitoring(detector_id)

                if result:
                    logger.info(f"Scheduled drift check completed for {detector_id}")

                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes

        except asyncio.CancelledError:
            logger.info(f"Monitoring loop cancelled for detector {detector_id}")
        except Exception as e:
            logger.error(f"Monitoring loop error for detector {detector_id}: {e}")

            # Update status with error
            status = self.drift_service.monitoring_status.get(detector_id)
            if status:
                status.update_check_status(success=False, error=str(e))

    async def _get_monitoring_data(
        self, detector_id: str, config: ModelMonitoringConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get reference and current data for monitoring.

        Args:
            detector_id: ID of the detector
            config: Monitoring configuration

        Returns:
            Tuple of (reference_data, current_data)
        """
        # Mock implementation - in real system would fetch from data repository
        # Generate sample data for demonstration

        # Reference data (larger, stable dataset)
        reference_data = np.random.randn(config.min_sample_size * 2, 5)

        # Current data (smaller, potentially drifted dataset)
        current_data = np.random.randn(config.min_sample_size, 5)

        # Add some drift for demonstration
        if np.random.random() > 0.7:  # 30% chance of drift
            drift_magnitude = np.random.uniform(0.5, 2.0)
            current_data = current_data + drift_magnitude

        return reference_data, current_data

    async def _update_monitoring_status(
        self, detector_id: str, result: DriftDetectionResult
    ):
        """Update monitoring status based on drift detection result.

        Args:
            detector_id: ID of the detector
            result: Drift detection result
        """
        status = self.drift_service.monitoring_status.get(detector_id)
        if not status:
            return

        # Update drift detection count
        if result.drift_detected:
            status.drift_detections += 1
            status.last_drift_detected = datetime.now()

            # Decrease health score
            health_penalty = (
                0.1
                if result.severity == DriftSeverity.LOW
                else (
                    0.2
                    if result.severity == DriftSeverity.MEDIUM
                    else 0.3 if result.severity == DriftSeverity.HIGH else 0.5
                )
            )
            status.overall_health_score = max(
                0.0, status.overall_health_score - health_penalty
            )
        else:
            # Improve health score gradually
            status.overall_health_score = min(1.0, status.overall_health_score + 0.05)

        # Update check status
        status.update_check_status(success=True)

        # Schedule next check
        config = self.drift_service.monitoring_configs.get(detector_id)
        if config:
            status.schedule_next_check(config.check_interval_hours)

    async def _handle_drift_alert(
        self, result: DriftDetectionResult, config: ModelMonitoringConfig
    ):
        """Handle drift alert creation and notification.

        Args:
            result: Drift detection result
            config: Monitoring configuration
        """
        try:
            # Create alert
            alert = await self.drift_service._create_drift_alert(result, config)

            # Update monitoring status
            status = self.drift_service.monitoring_status.get(result.detector_id)
            if status:
                status.active_alerts += 1

            # Send notifications if configured
            if self.notification_service and config.notification_channels:
                await self._send_drift_notifications(alert, config)

            logger.info(f"Drift alert handled for detector {result.detector_id}")

        except Exception as e:
            logger.error(f"Failed to handle drift alert: {e}")

    async def _send_drift_notifications(
        self, alert: DriftAlert, config: ModelMonitoringConfig
    ):
        """Send drift alert notifications.

        Args:
            alert: Drift alert to send
            config: Monitoring configuration
        """
        try:
            for channel in config.notification_channels:
                if channel == "email":
                    await self._send_email_notification(alert)
                elif channel == "slack":
                    await self._send_slack_notification(alert)
                elif channel == "webhook":
                    await self._send_webhook_notification(alert)

            logger.info(f"Notifications sent for alert {alert.id}")

        except Exception as e:
            logger.error(f"Failed to send notifications for alert {alert.id}: {e}")

    async def _send_email_notification(self, alert: DriftAlert):
        """Send email notification for drift alert."""
        # Mock implementation
        logger.info(f"Email notification sent for alert {alert.id}")

    async def _send_slack_notification(self, alert: DriftAlert):
        """Send Slack notification for drift alert."""
        # Mock implementation
        logger.info(f"Slack notification sent for alert {alert.id}")

    async def _send_webhook_notification(self, alert: DriftAlert):
        """Send webhook notification for drift alert."""
        # Mock implementation
        logger.info(f"Webhook notification sent for alert {alert.id}")
