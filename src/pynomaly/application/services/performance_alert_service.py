"""Performance degradation alert service."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pynomaly.domain.value_objects.performance_degradation_metrics import (
    DegradationAlert,
    DegradationSeverity,
    PerformanceDegradation,
)
from pynomaly.infrastructure.repositories.performance_degradation_repository import (
    PerformanceDegradationRepository,
)

logger = logging.getLogger(__name__)


class AlertChannel:
    """Base class for alert channels."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize alert channel.
        
        Args:
            name: Channel name
            config: Channel configuration
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
    
    async def send_alert(self, alert: DegradationAlert) -> bool:
        """Send alert through this channel.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        raise NotImplementedError("Subclasses must implement send_alert")
    
    def should_send(self, alert: DegradationAlert) -> bool:
        """Check if alert should be sent through this channel.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if should send
        """
        if not self.enabled:
            return False
        
        # Check severity threshold
        min_severity = self.config.get('min_severity', 'low')
        severity_order = ['low', 'medium', 'high', 'critical']
        
        alert_severity_idx = severity_order.index(alert.alert_level.value)
        min_severity_idx = severity_order.index(min_severity)
        
        return alert_severity_idx >= min_severity_idx


class ConsoleAlertChannel(AlertChannel):
    """Console logging alert channel."""
    
    async def send_alert(self, alert: DegradationAlert) -> bool:
        """Send alert to console."""
        try:
            severity_icons = {
                DegradationSeverity.CRITICAL: "🚨",
                DegradationSeverity.HIGH: "⚠️",
                DegradationSeverity.MEDIUM: "⚡",
                DegradationSeverity.LOW: "ℹ️"
            }
            
            icon = severity_icons.get(alert.alert_level, "📊")
            
            log_message = (
                f"{icon} PERFORMANCE DEGRADATION ALERT [{alert.alert_level.value.upper()}]\n"
                f"Model: {alert.model_id}\n"
                f"Metric: {alert.degradation.metric_name}\n"
                f"Current Value: {alert.degradation.current_value:.4f}\n"
                f"Baseline Value: {alert.degradation.baseline_value:.4f}\n"
                f"Degradation: {alert.degradation.degradation_percentage:.2f}%\n"
                f"Confidence: {alert.degradation.confidence_level:.2f}\n"
                f"Message: {alert.message}\n"
                f"Recommendations: {', '.join(alert.recommended_actions)}"
            )
            
            if alert.alert_level in [DegradationSeverity.CRITICAL, DegradationSeverity.HIGH]:
                logger.error(log_message)
            elif alert.alert_level == DegradationSeverity.MEDIUM:
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            return True
        except Exception as e:
            logger.error(f"Failed to send console alert: {e}")
            return False


class EmailAlertChannel(AlertChannel):
    """Email alert channel."""
    
    async def send_alert(self, alert: DegradationAlert) -> bool:
        """Send alert via email."""
        try:
            # In a real implementation, this would use SMTP
            # For now, we'll log the email content
            
            email_subject = f"Performance Alert: {alert.degradation.metric_name} degradation on model {alert.model_id}"
            
            email_body = f"""
Performance Degradation Alert

Alert Level: {alert.alert_level.value.upper()}
Model ID: {alert.model_id}
Metric: {alert.degradation.metric_name}
Degradation Type: {alert.degradation.degradation_type.value}

Performance Details:
- Current Value: {alert.degradation.current_value:.4f}
- Baseline Value: {alert.degradation.baseline_value:.4f}
- Degradation Amount: {alert.degradation.degradation_amount:.4f}
- Degradation Percentage: {alert.degradation.degradation_percentage:.2f}%
- Confidence Level: {alert.degradation.confidence_level:.2f}

Detection Details:
- Detection Method: {alert.degradation.detection_method}
- Threshold Violated: {alert.degradation.threshold_violated}
- Samples Used: {alert.degradation.samples_used}
- Detected At: {alert.degradation.detected_at}

Message:
{alert.message}

Recommended Actions:
{chr(10).join(f"- {action}" for action in alert.recommended_actions)}

Alert ID: {alert.alert_id}
Created At: {alert.created_at}
            """
            
            recipients = self.config.get('recipients', [])
            
            # Log email instead of actually sending
            logger.info(
                f"Email alert would be sent to {recipients}\n"
                f"Subject: {email_subject}\n"
                f"Body: {email_body}"
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class SlackAlertChannel(AlertChannel):
    """Slack webhook alert channel."""
    
    async def send_alert(self, alert: DegradationAlert) -> bool:
        """Send alert to Slack."""
        try:
            # In a real implementation, this would use Slack webhooks
            # For now, we'll log the Slack message
            
            severity_colors = {
                DegradationSeverity.CRITICAL: "#ff0000",
                DegradationSeverity.HIGH: "#ff8800",
                DegradationSeverity.MEDIUM: "#ffcc00",
                DegradationSeverity.LOW: "#32cd32"
            }
            
            color = severity_colors.get(alert.alert_level, "#808080")
            
            slack_payload = {
                "channel": self.config.get('channel', '#alerts'),
                "username": self.config.get('username', 'Performance Monitor'),
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"Performance Degradation: {alert.degradation.metric_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Model ID",
                                "value": alert.model_id,
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.alert_level.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": f"{alert.degradation.current_value:.4f}",
                                "short": True
                            },
                            {
                                "title": "Baseline Value",
                                "value": f"{alert.degradation.baseline_value:.4f}",
                                "short": True
                            },
                            {
                                "title": "Degradation",
                                "value": f"{alert.degradation.degradation_percentage:.2f}%",
                                "short": True
                            },
                            {
                                "title": "Confidence",
                                "value": f"{alert.degradation.confidence_level:.2f}",
                                "short": True
                            }
                        ],
                        "footer": "Performance Monitoring System",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            webhook_url = self.config.get('webhook_url')
            
            # Log Slack message instead of actually sending
            logger.info(
                f"Slack alert would be sent to {webhook_url}\n"
                f"Payload: {slack_payload}"
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class WebhookAlertChannel(AlertChannel):
    """Generic webhook alert channel."""
    
    async def send_alert(self, alert: DegradationAlert) -> bool:
        """Send alert via webhook."""
        try:
            # In a real implementation, this would make HTTP requests
            # For now, we'll log the webhook payload
            
            webhook_payload = {
                "alert_id": alert.alert_id,
                "model_id": alert.model_id,
                "alert_level": alert.alert_level.value,
                "message": alert.message,
                "degradation": {
                    "metric_name": alert.degradation.metric_name,
                    "degradation_type": alert.degradation.degradation_type.value,
                    "current_value": alert.degradation.current_value,
                    "baseline_value": alert.degradation.baseline_value,
                    "degradation_percentage": alert.degradation.degradation_percentage,
                    "confidence_level": alert.degradation.confidence_level,
                    "detected_at": alert.degradation.detected_at.isoformat()
                },
                "recommended_actions": alert.recommended_actions,
                "created_at": alert.created_at.isoformat(),
                "tags": alert.tags
            }
            
            webhook_url = self.config.get('url')
            
            # Log webhook instead of actually sending
            logger.info(
                f"Webhook alert would be sent to {webhook_url}\n"
                f"Payload: {webhook_payload}"
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class PerformanceAlertService:
    """Service for managing performance degradation alerts."""
    
    def __init__(
        self,
        repository: PerformanceDegradationRepository,
        alert_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the alert service.
        
        Args:
            repository: Repository for persistence
            alert_config: Alert configuration
        """
        self.repository = repository
        self.config = alert_config or {}
        self.channels = self._initialize_channels()
        self._alert_throttle = {}  # Track alert throttling
    
    def _initialize_channels(self) -> Dict[str, AlertChannel]:
        """Initialize alert channels."""
        channels = {}
        
        # Console channel (always available)
        console_config = self.config.get('console', {'enabled': True})
        channels['console'] = ConsoleAlertChannel('console', console_config)
        
        # Email channel
        email_config = self.config.get('email', {})
        if email_config.get('enabled', False):
            channels['email'] = EmailAlertChannel('email', email_config)
        
        # Slack channel
        slack_config = self.config.get('slack', {})
        if slack_config.get('enabled', False):
            channels['slack'] = SlackAlertChannel('slack', slack_config)
        
        # Webhook channels
        webhook_configs = self.config.get('webhooks', [])
        for i, webhook_config in enumerate(webhook_configs):
            if webhook_config.get('enabled', False):
                channels[f'webhook_{i}'] = WebhookAlertChannel(f'webhook_{i}', webhook_config)
        
        return channels
    
    async def create_alert(
        self,
        model_id: UUID,
        degradation: PerformanceDegradation,
        custom_message: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> DegradationAlert:
        """Create a degradation alert.
        
        Args:
            model_id: Model identifier
            degradation: Performance degradation
            custom_message: Custom alert message
            tags: Alert tags
            
        Returns:
            Created alert
        """
        alert_id = str(uuid4())
        
        # Generate default message if not provided
        if not custom_message:
            custom_message = self._generate_default_message(degradation)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(degradation)
        
        alert = DegradationAlert(
            alert_id=alert_id,
            model_id=str(model_id),
            degradation=degradation,
            alert_level=degradation.severity,
            message=custom_message,
            recommended_actions=recommendations,
            tags=tags or []
        )
        
        # Store alert
        await self.repository.store_alert(alert)
        
        return alert
    
    async def send_alert(self, alert: DegradationAlert) -> Dict[str, bool]:
        """Send alert through configured channels.
        
        Args:
            alert: Alert to send
            
        Returns:
            Dictionary of channel send results
        """
        results = {}
        
        # Check throttling
        if self._should_throttle_alert(alert):
            logger.info(f"Alert {alert.alert_id} throttled")
            return {}
        
        # Send through each channel
        for channel_name, channel in self.channels.items():
            if channel.should_send(alert):
                try:
                    success = await channel.send_alert(alert)
                    results[channel_name] = success
                    
                    if success:
                        logger.info(f"Alert {alert.alert_id} sent via {channel_name}")
                    else:
                        logger.warning(f"Failed to send alert {alert.alert_id} via {channel_name}")
                        
                except Exception as e:
                    logger.error(f"Error sending alert {alert.alert_id} via {channel_name}: {e}")
                    results[channel_name] = False
            else:
                logger.debug(f"Alert {alert.alert_id} filtered out for {channel_name}")
        
        # Update throttle tracking
        self._update_throttle_tracking(alert)
        
        return results
    
    async def process_degradations(
        self,
        model_id: UUID,
        degradations: List[PerformanceDegradation],
        send_alerts: bool = True
    ) -> List[DegradationAlert]:
        """Process multiple degradations and create alerts.
        
        Args:
            model_id: Model identifier
            degradations: List of degradations
            send_alerts: Whether to send alerts immediately
            
        Returns:
            List of created alerts
        """
        alerts = []
        
        for degradation in degradations:
            # Create alert
            alert = await self.create_alert(model_id, degradation)
            alerts.append(alert)
            
            # Send alert if requested
            if send_alerts:
                await self.send_alert(alert)
        
        return alerts
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: User acknowledging the alert
            timestamp: Acknowledgment timestamp
            
        Returns:
            True if acknowledged successfully
        """
        alert = await self.repository.get_alert(alert_id)
        if not alert:
            return False
        
        acknowledged_alert = alert.acknowledge(acknowledged_by, timestamp)
        await self.repository.update_alert(acknowledged_alert)
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert identifier
            resolved_by: User resolving the alert
            timestamp: Resolution timestamp
            
        Returns:
            True if resolved successfully
        """
        alert = await self.repository.get_alert(alert_id)
        if not alert:
            return False
        
        resolved_alert = alert.resolve(resolved_by, timestamp)
        await self.repository.update_alert(resolved_alert)
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True
    
    async def get_active_alerts(
        self,
        model_id: Optional[UUID] = None,
        severity_filter: Optional[str] = None
    ) -> List[DegradationAlert]:
        """Get active alerts.
        
        Args:
            model_id: Optional model filter
            severity_filter: Optional severity filter
            
        Returns:
            List of active alerts
        """
        alerts = await self.repository.get_active_alerts(model_id)
        
        if severity_filter:
            alerts = [
                alert for alert in alerts
                if alert.alert_level.value == severity_filter
            ]
        
        return alerts
    
    def _generate_default_message(self, degradation: PerformanceDegradation) -> str:
        """Generate default alert message."""
        return (
            f"Performance degradation detected in {degradation.metric_name}. "
            f"Current value ({degradation.current_value:.4f}) has degraded by "
            f"{abs(degradation.degradation_percentage):.2f}% from baseline "
            f"({degradation.baseline_value:.4f}). Confidence: {degradation.confidence_level:.2f}"
        )
    
    def _generate_recommendations(self, degradation: PerformanceDegradation) -> List[str]:
        """Generate recommendations for a degradation."""
        recommendations = []
        
        # Severity-based recommendations
        if degradation.severity == DegradationSeverity.CRITICAL:
            recommendations.append("🚨 Critical degradation - consider immediate model rollback or emergency retraining")
            recommendations.append("📊 Investigate root cause immediately")
            recommendations.append("🔄 Activate backup model if available")
        
        elif degradation.severity == DegradationSeverity.HIGH:
            recommendations.append("⚠️ High degradation - plan model update within 24-48 hours")
            recommendations.append("🔍 Analyze recent data changes and feature drift")
            recommendations.append("📈 Monitor closely for further degradation")
        
        elif degradation.severity == DegradationSeverity.MEDIUM:
            recommendations.append("⚡ Medium degradation - schedule model review within a week")
            recommendations.append("📊 Check training data quality and distribution")
        
        else:  # LOW
            recommendations.append("ℹ️ Low degradation - monitor trends and plan routine update")
        
        # Metric-specific recommendations
        metric_recommendations = {
            "accuracy": "Review feature engineering and data quality",
            "precision": "Check for class imbalance and false positive patterns",
            "recall": "Investigate missing features or data availability",
            "f1_score": "Balance precision and recall optimization",
            "roc_auc": "Review model calibration and threshold tuning",
            "rmse": "Check for outliers and scaling issues",
            "r2_score": "Evaluate feature selection and model complexity",
            "prediction_time_seconds": "Optimize model architecture or hardware",
            "memory_usage_mb": "Consider model compression techniques"
        }
        
        if degradation.metric_name in metric_recommendations:
            recommendations.append(metric_recommendations[degradation.metric_name])
        
        # Detection method specific recommendations
        if degradation.detection_method == "trend_analysis":
            recommendations.append("📈 Trend-based detection - monitor for sustained degradation pattern")
        
        return recommendations
    
    def _should_throttle_alert(self, alert: DegradationAlert) -> bool:
        """Check if alert should be throttled."""
        throttle_config = self.config.get('throttling', {})
        if not throttle_config.get('enabled', True):
            return False
        
        throttle_minutes = throttle_config.get('minutes', 30)
        max_alerts_per_hour = throttle_config.get('max_per_hour', 10)
        
        # Create throttle key
        throttle_key = f"{alert.model_id}_{alert.degradation.metric_name}"
        
        current_time = datetime.utcnow()
        
        # Check recent alerts for this key
        if throttle_key in self._alert_throttle:
            last_alert_time, recent_count = self._alert_throttle[throttle_key]
            
            # Check time-based throttling
            time_diff = (current_time - last_alert_time).total_seconds() / 60
            if time_diff < throttle_minutes:
                return True
            
            # Check rate limiting
            if recent_count >= max_alerts_per_hour:
                return True
        
        return False
    
    def _update_throttle_tracking(self, alert: DegradationAlert) -> None:
        """Update throttle tracking for an alert."""
        throttle_key = f"{alert.model_id}_{alert.degradation.metric_name}"
        
        current_time = datetime.utcnow()
        
        # Clean up old entries (older than 1 hour)
        cutoff_time = current_time.timestamp() - 3600
        self._alert_throttle = {
            k: v for k, v in self._alert_throttle.items()
            if v[0].timestamp() > cutoff_time
        }
        
        # Update tracking
        if throttle_key in self._alert_throttle:
            _, count = self._alert_throttle[throttle_key]
            self._alert_throttle[throttle_key] = (current_time, count + 1)
        else:
            self._alert_throttle[throttle_key] = (current_time, 1)