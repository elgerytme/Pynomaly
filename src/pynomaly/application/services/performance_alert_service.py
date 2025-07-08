"""Performance Alert Service for handling model performance degradation alerts.

This service specializes in creating, managing, and sending alerts related to
model performance degradation with intelligent correlation and notification capabilities.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pynomaly.application.services.intelligent_alert_service import IntelligentAlertService
from pynomaly.domain.entities.alert import (
    Alert,
    AlertCondition,
    AlertNotification,
    AlertSeverity,
    AlertSource,
    AlertStatus,
    AlertType,
    AlertCategory,
    AlertMetadata,
    NotificationChannel,
)
from pynomaly.application.services.model_performance_degradation_detector import (
    DegradationResult,
    DegradationDetails,
)

logger = logging.getLogger(__name__)


class PerformanceAlertService:
    """Service for handling model performance degradation alerts.
    
    This service provides specialized functionality for creating and managing
    alerts related to model performance degradation, including:
    - Intelligent severity mapping based on degradation magnitude
    - Correlation with existing alerts
    - Multi-channel notification support
    - Performance-specific alert metadata
    """

    def __init__(self, intelligent_alert_service: IntelligentAlertService):
        """Initialize the performance alert service.
        
        Args:
            intelligent_alert_service: The intelligent alert service for correlation and notifications
        """
        self.intelligent_alert_service = intelligent_alert_service
        self.logger = logging.getLogger(__name__)

    async def create_performance_alert(
        self,
        degradation_result: DegradationResult,
        model_id: str,
        model_name: str,
        detection_timestamp: Optional[datetime] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        notification_channels: Optional[List[NotificationChannel]] = None,
    ) -> Alert:
        """Create a performance degradation alert.
        
        Args:
            degradation_result: The degradation detection result
            model_id: Unique identifier for the model
            model_name: Human-readable name of the model
            detection_timestamp: When the degradation was detected (defaults to now)
            additional_context: Additional context information
            notification_channels: Channels to send notifications to
            
        Returns:
            Created alert with performance-specific metadata
        """
        if detection_timestamp is None:
            detection_timestamp = datetime.utcnow()
        
        # Calculate severity based on degradation magnitude
        severity = self._calculate_severity(degradation_result)
        
        # Create alert condition based on degradation details
        condition = self._create_alert_condition(degradation_result)
        
        # Build alert description
        description = self._build_alert_description(
            degradation_result, model_name, detection_timestamp
        )
        
        # Prepare metadata with performance-specific information
        metadata = self._prepare_alert_metadata(
            degradation_result, model_id, model_name, additional_context
        )
        
        # Create the alert using the intelligent alert service
        alert_metadata = AlertMetadata()
        alert_metadata.key = "performance_degradation"
        alert_metadata.value = metadata
        alert_metadata.value_type = "dict"
        
        alert = await self.intelligent_alert_service.create_alert(
            title=f"Model Performance Degradation: {model_name}",
            description=description,
            severity=severity,
            category=AlertCategory.PERFORMANCE,
            source=AlertSource.MODEL_MONITOR,
            metadata=alert_metadata,
            message=self._create_alert_message(degradation_result, model_name),
            details=degradation_result.to_dict(),
        )
        
        # Add performance-specific tags
        self._add_performance_tags(alert, degradation_result, model_name)
        
        # Set up notifications if specified
        if notification_channels:
            await self._setup_notifications(alert, notification_channels)
        
        self.logger.info(
            f"Created performance alert {alert.id} for model {model_name} "
            f"with severity {severity.value}"
        )
        
        return alert

    async def create_metric_specific_alert(
        self,
        degradation_detail: DegradationDetails,
        model_id: str,
        model_name: str,
        detection_timestamp: Optional[datetime] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create an alert for a specific metric degradation.
        
        Args:
            degradation_detail: Details of the specific metric degradation
            model_id: Unique identifier for the model
            model_name: Human-readable name of the model
            detection_timestamp: When the degradation was detected
            additional_context: Additional context information
            
        Returns:
            Created alert for the specific metric
        """
        if detection_timestamp is None:
            detection_timestamp = datetime.utcnow()
        
        # Calculate severity based on relative deviation
        severity = self._calculate_metric_severity(degradation_detail)
        
        # Create condition for the specific metric
        condition = AlertCondition(
            metric_name=degradation_detail.metric_name,
            operator="lt",  # Performance degradation means value is lower than baseline
            threshold=degradation_detail.baseline_value,
            description=f"{degradation_detail.metric_name} degradation detected",
        )
        
        # Build metric-specific description
        description = (
            f"Performance degradation detected in {degradation_detail.metric_name} "
            f"for model {model_name}. "
            f"Current value: {degradation_detail.current_value:.4f}, "
            f"Baseline: {degradation_detail.baseline_value:.4f}, "
            f"Deviation: {degradation_detail.relative_deviation:.2f}%"
        )
        
        # Prepare metadata
        metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "metric_name": degradation_detail.metric_name,
            "current_value": degradation_detail.current_value,
            "baseline_value": degradation_detail.baseline_value,
            "absolute_deviation": degradation_detail.deviation,
            "relative_deviation_percent": degradation_detail.relative_deviation,
            "detection_timestamp": detection_timestamp.isoformat(),
            "alert_source": AlertSource.MODEL_MONITOR.value,
            "statistical_significance": degradation_detail.statistical_significance,
        }
        
        if additional_context:
            metadata.update(additional_context)
        
        # Create the alert
        alert_metadata = AlertMetadata()
        alert_metadata.key = "metric_degradation"
        alert_metadata.value = metadata
        alert_metadata.value_type = "dict"
        
        alert = await self.intelligent_alert_service.create_alert(
            title=f"Metric Degradation: {degradation_detail.metric_name} - {model_name}",
            description=description,
            severity=severity,
            category=AlertCategory.PERFORMANCE,
            source=AlertSource.MODEL_MONITOR,
            metadata=alert_metadata,
            message=f"{degradation_detail.metric_name} degraded by {degradation_detail.relative_deviation:.2f}%",
            details=degradation_detail.__dict__,
        )
        
        # Add metric-specific tags
        alert.add_tag(f"metric:{degradation_detail.metric_name}")
        alert.add_tag(f"model:{model_name}")
        alert.add_tag("performance_degradation")
        
        return alert

    def _calculate_severity(self, degradation_result: DegradationResult) -> AlertSeverity:
        """Calculate alert severity based on degradation magnitude.
        
        Args:
            degradation_result: The degradation detection result
            
        Returns:
            Appropriate alert severity level
        """
        if not degradation_result.degrade_flag:
            return AlertSeverity.INFO
        
        # Use overall severity from degradation result
        severity_score = degradation_result.overall_severity
        
        # Map severity score to alert severity levels
        if severity_score >= 0.8:
            return AlertSeverity.CRITICAL
        elif severity_score >= 0.6:
            return AlertSeverity.HIGH
        elif severity_score >= 0.4:
            return AlertSeverity.MEDIUM
        elif severity_score >= 0.2:
            return AlertSeverity.LOW
        else:
            return AlertSeverity.INFO

    def _calculate_metric_severity(self, degradation_detail: DegradationDetails) -> AlertSeverity:
        """Calculate severity for a specific metric degradation.
        
        Args:
            degradation_detail: Details of the metric degradation
            
        Returns:
            Appropriate alert severity level
        """
        # Use relative deviation percentage to determine severity
        deviation_percent = abs(degradation_detail.relative_deviation)
        
        if deviation_percent >= 50:
            return AlertSeverity.CRITICAL
        elif deviation_percent >= 30:
            return AlertSeverity.HIGH
        elif deviation_percent >= 15:
            return AlertSeverity.MEDIUM
        elif deviation_percent >= 5:
            return AlertSeverity.LOW
        else:
            return AlertSeverity.INFO

    def _create_alert_condition(self, degradation_result: DegradationResult) -> AlertCondition:
        """Create an alert condition based on degradation result.
        
        Args:
            degradation_result: The degradation detection result
            
        Returns:
            Alert condition representing the degradation
        """
        if degradation_result.affected_metrics:
            # Use the first (most significant) affected metric
            primary_metric = degradation_result.affected_metrics[0]
            
            return AlertCondition(
                metric_name=primary_metric.metric_name,
                operator="lt",  # Performance degradation means value is lower
                threshold=primary_metric.baseline_value,
                description=f"Performance degradation in {primary_metric.metric_name}",
            )
        else:
            # Fallback condition for general degradation
            return AlertCondition(
                metric_name="overall_performance",
                operator="lt",
                threshold=1.0,
                description="General model performance degradation",
            )

    def _build_alert_description(
        self,
        degradation_result: DegradationResult,
        model_name: str,
        detection_timestamp: datetime,
    ) -> str:
        """Build a comprehensive alert description.
        
        Args:
            degradation_result: The degradation detection result
            model_name: Name of the model
            detection_timestamp: When degradation was detected
            
        Returns:
            Formatted alert description
        """
        description_parts = [
            f"Model performance degradation detected for {model_name} "
            f"at {detection_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}.",
        ]
        
        if degradation_result.affected_metrics:
            description_parts.append(
                f"Affected metrics ({len(degradation_result.affected_metrics)}):"
            )
            
            for metric in degradation_result.affected_metrics:
                description_parts.append(
                    f"  • {metric.metric_name}: {metric.current_value:.4f} "
                    f"(baseline: {metric.baseline_value:.4f}, "
                    f"deviation: {metric.relative_deviation:.2f}%)"
                )
        
        description_parts.append(
            f"Detection algorithm: {degradation_result.detection_algorithm.value}"
        )
        description_parts.append(
            f"Overall severity score: {degradation_result.overall_severity:.3f}"
        )
        
        return "\n".join(description_parts)

    def _prepare_alert_metadata(
        self,
        degradation_result: DegradationResult,
        model_id: str,
        model_name: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare comprehensive alert metadata.
        
        Args:
            degradation_result: The degradation detection result
            model_id: Unique identifier for the model
            model_name: Name of the model
            additional_context: Additional context information
            
        Returns:
            Complete metadata dictionary
        """
        metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "detection_algorithm": degradation_result.detection_algorithm.value,
            "overall_severity": degradation_result.overall_severity,
            "affected_metrics_count": len(degradation_result.affected_metrics),
            "degradation_flag": degradation_result.degrade_flag,
            "alert_source": AlertSource.MODEL_MONITOR.value,
            "alert_type": AlertType.MODEL_PERFORMANCE.value,
            "detection_metadata": degradation_result.metadata,
        }
        
        # Add affected metrics details
        if degradation_result.affected_metrics:
            metadata["affected_metrics"] = [
                {
                    "name": metric.metric_name,
                    "current_value": metric.current_value,
                    "baseline_value": metric.baseline_value,
                    "absolute_deviation": metric.deviation,
                    "relative_deviation_percent": metric.relative_deviation,
                    "statistical_significance": metric.statistical_significance,
                }
                for metric in degradation_result.affected_metrics
            ]
            
            # Add primary metric information
            primary_metric = degradation_result.affected_metrics[0]
            metadata["primary_metric"] = primary_metric.metric_name
            metadata["primary_metric_deviation"] = primary_metric.relative_deviation
        
        # Add additional context if provided
        if additional_context:
            metadata.update(additional_context)
        
        return metadata

    def _create_alert_message(
        self, degradation_result: DegradationResult, model_name: str
    ) -> str:
        """Create a concise alert message.
        
        Args:
            degradation_result: The degradation detection result
            model_name: Name of the model
            
        Returns:
            Concise alert message
        """
        if degradation_result.affected_metrics:
            metric_names = [m.metric_name for m in degradation_result.affected_metrics]
            if len(metric_names) == 1:
                return f"Performance degradation in {metric_names[0]} for model {model_name}"
            else:
                return f"Performance degradation in {len(metric_names)} metrics for model {model_name}"
        else:
            return f"General performance degradation detected for model {model_name}"

    def _add_performance_tags(
        self, alert: Alert, degradation_result: DegradationResult, model_name: str
    ) -> None:
        """Add performance-specific tags to the alert.
        
        Args:
            alert: The alert to add tags to
            degradation_result: The degradation detection result
            model_name: Name of the model
        """
        # Add basic tags
        alert.add_tag("performance_degradation")
        alert.add_tag(f"model:{model_name}")
        alert.add_tag(f"algorithm:{degradation_result.detection_algorithm.value}")
        
        # Add severity-based tags
        if degradation_result.overall_severity >= 0.8:
            alert.add_tag("critical_degradation")
        elif degradation_result.overall_severity >= 0.6:
            alert.add_tag("high_degradation")
        
        # Add metric-specific tags
        for metric in degradation_result.affected_metrics:
            alert.add_tag(f"metric:{metric.metric_name}")
            
            # Add deviation range tags
            deviation = abs(metric.relative_deviation)
            if deviation >= 50:
                alert.add_tag(f"severe_deviation:{metric.metric_name}")
            elif deviation >= 20:
                alert.add_tag(f"moderate_deviation:{metric.metric_name}")

    async def _setup_notifications(
        self, alert: Alert, notification_channels: List[NotificationChannel]
    ) -> None:
        """Set up notifications for the alert.
        
        Args:
            alert: The alert to set up notifications for
            notification_channels: Channels to send notifications to
        """
        for channel in notification_channels:
            # Create notification based on channel type
            if channel == NotificationChannel.EMAIL:
                recipient = "ml-ops@company.com"
            elif channel == NotificationChannel.SLACK:
                recipient = "#ml-alerts"
            elif channel == NotificationChannel.WEBHOOK:
                recipient = "https://api.company.com/webhooks/ml-alerts"
            else:
                recipient = "default@company.com"
            
            notification = AlertNotification(
                alert_id=alert.id,
                channel=channel,
                recipient=recipient,
                status="pending",
                metadata={
                    "alert_type": "performance_degradation",
                    "model_name": alert.metadata.get("model_name"),
                    "severity": alert.severity.value,
                },
            )
            
            alert.add_notification(notification)

    async def get_performance_alerts(
        self,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        status: Optional[AlertStatus] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get performance degradation alerts with filtering.
        
        Args:
            model_id: Filter by model ID
            model_name: Filter by model name
            severity: Filter by severity level
            status: Filter by alert status
            limit: Maximum number of alerts to return
            
        Returns:
            List of filtered performance alerts
        """
        # Get all alerts from the intelligent alert service
        alerts = await self.intelligent_alert_service.list_alerts(
            category_filter=AlertCategory.PERFORMANCE,
            severity_filter=severity,
            status_filter=status,
            limit=limit * 2,  # Get more to allow for additional filtering
        )
        
        # Apply performance-specific filters
        filtered_alerts = []
        for alert in alerts:
            if alert.source != AlertSource.MODEL_MONITOR.value:
                continue
                
            if model_id and alert.metadata.get("model_id") != model_id:
                continue
                
            if model_name and alert.metadata.get("model_name") != model_name:
                continue
                
            filtered_alerts.append(alert)
            
            if len(filtered_alerts) >= limit:
                break
        
        return filtered_alerts

    async def get_performance_alert_statistics(
        self, days: int = 7
    ) -> Dict[str, Any]:
        """Get statistics about performance alerts.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics about performance alerts
        """
        # Get analytics from the intelligent alert service
        analytics = await self.intelligent_alert_service.get_alert_analytics(days=days)
        
        # Filter for performance alerts and add performance-specific metrics
        performance_stats = {
            "total_performance_alerts": 0,
            "models_with_degradation": set(),
            "most_affected_metrics": {},
            "severity_distribution": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0,
            },
            "detection_algorithms": {},
        }
        
        # Get performance alerts for detailed analysis
        performance_alerts = await self.get_performance_alerts(limit=1000)
        
        for alert in performance_alerts:
            performance_stats["total_performance_alerts"] += 1
            
            # Track models with degradation
            model_name = alert.metadata.get("model_name")
            if model_name:
                performance_stats["models_with_degradation"].add(model_name)
            
            # Track affected metrics
            affected_metrics = alert.metadata.get("affected_metrics", [])
            for metric in affected_metrics:
                metric_name = metric.get("name")
                if metric_name:
                    performance_stats["most_affected_metrics"][metric_name] = (
                        performance_stats["most_affected_metrics"].get(metric_name, 0) + 1
                    )
            
            # Track severity distribution
            severity = alert.severity.value
            if severity in performance_stats["severity_distribution"]:
                performance_stats["severity_distribution"][severity] += 1
            
            # Track detection algorithms
            algorithm = alert.metadata.get("detection_algorithm")
            if algorithm:
                performance_stats["detection_algorithms"][algorithm] = (
                    performance_stats["detection_algorithms"].get(algorithm, 0) + 1
                )
        
        # Convert set to count
        performance_stats["models_with_degradation"] = len(performance_stats["models_with_degradation"])
        
        # Sort most affected metrics
        performance_stats["most_affected_metrics"] = dict(
            sorted(
                performance_stats["most_affected_metrics"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        
        return performance_stats
