"""Repository for performance degradation data persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from packages.data_science.domain.value_objects.model_performance_metrics import ModelPerformanceMetrics
from pynomaly.domain.value_objects.performance_degradation_metrics import (
    DegradationAlert,
    DegradationReport,
    MetricThreshold,
    PerformanceBaseline,
    PerformanceDegradation,
)

logger = logging.getLogger(__name__)


class PerformanceDegradationRepository:
    """Repository for managing performance degradation data."""

    def __init__(self):
        """Initialize the repository with in-memory storage."""
        # In-memory storage for development - replace with actual database in production
        self._performance_history: Dict[UUID, List[ModelPerformanceMetrics]] = {}
        self._baselines: Dict[str, PerformanceBaseline] = {}  # key: f"{model_id}_{metric_name}"
        self._degradations: Dict[str, List[PerformanceDegradation]] = {}  # key: model_id
        self._alerts: Dict[str, DegradationAlert] = {}  # key: alert_id
        self._reports: Dict[str, DegradationReport] = {}  # key: report_id
        self._thresholds: Dict[str, Dict[str, MetricThreshold]] = {}  # key: model_id, value: metric_name -> threshold

    # Performance History Management
    async def store_performance_metrics(
        self,
        model_id: UUID,
        metrics: ModelPerformanceMetrics,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Store performance metrics for a model.
        
        Args:
            model_id: Model identifier
            metrics: Performance metrics to store
            timestamp: Timestamp for the metrics (defaults to now)
        """
        if model_id not in self._performance_history:
            self._performance_history[model_id] = []
        
        # Set timestamp if not provided
        if timestamp:
            # Note: ModelPerformanceMetrics doesn't have timestamp field in current implementation
            # We'll store it separately or extend the metrics class
            pass
        
        self._performance_history[model_id].append(metrics)
        
        # Keep only last 1000 entries to manage memory
        if len(self._performance_history[model_id]) > 1000:
            self._performance_history[model_id] = self._performance_history[model_id][-1000:]
        
        logger.info(f"Stored performance metrics for model {model_id}")

    async def get_model_performance_history(
        self,
        model_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[ModelPerformanceMetrics]:
        """Get performance history for a model.
        
        Args:
            model_id: Model identifier
            start_date: Start date for filtering
            end_date: End date for filtering
            limit: Maximum number of records to return
            
        Returns:
            List of performance metrics
        """
        if model_id not in self._performance_history:
            return []
        
        metrics_list = self._performance_history[model_id]
        
        # Note: Since ModelPerformanceMetrics doesn't have timestamp,
        # we'll return all records for now. In a real implementation,
        # we'd filter by timestamp
        
        if limit:
            metrics_list = metrics_list[-limit:]
        
        return metrics_list

    async def get_latest_performance_metrics(self, model_id: UUID) -> Optional[ModelPerformanceMetrics]:
        """Get the latest performance metrics for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Latest performance metrics or None
        """
        history = await self.get_model_performance_history(model_id, limit=1)
        return history[0] if history else None

    # Baseline Management
    async def store_baseline(self, model_id: UUID, baseline: PerformanceBaseline) -> None:
        """Store a performance baseline.
        
        Args:
            model_id: Model identifier
            baseline: Performance baseline to store
        """
        key = f"{model_id}_{baseline.metric_name}"
        self._baselines[key] = baseline
        logger.info(f"Stored baseline for {baseline.metric_name} on model {model_id}")

    async def get_baseline(
        self, 
        model_id: UUID, 
        metric_name: str
    ) -> Optional[PerformanceBaseline]:
        """Get a performance baseline.
        
        Args:
            model_id: Model identifier
            metric_name: Name of the metric
            
        Returns:
            Performance baseline or None
        """
        key = f"{model_id}_{metric_name}"
        return self._baselines.get(key)

    async def get_all_baselines(self, model_id: UUID) -> Dict[str, PerformanceBaseline]:
        """Get all baselines for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary of metric name to baseline
        """
        baselines = {}
        prefix = f"{model_id}_"
        
        for key, baseline in self._baselines.items():
            if key.startswith(prefix):
                metric_name = key[len(prefix):]
                baselines[metric_name] = baseline
        
        return baselines

    async def update_baseline(self, model_id: UUID, baseline: PerformanceBaseline) -> None:
        """Update an existing baseline.
        
        Args:
            model_id: Model identifier
            baseline: Updated baseline
        """
        key = f"{model_id}_{baseline.metric_name}"
        if key in self._baselines:
            baseline_dict = baseline.dict()
            baseline_dict['last_updated'] = datetime.utcnow()
            self._baselines[key] = PerformanceBaseline(**baseline_dict)
            logger.info(f"Updated baseline for {baseline.metric_name} on model {model_id}")

    async def delete_baseline(self, model_id: UUID, metric_name: str) -> bool:
        """Delete a baseline.
        
        Args:
            model_id: Model identifier
            metric_name: Name of the metric
            
        Returns:
            True if deleted, False if not found
        """
        key = f"{model_id}_{metric_name}"
        if key in self._baselines:
            del self._baselines[key]
            logger.info(f"Deleted baseline for {metric_name} on model {model_id}")
            return True
        return False

    # Degradation Management
    async def store_degradation(self, model_id: UUID, degradation: PerformanceDegradation) -> None:
        """Store a performance degradation.
        
        Args:
            model_id: Model identifier
            degradation: Performance degradation to store
        """
        model_key = str(model_id)
        if model_key not in self._degradations:
            self._degradations[model_key] = []
        
        self._degradations[model_key].append(degradation)
        logger.info(f"Stored degradation for {degradation.metric_name} on model {model_id}")

    async def get_recent_degradations(
        self,
        model_id: UUID,
        hours: int = 24,
        severity_filter: Optional[str] = None
    ) -> List[PerformanceDegradation]:
        """Get recent degradations for a model.
        
        Args:
            model_id: Model identifier
            hours: Hours to look back
            severity_filter: Filter by severity level
            
        Returns:
            List of degradations
        """
        model_key = str(model_id)
        if model_key not in self._degradations:
            return []
        
        degradations = self._degradations[model_key]
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter by time and severity
        filtered = []
        for degradation in degradations:
            if degradation.detected_at >= cutoff_time:
                if not severity_filter or degradation.severity.value == severity_filter:
                    filtered.append(degradation)
        
        return filtered

    async def get_degradation_trends(
        self,
        model_id: UUID,
        metric_name: str,
        days: int = 30
    ) -> List[PerformanceDegradation]:
        """Get degradation trends for a specific metric.
        
        Args:
            model_id: Model identifier
            metric_name: Name of the metric
            days: Days to look back
            
        Returns:
            List of degradations for the metric
        """
        model_key = str(model_id)
        if model_key not in self._degradations:
            return []
        
        degradations = self._degradations[model_key]
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        return [
            d for d in degradations
            if d.metric_name == metric_name and d.detected_at >= cutoff_time
        ]

    # Alert Management
    async def store_alert(self, alert: DegradationAlert) -> None:
        """Store a degradation alert.
        
        Args:
            alert: Degradation alert to store
        """
        self._alerts[alert.alert_id] = alert
        logger.info(f"Stored alert {alert.alert_id} for model {alert.model_id}")

    async def get_alert(self, alert_id: str) -> Optional[DegradationAlert]:
        """Get an alert by ID.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            Alert or None
        """
        return self._alerts.get(alert_id)

    async def get_active_alerts(self, model_id: Optional[UUID] = None) -> List[DegradationAlert]:
        """Get active (unresolved) alerts.
        
        Args:
            model_id: Optional model filter
            
        Returns:
            List of active alerts
        """
        alerts = []
        for alert in self._alerts.values():
            if not alert.resolved:
                if not model_id or alert.model_id == str(model_id):
                    alerts.append(alert)
        
        return alerts

    async def update_alert(self, alert: DegradationAlert) -> None:
        """Update an existing alert.
        
        Args:
            alert: Updated alert
        """
        if alert.alert_id in self._alerts:
            self._alerts[alert.alert_id] = alert
            logger.info(f"Updated alert {alert.alert_id}")

    # Report Management
    async def store_report(self, report: DegradationReport) -> None:
        """Store a degradation report.
        
        Args:
            report: Degradation report to store
        """
        self._reports[report.report_id] = report
        logger.info(f"Stored report {report.report_id} for model {report.model_id}")

    async def get_report(self, report_id: str) -> Optional[DegradationReport]:
        """Get a report by ID.
        
        Args:
            report_id: Report identifier
            
        Returns:
            Report or None
        """
        return self._reports.get(report_id)

    async def get_model_reports(
        self,
        model_id: UUID,
        limit: int = 10
    ) -> List[DegradationReport]:
        """Get reports for a model.
        
        Args:
            model_id: Model identifier
            limit: Maximum number of reports
            
        Returns:
            List of reports
        """
        model_reports = [
            report for report in self._reports.values()
            if report.model_id == str(model_id)
        ]
        
        # Sort by generation date, most recent first
        model_reports.sort(key=lambda r: r.generated_at, reverse=True)
        
        return model_reports[:limit]

    # Threshold Management
    async def store_threshold_config(
        self,
        model_id: UUID,
        thresholds: Dict[str, MetricThreshold]
    ) -> None:
        """Store threshold configuration for a model.
        
        Args:
            model_id: Model identifier
            thresholds: Dictionary of metric thresholds
        """
        self._thresholds[str(model_id)] = thresholds
        logger.info(f"Stored threshold config for model {model_id}")

    async def get_threshold_config(self, model_id: UUID) -> Optional[Dict[str, MetricThreshold]]:
        """Get threshold configuration for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary of thresholds or None
        """
        return self._thresholds.get(str(model_id))

    async def update_threshold(
        self,
        model_id: UUID,
        metric_name: str,
        threshold: MetricThreshold
    ) -> None:
        """Update a specific threshold.
        
        Args:
            model_id: Model identifier
            metric_name: Name of the metric
            threshold: New threshold configuration
        """
        model_key = str(model_id)
        if model_key not in self._thresholds:
            self._thresholds[model_key] = {}
        
        self._thresholds[model_key][metric_name] = threshold
        logger.info(f"Updated threshold for {metric_name} on model {model_id}")

    # Analytics and Statistics
    async def get_model_health_summary(self, model_id: UUID) -> Dict[str, Any]:
        """Get health summary for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Health summary dictionary
        """
        # Get recent degradations
        recent_degradations = await self.get_recent_degradations(model_id, hours=24)
        
        # Get active alerts
        active_alerts = await self.get_active_alerts(model_id)
        
        # Get baselines
        baselines = await self.get_all_baselines(model_id)
        
        # Calculate health score
        health_score = 1.0
        if recent_degradations:
            severity_penalties = {
                'critical': 0.4,
                'high': 0.2,
                'medium': 0.1,
                'low': 0.05
            }
            
            penalty = sum(
                severity_penalties.get(d.severity.value, 0)
                for d in recent_degradations
            )
            health_score = max(0.0, 1.0 - penalty)
        
        return {
            'model_id': str(model_id),
            'health_score': health_score,
            'recent_degradations_count': len(recent_degradations),
            'active_alerts_count': len(active_alerts),
            'baselines_count': len(baselines),
            'degradations_by_severity': {
                severity: sum(1 for d in recent_degradations if d.severity.value == severity)
                for severity in ['critical', 'high', 'medium', 'low']
            },
            'most_degraded_metrics': [
                d.metric_name for d in sorted(
                    recent_degradations,
                    key=lambda x: abs(x.degradation_percentage),
                    reverse=True
                )[:5]
            ]
        }

    async def get_system_wide_health(self) -> Dict[str, Any]:
        """Get system-wide health statistics.
        
        Returns:
            System health summary
        """
        total_models = len(self._performance_history)
        total_alerts = len([a for a in self._alerts.values() if not a.resolved])
        total_baselines = len(self._baselines)
        
        # Count degradations by severity
        all_degradations = []
        for degradations in self._degradations.values():
            all_degradations.extend(degradations)
        
        recent_degradations = [
            d for d in all_degradations
            if d.detected_at >= datetime.utcnow() - timedelta(hours=24)
        ]
        
        return {
            'total_models_monitored': total_models,
            'total_active_alerts': total_alerts,
            'total_baselines': total_baselines,
            'recent_degradations_24h': len(recent_degradations),
            'system_health_score': 1.0 - min(len(recent_degradations) * 0.1, 1.0),
            'degradations_by_severity': {
                severity: sum(1 for d in recent_degradations if d.severity.value == severity)
                for severity in ['critical', 'high', 'medium', 'low']
            }
        }

    # Cleanup and Maintenance
    async def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data to manage storage.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Summary of cleanup operations
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        cleanup_summary = {
            'degradations_removed': 0,
            'alerts_removed': 0,
            'reports_removed': 0
        }
        
        # Clean old degradations
        for model_id, degradations in self._degradations.items():
            original_count = len(degradations)
            self._degradations[model_id] = [
                d for d in degradations if d.detected_at >= cutoff_date
            ]
            cleanup_summary['degradations_removed'] += original_count - len(self._degradations[model_id])
        
        # Clean old alerts
        original_alert_count = len(self._alerts)
        self._alerts = {
            alert_id: alert for alert_id, alert in self._alerts.items()
            if alert.created_at >= cutoff_date
        }
        cleanup_summary['alerts_removed'] = original_alert_count - len(self._alerts)
        
        # Clean old reports
        original_report_count = len(self._reports)
        self._reports = {
            report_id: report for report_id, report in self._reports.items()
            if report.generated_at >= cutoff_date
        }
        cleanup_summary['reports_removed'] = original_report_count - len(self._reports)
        
        logger.info(f"Cleaned up old data: {cleanup_summary}")
        return cleanup_summary