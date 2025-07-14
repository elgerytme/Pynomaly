"""Real-time data quality monitoring system with alerting and reporting."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import json
from uuid import UUID, uuid4
from collections import deque, defaultdict
import threading
import time

from ...domain.entities.quality_rule import QualityRule, ValidationResult, ValidationStatus, Severity
from .validation_engine import ValidationEngine

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringStatus(str, Enum):
    """Monitoring system status."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class MetricType(str, Enum):
    """Types of quality metrics."""
    PASS_RATE = "pass_rate"
    ERROR_RATE = "error_rate"
    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"


@dataclass
class QualityAlert:
    """Quality alert notification."""
    alert_id: UUID = field(default_factory=uuid4)
    rule_id: UUID
    dataset_id: UUID
    severity: AlertSeverity
    alert_type: str
    message: str
    current_value: float
    threshold_value: float
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetric:
    """Quality metric data point."""
    metric_id: UUID = field(default_factory=uuid4)
    metric_type: MetricType
    dataset_id: UUID
    rule_id: Optional[UUID] = None
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfiguration:
    """Configuration for quality monitoring."""
    monitoring_interval_seconds: int = 300  # 5 minutes
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)
    retention_days: int = 30
    max_alerts_per_hour: int = 10
    enable_trend_analysis: bool = True
    enable_anomaly_detection: bool = True


class QualityTrendAnalyzer:
    """Analyze quality trends and detect patterns."""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size  # Hours
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_metric(self, metric: QualityMetric) -> None:
        """Add a metric to the trend analysis buffer."""
        key = f"{metric.dataset_id}_{metric.metric_type}"
        self.metrics_buffer[key].append(metric)
    
    def detect_trend(self, dataset_id: UUID, metric_type: MetricType) -> Dict[str, Any]:
        """Detect trend for a specific metric."""
        key = f"{dataset_id}_{metric_type}"
        metrics = list(self.metrics_buffer[key])
        
        if len(metrics) < 3:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        # Calculate trend using linear regression
        values = [m.value for m in metrics]
        timestamps = [(m.timestamp - metrics[0].timestamp).total_seconds() for m in metrics]
        
        try:
            # Simple linear regression
            n = len(values)
            sum_x = sum(timestamps)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(timestamps, values))
            sum_x2 = sum(x * x for x in timestamps)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Determine trend direction
            if abs(slope) < 0.001:
                trend = "stable"
            elif slope > 0:
                trend = "improving"
            else:
                trend = "declining"
            
            # Calculate confidence based on correlation
            mean_x = sum_x / n
            mean_y = sum_y / n
            
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(timestamps, values))
            denominator_x = sum((x - mean_x) ** 2 for x in timestamps)
            denominator_y = sum((y - mean_y) ** 2 for y in values)
            
            if denominator_x > 0 and denominator_y > 0:
                correlation = numerator / (denominator_x * denominator_y) ** 0.5
                confidence = abs(correlation)
            else:
                confidence = 0.0
            
            return {
                "trend": trend,
                "slope": slope,
                "confidence": confidence,
                "data_points": len(metrics),
                "time_span_hours": (metrics[-1].timestamp - metrics[0].timestamp).total_seconds() / 3600
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return {"trend": "error", "confidence": 0.0, "error": str(e)}
    
    def detect_anomaly(self, dataset_id: UUID, metric_type: MetricType, 
                      current_value: float) -> Dict[str, Any]:
        """Detect if current value is anomalous."""
        key = f"{dataset_id}_{metric_type}"
        metrics = list(self.metrics_buffer[key])
        
        if len(metrics) < 5:
            return {"is_anomaly": False, "confidence": 0.0, "reason": "insufficient_data"}
        
        values = [m.value for m in metrics]
        
        # Calculate statistics
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return {"is_anomaly": False, "confidence": 0.0, "reason": "no_variance"}
        
        # Z-score based anomaly detection
        z_score = abs((current_value - mean) / std)
        
        # Consider anomaly if z-score > 2 (95% confidence)
        is_anomaly = z_score > 2.0
        confidence = min(z_score / 3.0, 1.0)  # Normalize to 0-1 range
        
        return {
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "z_score": z_score,
            "historical_mean": mean,
            "historical_std": std,
            "deviation": abs(current_value - mean)
        }


class AlertManager:
    """Manage quality alerts and notifications."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.active_alerts: Dict[UUID, QualityAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_rate_limiter: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.max_alerts_per_hour))
        self.notification_handlers: Dict[str, Callable] = {}
    
    def register_notification_handler(self, channel: str, handler: Callable) -> None:
        """Register a notification handler for a channel."""
        self.notification_handlers[channel] = handler
    
    def create_alert(self, rule_id: UUID, dataset_id: UUID, severity: AlertSeverity,
                    alert_type: str, message: str, current_value: float,
                    threshold_value: float, metadata: Optional[Dict[str, Any]] = None) -> QualityAlert:
        """Create a new quality alert."""
        # Check rate limiting
        rate_key = f"{dataset_id}_{alert_type}"
        current_time = datetime.utcnow()
        
        # Clean old entries from rate limiter
        hour_ago = current_time - timedelta(hours=1)
        while (self.alert_rate_limiter[rate_key] and 
               self.alert_rate_limiter[rate_key][0] < hour_ago):
            self.alert_rate_limiter[rate_key].popleft()
        
        # Check if rate limit exceeded
        if len(self.alert_rate_limiter[rate_key]) >= self.config.max_alerts_per_hour:
            logger.warning(f"Alert rate limit exceeded for {rate_key}")
            return None
        
        # Create alert
        alert = QualityAlert(
            rule_id=rule_id,
            dataset_id=dataset_id,
            severity=severity,
            alert_type=alert_type,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            metadata=metadata or {}
        )
        
        # Add to rate limiter
        self.alert_rate_limiter[rate_key].append(current_time)
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.info(f"Created {severity} alert: {message}")
        return alert
    
    def acknowledge_alert(self, alert_id: UUID, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    def close_alert(self, alert_id: UUID) -> bool:
        """Close an active alert."""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} closed")
            return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[QualityAlert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        active_alerts = list(self.active_alerts.values())
        
        summary = {
            "total_active_alerts": len(active_alerts),
            "by_severity": {
                "emergency": len([a for a in active_alerts if a.severity == AlertSeverity.EMERGENCY]),
                "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                "warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
                "info": len([a for a in active_alerts if a.severity == AlertSeverity.INFO])
            },
            "acknowledged": len([a for a in active_alerts if a.acknowledged]),
            "unacknowledged": len([a for a in active_alerts if not a.acknowledged]),
            "oldest_alert": min([a.triggered_at for a in active_alerts]) if active_alerts else None,
            "newest_alert": max([a.triggered_at for a in active_alerts]) if active_alerts else None
        }
        
        return summary
    
    def _send_notifications(self, alert: QualityAlert) -> None:
        """Send notifications for an alert."""
        for channel in self.config.notification_channels:
            if channel in self.notification_handlers:
                try:
                    self.notification_handlers[channel](alert)
                except Exception as e:
                    logger.error(f"Failed to send notification to {channel}: {e}")


class QualityMonitoringService:
    """Real-time data quality monitoring service."""
    
    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.status = MonitoringStatus.STOPPED
        self.validation_engine = ValidationEngine()
        self.trend_analyzer = QualityTrendAnalyzer()
        self.alert_manager = AlertManager(config)
        
        # Monitoring state
        self.monitored_datasets: Dict[UUID, Dict[str, Any]] = {}
        self.active_rules: Dict[UUID, QualityRule] = {}
        self.metrics_buffer: deque = deque(maxlen=10000)
        
        # Threading
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
    
    def add_dataset_monitoring(self, dataset_id: UUID, rules: List[QualityRule],
                              data_source_config: Dict[str, Any]) -> None:
        """Add a dataset to monitoring."""
        self.monitored_datasets[dataset_id] = {
            "rules": [rule.rule_id for rule in rules],
            "data_source_config": data_source_config,
            "last_check": None,
            "check_count": 0,
            "last_quality_score": None
        }
        
        # Store rules
        for rule in rules:
            self.active_rules[rule.rule_id] = rule
        
        logger.info(f"Added dataset {dataset_id} to monitoring with {len(rules)} rules")
    
    def remove_dataset_monitoring(self, dataset_id: UUID) -> None:
        """Remove a dataset from monitoring."""
        if dataset_id in self.monitored_datasets:
            # Remove associated rules
            rule_ids = self.monitored_datasets[dataset_id]["rules"]
            for rule_id in rule_ids:
                if rule_id in self.active_rules:
                    del self.active_rules[rule_id]
            
            del self.monitored_datasets[dataset_id]
            logger.info(f"Removed dataset {dataset_id} from monitoring")
    
    def start_monitoring(self) -> None:
        """Start the monitoring service."""
        if self.status == MonitoringStatus.ACTIVE:
            logger.warning("Monitoring service is already active")
            return
        
        self.status = MonitoringStatus.ACTIVE
        self.stop_event.clear()
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Quality monitoring service started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring service."""
        if self.status != MonitoringStatus.ACTIVE:
            return
        
        self.status = MonitoringStatus.STOPPED
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Quality monitoring service stopped")
    
    def pause_monitoring(self) -> None:
        """Pause the monitoring service."""
        if self.status == MonitoringStatus.ACTIVE:
            self.status = MonitoringStatus.PAUSED
            logger.info("Quality monitoring service paused")
    
    def resume_monitoring(self) -> None:
        """Resume the monitoring service."""
        if self.status == MonitoringStatus.PAUSED:
            self.status = MonitoringStatus.ACTIVE
            logger.info("Quality monitoring service resumed")
    
    def validate_dataset_realtime(self, dataset_id: UUID, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform real-time validation of a dataset."""
        if dataset_id not in self.monitored_datasets:
            return {"error": "Dataset not monitored", "success": False}
        
        start_time = time.time()
        
        # Get rules for this dataset
        rule_ids = self.monitored_datasets[dataset_id]["rules"]
        rules = [self.active_rules[rule_id] for rule_id in rule_ids if rule_id in self.active_rules]
        
        if not rules:
            return {"error": "No active rules for dataset", "success": False}
        
        # Execute validation
        validation_results = self.validation_engine.validate_dataset(rules, df, dataset_id)
        
        # Process results and generate metrics
        metrics = []
        alerts_created = []
        
        for result in validation_results:
            rule = self.active_rules.get(result.rule_id)
            if not rule:
                continue
            
            # Create quality metric
            metric = QualityMetric(
                metric_type=MetricType.PASS_RATE,
                dataset_id=dataset_id,
                rule_id=result.rule_id,
                value=result.pass_rate,
                metadata={
                    "rule_name": rule.rule_name,
                    "rule_type": rule.rule_type.value,
                    "total_records": result.total_records,
                    "records_failed": result.records_failed
                }
            )
            metrics.append(metric)
            self.metrics_buffer.append(metric)
            
            # Add to trend analysis
            if self.config.enable_trend_analysis:
                self.trend_analyzer.add_metric(metric)
            
            # Check for threshold violations and create alerts
            alerts_created.extend(self._check_thresholds_and_alert(rule, result, dataset_id))
        
        # Update dataset monitoring info
        self.monitored_datasets[dataset_id]["last_check"] = datetime.utcnow()
        self.monitored_datasets[dataset_id]["check_count"] += 1
        self.monitored_datasets[dataset_id]["last_quality_score"] = np.mean([m.value for m in metrics]) if metrics else 0
        
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "dataset_id": str(dataset_id),
            "validation_results": len(validation_results),
            "metrics_generated": len(metrics),
            "alerts_created": len(alerts_created),
            "execution_time_seconds": execution_time,
            "overall_quality_score": self.monitored_datasets[dataset_id]["last_quality_score"]
        }
    
    def get_quality_dashboard_data(self, dataset_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get data for quality monitoring dashboard."""
        # Filter metrics by dataset if specified
        if dataset_id:
            relevant_metrics = [m for m in self.metrics_buffer if m.dataset_id == dataset_id]
            datasets_info = {dataset_id: self.monitored_datasets.get(dataset_id, {})}
        else:
            relevant_metrics = list(self.metrics_buffer)
            datasets_info = self.monitored_datasets
        
        # Calculate summary statistics
        if relevant_metrics:
            latest_metrics = {}
            for metric in relevant_metrics:
                key = f"{metric.dataset_id}_{metric.metric_type}"
                if key not in latest_metrics or metric.timestamp > latest_metrics[key].timestamp:
                    latest_metrics[key] = metric
            
            avg_quality = np.mean([m.value for m in latest_metrics.values()])
            min_quality = min([m.value for m in latest_metrics.values()])
            max_quality = max([m.value for m in latest_metrics.values()])
        else:
            avg_quality = min_quality = max_quality = 0
        
        # Get alert summary
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Get trend analysis for key metrics
        trends = {}
        if self.config.enable_trend_analysis and dataset_id:
            for metric_type in MetricType:
                trend = self.trend_analyzer.detect_trend(dataset_id, metric_type)
                if trend["trend"] != "insufficient_data":
                    trends[metric_type.value] = trend
        
        return {
            "monitoring_status": self.status.value,
            "monitored_datasets": len(datasets_info),
            "total_active_rules": len(self.active_rules),
            "quality_summary": {
                "average_quality": avg_quality,
                "minimum_quality": min_quality,
                "maximum_quality": max_quality,
                "total_metrics": len(relevant_metrics)
            },
            "alert_summary": alert_summary,
            "trends": trends,
            "datasets_info": {str(k): v for k, v in datasets_info.items()},
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def get_quality_report(self, dataset_id: UUID, start_date: datetime, 
                          end_date: datetime) -> Dict[str, Any]:
        """Generate quality report for a date range."""
        # Filter metrics by dataset and date range
        relevant_metrics = [
            m for m in self.metrics_buffer 
            if m.dataset_id == dataset_id and start_date <= m.timestamp <= end_date
        ]
        
        if not relevant_metrics:
            return {"error": "No metrics found for specified criteria"}
        
        # Group metrics by type
        metrics_by_type = defaultdict(list)
        for metric in relevant_metrics:
            metrics_by_type[metric.metric_type].append(metric)
        
        # Calculate statistics for each metric type
        metric_stats = {}
        for metric_type, metrics in metrics_by_type.items():
            values = [m.value for m in metrics]
            metric_stats[metric_type.value] = {
                "count": len(values),
                "average": np.mean(values),
                "minimum": min(values),
                "maximum": max(values),
                "std_deviation": np.std(values),
                "trend": self.trend_analyzer.detect_trend(dataset_id, metric_type) if self.config.enable_trend_analysis else None
            }
        
        # Get alerts for the period
        period_alerts = [
            alert for alert in self.alert_manager.alert_history
            if alert.dataset_id == dataset_id and start_date <= alert.triggered_at <= end_date
        ]
        
        alert_stats = {
            "total_alerts": len(period_alerts),
            "by_severity": {
                severity.value: len([a for a in period_alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            "by_type": {}
        }
        
        # Group alerts by type
        for alert in period_alerts:
            alert_type = alert.alert_type
            if alert_type not in alert_stats["by_type"]:
                alert_stats["by_type"][alert_type] = 0
            alert_stats["by_type"][alert_type] += 1
        
        return {
            "dataset_id": str(dataset_id),
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "metric_statistics": metric_stats,
            "alert_statistics": alert_stats,
            "data_points": len(relevant_metrics),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        logger.info("Quality monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    self._perform_monitoring_cycle()
                
                # Sleep for monitoring interval
                self.stop_event.wait(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.status = MonitoringStatus.ERROR
                time.sleep(60)  # Wait a minute before retrying
        
        logger.info("Quality monitoring loop stopped")
    
    def _perform_monitoring_cycle(self) -> None:
        """Perform one monitoring cycle for all datasets."""
        logger.debug(f"Performing monitoring cycle for {len(self.monitored_datasets)} datasets")
        
        for dataset_id, dataset_info in self.monitored_datasets.items():
            try:
                # Load dataset based on data source configuration
                df = self._load_dataset(dataset_info["data_source_config"])
                
                if df is not None and len(df) > 0:
                    # Perform validation
                    result = self.validate_dataset_realtime(dataset_id, df)
                    
                    if result["success"]:
                        logger.debug(f"Monitoring cycle completed for dataset {dataset_id}")
                    else:
                        logger.warning(f"Monitoring failed for dataset {dataset_id}: {result.get('error', 'Unknown error')}")
                else:
                    logger.warning(f"Could not load data for dataset {dataset_id}")
                    
            except Exception as e:
                logger.error(f"Error monitoring dataset {dataset_id}: {e}")
    
    def _load_dataset(self, data_source_config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load dataset based on data source configuration."""
        try:
            source_type = data_source_config.get("type")
            
            if source_type == "file":
                file_path = data_source_config.get("path")
                return pd.read_csv(file_path)  # Simplified - could support multiple formats
            
            elif source_type == "database":
                # Database loading would be implemented here
                # For now, return None as placeholder
                return None
            
            else:
                logger.warning(f"Unsupported data source type: {source_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def _check_thresholds_and_alert(self, rule: QualityRule, result: ValidationResult, 
                                   dataset_id: UUID) -> List[QualityAlert]:
        """Check thresholds and create alerts if necessary."""
        alerts_created = []
        
        # Check pass rate threshold
        if result.pass_rate < rule.thresholds.critical_threshold:
            alert = self.alert_manager.create_alert(
                rule_id=result.rule_id,
                dataset_id=dataset_id,
                severity=AlertSeverity.CRITICAL,
                alert_type="critical_quality_threshold",
                message=f"Rule '{rule.rule_name}' pass rate {result.pass_rate:.2%} is below critical threshold {rule.thresholds.critical_threshold:.2%}",
                current_value=result.pass_rate,
                threshold_value=rule.thresholds.critical_threshold,
                metadata={"rule_type": rule.rule_type.value, "total_records": result.total_records}
            )
            if alert:
                alerts_created.append(alert)
        
        elif result.pass_rate < rule.thresholds.warning_threshold:
            alert = self.alert_manager.create_alert(
                rule_id=result.rule_id,
                dataset_id=dataset_id,
                severity=AlertSeverity.WARNING,
                alert_type="warning_quality_threshold",
                message=f"Rule '{rule.rule_name}' pass rate {result.pass_rate:.2%} is below warning threshold {rule.thresholds.warning_threshold:.2%}",
                current_value=result.pass_rate,
                threshold_value=rule.thresholds.warning_threshold,
                metadata={"rule_type": rule.rule_type.value, "total_records": result.total_records}
            )
            if alert:
                alerts_created.append(alert)
        
        # Check for anomalies if enabled
        if self.config.enable_anomaly_detection:
            anomaly_result = self.trend_analyzer.detect_anomaly(dataset_id, MetricType.PASS_RATE, result.pass_rate)
            
            if anomaly_result["is_anomaly"] and anomaly_result["confidence"] > 0.8:
                alert = self.alert_manager.create_alert(
                    rule_id=result.rule_id,
                    dataset_id=dataset_id,
                    severity=AlertSeverity.WARNING,
                    alert_type="quality_anomaly",
                    message=f"Quality anomaly detected for rule '{rule.rule_name}': pass rate {result.pass_rate:.2%} deviates significantly from historical average",
                    current_value=result.pass_rate,
                    threshold_value=anomaly_result["historical_mean"],
                    metadata={
                        "anomaly_confidence": anomaly_result["confidence"],
                        "z_score": anomaly_result["z_score"],
                        "historical_mean": anomaly_result["historical_mean"],
                        "deviation": anomaly_result["deviation"]
                    }
                )
                if alert:
                    alerts_created.append(alert)
        
        return alerts_created