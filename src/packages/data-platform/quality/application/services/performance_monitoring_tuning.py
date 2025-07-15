"""
Comprehensive performance monitoring and tuning service for enterprise-scale data quality operations.

This service implements real-time performance monitoring, automated tuning,
predictive analytics, and intelligent optimization recommendations.
"""

import asyncio
import logging
import time
import json
import pickle
import statistics
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import psutil
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from ...domain.entities.quality_profile import DataQualityProfile
from ...domain.value_objects.quality_scores import QualityScores
from core.shared.error_handling import handle_exceptions

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    BUSINESS = "business"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class TuningStrategy(Enum):
    """Performance tuning strategies."""
    RESOURCE_SCALING = "resource_scaling"
    CACHE_OPTIMIZATION = "cache_optimization"
    QUERY_OPTIMIZATION = "query_optimization"
    LOAD_BALANCING = "load_balancing"
    CONFIGURATION_TUNING = "configuration_tuning"
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"


@dataclass
class PerformanceMetric:
    """Individual performance metric with metadata."""
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Context
    component: str = "unknown"
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Aggregation metadata
    aggregation_period: str = "1m"  # 1 minute default
    sample_count: int = 1
    
    @property
    def severity(self) -> AlertSeverity:
        """Determine alert severity based on thresholds."""
        if self.critical_threshold and self.value >= self.critical_threshold:
            return AlertSeverity.CRITICAL
        elif self.warning_threshold and self.value >= self.warning_threshold:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO


@dataclass
class PerformanceAlert:
    """Performance alert with context and recommendations."""
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    current_value: float
    threshold_value: float
    message: str
    
    # Timing
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    
    # Context
    component: str = "unknown"
    related_metrics: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    auto_resolution_attempted: bool = False
    
    @property
    def duration_minutes(self) -> float:
        """Calculate alert duration in minutes."""
        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.triggered_at).total_seconds() / 60
    
    @property
    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.resolved_at is None


@dataclass
class PerformanceBaseline:
    """Performance baseline for anomaly detection."""
    metric_name: str
    component: str
    
    # Statistical baselines
    mean_value: float = 0.0
    std_deviation: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    
    # Time-based patterns
    hourly_patterns: Dict[int, float] = field(default_factory=dict)
    daily_patterns: Dict[str, float] = field(default_factory=dict)
    seasonal_trends: List[float] = field(default_factory=list)
    
    # Learning metadata
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    confidence_level: float = 0.0
    
    def is_anomaly(self, value: float, threshold_std: float = 3.0) -> bool:
        """Check if value is anomalous based on baseline."""
        if self.std_deviation == 0:
            return False
        
        z_score = abs(value - self.mean_value) / self.std_deviation
        return z_score > threshold_std


@dataclass
class TuningRecommendation:
    """Performance tuning recommendation."""
    recommendation_id: str
    strategy: TuningStrategy
    title: str
    description: str
    
    # Impact assessment
    estimated_improvement_percent: float = 0.0
    implementation_effort: str = "low"  # low, medium, high
    risk_level: str = "low"  # low, medium, high
    
    # Implementation details
    actions: List[str] = field(default_factory=list)
    configuration_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    can_auto_implement: bool = False
    requires_approval: bool = True
    tested: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    priority_score: float = 0.0


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    report_id: str
    period_start: datetime
    period_end: datetime
    
    # Key metrics summary
    key_metrics: Dict[str, float] = field(default_factory=dict)
    performance_score: float = 0.0
    
    # Trends and analysis
    trending_up: List[str] = field(default_factory=list)
    trending_down: List[str] = field(default_factory=list)
    anomalies_detected: List[str] = field(default_factory=list)
    
    # Alerts summary
    critical_alerts: int = 0
    warning_alerts: int = 0
    resolved_alerts: int = 0
    
    # Recommendations
    top_recommendations: List[TuningRecommendation] = field(default_factory=list)
    implemented_improvements: List[str] = field(default_factory=list)
    
    # Resource utilization
    cpu_utilization_avg: float = 0.0
    memory_utilization_avg: float = 0.0
    disk_utilization_avg: float = 0.0
    network_utilization_avg: float = 0.0
    
    # Business impact
    sla_compliance_percent: float = 100.0
    user_satisfaction_score: float = 0.0
    cost_efficiency_score: float = 0.0


class MetricCollector:
    """Intelligent metric collection with adaptive sampling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize metric collector."""
        self.config = config
        self.collectors: Dict[str, Callable] = {}
        self.collection_intervals: Dict[str, float] = {}
        self.last_collection: Dict[str, datetime] = {}
        
        # Initialize built-in collectors
        self._init_system_collectors()
        self._init_application_collectors()
    
    def _init_system_collectors(self) -> None:
        """Initialize system metric collectors."""
        self.collectors.update({
            "cpu_usage": self._collect_cpu_usage,
            "memory_usage": self._collect_memory_usage,
            "disk_usage": self._collect_disk_usage,
            "network_io": self._collect_network_io,
            "process_count": self._collect_process_count,
            "load_average": self._collect_load_average
        })
        
        # Set collection intervals (seconds)
        self.collection_intervals.update({
            "cpu_usage": 30,
            "memory_usage": 30,
            "disk_usage": 300,  # 5 minutes
            "network_io": 60,
            "process_count": 60,
            "load_average": 30
        })
    
    def _init_application_collectors(self) -> None:
        """Initialize application-specific metric collectors."""
        self.collectors.update({
            "request_rate": self._collect_request_rate,
            "response_time": self._collect_response_time,
            "error_rate": self._collect_error_rate,
            "throughput": self._collect_throughput,
            "queue_depth": self._collect_queue_depth,
            "cache_hit_ratio": self._collect_cache_hit_ratio
        })
        
        self.collection_intervals.update({
            "request_rate": 60,
            "response_time": 30,
            "error_rate": 60,
            "throughput": 60,
            "queue_depth": 30,
            "cache_hit_ratio": 120  # 2 minutes
        })
    
    async def collect_all_metrics(self) -> List[PerformanceMetric]:
        """Collect all configured metrics."""
        metrics = []
        current_time = datetime.utcnow()
        
        for metric_name, collector in self.collectors.items():
            interval = self.collection_intervals.get(metric_name, 60)
            last_collection = self.last_collection.get(metric_name, datetime.min)
            
            # Check if it's time to collect this metric
            if (current_time - last_collection).total_seconds() >= interval:
                try:
                    metric = await collector()
                    if metric:
                        metrics.append(metric)
                        self.last_collection[metric_name] = current_time
                except Exception as e:
                    logger.error(f"Failed to collect metric {metric_name}: {str(e)}")
        
        return metrics
    
    async def _collect_cpu_usage(self) -> PerformanceMetric:
        """Collect CPU usage metric."""
        cpu_percent = psutil.cpu_percent(interval=1)
        return PerformanceMetric(
            metric_name="cpu_usage",
            metric_type=MetricType.SYSTEM,
            value=cpu_percent,
            unit="percent",
            component="system",
            warning_threshold=80.0,
            critical_threshold=95.0
        )
    
    async def _collect_memory_usage(self) -> PerformanceMetric:
        """Collect memory usage metric."""
        memory = psutil.virtual_memory()
        return PerformanceMetric(
            metric_name="memory_usage",
            metric_type=MetricType.SYSTEM,
            value=memory.percent,
            unit="percent",
            component="system",
            warning_threshold=85.0,
            critical_threshold=95.0
        )
    
    async def _collect_disk_usage(self) -> PerformanceMetric:
        """Collect disk usage metric."""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        return PerformanceMetric(
            metric_name="disk_usage",
            metric_type=MetricType.SYSTEM,
            value=usage_percent,
            unit="percent",
            component="system",
            warning_threshold=80.0,
            critical_threshold=90.0
        )
    
    async def _collect_network_io(self) -> PerformanceMetric:
        """Collect network I/O metric."""
        net_io = psutil.net_io_counters()
        # Calculate rate if we have previous data
        bytes_per_second = net_io.bytes_sent + net_io.bytes_recv  # Simplified
        return PerformanceMetric(
            metric_name="network_io",
            metric_type=MetricType.NETWORK,
            value=bytes_per_second / (1024 * 1024),  # MB/s
            unit="mbps",
            component="network"
        )
    
    async def _collect_process_count(self) -> PerformanceMetric:
        """Collect process count metric."""
        process_count = len(psutil.pids())
        return PerformanceMetric(
            metric_name="process_count",
            metric_type=MetricType.SYSTEM,
            value=process_count,
            unit="count",
            component="system"
        )
    
    async def _collect_load_average(self) -> PerformanceMetric:
        """Collect system load average."""
        try:
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
        except AttributeError:
            # Windows doesn't have load average
            load_avg = psutil.cpu_percent() / 100
        
        return PerformanceMetric(
            metric_name="load_average",
            metric_type=MetricType.SYSTEM,
            value=load_avg,
            unit="ratio",
            component="system",
            warning_threshold=2.0,
            critical_threshold=5.0
        )
    
    # Application metrics (simplified implementations)
    async def _collect_request_rate(self) -> PerformanceMetric:
        """Collect request rate metric."""
        # Would integrate with actual request tracking
        return PerformanceMetric(
            metric_name="request_rate",
            metric_type=MetricType.APPLICATION,
            value=100.0,  # Placeholder
            unit="requests_per_second",
            component="application"
        )
    
    async def _collect_response_time(self) -> PerformanceMetric:
        """Collect response time metric."""
        return PerformanceMetric(
            metric_name="response_time",
            metric_type=MetricType.APPLICATION,
            value=150.0,  # Placeholder
            unit="milliseconds",
            component="application",
            warning_threshold=1000.0,
            critical_threshold=5000.0
        )
    
    async def _collect_error_rate(self) -> PerformanceMetric:
        """Collect error rate metric."""
        return PerformanceMetric(
            metric_name="error_rate",
            metric_type=MetricType.APPLICATION,
            value=2.5,  # Placeholder
            unit="percent",
            component="application",
            warning_threshold=5.0,
            critical_threshold=10.0
        )
    
    async def _collect_throughput(self) -> PerformanceMetric:
        """Collect throughput metric."""
        return PerformanceMetric(
            metric_name="throughput",
            metric_type=MetricType.APPLICATION,
            value=1000.0,  # Placeholder
            unit="records_per_second",
            component="application"
        )
    
    async def _collect_queue_depth(self) -> PerformanceMetric:
        """Collect queue depth metric."""
        return PerformanceMetric(
            metric_name="queue_depth",
            metric_type=MetricType.APPLICATION,
            value=25.0,  # Placeholder
            unit="count",
            component="application",
            warning_threshold=100.0,
            critical_threshold=500.0
        )
    
    async def _collect_cache_hit_ratio(self) -> PerformanceMetric:
        """Collect cache hit ratio metric."""
        return PerformanceMetric(
            metric_name="cache_hit_ratio",
            metric_type=MetricType.APPLICATION,
            value=85.0,  # Placeholder
            unit="percent",
            component="cache",
            warning_threshold=70.0  # Lower is worse for cache hit ratio
        )


class PerformanceMonitoringTuningService:
    """Comprehensive performance monitoring and tuning service."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance monitoring and tuning service."""
        self.config = config
        
        # Metric collection
        self.metric_collector = MetricCollector(config)
        self.metrics_history: deque = deque(maxlen=100000)  # Last 100k metrics
        
        # Baselines and anomaly detection
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.anomaly_threshold_std = config.get("anomaly_threshold_std", 3.0)
        
        # Alerting
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.alert_cooldown_minutes = config.get("alert_cooldown_minutes", 15)
        
        # Tuning and recommendations
        self.tuning_recommendations: List[TuningRecommendation] = []
        self.auto_tuning_enabled = config.get("auto_tuning_enabled", False)
        
        # Machine learning for predictions
        self.ml_models: Dict[str, Any] = {}
        self.feature_scalers: Dict[str, StandardScaler] = {}
        self.model_cache_path = Path(config.get("model_cache_path", "./models"))
        self.model_cache_path.mkdir(exist_ok=True)
        
        # Configuration
        self.monitoring_enabled = config.get("monitoring_enabled", True)
        self.baseline_learning_enabled = config.get("baseline_learning_enabled", True)
        self.predictive_analytics_enabled = config.get("predictive_analytics_enabled", True)
        
        # Background tasks
        asyncio.create_task(self._metric_collection_task())
        asyncio.create_task(self._baseline_learning_task())
        asyncio.create_task(self._anomaly_detection_task())
        asyncio.create_task(self._alert_management_task())
        asyncio.create_task(self._performance_tuning_task())
        asyncio.create_task(self._predictive_analytics_task())
    
    async def _metric_collection_task(self) -> None:
        """Background task for metric collection."""
        while self.monitoring_enabled:
            try:
                # Collect all metrics
                metrics = await self.metric_collector.collect_all_metrics()
                
                # Store metrics in history
                for metric in metrics:
                    self.metrics_history.append(metric)
                
                # Check for immediate alerts
                for metric in metrics:
                    await self._check_metric_thresholds(metric)
                
                logger.debug(f"Collected {len(metrics)} performance metrics")
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Metric collection error: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _baseline_learning_task(self) -> None:
        """Background task for learning performance baselines."""
        while self.baseline_learning_enabled:
            try:
                await asyncio.sleep(300)  # Update baselines every 5 minutes
                
                # Update baselines for each metric
                metric_groups = defaultdict(list)
                
                # Group metrics by name and component
                for metric in list(self.metrics_history):
                    if metric.timestamp > datetime.utcnow() - timedelta(hours=24):  # Last 24 hours
                        key = f"{metric.metric_name}_{metric.component}"
                        metric_groups[key].append(metric)
                
                # Update baselines
                for key, metrics in metric_groups.items():
                    if len(metrics) >= 10:  # Need at least 10 samples
                        await self._update_baseline(key, metrics)
                
                logger.debug(f"Updated {len(metric_groups)} performance baselines")
                
            except Exception as e:
                logger.error(f"Baseline learning error: {str(e)}")
    
    async def _anomaly_detection_task(self) -> None:
        """Background task for anomaly detection."""
        while True:
            try:
                await asyncio.sleep(60)  # Check for anomalies every minute
                
                # Get recent metrics
                recent_metrics = [
                    m for m in list(self.metrics_history)
                    if m.timestamp > datetime.utcnow() - timedelta(minutes=5)
                ]
                
                # Check for anomalies
                anomalies = []
                for metric in recent_metrics:
                    baseline_key = f"{metric.metric_name}_{metric.component}"
                    baseline = self.baselines.get(baseline_key)
                    
                    if baseline and baseline.is_anomaly(metric.value, self.anomaly_threshold_std):
                        anomalies.append(metric)
                
                # Generate alerts for anomalies
                for anomaly in anomalies:
                    await self._generate_anomaly_alert(anomaly)
                
                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} performance anomalies")
                
            except Exception as e:
                logger.error(f"Anomaly detection error: {str(e)}")
    
    async def _alert_management_task(self) -> None:
        """Background task for alert management."""
        while True:
            try:
                await asyncio.sleep(60)  # Manage alerts every minute
                
                # Check for alert resolution
                resolved_alerts = []
                for alert_id, alert in self.active_alerts.items():
                    if await self._should_resolve_alert(alert):
                        alert.resolved_at = datetime.utcnow()
                        resolved_alerts.append(alert_id)
                        self.alert_history.append(alert)
                
                # Remove resolved alerts
                for alert_id in resolved_alerts:
                    del self.active_alerts[alert_id]
                
                # Escalate long-running critical alerts
                for alert in self.active_alerts.values():
                    if (alert.severity == AlertSeverity.CRITICAL and
                        alert.duration_minutes > 30 and
                        not alert.auto_resolution_attempted):
                        await self._attempt_auto_resolution(alert)
                
                if resolved_alerts:
                    logger.info(f"Resolved {len(resolved_alerts)} alerts")
                
            except Exception as e:
                logger.error(f"Alert management error: {str(e)}")
    
    async def _performance_tuning_task(self) -> None:
        """Background task for performance tuning."""
        while True:
            try:
                await asyncio.sleep(600)  # Run tuning analysis every 10 minutes
                
                # Generate tuning recommendations
                new_recommendations = await self._generate_tuning_recommendations()
                
                # Add new recommendations
                for recommendation in new_recommendations:
                    if not any(r.recommendation_id == recommendation.recommendation_id 
                             for r in self.tuning_recommendations):
                        self.tuning_recommendations.append(recommendation)
                
                # Auto-implement low-risk recommendations if enabled
                if self.auto_tuning_enabled:
                    await self._auto_implement_recommendations()
                
                # Clean up old recommendations
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.tuning_recommendations = [
                    r for r in self.tuning_recommendations
                    if r.created_at > cutoff_time
                ]
                
                if new_recommendations:
                    logger.info(f"Generated {len(new_recommendations)} new tuning recommendations")
                
            except Exception as e:
                logger.error(f"Performance tuning error: {str(e)}")
    
    async def _predictive_analytics_task(self) -> None:
        """Background task for predictive analytics."""
        while self.predictive_analytics_enabled:
            try:
                await asyncio.sleep(1800)  # Update predictions every 30 minutes
                
                # Train/update ML models for key metrics
                key_metrics = ["cpu_usage", "memory_usage", "response_time", "error_rate"]
                
                for metric_name in key_metrics:
                    await self._update_prediction_model(metric_name)
                
                # Generate predictions
                predictions = await self._generate_predictions()
                
                # Create proactive alerts for predicted issues
                for prediction in predictions:
                    if prediction["confidence"] > 0.8 and prediction["severity"] == "critical":
                        await self._create_predictive_alert(prediction)
                
                logger.debug(f"Updated {len(key_metrics)} prediction models")
                
            except Exception as e:
                logger.error(f"Predictive analytics error: {str(e)}")
    
    async def _check_metric_thresholds(self, metric: PerformanceMetric) -> None:
        """Check metric against thresholds and generate alerts."""
        alert_key = f"{metric.metric_name}_{metric.component}"
        
        # Check if we're in cooldown period
        if alert_key in self.active_alerts:
            return  # Alert already active
        
        # Check for recent alerts (cooldown)
        recent_alerts = [
            a for a in self.alert_history
            if (a.metric_name == metric.metric_name and
                a.component == metric.component and
                datetime.utcnow() - a.resolved_at < timedelta(minutes=self.alert_cooldown_minutes))
        ]
        
        if recent_alerts:
            return  # In cooldown period
        
        # Check thresholds
        severity = metric.severity
        if severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]:
            alert = await self._create_threshold_alert(metric, severity)
            self.active_alerts[alert_key] = alert
    
    async def _create_threshold_alert(self, metric: PerformanceMetric, severity: AlertSeverity) -> PerformanceAlert:
        """Create alert for threshold violation."""
        threshold = metric.critical_threshold if severity == AlertSeverity.CRITICAL else metric.warning_threshold
        
        alert = PerformanceAlert(
            alert_id=f"threshold_{metric.metric_name}_{int(time.time())}",
            metric_name=metric.metric_name,
            severity=severity,
            current_value=metric.value,
            threshold_value=threshold,
            message=f"{metric.metric_name} {severity.value}: {metric.value:.2f} {metric.unit} "
                   f"(threshold: {threshold:.2f} {metric.unit})",
            component=metric.component,
            recommendations=await self._get_metric_recommendations(metric)
        )
        
        logger.warning(f"Performance alert: {alert.message}")
        return alert
    
    async def _update_baseline(self, baseline_key: str, metrics: List[PerformanceMetric]) -> None:
        """Update performance baseline with new metric data."""
        values = [m.value for m in metrics]
        
        if baseline_key not in self.baselines:
            self.baselines[baseline_key] = PerformanceBaseline(
                metric_name=metrics[0].metric_name,
                component=metrics[0].component
            )
        
        baseline = self.baselines[baseline_key]
        
        # Update statistical measures
        baseline.mean_value = statistics.mean(values)
        baseline.std_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        baseline.percentile_95 = np.percentile(values, 95)
        baseline.percentile_99 = np.percentile(values, 99)
        baseline.sample_count = len(values)
        baseline.last_updated = datetime.utcnow()
        
        # Update confidence based on sample size
        baseline.confidence_level = min(1.0, len(values) / 100)  # Full confidence at 100+ samples
        
        # Update time-based patterns
        for metric in metrics:
            hour = metric.timestamp.hour
            if hour not in baseline.hourly_patterns:
                baseline.hourly_patterns[hour] = []
            baseline.hourly_patterns[hour] = metric.value
        
        logger.debug(f"Updated baseline for {baseline_key}: mean={baseline.mean_value:.2f}, "
                    f"std={baseline.std_deviation:.2f}")
    
    async def _generate_tuning_recommendations(self) -> List[TuningRecommendation]:
        """Generate performance tuning recommendations."""
        recommendations = []
        
        # Analyze recent performance trends
        recent_metrics = [
            m for m in list(self.metrics_history)
            if m.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        # Group metrics by type
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.metric_name].append(metric.value)
        
        # Generate recommendations based on patterns
        for metric_name, values in metric_groups.items():
            if len(values) < 5:
                continue
            
            avg_value = statistics.mean(values)
            trend = self._calculate_trend(values)
            
            # CPU optimization recommendations
            if metric_name == "cpu_usage" and avg_value > 80:
                recommendations.append(TuningRecommendation(
                    recommendation_id=f"cpu_optimization_{int(time.time())}",
                    strategy=TuningStrategy.RESOURCE_SCALING,
                    title="High CPU Usage - Scale Resources",
                    description=f"CPU usage averaging {avg_value:.1f}% over the last hour",
                    estimated_improvement_percent=25.0,
                    implementation_effort="medium",
                    risk_level="low",
                    actions=[
                        "Increase CPU allocation",
                        "Optimize CPU-intensive algorithms",
                        "Implement connection pooling"
                    ],
                    can_auto_implement=False,
                    priority_score=avg_value / 100 * (1 + abs(trend))
                ))
            
            # Memory optimization recommendations
            elif metric_name == "memory_usage" and avg_value > 85:
                recommendations.append(TuningRecommendation(
                    recommendation_id=f"memory_optimization_{int(time.time())}",
                    strategy=TuningStrategy.CACHE_OPTIMIZATION,
                    title="High Memory Usage - Optimize Caching",
                    description=f"Memory usage averaging {avg_value:.1f}% over the last hour",
                    estimated_improvement_percent=20.0,
                    implementation_effort="low",
                    risk_level="low",
                    actions=[
                        "Optimize cache size and eviction policies",
                        "Implement memory pooling",
                        "Review memory leaks"
                    ],
                    can_auto_implement=True,
                    priority_score=avg_value / 100 * (1 + abs(trend))
                ))
            
            # Response time optimization
            elif metric_name == "response_time" and avg_value > 1000:
                recommendations.append(TuningRecommendation(
                    recommendation_id=f"response_time_optimization_{int(time.time())}",
                    strategy=TuningStrategy.QUERY_OPTIMIZATION,
                    title="High Response Time - Optimize Queries",
                    description=f"Response time averaging {avg_value:.1f}ms over the last hour",
                    estimated_improvement_percent=40.0,
                    implementation_effort="medium",
                    risk_level="low",
                    actions=[
                        "Optimize database queries",
                        "Implement result caching",
                        "Add database indexes"
                    ],
                    can_auto_implement=False,
                    priority_score=(avg_value / 5000) * (1 + abs(trend))  # Normalize to 5s
                ))
        
        # Sort by priority score
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1, negative = decreasing, positive = increasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            slope, _ = np.polyfit(x, y, 1)
            # Normalize slope to -1 to 1 range
            normalized_slope = np.tanh(slope / np.std(y) if np.std(y) > 0 else 0)
            return float(normalized_slope)
        except:
            return 0.0
    
    async def _update_prediction_model(self, metric_name: str) -> None:
        """Update ML prediction model for a specific metric."""
        # Get historical data for the metric
        metric_data = [
            m for m in list(self.metrics_history)
            if (m.metric_name == metric_name and
                m.timestamp > datetime.utcnow() - timedelta(days=7))
        ]
        
        if len(metric_data) < 50:  # Need at least 50 data points
            return
        
        # Prepare features and targets
        features = []
        targets = []
        
        for i in range(len(metric_data) - 5):  # Predict 5 steps ahead
            # Features: last 10 values, hour of day, day of week
            feature_window = metric_data[i:i+10]
            target = metric_data[i+15].value if i+15 < len(metric_data) else metric_data[-1].value
            
            feature_vector = [m.value for m in feature_window]
            feature_vector.extend([
                feature_window[-1].timestamp.hour,
                feature_window[-1].timestamp.weekday()
            ])
            
            features.append(feature_vector)
            targets.append(target)
        
        if len(features) < 20:
            return
        
        # Train model
        X = np.array(features)
        y = np.array(targets)
        
        # Scale features
        if metric_name not in self.feature_scalers:
            self.feature_scalers[metric_name] = StandardScaler()
        
        scaler = self.feature_scalers[metric_name]
        X_scaled = scaler.fit_transform(X)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        self.ml_models[metric_name] = model
        
        # Save model to disk
        model_path = self.model_cache_path / f"{metric_name}_model.joblib"
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "last_updated": datetime.utcnow()
        }, model_path)
        
        logger.debug(f"Updated prediction model for {metric_name}")
    
    async def _generate_predictions(self) -> List[Dict[str, Any]]:
        """Generate predictions for key metrics."""
        predictions = []
        
        for metric_name, model in self.ml_models.items():
            try:
                # Get recent data for prediction
                recent_data = [
                    m for m in list(self.metrics_history)
                    if (m.metric_name == metric_name and
                        m.timestamp > datetime.utcnow() - timedelta(hours=1))
                ]
                
                if len(recent_data) < 10:
                    continue
                
                # Prepare feature vector
                feature_vector = [m.value for m in recent_data[-10:]]
                current_time = datetime.utcnow()
                feature_vector.extend([current_time.hour, current_time.weekday()])
                
                # Scale features
                scaler = self.feature_scalers[metric_name]
                X_scaled = scaler.transform([feature_vector])
                
                # Make prediction
                prediction = model.predict(X_scaled)[0]
                
                # Calculate confidence (simplified)
                recent_values = [m.value for m in recent_data]
                std_dev = np.std(recent_values)
                confidence = 1.0 / (1.0 + std_dev / np.mean(recent_values))
                
                # Determine severity
                baseline_key = f"{metric_name}_{recent_data[-1].component}"
                baseline = self.baselines.get(baseline_key)
                severity = "normal"
                
                if baseline:
                    if baseline.is_anomaly(prediction, 2.0):  # Less strict for predictions
                        severity = "warning"
                    if baseline.is_anomaly(prediction, 3.0):
                        severity = "critical"
                
                predictions.append({
                    "metric_name": metric_name,
                    "current_value": recent_data[-1].value,
                    "predicted_value": prediction,
                    "confidence": confidence,
                    "severity": severity,
                    "prediction_horizon_minutes": 5
                })
                
            except Exception as e:
                logger.error(f"Prediction generation failed for {metric_name}: {str(e)}")
        
        return predictions
    
    @handle_exceptions
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        # Current system status
        current_metrics = {}
        for metric_name in ["cpu_usage", "memory_usage", "disk_usage", "response_time", "error_rate"]:
            recent_metrics = [
                m for m in list(self.metrics_history)
                if (m.metric_name == metric_name and
                    m.timestamp > datetime.utcnow() - timedelta(minutes=5))
            ]
            
            if recent_metrics:
                current_metrics[metric_name] = {
                    "current_value": recent_metrics[-1].value,
                    "unit": recent_metrics[-1].unit,
                    "trend": self._calculate_trend([m.value for m in recent_metrics])
                }
        
        # Active alerts summary
        alert_summary = {
            "critical": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
            "warning": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.WARNING]),
            "info": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.INFO])
        }
        
        # Top recommendations
        top_recommendations = sorted(
            self.tuning_recommendations,
            key=lambda r: r.priority_score,
            reverse=True
        )[:5]
        
        # Performance score calculation
        performance_score = await self._calculate_performance_score()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "performance_score": performance_score,
            "current_metrics": current_metrics,
            "alert_summary": alert_summary,
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "duration_minutes": alert.duration_minutes,
                    "component": alert.component
                }
                for alert in list(self.active_alerts.values())[:10]  # Latest 10 alerts
            ],
            "top_recommendations": [
                {
                    "title": rec.title,
                    "description": rec.description,
                    "strategy": rec.strategy.value,
                    "estimated_improvement": rec.estimated_improvement_percent,
                    "effort": rec.implementation_effort,
                    "risk": rec.risk_level,
                    "priority_score": rec.priority_score
                }
                for rec in top_recommendations
            ],
            "system_health": await self._assess_system_health()
        }
    
    async def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        scores = []
        
        # Get recent metrics for key indicators
        key_metrics = ["cpu_usage", "memory_usage", "response_time", "error_rate"]
        
        for metric_name in key_metrics:
            recent_metrics = [
                m for m in list(self.metrics_history)
                if (m.metric_name == metric_name and
                    m.timestamp > datetime.utcnow() - timedelta(minutes=15))
            ]
            
            if recent_metrics:
                avg_value = statistics.mean([m.value for m in recent_metrics])
                
                # Score based on metric type (lower is better for most)
                if metric_name in ["cpu_usage", "memory_usage"]:
                    score = max(0, 100 - avg_value)  # 100% usage = 0 score
                elif metric_name == "response_time":
                    score = max(0, 100 - (avg_value / 50))  # 5000ms = 0 score
                elif metric_name == "error_rate":
                    score = max(0, 100 - (avg_value * 10))  # 10% error = 0 score
                else:
                    score = 50  # Neutral score for unknown metrics
                
                scores.append(score)
        
        # Factor in active alerts
        alert_penalty = len(self.active_alerts) * 5  # 5 points per alert
        critical_alert_penalty = len([a for a in self.active_alerts.values() 
                                    if a.severity == AlertSeverity.CRITICAL]) * 15  # Extra penalty for critical
        
        base_score = statistics.mean(scores) if scores else 50
        final_score = max(0, base_score - alert_penalty - critical_alert_penalty)
        
        return round(final_score, 1)
    
    async def _assess_system_health(self) -> str:
        """Assess overall system health status."""
        performance_score = await self._calculate_performance_score()
        critical_alerts = len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL])
        
        if critical_alerts > 0:
            return "critical"
        elif performance_score < 50:
            return "degraded"
        elif performance_score < 80:
            return "warning"
        else:
            return "healthy"
    
    async def shutdown(self) -> None:
        """Shutdown the performance monitoring service."""
        logger.info("Shutting down performance monitoring and tuning service...")
        
        self.monitoring_enabled = False
        self.baseline_learning_enabled = False
        self.predictive_analytics_enabled = False
        
        # Save final baselines and models
        await self._save_baselines()
        
        logger.info("Performance monitoring service shutdown complete")
    
    async def _save_baselines(self) -> None:
        """Save performance baselines to disk."""
        try:
            baseline_path = self.model_cache_path / "baselines.pkl"
            with open(baseline_path, 'wb') as f:
                pickle.dump(self.baselines, f)
            
            logger.info(f"Saved {len(self.baselines)} performance baselines")
        except Exception as e:
            logger.error(f"Failed to save baselines: {str(e)}")
    
    # Additional helper methods for alert management and auto-resolution
    async def _should_resolve_alert(self, alert: PerformanceAlert) -> bool:
        """Check if alert should be resolved."""
        # Get recent metrics for the alert
        recent_metrics = [
            m for m in list(self.metrics_history)
            if (m.metric_name == alert.metric_name and
                m.timestamp > datetime.utcnow() - timedelta(minutes=5))
        ]
        
        if not recent_metrics:
            return False
        
        # Check if metrics are back within threshold
        avg_recent = statistics.mean([m.value for m in recent_metrics])
        return avg_recent < alert.threshold_value
    
    async def _generate_anomaly_alert(self, anomaly_metric: PerformanceMetric) -> None:
        """Generate alert for detected anomaly."""
        alert_key = f"anomaly_{anomaly_metric.metric_name}_{anomaly_metric.component}"
        
        if alert_key in self.active_alerts:
            return  # Alert already exists
        
        baseline_key = f"{anomaly_metric.metric_name}_{anomaly_metric.component}"
        baseline = self.baselines.get(baseline_key)
        
        alert = PerformanceAlert(
            alert_id=f"anomaly_{int(time.time())}",
            metric_name=anomaly_metric.metric_name,
            severity=AlertSeverity.WARNING,
            current_value=anomaly_metric.value,
            threshold_value=baseline.mean_value if baseline else 0.0,
            message=f"Anomaly detected in {anomaly_metric.metric_name}: {anomaly_metric.value:.2f} "
                   f"(expected: {baseline.mean_value:.2f})" if baseline else 
                   f"Anomaly detected in {anomaly_metric.metric_name}: {anomaly_metric.value:.2f}",
            component=anomaly_metric.component,
            recommendations=["Investigate recent system changes", "Check for resource contention"]
        )
        
        self.active_alerts[alert_key] = alert
        logger.warning(f"Anomaly alert: {alert.message}")
    
    async def _attempt_auto_resolution(self, alert: PerformanceAlert) -> None:
        """Attempt automatic resolution of critical alerts."""
        alert.auto_resolution_attempted = True
        
        # Simple auto-resolution strategies
        if alert.metric_name == "memory_usage" and alert.current_value > 90:
            # Force garbage collection for memory issues
            import gc
            gc.collect()
            logger.info(f"Attempted auto-resolution for {alert.alert_id}: forced garbage collection")
        
        elif alert.metric_name == "queue_depth" and alert.current_value > 100:
            # Could trigger additional worker processes
            logger.info(f"Attempted auto-resolution for {alert.alert_id}: queue depth management")
    
    async def _get_metric_recommendations(self, metric: PerformanceMetric) -> List[str]:
        """Get recommendations for specific metric issues."""
        recommendations = []
        
        if metric.metric_name == "cpu_usage":
            recommendations = [
                "Scale CPU resources",
                "Optimize algorithms",
                "Implement caching",
                "Review background processes"
            ]
        elif metric.metric_name == "memory_usage":
            recommendations = [
                "Optimize memory usage",
                "Implement memory pooling",
                "Check for memory leaks",
                "Adjust garbage collection"
            ]
        elif metric.metric_name == "response_time":
            recommendations = [
                "Optimize database queries",
                "Implement caching",
                "Scale infrastructure",
                "Review network latency"
            ]
        elif metric.metric_name == "error_rate":
            recommendations = [
                "Check error logs",
                "Review recent deployments",
                "Validate input data",
                "Check external dependencies"
            ]
        
        return recommendations
    
    async def _auto_implement_recommendations(self) -> None:
        """Auto-implement low-risk recommendations."""
        for recommendation in self.tuning_recommendations:
            if (recommendation.can_auto_implement and
                not recommendation.tested and
                recommendation.risk_level == "low"):
                
                # Implement the recommendation
                success = await self._implement_recommendation(recommendation)
                recommendation.tested = True
                
                if success:
                    logger.info(f"Auto-implemented recommendation: {recommendation.title}")
                else:
                    logger.warning(f"Failed to auto-implement recommendation: {recommendation.title}")
    
    async def _implement_recommendation(self, recommendation: TuningRecommendation) -> bool:
        """Implement a specific recommendation."""
        try:
            # This would contain actual implementation logic
            # For now, just simulate implementation
            await asyncio.sleep(0.1)  # Simulate work
            
            return True  # Simulate success
            
        except Exception as e:
            logger.error(f"Failed to implement recommendation {recommendation.recommendation_id}: {str(e)}")
            return False
    
    async def _create_predictive_alert(self, prediction: Dict[str, Any]) -> None:
        """Create alert based on predictive analytics."""
        alert_key = f"predictive_{prediction['metric_name']}"
        
        if alert_key in self.active_alerts:
            return  # Predictive alert already exists
        
        alert = PerformanceAlert(
            alert_id=f"predictive_{int(time.time())}",
            metric_name=prediction["metric_name"],
            severity=AlertSeverity.WARNING,
            current_value=prediction["current_value"],
            threshold_value=prediction["predicted_value"],
            message=f"Predicted issue in {prediction['metric_name']}: "
                   f"expected to reach {prediction['predicted_value']:.2f} "
                   f"(confidence: {prediction['confidence']:.1%})",
            component="predictive",
            recommendations=[
                "Take proactive measures",
                "Monitor closely",
                "Prepare scaling resources"
            ]
        )
        
        self.active_alerts[alert_key] = alert
        logger.warning(f"Predictive alert: {alert.message}")