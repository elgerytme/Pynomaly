"""
Monitoring and observability service for integration layer.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from enum import Enum

from data_platform.integration.domain.entities.integration_config import IntegrationConfig
from data_platform.integration.domain.value_objects.performance_metrics import (
    PerformanceMetrics, SystemMetrics, ApplicationMetrics, PackageMetrics, WorkflowMetrics
)
from software.interfaces.shared.error_handling import handle_exceptions


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert entity."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    component: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


@dataclass
class MetricTimeSeries:
    """Time series data for a metric."""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a value to the time series."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.values.append(value)
        self.timestamps.append(timestamp)
        self.last_updated = timestamp
    
    def get_average(self, minutes: int = 5) -> float:
        """Get average value for the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        recent_values = [
            value for value, timestamp in zip(self.values, self.timestamps)
            if timestamp > cutoff_time
        ]
        
        return sum(recent_values) / len(recent_values) if recent_values else 0.0
    
    def get_max(self, minutes: int = 5) -> float:
        """Get maximum value for the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        recent_values = [
            value for value, timestamp in zip(self.values, self.timestamps)
            if timestamp > cutoff_time
        ]
        
        return max(recent_values) if recent_values else 0.0


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    target: str
    check_function: Callable[[], bool]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    success_threshold: int = 2
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    is_healthy: bool = True
    last_check: Optional[datetime] = None
    last_error: Optional[str] = None


class MonitoringService:
    """Service for monitoring and observability of integrated packages."""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize the monitoring service."""
        self.config = config
        self.metrics_timeseries: Dict[str, MetricTimeSeries] = {}
        self.alerts: Dict[str, Alert] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        
        # Distributed tracing
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.active_spans: Dict[str, Dict[str, Any]] = {}
        
        # Initialize monitoring components
        self._initialize_health_checks()
        self._initialize_dashboards()
        
        # Start monitoring tasks
        asyncio.create_task(self._metrics_collection_task())
        asyncio.create_task(self._health_check_task())
        asyncio.create_task(self._alert_evaluation_task())
    
    def _initialize_health_checks(self) -> None:
        """Initialize health checks for all packages."""
        for package_name, package_config in self.config.packages.items():
            if package_config.enabled:
                health_check = HealthCheck(
                    name=f"{package_name}_health",
                    target=package_name,
                    check_function=lambda: self._check_package_health(package_name),
                    interval_seconds=self.config.monitoring.health_check_interval,
                    timeout_seconds=package_config.timeout_seconds,
                    failure_threshold=3,
                    success_threshold=2
                )
                self.health_checks[package_name] = health_check
    
    def _initialize_dashboards(self) -> None:
        """Initialize monitoring dashboards."""
        # System overview dashboard
        self.dashboards["system_overview"] = {
            "title": "System Overview",
            "panels": [
                {
                    "title": "CPU Usage",
                    "type": "gauge",
                    "metric": "system.cpu_usage_percent",
                    "unit": "%",
                    "thresholds": [70, 85, 95]
                },
                {
                    "title": "Memory Usage",
                    "type": "gauge",
                    "metric": "system.memory_usage_percent",
                    "unit": "%",
                    "thresholds": [70, 85, 95]
                },
                {
                    "title": "Request Rate",
                    "type": "graph",
                    "metric": "application.request_rate",
                    "unit": "req/s",
                    "timespan": "1h"
                },
                {
                    "title": "Error Rate",
                    "type": "graph",
                    "metric": "application.error_rate",
                    "unit": "%",
                    "timespan": "1h"
                }
            ]
        }
        
        # Package performance dashboard
        self.dashboards["package_performance"] = {
            "title": "Package Performance",
            "panels": [
                {
                    "title": "Response Times",
                    "type": "graph",
                    "metric": "package.response_time",
                    "unit": "ms",
                    "timespan": "1h"
                },
                {
                    "title": "Success Rate",
                    "type": "stat",
                    "metric": "package.success_rate",
                    "unit": "%",
                    "timespan": "1h"
                },
                {
                    "title": "Throughput",
                    "type": "graph",
                    "metric": "package.throughput",
                    "unit": "ops/s",
                    "timespan": "1h"
                }
            ]
        }
        
        # Workflow monitoring dashboard
        self.dashboards["workflow_monitoring"] = {
            "title": "Workflow Monitoring",
            "panels": [
                {
                    "title": "Active Workflows",
                    "type": "stat",
                    "metric": "workflow.active_count",
                    "unit": "count"
                },
                {
                    "title": "Workflow Success Rate",
                    "type": "gauge",
                    "metric": "workflow.success_rate",
                    "unit": "%",
                    "thresholds": [80, 90, 95]
                },
                {
                    "title": "Average Execution Time",
                    "type": "graph",
                    "metric": "workflow.execution_time",
                    "unit": "s",
                    "timespan": "1h"
                }
            ]
        }
    
    async def _metrics_collection_task(self) -> None:
        """Background task for collecting metrics."""
        interval = self.config.monitoring.metrics_collection_interval
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                await self._store_metrics("system", system_metrics)
                
                # Collect application metrics
                app_metrics = await self._collect_application_metrics()
                await self._store_metrics("application", app_metrics)
                
                # Collect package metrics
                for package_name in self.config.get_enabled_packages():
                    package_metrics = await self._collect_package_metrics(package_name)
                    await self._store_metrics(f"package.{package_name}", package_metrics)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {str(e)}")
    
    async def _health_check_task(self) -> None:
        """Background task for health checks."""
        while True:
            try:
                for health_check in self.health_checks.values():
                    await self._run_health_check(health_check)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
    
    async def _alert_evaluation_task(self) -> None:
        """Background task for evaluating alerts."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Evaluate alert conditions
                await self._evaluate_system_alerts()
                await self._evaluate_package_alerts()
                await self._evaluate_workflow_alerts()
                
                # Auto-resolve alerts
                await self._auto_resolve_alerts()
                
            except Exception as e:
                logger.error(f"Alert evaluation error: {str(e)}")
    
    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            return {
                "cpu_usage_percent": cpu_percent,
                "cpu_count": cpu_count,
                "memory_usage_percent": memory.percent,
                "memory_total_mb": memory.total / 1024 / 1024,
                "memory_available_mb": memory.available / 1024 / 1024,
                "disk_usage_percent": disk.percent,
                "disk_total_gb": disk.total / 1024 / 1024 / 1024,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "network_packets_sent": network.packets_sent,
                "network_packets_recv": network.packets_recv
            }
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return {}
    
    async def _collect_application_metrics(self) -> Dict[str, float]:
        """Collect application-level metrics."""
        current_time = time.time()
        
        # Calculate request rate (requests per second)
        total_requests = sum(self.request_counts.values())
        request_rate = total_requests / 60  # Approximate rate per second
        
        # Calculate error rate
        total_errors = sum(self.error_counts.values())
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0
        
        # Calculate average response time
        all_response_times = []
        for times in self.response_times.values():
            all_response_times.extend(times)
        
        avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0.0
        
        # Get process metrics
        try:
            import psutil
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            process_cpu = process.cpu_percent()
        except:
            process_memory = 0.0
            process_cpu = 0.0
        
        return {
            "request_rate": request_rate,
            "error_rate": error_rate,
            "average_response_time": avg_response_time,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "process_memory_mb": process_memory,
            "process_cpu_percent": process_cpu,
            "active_connections": len(self.active_spans)
        }
    
    async def _collect_package_metrics(self, package_name: str) -> Dict[str, float]:
        """Collect metrics for a specific package."""
        package_requests = self.request_counts.get(package_name, 0)
        package_errors = self.error_counts.get(package_name, 0)
        package_response_times = self.response_times.get(package_name, [])
        
        return {
            "request_count": package_requests,
            "error_count": package_errors,
            "success_rate": ((package_requests - package_errors) / package_requests * 100) if package_requests > 0 else 0.0,
            "average_response_time": sum(package_response_times) / len(package_response_times) if package_response_times else 0.0,
            "throughput": package_requests / 60  # Approximate throughput per second
        }
    
    async def _store_metrics(self, metric_prefix: str, metrics: Dict[str, float]) -> None:
        """Store metrics in time series."""
        timestamp = datetime.utcnow()
        
        for metric_name, value in metrics.items():
            full_metric_name = f"{metric_prefix}.{metric_name}"
            
            if full_metric_name not in self.metrics_timeseries:
                self.metrics_timeseries[full_metric_name] = MetricTimeSeries(name=full_metric_name)
            
            self.metrics_timeseries[full_metric_name].add_value(value, timestamp)
    
    async def _run_health_check(self, health_check: HealthCheck) -> None:
        """Run a single health check."""
        try:
            # Check if it's time to run the health check
            if (health_check.last_check and 
                datetime.utcnow() - health_check.last_check < timedelta(seconds=health_check.interval_seconds)):
                return
            
            # Run the health check
            start_time = time.time()
            is_healthy = await asyncio.wait_for(
                asyncio.create_task(asyncio.to_thread(health_check.check_function)),
                timeout=health_check.timeout_seconds
            )
            execution_time = time.time() - start_time
            
            health_check.last_check = datetime.utcnow()
            health_check.last_error = None
            
            if is_healthy:
                health_check.consecutive_successes += 1
                health_check.consecutive_failures = 0
                
                # Mark as healthy if success threshold is met
                if health_check.consecutive_successes >= health_check.success_threshold:
                    if not health_check.is_healthy:
                        health_check.is_healthy = True
                        await self._create_alert(
                            f"{health_check.name}_recovered",
                            f"Health check {health_check.name} recovered",
                            f"Health check for {health_check.target} is now healthy after {health_check.consecutive_successes} consecutive successes",
                            AlertSeverity.LOW,
                            health_check.target
                        )
            else:
                health_check.consecutive_failures += 1
                health_check.consecutive_successes = 0
                
                # Mark as unhealthy if failure threshold is met
                if health_check.consecutive_failures >= health_check.failure_threshold:
                    if health_check.is_healthy:
                        health_check.is_healthy = False
                        await self._create_alert(
                            f"{health_check.name}_failed",
                            f"Health check {health_check.name} failed",
                            f"Health check for {health_check.target} failed after {health_check.consecutive_failures} consecutive failures",
                            AlertSeverity.HIGH,
                            health_check.target
                        )
            
            # Store health check metrics
            await self._store_metrics(f"health.{health_check.name}", {
                "is_healthy": 1.0 if is_healthy else 0.0,
                "execution_time": execution_time,
                "consecutive_failures": health_check.consecutive_failures,
                "consecutive_successes": health_check.consecutive_successes
            })
            
        except asyncio.TimeoutError:
            health_check.last_error = "Health check timed out"
            health_check.consecutive_failures += 1
            health_check.consecutive_successes = 0
            
        except Exception as e:
            health_check.last_error = str(e)
            health_check.consecutive_failures += 1
            health_check.consecutive_successes = 0
    
    def _check_package_health(self, package_name: str) -> bool:
        """Check health of a specific package."""
        # This is a simplified health check
        # In a real implementation, you'd check if the package is responding
        return package_name in self.config.get_enabled_packages()
    
    async def _evaluate_system_alerts(self) -> None:
        """Evaluate system-level alerts."""
        thresholds = self.config.monitoring.alert_thresholds
        
        # CPU usage alert
        cpu_metric = self.metrics_timeseries.get("system.cpu_usage_percent")
        if cpu_metric:
            avg_cpu = cpu_metric.get_average(5)
            if avg_cpu > thresholds.get("cpu_usage", 80):
                await self._create_alert(
                    "high_cpu_usage",
                    "High CPU Usage",
                    f"CPU usage is {avg_cpu:.1f}% (threshold: {thresholds.get('cpu_usage', 80)}%)",
                    AlertSeverity.HIGH,
                    "system"
                )
        
        # Memory usage alert
        memory_metric = self.metrics_timeseries.get("system.memory_usage_percent")
        if memory_metric:
            avg_memory = memory_metric.get_average(5)
            if avg_memory > thresholds.get("memory_usage", 85):
                await self._create_alert(
                    "high_memory_usage",
                    "High Memory Usage",
                    f"Memory usage is {avg_memory:.1f}% (threshold: {thresholds.get('memory_usage', 85)}%)",
                    AlertSeverity.HIGH,
                    "system"
                )
        
        # Disk usage alert
        disk_metric = self.metrics_timeseries.get("system.disk_usage_percent")
        if disk_metric:
            avg_disk = disk_metric.get_average(5)
            if avg_disk > thresholds.get("disk_usage", 90):
                await self._create_alert(
                    "high_disk_usage",
                    "High Disk Usage",
                    f"Disk usage is {avg_disk:.1f}% (threshold: {thresholds.get('disk_usage', 90)}%)",
                    AlertSeverity.CRITICAL,
                    "system"
                )
    
    async def _evaluate_package_alerts(self) -> None:
        """Evaluate package-specific alerts."""
        for package_name in self.config.get_enabled_packages():
            # Error rate alert
            error_rate_metric = self.metrics_timeseries.get(f"package.{package_name}.error_rate")
            if error_rate_metric:
                avg_error_rate = error_rate_metric.get_average(5)
                if avg_error_rate > 5.0:  # 5% error rate threshold
                    await self._create_alert(
                        f"high_error_rate_{package_name}",
                        f"High Error Rate - {package_name}",
                        f"Error rate for {package_name} is {avg_error_rate:.1f}% (threshold: 5%)",
                        AlertSeverity.HIGH,
                        package_name
                    )
            
            # Response time alert
            response_time_metric = self.metrics_timeseries.get(f"package.{package_name}.average_response_time")
            if response_time_metric:
                avg_response_time = response_time_metric.get_average(5)
                if avg_response_time > 5000:  # 5 second threshold
                    await self._create_alert(
                        f"slow_response_{package_name}",
                        f"Slow Response Time - {package_name}",
                        f"Response time for {package_name} is {avg_response_time:.0f}ms (threshold: 5000ms)",
                        AlertSeverity.MEDIUM,
                        package_name
                    )
    
    async def _evaluate_workflow_alerts(self) -> None:
        """Evaluate workflow-specific alerts."""
        # This would evaluate workflow metrics if available
        # For now, this is a placeholder
        pass
    
    async def _create_alert(self, alert_id: str, title: str, description: str, 
                          severity: AlertSeverity, component: str, 
                          metric_name: str = "", metric_value: float = 0.0, 
                          threshold: float = 0.0) -> None:
        """Create a new alert."""
        # Check if alert already exists and is not resolved
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            return
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            component=component,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )
        
        self.alerts[alert_id] = alert
        
        logger.warning(f"Alert created: {title} - {description}")
        
        # Send notifications if configured
        await self._send_alert_notification(alert)
    
    async def _send_alert_notification(self, alert: Alert) -> None:
        """Send alert notification to configured channels."""
        # This is a placeholder for notification logic
        # In a real implementation, you'd integrate with notification services
        logger.info(f"Sending alert notification: {alert.title}")
    
    async def _auto_resolve_alerts(self) -> None:
        """Auto-resolve alerts based on current metrics."""
        for alert_id, alert in self.alerts.items():
            if alert.resolved:
                continue
            
            # Check if alert condition is no longer met
            should_resolve = False
            
            if alert.metric_name and alert.metric_name in self.metrics_timeseries:
                metric = self.metrics_timeseries[alert.metric_name]
                current_value = metric.get_average(2)  # Check last 2 minutes
                
                # Simple resolution logic - if value is below threshold for 2 minutes
                if current_value < alert.threshold:
                    should_resolve = True
            
            if should_resolve:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                logger.info(f"Alert auto-resolved: {alert.title}")
    
    @handle_exceptions
    async def track_request(self, component: str, operation: str, 
                          execution_time: float, success: bool) -> None:
        """Track a request for monitoring."""
        self.request_counts[component] += 1
        
        if not success:
            self.error_counts[component] += 1
        
        self.response_times[component].append(execution_time)
        
        # Keep only recent response times
        if len(self.response_times[component]) > 1000:
            self.response_times[component] = self.response_times[component][-1000:]
        
        # Create trace span
        span_id = f"{component}_{operation}_{time.time()}"
        span = {
            "span_id": span_id,
            "component": component,
            "operation": operation,
            "start_time": time.time() - execution_time,
            "end_time": time.time(),
            "duration": execution_time,
            "success": success,
            "tags": {}
        }
        
        self.traces[span_id] = span
    
    @handle_exceptions
    async def start_trace(self, trace_id: str, operation: str) -> str:
        """Start a distributed trace."""
        span_id = f"{trace_id}_{operation}_{time.time()}"
        
        span = {
            "span_id": span_id,
            "trace_id": trace_id,
            "operation": operation,
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "success": None,
            "tags": {},
            "parent_span": None
        }
        
        self.active_spans[span_id] = span
        return span_id
    
    @handle_exceptions
    async def end_trace(self, span_id: str, success: bool, tags: Optional[Dict[str, Any]] = None) -> None:
        """End a distributed trace span."""
        if span_id not in self.active_spans:
            return
        
        span = self.active_spans[span_id]
        span["end_time"] = time.time()
        span["duration"] = span["end_time"] - span["start_time"]
        span["success"] = success
        
        if tags:
            span["tags"].update(tags)
        
        # Move to completed traces
        self.traces[span_id] = span
        del self.active_spans[span_id]
    
    @handle_exceptions
    async def get_metrics(self, metric_names: Optional[List[str]] = None, 
                         minutes: int = 60) -> Dict[str, Any]:
        """Get metrics data."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        if metric_names is None:
            metric_names = list(self.metrics_timeseries.keys())
        
        result = {}
        
        for metric_name in metric_names:
            if metric_name in self.metrics_timeseries:
                metric = self.metrics_timeseries[metric_name]
                
                # Get recent values
                recent_data = [
                    {"value": value, "timestamp": timestamp}
                    for value, timestamp in zip(metric.values, metric.timestamps)
                    if timestamp > cutoff_time
                ]
                
                result[metric_name] = {
                    "name": metric_name,
                    "data": recent_data,
                    "average": metric.get_average(minutes),
                    "max": metric.get_max(minutes),
                    "last_value": list(metric.values)[-1] if metric.values else None,
                    "last_updated": metric.last_updated
                }
        
        return result
    
    @handle_exceptions
    async def get_alerts(self, resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get alerts."""
        alerts = []
        
        for alert in self.alerts.values():
            if resolved is None or alert.resolved == resolved:
                alerts.append({
                    "id": alert.id,
                    "title": alert.title,
                    "description": alert.description,
                    "severity": alert.severity.value,
                    "component": alert.component,
                    "metric_name": alert.metric_name,
                    "metric_value": alert.metric_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at,
                    "acknowledged": alert.acknowledged,
                    "acknowledged_at": alert.acknowledged_at,
                    "acknowledged_by": alert.acknowledged_by
                })
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return alerts
    
    @handle_exceptions
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        overall_healthy = True
        component_health = {}
        
        for name, health_check in self.health_checks.items():
            component_health[name] = {
                "healthy": health_check.is_healthy,
                "last_check": health_check.last_check,
                "consecutive_failures": health_check.consecutive_failures,
                "consecutive_successes": health_check.consecutive_successes,
                "last_error": health_check.last_error
            }
            
            if not health_check.is_healthy:
                overall_healthy = False
        
        return {
            "overall_healthy": overall_healthy,
            "components": component_health,
            "timestamp": datetime.utcnow()
        }
    
    @handle_exceptions
    async def get_dashboard_data(self, dashboard_name: str) -> Dict[str, Any]:
        """Get dashboard data."""
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_name}")
        
        dashboard = self.dashboards[dashboard_name]
        
        # Get metrics for all panels
        for panel in dashboard["panels"]:
            metric_name = panel["metric"]
            if metric_name in self.metrics_timeseries:
                metric = self.metrics_timeseries[metric_name]
                
                if panel["type"] == "gauge":
                    panel["current_value"] = list(metric.values)[-1] if metric.values else 0
                    panel["average_value"] = metric.get_average(5)
                elif panel["type"] == "graph":
                    timespan_minutes = 60  # Default
                    if panel.get("timespan") == "30m":
                        timespan_minutes = 30
                    elif panel.get("timespan") == "1h":
                        timespan_minutes = 60
                    elif panel.get("timespan") == "24h":
                        timespan_minutes = 1440
                    
                    cutoff_time = datetime.utcnow() - timedelta(minutes=timespan_minutes)
                    panel["data"] = [
                        {"value": value, "timestamp": timestamp}
                        for value, timestamp in zip(metric.values, metric.timestamps)
                        if timestamp > cutoff_time
                    ]
                elif panel["type"] == "stat":
                    panel["value"] = metric.get_average(60)
        
        return dashboard
    
    async def shutdown(self) -> None:
        """Shutdown the monitoring service."""
        logger.info("Shutting down monitoring service...")
        
        # Clear all data
        self.metrics_timeseries.clear()
        self.alerts.clear()
        self.health_checks.clear()
        self.traces.clear()
        self.active_spans.clear()
        
        logger.info("Monitoring service shutdown complete")