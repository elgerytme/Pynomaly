#!/usr/bin/env python3
"""
Production Monitoring and Alerting System
Comprehensive monitoring for the hexagonal architecture deployment
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import aiohttp
import psutil
import subprocess
import yaml
from pathlib import Path


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricValue:
    """Represents a metric value with metadata"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    labels: Dict[str, str]
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class Alert:
    """Represents a monitoring alert"""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    service: str
    metric: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Health check configuration and result"""
    name: str
    endpoint: str
    method: str = "GET"
    expected_status: int = 200
    timeout: int = 30
    interval: int = 60
    retries: int = 3
    last_check: Optional[datetime] = None
    status: ServiceStatus = ServiceStatus.UNKNOWN
    response_time: Optional[float] = None
    error_message: Optional[str] = None


class ProductionMonitor:
    """Main production monitoring system"""
    
    def __init__(self, config_path: str = "monitoring-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.metrics: Dict[str, MetricValue] = {}
        self.alerts: Dict[str, Alert] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alert_handlers: List[Callable] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'/tmp/production-monitoring-{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self._initialize_health_checks()
        self._register_alert_handlers()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "services": {
                "api-gateway": {
                    "health_endpoint": "/health",
                    "metrics_endpoint": "/metrics",
                    "port": 8080
                },
                "data-quality-service": {
                    "health_endpoint": "/health",
                    "metrics_endpoint": "/metrics", 
                    "port": 8081
                },
                "anomaly-detection-service": {
                    "health_endpoint": "/health",
                    "metrics_endpoint": "/metrics",
                    "port": 8082
                },
                "workflow-engine": {
                    "health_endpoint": "/health",
                    "metrics_endpoint": "/metrics",
                    "port": 8083
                }
            },
            "thresholds": {
                "cpu_usage": {"warning": 70.0, "critical": 90.0},
                "memory_usage": {"warning": 80.0, "critical": 95.0},
                "disk_usage": {"warning": 85.0, "critical": 95.0},
                "response_time": {"warning": 1000.0, "critical": 5000.0},
                "error_rate": {"warning": 1.0, "critical": 5.0}
            },
            "intervals": {
                "health_check": 30,
                "metrics_collection": 60,
                "alert_evaluation": 30
            },
            "alerting": {
                "slack_webhook": "",
                "email_recipients": [],
                "pagerduty_key": ""
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge user config with defaults
                return {**default_config, **user_config}
        
        return default_config
    
    def _initialize_health_checks(self):
        """Initialize health checks for all services"""
        for service_name, service_config in self.config["services"].items():
            health_check = HealthCheck(
                name=service_name,
                endpoint=f"http://localhost:{service_config['port']}{service_config['health_endpoint']}",
                interval=self.config["intervals"]["health_check"]
            )
            self.health_checks[service_name] = health_check
    
    def _register_alert_handlers(self):
        """Register alert notification handlers"""
        if self.config["alerting"]["slack_webhook"]:
            self.alert_handlers.append(self._send_slack_alert)
        
        if self.config["alerting"]["email_recipients"]:
            self.alert_handlers.append(self._send_email_alert)
        
        if self.config["alerting"]["pagerduty_key"]:
            self.alert_handlers.append(self._send_pagerduty_alert)
    
    async def start_monitoring(self):
        """Start the main monitoring loop"""
        self.logger.info("Starting production monitoring system")
        
        # Create monitoring tasks
        tasks = [
            self._health_check_loop(),
            self._metrics_collection_loop(),
            self._alert_evaluation_loop(),
            self._kubernetes_monitoring_loop(),
            self._log_analysis_loop()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            raise
    
    async def _health_check_loop(self):
        """Continuous health checking of services"""
        while True:
            try:
                tasks = []
                for health_check in self.health_checks.values():
                    tasks.append(self._perform_health_check(health_check))
                
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(self.config["intervals"]["health_check"])
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(30)
    
    async def _perform_health_check(self, health_check: HealthCheck):
        """Perform individual health check"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=health_check.timeout)) as session:
                async with session.request(
                    health_check.method,
                    health_check.endpoint
                ) as response:
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    if response.status == health_check.expected_status:
                        health_check.status = ServiceStatus.HEALTHY
                        health_check.error_message = None
                    else:
                        health_check.status = ServiceStatus.UNHEALTHY
                        health_check.error_message = f"HTTP {response.status}"
                    
                    health_check.response_time = response_time
                    health_check.last_check = datetime.now()
                    
                    # Check response time thresholds
                    response_time_threshold = self.config["thresholds"]["response_time"]
                    if response_time > response_time_threshold["critical"]:
                        await self._create_alert(
                            f"high_response_time_{health_check.name}",
                            AlertSeverity.CRITICAL,
                            f"High Response Time - {health_check.name}",
                            f"Response time {response_time:.2f}ms exceeds critical threshold {response_time_threshold['critical']}ms",
                            health_check.name,
                            "response_time",
                            response_time,
                            response_time_threshold["critical"]
                        )
                    elif response_time > response_time_threshold["warning"]:
                        await self._create_alert(
                            f"high_response_time_{health_check.name}",
                            AlertSeverity.WARNING,
                            f"High Response Time - {health_check.name}",
                            f"Response time {response_time:.2f}ms exceeds warning threshold {response_time_threshold['warning']}ms",
                            health_check.name,
                            "response_time",
                            response_time,
                            response_time_threshold["warning"]
                        )
                    
        except asyncio.TimeoutError:
            health_check.status = ServiceStatus.UNHEALTHY
            health_check.error_message = "Timeout"
            health_check.response_time = None
            health_check.last_check = datetime.now()
            
            await self._create_alert(
                f"service_timeout_{health_check.name}",
                AlertSeverity.CRITICAL,
                f"Service Timeout - {health_check.name}",
                f"Service {health_check.name} is not responding within {health_check.timeout}s timeout",
                health_check.name,
                "availability",
                0.0,
                1.0
            )
            
        except Exception as e:
            health_check.status = ServiceStatus.UNHEALTHY
            health_check.error_message = str(e)
            health_check.response_time = None
            health_check.last_check = datetime.now()
            
            self.logger.error(f"Health check failed for {health_check.name}: {e}")
    
    async def _metrics_collection_loop(self):
        """Collect system and application metrics"""
        while True:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await asyncio.sleep(self.config["intervals"]["metrics_collection"])
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics["cpu_usage"] = MetricValue(
            name="cpu_usage",
            value=cpu_percent,
            unit="percent",
            timestamp=timestamp,
            labels={"type": "system"},
            threshold_warning=self.config["thresholds"]["cpu_usage"]["warning"],
            threshold_critical=self.config["thresholds"]["cpu_usage"]["critical"]
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics["memory_usage"] = MetricValue(
            name="memory_usage",
            value=memory.percent,
            unit="percent",
            timestamp=timestamp,
            labels={"type": "system"},
            threshold_warning=self.config["thresholds"]["memory_usage"]["warning"],
            threshold_critical=self.config["thresholds"]["memory_usage"]["critical"]
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.metrics["disk_usage"] = MetricValue(
            name="disk_usage",
            value=disk_percent,
            unit="percent",
            timestamp=timestamp,
            labels={"type": "system", "mount": "/"},
            threshold_warning=self.config["thresholds"]["disk_usage"]["warning"],
            threshold_critical=self.config["thresholds"]["disk_usage"]["critical"]
        )
        
        # Network I/O
        network = psutil.net_io_counters()
        self.metrics["network_bytes_sent"] = MetricValue(
            name="network_bytes_sent",
            value=network.bytes_sent,
            unit="bytes",
            timestamp=timestamp,
            labels={"type": "system"}
        )
        
        self.metrics["network_bytes_recv"] = MetricValue(
            name="network_bytes_recv", 
            value=network.bytes_recv,
            unit="bytes",
            timestamp=timestamp,
            labels={"type": "system"}
        )
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        for service_name, service_config in self.config["services"].items():
            try:
                metrics_url = f"http://localhost:{service_config['port']}{service_config['metrics_endpoint']}"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(metrics_url, timeout=10) as response:
                        if response.status == 200:
                            metrics_text = await response.text()
                            # Parse Prometheus-style metrics
                            await self._parse_prometheus_metrics(service_name, metrics_text)
                        
            except Exception as e:
                self.logger.warning(f"Failed to collect metrics from {service_name}: {e}")
    
    async def _parse_prometheus_metrics(self, service_name: str, metrics_text: str):
        """Parse Prometheus-format metrics"""
        timestamp = datetime.now()
        
        for line in metrics_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        metric_value = float(parts[1])
                        
                        # Store metric with service label
                        key = f"{service_name}_{metric_name}"
                        self.metrics[key] = MetricValue(
                            name=metric_name,
                            value=metric_value,
                            unit="",
                            timestamp=timestamp,
                            labels={"service": service_name}
                        )
                        
                except (ValueError, IndexError):
                    continue
    
    async def _kubernetes_monitoring_loop(self):
        """Monitor Kubernetes cluster health"""
        while True:
            try:
                await self._check_kubernetes_cluster()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Kubernetes monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_kubernetes_cluster(self):
        """Check Kubernetes cluster health"""
        try:
            # Check node status
            result = subprocess.run(
                ["kubectl", "get", "nodes", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                nodes = result.stdout.strip().split('\n')
                unhealthy_nodes = []
                
                for node_line in nodes:
                    if node_line.strip():
                        parts = node_line.split()
                        if len(parts) >= 2:
                            node_name = parts[0]
                            node_status = parts[1]
                            
                            if node_status != "Ready":
                                unhealthy_nodes.append(f"{node_name}: {node_status}")
                
                if unhealthy_nodes:
                    await self._create_alert(
                        "kubernetes_unhealthy_nodes",
                        AlertSeverity.CRITICAL,
                        "Kubernetes Unhealthy Nodes",
                        f"Found unhealthy nodes: {', '.join(unhealthy_nodes)}",
                        "kubernetes",
                        "node_health",
                        len(unhealthy_nodes),
                        0
                    )
            
            # Check pod status
            result = subprocess.run(
                ["kubectl", "get", "pods", "--all-namespaces", "--field-selector=status.phase!=Running", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                unhealthy_pods = result.stdout.strip().split('\n')
                if len(unhealthy_pods) > 5:  # Threshold for alert
                    await self._create_alert(
                        "kubernetes_unhealthy_pods",
                        AlertSeverity.WARNING,
                        "Kubernetes Unhealthy Pods",
                        f"Found {len(unhealthy_pods)} unhealthy pods",
                        "kubernetes",
                        "pod_health",
                        len(unhealthy_pods),
                        5
                    )
                        
        except subprocess.TimeoutExpired:
            await self._create_alert(
                "kubernetes_api_timeout",
                AlertSeverity.CRITICAL,
                "Kubernetes API Timeout",
                "Kubernetes API is not responding",
                "kubernetes",
                "api_availability",
                0.0,
                1.0
            )
        except Exception as e:
            self.logger.error(f"Kubernetes check failed: {e}")
    
    async def _log_analysis_loop(self):
        """Analyze application logs for anomalies"""
        while True:
            try:
                await self._analyze_application_logs()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Log analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_application_logs(self):
        """Analyze application logs for error patterns"""
        try:
            # Get recent logs from kubectl
            result = subprocess.run([
                "kubectl", "logs", "--since=5m", "--all-containers=true",
                "-l", "app.kubernetes.io/part-of=hexagonal-architecture",
                "--tail=1000"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logs = result.stdout
                error_count = logs.lower().count('error')
                warning_count = logs.lower().count('warning')
                exception_count = logs.lower().count('exception')
                
                total_issues = error_count + exception_count
                
                if total_issues > 50:  # Threshold for high error rate
                    await self._create_alert(
                        "high_error_rate_logs",
                        AlertSeverity.CRITICAL,
                        "High Error Rate in Logs",
                        f"Found {total_issues} errors/exceptions in last 5 minutes (errors: {error_count}, exceptions: {exception_count})",
                        "application",
                        "error_rate",
                        total_issues,
                        50
                    )
                elif total_issues > 20:
                    await self._create_alert(
                        "elevated_error_rate_logs",
                        AlertSeverity.WARNING,
                        "Elevated Error Rate in Logs",
                        f"Found {total_issues} errors/exceptions in last 5 minutes (errors: {error_count}, exceptions: {exception_count})",
                        "application",
                        "error_rate",
                        total_issues,
                        20
                    )
                        
        except subprocess.TimeoutExpired:
            self.logger.warning("Log analysis timed out")
        except Exception as e:
            self.logger.error(f"Log analysis failed: {e}")
    
    async def _alert_evaluation_loop(self):
        """Evaluate metrics against thresholds and generate alerts"""
        while True:
            try:
                await self._evaluate_metric_thresholds()
                await asyncio.sleep(self.config["intervals"]["alert_evaluation"])
                
            except Exception as e:
                self.logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(30)
    
    async def _evaluate_metric_thresholds(self):
        """Evaluate all metrics against their thresholds"""
        for metric_name, metric in self.metrics.items():
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                await self._create_alert(
                    f"critical_{metric_name}",
                    AlertSeverity.CRITICAL,
                    f"Critical {metric.name}",
                    f"{metric.name} is {metric.value}{metric.unit}, exceeding critical threshold {metric.threshold_critical}{metric.unit}",
                    metric.labels.get("service", "system"),
                    metric.name,
                    metric.value,
                    metric.threshold_critical
                )
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                await self._create_alert(
                    f"warning_{metric_name}",
                    AlertSeverity.WARNING,
                    f"Warning {metric.name}",
                    f"{metric.name} is {metric.value}{metric.unit}, exceeding warning threshold {metric.threshold_warning}{metric.unit}",
                    metric.labels.get("service", "system"),
                    metric.name,
                    metric.value,
                    metric.threshold_warning
                )
    
    async def _create_alert(self, alert_id: str, severity: AlertSeverity, title: str, 
                          description: str, service: str, metric: str, 
                          current_value: float, threshold: float):
        """Create or update an alert"""
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            # Alert already exists and is not resolved
            return
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            description=description,
            service=service,
            metric=metric,
            current_value=current_value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        self.alerts[alert_id] = alert
        self.logger.warning(f"ALERT [{severity.value.upper()}] {title}: {description}")
        
        # Send notifications
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        webhook_url = self.config["alerting"]["slack_webhook"]
        if not webhook_url:
            return
        
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffeb3b",
            AlertSeverity.CRITICAL: "#f44336",
            AlertSeverity.EMERGENCY: "#9c27b0"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, "#36a64f"),
                "title": f"🚨 {alert.title}",
                "text": alert.description,
                "fields": [
                    {"title": "Service", "value": alert.service, "short": True},
                    {"title": "Metric", "value": alert.metric, "short": True},
                    {"title": "Current Value", "value": str(alert.current_value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True},
                    {"title": "Timestamp", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": False}
                ],
                "footer": "Production Monitoring System"
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=payload)
    
    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        # Email implementation would go here
        self.logger.info(f"Would send email alert: {alert.title}")
    
    async def _send_pagerduty_alert(self, alert: Alert):
        """Send alert to PagerDuty"""
        # PagerDuty implementation would go here
        self.logger.info(f"Would send PagerDuty alert: {alert.title}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        now = datetime.now()
        
        # Service health summary
        healthy_services = sum(1 for hc in self.health_checks.values() if hc.status == ServiceStatus.HEALTHY)
        total_services = len(self.health_checks)
        
        # Active alerts by severity
        active_alerts = {alert for alert in self.alerts.values() if not alert.resolved}
        alerts_by_severity = {
            severity: len([a for a in active_alerts if a.severity == severity])
            for severity in AlertSeverity
        }
        
        # Recent metrics
        recent_metrics = {
            name: {
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
                "age_seconds": (now - metric.timestamp).total_seconds()
            }
            for name, metric in self.metrics.items()
            if (now - metric.timestamp).total_seconds() < 300  # Last 5 minutes
        }
        
        return {
            "timestamp": now.isoformat(),
            "overall_health": "healthy" if healthy_services == total_services and not alerts_by_severity[AlertSeverity.CRITICAL] else "degraded",
            "services": {
                "total": total_services,
                "healthy": healthy_services,
                "unhealthy": total_services - healthy_services,
                "details": {
                    name: {
                        "status": hc.status.value,
                        "last_check": hc.last_check.isoformat() if hc.last_check else None,
                        "response_time_ms": hc.response_time,
                        "error": hc.error_message
                    }
                    for name, hc in self.health_checks.items()
                }
            },
            "alerts": {
                "total_active": len(active_alerts),
                "by_severity": {severity.value: count for severity, count in alerts_by_severity.items()},
                "recent": [
                    {
                        "id": alert.id,
                        "severity": alert.severity.value,
                        "title": alert.title,
                        "service": alert.service,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:10]
                ]
            },
            "metrics": recent_metrics
        }
    
    async def resolve_alert(self, alert_id: str):
        """Manually resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_time = datetime.now()
            self.logger.info(f"Alert resolved: {alert_id}")
        else:
            self.logger.warning(f"Alert not found: {alert_id}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Monitoring System")
    parser.add_argument("--config", default="monitoring-config.yaml", help="Configuration file path")
    parser.add_argument("--status", action="store_true", help="Show status report and exit")
    parser.add_argument("--resolve-alert", help="Resolve specific alert by ID")
    args = parser.parse_args()
    
    monitor = ProductionMonitor(args.config)
    
    if args.status:
        # Show status report
        status = monitor.get_status_report()
        print(json.dumps(status, indent=2))
        return
    
    if args.resolve_alert:
        # Resolve specific alert
        await monitor.resolve_alert(args.resolve_alert)
        return
    
    # Start monitoring
    await monitor.start_monitoring()


if __name__ == "__main__":
    asyncio.run(main())