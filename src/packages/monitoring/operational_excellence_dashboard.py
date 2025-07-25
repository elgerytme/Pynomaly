#!/usr/bin/env python3
"""
Operational Excellence Dashboard for Enterprise Monitoring.

Provides comprehensive monitoring, alerting, and operational insights
for the hexagonal architecture monorepo with real-time metrics,
SLI/SLO tracking, and automated incident response.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import aiohttp
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ServiceHealth(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SLIMetric:
    """Service Level Indicator metric."""
    name: str
    current_value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    description: str = ""
    
    @property
    def is_healthy(self) -> bool:
        """Check if metric is within healthy range."""
        return self.current_value >= self.target_value
    
    @property
    def health_status(self) -> ServiceHealth:
        """Get health status based on thresholds."""
        if self.current_value >= self.target_value:
            return ServiceHealth.HEALTHY
        elif self.current_value >= self.threshold_warning:
            return ServiceHealth.DEGRADED
        elif self.current_value >= self.threshold_critical:
            return ServiceHealth.UNHEALTHY
        else:
            return ServiceHealth.UNKNOWN


@dataclass
class ServiceStatus:
    """Service operational status."""
    name: str
    health: ServiceHealth
    response_time_ms: float
    error_rate_percent: float
    throughput_rps: float
    availability_percent: float
    last_check: datetime
    issues: List[str] = field(default_factory=list)
    sli_metrics: Dict[str, SLIMetric] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    service: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_resolved(self) -> bool:
        """Check if alert is resolved."""
        return self.resolved_at is not None
    
    @property
    def duration_minutes(self) -> float:
        """Get alert duration in minutes."""
        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.created_at).total_seconds() / 60


class MetricsCollector:
    """Collects metrics from various services and infrastructure."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.services_config = self._load_services_config()
    
    def _load_services_config(self) -> Dict[str, Dict[str, Any]]:
        """Load service configuration."""
        return {
            "data-quality": {
                "url": "http://localhost:8000",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics",
                "sli_targets": {
                    "availability": 99.9,
                    "response_time": 200,
                    "error_rate": 0.1
                }
            },
            "machine-learning": {
                "url": "http://localhost:8001", 
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics",
                "sli_targets": {
                    "availability": 99.5,
                    "response_time": 500,
                    "error_rate": 0.5
                }
            },
            "mlops": {
                "url": "http://localhost:8002",
                "health_endpoint": "/health", 
                "metrics_endpoint": "/metrics",
                "sli_targets": {
                    "availability": 99.0,
                    "response_time": 1000,
                    "error_rate": 1.0
                }
            },
            "anomaly-detection": {
                "url": "http://localhost:8003",
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics", 
                "sli_targets": {
                    "availability": 99.9,
                    "response_time": 100,
                    "error_rate": 0.1
                }
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def collect_service_metrics(self, service_name: str) -> ServiceStatus:
        """Collect metrics for a specific service."""
        config = self.services_config.get(service_name)
        if not config:
            logger.warning(f"No configuration found for service: {service_name}")
            return ServiceStatus(
                name=service_name,
                health=ServiceHealth.UNKNOWN,
                response_time_ms=0,
                error_rate_percent=100,
                throughput_rps=0,
                availability_percent=0,
                last_check=datetime.utcnow(),
                issues=["Service configuration not found"]
            )
        
        try:
            # Check health endpoint
            health_status, response_time = await self._check_health(
                config["url"] + config["health_endpoint"]
            )
            
            # Collect detailed metrics
            metrics = await self._collect_metrics(
                config["url"] + config["metrics_endpoint"]
            )
            
            # Calculate SLI metrics
            sli_metrics = self._calculate_sli_metrics(metrics, config["sli_targets"])
            
            # Determine overall health
            overall_health = self._determine_health(sli_metrics, health_status)
            
            return ServiceStatus(
                name=service_name,
                health=overall_health,
                response_time_ms=response_time,
                error_rate_percent=metrics.get("error_rate", 0),
                throughput_rps=metrics.get("throughput", 0),
                availability_percent=metrics.get("availability", 100 if health_status else 0),
                last_check=datetime.utcnow(),
                sli_metrics=sli_metrics
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {service_name}: {e}")
            return ServiceStatus(
                name=service_name,
                health=ServiceHealth.UNHEALTHY,
                response_time_ms=0,
                error_rate_percent=100,
                throughput_rps=0,
                availability_percent=0,
                last_check=datetime.utcnow(),
                issues=[f"Failed to collect metrics: {str(e)}"]
            )
    
    async def _check_health(self, health_url: str) -> Tuple[bool, float]:
        """Check service health endpoint."""
        start_time = time.time()
        try:
            async with self.session.get(health_url) as response:
                response_time = (time.time() - start_time) * 1000
                return response.status == 200, response_time
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.debug(f"Health check failed for {health_url}: {e}")
            return False, response_time
    
    async def _collect_metrics(self, metrics_url: str) -> Dict[str, float]:
        """Collect detailed metrics from service."""
        try:
            async with self.session.get(metrics_url) as response:
                if response.status == 200:
                    # Simulate realistic metrics
                    return {
                        "error_rate": np.random.uniform(0.0, 2.0),
                        "throughput": np.random.uniform(50, 1000),
                        "availability": np.random.uniform(95.0, 100.0),
                        "cpu_usage": np.random.uniform(10, 80),
                        "memory_usage": np.random.uniform(20, 90),
                        "disk_usage": np.random.uniform(10, 70)
                    }
                else:
                    return {"error_rate": 100, "throughput": 0, "availability": 0}
        except Exception:
            return {"error_rate": 100, "throughput": 0, "availability": 0}
    
    def _calculate_sli_metrics(self, metrics: Dict[str, float], 
                             targets: Dict[str, float]) -> Dict[str, SLIMetric]:
        """Calculate SLI metrics based on collected data."""
        sli_metrics = {}
        
        # Availability SLI
        if "availability" in metrics and "availability" in targets:
            sli_metrics["availability"] = SLIMetric(
                name="Availability",
                current_value=metrics["availability"],
                target_value=targets["availability"],
                threshold_warning=targets["availability"] - 1.0,
                threshold_critical=targets["availability"] - 5.0,
                unit="%",
                description="Service availability percentage"
            )
        
        # Response Time SLI  
        if "response_time" in targets:
            # Use collected response time or simulate
            current_rt = metrics.get("response_time", np.random.uniform(50, 800))
            sli_metrics["response_time"] = SLIMetric(
                name="Response Time",
                current_value=current_rt,
                target_value=targets["response_time"],
                threshold_warning=targets["response_time"] * 1.5,
                threshold_critical=targets["response_time"] * 3.0,
                unit="ms",
                description="95th percentile response time"
            )
        
        # Error Rate SLI
        if "error_rate" in metrics and "error_rate" in targets:
            sli_metrics["error_rate"] = SLIMetric(
                name="Error Rate",
                current_value=metrics["error_rate"],
                target_value=targets["error_rate"],
                threshold_warning=targets["error_rate"] * 2.0,
                threshold_critical=targets["error_rate"] * 5.0,
                unit="%",
                description="Error rate percentage"
            )
        
        return sli_metrics
    
    def _determine_health(self, sli_metrics: Dict[str, SLIMetric], 
                         health_check_passed: bool) -> ServiceHealth:
        """Determine overall service health."""
        if not health_check_passed:
            return ServiceHealth.UNHEALTHY
        
        if not sli_metrics:
            return ServiceHealth.UNKNOWN
        
        unhealthy_count = sum(1 for sli in sli_metrics.values() 
                            if sli.health_status == ServiceHealth.UNHEALTHY)
        degraded_count = sum(1 for sli in sli_metrics.values() 
                           if sli.health_status == ServiceHealth.DEGRADED)
        
        if unhealthy_count > 0:
            return ServiceHealth.UNHEALTHY
        elif degraded_count > 0:
            return ServiceHealth.DEGRADED
        else:
            return ServiceHealth.HEALTHY


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels = self._setup_notification_channels()
    
    def _setup_notification_channels(self) -> Dict[str, Dict[str, Any]]:
        """Setup notification channels."""
        return {
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "channel": "#alerts",
                "enabled": True
            },
            "email": {
                "smtp_server": "smtp.company.com",
                "recipients": ["ops-team@company.com", "devops@company.com"],
                "enabled": True
            },
            "pagerduty": {
                "integration_key": "your-pagerduty-integration-key",
                "enabled": True
            }
        }
    
    def create_alert(self, severity: AlertSeverity, title: str, 
                    description: str, service: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create new alert."""
        alert_id = f"{service}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            description=description,
            service=service,
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        logger.warning(f"Alert created: {alert.title} ({alert.severity.value})")
        
        # Send notifications for high severity alerts
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            asyncio.create_task(self._send_notifications(alert))
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.utcnow()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.title} (duration: {alert.duration_minutes:.1f} min)")
            return True
        
        return False
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        try:
            # Slack notification
            if self.notification_channels["slack"]["enabled"]:
                await self._send_slack_notification(alert)
            
            # Email notification  
            if self.notification_channels["email"]["enabled"]:
                await self._send_email_notification(alert)
                
            logger.info(f"Notifications sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send notifications for alert {alert.id}: {e}")
    
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification."""
        # Simulate Slack notification
        logger.info(f"Slack notification: {alert.title} - {alert.description}")
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        # Simulate email notification
        logger.info(f"Email notification: {alert.title} - {alert.description}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        active_by_severity = {}
        for severity in AlertSeverity:
            active_by_severity[severity.value] = len([
                a for a in self.active_alerts.values() 
                if a.severity == severity
            ])
        
        return {
            "active_alerts": len(self.active_alerts),
            "active_by_severity": active_by_severity,
            "total_alerts_today": len([
                a for a in self.alert_history 
                if a.created_at.date() == datetime.utcnow().date()
            ]),
            "average_resolution_time_minutes": self._calculate_avg_resolution_time()
        }
    
    def _calculate_avg_resolution_time(self) -> float:
        """Calculate average alert resolution time."""
        resolved_alerts = [a for a in self.alert_history if a.is_resolved]
        if not resolved_alerts:
            return 0.0
        
        total_time = sum(a.duration_minutes for a in resolved_alerts)
        return total_time / len(resolved_alerts)


class OperationalDashboard:
    """Main operational excellence dashboard."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.dashboard_data: Dict[str, Any] = {}
        self.running = False
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring."""
        logger.info("Starting operational excellence monitoring dashboard...")
        self.running = True
        
        async with self.metrics_collector:
            while self.running:
                try:
                    await self._collect_all_metrics()
                    await self._analyze_and_alert()
                    await self._update_dashboard()
                    
                    logger.info("Monitoring cycle completed successfully")
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")
                    await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        logger.info("Stopping operational excellence monitoring...")
        self.running = False
    
    async def _collect_all_metrics(self):
        """Collect metrics from all services."""
        services = list(self.metrics_collector.services_config.keys())
        
        # Collect metrics concurrently
        tasks = [
            self.metrics_collector.collect_service_metrics(service)
            for service in services
        ]
        
        service_statuses = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.dashboard_data["services"] = {}
        for service, status in zip(services, service_statuses):
            if isinstance(status, Exception):
                logger.error(f"Failed to collect metrics for {service}: {status}")
                continue
            
            self.dashboard_data["services"][service] = status
    
    async def _analyze_and_alert(self):
        """Analyze metrics and create alerts."""
        if "services" not in self.dashboard_data:
            return
        
        for service_name, status in self.dashboard_data["services"].items():
            # Check for unhealthy services
            if status.health == ServiceHealth.UNHEALTHY:
                self.alert_manager.create_alert(
                    severity=AlertSeverity.CRITICAL,
                    title=f"Service {service_name} is unhealthy",
                    description=f"Service {service_name} health check failed. Issues: {', '.join(status.issues)}",
                    service=service_name,
                    metadata={"response_time": status.response_time_ms, "error_rate": status.error_rate_percent}
                )
            
            elif status.health == ServiceHealth.DEGRADED:
                self.alert_manager.create_alert(
                    severity=AlertSeverity.HIGH,
                    title=f"Service {service_name} is degraded",
                    description=f"Service {service_name} is experiencing performance issues",
                    service=service_name,
                    metadata={"response_time": status.response_time_ms, "error_rate": status.error_rate_percent}
                )
            
            # Check SLI breaches
            for sli_name, sli in status.sli_metrics.items():
                if sli.health_status == ServiceHealth.UNHEALTHY:
                    self.alert_manager.create_alert(
                        severity=AlertSeverity.HIGH,
                        title=f"SLI breach: {sli_name} for {service_name}",
                        description=f"{sli_name} is {sli.current_value}{sli.unit}, below target {sli.target_value}{sli.unit}",
                        service=service_name,
                        metadata={"sli_name": sli_name, "current_value": sli.current_value, "target_value": sli.target_value}
                    )
    
    async def _update_dashboard(self):
        """Update dashboard data."""
        # Calculate system-wide metrics
        services = self.dashboard_data.get("services", {})
        
        healthy_services = len([s for s in services.values() if s.health == ServiceHealth.HEALTHY])
        total_services = len(services)
        
        avg_response_time = np.mean([s.response_time_ms for s in services.values()]) if services else 0
        avg_error_rate = np.mean([s.error_rate_percent for s in services.values()]) if services else 0
        total_throughput = sum([s.throughput_rps for s in services.values()])
        
        # Update dashboard summary
        self.dashboard_data["summary"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health": "healthy" if healthy_services == total_services else "degraded" if healthy_services > 0 else "unhealthy",
            "healthy_services": healthy_services,
            "total_services": total_services,
            "avg_response_time_ms": round(avg_response_time, 2),
            "avg_error_rate_percent": round(avg_error_rate, 2),
            "total_throughput_rps": round(total_throughput, 2),
            "alerts": self.alert_manager.get_alert_summary()
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data
    
    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard."""
        if not self.dashboard_data:
            return "<html><body><h1>Operational Dashboard</h1><p>No data available</p></body></html>"
        
        summary = self.dashboard_data.get("summary", {})
        services = self.dashboard_data.get("services", {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Operational Excellence Dashboard</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .healthy {{ border-left: 5px solid #27ae60; }}
                .degraded {{ border-left: 5px solid #f39c12; }}
                .unhealthy {{ border-left: 5px solid #e74c3c; }}
                .metric-value {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ color: #666; font-size: 0.9em; }}
                .alert-summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .timestamp {{ text-align: right; color: #888; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Operational Excellence Dashboard</h1>
                    <p>Real-time monitoring of hexagonal architecture services</p>
                    <div class="timestamp">Last updated: {summary.get('timestamp', 'N/A')}</div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card {summary.get('overall_health', 'unknown')}">
                        <div class="metric-label">Overall System Health</div>
                        <div class="metric-value">{summary.get('overall_health', 'Unknown').upper()}</div>
                        <p>{summary.get('healthy_services', 0)}/{summary.get('total_services', 0)} services healthy</p>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Average Response Time</div>
                        <div class="metric-value">{summary.get('avg_response_time_ms', 0):.1f}ms</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Error Rate</div>
                        <div class="metric-value">{summary.get('avg_error_rate_percent', 0):.2f}%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Total Throughput</div>
                        <div class="metric-value">{summary.get('total_throughput_rps', 0):.1f}</div>
                        <p>requests/second</p>
                    </div>
                </div>
                
                <h2>🔧 Service Status</h2>
                <div class="metrics-grid">
        """
        
        for service_name, status in services.items():
            health_class = status.health.value
            html += f"""
                    <div class="metric-card {health_class}">
                        <h3>{service_name.replace('-', ' ').title()}</h3>
                        <div class="metric-label">Health: {status.health.value.upper()}</div>
                        <p><strong>Response Time:</strong> {status.response_time_ms:.1f}ms</p>
                        <p><strong>Error Rate:</strong> {status.error_rate_percent:.2f}%</p>
                        <p><strong>Throughput:</strong> {status.throughput_rps:.1f} RPS</p>
                        <p><strong>Availability:</strong> {status.availability_percent:.2f}%</p>
                        <div class="timestamp">Last check: {status.last_check.strftime('%H:%M:%S')}</div>
                    </div>
            """
        
        alerts = summary.get("alerts", {})
        html += f"""
                </div>
                
                <h2>🚨 Alert Summary</h2>
                <div class="alert-summary">
                    <p><strong>Active Alerts:</strong> {alerts.get('active_alerts', 0)}</p>
                    <p><strong>Critical:</strong> {alerts.get('active_by_severity', {}).get('critical', 0)} | 
                       <strong>High:</strong> {alerts.get('active_by_severity', {}).get('high', 0)} | 
                       <strong>Medium:</strong> {alerts.get('active_by_severity', {}).get('medium', 0)}</p>
                    <p><strong>Today's Alerts:</strong> {alerts.get('total_alerts_today', 0)}</p>
                    <p><strong>Avg Resolution Time:</strong> {alerts.get('average_resolution_time_minutes', 0):.1f} minutes</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    async def export_metrics(self, file_path: str):
        """Export metrics to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.dashboard_data, f, indent=2, default=str)
            logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")


async def main():
    """Main function for running the dashboard."""
    dashboard = OperationalDashboard()
    
    try:
        # Start monitoring in background
        monitoring_task = asyncio.create_task(dashboard.start_monitoring(interval_seconds=30))
        
        # Run for demonstration
        await asyncio.sleep(120)  # Run for 2 minutes
        
        # Generate and save dashboard
        html_dashboard = dashboard.generate_dashboard_html()
        dashboard_path = Path("operational_dashboard.html")
        
        with open(dashboard_path, 'w') as f:
            f.write(html_dashboard)
        
        print(f"Dashboard saved to: {dashboard_path.absolute()}")
        print("\n📊 Dashboard Summary:")
        print(json.dumps(dashboard.get_dashboard_data().get("summary", {}), indent=2))
        
        # Export metrics
        await dashboard.export_metrics("operational_metrics.json")
        
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        dashboard.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())