#!/usr/bin/env python3
"""
Package Health Monitoring System

This script implements comprehensive package health monitoring including:
- Runtime metrics collection
- Usage analytics
- Performance monitoring dashboards
- Health check endpoints
- Monitoring alerting system
"""

import json
import logging
import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import psutil
import requests
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from prometheus_client.core import CollectorRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    'pynomaly_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)
REQUEST_DURATION = Histogram(
    'pynomaly_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint'],
    registry=REGISTRY
)
ACTIVE_CONNECTIONS = Gauge(
    'pynomaly_active_connections',
    'Active connections',
    registry=REGISTRY
)
SYSTEM_CPU = Gauge(
    'pynomaly_system_cpu_percent',
    'System CPU usage',
    registry=REGISTRY
)
SYSTEM_MEMORY = Gauge(
    'pynomaly_system_memory_percent',
    'System memory usage',
    registry=REGISTRY
)
SYSTEM_DISK = Gauge(
    'pynomaly_system_disk_percent',
    'System disk usage',
    registry=REGISTRY
)
DETECTOR_COUNT = Gauge(
    'pynomaly_active_detectors',
    'Number of active detectors',
    registry=REGISTRY
)
DETECTION_RATE = Counter(
    'pynomaly_detections_total',
    'Total detections performed',
    ['detector_type', 'status'],
    registry=REGISTRY
)

@dataclass
class HealthMetrics:
    """Health metrics data structure."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_connections: int
    request_count: int
    error_count: int
    response_time_avg: float
    active_detectors: int
    detections_per_minute: float

@dataclass
class ComponentStatus:
    """Component status data structure."""
    name: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class HealthMonitor:
    """Main health monitoring class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.metrics_buffer = deque(maxlen=1000)
        self.component_statuses: Dict[str, ComponentStatus] = {}
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        self.running = False
        self.monitor_thread = None
        
        # Initialize component monitors
        self._initialize_monitors()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'collection_interval': 30,  # seconds
            'retention_period': 24 * 60 * 60,  # 24 hours
            'alert_thresholds': {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'disk_percent': 90.0,
                'error_rate': 5.0,
                'response_time': 2.0
            },
            'endpoints': {
                'health': '/health',
                'metrics': '/metrics',
                'detailed_health': '/health/detailed'
            },
            'prometheus': {
                'port': 8001,
                'enabled': True
            }
        }
    
    def _initialize_monitors(self):
        """Initialize component monitors."""
        components = [
            'system',
            'database',
            'redis',
            'api',
            'detectors',
            'workers'
        ]
        
        for component in components:
            self.component_statuses[component] = ComponentStatus(
                name=component,
                status='unknown',
                last_check=datetime.now(),
                message='Initializing'
            )
    
    def start(self):
        """Start health monitoring."""
        if self.running:
            logger.warning("Health monitoring already running")
            return
        
        logger.info("Starting health monitoring...")
        self.running = True
        
        # Start Prometheus metrics server
        if self.config['prometheus']['enabled']:
            start_http_server(
                self.config['prometheus']['port'],
                registry=REGISTRY
            )
            logger.info(f"Prometheus metrics server started on port {self.config['prometheus']['port']}")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Health monitoring started successfully")
    
    def stop(self):
        """Stop health monitoring."""
        if not self.running:
            return
        
        logger.info("Stopping health monitoring...")
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_buffer.append(metrics)
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(metrics)
                
                # Check component health
                self._check_component_health()
                
                # Check alert conditions
                self._check_alerts(metrics)
                
                # Sleep until next collection
                time.sleep(self.config['collection_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect system and application metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connections
            connections = len(psutil.net_connections())
            
            # Application metrics (mock for now)
            request_count = self._get_request_count()
            error_count = self._get_error_count()
            response_time_avg = self._get_avg_response_time()
            active_detectors = self._get_active_detectors()
            detections_per_minute = self._get_detections_per_minute()
            
            return HealthMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                active_connections=connections,
                request_count=request_count,
                error_count=error_count,
                response_time_avg=response_time_avg,
                active_detectors=active_detectors,
                detections_per_minute=detections_per_minute
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return HealthMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                active_connections=0,
                request_count=0,
                error_count=0,
                response_time_avg=0.0,
                active_detectors=0,
                detections_per_minute=0.0
            )
    
    def _get_request_count(self) -> int:
        """Get total request count."""
        # In a real implementation, this would query the application
        return len(self.metrics_buffer) * 100
    
    def _get_error_count(self) -> int:
        """Get total error count."""
        # In a real implementation, this would query error logs
        return max(0, len(self.metrics_buffer) * 2 - 100)
    
    def _get_avg_response_time(self) -> float:
        """Get average response time."""
        # In a real implementation, this would calculate from request logs
        return 0.5 + (len(self.metrics_buffer) % 10) * 0.1
    
    def _get_active_detectors(self) -> int:
        """Get number of active detectors."""
        # In a real implementation, this would query the detector registry
        return 5 + (len(self.metrics_buffer) % 3)
    
    def _get_detections_per_minute(self) -> float:
        """Get detections per minute rate."""
        # In a real implementation, this would calculate from detection logs
        return 10.0 + (len(self.metrics_buffer) % 5) * 2.0
    
    def _update_prometheus_metrics(self, metrics: HealthMetrics):
        """Update Prometheus metrics."""
        SYSTEM_CPU.set(metrics.cpu_percent)
        SYSTEM_MEMORY.set(metrics.memory_percent)
        SYSTEM_DISK.set(metrics.disk_percent)
        ACTIVE_CONNECTIONS.set(metrics.active_connections)
        DETECTOR_COUNT.set(metrics.active_detectors)
    
    def _check_component_health(self):
        """Check health of individual components."""
        components_to_check = [
            ('system', self._check_system_health),
            ('database', self._check_database_health),
            ('redis', self._check_redis_health),
            ('api', self._check_api_health),
            ('detectors', self._check_detectors_health),
            ('workers', self._check_workers_health)
        ]
        
        for component_name, check_func in components_to_check:
            try:
                status = check_func()
                self.component_statuses[component_name] = status
            except Exception as e:
                logger.error(f"Error checking {component_name} health: {e}")
                self.component_statuses[component_name] = ComponentStatus(
                    name=component_name,
                    status='unhealthy',
                    last_check=datetime.now(),
                    message=f"Health check failed: {str(e)}"
                )
    
    def _check_system_health(self) -> ComponentStatus:
        """Check system health."""
        if not self.metrics_buffer:
            return ComponentStatus(
                name='system',
                status='unknown',
                last_check=datetime.now(),
                message='No metrics available'
            )
        
        latest_metrics = self.metrics_buffer[-1]
        
        # Determine status based on thresholds
        if (latest_metrics.cpu_percent > self.alert_thresholds['cpu_percent'] or
            latest_metrics.memory_percent > self.alert_thresholds['memory_percent'] or
            latest_metrics.disk_percent > self.alert_thresholds['disk_percent']):
            status = 'degraded'
            message = 'System resources under pressure'
        else:
            status = 'healthy'
            message = 'System resources normal'
        
        return ComponentStatus(
            name='system',
            status=status,
            last_check=datetime.now(),
            message=message,
            metrics={
                'cpu_percent': latest_metrics.cpu_percent,
                'memory_percent': latest_metrics.memory_percent,
                'disk_percent': latest_metrics.disk_percent
            }
        )
    
    def _check_database_health(self) -> ComponentStatus:
        """Check database health."""
        # Mock database health check
        return ComponentStatus(
            name='database',
            status='healthy',
            last_check=datetime.now(),
            message='Database connections normal',
            metrics={'connection_count': 10, 'query_time_avg': 0.05}
        )
    
    def _check_redis_health(self) -> ComponentStatus:
        """Check Redis health."""
        # Mock Redis health check
        return ComponentStatus(
            name='redis',
            status='healthy',
            last_check=datetime.now(),
            message='Redis cache operational',
            metrics={'memory_usage': 45.2, 'hit_rate': 0.95}
        )
    
    def _check_api_health(self) -> ComponentStatus:
        """Check API health."""
        if not self.metrics_buffer:
            return ComponentStatus(
                name='api',
                status='unknown',
                last_check=datetime.now(),
                message='No metrics available'
            )
        
        latest_metrics = self.metrics_buffer[-1]
        
        # Check error rate
        if latest_metrics.request_count > 0:
            error_rate = (latest_metrics.error_count / latest_metrics.request_count) * 100
        else:
            error_rate = 0.0
        
        if (error_rate > self.alert_thresholds['error_rate'] or
            latest_metrics.response_time_avg > self.alert_thresholds['response_time']):
            status = 'degraded'
            message = f'API performance degraded (error rate: {error_rate:.1f}%)'
        else:
            status = 'healthy'
            message = 'API performance normal'
        
        return ComponentStatus(
            name='api',
            status=status,
            last_check=datetime.now(),
            message=message,
            metrics={
                'error_rate': error_rate,
                'response_time_avg': latest_metrics.response_time_avg,
                'request_count': latest_metrics.request_count
            }
        )
    
    def _check_detectors_health(self) -> ComponentStatus:
        """Check detectors health."""
        if not self.metrics_buffer:
            return ComponentStatus(
                name='detectors',
                status='unknown',
                last_check=datetime.now(),
                message='No metrics available'
            )
        
        latest_metrics = self.metrics_buffer[-1]
        
        if latest_metrics.active_detectors == 0:
            status = 'unhealthy'
            message = 'No active detectors'
        elif latest_metrics.detections_per_minute < 1.0:
            status = 'degraded'
            message = 'Low detection activity'
        else:
            status = 'healthy'
            message = 'Detectors operating normally'
        
        return ComponentStatus(
            name='detectors',
            status=status,
            last_check=datetime.now(),
            message=message,
            metrics={
                'active_detectors': latest_metrics.active_detectors,
                'detections_per_minute': latest_metrics.detections_per_minute
            }
        )
    
    def _check_workers_health(self) -> ComponentStatus:
        """Check workers health."""
        # Mock worker health check
        return ComponentStatus(
            name='workers',
            status='healthy',
            last_check=datetime.now(),
            message='Background workers active',
            metrics={'active_workers': 3, 'queue_size': 5}
        )
    
    def _check_alerts(self, metrics: HealthMetrics):
        """Check for alert conditions."""
        alerts = []
        
        # System resource alerts
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_percent > self.alert_thresholds['disk_percent']:
            alerts.append(f"High disk usage: {metrics.disk_percent:.1f}%")
        
        # Application alerts
        if metrics.request_count > 0:
            error_rate = (metrics.error_count / metrics.request_count) * 100
            if error_rate > self.alert_thresholds['error_rate']:
                alerts.append(f"High error rate: {error_rate:.1f}%")
        
        if metrics.response_time_avg > self.alert_thresholds['response_time']:
            alerts.append(f"High response time: {metrics.response_time_avg:.2f}s")
        
        # Send alerts if any
        if alerts:
            self._send_alerts(alerts)
    
    def _send_alerts(self, alerts: List[str]):
        """Send alerts to configured channels."""
        alert_message = {
            'timestamp': datetime.now().isoformat(),
            'alerts': alerts,
            'severity': 'warning'
        }
        
        logger.warning(f"ALERT: {', '.join(alerts)}")
        
        # In a real implementation, this would send to Slack, email, etc.
        # For now, just log the alert
        print(f"🚨 ALERT: {json.dumps(alert_message, indent=2)}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        overall_status = 'healthy'
        
        # Determine overall status
        for component in self.component_statuses.values():
            if component.status == 'unhealthy':
                overall_status = 'unhealthy'
                break
            elif component.status == 'degraded':
                overall_status = 'degraded'
        
        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'components': {
                name: {
                    'status': component.status,
                    'last_check': component.last_check.isoformat(),
                    'message': component.message,
                    'metrics': component.metrics
                }
                for name, component in self.component_statuses.items()
            }
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.metrics_buffer:
            return {'message': 'No metrics available'}
        
        recent_metrics = list(self.metrics_buffer)[-10:]  # Last 10 metrics
        
        return {
            'current': asdict(recent_metrics[-1]) if recent_metrics else None,
            'average': {
                'cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                'memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                'response_time_avg': sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics)
            },
            'trends': {
                'cpu_trend': 'stable',  # Would calculate trend in real implementation
                'memory_trend': 'stable',
                'response_time_trend': 'stable'
            }
        }

class HealthDashboard:
    """Simple health dashboard."""
    
    def __init__(self, monitor: HealthMonitor):
        self.monitor = monitor
    
    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard."""
        health_status = self.monitor.get_health_status()
        metrics_summary = self.monitor.get_metrics_summary()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pynomaly Health Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .status-healthy {{ color: green; }}
                .status-degraded {{ color: orange; }}
                .status-unhealthy {{ color: red; }}
                .component {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 5px 10px; }}
                .refresh {{ position: fixed; top: 10px; right: 10px; }}
            </style>
            <script>
                function refreshPage() {{
                    location.reload();
                }}
                setInterval(refreshPage, 30000); // Refresh every 30 seconds
            </script>
        </head>
        <body>
            <h1>Pynomaly Health Dashboard</h1>
            <button class="refresh" onclick="refreshPage()">Refresh Now</button>
            
            <h2>Overall Status: <span class="status-{health_status['status']}">{health_status['status'].upper()}</span></h2>
            <p>Last updated: {health_status['timestamp']}</p>
            
            <h3>Component Status</h3>
        """
        
        for name, component in health_status['components'].items():
            html += f"""
            <div class="component">
                <h4>{name.title()}: <span class="status-{component['status']}">{component['status'].upper()}</span></h4>
                <p>{component['message']}</p>
                <p><small>Last checked: {component['last_check']}</small></p>
            </div>
            """
        
        if metrics_summary.get('current'):
            current = metrics_summary['current']
            html += f"""
            <h3>Current Metrics</h3>
            <div class="metric">CPU: {current['cpu_percent']:.1f}%</div>
            <div class="metric">Memory: {current['memory_percent']:.1f}%</div>
            <div class="metric">Disk: {current['disk_percent']:.1f}%</div>
            <div class="metric">Response Time: {current['response_time_avg']:.2f}s</div>
            <div class="metric">Active Detectors: {current['active_detectors']}</div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html

def main():
    """Main function for standalone execution."""
    print("Starting Pynomaly Health Monitoring System...")
    
    # Create health monitor
    monitor = HealthMonitor()
    
    try:
        # Start monitoring
        monitor.start()
        
        # Generate initial dashboard
        dashboard = HealthDashboard(monitor)
        dashboard_html = dashboard.generate_dashboard_html()
        
        # Save dashboard to file
        dashboard_path = Path('health_dashboard.html')
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        print(f"Health dashboard saved to: {dashboard_path}")
        print("Health monitoring is running...")
        print("Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(60)
            
            # Update dashboard every minute
            dashboard_html = dashboard.generate_dashboard_html()
            with open(dashboard_path, 'w') as f:
                f.write(dashboard_html)
            
            # Print status update
            health_status = monitor.get_health_status()
            print(f"Status: {health_status['status']} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
    except KeyboardInterrupt:
        print("\nStopping health monitoring...")
        monitor.stop()
        print("Health monitoring stopped.")

if __name__ == "__main__":
    main()