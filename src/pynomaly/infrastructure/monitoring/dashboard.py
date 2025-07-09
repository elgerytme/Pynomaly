#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for Pynomaly

This module provides a web-based dashboard for monitoring system metrics,
alerts, and application performance.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import aiohttp
from aiohttp import web
import aiohttp_jinja2
import jinja2
from aiohttp_session import setup
from aiohttp_session.cookie_storage import EncryptedCookieStorage
import base64
import secrets

from .alerts import AlertManager, Alert, AlertSeverity, AlertStatus


class DashboardAPI:
    """API endpoints for the monitoring dashboard."""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
    
    async def get_metrics(self, request: web.Request) -> web.Response:
        """Get current metrics."""
        try:
            metrics = self.alert_manager.metrics_collector.get_all_metrics()
            
            # Convert datetime objects to ISO format
            for key, value in metrics.items():
                if isinstance(value, datetime):
                    metrics[key] = value.isoformat()
            
            return web.json_response(metrics)
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_alerts(self, request: web.Request) -> web.Response:
        """Get current alerts."""
        try:
            active_alerts = self.alert_manager.get_active_alerts()
            alerts_data = [alert.to_dict() for alert in active_alerts]
            
            return web.json_response({"alerts": alerts_data})
        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_alert_history(self, request: web.Request) -> web.Response:
        """Get alert history."""
        try:
            limit = int(request.query.get('limit', 100))
            history = self.alert_manager.get_alert_history()
            
            # Sort by timestamp descending and limit
            history.sort(key=lambda x: x.timestamp, reverse=True)
            history = history[:limit]
            
            history_data = [alert.to_dict() for alert in history]
            
            return web.json_response({"history": history_data})
        except Exception as e:
            self.logger.error(f"Error getting alert history: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def acknowledge_alert(self, request: web.Request) -> web.Response:
        """Acknowledge an alert."""
        try:
            data = await request.json()
            alert_id = data.get('alert_id')
            acknowledged_by = data.get('acknowledged_by', 'unknown')
            
            if not alert_id:
                return web.json_response({"error": "alert_id is required"}, status=400)
            
            self.alert_manager.acknowledge_alert(alert_id, acknowledged_by)
            
            return web.json_response({"message": "Alert acknowledged"})
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_system_status(self, request: web.Request) -> web.Response:
        """Get overall system status."""
        try:
            metrics = self.alert_manager.metrics_collector.get_all_metrics()
            active_alerts = self.alert_manager.get_active_alerts()
            
            # Calculate status
            critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
            warning_alerts = [a for a in active_alerts if a.severity == AlertSeverity.WARNING]
            
            if critical_alerts:
                status = "critical"
            elif warning_alerts:
                status = "warning"
            else:
                status = "healthy"
            
            cpu_usage = metrics.get("system.cpu.percent", 0)
            memory_usage = metrics.get("system.memory.percent", 0)
            disk_usage = metrics.get("system.disk.percent", 0)
            
            return web.json_response({
                "status": status,
                "alerts": {
                    "critical": len(critical_alerts),
                    "warning": len(warning_alerts),
                    "total": len(active_alerts)
                },
                "metrics": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage
                },
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return web.json_response({"error": str(e)}, status=500)


class DashboardWebServer:
    """Web server for the monitoring dashboard."""
    
    def __init__(self, alert_manager: AlertManager, host: str = "0.0.0.0", port: int = 8080):
        self.alert_manager = alert_manager
        self.host = host
        self.port = port
        self.app = web.Application()
        self.logger = logging.getLogger(__name__)
        
        # Setup API
        self.api = DashboardAPI(alert_manager)
        
        # Setup templates
        self._setup_templates()
        
        # Setup routes
        self._setup_routes()
        
        # Setup session
        self._setup_session()
    
    def _setup_templates(self):
        """Setup Jinja2 templates."""
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        
        aiohttp_jinja2.setup(
            self.app,
            loader=jinja2.FileSystemLoader(str(template_dir))
        )
    
    def _setup_routes(self):
        """Setup web routes."""
        # Static files
        self.app.router.add_static('/static', 'static', name='static')
        
        # Web pages
        self.app.router.add_get('/', self.dashboard_page)
        self.app.router.add_get('/alerts', self.alerts_page)
        self.app.router.add_get('/metrics', self.metrics_page)
        self.app.router.add_get('/health', self.health_check)
        
        # API endpoints
        self.app.router.add_get('/api/metrics', self.api.get_metrics)
        self.app.router.add_get('/api/alerts', self.api.get_alerts)
        self.app.router.add_get('/api/alerts/history', self.api.get_alert_history)
        self.app.router.add_post('/api/alerts/acknowledge', self.api.acknowledge_alert)
        self.app.router.add_get('/api/status', self.api.get_system_status)
    
    def _setup_session(self):
        """Setup secure sessions."""
        secret_key = base64.urlsafe_b64encode(secrets.token_bytes(32))
        setup(self.app, EncryptedCookieStorage(secret_key))
    
    @aiohttp_jinja2.template('dashboard.html')
    async def dashboard_page(self, request: web.Request) -> Dict[str, Any]:
        """Dashboard page."""
        return {
            "title": "Pynomaly Monitoring Dashboard",
            "refresh_interval": 30
        }
    
    @aiohttp_jinja2.template('alerts.html')
    async def alerts_page(self, request: web.Request) -> Dict[str, Any]:
        """Alerts page."""
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "title": "Active Alerts",
            "alerts": active_alerts,
            "refresh_interval": 10
        }
    
    @aiohttp_jinja2.template('metrics.html')
    async def metrics_page(self, request: web.Request) -> Dict[str, Any]:
        """Metrics page."""
        metrics = self.alert_manager.metrics_collector.get_all_metrics()
        
        return {
            "title": "System Metrics",
            "metrics": metrics,
            "refresh_interval": 30
        }
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "pynomaly-monitoring-dashboard"
        })
    
    async def start(self):
        """Start the web server."""
        try:
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            
            self.logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
            
            # Keep server running
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                self.logger.info("Shutting down dashboard server...")
                await runner.cleanup()
                
        except Exception as e:
            self.logger.error(f"Error starting dashboard server: {e}")
            raise


def create_dashboard_templates():
    """Create HTML templates for the dashboard."""
    
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    # Base template
    base_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Pynomaly Monitoring{% endblock %}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
            color: #333;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 1.5rem;
        }
        .nav {
            margin-top: 1rem;
        }
        .nav a {
            color: #ecf0f1;
            text-decoration: none;
            margin-right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        .nav a:hover {
            background: #34495e;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        .alert {
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            border-left: 4px solid;
        }
        .alert-critical {
            background: #fdf2f2;
            border-color: #e53e3e;
            color: #742a2a;
        }
        .alert-warning {
            background: #fffbeb;
            border-color: #ed8936;
            color: #744210;
        }
        .alert-info {
            background: #ebf8ff;
            border-color: #3182ce;
            color: #2c5282;
        }
        .status-healthy {
            color: #38a169;
        }
        .status-warning {
            color: #ed8936;
        }
        .status-critical {
            color: #e53e3e;
        }
        .refresh-info {
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 2rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
        }
        .btn {
            background: #3182ce;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .btn:hover {
            background: #2c5282;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Pynomaly Monitoring</h1>
        <nav class="nav">
            <a href="/">Dashboard</a>
            <a href="/alerts">Alerts</a>
            <a href="/metrics">Metrics</a>
        </nav>
    </div>
    
    <div class="container">
        {% block content %}{% endblock %}
    </div>
    
    <script>
        // Auto-refresh functionality
        const refreshInterval = {{ refresh_interval or 30 }};
        
        if (refreshInterval > 0) {
            setTimeout(() => {
                location.reload();
            }, refreshInterval * 1000);
        }
        
        // Real-time updates via WebSocket (placeholder)
        function updateMetrics() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update status indicators
                    const statusElements = document.querySelectorAll('.status-indicator');
                    statusElements.forEach(el => {
                        el.className = `status-indicator status-${data.status}`;
                        el.textContent = data.status.toUpperCase();
                    });
                })
                .catch(error => console.error('Error updating metrics:', error));
        }
        
        // Update every 10 seconds
        setInterval(updateMetrics, 10000);
    </script>
</body>
</html>"""
    
    # Dashboard template
    dashboard_template = """{% extends "base.html" %}

{% block title %}{{ title }}{% endblock %}

{% block content %}
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-value status-indicator" id="system-status">LOADING</div>
        <div class="metric-label">System Status</div>
    </div>
    
    <div class="metric-card">
        <div class="metric-value" id="active-alerts">-</div>
        <div class="metric-label">Active Alerts</div>
    </div>
    
    <div class="metric-card">
        <div class="metric-value" id="cpu-usage">-</div>
        <div class="metric-label">CPU Usage (%)</div>
    </div>
    
    <div class="metric-card">
        <div class="metric-value" id="memory-usage">-</div>
        <div class="metric-label">Memory Usage (%)</div>
    </div>
    
    <div class="metric-card">
        <div class="metric-value" id="disk-usage">-</div>
        <div class="metric-label">Disk Usage (%)</div>
    </div>
</div>

<div class="card">
    <h2>Recent Alerts</h2>
    <div id="recent-alerts">
        Loading alerts...
    </div>
</div>

<div class="refresh-info">
    Page refreshes every {{ refresh_interval }} seconds
</div>

<script>
    // Load dashboard data
    async function loadDashboardData() {
        try {
            const [statusResponse, alertsResponse] = await Promise.all([
                fetch('/api/status'),
                fetch('/api/alerts')
            ]);
            
            const status = await statusResponse.json();
            const alerts = await alertsResponse.json();
            
            // Update status metrics
            document.getElementById('system-status').textContent = status.status.toUpperCase();
            document.getElementById('system-status').className = `metric-value status-${status.status}`;
            document.getElementById('active-alerts').textContent = status.alerts.total;
            document.getElementById('cpu-usage').textContent = status.metrics.cpu_usage.toFixed(1);
            document.getElementById('memory-usage').textContent = status.metrics.memory_usage.toFixed(1);
            document.getElementById('disk-usage').textContent = status.metrics.disk_usage.toFixed(1);
            
            // Update alerts
            const alertsContainer = document.getElementById('recent-alerts');
            if (alerts.alerts.length === 0) {
                alertsContainer.innerHTML = '<p>No active alerts</p>';
            } else {
                alertsContainer.innerHTML = alerts.alerts.slice(0, 5).map(alert => `
                    <div class="alert alert-${alert.severity}">
                        <strong>${alert.title}</strong><br>
                        ${alert.description}<br>
                        <small>${new Date(alert.timestamp).toLocaleString()}</small>
                    </div>
                `).join('');
            }
            
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }
    
    // Load data on page load
    loadDashboardData();
    
    // Update every 10 seconds
    setInterval(loadDashboardData, 10000);
</script>
{% endblock %}"""
    
    # Alerts template
    alerts_template = """{% extends "base.html" %}

{% block title %}Active Alerts{% endblock %}

{% block content %}
<div class="card">
    <h2>Active Alerts</h2>
    <div id="alerts-container">
        Loading alerts...
    </div>
</div>

<div class="card">
    <h2>Alert History</h2>
    <div id="history-container">
        Loading history...
    </div>
</div>

<div class="refresh-info">
    Page refreshes every {{ refresh_interval }} seconds
</div>

<script>
    async function loadAlerts() {
        try {
            const [alertsResponse, historyResponse] = await Promise.all([
                fetch('/api/alerts'),
                fetch('/api/alerts/history?limit=20')
            ]);
            
            const alerts = await alertsResponse.json();
            const history = await historyResponse.json();
            
            // Update active alerts
            const alertsContainer = document.getElementById('alerts-container');
            if (alerts.alerts.length === 0) {
                alertsContainer.innerHTML = '<p>No active alerts</p>';
            } else {
                alertsContainer.innerHTML = alerts.alerts.map(alert => `
                    <div class="alert alert-${alert.severity}">
                        <strong>${alert.title}</strong><br>
                        ${alert.description}<br>
                        <small>Since: ${new Date(alert.timestamp).toLocaleString()}</small>
                        ${alert.status === 'active' ? `<br><button class="btn" onclick="acknowledgeAlert('${alert.id}')">Acknowledge</button>` : ''}
                    </div>
                `).join('');
            }
            
            // Update history
            const historyContainer = document.getElementById('history-container');
            if (history.history.length === 0) {
                historyContainer.innerHTML = '<p>No alert history</p>';
            } else {
                historyContainer.innerHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th>Alert</th>
                                <th>Severity</th>
                                <th>Status</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${history.history.map(alert => `
                                <tr>
                                    <td>${alert.title}</td>
                                    <td><span class="status-${alert.severity}">${alert.severity.toUpperCase()}</span></td>
                                    <td>${alert.status}</td>
                                    <td>${new Date(alert.timestamp).toLocaleString()}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
            
        } catch (error) {
            console.error('Error loading alerts:', error);
        }
    }
    
    async function acknowledgeAlert(alertId) {
        try {
            await fetch('/api/alerts/acknowledge', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    alert_id: alertId,
                    acknowledged_by: 'dashboard_user'
                })
            });
            
            loadAlerts(); // Refresh alerts
        } catch (error) {
            console.error('Error acknowledging alert:', error);
        }
    }
    
    // Load alerts on page load
    loadAlerts();
    
    // Update every 10 seconds
    setInterval(loadAlerts, 10000);
</script>
{% endblock %}"""
    
    # Metrics template
    metrics_template = """{% extends "base.html" %}

{% block title %}System Metrics{% endblock %}

{% block content %}
<div class="card">
    <h2>System Metrics</h2>
    <div id="metrics-container">
        Loading metrics...
    </div>
</div>

<div class="refresh-info">
    Page refreshes every {{ refresh_interval }} seconds
</div>

<script>
    async function loadMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const metrics = await response.json();
            
            const container = document.getElementById('metrics-container');
            container.innerHTML = `
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">${metrics['system.cpu.percent']?.toFixed(1) || 'N/A'}</div>
                        <div class="metric-label">CPU Usage (%)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics['system.memory.percent']?.toFixed(1) || 'N/A'}</div>
                        <div class="metric-label">Memory Usage (%)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics['system.disk.percent']?.toFixed(1) || 'N/A'}</div>
                        <div class="metric-label">Disk Usage (%)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics['system.load.1min']?.toFixed(2) || 'N/A'}</div>
                        <div class="metric-label">Load Average (1m)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics['app.api.response_time']?.toFixed(0) || 'N/A'}ms</div>
                        <div class="metric-label">API Response Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(metrics['app.error.rate'] * 100)?.toFixed(2) || 'N/A'}%</div>
                        <div class="metric-label">Error Rate</div>
                    </div>
                </div>
                
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Last Updated</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(metrics).map(([key, value]) => {
                            if (key === 'timestamp') return '';
                            return `
                                <tr>
                                    <td>${key}</td>
                                    <td>${typeof value === 'number' ? value.toFixed(2) : value}</td>
                                    <td>${metrics.timestamp ? new Date(metrics.timestamp).toLocaleString() : 'N/A'}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            `;
            
        } catch (error) {
            console.error('Error loading metrics:', error);
        }
    }
    
    // Load metrics on page load
    loadMetrics();
    
    // Update every 30 seconds
    setInterval(loadMetrics, 30000);
</script>
{% endblock %}"""
    
    # Write templates to files
    with open(template_dir / "base.html", "w") as f:
        f.write(base_template)
    
    with open(template_dir / "dashboard.html", "w") as f:
        f.write(dashboard_template)
    
    with open(template_dir / "alerts.html", "w") as f:
        f.write(alerts_template)
    
    with open(template_dir / "metrics.html", "w") as f:
        f.write(metrics_template)


async def main():
    """Main function for testing."""
    from .alerts import AlertManager
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create templates
    create_dashboard_templates()
    
    # Create alert manager
    alert_manager = AlertManager()
    
    # Create dashboard server
    dashboard = DashboardWebServer(alert_manager)
    
    try:
        # Start alert manager
        alert_manager.start()
        
        # Start dashboard server
        await dashboard.start()
        
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        alert_manager.stop()
        print("Dashboard stopped.")


if __name__ == "__main__":
    asyncio.run(main())