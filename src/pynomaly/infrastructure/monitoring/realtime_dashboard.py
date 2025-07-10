#!/usr/bin/env python3
"""
Real-time Monitoring and Alerting Dashboard for Pynomaly.

This module provides a comprehensive real-time monitoring dashboard with:
- Live metrics visualization
- Real-time alerts and notifications
- System health monitoring
- Performance metrics
- Interactive charts and graphs
- WebSocket support for live updates
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import uuid

from aiohttp import web, WSMsgType
from aiohttp_cors import CorsConfig, setup as cors_setup
import aiohttp_session
from aiohttp_session.cookie_storage import EncryptedCookieStorage
import asyncpg
import redis.asyncio as redis
import psutil
import numpy as np
from prometheus_client import CollectorRegistry, Histogram, Counter, Gauge, generate_latest
from prometheus_client.exposition import choose_encoder

from ..alerting.alert_manager import AlertManager
from ..alerting.metric_collector import MetricCollector
from ...shared.logging import get_logger

logger = get_logger(__name__)

class RealtimeDashboard:
    """Real-time monitoring dashboard with WebSocket support."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.websocket_connections: Set[web.WebSocketResponse] = set()
        self.alert_manager = AlertManager()
        self.metric_collector = MetricCollector()
        self.running = False
        
        # Metrics
        self.registry = CollectorRegistry()
        self.dashboard_requests = Counter(
            'dashboard_requests_total',
            'Total dashboard requests',
            ['method', 'endpoint'],
            registry=self.registry
        )
        self.websocket_connections_gauge = Gauge(
            'websocket_connections_current',
            'Current WebSocket connections',
            registry=self.registry
        )
        self.alert_processing_time = Histogram(
            'alert_processing_seconds',
            'Time spent processing alerts',
            registry=self.registry
        )
        
        # Configuration
        self.config = {
            'refresh_interval': 5,  # seconds
            'metrics_history_size': 100,
            'alert_history_size': 50,
            'chart_update_interval': 10,  # seconds
            'websocket_heartbeat': 30,  # seconds
        }
        
        # In-memory storage for real-time data
        self.metrics_history: List[Dict[str, Any]] = []
        self.alerts_history: List[Dict[str, Any]] = []
        self.system_status = {
            'status': 'healthy',
            'uptime': 0,
            'last_update': datetime.now().isoformat()
        }
        
        self._setup_routes()
        self._setup_middleware()
        
    def _setup_routes(self):
        """Setup HTTP routes for the dashboard."""
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/dashboard', self.dashboard)
        self.app.router.add_get('/ws', self.websocket_handler)
        
        # API routes
        self.app.router.add_get('/api/metrics', self.get_metrics)
        self.app.router.add_get('/api/metrics/history', self.get_metrics_history)
        self.app.router.add_get('/api/alerts', self.get_alerts)
        self.app.router.add_get('/api/alerts/history', self.get_alerts_history)
        self.app.router.add_post('/api/alerts/{alert_id}/acknowledge', self.acknowledge_alert)
        self.app.router.add_get('/api/status', self.get_system_status)
        self.app.router.add_get('/api/health', self.health_check)
        
        # Configuration routes
        self.app.router.add_get('/api/config', self.get_config)
        self.app.router.add_post('/api/config', self.update_config)
        
        # Export routes
        self.app.router.add_get('/api/export/metrics', self.export_metrics)
        self.app.router.add_get('/api/export/alerts', self.export_alerts)
        
        # Prometheus metrics endpoint
        self.app.router.add_get('/metrics', self.prometheus_metrics)
        
        # Static files
        self.app.router.add_static('/static/', path=Path(__file__).parent / 'static')
        
    def _setup_middleware(self):
        """Setup middleware for the application."""
        # CORS
        cors = cors_setup(self.app, defaults={
            "*": CorsConfig(
                allow_credentials=True,
                allow_headers="*",
                allow_methods="*",
            )
        })
        
        # Session middleware
        secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
        aiohttp_session.setup(self.app, EncryptedCookieStorage(secret_key.encode()))
        
        # Request logging middleware
        async def logging_middleware(request, handler):
            start_time = datetime.now()
            response = await handler(request)
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"{request.method} {request.path} - {response.status} - {duration:.3f}s"
            )
            
            self.dashboard_requests.labels(
                method=request.method,
                endpoint=request.path
            ).inc()
            
            return response
        
        self.app.middlewares.append(logging_middleware)
        
    async def index(self, request):
        """Redirect to dashboard."""
        return web.HTTPFound('/dashboard')
        
    async def dashboard(self, request):
        """Serve the main dashboard page."""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynomaly Real-time Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.28.0/index.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1419;
            color: #e6e6e6;
            overflow-x: hidden;
        }
        
        .header {
            background: #1a1f2e;
            padding: 1rem 2rem;
            border-bottom: 1px solid #2a2f3e;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            color: #00d4ff;
            font-size: 1.5rem;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff41;
            box-shadow: 0 0 10px #00ff41;
            animation: pulse 2s infinite;
        }
        
        .status-dot.warning {
            background: #ff9500;
            box-shadow: 0 0 10px #ff9500;
        }
        
        .status-dot.critical {
            background: #ff003d;
            box-shadow: 0 0 10px #ff003d;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .container {
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: #1a1f2e;
            border: 1px solid #2a2f3e;
            border-radius: 8px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: #00d4ff;
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
        }
        
        .card h3 {
            color: #00d4ff;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00ff41;
            margin-bottom: 0.5rem;
        }
        
        .metric-value.warning {
            color: #ff9500;
        }
        
        .metric-value.critical {
            color: #ff003d;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #888;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 1rem;
        }
        
        .alerts-section {
            background: #1a1f2e;
            border: 1px solid #2a2f3e;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .alert-item {
            display: flex;
            align-items: center;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 4px;
            border-left: 4px solid;
        }
        
        .alert-item.critical {
            background: rgba(255, 0, 61, 0.1);
            border-left-color: #ff003d;
        }
        
        .alert-item.warning {
            background: rgba(255, 149, 0, 0.1);
            border-left-color: #ff9500;
        }
        
        .alert-item.info {
            background: rgba(0, 212, 255, 0.1);
            border-left-color: #00d4ff;
        }
        
        .alert-content {
            flex: 1;
        }
        
        .alert-title {
            font-weight: bold;
            margin-bottom: 0.25rem;
        }
        
        .alert-description {
            font-size: 0.9rem;
            color: #ccc;
        }
        
        .alert-time {
            font-size: 0.8rem;
            color: #888;
        }
        
        .alert-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: #00d4ff;
            color: #0f1419;
        }
        
        .btn-primary:hover {
            background: #00b8e6;
        }
        
        .btn-secondary {
            background: #2a2f3e;
            color: #e6e6e6;
        }
        
        .btn-secondary:hover {
            background: #3a3f4e;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-size: 0.9rem;
            z-index: 1000;
        }
        
        .connection-status.connected {
            background: #00ff41;
            color: #0f1419;
        }
        
        .connection-status.disconnected {
            background: #ff003d;
            color: #e6e6e6;
        }
        
        .loading {
            text-align: center;
            padding: 2rem;
            color: #888;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: #1a1f2e;
            border: 1px solid #2a2f3e;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        
        .metric-card h4 {
            color: #00d4ff;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        
        .chart-section {
            background: #1a1f2e;
            border: 1px solid #2a2f3e;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .chart-tabs {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .chart-tab {
            padding: 0.5rem 1rem;
            border: 1px solid #2a2f3e;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .chart-tab.active {
            background: #00d4ff;
            color: #0f1419;
            border-color: #00d4ff;
        }
        
        .chart-content {
            display: none;
        }
        
        .chart-content.active {
            display: block;
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            color: #888;
            border-top: 1px solid #2a2f3e;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .header {
                padding: 1rem;
            }
            
            .header h1 {
                font-size: 1.2rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 Pynomaly Monitoring Dashboard</h1>
        <div class="status-indicator">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">System Healthy</span>
        </div>
    </div>
    
    <div class="connection-status disconnected" id="connectionStatus">
        Connecting...
    </div>
    
    <div class="container">
        <!-- Key Metrics -->
        <div class="metrics-grid" id="metricsGrid">
            <div class="metric-card">
                <h4>API Response Time</h4>
                <div class="metric-value" id="apiResponseTime">--</div>
                <div class="metric-label">ms</div>
            </div>
            <div class="metric-card">
                <h4>Request Rate</h4>
                <div class="metric-value" id="requestRate">--</div>
                <div class="metric-label">req/min</div>
            </div>
            <div class="metric-card">
                <h4>Error Rate</h4>
                <div class="metric-value" id="errorRate">--</div>
                <div class="metric-label">%</div>
            </div>
            <div class="metric-card">
                <h4>CPU Usage</h4>
                <div class="metric-value" id="cpuUsage">--</div>
                <div class="metric-label">%</div>
            </div>
            <div class="metric-card">
                <h4>Memory Usage</h4>
                <div class="metric-value" id="memoryUsage">--</div>
                <div class="metric-label">%</div>
            </div>
            <div class="metric-card">
                <h4>Active Alerts</h4>
                <div class="metric-value" id="activeAlerts">--</div>
                <div class="metric-label">alerts</div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="chart-section">
            <h3>📊 Performance Charts</h3>
            <div class="chart-tabs">
                <div class="chart-tab active" data-chart="performance">Performance</div>
                <div class="chart-tab" data-chart="resources">Resources</div>
                <div class="chart-tab" data-chart="alerts">Alerts</div>
            </div>
            <div class="chart-content active" id="performanceChart">
                <div class="chart-container">
                    <canvas id="performanceCanvas"></canvas>
                </div>
            </div>
            <div class="chart-content" id="resourcesChart">
                <div class="chart-container">
                    <canvas id="resourcesCanvas"></canvas>
                </div>
            </div>
            <div class="chart-content" id="alertsChart">
                <div class="chart-container">
                    <canvas id="alertsCanvas"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Alerts -->
        <div class="alerts-section">
            <h3>🚨 Active Alerts</h3>
            <div id="alertsList">
                <div class="loading">Loading alerts...</div>
            </div>
        </div>
        
        <!-- System Status -->
        <div class="dashboard-grid">
            <div class="card">
                <h3>📡 System Status</h3>
                <div id="systemStatus">
                    <div class="loading">Loading system status...</div>
                </div>
            </div>
            <div class="card">
                <h3>📈 Performance Metrics</h3>
                <div id="performanceMetrics">
                    <div class="loading">Loading performance metrics...</div>
                </div>
            </div>
            <div class="card">
                <h3>🔧 Configuration</h3>
                <div id="configStatus">
                    <div class="loading">Loading configuration...</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>&copy; 2024 Pynomaly. Real-time monitoring dashboard powered by WebSocket.</p>
    </div>
    
    <script>
        class RealtimeDashboard {
            constructor() {
                this.ws = null;
                this.charts = {};
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.reconnectInterval = 5000;
                this.heartbeatInterval = 30000;
                this.heartbeatTimer = null;
                
                this.init();
            }
            
            init() {
                this.setupWebSocket();
                this.setupCharts();
                this.setupEventListeners();
                this.startHeartbeat();
            }
            
            setupWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.reconnectAttempts = 0;
                    this.updateConnectionStatus(true);
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.updateConnectionStatus(false);
                    this.attemptReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus(false);
                };
            }
            
            attemptReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                    setTimeout(() => this.setupWebSocket(), this.reconnectInterval);
                } else {
                    console.error('Max reconnection attempts reached');
                }
            }
            
            updateConnectionStatus(connected) {
                const statusEl = document.getElementById('connectionStatus');
                if (connected) {
                    statusEl.className = 'connection-status connected';
                    statusEl.textContent = 'Connected';
                } else {
                    statusEl.className = 'connection-status disconnected';
                    statusEl.textContent = 'Disconnected';
                }
            }
            
            handleWebSocketMessage(data) {
                switch (data.type) {
                    case 'metrics':
                        this.updateMetrics(data.payload);
                        break;
                    case 'alerts':
                        this.updateAlerts(data.payload);
                        break;
                    case 'system_status':
                        this.updateSystemStatus(data.payload);
                        break;
                    case 'heartbeat':
                        this.handleHeartbeat();
                        break;
                    default:
                        console.log('Unknown message type:', data.type);
                }
            }
            
            updateMetrics(metrics) {
                // Update key metrics
                document.getElementById('apiResponseTime').textContent = 
                    metrics.api_response_time ? metrics.api_response_time.toFixed(1) : '--';
                document.getElementById('requestRate').textContent = 
                    metrics.request_rate ? metrics.request_rate.toFixed(0) : '--';
                document.getElementById('errorRate').textContent = 
                    metrics.error_rate ? metrics.error_rate.toFixed(1) : '--';
                document.getElementById('cpuUsage').textContent = 
                    metrics.cpu_usage ? metrics.cpu_usage.toFixed(1) : '--';
                document.getElementById('memoryUsage').textContent = 
                    metrics.memory_usage ? metrics.memory_usage.toFixed(1) : '--';
                document.getElementById('activeAlerts').textContent = 
                    metrics.active_alerts || '--';
                
                // Update metric card colors based on thresholds
                this.updateMetricColors(metrics);
                
                // Update charts
                this.updateCharts(metrics);
            }
            
            updateMetricColors(metrics) {
                const updateColor = (elementId, value, warningThreshold, criticalThreshold) => {
                    const element = document.getElementById(elementId);
                    element.className = 'metric-value';
                    
                    if (value >= criticalThreshold) {
                        element.className += ' critical';
                    } else if (value >= warningThreshold) {
                        element.className += ' warning';
                    }
                };
                
                updateColor('apiResponseTime', metrics.api_response_time, 1000, 3000);
                updateColor('errorRate', metrics.error_rate, 5, 10);
                updateColor('cpuUsage', metrics.cpu_usage, 80, 90);
                updateColor('memoryUsage', metrics.memory_usage, 80, 90);
            }
            
            updateAlerts(alerts) {
                const alertsList = document.getElementById('alertsList');
                
                if (!alerts || alerts.length === 0) {
                    alertsList.innerHTML = '<div class="loading">No active alerts</div>';
                    return;
                }
                
                alertsList.innerHTML = alerts.map(alert => `
                    <div class="alert-item ${alert.severity}">
                        <div class="alert-content">
                            <div class="alert-title">${alert.title}</div>
                            <div class="alert-description">${alert.description}</div>
                            <div class="alert-time">${new Date(alert.timestamp).toLocaleString()}</div>
                        </div>
                        <div class="alert-actions">
                            <button class="btn btn-primary" onclick="dashboard.acknowledgeAlert('${alert.id}')">
                                Acknowledge
                            </button>
                        </div>
                    </div>
                `).join('');
            }
            
            updateSystemStatus(status) {
                const statusDot = document.getElementById('statusDot');
                const statusText = document.getElementById('statusText');
                
                statusDot.className = 'status-dot';
                
                switch (status.status) {
                    case 'healthy':
                        statusDot.className += ' healthy';
                        statusText.textContent = 'System Healthy';
                        break;
                    case 'warning':
                        statusDot.className += ' warning';
                        statusText.textContent = 'System Warning';
                        break;
                    case 'critical':
                        statusDot.className += ' critical';
                        statusText.textContent = 'System Critical';
                        break;
                    default:
                        statusText.textContent = 'System Unknown';
                }
                
                document.getElementById('systemStatus').innerHTML = `
                    <div>Status: ${status.status}</div>
                    <div>Uptime: ${status.uptime}</div>
                    <div>Last Update: ${new Date(status.last_update).toLocaleString()}</div>
                `;
            }
            
            setupCharts() {
                // Performance Chart
                this.charts.performance = new Chart(document.getElementById('performanceCanvas'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Response Time (ms)',
                            data: [],
                            borderColor: '#00d4ff',
                            backgroundColor: 'rgba(0, 212, 255, 0.1)',
                            tension: 0.4
                        }, {
                            label: 'Error Rate (%)',
                            data: [],
                            borderColor: '#ff003d',
                            backgroundColor: 'rgba(255, 0, 61, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: { color: '#888' }
                            },
                            x: {
                                ticks: { color: '#888' }
                            }
                        },
                        plugins: {
                            legend: {
                                labels: { color: '#e6e6e6' }
                            }
                        }
                    }
                });
                
                // Resources Chart
                this.charts.resources = new Chart(document.getElementById('resourcesCanvas'), {
                    type: 'doughnut',
                    data: {
                        labels: ['CPU', 'Memory', 'Disk'],
                        datasets: [{
                            data: [0, 0, 0],
                            backgroundColor: ['#00d4ff', '#00ff41', '#ff9500']
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                labels: { color: '#e6e6e6' }
                            }
                        }
                    }
                });
                
                // Alerts Chart
                this.charts.alerts = new Chart(document.getElementById('alertsCanvas'), {
                    type: 'bar',
                    data: {
                        labels: ['Critical', 'Warning', 'Info'],
                        datasets: [{
                            label: 'Active Alerts',
                            data: [0, 0, 0],
                            backgroundColor: ['#ff003d', '#ff9500', '#00d4ff']
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: { color: '#888' }
                            },
                            x: {
                                ticks: { color: '#888' }
                            }
                        },
                        plugins: {
                            legend: {
                                labels: { color: '#e6e6e6' }
                            }
                        }
                    }
                });
            }
            
            updateCharts(metrics) {
                const now = new Date().toLocaleTimeString();
                
                // Update performance chart
                const perfChart = this.charts.performance;
                if (perfChart.data.labels.length > 20) {
                    perfChart.data.labels.shift();
                    perfChart.data.datasets[0].data.shift();
                    perfChart.data.datasets[1].data.shift();
                }
                
                perfChart.data.labels.push(now);
                perfChart.data.datasets[0].data.push(metrics.api_response_time || 0);
                perfChart.data.datasets[1].data.push(metrics.error_rate || 0);
                perfChart.update('none');
                
                // Update resources chart
                this.charts.resources.data.datasets[0].data = [
                    metrics.cpu_usage || 0,
                    metrics.memory_usage || 0,
                    metrics.disk_usage || 0
                ];
                this.charts.resources.update('none');
                
                // Update alerts chart
                this.charts.alerts.data.datasets[0].data = [
                    metrics.critical_alerts || 0,
                    metrics.warning_alerts || 0,
                    metrics.info_alerts || 0
                ];
                this.charts.alerts.update('none');
            }
            
            setupEventListeners() {
                // Chart tabs
                document.querySelectorAll('.chart-tab').forEach(tab => {
                    tab.addEventListener('click', (e) => {
                        const chartType = e.target.dataset.chart;
                        
                        // Update active tab
                        document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
                        e.target.classList.add('active');
                        
                        // Update active chart
                        document.querySelectorAll('.chart-content').forEach(c => c.classList.remove('active'));
                        document.getElementById(`${chartType}Chart`).classList.add('active');
                    });
                });
                
                // Keyboard shortcuts
                document.addEventListener('keydown', (e) => {
                    if (e.ctrlKey && e.key === 'r') {
                        e.preventDefault();
                        this.refreshData();
                    }
                });
            }
            
            startHeartbeat() {
                this.heartbeatTimer = setInterval(() => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({ type: 'heartbeat' }));
                    }
                }, this.heartbeatInterval);
            }
            
            handleHeartbeat() {
                console.log('Heartbeat received');
            }
            
            acknowledgeAlert(alertId) {
                fetch(`/api/alerts/${alertId}/acknowledge`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                }).then(response => {
                    if (response.ok) {
                        console.log('Alert acknowledged');
                    } else {
                        console.error('Failed to acknowledge alert');
                    }
                });
            }
            
            refreshData() {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'refresh' }));
                }
            }
        }
        
        // Initialize dashboard
        const dashboard = new RealtimeDashboard();
    </script>
</body>
</html>
        """
        
        return web.Response(text=html_content, content_type='text/html')
        
    async def websocket_handler(self, request):
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        self.websocket_connections_gauge.set(len(self.websocket_connections))
        
        logger.info(f"WebSocket client connected. Total connections: {len(self.websocket_connections)}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON received: {msg.data}")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.websocket_connections.discard(ws)
            self.websocket_connections_gauge.set(len(self.websocket_connections))
            logger.info(f"WebSocket client disconnected. Total connections: {len(self.websocket_connections)}")
        
        return ws
        
    async def _handle_websocket_message(self, ws: web.WebSocketResponse, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        msg_type = data.get('type')
        
        if msg_type == 'heartbeat':
            await ws.send_str(json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()}))
        elif msg_type == 'refresh':
            # Send current data
            await self._send_current_data(ws)
        else:
            logger.warning(f"Unknown WebSocket message type: {msg_type}")
            
    async def _send_current_data(self, ws: web.WebSocketResponse):
        """Send current data to WebSocket client."""
        try:
            # Get current metrics
            metrics = await self.get_current_metrics()
            await ws.send_str(json.dumps({'type': 'metrics', 'payload': metrics}))
            
            # Get current alerts
            alerts = await self.get_current_alerts()
            await ws.send_str(json.dumps({'type': 'alerts', 'payload': alerts}))
            
            # Get system status
            status = await self.get_current_system_status()
            await ws.send_str(json.dumps({'type': 'system_status', 'payload': status}))
            
        except Exception as e:
            logger.error(f"Error sending current data: {e}")
            
    async def broadcast_update(self, msg_type: str, payload: Any):
        """Broadcast updates to all connected WebSocket clients."""
        if not self.websocket_connections:
            return
            
        message = json.dumps({'type': msg_type, 'payload': payload})
        
        # Send to all connected clients
        disconnected_clients = set()
        for ws in self.websocket_connections:
            try:
                await ws.send_str(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                disconnected_clients.add(ws)
        
        # Remove disconnected clients
        for ws in disconnected_clients:
            self.websocket_connections.discard(ws)
            
        self.websocket_connections_gauge.set(len(self.websocket_connections))
        
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get application metrics from metric collector
            app_metrics = await self.metric_collector.collect_metrics()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': (disk.used / disk.total) * 100,
                'api_response_time': app_metrics.get('api_response_time', 0),
                'request_rate': app_metrics.get('request_rate', 0),
                'error_rate': app_metrics.get('error_rate', 0),
                'active_alerts': len(await self.get_current_alerts()),
                'critical_alerts': len([a for a in await self.get_current_alerts() if a.get('severity') == 'critical']),
                'warning_alerts': len([a for a in await self.get_current_alerts() if a.get('severity') == 'warning']),
                'info_alerts': len([a for a in await self.get_current_alerts() if a.get('severity') == 'info']),
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {}
            
    async def get_current_alerts(self) -> List[Dict[str, Any]]:
        """Get current active alerts."""
        try:
            alerts = await self.alert_manager.get_active_alerts()
            return [
                {
                    'id': alert.id,
                    'title': alert.title,
                    'description': alert.description,
                    'severity': alert.severity,
                    'timestamp': alert.timestamp.isoformat(),
                    'status': alert.status,
                }
                for alert in alerts
            ]
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
            
    async def get_current_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            alerts = await self.get_current_alerts()
            critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
            warning_alerts = [a for a in alerts if a.get('severity') == 'warning']
            
            if critical_alerts:
                status = 'critical'
            elif warning_alerts:
                status = 'warning'
            else:
                status = 'healthy'
            
            return {
                'status': status,
                'uptime': self._get_uptime(),
                'last_update': datetime.now().isoformat(),
                'total_alerts': len(alerts),
                'critical_alerts': len(critical_alerts),
                'warning_alerts': len(warning_alerts),
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'status': 'unknown',
                'uptime': 'unknown',
                'last_update': datetime.now().isoformat(),
            }
            
    def _get_uptime(self) -> str:
        """Get system uptime."""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = datetime.now().timestamp() - boot_time
            uptime_delta = timedelta(seconds=uptime_seconds)
            
            days = uptime_delta.days
            hours, remainder = divmod(uptime_delta.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            
            return f"{days}d {hours}h {minutes}m"
        except Exception:
            return "unknown"
            
    # HTTP API endpoints
    async def get_metrics(self, request):
        """Get current metrics via HTTP."""
        metrics = await self.get_current_metrics()
        return web.json_response(metrics)
        
    async def get_metrics_history(self, request):
        """Get metrics history."""
        return web.json_response(self.metrics_history)
        
    async def get_alerts(self, request):
        """Get current alerts via HTTP."""
        alerts = await self.get_current_alerts()
        return web.json_response(alerts)
        
    async def get_alerts_history(self, request):
        """Get alerts history."""
        return web.json_response(self.alerts_history)
        
    async def acknowledge_alert(self, request):
        """Acknowledge an alert."""
        alert_id = request.match_info['alert_id']
        
        try:
            await self.alert_manager.acknowledge_alert(alert_id)
            return web.json_response({'status': 'success'})
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def get_system_status(self, request):
        """Get system status via HTTP."""
        status = await self.get_current_system_status()
        return web.json_response(status)
        
    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'uptime': self._get_uptime(),
        })
        
    async def get_config(self, request):
        """Get dashboard configuration."""
        return web.json_response(self.config)
        
    async def update_config(self, request):
        """Update dashboard configuration."""
        try:
            data = await request.json()
            self.config.update(data)
            return web.json_response({'status': 'success'})
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
            
    async def export_metrics(self, request):
        """Export metrics data."""
        export_format = request.query.get('format', 'json')
        
        if export_format == 'csv':
            # CSV export logic
            pass
        else:
            metrics = await self.get_current_metrics()
            return web.json_response(metrics)
            
    async def export_alerts(self, request):
        """Export alerts data."""
        alerts = await self.get_current_alerts()
        return web.json_response(alerts)
        
    async def prometheus_metrics(self, request):
        """Prometheus metrics endpoint."""
        encoder, content_type = choose_encoder(request.headers.get('Accept'))
        output = encoder(self.registry)
        return web.Response(body=output, content_type=content_type)
        
    async def start_background_tasks(self):
        """Start background tasks for metrics collection and broadcasting."""
        self.running = True
        
        # Start metrics collection task
        asyncio.create_task(self._metrics_collection_task())
        
        # Start broadcasting task
        asyncio.create_task(self._broadcasting_task())
        
    async def _metrics_collection_task(self):
        """Background task for collecting metrics."""
        while self.running:
            try:
                metrics = await self.get_current_metrics()
                
                # Store in history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.config['metrics_history_size']:
                    self.metrics_history.pop(0)
                
                # Update system status
                self.system_status = await self.get_current_system_status()
                
            except Exception as e:
                logger.error(f"Error in metrics collection task: {e}")
                
            await asyncio.sleep(self.config['refresh_interval'])
            
    async def _broadcasting_task(self):
        """Background task for broadcasting updates."""
        while self.running:
            try:
                # Broadcast metrics
                metrics = await self.get_current_metrics()
                await self.broadcast_update('metrics', metrics)
                
                # Broadcast alerts
                alerts = await self.get_current_alerts()
                await self.broadcast_update('alerts', alerts)
                
                # Broadcast system status
                status = await self.get_current_system_status()
                await self.broadcast_update('system_status', status)
                
            except Exception as e:
                logger.error(f"Error in broadcasting task: {e}")
                
            await asyncio.sleep(self.config['refresh_interval'])
            
    async def run(self):
        """Start the dashboard server."""
        logger.info(f"Starting Real-time Dashboard on {self.host}:{self.port}")
        
        # Start background tasks
        await self.start_background_tasks()
        
        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Dashboard started at http://{self.host}:{self.port}")
        
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down dashboard...")
        finally:
            self.running = False
            await runner.cleanup()


if __name__ == "__main__":
    dashboard = RealtimeDashboard()
    asyncio.run(dashboard.run())