/**
 * Real-time Monitoring Dashboard Component
 *
 * Provides comprehensive real-time monitoring dashboard with WebSocket integration,
 * performance metrics, error tracking, user analytics, and system health monitoring.
 */

class MonitoringDashboard {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            autoRefresh: true,
            refreshInterval: 5000,
            enableWebSocket: true,
            apiEndpoint: '/api/v1/monitoring',
            wsEndpoint: '/api/v1/monitoring/ws',
            ...options
        };

        // WebSocket connection
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;

        // Data storage
        this.dashboardData = {};
        this.performanceHistory = [];
        this.errorHistory = [];
        this.userMetrics = {};

        // Chart instances
        this.charts = {};

        // Event handlers
        this.eventHandlers = {};

        this.init();
    }

    async init() {
        try {
            await this.createDashboardLayout();
            await this.initializeCharts();
            await this.loadInitialData();

            if (this.options.enableWebSocket) {
                this.connectWebSocket();
            }

            if (this.options.autoRefresh) {
                this.startAutoRefresh();
            }

            this.setupEventListeners();
            console.log('Monitoring dashboard initialized successfully');
        } catch (error) {
            console.error('Failed to initialize monitoring dashboard:', error);
            this.showError('Failed to initialize dashboard');
        }
    }

    createDashboardLayout() {
        this.container.innerHTML = `
            <div class="monitoring-dashboard">
                <!-- Header with system status -->
                <div class="dashboard-header">
                    <div class="system-status">
                        <div class="status-indicator" id="system-status-indicator">
                            <span class="status-dot"></span>
                            <span class="status-text">Initializing...</span>
                        </div>
                        <div class="health-score" id="health-score">
                            <span class="score-label">Health Score:</span>
                            <span class="score-value">--</span>
                        </div>
                        <div class="connection-status" id="connection-status">
                            <span class="connection-indicator"></span>
                            <span class="connection-text">Connecting...</span>
                        </div>
                    </div>
                    <div class="dashboard-controls">
                        <button id="refresh-btn" class="btn btn-primary">
                            <i class="icon-refresh"></i> Refresh
                        </button>
                        <select id="time-range" class="form-select">
                            <option value="1">Last Hour</option>
                            <option value="6">Last 6 Hours</option>
                            <option value="24" selected>Last 24 Hours</option>
                            <option value="168">Last Week</option>
                        </select>
                    </div>
                </div>

                <!-- Key Metrics Cards -->
                <div class="metrics-grid">
                    <div class="metric-card" id="active-users-card">
                        <div class="metric-header">
                            <h3>Active Users</h3>
                            <i class="icon-users"></i>
                        </div>
                        <div class="metric-value" id="active-users-value">0</div>
                        <div class="metric-change" id="active-users-change">--</div>
                    </div>

                    <div class="metric-card" id="response-time-card">
                        <div class="metric-header">
                            <h3>Avg Response Time</h3>
                            <i class="icon-clock"></i>
                        </div>
                        <div class="metric-value" id="response-time-value">0ms</div>
                        <div class="metric-change" id="response-time-change">--</div>
                    </div>

                    <div class="metric-card" id="error-rate-card">
                        <div class="metric-header">
                            <h3>Error Rate</h3>
                            <i class="icon-alert"></i>
                        </div>
                        <div class="metric-value" id="error-rate-value">0%</div>
                        <div class="metric-change" id="error-rate-change">--</div>
                    </div>

                    <div class="metric-card" id="requests-per-second-card">
                        <div class="metric-header">
                            <h3>Requests/Second</h3>
                            <i class="icon-activity"></i>
                        </div>
                        <div class="metric-value" id="requests-per-second-value">0</div>
                        <div class="metric-change" id="requests-per-second-change">--</div>
                    </div>
                </div>

                <!-- Charts Section -->
                <div class="charts-section">
                    <div class="chart-row">
                        <div class="chart-container">
                            <h3>Performance Metrics</h3>
                            <canvas id="performance-chart"></canvas>
                        </div>
                        <div class="chart-container">
                            <h3>Error Trends</h3>
                            <canvas id="error-chart"></canvas>
                        </div>
                    </div>

                    <div class="chart-row">
                        <div class="chart-container">
                            <h3>System Resources</h3>
                            <canvas id="resources-chart"></canvas>
                        </div>
                        <div class="chart-container">
                            <h3>User Activity</h3>
                            <canvas id="user-activity-chart"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Alerts and Errors Section -->
                <div class="alerts-section">
                    <div class="section-header">
                        <h3>Active Alerts</h3>
                        <span class="alert-count" id="alert-count">0</span>
                    </div>
                    <div class="alerts-list" id="alerts-list">
                        <div class="no-alerts">No active alerts</div>
                    </div>
                </div>

                <!-- Recent Errors Section -->
                <div class="errors-section">
                    <div class="section-header">
                        <h3>Recent Errors</h3>
                        <button id="view-all-errors" class="btn btn-secondary">View All</button>
                    </div>
                    <div class="errors-list" id="errors-list">
                        <div class="no-errors">No recent errors</div>
                    </div>
                </div>

                <!-- Web Vitals Section -->
                <div class="web-vitals-section">
                    <h3>Core Web Vitals</h3>
                    <div class="vitals-grid">
                        <div class="vital-metric" id="fcp-metric">
                            <div class="vital-label">First Contentful Paint</div>
                            <div class="vital-value">--</div>
                            <div class="vital-rating">--</div>
                        </div>
                        <div class="vital-metric" id="lcp-metric">
                            <div class="vital-label">Largest Contentful Paint</div>
                            <div class="vital-value">--</div>
                            <div class="vital-rating">--</div>
                        </div>
                        <div class="vital-metric" id="fid-metric">
                            <div class="vital-label">First Input Delay</div>
                            <div class="vital-value">--</div>
                            <div class="vital-rating">--</div>
                        </div>
                        <div class="vital-metric" id="cls-metric">
                            <div class="vital-label">Cumulative Layout Shift</div>
                            <div class="vital-value">--</div>
                            <div class="vital-rating">--</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    async initializeCharts() {
        // Performance Chart
        const perfCtx = document.getElementById('performance-chart').getContext('2d');
        this.charts.performance = new Chart(perfCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Response Time (ms)',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Requests/sec',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });

        // Error Chart
        const errorCtx = document.getElementById('error-chart').getContext('2d');
        this.charts.errors = new Chart(errorCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Error Count',
                    data: [],
                    backgroundColor: '#ef4444',
                    borderColor: '#dc2626',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        // Resources Chart
        const resourcesCtx = document.getElementById('resources-chart').getContext('2d');
        this.charts.resources = new Chart(resourcesCtx, {
            type: 'doughnut',
            data: {
                labels: ['CPU', 'Memory', 'Disk'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#f59e0b', '#8b5cf6', '#06b6d4'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        // User Activity Chart
        const userCtx = document.getElementById('user-activity-chart').getContext('2d');
        this.charts.userActivity = new Chart(userCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Active Users',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    async loadInitialData() {
        try {
            const response = await fetch(`${this.options.apiEndpoint}/dashboard/metrics`);
            if (!response.ok) throw new Error('Failed to load dashboard data');

            this.dashboardData = await response.json();
            this.updateDashboard(this.dashboardData);
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}${this.options.wsEndpoint}`;

            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');

                // Subscribe to dashboard metrics
                this.websocket.send(JSON.stringify({
                    type: 'subscribe',
                    topics: ['dashboard_metrics', 'performance_snapshot', 'user_event', 'error_tracking']
                }));
            };

            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            this.websocket.onclose = () => {
                console.log('WebSocket connection closed');
                this.updateConnectionStatus('disconnected');
                this.attemptReconnect();
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('error');
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus('error');
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

            console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);

            setTimeout(() => {
                this.connectWebSocket();
            }, delay);
        } else {
            console.error('Max reconnection attempts reached');
            this.updateConnectionStatus('failed');
        }
    }

    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'dashboard_data':
                this.updateDashboard(message.data);
                break;
            case 'performance_snapshot':
                this.updatePerformanceMetrics(message.data);
                break;
            case 'monitoring_event':
                this.handleMonitoringEvent(message.data);
                break;
            case 'topic_message':
                this.handleTopicMessage(message.topic, message.data);
                break;
            case 'server_ping':
                // Respond to server ping
                this.websocket.send(JSON.stringify({ type: 'ping' }));
                break;
            default:
                console.log('Unknown WebSocket message type:', message.type);
        }
    }

    handleTopicMessage(topic, data) {
        switch (topic) {
            case 'dashboard_metrics':
                this.updateDashboard(data);
                break;
            case 'performance_snapshot':
                this.updatePerformanceMetrics(data);
                break;
            case 'user_event':
                this.handleUserEvent(data);
                break;
            case 'error_tracking':
                this.handleErrorEvent(data);
                break;
        }
    }

    updateDashboard(data) {
        this.dashboardData = data;

        // Update system status
        this.updateSystemStatus(data.system_health);

        // Update key metrics
        this.updateKeyMetrics(data);

        // Update charts
        this.updateCharts(data);

        // Update alerts
        this.updateAlerts(data.alerts || []);

        // Update Web Vitals
        if (data.performance && data.performance.web_vitals) {
            this.updateWebVitals(data.performance.web_vitals);
        }
    }

    updateSystemStatus(systemHealth) {
        const statusIndicator = document.getElementById('system-status-indicator');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('.status-text');
        const healthScoreValue = document.querySelector('#health-score .score-value');

        const status = systemHealth?.status || 'unknown';
        const healthScore = systemHealth?.health_score || 0;

        // Update status indicator
        statusDot.className = `status-dot status-${status}`;
        statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);

        // Update health score
        healthScoreValue.textContent = `${Math.round(healthScore)}%`;
        healthScoreValue.className = `score-value score-${this.getHealthScoreClass(healthScore)}`;
    }

    getHealthScoreClass(score) {
        if (score >= 90) return 'excellent';
        if (score >= 75) return 'good';
        if (score >= 50) return 'warning';
        return 'critical';
    }

    updateKeyMetrics(data) {
        // Active Users
        const activeUsers = data.users?.active_users || 0;
        document.getElementById('active-users-value').textContent = activeUsers;

        // Response Time
        const responseTime = data.api?.avg_response_time || 0;
        document.getElementById('response-time-value').textContent = `${Math.round(responseTime)}ms`;

        // Error Rate
        const errorRate = data.api?.error_rate || 0;
        document.getElementById('error-rate-value').textContent = `${errorRate.toFixed(2)}%`;

        // Requests per Second
        const rps = data.api?.requests_per_second || 0;
        document.getElementById('requests-per-second-value').textContent = rps.toFixed(1);
    }

    updateCharts(data) {
        // Add current timestamp to history
        const timestamp = new Date().toLocaleTimeString();

        // Update performance chart
        if (this.charts.performance) {
            const perfData = this.charts.performance.data;
            perfData.labels.push(timestamp);
            perfData.datasets[0].data.push(data.api?.avg_response_time || 0);
            perfData.datasets[1].data.push(data.api?.requests_per_second || 0);

            // Keep only last 20 data points
            if (perfData.labels.length > 20) {
                perfData.labels.shift();
                perfData.datasets.forEach(dataset => dataset.data.shift());
            }

            this.charts.performance.update('none');
        }

        // Update resources chart
        if (this.charts.resources && data.performance?.resources) {
            const resources = data.performance.resources;
            this.charts.resources.data.datasets[0].data = [
                resources.cpu || 0,
                resources.memory || 0,
                resources.disk || 0
            ];
            this.charts.resources.update('none');
        }

        // Update user activity chart
        if (this.charts.userActivity) {
            const userData = this.charts.userActivity.data;
            userData.labels.push(timestamp);
            userData.datasets[0].data.push(data.users?.active_users || 0);

            // Keep only last 20 data points
            if (userData.labels.length > 20) {
                userData.labels.shift();
                userData.datasets[0].data.shift();
            }

            this.charts.userActivity.update('none');
        }
    }

    updateAlerts(alerts) {
        const alertsList = document.getElementById('alerts-list');
        const alertCount = document.getElementById('alert-count');

        alertCount.textContent = alerts.length;

        if (alerts.length === 0) {
            alertsList.innerHTML = '<div class="no-alerts">No active alerts</div>';
            return;
        }

        alertsList.innerHTML = alerts.map(alert => `
            <div class="alert-item alert-${alert.severity}">
                <div class="alert-header">
                    <span class="alert-severity">${alert.severity.toUpperCase()}</span>
                    <span class="alert-time">${new Date(alert.triggered_at).toLocaleTimeString()}</span>
                </div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-metric">
                    ${alert.metric_name}: ${alert.metric_value} (threshold: ${alert.threshold})
                </div>
            </div>
        `).join('');
    }

    updateWebVitals(webVitals) {
        const vitals = [
            { id: 'fcp', key: 'fcp', label: 'FCP', unit: 'ms' },
            { id: 'lcp', key: 'lcp', label: 'LCP', unit: 'ms' },
            { id: 'fid', key: 'fid', label: 'FID', unit: 'ms' },
            { id: 'cls', key: 'cls', label: 'CLS', unit: '' }
        ];

        vitals.forEach(vital => {
            const element = document.getElementById(`${vital.id}-metric`);
            const value = webVitals[vital.key] || 0;
            const rating = this.getWebVitalRating(vital.key, value);

            element.querySelector('.vital-value').textContent = `${value.toFixed(vital.key === 'cls' ? 3 : 0)}${vital.unit}`;
            element.querySelector('.vital-rating').textContent = rating;
            element.querySelector('.vital-rating').className = `vital-rating rating-${rating}`;
        });
    }

    getWebVitalRating(vital, value) {
        const thresholds = {
            fcp: { good: 1800, poor: 3000 },
            lcp: { good: 2500, poor: 4000 },
            fid: { good: 100, poor: 300 },
            cls: { good: 0.1, poor: 0.25 }
        };

        const threshold = thresholds[vital];
        if (!threshold) return 'unknown';

        if (value <= threshold.good) return 'good';
        if (value <= threshold.poor) return 'needs-improvement';
        return 'poor';
    }

    updateConnectionStatus(status) {
        const connectionStatus = document.getElementById('connection-status');
        const indicator = connectionStatus.querySelector('.connection-indicator');
        const text = connectionStatus.querySelector('.connection-text');

        indicator.className = `connection-indicator connection-${status}`;

        switch (status) {
            case 'connected':
                text.textContent = 'Live';
                break;
            case 'disconnected':
                text.textContent = 'Disconnected';
                break;
            case 'error':
                text.textContent = 'Connection Error';
                break;
            case 'failed':
                text.textContent = 'Connection Failed';
                break;
            default:
                text.textContent = 'Connecting...';
        }
    }

    setupEventListeners() {
        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadInitialData();
        });

        // Time range selector
        document.getElementById('time-range').addEventListener('change', (e) => {
            this.changeTimeRange(e.target.value);
        });

        // View all errors button
        document.getElementById('view-all-errors').addEventListener('click', () => {
            this.showErrorsModal();
        });
    }

    startAutoRefresh() {
        setInterval(() => {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                this.loadInitialData();
            }
        }, this.options.refreshInterval);
    }

    async changeTimeRange(hours) {
        try {
            const response = await fetch(`${this.options.apiEndpoint}/dashboard/performance?time_window_hours=${hours}`);
            const data = await response.json();

            // Update charts with historical data
            this.updateChartsWithHistoricalData(data);
        } catch (error) {
            console.error('Failed to load historical data:', error);
        }
    }

    showError(message) {
        // Create a simple error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.textContent = message;

        document.body.appendChild(errorDiv);

        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    showErrorsModal() {
        // This would open a modal with detailed error information
        console.log('Opening errors modal...');
    }

    destroy() {
        if (this.websocket) {
            this.websocket.close();
        }

        // Destroy charts
        Object.values(this.charts).forEach(chart => {
            if (chart.destroy) chart.destroy();
        });

        // Clear intervals
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }
}

// CSS styles for the monitoring dashboard
const dashboardStyles = `
<style>
.monitoring-dashboard {
    padding: 20px;
    background: #f8fafc;
    min-height: 100vh;
}

.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
    padding: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.system-status {
    display: flex;
    align-items: center;
    gap: 30px;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

.status-dot.status-healthy { background: #10b981; }
.status-dot.status-degraded { background: #f59e0b; }
.status-dot.status-unhealthy { background: #ef4444; }
.status-dot.status-unknown { background: #6b7280; }

.health-score {
    display: flex;
    align-items: center;
    gap: 8px;
}

.score-value.score-excellent { color: #10b981; }
.score-value.score-good { color: #3b82f6; }
.score-value.score-warning { color: #f59e0b; }
.score-value.score-critical { color: #ef4444; }

.connection-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
}

.connection-indicator.connection-connected { background: #10b981; }
.connection-indicator.connection-disconnected { background: #6b7280; }
.connection-indicator.connection-error { background: #ef4444; }
.connection-indicator.connection-failed { background: #ef4444; }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.metric-header h3 {
    margin: 0;
    font-size: 14px;
    color: #6b7280;
    text-transform: uppercase;
    font-weight: 600;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #1f2937;
    margin-bottom: 5px;
}

.metric-change {
    font-size: 12px;
    color: #6b7280;
}

.charts-section {
    margin-bottom: 30px;
}

.chart-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

.chart-container {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.chart-container h3 {
    margin: 0 0 20px 0;
    font-size: 16px;
    color: #1f2937;
}

.alerts-section, .errors-section {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.alert-count {
    background: #ef4444;
    color: white;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
}

.alert-item {
    padding: 15px;
    margin-bottom: 10px;
    border-radius: 6px;
    border-left: 4px solid;
}

.alert-item.alert-critical {
    background: #fef2f2;
    border-left-color: #ef4444;
}

.alert-item.alert-warning {
    background: #fffbeb;
    border-left-color: #f59e0b;
}

.alert-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
}

.alert-severity {
    font-weight: bold;
    font-size: 12px;
}

.alert-time {
    font-size: 12px;
    color: #6b7280;
}

.web-vitals-section {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.vitals-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.vital-metric {
    text-align: center;
    padding: 15px;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
}

.vital-label {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 8px;
}

.vital-value {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 4px;
}

.vital-rating {
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 12px;
    text-transform: uppercase;
    font-weight: bold;
}

.vital-rating.rating-good {
    background: #d1fae5;
    color: #065f46;
}

.vital-rating.rating-needs-improvement {
    background: #fef3c7;
    color: #92400e;
}

.vital-rating.rating-poor {
    background: #fee2e2;
    color: #991b1b;
}

.no-alerts, .no-errors {
    text-align: center;
    color: #6b7280;
    padding: 20px;
}

.error-notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: #ef4444;
    color: white;
    padding: 15px 20px;
    border-radius: 6px;
    z-index: 1000;
    animation: slideIn 0.3s ease-out;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes slideIn {
    from { transform: translateX(100%); }
    to { transform: translateX(0); }
}

@media (max-width: 768px) {
    .chart-row {
        grid-template-columns: 1fr;
    }

    .metrics-grid {
        grid-template-columns: 1fr;
    }

    .vitals-grid {
        grid-template-columns: 1fr 1fr;
    }
}
</style>
`;

// Add styles to the document
if (!document.querySelector('#monitoring-dashboard-styles')) {
    const styleElement = document.createElement('style');
    styleElement.id = 'monitoring-dashboard-styles';
    styleElement.textContent = dashboardStyles.replace('<style>', '').replace('</style>', '');
    document.head.appendChild(styleElement);
}

// Export for use
window.MonitoringDashboard = MonitoringDashboard;
