/**
 * Performance Dashboard Utility
 * Real-time performance monitoring and visualization
 */

class PerformanceDashboard {
    constructor() {
        this.metrics = {
            pageLoads: [],
            apiCalls: [],
            coreWebVitals: {
                LCP: [],
                FID: [],
                CLS: [],
                FCP: [],
                TTFB: []
            },
            resources: [],
            errors: []
        };
        
        this.dashboardElement = null;
        this.chartInstances = {};
        this.updateInterval = 5000; // 5 seconds
        this.isVisible = false;
        this.init();
    }

    init() {
        this.createDashboard();
        this.setupEventListeners();
        this.startDataCollection();
    }

    createDashboard() {
        // Create dashboard container
        this.dashboardElement = document.createElement('div');
        this.dashboardElement.id = 'performance-dashboard';
        this.dashboardElement.className = 'performance-dashboard';
        this.dashboardElement.innerHTML = this.getDashboardHTML();
        
        // Add styles
        this.addStyles();
        
        // Initially hidden
        this.dashboardElement.style.display = 'none';
        document.body.appendChild(this.dashboardElement);
    }

    getDashboardHTML() {
        return `
            <div class="dashboard-header">
                <h3>Performance Dashboard</h3>
                <div class="dashboard-controls">
                    <button id="refresh-dashboard" class="btn btn-sm">Refresh</button>
                    <button id="export-data" class="btn btn-sm">Export</button>
                    <button id="close-dashboard" class="btn btn-sm">×</button>
                </div>
            </div>
            
            <div class="dashboard-content">
                <div class="metrics-grid">
                    <!-- Core Web Vitals -->
                    <div class="metric-card">
                        <h4>Core Web Vitals</h4>
                        <div class="vitals-grid">
                            <div class="vital-item">
                                <span class="vital-label">LCP</span>
                                <span class="vital-value" id="lcp-value">--</span>
                                <span class="vital-status" id="lcp-status"></span>
                            </div>
                            <div class="vital-item">
                                <span class="vital-label">FID</span>
                                <span class="vital-value" id="fid-value">--</span>
                                <span class="vital-status" id="fid-status"></span>
                            </div>
                            <div class="vital-item">
                                <span class="vital-label">CLS</span>
                                <span class="vital-value" id="cls-value">--</span>
                                <span class="vital-status" id="cls-status"></span>
                            </div>
                            <div class="vital-item">
                                <span class="vital-label">FCP</span>
                                <span class="vital-value" id="fcp-value">--</span>
                                <span class="vital-status" id="fcp-status"></span>
                            </div>
                        </div>
                        <canvas id="vitals-chart" width="400" height="200"></canvas>
                    </div>

                    <!-- Page Performance -->
                    <div class="metric-card">
                        <h4>Page Performance</h4>
                        <div class="performance-stats">
                            <div class="stat-item">
                                <span class="stat-label">Avg Load Time</span>
                                <span class="stat-value" id="avg-load-time">--</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Total Requests</span>
                                <span class="stat-value" id="total-requests">--</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Failed Requests</span>
                                <span class="stat-value" id="failed-requests">--</span>
                            </div>
                        </div>
                        <canvas id="performance-chart" width="400" height="200"></canvas>
                    </div>

                    <!-- API Performance -->
                    <div class="metric-card">
                        <h4>API Performance</h4>
                        <div class="api-stats">
                            <div class="stat-item">
                                <span class="stat-label">Avg Response Time</span>
                                <span class="stat-value" id="avg-api-time">--</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">API Calls</span>
                                <span class="stat-value" id="api-calls">--</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Error Rate</span>
                                <span class="stat-value" id="error-rate">--</span>
                            </div>
                        </div>
                        <canvas id="api-chart" width="400" height="200"></canvas>
                    </div>

                    <!-- Resource Loading -->
                    <div class="metric-card">
                        <h4>Resource Loading</h4>
                        <div class="resource-list" id="resource-list">
                            <!-- Resources will be populated here -->
                        </div>
                    </div>

                    <!-- Real-time Metrics -->
                    <div class="metric-card">
                        <h4>Real-time Metrics</h4>
                        <div class="realtime-metrics">
                            <div class="metric-item">
                                <span class="metric-label">Memory Usage</span>
                                <span class="metric-value" id="memory-usage">--</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Connection Type</span>
                                <span class="metric-value" id="connection-type">--</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">Effective Type</span>
                                <span class="metric-value" id="effective-type">--</span>
                            </div>
                        </div>
                        <canvas id="realtime-chart" width="400" height="200"></canvas>
                    </div>

                    <!-- Error Log -->
                    <div class="metric-card">
                        <h4>Error Log</h4>
                        <div class="error-log" id="error-log">
                            <!-- Errors will be populated here -->
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .performance-dashboard {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.95);
                z-index: 10000;
                overflow-y: auto;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                color: #fff;
            }

            .dashboard-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px;
                background: #1a1a1a;
                border-bottom: 1px solid #333;
            }

            .dashboard-header h3 {
                margin: 0;
                font-size: 1.5rem;
            }

            .dashboard-controls {
                display: flex;
                gap: 10px;
            }

            .btn {
                padding: 8px 16px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
            }

            .btn:hover {
                background: #0056b3;
            }

            .dashboard-content {
                padding: 20px;
            }

            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
            }

            .metric-card {
                background: #2a2a2a;
                border-radius: 8px;
                padding: 20px;
                border: 1px solid #444;
            }

            .metric-card h4 {
                margin: 0 0 15px 0;
                color: #fff;
                font-size: 1.2rem;
            }

            .vitals-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }

            .vital-item {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 10px;
                background: #333;
                border-radius: 4px;
            }

            .vital-label {
                font-size: 0.9rem;
                color: #ccc;
            }

            .vital-value {
                font-size: 1.5rem;
                font-weight: bold;
                margin: 5px 0;
            }

            .vital-status {
                font-size: 0.8rem;
                padding: 2px 6px;
                border-radius: 10px;
            }

            .status-good { background: #28a745; color: white; }
            .status-needs-improvement { background: #ffc107; color: black; }
            .status-poor { background: #dc3545; color: white; }

            .performance-stats, .api-stats {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }

            .stat-item {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 10px;
                background: #333;
                border-radius: 4px;
            }

            .stat-label {
                font-size: 0.9rem;
                color: #ccc;
            }

            .stat-value {
                font-size: 1.3rem;
                font-weight: bold;
                margin-top: 5px;
            }

            .resource-list, .error-log {
                max-height: 200px;
                overflow-y: auto;
                background: #333;
                border-radius: 4px;
                padding: 10px;
            }

            .resource-item, .error-item {
                display: flex;
                justify-content: space-between;
                padding: 5px 0;
                border-bottom: 1px solid #444;
                font-size: 0.9rem;
            }

            .resource-item:last-child, .error-item:last-child {
                border-bottom: none;
            }

            .realtime-metrics {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }

            .metric-item {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 10px;
                background: #333;
                border-radius: 4px;
            }

            .metric-label {
                font-size: 0.9rem;
                color: #ccc;
            }

            .metric-value {
                font-size: 1.1rem;
                font-weight: bold;
                margin-top: 5px;
            }

            canvas {
                width: 100%;
                height: 200px;
                background: #1a1a1a;
                border-radius: 4px;
            }
        `;
        document.head.appendChild(style);
    }

    setupEventListeners() {
        // Dashboard controls
        document.getElementById('refresh-dashboard').addEventListener('click', () => {
            this.updateDashboard();
        });

        document.getElementById('export-data').addEventListener('click', () => {
            this.exportData();
        });

        document.getElementById('close-dashboard').addEventListener('click', () => {
            this.hide();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isVisible) {
                this.hide();
            }
        });
    }

    startDataCollection() {
        // Update dashboard periodically
        setInterval(() => {
            if (this.isVisible) {
                this.updateDashboard();
            }
        }, this.updateInterval);

        // Collect real-time metrics
        this.collectNetworkInformation();
        this.collectMemoryUsage();
        this.monitorResources();
    }

    collectNetworkInformation() {
        if ('connection' in navigator) {
            const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
            
            if (connection) {
                document.getElementById('connection-type').textContent = connection.type || 'Unknown';
                document.getElementById('effective-type').textContent = connection.effectiveType || 'Unknown';
            }
        }
    }

    collectMemoryUsage() {
        setInterval(() => {
            if (performance.memory) {
                const used = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
                const total = Math.round(performance.memory.totalJSHeapSize / 1024 / 1024);
                document.getElementById('memory-usage').textContent = `${used}MB / ${total}MB`;
            }
        }, 1000);
    }

    monitorResources() {
        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.entryType === 'resource') {
                    this.metrics.resources.push({
                        name: entry.name,
                        duration: entry.duration,
                        size: entry.transferSize,
                        timestamp: entry.startTime
                    });
                }
            }
        });

        observer.observe({ entryTypes: ['resource'] });
    }

    updateDashboard() {
        this.updateCoreWebVitals();
        this.updatePagePerformance();
        this.updateAPIPerformance();
        this.updateResourceList();
        this.updateErrorLog();
        this.updateCharts();
    }

    updateCoreWebVitals() {
        const vitals = this.metrics.coreWebVitals;
        
        Object.entries(vitals).forEach(([metric, values]) => {
            if (values.length > 0) {
                const latest = values[values.length - 1];
                const element = document.getElementById(`${metric.toLowerCase()}-value`);
                const statusElement = document.getElementById(`${metric.toLowerCase()}-status`);
                
                if (element) {
                    element.textContent = this.formatValue(latest.value, metric);
                }
                
                if (statusElement) {
                    const status = this.getVitalStatus(metric, latest.value);
                    statusElement.textContent = status.text;
                    statusElement.className = `vital-status ${status.class}`;
                }
            }
        });
    }

    updatePagePerformance() {
        const loads = this.metrics.pageLoads;
        
        if (loads.length > 0) {
            const avgLoadTime = loads.reduce((sum, load) => sum + load.loadTime, 0) / loads.length;
            document.getElementById('avg-load-time').textContent = `${avgLoadTime.toFixed(0)}ms`;
        }
        
        document.getElementById('total-requests').textContent = loads.length;
        
        const failed = loads.filter(load => load.failed).length;
        document.getElementById('failed-requests').textContent = failed;
    }

    updateAPIPerformance() {
        const apiCalls = this.metrics.apiCalls;
        
        if (apiCalls.length > 0) {
            const avgResponseTime = apiCalls.reduce((sum, call) => sum + call.responseTime, 0) / apiCalls.length;
            document.getElementById('avg-api-time').textContent = `${avgResponseTime.toFixed(0)}ms`;
        }
        
        document.getElementById('api-calls').textContent = apiCalls.length;
        
        const errors = apiCalls.filter(call => call.error).length;
        const errorRate = apiCalls.length > 0 ? (errors / apiCalls.length * 100).toFixed(1) : 0;
        document.getElementById('error-rate').textContent = `${errorRate}%`;
    }

    updateResourceList() {
        const resourceList = document.getElementById('resource-list');
        const resources = this.metrics.resources
            .slice(-20) // Show last 20 resources
            .sort((a, b) => b.duration - a.duration);

        resourceList.innerHTML = resources.map(resource => `
            <div class="resource-item">
                <span class="resource-name">${this.truncateText(resource.name, 40)}</span>
                <span class="resource-duration">${resource.duration.toFixed(0)}ms</span>
                <span class="resource-size">${this.formatBytes(resource.size)}</span>
            </div>
        `).join('');
    }

    updateErrorLog() {
        const errorLog = document.getElementById('error-log');
        const errors = this.metrics.errors.slice(-10); // Show last 10 errors

        errorLog.innerHTML = errors.map(error => `
            <div class="error-item">
                <span class="error-message">${this.truncateText(error.message, 50)}</span>
                <span class="error-time">${new Date(error.timestamp).toLocaleTimeString()}</span>
            </div>
        `).join('');
    }

    updateCharts() {
        // Update charts if charting library is available
        if (typeof Chart !== 'undefined') {
            this.updateVitalsChart();
            this.updatePerformanceChart();
            this.updateAPIChart();
            this.updateRealtimeChart();
        }
    }

    updateVitalsChart() {
        const ctx = document.getElementById('vitals-chart').getContext('2d');
        const vitals = this.metrics.coreWebVitals;
        
        if (this.chartInstances.vitals) {
            this.chartInstances.vitals.destroy();
        }
        
        this.chartInstances.vitals = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 20}, (_, i) => i),
                datasets: Object.entries(vitals).map(([metric, values]) => ({
                    label: metric,
                    data: values.slice(-20).map(v => v.value),
                    borderColor: this.getChartColor(metric),
                    backgroundColor: this.getChartColor(metric, 0.2),
                    tension: 0.1
                }))
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#fff'
                        }
                    }
                }
            }
        });
    }

    formatValue(value, metric) {
        switch (metric) {
            case 'LCP':
            case 'FCP':
            case 'FID':
                return `${value.toFixed(0)}ms`;
            case 'CLS':
                return value.toFixed(3);
            case 'TTFB':
                return `${value.toFixed(0)}ms`;
            default:
                return value.toFixed(2);
        }
    }

    getVitalStatus(metric, value) {
        const thresholds = {
            LCP: { good: 2500, needsImprovement: 4000 },
            FID: { good: 100, needsImprovement: 300 },
            CLS: { good: 0.1, needsImprovement: 0.25 },
            FCP: { good: 1800, needsImprovement: 3000 },
            TTFB: { good: 800, needsImprovement: 1800 }
        };

        const threshold = thresholds[metric];
        if (!threshold) return { text: 'Unknown', class: 'status-unknown' };

        if (value <= threshold.good) {
            return { text: 'Good', class: 'status-good' };
        } else if (value <= threshold.needsImprovement) {
            return { text: 'Needs Improvement', class: 'status-needs-improvement' };
        } else {
            return { text: 'Poor', class: 'status-poor' };
        }
    }

    getChartColor(metric, alpha = 1) {
        const colors = {
            LCP: `rgba(255, 99, 132, ${alpha})`,
            FID: `rgba(54, 162, 235, ${alpha})`,
            CLS: `rgba(255, 205, 86, ${alpha})`,
            FCP: `rgba(75, 192, 192, ${alpha})`,
            TTFB: `rgba(153, 102, 255, ${alpha})`
        };
        return colors[metric] || `rgba(255, 255, 255, ${alpha})`;
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Public API
    show() {
        this.dashboardElement.style.display = 'block';
        this.isVisible = true;
        this.updateDashboard();
    }

    hide() {
        this.dashboardElement.style.display = 'none';
        this.isVisible = false;
    }

    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    addMetric(type, data) {
        switch (type) {
            case 'core-web-vital':
                if (this.metrics.coreWebVitals[data.metric]) {
                    this.metrics.coreWebVitals[data.metric].push({
                        value: data.value,
                        timestamp: data.timestamp || Date.now()
                    });
                }
                break;
            case 'page-load':
                this.metrics.pageLoads.push({
                    loadTime: data.loadTime,
                    page: data.page,
                    failed: data.failed || false,
                    timestamp: data.timestamp || Date.now()
                });
                break;
            case 'api-call':
                this.metrics.apiCalls.push({
                    endpoint: data.endpoint,
                    responseTime: data.responseTime,
                    error: data.error || false,
                    timestamp: data.timestamp || Date.now()
                });
                break;
            case 'error':
                this.metrics.errors.push({
                    message: data.message,
                    stack: data.stack,
                    timestamp: data.timestamp || Date.now()
                });
                break;
        }
    }

    exportData() {
        const data = {
            metrics: this.metrics,
            exported: new Date().toISOString(),
            url: window.location.href
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `performance-data-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    reset() {
        this.metrics = {
            pageLoads: [],
            apiCalls: [],
            coreWebVitals: {
                LCP: [],
                FID: [],
                CLS: [],
                FCP: [],
                TTFB: []
            },
            resources: [],
            errors: []
        };
        this.updateDashboard();
    }
}

// Create global instance
window.performanceDashboard = new PerformanceDashboard();

// Add keyboard shortcut to toggle dashboard
document.addEventListener('keydown', (e) => {
    if (e.key === 'F12' && e.ctrlKey) {
        e.preventDefault();
        window.performanceDashboard.toggle();
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceDashboard;
}