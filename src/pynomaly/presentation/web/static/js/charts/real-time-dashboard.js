/**
 * Real-Time Anomaly Detection Dashboard
 * WebSocket-enabled dashboard for live anomaly monitoring
 */

import { AnomalyTimelineChart } from './anomaly-timeline.js';
import { AnomalyHeatmapChart } from './anomaly-heatmap.js';

export class RealTimeDashboard {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      websocketUrl: 'ws://localhost:8000/ws/anomalies',
      updateInterval: 1000, // 1 second
      maxDataPoints: 1000,
      alertThreshold: 0.8,
      ...options
    };
    
    this.websocket = null;
    this.charts = new Map();
    this.dataBuffers = new Map();
    this.isConnected = false;
    this.alertCount = 0;
    this.totalDataPoints = 0;
    
    this.init();
  }
  
  init() {
    this.createLayout();
    this.initializeCharts();
    this.setupWebSocket();
    this.setupEventHandlers();
    this.startHeartbeat();
  }
  
  createLayout() {
    this.container.innerHTML = `
      <div class="dashboard-header">
        <div class="dashboard-title">
          <h2>Real-Time Anomaly Detection</h2>
          <div class="connection-status" id="connection-status">
            <span class="status-indicator disconnected"></span>
            <span class="status-text">Connecting...</span>
          </div>
        </div>
        <div class="dashboard-controls">
          <button id="pause-btn" class="btn-secondary btn-sm">Pause</button>
          <button id="clear-btn" class="btn-secondary btn-sm">Clear Data</button>
          <button id="export-btn" class="btn-primary btn-sm">Export</button>
        </div>
      </div>
      
      <div class="dashboard-metrics">
        <div class="metric-card" id="total-points">
          <div class="metric-icon">📊</div>
          <div class="metric-content">
            <div class="metric-value">0</div>
            <div class="metric-label">Total Data Points</div>
          </div>
        </div>
        <div class="metric-card" id="anomaly-count">
          <div class="metric-icon">⚠️</div>
          <div class="metric-content">
            <div class="metric-value">0</div>
            <div class="metric-label">Anomalies Detected</div>
          </div>
        </div>
        <div class="metric-card" id="anomaly-rate">
          <div class="metric-icon">📈</div>
          <div class="metric-content">
            <div class="metric-value">0.0%</div>
            <div class="metric-label">Anomaly Rate</div>
          </div>
        </div>
        <div class="metric-card" id="data-rate">
          <div class="metric-icon">⚡</div>
          <div class="metric-content">
            <div class="metric-value">0</div>
            <div class="metric-label">Data Points/sec</div>
          </div>
        </div>
      </div>
      
      <div class="dashboard-charts">
        <div class="chart-section">
          <div class="chart-header">
            <h3>Anomaly Timeline</h3>
            <div class="chart-controls">
              <select id="timeline-interval">
                <option value="1">Last 1 hour</option>
                <option value="6">Last 6 hours</option>
                <option value="24" selected>Last 24 hours</option>
                <option value="168">Last week</option>
              </select>
            </div>
          </div>
          <div id="timeline-chart" class="chart-container"></div>
        </div>
        
        <div class="chart-section">
          <div class="chart-header">
            <h3>Feature Anomaly Heatmap</h3>
            <div class="chart-controls">
              <select id="heatmap-aggregation">
                <option value="hour">Hourly</option>
                <option value="15min" selected>15 Minutes</option>
                <option value="5min">5 Minutes</option>
              </select>
            </div>
          </div>
          <div id="heatmap-chart" class="chart-container"></div>
        </div>
      </div>
      
      <div class="dashboard-alerts">
        <div class="alerts-header">
          <h3>Recent Alerts</h3>
          <button id="clear-alerts" class="btn-ghost btn-sm">Clear All</button>
        </div>
        <div id="alerts-container" class="alerts-list"></div>
      </div>
      
      <div class="dashboard-data-sources">
        <div class="data-sources-header">
          <h3>Active Data Sources</h3>
          <button id="add-source" class="btn-primary btn-sm">Add Source</button>
        </div>
        <div id="data-sources-list" class="data-sources-grid"></div>
      </div>
    `;
  }
  
  initializeCharts() {
    // Initialize timeline chart
    const timelineContainer = this.container.querySelector('#timeline-chart');
    this.charts.set('timeline', new AnomalyTimelineChart(timelineContainer, {
      width: timelineContainer.clientWidth,
      height: 300,
      interactive: true,
      showLegend: true,
      realTimeUpdates: true
    }));
    
    // Initialize heatmap chart
    const heatmapContainer = this.container.querySelector('#heatmap-chart');
    this.charts.set('heatmap', new AnomalyHeatmapChart(heatmapContainer, {
      width: heatmapContainer.clientWidth,
      height: 400,
      interactive: true,
      showLegend: true
    }));
    
    // Initialize data buffers
    this.dataBuffers.set('timeline', []);
    this.dataBuffers.set('heatmap', []);
  }
  
  setupWebSocket() {
    this.websocket = new WebSocket(this.options.websocketUrl);
    
    this.websocket.onopen = () => {
      this.isConnected = true;
      this.updateConnectionStatus('connected', 'Connected');
      console.log('WebSocket connected');
    };
    
    this.websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleIncomingData(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    this.websocket.onclose = () => {
      this.isConnected = false;
      this.updateConnectionStatus('disconnected', 'Disconnected');
      console.log('WebSocket disconnected');
      
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (!this.isConnected) {
          this.setupWebSocket();
        }
      }, 5000);
    };
    
    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.updateConnectionStatus('error', 'Connection Error');
    };
  }
  
  setupEventHandlers() {
    // Pause/Resume button
    const pauseBtn = this.container.querySelector('#pause-btn');
    let isPaused = false;
    
    pauseBtn.addEventListener('click', () => {
      isPaused = !isPaused;
      pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';
      pauseBtn.className = isPaused ? 'btn-primary btn-sm' : 'btn-secondary btn-sm';
      
      if (isPaused) {
        this.pauseUpdates();
      } else {
        this.resumeUpdates();
      }
    });
    
    // Clear data button
    this.container.querySelector('#clear-btn').addEventListener('click', () => {
      this.clearAllData();
    });
    
    // Export button
    this.container.querySelector('#export-btn').addEventListener('click', () => {
      this.exportData();
    });
    
    // Timeline interval selector
    this.container.querySelector('#timeline-interval').addEventListener('change', (e) => {
      this.updateTimelineRange(parseInt(e.target.value));
    });
    
    // Heatmap aggregation selector
    this.container.querySelector('#heatmap-aggregation').addEventListener('change', (e) => {
      this.updateHeatmapAggregation(e.target.value);
    });
    
    // Clear alerts
    this.container.querySelector('#clear-alerts').addEventListener('click', () => {
      this.clearAlerts();
    });
    
    // Add data source
    this.container.querySelector('#add-source').addEventListener('click', () => {
      this.showAddSourceDialog();
    });
  }
  
  handleIncomingData(data) {
    this.totalDataPoints++;
    
    // Process different data types
    switch (data.type) {
      case 'anomaly_point':
        this.handleAnomalyPoint(data);
        break;
      case 'batch_update':
        this.handleBatchUpdate(data);
        break;
      case 'system_status':
        this.handleSystemStatus(data);
        break;
      default:
        console.warn('Unknown data type:', data.type);
    }
    
    this.updateMetrics();
  }
  
  handleAnomalyPoint(data) {
    const point = {
      id: data.id,
      timestamp: new Date(data.timestamp),
      value: data.value,
      feature: data.feature,
      isAnomaly: data.isAnomaly,
      confidence: data.confidence || 0,
      anomalyScore: data.anomalyScore || 0
    };
    
    // Add to timeline buffer
    const timelineBuffer = this.dataBuffers.get('timeline');
    timelineBuffer.push(point);
    
    // Maintain buffer size
    if (timelineBuffer.length > this.options.maxDataPoints) {
      timelineBuffer.shift();
    }
    
    // Update timeline chart
    this.charts.get('timeline').addDataPoint(point);
    
    // Process for heatmap
    this.processHeatmapData(point);
    
    // Check for alerts
    if (point.isAnomaly && point.confidence >= this.options.alertThreshold) {
      this.createAlert(point);
    }
  }
  
  handleBatchUpdate(data) {
    data.points.forEach(point => {
      this.handleAnomalyPoint({ ...point, type: 'anomaly_point' });
    });
  }
  
  handleSystemStatus(data) {
    this.updateSystemStatus(data);
  }
  
  processHeatmapData(point) {
    const heatmapBuffer = this.dataBuffers.get('heatmap');
    
    // Group data by feature and time interval
    const timeInterval = this.getTimeInterval(point.timestamp);
    const key = `${point.feature}-${timeInterval}`;
    
    let existing = heatmapBuffer.find(d => d.key === key);
    if (existing) {
      existing.values.push(point);
      existing.anomalyScore = Math.max(existing.anomalyScore, point.anomalyScore);
    } else {
      heatmapBuffer.push({
        key,
        x: timeInterval,
        y: point.feature,
        values: [point],
        value: point.value,
        anomalyScore: point.anomalyScore
      });
    }
    
    // Update heatmap with aggregated data
    const heatmapData = heatmapBuffer.map(d => ({
      x: d.x,
      y: d.y,
      value: d3.mean(d.values, v => v.value),
      anomalyScore: d.anomalyScore
    }));
    
    this.charts.get('heatmap').setData(heatmapData);
  }
  
  getTimeInterval(timestamp) {
    const aggregation = this.container.querySelector('#heatmap-aggregation').value;
    const date = new Date(timestamp);
    
    switch (aggregation) {
      case '5min':
        const minutes5 = Math.floor(date.getMinutes() / 5) * 5;
        return `${date.getHours().toString().padStart(2, '0')}:${minutes5.toString().padStart(2, '0')}`;
      case '15min':
        const minutes15 = Math.floor(date.getMinutes() / 15) * 15;
        return `${date.getHours().toString().padStart(2, '0')}:${minutes15.toString().padStart(2, '0')}`;
      case 'hour':
      default:
        return `${date.getHours().toString().padStart(2, '0')}:00`;
    }
  }
  
  createAlert(point) {
    this.alertCount++;
    
    const alertsContainer = this.container.querySelector('#alerts-container');
    const alertElement = document.createElement('div');
    alertElement.className = 'alert-item';
    alertElement.innerHTML = `
      <div class="alert-icon">🚨</div>
      <div class="alert-content">
        <div class="alert-title">High Confidence Anomaly</div>
        <div class="alert-details">
          Feature: ${point.feature} | 
          Confidence: ${Math.round(point.confidence * 100)}% | 
          ${new Date(point.timestamp).toLocaleTimeString()}
        </div>
      </div>
      <button class="alert-dismiss" onclick="this.parentElement.remove()">×</button>
    `;
    
    alertsContainer.insertBefore(alertElement, alertsContainer.firstChild);
    
    // Auto-remove after 30 seconds
    setTimeout(() => {
      if (alertElement.parentElement) {
        alertElement.remove();
      }
    }, 30000);
    
    // Show browser notification if permitted
    this.showBrowserNotification(point);
  }
  
  showBrowserNotification(point) {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Anomaly Detected', {
        body: `High confidence anomaly in ${point.feature} (${Math.round(point.confidence * 100)}%)`,
        icon: '/static/images/alert-icon.png',
        tag: 'anomaly-alert'
      });
    }
  }
  
  updateMetrics() {
    // Update total data points
    this.container.querySelector('#total-points .metric-value').textContent = 
      this.totalDataPoints.toLocaleString();
    
    // Update anomaly count
    const timelineData = this.dataBuffers.get('timeline');
    const anomalyCount = timelineData.filter(d => d.isAnomaly).length;
    this.container.querySelector('#anomaly-count .metric-value').textContent = 
      anomalyCount.toLocaleString();
    
    // Update anomaly rate
    const anomalyRate = timelineData.length > 0 ? 
      (anomalyCount / timelineData.length * 100).toFixed(1) : '0.0';
    this.container.querySelector('#anomaly-rate .metric-value').textContent = 
      `${anomalyRate}%`;
    
    // Update data rate (points per second)
    this.updateDataRate();
  }
  
  updateDataRate() {
    const now = Date.now();
    const oneSecondAgo = now - 1000;
    const timelineData = this.dataBuffers.get('timeline');
    
    const recentPoints = timelineData.filter(d => 
      d.timestamp.getTime() > oneSecondAgo
    ).length;
    
    this.container.querySelector('#data-rate .metric-value').textContent = 
      recentPoints.toString();
  }
  
  updateConnectionStatus(status, text) {
    const statusElement = this.container.querySelector('#connection-status');
    const indicator = statusElement.querySelector('.status-indicator');
    const textElement = statusElement.querySelector('.status-text');
    
    indicator.className = `status-indicator ${status}`;
    textElement.textContent = text;
  }
  
  updateTimelineRange(hours) {
    const cutoffTime = new Date(Date.now() - hours * 60 * 60 * 1000);
    const timelineData = this.dataBuffers.get('timeline');
    const filteredData = timelineData.filter(d => d.timestamp >= cutoffTime);
    
    this.charts.get('timeline').setData(filteredData);
  }
  
  updateHeatmapAggregation(aggregation) {
    // Reprocess heatmap data with new aggregation
    const timelineData = this.dataBuffers.get('timeline');
    this.dataBuffers.set('heatmap', []);
    
    timelineData.forEach(point => {
      this.processHeatmapData(point);
    });
  }
  
  clearAllData() {
    this.dataBuffers.set('timeline', []);
    this.dataBuffers.set('heatmap', []);
    this.totalDataPoints = 0;
    this.alertCount = 0;
    
    this.charts.get('timeline').setData([]);
    this.charts.get('heatmap').setData([]);
    
    this.updateMetrics();
    this.clearAlerts();
  }
  
  clearAlerts() {
    this.container.querySelector('#alerts-container').innerHTML = '';
  }
  
  pauseUpdates() {
    // Implement pause logic
    this.isPaused = true;
  }
  
  resumeUpdates() {
    // Implement resume logic
    this.isPaused = false;
  }
  
  exportData() {
    const data = {
      timeline: this.dataBuffers.get('timeline'),
      heatmap: this.dataBuffers.get('heatmap'),
      exportTime: new Date().toISOString(),
      totalDataPoints: this.totalDataPoints,
      alertCount: this.alertCount
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const downloadLink = document.createElement('a');
    downloadLink.download = `anomaly-data-${new Date().toISOString().split('T')[0]}.json`;
    downloadLink.href = url;
    downloadLink.click();
    
    URL.revokeObjectURL(url);
  }
  
  showAddSourceDialog() {
    // Implementation for adding new data sources
    const dialog = document.createElement('div');
    dialog.className = 'modal-overlay';
    dialog.innerHTML = `
      <div class="modal-dialog">
        <div class="modal-header">
          <h3>Add Data Source</h3>
          <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">×</button>
        </div>
        <div class="modal-body">
          <form id="add-source-form">
            <div class="form-group">
              <label class="form-label">Source Type</label>
              <select class="form-select" name="type">
                <option value="websocket">WebSocket</option>
                <option value="http">HTTP Endpoint</option>
                <option value="file">File Upload</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">Source URL/Path</label>
              <input type="text" class="form-input" name="url" placeholder="Enter URL or file path">
            </div>
            <div class="form-group">
              <label class="form-label">Update Interval (seconds)</label>
              <input type="number" class="form-input" name="interval" value="1" min="1">
            </div>
          </form>
        </div>
        <div class="modal-footer">
          <button class="btn-secondary" onclick="this.closest('.modal-overlay').remove()">Cancel</button>
          <button class="btn-primary" onclick="this.closest('.modal-dialog').querySelector('form').dispatchEvent(new Event('submit'))">Add Source</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(dialog);
    
    dialog.querySelector('#add-source-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const formData = new FormData(e.target);
      this.addDataSource({
        type: formData.get('type'),
        url: formData.get('url'),
        interval: parseInt(formData.get('interval'))
      });
      dialog.remove();
    });
  }
  
  addDataSource(config) {
    // Implementation for adding new data sources
    console.log('Adding data source:', config);
    // This would typically involve configuring new WebSocket connections
    // or HTTP polling for additional data streams
  }
  
  updateSystemStatus(status) {
    // Update system status indicators
    console.log('System status:', status);
  }
  
  startHeartbeat() {
    // Send periodic heartbeat to maintain connection
    setInterval(() => {
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify({ type: 'heartbeat' }));
      }
    }, 30000); // Every 30 seconds
  }
  
  requestNotificationPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }
  
  destroy() {
    if (this.websocket) {
      this.websocket.close();
    }
    
    this.charts.forEach(chart => chart.destroy());
    this.charts.clear();
    this.dataBuffers.clear();
  }
}

// Factory function for easy instantiation
export function createRealTimeDashboard(container, options = {}) {
  return new RealTimeDashboard(container, options);
}

// Auto-initialize dashboard if container exists
export function initializeDashboard() {
  const container = document.querySelector('[data-component="real-time-dashboard"]');
  if (!container) return;
  
  const dashboard = new RealTimeDashboard(container);
  
  // Request notification permission
  dashboard.requestNotificationPermission();
  
  return dashboard;
}

// Export for global access
window.RealTimeDashboard = RealTimeDashboard;
