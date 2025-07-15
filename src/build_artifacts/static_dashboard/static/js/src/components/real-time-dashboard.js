/**
 * Real-Time Dashboard Component
 *
 * Advanced dashboard for live anomaly detection monitoring with
 * real-time charts, alerts, and system metrics visualization
 */

import { AnomalyWebSocketClient } from "../services/websocket-service.js";
import { AnomalyTimeline } from "./anomaly-timeline.js";
import { DashboardLayout } from "./dashboard-layout.js";

export class RealTimeDashboard {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      enableRealTime: true,
      updateInterval: 1000,
      maxDataPoints: 1000,
      alertThreshold: 0.7,
      enableSound: false,
      enableNotifications: true,
      autoScroll: true,
      theme: "light",
      ...options,
    };

    this.wsClient = null;
    this.dashboardLayout = null;
    this.timeline = null;
    this.isActive = false;
    this.lastUpdate = null;

    this.data = {
      realTimeData: [],
      alerts: [],
      systemMetrics: {},
      connectionStatus: "disconnected",
      statistics: {
        totalAnomalies: 0,
        criticalAlerts: 0,
        avgScore: 0,
        detectionRate: 0,
      },
    };

    this.charts = new Map();
    this.widgets = new Map();
    this.updateTimers = new Map();

    this.init();
  }

  init() {
    this.setupContainer();
    this.initializeWebSocket();
    this.createDashboardLayout();
    this.bindEvents();

    if (this.options.enableRealTime) {
      this.start();
    }
  }

  setupContainer() {
    this.container.classList.add("real-time-dashboard");
    this.container.innerHTML = "";

    // Add dashboard header
    this.header = document.createElement("div");
    this.header.className = "dashboard-header";
    this.header.innerHTML = `
            <div class="dashboard-title">
                <h2>Real-Time Anomaly Detection</h2>
                <div class="connection-status" id="connection-status">
                    <span class="status-indicator"></span>
                    <span class="status-text">Connecting...</span>
                </div>
            </div>
            <div class="dashboard-controls">
                <button class="btn btn-primary" id="start-btn">Start Monitoring</button>
                <button class="btn btn-secondary" id="pause-btn" disabled>Pause</button>
                <button class="btn btn-ghost" id="settings-btn">Settings</button>
                <button class="btn btn-ghost" id="fullscreen-btn">⛶</button>
            </div>
        `;

    // Dashboard content area
    this.content = document.createElement("div");
    this.content.className = "dashboard-content";

    this.container.appendChild(this.header);
    this.container.appendChild(this.content);
  }

  initializeWebSocket() {
    this.wsClient = new AnomalyWebSocketClient({
      enableLogging: true,
      autoConnect: false,
      enableRealTimeDetection: true,
      enableAlerts: true,
      enableMetrics: true,
    });

    // Set up WebSocket event handlers
    this.wsClient.on("connected", () => {
      this.updateConnectionStatus("connected");
      this.onWebSocketConnected();
    });

    this.wsClient.on("disconnected", () => {
      this.updateConnectionStatus("disconnected");
      this.onWebSocketDisconnected();
    });

    this.wsClient.on("anomaly_detected", (anomaly) => {
      this.onAnomalyDetected(anomaly);
    });

    this.wsClient.on("real_time_data", (data) => {
      this.onRealTimeData(data);
    });

    this.wsClient.on("alert", (alert) => {
      this.onAlert(alert);
    });

    this.wsClient.on("system_metrics", (metrics) => {
      this.onSystemMetrics(metrics);
    });
  }

  createDashboardLayout() {
    this.dashboardLayout = new DashboardLayout(this.content, {
      columns: 12,
      rowHeight: 80,
      margin: [10, 10],
      isDraggable: true,
      isResizable: true,
    });

    // Add default widgets
    this.addWidget({
      id: "connection-info",
      title: "Connection Status",
      type: "status",
      x: 0,
      y: 0,
      w: 3,
      h: 2,
      component: () => this.createConnectionWidget(),
    });

    this.addWidget({
      id: "live-timeline",
      title: "Live Anomaly Timeline",
      type: "chart",
      x: 3,
      y: 0,
      w: 9,
      h: 6,
      component: () => this.createTimelineWidget(),
    });

    this.addWidget({
      id: "statistics",
      title: "Detection Statistics",
      type: "metrics",
      x: 0,
      y: 2,
      w: 3,
      h: 4,
      component: () => this.createStatisticsWidget(),
    });

    this.addWidget({
      id: "live-alerts",
      title: "Live Alerts",
      type: "alerts",
      x: 0,
      y: 6,
      w: 6,
      h: 4,
      component: () => this.createAlertsWidget(),
    });

    this.addWidget({
      id: "system-metrics",
      title: "System Metrics",
      type: "metrics",
      x: 6,
      y: 6,
      w: 6,
      h: 4,
      component: () => this.createSystemMetricsWidget(),
    });

    this.addWidget({
      id: "data-stream",
      title: "Data Stream",
      type: "stream",
      x: 0,
      y: 10,
      w: 12,
      h: 3,
      component: () => this.createDataStreamWidget(),
    });
  }

  addWidget(config) {
    this.dashboardLayout.addWidget(config);
    return config.id;
  }

  createConnectionWidget() {
    const widget = document.createElement("div");
    widget.className = "connection-widget";

    const update = () => {
      const info = this.wsClient.getConnectionInfo();
      widget.innerHTML = `
                <div class="connection-info">
                    <div class="status-row">
                        <span class="label">Status:</span>
                        <span class="value status-${info.state}">${info.state}</span>
                    </div>
                    <div class="status-row">
                        <span class="label">Connection ID:</span>
                        <span class="value">${info.connectionId || "N/A"}</span>
                    </div>
                    <div class="status-row">
                        <span class="label">Subscriptions:</span>
                        <span class="value">${info.subscriptions.length}</span>
                    </div>
                    <div class="status-row">
                        <span class="label">Queued:</span>
                        <span class="value">${info.queuedMessages}</span>
                    </div>
                    ${
                      info.state === "connecting"
                        ? `
                        <div class="status-row">
                            <span class="label">Attempts:</span>
                            <span class="value">${info.reconnectAttempts}</span>
                        </div>
                    `
                        : ""
                    }
                </div>
            `;
    };

    update();
    this.updateTimers.set("connection", setInterval(update, 2000));

    return widget;
  }

  createTimelineWidget() {
    const widget = document.createElement("div");
    widget.className = "timeline-widget";
    widget.style.height = "100%";

    // Initialize timeline component
    this.timeline = new AnomalyTimeline(widget, {
      width: widget.clientWidth || 800,
      height: 400,
      enableRealTime: true,
      enableZoom: true,
      enableBrush: true,
      maxDataPoints: this.options.maxDataPoints,
    });

    return widget;
  }

  createStatisticsWidget() {
    const widget = document.createElement("div");
    widget.className = "statistics-widget";

    const update = () => {
      const stats = this.data.statistics;
      widget.innerHTML = `
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">${stats.totalAnomalies}</div>
                        <div class="stat-label">Total Anomalies</div>
                    </div>
                    <div class="stat-item critical">
                        <div class="stat-value">${stats.criticalAlerts}</div>
                        <div class="stat-label">Critical Alerts</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.avgScore.toFixed(3)}</div>
                        <div class="stat-label">Avg Score</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${stats.detectionRate.toFixed(1)}/min</div>
                        <div class="stat-label">Detection Rate</div>
                    </div>
                </div>
            `;
    };

    update();
    this.updateTimers.set("statistics", setInterval(update, 1000));

    return widget;
  }

  createAlertsWidget() {
    const widget = document.createElement("div");
    widget.className = "alerts-widget";

    widget.innerHTML = `
            <div class="alerts-header">
                <span class="alerts-count">0 active alerts</span>
                <button class="btn btn-sm btn-ghost" id="clear-alerts">Clear All</button>
            </div>
            <div class="alerts-list" id="alerts-list"></div>
        `;

    widget.querySelector("#clear-alerts").onclick = () => {
      this.clearAlerts();
    };

    const update = () => {
      const alertsList = widget.querySelector("#alerts-list");
      const alertsCount = widget.querySelector(".alerts-count");
      const unacknowledged = this.wsClient.getUnacknowledgedAlerts();

      alertsCount.textContent = `${unacknowledged.length} active alerts`;

      alertsList.innerHTML = unacknowledged
        .slice(0, 10)
        .map(
          (alert) => `
                <div class="alert-item severity-${alert.severity}" data-alert-id="${alert.id}">
                    <div class="alert-content">
                        <div class="alert-message">${alert.message}</div>
                        <div class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</div>
                    </div>
                    <button class="alert-dismiss" onclick="dashboard.acknowledgeAlert('${alert.id}')">×</button>
                </div>
            `,
        )
        .join("");
    };

    update();
    this.updateTimers.set("alerts", setInterval(update, 500));

    return widget;
  }

  createSystemMetricsWidget() {
    const widget = document.createElement("div");
    widget.className = "system-metrics-widget";

    const update = () => {
      const metrics = this.data.systemMetrics;
      widget.innerHTML = `
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">CPU Usage</div>
                        <div class="metric-value">${(metrics.cpu * 100).toFixed(1)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${metrics.cpu * 100}%"></div>
                        </div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Memory Usage</div>
                        <div class="metric-value">${(metrics.memory * 100).toFixed(1)}%</div>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${metrics.memory * 100}%"></div>
                        </div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Active Models</div>
                        <div class="metric-value">${metrics.activeModels || 0}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Queue Size</div>
                        <div class="metric-value">${metrics.queueSize || 0}</div>
                    </div>
                </div>
            `;
    };

    update();
    this.updateTimers.set("system-metrics", setInterval(update, 2000));

    return widget;
  }

  createDataStreamWidget() {
    const widget = document.createElement("div");
    widget.className = "data-stream-widget";

    widget.innerHTML = `
            <div class="stream-header">
                <span>Live Data Stream</span>
                <div class="stream-controls">
                    <button class="btn btn-sm" id="pause-stream">Pause</button>
                    <button class="btn btn-sm" id="clear-stream">Clear</button>
                </div>
            </div>
            <div class="stream-content" id="stream-content"></div>
        `;

    const streamContent = widget.querySelector("#stream-content");
    let isPaused = false;

    widget.querySelector("#pause-stream").onclick = (e) => {
      isPaused = !isPaused;
      e.target.textContent = isPaused ? "Resume" : "Pause";
    };

    widget.querySelector("#clear-stream").onclick = () => {
      streamContent.innerHTML = "";
    };

    const update = () => {
      if (isPaused) return;

      const recentData = this.data.realTimeData.slice(-50);
      streamContent.innerHTML = recentData
        .map(
          (item) => `
                <div class="stream-item">
                    <span class="stream-time">${new Date(item.timestamp).toLocaleTimeString()}</span>
                    <span class="stream-data">${JSON.stringify(item.data).substring(0, 100)}</span>
                    ${item.anomaly ? '<span class="stream-anomaly">⚠️</span>' : ""}
                </div>
            `,
        )
        .join("");

      if (this.options.autoScroll) {
        streamContent.scrollTop = streamContent.scrollHeight;
      }
    };

    this.updateTimers.set("stream", setInterval(update, 1000));

    return widget;
  }

  bindEvents() {
    // Header controls
    this.header.querySelector("#start-btn").onclick = () => this.start();
    this.header.querySelector("#pause-btn").onclick = () => this.pause();
    this.header.querySelector("#settings-btn").onclick = () =>
      this.showSettings();
    this.header.querySelector("#fullscreen-btn").onclick = () =>
      this.toggleFullscreen();

    // Global keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case "p":
            e.preventDefault();
            this.isActive ? this.pause() : this.start();
            break;
          case "r":
            e.preventDefault();
            this.refresh();
            break;
          case "f":
            e.preventDefault();
            this.toggleFullscreen();
            break;
        }
      }
    });
  }

  // WebSocket event handlers
  onWebSocketConnected() {
    console.log("Real-time dashboard connected");
    this.updateStatistics();
  }

  onWebSocketDisconnected() {
    console.log("Real-time dashboard disconnected");
  }

  onAnomalyDetected(anomaly) {
    // Add to timeline
    if (this.timeline) {
      this.timeline.addRealTimeData([anomaly]);
    }

    // Update statistics
    this.data.statistics.totalAnomalies++;
    if (anomaly.severity === "critical") {
      this.data.statistics.criticalAlerts++;
    }

    // Calculate rolling average score
    const recentScores = this.data.realTimeData
      .filter((item) => item.anomaly)
      .slice(-100)
      .map((item) => item.anomaly.score);

    if (recentScores.length > 0) {
      this.data.statistics.avgScore =
        recentScores.reduce((a, b) => a + b) / recentScores.length;
    }

    // Play notification sound
    if (this.options.enableSound && anomaly.severity === "critical") {
      this.playNotificationSound();
    }

    // Show browser notification
    if (this.options.enableNotifications && anomaly.severity === "critical") {
      this.showBrowserNotification(anomaly);
    }
  }

  onRealTimeData(data) {
    this.data.realTimeData.push({
      ...data,
      timestamp: Date.now(),
    });

    // Maintain buffer size
    if (this.data.realTimeData.length > this.options.maxDataPoints) {
      this.data.realTimeData.shift();
    }

    this.lastUpdate = Date.now();
    this.updateDetectionRate();
  }

  onAlert(alert) {
    this.data.alerts.unshift(alert);

    // Maintain alerts buffer
    if (this.data.alerts.length > 1000) {
      this.data.alerts.pop();
    }

    // Update critical alerts count
    if (alert.severity === "critical") {
      this.data.statistics.criticalAlerts++;
    }
  }

  onSystemMetrics(metrics) {
    this.data.systemMetrics = {
      ...this.data.systemMetrics,
      ...metrics,
      lastUpdate: Date.now(),
    };
  }

  // Control methods
  start() {
    if (this.isActive) return;

    this.isActive = true;
    this.wsClient.connect();

    this.header.querySelector("#start-btn").disabled = true;
    this.header.querySelector("#pause-btn").disabled = false;

    this.updateConnectionStatus("connecting");
  }

  pause() {
    if (!this.isActive) return;

    this.isActive = false;
    this.wsClient.disconnect();

    this.header.querySelector("#start-btn").disabled = false;
    this.header.querySelector("#pause-btn").disabled = true;

    this.updateConnectionStatus("paused");
  }

  refresh() {
    this.data.realTimeData = [];
    this.data.alerts = [];
    this.data.statistics = {
      totalAnomalies: 0,
      criticalAlerts: 0,
      avgScore: 0,
      detectionRate: 0,
    };

    if (this.timeline) {
      this.timeline.setData([]);
    }
  }

  showSettings() {
    // Implementation for settings modal
    console.log("Settings modal would open here");
  }

  toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      this.container.requestFullscreen();
    }
  }

  // Utility methods
  updateConnectionStatus(status) {
    this.data.connectionStatus = status;

    const statusElement = this.header.querySelector("#connection-status");
    const indicator = statusElement.querySelector(".status-indicator");
    const text = statusElement.querySelector(".status-text");

    indicator.className = `status-indicator status-${status}`;
    text.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  }

  updateStatistics() {
    // Calculate detection rate (detections per minute)
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    const recentDetections = this.data.realTimeData.filter(
      (item) => item.timestamp > oneMinuteAgo,
    ).length;

    this.data.statistics.detectionRate = recentDetections;
  }

  updateDetectionRate() {
    // Update detection rate every 10 seconds
    if (!this.detectionRateTimer) {
      this.detectionRateTimer = setInterval(() => {
        this.updateStatistics();
      }, 10000);
    }
  }

  acknowledgeAlert(alertId) {
    this.wsClient.acknowledgeAlert(alertId);

    // Remove from UI
    const alertElement = document.querySelector(`[data-alert-id="${alertId}"]`);
    if (alertElement) {
      alertElement.remove();
    }
  }

  clearAlerts() {
    const unacknowledged = this.wsClient.getUnacknowledgedAlerts();
    unacknowledged.forEach((alert) => {
      this.wsClient.acknowledgeAlert(alert.id);
    });
  }

  playNotificationSound() {
    if ("AudioContext" in window) {
      const audioContext = new AudioContext();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      oscillator.frequency.value = 800;
      oscillator.type = "square";
      gainNode.gain.value = 0.1;

      oscillator.start();
      oscillator.stop(audioContext.currentTime + 0.2);
    }
  }

  showBrowserNotification(anomaly) {
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification("Critical Anomaly Detected", {
        body: `Anomaly score: ${anomaly.score.toFixed(3)}`,
        icon: "/static/icons/alert.png",
        tag: "anomaly-alert",
      });
    }
  }

  // Data export methods
  exportData(format = "json") {
    const data = {
      timestamp: new Date().toISOString(),
      statistics: this.data.statistics,
      recentData: this.data.realTimeData.slice(-1000),
      alerts: this.data.alerts.slice(0, 100),
      systemMetrics: this.data.systemMetrics,
    };

    switch (format) {
      case "json":
        return JSON.stringify(data, null, 2);
      case "csv":
        return this.convertToCSV(data.recentData);
      default:
        return data;
    }
  }

  convertToCSV(data) {
    if (data.length === 0) return "";

    const headers = Object.keys(data[0]);
    const rows = data.map((item) =>
      headers.map((header) => JSON.stringify(item[header] || "")).join(","),
    );

    return [headers.join(","), ...rows].join("\n");
  }

  // Cleanup
  destroy() {
    // Clear all timers
    this.updateTimers.forEach((timer) => clearInterval(timer));
    this.updateTimers.clear();

    if (this.detectionRateTimer) {
      clearInterval(this.detectionRateTimer);
    }

    // Disconnect WebSocket
    if (this.wsClient) {
      this.wsClient.destroy();
    }

    // Destroy components
    if (this.timeline) {
      this.timeline.destroy();
    }

    if (this.dashboardLayout) {
      this.dashboardLayout.destroy();
    }

    // Clear data
    this.data = null;
    this.charts.clear();
    this.widgets.clear();
  }
}

// Make globally available for onclick handlers
if (typeof window !== "undefined") {
  window.dashboard = null;
}

export default RealTimeDashboard;
