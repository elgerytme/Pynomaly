/**
 * ECharts Demo Application
 * Interactive demonstration of all ECharts dashboard components
 */

class EChartsDemo {
  constructor() {
    this.charts = new Map();
    this.demoData = this.generateDemoData();
    this.realTimeInterval = null;
    this.isRealTimeEnabled = false;

    this.init();
  }

  init() {
    this.setupDemoControls();
    this.createAllDemos();
    this.setupEventListeners();
  }

  generateDemoData() {
    const now = new Date();

    // Performance metrics data
    const performanceData = [];
    for (let i = 0; i < 50; i++) {
      const timestamp = new Date(now.getTime() - (50 - i) * 60000);
      performanceData.push({
        timestamp: timestamp.toISOString(),
        cpu: Math.random() * 80 + 10,
        memory: Math.random() * 70 + 20,
        network: Math.random() * 50 + 5,
        disk: Math.random() * 60 + 10,
      });
    }

    // Anomaly distribution data
    const anomalyTypes = [
      "Statistical Outlier",
      "Temporal Anomaly",
      "Pattern Deviation",
      "Threshold Violation",
      "Trend Anomaly",
    ];

    const anomalyData = [];
    for (let i = 0; i < 100; i++) {
      const confidence = Math.random();
      anomalyData.push({
        id: i,
        type: anomalyTypes[Math.floor(Math.random() * anomalyTypes.length)],
        confidence: confidence,
        score: confidence,
        timestamp: new Date(
          now.getTime() - Math.random() * 86400000 * 7,
        ).toISOString(),
        severity:
          confidence > 0.8 ? "high" : confidence > 0.5 ? "medium" : "low",
      });
    }

    // Detection timeline data
    const timelineData = [];
    for (let i = 0; i < 200; i++) {
      const confidence = Math.random();
      timelineData.push({
        timestamp: new Date(
          now.getTime() - Math.random() * 86400000,
        ).toISOString(),
        confidence: confidence,
        score: confidence,
        type: anomalyTypes[Math.floor(Math.random() * anomalyTypes.length)],
      });
    }

    return {
      performance: performanceData,
      anomalies: anomalyData,
      timeline: timelineData,
    };
  }

  setupDemoControls() {
    const controlsContainer = document.getElementById("echarts-demo-controls");
    if (!controlsContainer) return;

    controlsContainer.innerHTML = `
      <div class="demo-controls-grid">
        <div class="control-group">
          <h3>Real-Time Simulation</h3>
          <button id="echarts-realtime-toggle" class="btn btn-primary">Start Real-Time</button>
          <label>
            Update Interval:
            <select id="echarts-update-interval">
              <option value="1000">1 second</option>
              <option value="2000" selected>2 seconds</option>
              <option value="5000">5 seconds</option>
            </select>
          </label>
        </div>

        <div class="control-group">
          <h3>Theme</h3>
          <button id="echarts-theme-toggle" class="btn btn-secondary">Switch to Dark</button>
          <label>
            <input type="checkbox" id="echarts-high-contrast" /> High Contrast
          </label>
        </div>

        <div class="control-group">
          <h3>Chart Controls</h3>
          <button id="echarts-refresh-data" class="btn btn-secondary">Refresh Data</button>
          <button id="echarts-export-charts" class="btn btn-secondary">Export Charts</button>
        </div>

        <div class="control-group">
          <h3>Performance</h3>
          <label>
            Metrics:
            <select id="performance-metrics" multiple>
              <option value="cpu" selected>CPU</option>
              <option value="memory" selected>Memory</option>
              <option value="network" selected>Network</option>
              <option value="disk">Disk I/O</option>
            </select>
          </label>
        </div>

        <div class="control-group">
          <h3>Anomaly Charts</h3>
          <label>
            Distribution Type:
            <select id="distribution-type">
              <option value="pie" selected>Pie Chart</option>
              <option value="bar">Bar Chart</option>
              <option value="histogram">Histogram</option>
            </select>
          </label>
        </div>

        <div class="control-group">
          <h3>Timeline</h3>
          <label>
            Time Range:
            <select id="timeline-range">
              <option value="1h">1 Hour</option>
              <option value="6h">6 Hours</option>
              <option value="24h" selected>24 Hours</option>
              <option value="7d">7 Days</option>
              <option value="30d">30 Days</option>
            </select>
          </label>
        </div>
      </div>

      <div id="echarts-announcer" aria-live="polite" class="sr-only"></div>
    `;
  }

  createAllDemos() {
    this.createPerformanceDemo();
    this.createAnomalyDistributionDemo();
    this.createTimelineDemo();
    this.createDashboardDemo();
  }

  createPerformanceDemo() {
    const container = document.getElementById("performance-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Performance Metrics Chart</h2>
        <p>Real-time system performance monitoring with CPU, memory, network, and disk utilization metrics.</p>

        <div class="chart-controls">
          <label>
            <input type="checkbox" id="performance-animation" checked /> Enable Animations
          </label>
          <label>
            Max Data Points:
            <select id="performance-max-points">
              <option value="50">50 points</option>
              <option value="100" selected>100 points</option>
              <option value="200">200 points</option>
            </select>
          </label>
          <button id="performance-reset" class="btn btn-sm">Reset Data</button>
        </div>

        <div id="performance-chart" class="chart-container" style="height: 400px;"></div>

        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Real-time data streaming</li>
            <li>Multiple metric tracking</li>
            <li>Smooth line animations</li>
            <li>Interactive tooltips</li>
            <li>Responsive design</li>
            <li>Alert notifications</li>
          </ul>
        </div>
      </div>
    `;

    // Initialize chart
    const chart = new PerformanceMetricsChart("#performance-chart", {
      title: "System Performance Metrics",
      description:
        "Real-time monitoring of system CPU, memory, and network utilization",
      metrics: ["cpu", "memory", "network"],
      updateInterval: 0, // We'll control updates manually
      maxDataPoints: 100,
      animation: true,
    });

    chart.setData(this.demoData.performance);
    this.charts.set("performance", chart);

    // Setup controls
    this.setupPerformanceControls();
  }

  setupPerformanceControls() {
    document
      .getElementById("performance-animation")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("performance");
        if (chart) {
          chart.options.animation = e.target.checked;
          chart.updateChart();
        }
      });

    document
      .getElementById("performance-max-points")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("performance");
        if (chart) {
          chart.options.maxDataPoints = parseInt(e.target.value);
          chart.setData(
            this.demoData.performance.slice(-chart.options.maxDataPoints),
          );
        }
      });

    document
      .getElementById("performance-reset")
      ?.addEventListener("click", () => {
        this.demoData.performance = this.generateDemoData().performance;
        const chart = this.charts.get("performance");
        if (chart) {
          chart.setData(this.demoData.performance);
        }
      });

    document
      .getElementById("performance-metrics")
      ?.addEventListener("change", (e) => {
        const selectedMetrics = Array.from(e.target.selectedOptions).map(
          (option) => option.value,
        );
        const chart = this.charts.get("performance");
        if (chart) {
          chart.options.metrics = selectedMetrics;
          chart.updateChart();
        }
      });
  }

  createAnomalyDistributionDemo() {
    const container = document.getElementById("anomaly-distribution-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Anomaly Distribution Analysis</h2>
        <p>Statistical visualization of detected anomalies by type, confidence level, and distribution patterns.</p>

        <div class="chart-controls">
          <label>
            Chart Type:
            <select id="distribution-chart-type">
              <option value="pie" selected>Pie Chart</option>
              <option value="bar">Bar Chart</option>
              <option value="histogram">Histogram</option>
            </select>
          </label>
          <label>
            <input type="checkbox" id="distribution-animation" checked /> Enable Animations
          </label>
          <button id="distribution-randomize" class="btn btn-sm">Randomize Data</button>
        </div>

        <div id="anomaly-distribution-chart" class="chart-container" style="height: 400px;"></div>

        <div class="stats-panel">
          <h4>Statistics:</h4>
          <div id="distribution-stats">
            <div class="stat-item">
              <span class="stat-label">Total Anomalies:</span>
              <span class="stat-value" id="total-anomalies-count">-</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Average Confidence:</span>
              <span class="stat-value" id="avg-confidence-value">-</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">High Confidence:</span>
              <span class="stat-value" id="high-confidence-count">-</span>
            </div>
          </div>
        </div>

        <div class="chart-info">
          <h4>Chart Types:</h4>
          <ul>
            <li><strong>Pie Chart</strong> - Distribution by anomaly type</li>
            <li><strong>Bar Chart</strong> - Confidence level ranges</li>
            <li><strong>Histogram</strong> - Score distribution density</li>
          </ul>
        </div>
      </div>
    `;

    // Initialize chart
    const chart = new AnomalyDistributionChart("#anomaly-distribution-chart", {
      title: "Anomaly Distribution by Type",
      description:
        "Pie chart showing distribution of anomaly types detected in the system",
      chartType: "pie",
      animation: true,
    });

    chart.setData(this.demoData.anomalies);
    this.charts.set("distribution", chart);

    // Update statistics
    this.updateDistributionStats();

    // Setup controls
    this.setupDistributionControls();
  }

  setupDistributionControls() {
    document
      .getElementById("distribution-chart-type")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("distribution");
        if (chart) {
          chart.options.chartType = e.target.value;

          // Update title based on chart type
          const titles = {
            pie: "Anomaly Distribution by Type",
            bar: "Anomalies by Confidence Level",
            histogram: "Anomaly Score Distribution",
          };
          chart.options.title = titles[e.target.value];
          chart.updateChart();
        }
      });

    document
      .getElementById("distribution-animation")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("distribution");
        if (chart) {
          chart.options.animation = e.target.checked;
          chart.updateChart();
        }
      });

    document
      .getElementById("distribution-randomize")
      ?.addEventListener("click", () => {
        this.demoData.anomalies = this.generateDemoData().anomalies;
        const chart = this.charts.get("distribution");
        if (chart) {
          chart.setData(this.demoData.anomalies);
          this.updateDistributionStats();
        }
      });

    document
      .getElementById("distribution-type")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("distribution");
        if (chart) {
          chart.options.chartType = e.target.value;
          chart.updateChart();
        }
      });
  }

  updateDistributionStats() {
    const data = this.demoData.anomalies;
    const total = data.length;
    const avgConfidence =
      data.reduce((sum, d) => sum + d.confidence, 0) / total;
    const highConfidence = data.filter((d) => d.confidence > 0.8).length;

    document.getElementById("total-anomalies-count").textContent = total;
    document.getElementById("avg-confidence-value").textContent =
      (avgConfidence * 100).toFixed(1) + "%";
    document.getElementById("high-confidence-count").textContent =
      highConfidence;
  }

  createTimelineDemo() {
    const container = document.getElementById("timeline-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Detection Timeline</h2>
        <p>Chronological visualization of anomaly detection events with severity indicators and time-based analysis.</p>

        <div class="chart-controls">
          <label>
            Time Range:
            <select id="timeline-time-range">
              <option value="1h">1 Hour</option>
              <option value="6h">6 Hours</option>
              <option value="24h" selected>24 Hours</option>
              <option value="7d">7 Days</option>
              <option value="30d">30 Days</option>
            </select>
          </label>
          <label>
            <input type="checkbox" id="timeline-severity" checked /> Show Severity Levels
          </label>
          <button id="timeline-refresh" class="btn btn-sm">Refresh Timeline</button>
        </div>

        <div id="detection-timeline-chart" class="chart-container" style="height: 400px;"></div>

        <div class="timeline-summary">
          <h4>Timeline Summary:</h4>
          <div id="timeline-summary-content">
            <div class="summary-item">
              <span class="severity-indicator critical"></span>
              <span>Critical: <strong id="critical-count">0</strong></span>
            </div>
            <div class="summary-item">
              <span class="severity-indicator high"></span>
              <span>High: <strong id="high-count">0</strong></span>
            </div>
            <div class="summary-item">
              <span class="severity-indicator medium"></span>
              <span>Medium: <strong id="medium-count">0</strong></span>
            </div>
            <div class="summary-item">
              <span class="severity-indicator low"></span>
              <span>Low: <strong id="low-count">0</strong></span>
            </div>
          </div>
        </div>

        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Multiple time range options</li>
            <li>Severity level stacking</li>
            <li>Interactive timeline navigation</li>
            <li>Real-time updates</li>
            <li>Trend analysis</li>
          </ul>
        </div>
      </div>
    `;

    // Initialize chart
    const chart = new DetectionTimelineChart("#detection-timeline-chart", {
      title: "Anomaly Detection Timeline - 24 Hours",
      description:
        "Stacked bar chart showing anomaly detection events over time by severity level",
      timeRange: "24h",
      showSeverityLevels: true,
      animation: true,
    });

    chart.setData(this.demoData.timeline);
    this.charts.set("timeline", chart);

    // Update summary
    this.updateTimelineSummary();

    // Setup controls
    this.setupTimelineControls();
  }

  setupTimelineControls() {
    document
      .getElementById("timeline-time-range")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("timeline");
        if (chart) {
          chart.options.timeRange = e.target.value;
          chart.options.title = `Anomaly Detection Timeline - ${e.target.value.toUpperCase()}`;
          chart.updateChart();
        }
      });

    document
      .getElementById("timeline-severity")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("timeline");
        if (chart) {
          chart.options.showSeverityLevels = e.target.checked;
          chart.updateChart();
        }
      });

    document
      .getElementById("timeline-refresh")
      ?.addEventListener("click", () => {
        this.demoData.timeline = this.generateDemoData().timeline;
        const chart = this.charts.get("timeline");
        if (chart) {
          chart.setData(this.demoData.timeline);
          this.updateTimelineSummary();
        }
      });

    document
      .getElementById("timeline-range")
      ?.addEventListener("change", (e) => {
        const chart = this.charts.get("timeline");
        if (chart) {
          chart.options.timeRange = e.target.value;
          chart.updateChart();
        }
      });
  }

  updateTimelineSummary() {
    const data = this.demoData.timeline;
    const counts = { critical: 0, high: 0, medium: 0, low: 0 };

    data.forEach((item) => {
      const confidence = item.confidence || 0;
      if (confidence >= 0.95) counts.critical++;
      else if (confidence >= 0.8) counts.high++;
      else if (confidence >= 0.5) counts.medium++;
      else counts.low++;
    });

    document.getElementById("critical-count").textContent = counts.critical;
    document.getElementById("high-count").textContent = counts.high;
    document.getElementById("medium-count").textContent = counts.medium;
    document.getElementById("low-count").textContent = counts.low;
  }

  createDashboardDemo() {
    const container = document.getElementById("echarts-dashboard-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Integrated Dashboard</h2>
        <p>Comprehensive anomaly detection dashboard combining multiple chart types for complete system overview.</p>

        <div class="dashboard-grid">
          <div class="dashboard-widget">
            <h3>Performance Overview</h3>
            <div id="dashboard-performance" class="mini-chart" style="height: 200px;"></div>
          </div>

          <div class="dashboard-widget">
            <h3>Anomaly Types</h3>
            <div id="dashboard-distribution" class="mini-chart" style="height: 200px;"></div>
          </div>

          <div class="dashboard-widget">
            <h3>Detection Events</h3>
            <div id="dashboard-timeline" class="mini-chart" style="height: 200px;"></div>
          </div>

          <div class="dashboard-widget stats-summary">
            <h3>Key Metrics</h3>
            <div class="metrics-grid">
              <div class="metric">
                <div class="metric-value" id="dashboard-total-anomalies">0</div>
                <div class="metric-label">Total Anomalies</div>
              </div>
              <div class="metric">
                <div class="metric-value" id="dashboard-avg-confidence">0%</div>
                <div class="metric-label">Avg Confidence</div>
              </div>
              <div class="metric">
                <div class="metric-value" id="dashboard-critical-alerts">0</div>
                <div class="metric-label">Critical Alerts</div>
              </div>
              <div class="metric">
                <div class="metric-value" id="dashboard-system-health">Good</div>
                <div class="metric-label">System Health</div>
              </div>
            </div>
          </div>
        </div>

        <div class="dashboard-controls">
          <button id="dashboard-refresh" class="btn btn-primary">Refresh Dashboard</button>
          <button id="dashboard-export" class="btn btn-secondary">Export Data</button>
          <label>
            <input type="checkbox" id="dashboard-realtime" /> Real-time Updates
          </label>
        </div>
      </div>
    `;

    // Initialize mini charts
    this.createDashboardCharts();
    this.updateDashboardMetrics();
  }

  createDashboardCharts() {
    // Mini performance chart
    const perfChart = new PerformanceMetricsChart("#dashboard-performance", {
      title: "",
      metrics: ["cpu", "memory"],
      updateInterval: 0,
      maxDataPoints: 20,
      animation: false,
    });
    perfChart.setData(this.demoData.performance.slice(-20));
    this.charts.set("dashboard-performance", perfChart);

    // Mini distribution chart
    const distChart = new AnomalyDistributionChart("#dashboard-distribution", {
      title: "",
      chartType: "pie",
      animation: false,
    });
    distChart.setData(this.demoData.anomalies.slice(0, 50));
    this.charts.set("dashboard-distribution", distChart);

    // Mini timeline chart
    const timelineChart = new DetectionTimelineChart("#dashboard-timeline", {
      title: "",
      timeRange: "6h",
      animation: false,
    });
    timelineChart.setData(this.demoData.timeline.slice(-50));
    this.charts.set("dashboard-timeline", timelineChart);

    // Setup dashboard controls
    this.setupDashboardControls();
  }

  setupDashboardControls() {
    document
      .getElementById("dashboard-refresh")
      ?.addEventListener("click", () => {
        this.refreshDashboard();
      });

    document
      .getElementById("dashboard-export")
      ?.addEventListener("click", () => {
        this.exportDashboardData();
      });

    document
      .getElementById("dashboard-realtime")
      ?.addEventListener("change", (e) => {
        if (e.target.checked) {
          this.startDashboardRealTime();
        } else {
          this.stopDashboardRealTime();
        }
      });
  }

  updateDashboardMetrics() {
    const totalAnomalies = this.demoData.anomalies.length;
    const avgConfidence =
      this.demoData.anomalies.reduce((sum, d) => sum + d.confidence, 0) /
      totalAnomalies;
    const criticalAlerts = this.demoData.anomalies.filter(
      (d) => d.confidence > 0.95,
    ).length;
    const systemHealth =
      criticalAlerts > 5 ? "Critical" : criticalAlerts > 2 ? "Warning" : "Good";

    document.getElementById("dashboard-total-anomalies").textContent =
      totalAnomalies;
    document.getElementById("dashboard-avg-confidence").textContent =
      (avgConfidence * 100).toFixed(1) + "%";
    document.getElementById("dashboard-critical-alerts").textContent =
      criticalAlerts;
    document.getElementById("dashboard-system-health").textContent =
      systemHealth;
  }

  refreshDashboard() {
    this.demoData = this.generateDemoData();

    // Update all dashboard charts
    const perfChart = this.charts.get("dashboard-performance");
    if (perfChart) {
      perfChart.setData(this.demoData.performance.slice(-20));
    }

    const distChart = this.charts.get("dashboard-distribution");
    if (distChart) {
      distChart.setData(this.demoData.anomalies.slice(0, 50));
    }

    const timelineChart = this.charts.get("dashboard-timeline");
    if (timelineChart) {
      timelineChart.setData(this.demoData.timeline.slice(-50));
    }

    this.updateDashboardMetrics();
    this.announceToScreenReader("Dashboard refreshed with new data");
  }

  startDashboardRealTime() {
    if (this.dashboardTimer) return;

    this.dashboardTimer = setInterval(() => {
      // Add new performance data
      const now = new Date();
      const newPerfPoint = {
        timestamp: now.toISOString(),
        cpu: Math.random() * 80 + 10,
        memory: Math.random() * 70 + 20,
        network: Math.random() * 50 + 5,
      };

      this.demoData.performance.push(newPerfPoint);
      this.demoData.performance = this.demoData.performance.slice(-100);

      // Update performance chart
      const perfChart = this.charts.get("dashboard-performance");
      if (perfChart) {
        perfChart.setData(this.demoData.performance.slice(-20));
      }

      // Occasionally add new anomaly
      if (Math.random() < 0.1) {
        const newAnomaly = {
          id: Date.now(),
          type: [
            "Statistical Outlier",
            "Temporal Anomaly",
            "Pattern Deviation",
          ][Math.floor(Math.random() * 3)],
          confidence: Math.random(),
          timestamp: now.toISOString(),
        };

        this.demoData.anomalies.push(newAnomaly);
        this.demoData.timeline.push(newAnomaly);

        this.updateDashboardMetrics();
      }
    }, 3000);
  }

  stopDashboardRealTime() {
    if (this.dashboardTimer) {
      clearInterval(this.dashboardTimer);
      this.dashboardTimer = null;
    }
  }

  exportDashboardData() {
    const exportData = {
      timestamp: new Date().toISOString(),
      performance: this.demoData.performance,
      anomalies: this.demoData.anomalies,
      timeline: this.demoData.timeline,
      metrics: {
        totalAnomalies: this.demoData.anomalies.length,
        avgConfidence:
          this.demoData.anomalies.reduce((sum, d) => sum + d.confidence, 0) /
          this.demoData.anomalies.length,
        criticalAlerts: this.demoData.anomalies.filter(
          (d) => d.confidence > 0.95,
        ).length,
      },
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `echarts-dashboard-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.announceToScreenReader("Dashboard data exported successfully");
  }

  setupEventListeners() {
    // Real-time toggle
    document
      .getElementById("echarts-realtime-toggle")
      ?.addEventListener("click", (e) => {
        this.toggleRealTime();
        e.target.textContent = this.isRealTimeEnabled
          ? "Stop Real-Time"
          : "Start Real-Time";
      });

    // Theme toggle
    document
      .getElementById("echarts-theme-toggle")
      ?.addEventListener("click", (e) => {
        const newTheme =
          echartsManager.currentTheme === "light" ? "dark" : "light";
        this.switchTheme(newTheme);
        e.target.textContent = `Switch to ${newTheme === "light" ? "Dark" : "Light"}`;
      });

    // Refresh data
    document
      .getElementById("echarts-refresh-data")
      ?.addEventListener("click", () => {
        this.refreshAllData();
      });

    // Export charts
    document
      .getElementById("echarts-export-charts")
      ?.addEventListener("click", () => {
        this.exportCharts();
      });
  }

  toggleRealTime() {
    if (this.isRealTimeEnabled) {
      clearInterval(this.realTimeInterval);
      this.isRealTimeEnabled = false;
    } else {
      const interval = parseInt(
        document.getElementById("echarts-update-interval")?.value || "2000",
      );
      this.realTimeInterval = setInterval(() => {
        this.updateRealTimeData();
      }, interval);
      this.isRealTimeEnabled = true;
    }
  }

  updateRealTimeData() {
    // Update performance chart
    const perfChart = this.charts.get("performance");
    if (perfChart) {
      perfChart.addRandomDataPoint();
    }

    // Occasionally add new anomaly
    if (Math.random() < 0.05) {
      const now = new Date();
      const newAnomaly = {
        id: Date.now(),
        type: ["Statistical Outlier", "Temporal Anomaly", "Pattern Deviation"][
          Math.floor(Math.random() * 3)
        ],
        confidence: Math.random(),
        timestamp: now.toISOString(),
      };

      this.demoData.anomalies.push(newAnomaly);
      this.demoData.timeline.push(newAnomaly);

      // Update distribution chart
      const distChart = this.charts.get("distribution");
      if (distChart) {
        distChart.setData(this.demoData.anomalies);
        this.updateDistributionStats();
      }

      // Update timeline chart
      const timelineChart = this.charts.get("timeline");
      if (timelineChart) {
        timelineChart.setData(this.demoData.timeline);
        this.updateTimelineSummary();
      }

      if (newAnomaly.confidence > 0.9) {
        this.announceToScreenReader(
          `High confidence anomaly detected: ${newAnomaly.type}, confidence ${(newAnomaly.confidence * 100).toFixed(1)}%`,
        );
      }
    }
  }

  switchTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);

    // Emit theme change event
    document.dispatchEvent(
      new CustomEvent("theme-changed", {
        detail: { theme },
      }),
    );
  }

  refreshAllData() {
    this.demoData = this.generateDemoData();

    this.charts.forEach((chart, key) => {
      if (key.includes("performance")) {
        chart.setData(this.demoData.performance);
      } else if (key.includes("distribution")) {
        chart.setData(this.demoData.anomalies);
        this.updateDistributionStats();
      } else if (key.includes("timeline")) {
        chart.setData(this.demoData.timeline);
        this.updateTimelineSummary();
      }
    });

    this.updateDashboardMetrics();
    this.announceToScreenReader("All chart data refreshed");
  }

  exportCharts() {
    const exportData = {
      timestamp: new Date().toISOString(),
      chartTypes: ["performance", "distribution", "timeline"],
      data: this.demoData,
      chartConfigurations: {},
    };

    this.charts.forEach((chart, key) => {
      exportData.chartConfigurations[key] = {
        type: chart.constructor.name,
        options: chart.options,
      };
    });

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `echarts-demo-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.announceToScreenReader("ECharts data exported successfully");
  }

  announceToScreenReader(message) {
    const announcer = document.getElementById("echarts-announcer");
    if (announcer) {
      announcer.textContent = message;
    }
  }

  destroy() {
    if (this.realTimeInterval) {
      clearInterval(this.realTimeInterval);
    }

    if (this.dashboardTimer) {
      clearInterval(this.dashboardTimer);
    }

    this.charts.forEach((chart) => {
      chart.dispose();
    });

    this.charts.clear();
  }
}

// Auto-initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  if (document.querySelector(".echarts-demo")) {
    window.echartsDemo = new EChartsDemo();
  }
});

// Export for module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = EChartsDemo;
} else {
  window.EChartsDemo = EChartsDemo;
}
