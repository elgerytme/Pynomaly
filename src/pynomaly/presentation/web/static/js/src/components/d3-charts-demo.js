/**
 * D3.js Chart Library Demo and Usage Examples
 * Interactive demonstrations of all chart types with real-time data
 */

class D3ChartsDemo {
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
    const timeSeriesData = [];
    const scatterData = [];
    const heatmapData = [];

    // Generate time series data with anomalies
    for (let i = 0; i < 100; i++) {
      const timestamp = new Date(now.getTime() - (100 - i) * 60000);
      const baseValue = 50 + 20 * Math.sin(i * 0.1) + Math.random() * 10;

      // Inject some anomalies
      const isAnomaly = Math.random() < 0.05;
      const value = isAnomaly
        ? baseValue + (Math.random() - 0.5) * 80
        : baseValue;
      const confidence = isAnomaly
        ? 0.7 + Math.random() * 0.3
        : 0.1 + Math.random() * 0.3;

      timeSeriesData.push({
        timestamp: timestamp,
        value: value,
        isAnomaly: isAnomaly,
        confidence: confidence,
      });
    }

    // Generate scatter plot data
    for (let i = 0; i < 200; i++) {
      const x = Math.random() * 100;
      const y = Math.random() * 100;

      // Create some clusters and anomalies
      const isCluster1 = Math.sqrt((x - 25) ** 2 + (y - 25) ** 2) < 15;
      const isCluster2 = Math.sqrt((x - 75) ** 2 + (y - 75) ** 2) < 15;
      const isAnomaly = !isCluster1 && !isCluster2 && Math.random() < 0.1;

      const anomalyScore = isAnomaly
        ? 0.7 + Math.random() * 0.3
        : Math.random() * 0.3;
      const confidence = isAnomaly
        ? 0.8 + Math.random() * 0.2
        : 0.3 + Math.random() * 0.4;

      scatterData.push({
        x: x,
        y: y,
        anomalyScore: anomalyScore,
        confidence: confidence,
        isAnomaly: isAnomaly,
      });
    }

    // Generate heatmap data
    const features = ["CPU", "Memory", "Disk", "Network", "Response Time"];
    const timeSlots = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"];

    for (let feature of features) {
      for (let time of timeSlots) {
        heatmapData.push({
          x: time,
          y: feature,
          value: Math.random(),
        });
      }
    }

    return {
      timeSeries: timeSeriesData,
      scatter: scatterData,
      heatmap: heatmapData,
    };
  }

  setupDemoControls() {
    const controlsContainer = document.getElementById("demo-controls");
    if (!controlsContainer) return;

    controlsContainer.innerHTML = `
      <div class="demo-controls-grid">
        <div class="control-group">
          <h3>Real-Time Simulation</h3>
          <button id="realtime-toggle" class="btn btn-primary">Start Real-Time</button>
          <label>
            Update Interval:
            <select id="update-interval">
              <option value="1000">1 second</option>
              <option value="2000" selected>2 seconds</option>
              <option value="5000">5 seconds</option>
            </select>
          </label>
        </div>
        
        <div class="control-group">
          <h3>Theme</h3>
          <button id="theme-toggle" class="btn btn-secondary">Switch to Dark</button>
        </div>
        
        <div class="control-group">
          <h3>Chart Controls</h3>
          <button id="refresh-data" class="btn btn-secondary">Refresh Data</button>
          <button id="export-charts" class="btn btn-secondary">Export Charts</button>
        </div>
        
        <div class="control-group">
          <h3>Accessibility</h3>
          <button id="announce-data" class="btn btn-secondary">Announce Data</button>
          <label>
            <input type="checkbox" id="high-contrast" /> High Contrast
          </label>
        </div>
      </div>
      
      <div id="chart-announcer" aria-live="polite" class="sr-only"></div>
    `;
  }

  createAllDemos() {
    this.createTimeSeriesDemo();
    this.createScatterPlotDemo();
    this.createHeatmapDemo();
    this.createInteractiveDemo();
  }

  createTimeSeriesDemo() {
    const container = document.getElementById("timeseries-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Time Series Chart - Anomaly Detection Timeline</h2>
        <p>Interactive time series visualization showing anomaly detection results over time with real-time updates.</p>
        
        <div class="chart-controls">
          <label>
            <input type="checkbox" id="show-confidence" checked /> Show Confidence Bands
          </label>
          <label>
            <input type="checkbox" id="show-anomalies" checked /> Show Anomaly Markers
          </label>
          <button id="zoom-reset-ts" class="btn btn-sm">Reset Zoom</button>
        </div>
        
        <div id="timeseries-chart" class="chart-container"></div>
        
        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Real-time data updates</li>
            <li>Interactive tooltips</li>
            <li>Anomaly highlighting</li>
            <li>Confidence bands</li>
            <li>Keyboard navigation</li>
            <li>Screen reader support</li>
          </ul>
        </div>
      </div>
    `;

    const chart = new TimeSeriesChart("#timeseries-chart", {
      title: "Anomaly Detection Timeline",
      description:
        "Time series chart showing detected anomalies with confidence intervals",
      showConfidenceBands: true,
      animated: true,
      responsive: true,
    });

    chart.setData(this.demoData.timeSeries);
    this.charts.set("timeseries", chart);

    // Setup controls
    document
      .getElementById("show-confidence")
      ?.addEventListener("change", (e) => {
        chart.options.showConfidenceBands = e.target.checked;
        chart.render();
      });

    document
      .getElementById("show-anomalies")
      ?.addEventListener("change", (e) => {
        chart.options.showAnomalies = e.target.checked;
        chart.render();
      });

    // Setup chart event listeners
    container.addEventListener("anomaly-selected", (e) => {
      const { data } = e.detail;
      this.showAnomalyDetails(data, "Time Series");
    });
  }

  createScatterPlotDemo() {
    const container = document.getElementById("scatter-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Scatter Plot - 2D Anomaly Detection</h2>
        <p>Interactive scatter plot for detecting anomalies in two-dimensional data space with brushing and zoom.</p>
        
        <div class="chart-controls">
          <label>
            <input type="checkbox" id="enable-brushing" checked /> Enable Brushing
          </label>
          <label>
            <input type="checkbox" id="enable-zoom" checked /> Enable Zoom
          </label>
          <button id="clear-selection" class="btn btn-sm">Clear Selection</button>
        </div>
        
        <div id="scatter-chart" class="chart-container"></div>
        
        <div class="selection-info">
          <h4>Selection Info:</h4>
          <div id="selection-details">No points selected</div>
        </div>
        
        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Brush selection</li>
            <li>Zoom and pan</li>
            <li>Color-coded anomaly scores</li>
            <li>Size-coded confidence</li>
            <li>Interactive legends</li>
          </ul>
        </div>
      </div>
    `;

    const chart = new ScatterPlotChart("#scatter-chart", {
      title: "2D Anomaly Detection",
      description:
        "Scatter plot showing anomalies in two-dimensional feature space",
      xLabel: "Feature 1",
      yLabel: "Feature 2",
      enableBrushing: true,
      enableZoom: true,
      animated: true,
    });

    chart.setData(this.demoData.scatter);
    this.charts.set("scatter", chart);

    // Setup controls
    document
      .getElementById("enable-brushing")
      ?.addEventListener("change", (e) => {
        chart.options.enableBrushing = e.target.checked;
        chart.render();
      });

    document.getElementById("enable-zoom")?.addEventListener("change", (e) => {
      chart.options.enableZoom = e.target.checked;
      chart.render();
    });

    // Setup chart event listeners
    container.addEventListener("points-selected", (e) => {
      const { data } = e.detail;
      this.updateSelectionInfo(data);
    });

    container.addEventListener("point-selected", (e) => {
      const { data } = e.detail;
      this.showAnomalyDetails(data, "Scatter Plot");
    });
  }

  createHeatmapDemo() {
    const container = document.getElementById("heatmap-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Heatmap - Feature Correlation Matrix</h2>
        <p>Interactive heatmap showing correlations between features and anomaly densities across time periods.</p>
        
        <div class="chart-controls">
          <label>
            <input type="checkbox" id="show-labels" checked /> Show Value Labels
          </label>
          <select id="color-scheme">
            <option value="interpolateViridis">Viridis</option>
            <option value="interpolatePlasma">Plasma</option>
            <option value="interpolateInferno">Inferno</option>
            <option value="interpolateTurbo">Turbo</option>
          </select>
        </div>
        
        <div id="heatmap-chart" class="chart-container"></div>
        
        <div class="chart-info">
          <h4>Features:</h4>
          <ul>
            <li>Interactive cells</li>
            <li>Color legends</li>
            <li>Value labels</li>
            <li>Multiple color schemes</li>
            <li>Responsive design</li>
          </ul>
        </div>
      </div>
    `;

    const chart = new HeatmapChart("#heatmap-chart", {
      title: "Feature Correlation Heatmap",
      description:
        "Heatmap showing correlation values between different system features",
      showLabels: true,
      animated: true,
    });

    chart.setData(this.demoData.heatmap);
    this.charts.set("heatmap", chart);

    // Setup controls
    document.getElementById("show-labels")?.addEventListener("change", (e) => {
      chart.options.showLabels = e.target.checked;
      chart.render();
    });

    document.getElementById("color-scheme")?.addEventListener("change", (e) => {
      chart.options.colorScheme = d3[e.target.value];
      chart.render();
    });

    // Setup chart event listeners
    container.addEventListener("cell-selected", (e) => {
      const { data } = e.detail;
      this.showCellDetails(data);
    });
  }

  createInteractiveDemo() {
    const container = document.getElementById("interactive-demo");
    if (!container) return;

    container.innerHTML = `
      <div class="demo-section">
        <h2>Interactive Dashboard Demo</h2>
        <p>Combined visualization dashboard showing real-time anomaly detection across multiple chart types.</p>
        
        <div class="dashboard-grid">
          <div class="dashboard-item">
            <h3>Real-Time Stream</h3>
            <div id="realtime-chart" class="mini-chart"></div>
          </div>
          
          <div class="dashboard-item">
            <h3>Anomaly Distribution</h3>
            <div id="distribution-chart" class="mini-chart"></div>
          </div>
          
          <div class="dashboard-item">
            <h3>Feature Correlation</h3>
            <div id="correlation-chart" class="mini-chart"></div>
          </div>
          
          <div class="dashboard-item stats-panel">
            <h3>Statistics</h3>
            <div id="stats-content">
              <div class="stat-item">
                <span class="stat-label">Total Anomalies:</span>
                <span class="stat-value" id="total-anomalies">-</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">Avg Confidence:</span>
                <span class="stat-value" id="avg-confidence">-</span>
              </div>
              <div class="stat-item">
                <span class="stat-label">Last Updated:</span>
                <span class="stat-value" id="last-updated">-</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    // Create mini charts for dashboard
    const realtimeChart = new TimeSeriesChart("#realtime-chart", {
      title: "Real-time Anomaly Stream",
      height: 200,
      margin: { top: 10, right: 20, bottom: 30, left: 40 },
      showConfidenceBands: false,
      animated: false,
    });

    const distributionChart = new ScatterPlotChart("#distribution-chart", {
      title: "Anomaly Distribution",
      height: 200,
      margin: { top: 10, right: 20, bottom: 30, left: 40 },
      enableBrushing: false,
      enableZoom: false,
      animated: false,
    });

    const correlationChart = new HeatmapChart("#correlation-chart", {
      title: "Feature Correlation",
      height: 200,
      margin: { top: 10, right: 20, bottom: 30, left: 40 },
      showLabels: false,
      animated: false,
    });

    // Set initial data
    realtimeChart.setData(this.demoData.timeSeries.slice(-20));
    distributionChart.setData(this.demoData.scatter.slice(0, 50));
    correlationChart.setData(this.demoData.heatmap);

    this.charts.set("realtime", realtimeChart);
    this.charts.set("distribution", distributionChart);
    this.charts.set("correlation", correlationChart);

    this.updateStatistics();
  }

  setupEventListeners() {
    // Real-time toggle
    document
      .getElementById("realtime-toggle")
      ?.addEventListener("click", (e) => {
        this.toggleRealTime();
        e.target.textContent = this.isRealTimeEnabled
          ? "Stop Real-Time"
          : "Start Real-Time";
      });

    // Theme toggle
    document.getElementById("theme-toggle")?.addEventListener("click", (e) => {
      const newTheme = chartLibrary.currentTheme === "light" ? "dark" : "light";
      this.switchTheme(newTheme);
      e.target.textContent = `Switch to ${newTheme === "light" ? "Dark" : "Light"}`;
    });

    // Refresh data
    document.getElementById("refresh-data")?.addEventListener("click", () => {
      this.refreshAllData();
    });

    // Export charts
    document.getElementById("export-charts")?.addEventListener("click", () => {
      this.exportCharts();
    });

    // Announce data
    document.getElementById("announce-data")?.addEventListener("click", () => {
      this.announceDataSummary();
    });

    // High contrast
    document
      .getElementById("high-contrast")
      ?.addEventListener("change", (e) => {
        document.body.classList.toggle("high-contrast", e.target.checked);
      });
  }

  toggleRealTime() {
    if (this.isRealTimeEnabled) {
      clearInterval(this.realTimeInterval);
      this.isRealTimeEnabled = false;
    } else {
      const interval = parseInt(
        document.getElementById("update-interval")?.value || "2000",
      );
      this.realTimeInterval = setInterval(() => {
        this.updateRealTimeData();
      }, interval);
      this.isRealTimeEnabled = true;
    }
  }

  updateRealTimeData() {
    // Add new data point to time series
    const lastPoint =
      this.demoData.timeSeries[this.demoData.timeSeries.length - 1];
    const now = new Date();
    const baseValue =
      50 + 20 * Math.sin(Date.now() * 0.0001) + Math.random() * 10;
    const isAnomaly = Math.random() < 0.05;
    const value = isAnomaly
      ? baseValue + (Math.random() - 0.5) * 80
      : baseValue;

    const newPoint = {
      timestamp: now,
      value: value,
      isAnomaly: isAnomaly,
      confidence: isAnomaly
        ? 0.7 + Math.random() * 0.3
        : 0.1 + Math.random() * 0.3,
    };

    // Update main time series chart
    const timeSeriesChart = this.charts.get("timeseries");
    if (timeSeriesChart) {
      timeSeriesChart.addDataPoint(newPoint);
    }

    // Update real-time mini chart
    const realtimeChart = this.charts.get("realtime");
    if (realtimeChart) {
      realtimeChart.addDataPoint(newPoint, 20); // Keep only last 20 points
    }

    // Update statistics
    this.updateStatistics();

    // Announce significant anomalies
    if (isAnomaly && newPoint.confidence > 0.8) {
      this.announceToScreenReader(
        `High confidence anomaly detected: value ${value.toFixed(2)}, confidence ${(newPoint.confidence * 100).toFixed(1)}%`,
      );
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
      if (key === "timeseries" || key === "realtime") {
        chart.setData(this.demoData.timeSeries);
      } else if (key === "scatter" || key === "distribution") {
        chart.setData(this.demoData.scatter);
      } else if (key === "heatmap" || key === "correlation") {
        chart.setData(this.demoData.heatmap);
      }
    });

    this.updateStatistics();
    this.announceToScreenReader("All chart data refreshed");
  }

  exportCharts() {
    const exportData = {
      timestamp: new Date().toISOString(),
      charts: {},
    };

    this.charts.forEach((chart, key) => {
      if (chart.data) {
        exportData.charts[key] = {
          type: chart.constructor.name,
          data: chart.data,
          options: chart.options,
        };
      }
    });

    // Create downloadable JSON file
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pynomaly-charts-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.announceToScreenReader("Chart data exported successfully");
  }

  updateStatistics() {
    const allAnomalies = this.demoData.timeSeries.filter((d) => d.isAnomaly);
    const totalAnomalies = allAnomalies.length;
    const avgConfidence =
      totalAnomalies > 0
        ? allAnomalies.reduce((sum, d) => sum + d.confidence, 0) /
          totalAnomalies
        : 0;

    document.getElementById("total-anomalies").textContent = totalAnomalies;
    document.getElementById("avg-confidence").textContent =
      (avgConfidence * 100).toFixed(1) + "%";
    document.getElementById("last-updated").textContent =
      new Date().toLocaleTimeString();
  }

  updateSelectionInfo(selectedData) {
    const detailsElement = document.getElementById("selection-details");
    if (!detailsElement) return;

    if (selectedData.length === 0) {
      detailsElement.textContent = "No points selected";
      return;
    }

    const anomalies = selectedData.filter((d) => d.isAnomaly).length;
    const avgScore =
      selectedData.reduce((sum, d) => sum + d.anomalyScore, 0) /
      selectedData.length;

    detailsElement.innerHTML = `
      <strong>${selectedData.length} points selected</strong><br/>
      Anomalies: ${anomalies}<br/>
      Average Score: ${avgScore.toFixed(3)}
    `;
  }

  showAnomalyDetails(data, chartType) {
    const modal = document.createElement("div");
    modal.className = "anomaly-modal";
    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <h3>Anomaly Details - ${chartType}</h3>
          <button class="modal-close" aria-label="Close">&times;</button>
        </div>
        <div class="modal-body">
          <div class="detail-grid">
            ${data.timestamp ? `<div><strong>Timestamp:</strong> ${data.timestamp.toLocaleString()}</div>` : ""}
            ${data.value !== undefined ? `<div><strong>Value:</strong> ${data.value.toFixed(3)}</div>` : ""}
            ${data.x !== undefined ? `<div><strong>X:</strong> ${data.x.toFixed(3)}</div>` : ""}
            ${data.y !== undefined ? `<div><strong>Y:</strong> ${data.y.toFixed(3)}</div>` : ""}
            ${data.confidence !== undefined ? `<div><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</div>` : ""}
            ${data.anomalyScore !== undefined ? `<div><strong>Anomaly Score:</strong> ${data.anomalyScore.toFixed(3)}</div>` : ""}
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary modal-close">Close</button>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    // Setup close handlers
    modal.querySelectorAll(".modal-close").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.body.removeChild(modal);
      });
    });

    // Close on backdrop click
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        document.body.removeChild(modal);
      }
    });

    // Focus management
    const firstButton = modal.querySelector("button");
    firstButton?.focus();
  }

  showCellDetails(data) {
    this.announceToScreenReader(
      `Heatmap cell selected: ${data.x}, ${data.y}, value ${data.value.toFixed(3)}`,
    );
  }

  announceDataSummary() {
    const totalPoints = this.demoData.timeSeries.length;
    const anomalies = this.demoData.timeSeries.filter(
      (d) => d.isAnomaly,
    ).length;
    const anomalyRate = ((anomalies / totalPoints) * 100).toFixed(1);

    const message = `Data summary: ${totalPoints} total data points, ${anomalies} anomalies detected, ${anomalyRate}% anomaly rate`;
    this.announceToScreenReader(message);
  }

  announceToScreenReader(message) {
    const announcer = document.getElementById("chart-announcer");
    if (announcer) {
      announcer.textContent = message;
    }
  }

  destroy() {
    if (this.realTimeInterval) {
      clearInterval(this.realTimeInterval);
    }

    this.charts.forEach((chart) => {
      chart.destroy();
    });

    this.charts.clear();
  }
}

// Auto-initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  if (document.querySelector(".d3-charts-demo")) {
    window.d3ChartsDemo = new D3ChartsDemo();
  }
});

// Export for module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = D3ChartsDemo;
} else {
  window.D3ChartsDemo = D3ChartsDemo;
}
