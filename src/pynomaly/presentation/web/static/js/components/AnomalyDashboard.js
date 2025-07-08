/**
 * Advanced Anomaly Detection Dashboard
 *
 * Features:
 * - Multiple synchronized visualizations using ECharts
 * - Real-time data updates with WebSocket support
 * - Interactive brushing and linking between charts
 * - Responsive layout with grid system
 * - Advanced filtering and data exploration
 * - Accessibility support with screen reader compatibility
 * - Performance optimization for large datasets
 */

class AnomalyDashboard {
  constructor(container, options = {}) {
    this.container = d3.select(container);
    this.options = {
      layout: {
        columns: 2,
        rows: 2,
        gap: 20,
      },
      realTime: {
        enabled: false,
        interval: 5000,
        maxDataPoints: 1000,
      },
      theme: "light",
      animation: {
        duration: 300,
        easing: "cubicOut",
      },
      colors: {
        normal: "#3b82f6",
        anomaly: "#ef4444",
        warning: "#f59e0b",
        background: "#ffffff",
        text: "#1f2937",
      },
      ...options,
    };

    this.data = {
      timeSeries: [],
      anomalies: [],
      features: [],
      summary: {},
    };

    this.charts = {
      timeSeries: null,
      heatmap: null,
      distribution: null,
      summary: null,
    };

    this.filters = {
      timeRange: null,
      features: [],
      anomalyTypes: [],
      severityThreshold: 0,
    };

    this.init();
    this.setupAccessibility();
    this.bindEvents();
  }

  init() {
    this.createLayout();
    this.createCharts();
    this.createControls();
    this.createStatusBar();
  }

  createLayout() {
    this.container
      .attr("class", "anomaly-dashboard")
      .style("display", "grid")
      .style(
        "grid-template-columns",
        `repeat(${this.options.layout.columns}, 1fr)`,
      )
      .style(
        "grid-template-rows",
        `auto repeat(${this.options.layout.rows}, 1fr) auto`,
      )
      .style("gap", `${this.options.layout.gap}px`)
      .style("height", "100vh")
      .style("padding", "20px");

    // Create chart containers
    this.chartContainers = {
      timeSeries: this.container
        .append("div")
        .attr("class", "chart-container chart-container--time-series")
        .style("grid-column", "1 / -1")
        .style("background", "var(--color-bg-primary)")
        .style("border-radius", "var(--border-radius-lg)")
        .style("box-shadow", "var(--shadow-sm)")
        .style("padding", "20px"),

      heatmap: this.container
        .append("div")
        .attr("class", "chart-container chart-container--heatmap")
        .style("background", "var(--color-bg-primary)")
        .style("border-radius", "var(--border-radius-lg)")
        .style("box-shadow", "var(--shadow-sm)")
        .style("padding", "20px"),

      distribution: this.container
        .append("div")
        .attr("class", "chart-container chart-container--distribution")
        .style("background", "var(--color-bg-primary)")
        .style("border-radius", "var(--border-radius-lg)")
        .style("box-shadow", "var(--shadow-sm)")
        .style("padding", "20px"),

      summary: this.container
        .append("div")
        .attr("class", "chart-container chart-container--summary")
        .style("grid-column", "1 / -1")
        .style("background", "var(--color-bg-primary)")
        .style("border-radius", "var(--border-radius-lg)")
        .style("box-shadow", "var(--shadow-sm)")
        .style("padding", "20px"),
    };

    // Add chart titles
    Object.keys(this.chartContainers).forEach((key) => {
      this.chartContainers[key]
        .append("h3")
        .attr("class", "chart-title")
        .style("margin", "0 0 15px 0")
        .style("font-size", "var(--font-size-lg)")
        .style("font-weight", "var(--font-weight-semibold)")
        .style("color", "var(--color-text-primary)")
        .text(this.getChartTitle(key));
    });
  }

  getChartTitle(chartType) {
    const titles = {
      timeSeries: "Time Series Analysis",
      heatmap: "Feature Correlation Heatmap",
      distribution: "Anomaly Score Distribution",
      summary: "Summary Statistics",
    };
    return titles[chartType] || chartType;
  }

  createCharts() {
    // Time Series Chart
    this.charts.timeSeries = echarts.init(
      this.chartContainers.timeSeries
        .append("div")
        .style("width", "100%")
        .style("height", "300px")
        .node(),
    );

    // Heatmap Chart
    this.charts.heatmap = echarts.init(
      this.chartContainers.heatmap
        .append("div")
        .style("width", "100%")
        .style("height", "300px")
        .node(),
    );

    // Distribution Chart
    this.charts.distribution = echarts.init(
      this.chartContainers.distribution
        .append("div")
        .style("width", "100%")
        .style("height", "300px")
        .node(),
    );

    // Summary Chart
    this.charts.summary = echarts.init(
      this.chartContainers.summary
        .append("div")
        .style("width", "100%")
        .style("height", "200px")
        .node(),
    );

    // Configure chart interactions
    this.setupChartInteractions();
  }

  setupChartInteractions() {
    // Time series brush selection
    this.charts.timeSeries.on("brushSelected", (params) => {
      const selected = params.batch[0].selected[0];
      if (selected) {
        const timeRange = [
          selected.dataIndex[0],
          selected.dataIndex[selected.dataIndex.length - 1],
        ];
        this.handleTimeRangeSelection(timeRange);
      }
    });

    // Heatmap cell selection
    this.charts.heatmap.on("click", (params) => {
      if (params.componentType === "series") {
        this.handleFeatureSelection(params.data);
      }
    });

    // Distribution chart selection
    this.charts.distribution.on("click", (params) => {
      if (params.componentType === "series") {
        this.handleSeveritySelection(params.data);
      }
    });
  }

  createControls() {
    const controlsContainer = this.container
      .insert("div", ":first-child")
      .attr("class", "dashboard-controls")
      .style("grid-column", "1 / -1")
      .style("display", "flex")
      .style("flex-wrap", "wrap")
      .style("gap", "15px")
      .style("align-items", "center")
      .style("padding", "15px")
      .style("background", "var(--color-bg-secondary)")
      .style("border-radius", "var(--border-radius-lg)")
      .style("margin-bottom", "10px");

    // Real-time toggle
    const realTimeToggle = controlsContainer
      .append("label")
      .attr("class", "control-group")
      .style("display", "flex")
      .style("align-items", "center")
      .style("gap", "8px");

    realTimeToggle
      .append("input")
      .attr("type", "checkbox")
      .attr("id", "real-time-toggle")
      .style("margin", "0")
      .on("change", (event) => {
        this.options.realTime.enabled = event.target.checked;
        this.toggleRealTime();
      });

    realTimeToggle
      .append("span")
      .text("Real-time Updates")
      .style("font-size", "var(--font-size-sm)")
      .style("color", "var(--color-text-primary)");

    // Time range selector
    const timeRangeSelector = controlsContainer
      .append("div")
      .attr("class", "control-group")
      .style("display", "flex")
      .style("align-items", "center")
      .style("gap", "8px");

    timeRangeSelector
      .append("label")
      .text("Time Range:")
      .style("font-size", "var(--font-size-sm)")
      .style("color", "var(--color-text-primary)");

    const timeSelect = timeRangeSelector
      .append("select")
      .attr("class", "form-select")
      .style("padding", "4px 8px")
      .style("border", "1px solid var(--color-border-medium)")
      .style("border-radius", "var(--border-radius-md)")
      .on("change", (event) => {
        this.handleTimeRangeChange(event.target.value);
      });

    timeSelect
      .selectAll("option")
      .data([
        { value: "1h", text: "Last Hour" },
        { value: "6h", text: "Last 6 Hours" },
        { value: "24h", text: "Last 24 Hours" },
        { value: "7d", text: "Last 7 Days" },
        { value: "30d", text: "Last 30 Days" },
      ])
      .enter()
      .append("option")
      .attr("value", (d) => d.value)
      .text((d) => d.text);

    // Severity threshold slider
    const severityControl = controlsContainer
      .append("div")
      .attr("class", "control-group")
      .style("display", "flex")
      .style("align-items", "center")
      .style("gap", "8px");

    severityControl
      .append("label")
      .text("Min Severity:")
      .style("font-size", "var(--font-size-sm)")
      .style("color", "var(--color-text-primary)");

    const severitySlider = severityControl
      .append("input")
      .attr("type", "range")
      .attr("min", "0")
      .attr("max", "1")
      .attr("step", "0.1")
      .attr("value", "0")
      .style("width", "100px")
      .on("input", (event) => {
        this.filters.severityThreshold = parseFloat(event.target.value);
        this.updateSeverityDisplay(event.target.value);
        this.applyFilters();
      });

    this.severityDisplay = severityControl
      .append("span")
      .attr("class", "severity-display")
      .style("font-size", "var(--font-size-sm)")
      .style("color", "var(--color-text-secondary)")
      .style("min-width", "30px")
      .text("0.0");

    // Export button
    controlsContainer
      .append("button")
      .attr("class", "btn btn--secondary")
      .style("margin-left", "auto")
      .text("Export Data")
      .on("click", () => this.exportData());

    // Clear filters button
    controlsContainer
      .append("button")
      .attr("class", "btn btn--outline")
      .text("Clear Filters")
      .on("click", () => this.clearFilters());
  }

  createStatusBar() {
    this.statusBar = this.container
      .append("div")
      .attr("class", "dashboard-status")
      .style("grid-column", "1 / -1")
      .style("display", "flex")
      .style("justify-content", "space-between")
      .style("align-items", "center")
      .style("padding", "10px 15px")
      .style("background", "var(--color-bg-tertiary)")
      .style("border-radius", "var(--border-radius-md)")
      .style("font-size", "var(--font-size-sm)")
      .style("color", "var(--color-text-secondary)");

    this.statusInfo = this.statusBar.append("div").attr("class", "status-info");

    this.statusActions = this.statusBar
      .append("div")
      .attr("class", "status-actions")
      .style("display", "flex")
      .style("gap", "10px");
  }

  setupAccessibility() {
    // Add ARIA labels and descriptions
    this.container
      .attr("role", "application")
      .attr(
        "aria-label",
        "Anomaly detection dashboard with multiple interactive charts",
      );

    // Add live region for announcements
    this.liveRegion = this.container
      .append("div")
      .attr("class", "dashboard-live-region sr-only")
      .attr("aria-live", "polite")
      .attr("aria-atomic", "true");

    // Add keyboard navigation
    this.container
      .attr("tabindex", "0")
      .on("keydown", this.handleKeydown.bind(this));
  }

  bindEvents() {
    // Resize observer
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(() => {
        this.resizeCharts();
      });
      this.resizeObserver.observe(this.container.node());
    }

    window.addEventListener(
      "resize",
      this.debounce(() => {
        this.resizeCharts();
      }, 250),
    );
  }

  setData(data) {
    this.data = {
      timeSeries: data.timeSeries || [],
      anomalies: data.anomalies || [],
      features: data.features || [],
      summary: data.summary || {},
    };

    this.updateAllCharts();
    this.updateStatusBar();
    this.announceDataUpdate();
  }

  updateAllCharts() {
    this.updateTimeSeriesChart();
    this.updateHeatmapChart();
    this.updateDistributionChart();
    this.updateSummaryChart();
  }

  updateTimeSeriesChart() {
    const filteredData = this.applyTimeSeriesFilters();

    const option = {
      title: {
        show: false,
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
        },
        formatter: (params) => {
          let tooltip = `<div><strong>${params[0].axisValueLabel}</strong></div>`;
          params.forEach((param) => {
            tooltip += `<div style="display: flex; align-items: center; gap: 8px;">
              <span style="display: inline-block; width: 10px; height: 10px; background-color: ${param.color}; border-radius: 50%;"></span>
              <span>${param.seriesName}: ${param.value[1]}</span>
            </div>`;
          });
          return tooltip;
        },
      },
      legend: {
        data: ["Normal Values", "Anomalies"],
        bottom: 0,
      },
      toolbox: {
        feature: {
          brush: {
            type: ["lineX", "clear"],
          },
          saveAsImage: {},
          restore: {},
        },
      },
      brush: {
        toolbox: ["lineX", "clear"],
        xAxisIndex: 0,
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "15%",
        containLabel: true,
      },
      xAxis: {
        type: "time",
        boundaryGap: false,
        axisLabel: {
          formatter: (value) => {
            const date = new Date(value);
            return date.toLocaleTimeString();
          },
        },
      },
      yAxis: {
        type: "value",
        name: "Value",
        nameLocation: "middle",
        nameGap: 40,
      },
      series: [
        {
          name: "Normal Values",
          type: "line",
          data: filteredData.normal,
          itemStyle: {
            color: this.options.colors.normal,
          },
          lineStyle: {
            width: 2,
          },
          symbol: "none",
          sampling: "lttb",
        },
        {
          name: "Anomalies",
          type: "scatter",
          data: filteredData.anomalies,
          itemStyle: {
            color: this.options.colors.anomaly,
          },
          symbolSize: (data) => {
            return Math.max(6, Math.min(12, data[2] * 10)); // Size based on anomaly score
          },
          emphasis: {
            scale: 1.2,
          },
        },
      ],
      animation: true,
      animationDuration: this.options.animation.duration,
      animationEasing: this.options.animation.easing,
    };

    this.charts.timeSeries.setOption(option, true);
  }

  updateHeatmapChart() {
    const heatmapData = this.prepareHeatmapData();

    const option = {
      title: {
        show: false,
      },
      tooltip: {
        position: "top",
        formatter: (params) => {
          return `Feature ${params.data[0]} vs ${params.data[1]}<br/>
                  Correlation: ${params.data[2].toFixed(3)}`;
        },
      },
      grid: {
        height: "80%",
        top: "10%",
      },
      xAxis: {
        type: "category",
        data: heatmapData.features,
        splitArea: {
          show: true,
        },
        axisLabel: {
          rotate: 45,
        },
      },
      yAxis: {
        type: "category",
        data: heatmapData.features,
        splitArea: {
          show: true,
        },
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: "horizontal",
        left: "center",
        bottom: "0%",
        inRange: {
          color: [
            "#313695",
            "#4575b4",
            "#74add1",
            "#abd9e9",
            "#e0f3f8",
            "#ffffcc",
            "#fee090",
            "#fdae61",
            "#f46d43",
            "#d73027",
            "#a50026",
          ],
        },
      },
      series: [
        {
          name: "Correlation",
          type: "heatmap",
          data: heatmapData.data,
          label: {
            show: false,
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
        },
      ],
    };

    this.charts.heatmap.setOption(option, true);
  }

  updateDistributionChart() {
    const distributionData = this.prepareDistributionData();

    const option = {
      title: {
        show: false,
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
      },
      legend: {
        data: ["Normal", "Anomalies"],
        bottom: 0,
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "15%",
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: distributionData.bins,
        name: "Anomaly Score Range",
      },
      yAxis: {
        type: "value",
        name: "Count",
      },
      series: [
        {
          name: "Normal",
          type: "bar",
          stack: "total",
          data: distributionData.normal,
          itemStyle: {
            color: this.options.colors.normal,
            opacity: 0.8,
          },
        },
        {
          name: "Anomalies",
          type: "bar",
          stack: "total",
          data: distributionData.anomalies,
          itemStyle: {
            color: this.options.colors.anomaly,
            opacity: 0.8,
          },
        },
      ],
    };

    this.charts.distribution.setOption(option, true);
  }

  updateSummaryChart() {
    const summaryData = this.prepareSummaryData();

    const option = {
      title: {
        show: false,
      },
      tooltip: {
        trigger: "item",
        formatter: "{a} <br/>{b}: {c} ({d}%)",
      },
      legend: {
        orient: "vertical",
        left: "left",
        data: summaryData.categories,
      },
      series: [
        {
          name: "Detection Results",
          type: "pie",
          radius: ["40%", "70%"],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: "#fff",
            borderWidth: 2,
          },
          label: {
            show: false,
            position: "center",
          },
          emphasis: {
            label: {
              show: true,
              fontSize: "30",
              fontWeight: "bold",
            },
          },
          labelLine: {
            show: false,
          },
          data: summaryData.data,
        },
      ],
    };

    this.charts.summary.setOption(option, true);
  }

  applyTimeSeriesFilters() {
    const normal = [];
    const anomalies = [];

    this.data.timeSeries.forEach((point) => {
      const timestamp = new Date(point.timestamp);
      const value = [timestamp, point.value];

      if (point.isAnomaly) {
        // Drill down on anomaly click
        this.charts.timeSeries.on('click', (params) => {
          if (params.componentType === 'series' && params.seriesName === 'Anomalies') {
            this.openInvestigationDashboard(params.data);
          }
        });
      }

      if (
        point.isAnomaly &&
        point.anomalyScore >= this.filters.severityThreshold
      ) {
        anomalies.push([timestamp, point.value, point.anomalyScore]);
      } else {
        normal.push(value);
      }
    });

    return { normal, anomalies };
  }

  prepareHeatmapData() {
    // Create correlation matrix from features
    const features = this.data.features.slice(0, 10); // Limit to first 10 features
    const featureNames = features.map((_, i) => `Feature ${i + 1}`);
    const data = [];

    for (let i = 0; i < features.length; i++) {
      for (let j = 0; j < features.length; j++) {
        const correlation = i === j ? 1 : Math.random() * 2 - 1; // Mock correlation
        data.push([i, j, correlation]);
      }
    }

    return {
      features: featureNames,
      data,
    };
  }

  prepareDistributionData() {
    const bins = [
      "0.0-0.1",
      "0.1-0.2",
      "0.2-0.3",
      "0.3-0.4",
      "0.4-0.5",
      "0.5-0.6",
      "0.6-0.7",
      "0.7-0.8",
      "0.8-0.9",
      "0.9-1.0",
    ];

    const normal = new Array(10).fill(0);
    const anomalies = new Array(10).fill(0);

    this.data.timeSeries.forEach((point) => {
      const binIndex = Math.min(9, Math.floor(point.anomalyScore * 10));
      if (point.isAnomaly) {
        anomalies[binIndex]++;
      } else {
        normal[binIndex]++;
      }
    });

    return { bins, normal, anomalies };
  }

  prepareSummaryData() {
    const totalPoints = this.data.timeSeries.length;
    const anomalyPoints = this.data.timeSeries.filter(
      (p) => p.isAnomaly,
    ).length;
    const normalPoints = totalPoints - anomalyPoints;

    const categories = [
      "Normal",
      "Low Severity",
      "Medium Severity",
      "High Severity",
    ];
    const data = [
      {
        value: normalPoints,
        name: "Normal",
        itemStyle: { color: this.options.colors.normal },
      },
      {
        value: Math.floor(anomalyPoints * 0.5),
        name: "Low Severity",
        itemStyle: { color: "#fbbf24" },
      },
      {
        value: Math.floor(anomalyPoints * 0.3),
        name: "Medium Severity",
        itemStyle: { color: this.options.colors.warning },
      },
      {
        value: Math.floor(anomalyPoints * 0.2),
        name: "High Severity",
        itemStyle: { color: this.options.colors.anomaly },
      },
    ];

    return { categories, data };
  }

  handleTimeRangeSelection(range) {
    this.filters.timeRange = range;
    this.applyFilters();
    this.announceFilterChange("Time range selected");
  }

  handleFeatureSelection(data) {
    this.filters.features = [data[0], data[1]];
    this.applyFilters();
    this.announceFilterChange(`Features ${data[0]} and ${data[1]} selected`);
  }

  handleSeveritySelection(data) {
    this.filters.severityThreshold = data.value;
    this.applyFilters();
    this.announceFilterChange(`Severity threshold set to ${data.value}`);
  }

  openInvestigationDashboard(data) {
const investigationDashboard = new InteractiveInvestigationDashboard();
investigationDashboard.open(data);
  }

  applyFilters() {
    this.updateAllCharts();
    this.updateStatusBar();
  }

  clearFilters() {
    this.filters = {
      timeRange: null,
      features: [],
      anomalyTypes: [],
      severityThreshold: 0,
    };

    // Reset UI controls
    this.container.select("#real-time-toggle").property("checked", false);
    this.container.select('input[type="range"]').property("value", 0);
    this.updateSeverityDisplay("0");

    this.applyFilters();
    this.announceFilterChange("All filters cleared");
  }

  updateSeverityDisplay(value) {
    this.severityDisplay.text(parseFloat(value).toFixed(1));
  }

  updateStatusBar() {
    const totalPoints = this.data.timeSeries.length;
    const anomalies = this.data.timeSeries.filter((p) => p.isAnomaly).length;
    const filtered = this.data.timeSeries.filter(
      (p) => p.anomalyScore >= this.filters.severityThreshold,
    ).length;

    this.statusInfo.html(`
      <span>Total Points: <strong>${totalPoints}</strong></span>
      <span style="margin-left: 20px;">Anomalies: <strong>${anomalies}</strong></span>
      <span style="margin-left: 20px;">Filtered: <strong>${filtered}</strong></span>
      <span style="margin-left: 20px;">Last Update: <strong>${new Date().toLocaleTimeString()}</strong></span>
    `);
  }

  announceDataUpdate() {
    const anomalies = this.data.timeSeries.filter((p) => p.isAnomaly).length;
    const total = this.data.timeSeries.length;

    this.liveRegion.text(
      `Dashboard updated with ${total} data points, ${anomalies} anomalies detected.`,
    );
  }

  announceFilterChange(message) {
    this.liveRegion.text(message);
  }

  toggleRealTime() {
    if (this.options.realTime.enabled) {
      this.startRealTime();
    } else {
      this.stopRealTime();
    }
  }

  startRealTime() {
    this.realTimeTimer = setInterval(() => {
      this.fetchRealTimeData();
    }, this.options.realTime.interval);

    this.announceFilterChange("Real-time updates enabled");
  }

  stopRealTime() {
    if (this.realTimeTimer) {
      clearInterval(this.realTimeTimer);
      this.realTimeTimer = null;
    }

    this.announceFilterChange("Real-time updates disabled");
  }

  fetchRealTimeData() {
    // Mock real-time data generation
    const newPoint = {
      timestamp: new Date(),
      value: Math.random() * 100,
      anomalyScore: Math.random(),
      isAnomaly: Math.random() > 0.8,
    };

    this.data.timeSeries.push(newPoint);

    // Keep only recent data
    if (this.data.timeSeries.length > this.options.realTime.maxDataPoints) {
      this.data.timeSeries = this.data.timeSeries.slice(
        -this.options.realTime.maxDataPoints,
      );
    }

    this.updateTimeSeriesChart();
    this.updateStatusBar();
  }

  resizeCharts() {
    Object.values(this.charts).forEach((chart) => {
      if (chart) {
        chart.resize();
      }
    });
  }

  exportData() {
    const exportData = {
      timeSeries: this.data.timeSeries,
      anomalies: this.data.anomalies,
      summary: this.data.summary,
      filters: this.filters,
      exported: new Date().toISOString(),
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `anomaly-dashboard-${Date.now()}.json`;
    link.click();

    URL.revokeObjectURL(url);
    this.announceFilterChange("Data exported successfully");
  }

  handleKeydown(event) {
    switch (event.key) {
      case "Escape":
        this.clearFilters();
        break;
      case "r":
      case "R":
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.options.realTime.enabled = !this.options.realTime.enabled;
          this.toggleRealTime();
        }
        break;
      case "e":
      case "E":
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.exportData();
        }
        break;
    }
  }

  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  destroy() {
    if (this.realTimeTimer) {
      clearInterval(this.realTimeTimer);
    }

    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    Object.values(this.charts).forEach((chart) => {
      if (chart) {
        chart.dispose();
      }
    });

    this.container.selectAll("*").remove();
  }
}

// Export for module systems
if (typeof module !== "undefined" && module.exports) {
  module.exports = AnomalyDashboard;
}
