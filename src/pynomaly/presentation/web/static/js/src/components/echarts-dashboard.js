/**
 * ECharts Dashboard Components for Pynomaly
 * Production-ready statistical charts, performance metrics, and anomaly visualization
 * with responsive design and accessibility features
 */

class EChartsDashboard {
  constructor() {
    this.charts = new Map();
    this.themes = {
      light: {
        backgroundColor: "#ffffff",
        textStyle: {
          color: "#1e293b",
        },
        title: {
          textStyle: {
            color: "#1e293b",
          },
        },
        legend: {
          textStyle: {
            color: "#64748b",
          },
        },
        grid: {
          borderColor: "#e2e8f0",
        },
        categoryAxis: {
          axisLine: { lineStyle: { color: "#e2e8f0" } },
          axisTick: { lineStyle: { color: "#e2e8f0" } },
          axisLabel: { color: "#64748b" },
          splitLine: { lineStyle: { color: "#f1f5f9" } },
        },
        valueAxis: {
          axisLine: { lineStyle: { color: "#e2e8f0" } },
          axisTick: { lineStyle: { color: "#e2e8f0" } },
          axisLabel: { color: "#64748b" },
          splitLine: { lineStyle: { color: "#f1f5f9" } },
        },
        colorBy: "series",
        color: [
          "#0ea5e9",
          "#22c55e",
          "#f59e0b",
          "#ef4444",
          "#8b5cf6",
          "#06b6d4",
        ],
      },
      dark: {
        backgroundColor: "#1e293b",
        textStyle: {
          color: "#f1f5f9",
        },
        title: {
          textStyle: {
            color: "#f1f5f9",
          },
        },
        legend: {
          textStyle: {
            color: "#94a3b8",
          },
        },
        grid: {
          borderColor: "#4b5563",
        },
        categoryAxis: {
          axisLine: { lineStyle: { color: "#4b5563" } },
          axisTick: { lineStyle: { color: "#4b5563" } },
          axisLabel: { color: "#94a3b8" },
          splitLine: { lineStyle: { color: "#374151" } },
        },
        valueAxis: {
          axisLine: { lineStyle: { color: "#4b5563" } },
          axisTick: { lineStyle: { color: "#4b5563" } },
          axisLabel: { color: "#94a3b8" },
          splitLine: { lineStyle: { color: "#374151" } },
        },
        colorBy: "series",
        color: [
          "#38bdf8",
          "#4ade80",
          "#fbbf24",
          "#f87171",
          "#a78bfa",
          "#22d3ee",
        ],
      },
    };
    this.currentTheme = "light";
    this.setupEventListeners();
  }

  setupEventListeners() {
    // Theme change listener
    document.addEventListener("theme-changed", (event) => {
      this.currentTheme = event.detail.theme;
      this.updateAllChartsTheme();
    });

    // Resize listener
    window.addEventListener(
      "resize",
      this.debounce(() => {
        this.resizeAllCharts();
      }, 250),
    );
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

  getTheme() {
    return this.themes[this.currentTheme];
  }

  registerChart(id, chart) {
    this.charts.set(id, chart);
  }

  removeChart(id) {
    const chart = this.charts.get(id);
    if (chart) {
      chart.dispose();
    }
    this.charts.delete(id);
  }

  updateAllChartsTheme() {
    this.charts.forEach((chart) => {
      if (chart && !chart.isDisposed()) {
        const theme = this.getTheme();
        chart.setOption(theme, true);
        chart.resize();
      }
    });
  }

  resizeAllCharts() {
    this.charts.forEach((chart) => {
      if (chart && !chart.isDisposed()) {
        chart.resize();
      }
    });
  }
}

/**
 * Base ECharts Component
 * Provides common functionality for all chart types
 */
class BaseEChart {
  constructor(container, options = {}) {
    this.container =
      typeof container === "string"
        ? document.querySelector(container)
        : container;
    this.options = {
      responsive: true,
      accessibility: true,
      animation: true,
      theme: "light",
      ...options,
    };

    this.chart = null;
    this.data = [];

    this.init();
    this.setupAccessibility();

    // Register with dashboard
    const chartId = this.container.id || `chart-${Date.now()}`;
    echartsManager.registerChart(chartId, this.chart);
  }

  init() {
    if (!this.container) {
      throw new Error("Container element not found");
    }

    // Initialize ECharts instance
    this.chart = echarts.init(this.container, null, {
      renderer: "canvas",
      useDirtyRect: true, // Performance optimization
    });

    // Apply theme
    this.applyTheme();

    // Setup resize observer
    if (this.options.responsive) {
      this.setupResizeObserver();
    }

    // Setup event listeners
    this.setupEventListeners();
  }

  setupAccessibility() {
    if (!this.options.accessibility) return;

    // Add ARIA labels
    this.container.setAttribute("role", "img");
    this.container.setAttribute(
      "aria-label",
      this.options.title || "Chart visualization",
    );

    // Add description if provided
    if (this.options.description) {
      const descId = `${this.container.id || "chart"}-desc`;
      let descElement = document.getElementById(descId);

      if (!descElement) {
        descElement = document.createElement("div");
        descElement.id = descId;
        descElement.className = "sr-only";
        descElement.textContent = this.options.description;
        this.container.parentNode.insertBefore(
          descElement,
          this.container.nextSibling,
        );
      }

      this.container.setAttribute("aria-describedby", descId);
    }

    // Make container focusable
    this.container.setAttribute("tabindex", "0");
  }

  setupResizeObserver() {
    if ("ResizeObserver" in window) {
      this.resizeObserver = new ResizeObserver(() => {
        if (this.chart && !this.chart.isDisposed()) {
          this.chart.resize();
        }
      });
      this.resizeObserver.observe(this.container);
    }
  }

  setupEventListeners() {
    // Chart click events
    this.chart.on("click", (params) => {
      this.container.dispatchEvent(
        new CustomEvent("chart-click", {
          detail: { params, chart: this },
        }),
      );
    });

    // Chart hover events
    this.chart.on("mouseover", (params) => {
      this.container.dispatchEvent(
        new CustomEvent("chart-hover", {
          detail: { params, chart: this },
        }),
      );
    });

    // Keyboard navigation
    this.container.addEventListener("keydown", (event) => {
      this.handleKeyboardNavigation(event);
    });
  }

  handleKeyboardNavigation(event) {
    // Basic keyboard navigation for accessibility
    switch (event.key) {
      case "Enter":
      case " ":
        event.preventDefault();
        this.announceChartData();
        break;
      case "Escape":
        this.container.blur();
        break;
    }
  }

  announceChartData() {
    if (!this.options.accessibility) return;

    const announcement = this.generateDataAnnouncement();
    this.announceToScreenReader(announcement);
  }

  generateDataAnnouncement() {
    // Override in subclasses for specific announcements
    return `Chart with ${this.data.length} data points`;
  }

  announceToScreenReader(message) {
    const announcer =
      document.getElementById("chart-announcer") ||
      document.querySelector('[aria-live="polite"]');
    if (announcer) {
      announcer.textContent = message;
    }
  }

  applyTheme() {
    const theme = echartsManager.getTheme();
    this.chart.setOption(theme, true);
  }

  setData(data) {
    this.data = data;
    this.updateChart();
  }

  updateChart() {
    // Override in subclasses
    throw new Error("updateChart() method must be implemented by subclass");
  }

  resize() {
    if (this.chart && !this.chart.isDisposed()) {
      this.chart.resize();
    }
  }

  dispose() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    if (this.chart && !this.chart.isDisposed()) {
      this.chart.dispose();
    }
  }
}

/**
 * Performance Metrics Chart
 * Real-time system performance visualization
 */
class PerformanceMetricsChart extends BaseEChart {
  constructor(container, options = {}) {
    super(container, {
      title: "System Performance Metrics",
      description:
        "Real-time visualization of CPU, memory, and network utilization",
      metrics: ["cpu", "memory", "network"],
      updateInterval: 2000,
      maxDataPoints: 100,
      ...options,
    });

    this.metricColors = {
      cpu: "#ef4444",
      memory: "#f59e0b",
      network: "#22c55e",
      disk: "#8b5cf6",
      response_time: "#06b6d4",
    };

    this.setupRealTimeUpdates();
  }

  updateChart() {
    const option = {
      title: {
        text: this.options.title,
        left: "center",
        textStyle: {
          fontSize: 16,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
          label: {
            backgroundColor: "#6a7985",
          },
        },
        formatter: (params) => {
          let tooltip = `<strong>${params[0].axisValue}</strong><br/>`;
          params.forEach((param) => {
            const value =
              typeof param.value === "number"
                ? param.value.toFixed(2)
                : param.value;
            tooltip += `${param.marker} ${param.seriesName}: ${value}%<br/>`;
          });
          return tooltip;
        },
      },
      legend: {
        data: this.options.metrics.map((metric) => metric.toUpperCase()),
        top: 30,
        textStyle: {
          fontSize: 12,
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        top: 80,
        containLabel: true,
      },
      xAxis: {
        type: "category",
        boundaryGap: false,
        data: this.data.map((d) => d.timestamp),
        axisLabel: {
          formatter: (value) => {
            const date = new Date(value);
            return date.toLocaleTimeString();
          },
        },
      },
      yAxis: {
        type: "value",
        min: 0,
        max: 100,
        axisLabel: {
          formatter: "{value}%",
        },
      },
      series: this.options.metrics.map((metric) => ({
        name: metric.toUpperCase(),
        type: "line",
        stack: false,
        smooth: true,
        symbol: "circle",
        symbolSize: 4,
        lineStyle: {
          width: 2,
        },
        areaStyle: {
          opacity: 0.1,
        },
        data: this.data.map((d) => d[metric] || 0),
        itemStyle: {
          color: this.metricColors[metric] || "#0ea5e9",
        },
      })),
      animation: this.options.animation,
      animationDuration: 1000,
      animationEasing: "cubicOut",
    };

    this.chart.setOption(option);
  }

  setupRealTimeUpdates() {
    if (this.options.updateInterval > 0) {
      this.updateTimer = setInterval(() => {
        this.addRandomDataPoint();
      }, this.options.updateInterval);
    }
  }

  addRandomDataPoint() {
    const now = new Date();
    const newPoint = {
      timestamp: now.toISOString(),
    };

    // Generate realistic performance data
    this.options.metrics.forEach((metric) => {
      switch (metric) {
        case "cpu":
          newPoint[metric] = Math.max(
            0,
            Math.min(
              100,
              (this.data.length > 0
                ? this.data[this.data.length - 1][metric]
                : 50) +
                (Math.random() - 0.5) * 20,
            ),
          );
          break;
        case "memory":
          newPoint[metric] = Math.max(
            0,
            Math.min(
              100,
              (this.data.length > 0
                ? this.data[this.data.length - 1][metric]
                : 60) +
                (Math.random() - 0.5) * 10,
            ),
          );
          break;
        case "network":
          newPoint[metric] = Math.max(
            0,
            Math.min(100, Math.random() * 30 + 10),
          );
          break;
        default:
          newPoint[metric] = Math.random() * 100;
      }
    });

    this.data.push(newPoint);

    // Keep only recent data points
    if (this.data.length > this.options.maxDataPoints) {
      this.data = this.data.slice(-this.options.maxDataPoints);
    }

    this.updateChart();

    // Announce significant changes
    if (this.options.accessibility) {
      const highUsage = this.options.metrics.find(
        (metric) => newPoint[metric] > 90,
      );
      if (highUsage) {
        this.announceToScreenReader(
          `High ${highUsage} usage detected: ${newPoint[highUsage].toFixed(1)}%`,
        );
      }
    }
  }

  generateDataAnnouncement() {
    if (this.data.length === 0) return "No performance data available";

    const latest = this.data[this.data.length - 1];
    const metrics = this.options.metrics
      .map((metric) => `${metric}: ${(latest[metric] || 0).toFixed(1)}%`)
      .join(", ");

    return `Current performance metrics: ${metrics}`;
  }

  dispose() {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
    }
    super.dispose();
  }
}

/**
 * Anomaly Distribution Chart
 * Statistical visualization of anomaly patterns
 */
class AnomalyDistributionChart extends BaseEChart {
  constructor(container, options = {}) {
    super(container, {
      title: "Anomaly Distribution Analysis",
      description:
        "Statistical distribution of detected anomalies by type and confidence",
      chartType: "pie", // 'pie', 'bar', 'histogram'
      showConfidenceRanges: true,
      ...options,
    });
  }

  updateChart() {
    if (this.options.chartType === "pie") {
      this.updatePieChart();
    } else if (this.options.chartType === "bar") {
      this.updateBarChart();
    } else if (this.options.chartType === "histogram") {
      this.updateHistogramChart();
    }
  }

  updatePieChart() {
    // Group anomalies by type
    const typeDistribution = this.data.reduce((acc, anomaly) => {
      const type = anomaly.type || "Unknown";
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});

    const pieData = Object.entries(typeDistribution).map(([type, count]) => ({
      name: type,
      value: count,
      itemStyle: {
        color: this.getAnomalyTypeColor(type),
      },
    }));

    const option = {
      title: {
        text: this.options.title,
        left: "center",
        textStyle: {
          fontSize: 16,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "item",
        formatter: (params) => {
          const percentage = ((params.value / this.data.length) * 100).toFixed(
            1,
          );
          return `<strong>${params.name}</strong><br/>
                  Count: ${params.value}<br/>
                  Percentage: ${percentage}%`;
        },
      },
      legend: {
        orient: "vertical",
        left: "left",
        top: "center",
        textStyle: {
          fontSize: 12,
        },
      },
      series: [
        {
          name: "Anomaly Types",
          type: "pie",
          radius: ["40%", "70%"],
          center: ["60%", "50%"],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 4,
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
              fontSize: "16",
              fontWeight: "bold",
            },
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
          labelLine: {
            show: false,
          },
          data: pieData,
        },
      ],
      animation: this.options.animation,
      animationType: "expansion",
      animationEasing: "elasticOut",
      animationDelay: (idx) => idx * 100,
    };

    this.chart.setOption(option);
  }

  updateBarChart() {
    // Group by confidence ranges
    const confidenceRanges = {
      "Low (0-0.5)": 0,
      "Medium (0.5-0.8)": 0,
      "High (0.8-0.95)": 0,
      "Very High (0.95+)": 0,
    };

    this.data.forEach((anomaly) => {
      const confidence = anomaly.confidence || 0;
      if (confidence < 0.5) {
        confidenceRanges["Low (0-0.5)"]++;
      } else if (confidence < 0.8) {
        confidenceRanges["Medium (0.5-0.8)"]++;
      } else if (confidence < 0.95) {
        confidenceRanges["High (0.8-0.95)"]++;
      } else {
        confidenceRanges["Very High (0.95+)"]++;
      }
    });

    const option = {
      title: {
        text: "Anomalies by Confidence Level",
        left: "center",
        textStyle: {
          fontSize: 16,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
        formatter: (params) => {
          const param = params[0];
          const percentage = ((param.value / this.data.length) * 100).toFixed(
            1,
          );
          return `<strong>${param.name}</strong><br/>
                  Count: ${param.value}<br/>
                  Percentage: ${percentage}%`;
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        top: 60,
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: Object.keys(confidenceRanges),
        axisLabel: {
          interval: 0,
          rotate: 45,
        },
      },
      yAxis: {
        type: "value",
        name: "Count",
        nameLocation: "middle",
        nameGap: 50,
      },
      series: [
        {
          name: "Anomalies",
          type: "bar",
          data: Object.values(confidenceRanges),
          itemStyle: {
            color: (params) => {
              const colors = ["#fbbf24", "#f59e0b", "#ef4444", "#dc2626"];
              return colors[params.dataIndex] || "#0ea5e9";
            },
            borderRadius: [4, 4, 0, 0],
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: "rgba(0, 0, 0, 0.3)",
            },
          },
        },
      ],
      animation: this.options.animation,
      animationDelay: (idx) => idx * 100,
    };

    this.chart.setOption(option);
  }

  updateHistogramChart() {
    // Create histogram of anomaly scores
    const scores = this.data.map((d) => d.score || d.confidence || 0);
    const bins = this.createHistogramBins(scores, 20);

    const option = {
      title: {
        text: "Anomaly Score Distribution",
        left: "center",
        textStyle: {
          fontSize: 16,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "shadow",
        },
        formatter: (params) => {
          const param = params[0];
          return `<strong>Score Range: ${param.name}</strong><br/>
                  Count: ${param.value}<br/>
                  Density: ${(param.value / this.data.length).toFixed(3)}`;
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        top: 60,
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: bins.map((bin) => bin.range),
        name: "Anomaly Score",
        nameLocation: "middle",
        nameGap: 30,
      },
      yAxis: {
        type: "value",
        name: "Frequency",
        nameLocation: "middle",
        nameGap: 50,
      },
      series: [
        {
          name: "Frequency",
          type: "bar",
          data: bins.map((bin) => bin.count),
          itemStyle: {
            color: "#0ea5e9",
            borderRadius: [2, 2, 0, 0],
          },
          emphasis: {
            itemStyle: {
              color: "#0284c7",
            },
          },
        },
      ],
      animation: this.options.animation,
    };

    this.chart.setOption(option);
  }

  createHistogramBins(data, numBins) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / numBins;

    const bins = Array.from({ length: numBins }, (_, i) => ({
      range: `${(min + i * binWidth).toFixed(2)}-${(min + (i + 1) * binWidth).toFixed(2)}`,
      count: 0,
      min: min + i * binWidth,
      max: min + (i + 1) * binWidth,
    }));

    data.forEach((value) => {
      const binIndex = Math.min(
        Math.floor((value - min) / binWidth),
        numBins - 1,
      );
      bins[binIndex].count++;
    });

    return bins;
  }

  getAnomalyTypeColor(type) {
    const colors = {
      "Statistical Outlier": "#ef4444",
      "Temporal Anomaly": "#f59e0b",
      "Pattern Deviation": "#22c55e",
      "Threshold Violation": "#8b5cf6",
      "Trend Anomaly": "#06b6d4",
      Unknown: "#6b7280",
    };
    return colors[type] || "#0ea5e9";
  }

  generateDataAnnouncement() {
    if (this.data.length === 0) return "No anomaly data available";

    const types = [...new Set(this.data.map((d) => d.type || "Unknown"))];
    const avgConfidence =
      this.data.reduce((sum, d) => sum + (d.confidence || 0), 0) /
      this.data.length;

    return `${this.data.length} anomalies detected across ${types.length} types. Average confidence: ${(avgConfidence * 100).toFixed(1)}%`;
  }
}

/**
 * Detection Timeline Chart
 * Time-based visualization of anomaly detection events
 */
class DetectionTimelineChart extends BaseEChart {
  constructor(container, options = {}) {
    super(container, {
      title: "Anomaly Detection Timeline",
      description:
        "Chronological view of anomaly detection events with severity indicators",
      timeRange: "24h", // '1h', '6h', '24h', '7d', '30d'
      showSeverityLevels: true,
      groupByHour: false,
      ...options,
    });
  }

  updateChart() {
    // Process data based on time range and grouping
    const processedData = this.processTimelineData();

    const option = {
      title: {
        text: this.options.title,
        left: "center",
        textStyle: {
          fontSize: 16,
          fontWeight: "bold",
        },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
          animation: false,
        },
        formatter: (params) => {
          let tooltip = `<strong>${params[0].axisValue}</strong><br/>`;
          params.forEach((param) => {
            tooltip += `${param.marker} ${param.seriesName}: ${param.value}<br/>`;
          });
          return tooltip;
        },
      },
      legend: {
        data: ["Low Severity", "Medium Severity", "High Severity", "Critical"],
        top: 30,
        textStyle: {
          fontSize: 12,
        },
      },
      grid: {
        left: "3%",
        right: "4%",
        bottom: "3%",
        top: 80,
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: processedData.timeline,
        axisLabel: {
          formatter: (value) => {
            const date = new Date(value);
            if (
              this.options.timeRange === "1h" ||
              this.options.timeRange === "6h"
            ) {
              return date.toLocaleTimeString();
            } else if (this.options.timeRange === "24h") {
              return `${date.getHours()}:00`;
            } else {
              return date.toLocaleDateString();
            }
          },
        },
      },
      yAxis: {
        type: "value",
        name: "Count",
        nameLocation: "middle",
        nameGap: 50,
        min: 0,
      },
      series: [
        {
          name: "Low Severity",
          type: "bar",
          stack: "severity",
          data: processedData.low,
          itemStyle: { color: "#22c55e" },
        },
        {
          name: "Medium Severity",
          type: "bar",
          stack: "severity",
          data: processedData.medium,
          itemStyle: { color: "#f59e0b" },
        },
        {
          name: "High Severity",
          type: "bar",
          stack: "severity",
          data: processedData.high,
          itemStyle: { color: "#ef4444" },
        },
        {
          name: "Critical",
          type: "bar",
          stack: "severity",
          data: processedData.critical,
          itemStyle: { color: "#dc2626" },
        },
      ],
      animation: this.options.animation,
      animationDelay: (idx) => idx * 50,
    };

    this.chart.setOption(option);
  }

  processTimelineData() {
    // Create time buckets based on time range
    const buckets = this.createTimeBuckets();

    // Initialize data structure
    const result = {
      timeline: buckets.map((bucket) => bucket.label),
      low: new Array(buckets.length).fill(0),
      medium: new Array(buckets.length).fill(0),
      high: new Array(buckets.length).fill(0),
      critical: new Array(buckets.length).fill(0),
    };

    // Categorize and count anomalies
    this.data.forEach((anomaly) => {
      const timestamp = new Date(anomaly.timestamp);
      const bucketIndex = this.findTimeBucket(timestamp, buckets);

      if (bucketIndex >= 0) {
        const severity = this.getSeverityLevel(anomaly);
        result[severity][bucketIndex]++;
      }
    });

    return result;
  }

  createTimeBuckets() {
    const now = new Date();
    const buckets = [];

    switch (this.options.timeRange) {
      case "1h":
        for (let i = 59; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 60000);
          buckets.push({
            start: time,
            end: new Date(time.getTime() + 60000),
            label: time.toISOString(),
          });
        }
        break;

      case "6h":
        for (let i = 35; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 600000); // 10-minute buckets
          buckets.push({
            start: time,
            end: new Date(time.getTime() + 600000),
            label: time.toISOString(),
          });
        }
        break;

      case "24h":
        for (let i = 23; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 3600000); // 1-hour buckets
          buckets.push({
            start: time,
            end: new Date(time.getTime() + 3600000),
            label: time.toISOString(),
          });
        }
        break;

      case "7d":
        for (let i = 6; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 86400000); // 1-day buckets
          buckets.push({
            start: time,
            end: new Date(time.getTime() + 86400000),
            label: time.toISOString(),
          });
        }
        break;

      default: // 30d
        for (let i = 29; i >= 0; i--) {
          const time = new Date(now.getTime() - i * 86400000); // 1-day buckets
          buckets.push({
            start: time,
            end: new Date(time.getTime() + 86400000),
            label: time.toISOString(),
          });
        }
    }

    return buckets;
  }

  findTimeBucket(timestamp, buckets) {
    return buckets.findIndex(
      (bucket) => timestamp >= bucket.start && timestamp < bucket.end,
    );
  }

  getSeverityLevel(anomaly) {
    const confidence = anomaly.confidence || 0;
    const score = anomaly.score || confidence;

    if (score >= 0.95) return "critical";
    if (score >= 0.8) return "high";
    if (score >= 0.5) return "medium";
    return "low";
  }

  generateDataAnnouncement() {
    if (this.data.length === 0) return "No detection events in timeline";

    const severityCounts = {
      low: 0,
      medium: 0,
      high: 0,
      critical: 0,
    };

    this.data.forEach((anomaly) => {
      const severity = this.getSeverityLevel(anomaly);
      severityCounts[severity]++;
    });

    const total = this.data.length;
    const critical = severityCounts.critical;
    const high = severityCounts.high;

    return `Timeline shows ${total} detection events. ${critical} critical and ${high} high severity anomalies detected.`;
  }
}

// Initialize the ECharts dashboard manager
const echartsManager = new EChartsDashboard();

// Export classes for use in other modules
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    EChartsDashboard,
    BaseEChart,
    PerformanceMetricsChart,
    AnomalyDistributionChart,
    DetectionTimelineChart,
  };
} else {
  // Browser environment
  window.EChartsDashboard = EChartsDashboard;
  window.BaseEChart = BaseEChart;
  window.PerformanceMetricsChart = PerformanceMetricsChart;
  window.AnomalyDistributionChart = AnomalyDistributionChart;
  window.DetectionTimelineChart = DetectionTimelineChart;
  window.echartsManager = echartsManager;
}
