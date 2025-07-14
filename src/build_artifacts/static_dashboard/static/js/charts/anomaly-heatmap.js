/**
 * Advanced Anomaly Heatmap Chart Component
 * Interactive D3.js-based heatmap for multi-dimensional anomaly visualization
 */

import * as d3 from "d3";

export class AnomalyHeatmapChart {
  constructor(container, options = {}) {
    this.container = d3.select(container);
    this.options = {
      width: 800,
      height: 500,
      margin: { top: 60, right: 100, bottom: 60, left: 100 },
      cellPadding: 2,
      animationDuration: 750,
      showLegend: true,
      showTooltip: true,
      interactive: true,
      colorScheme: "RdYlBu",
      ...options,
    };

    this.data = [];
    this.processedData = [];
    this.scales = {};
    this.svg = null;
    this.tooltip = null;

    this.init();
  }

  init() {
    this.createSVG();
    this.createScales();
    this.createTooltip();
    this.setupInteractions();
    this.setupResize();
  }

  createSVG() {
    // Clear existing content
    this.container.selectAll("*").remove();

    const { width, height } = this.options;

    this.svg = this.container
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("class", "anomaly-heatmap-chart")
      .attr("role", "img")
      .attr(
        "aria-label",
        "Anomaly heatmap showing anomaly scores across different features and time periods",
      );

    // Create main chart group
    this.chartGroup = this.svg
      .append("g")
      .attr(
        "transform",
        `translate(${this.options.margin.left}, ${this.options.margin.top})`,
      );

    // Create groups for different elements
    this.cellsGroup = this.chartGroup.append("g").attr("class", "cells");
    this.xAxisGroup = this.chartGroup.append("g").attr("class", "x-axis");
    this.yAxisGroup = this.chartGroup.append("g").attr("class", "y-axis");
    this.legendGroup = this.svg.append("g").attr("class", "legend");
  }

  createScales() {
    const { width, height, margin } = this.options;
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    this.scales.x = d3.scaleBand().range([0, chartWidth]).padding(0.1);

    this.scales.y = d3.scaleBand().range([0, chartHeight]).padding(0.1);

    // Color scale for anomaly scores
    this.scales.color = d3
      .scaleSequential()
      .interpolator(d3.interpolateRdYlBu)
      .domain([1, 0]); // Reverse: Red for high anomaly scores, Blue for low

    // Size scale for enhanced visualization
    this.scales.size = d3.scaleLinear().domain([0, 1]).range([0.3, 1]);
  }

  createTooltip() {
    if (!this.options.showTooltip) return;

    this.tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "heatmap-tooltip")
      .style("position", "absolute")
      .style("visibility", "hidden")
      .style("background", "rgba(0, 0, 0, 0.9)")
      .style("color", "white")
      .style("padding", "12px 16px")
      .style("border-radius", "6px")
      .style("font-size", "13px")
      .style("line-height", "1.4")
      .style("pointer-events", "none")
      .style("z-index", "1000")
      .style("box-shadow", "0 4px 12px rgba(0, 0, 0, 0.3)");
  }

  setupInteractions() {
    if (!this.options.interactive) return;

    // Add zoom capability
    this.zoom = d3
      .zoom()
      .scaleExtent([0.5, 5])
      .on("zoom", (event) => this.handleZoom(event));

    this.svg.call(this.zoom);
  }

  setupResize() {
    const resizeObserver = new ResizeObserver(() => {
      this.resize();
    });

    resizeObserver.observe(this.container.node());
  }

  setData(data) {
    this.data = data;
    this.processData();
    this.updateScales();
    this.render();
  }

  processData() {
    // Transform data into heatmap format
    // Expected format: { x: string, y: string, value: number, anomalyScore: number }
    this.processedData = this.data.map((d) => ({
      ...d,
      value: +d.value,
      anomalyScore: +d.anomalyScore || 0,
      x: String(d.x),
      y: String(d.y),
    }));
  }

  updateScales() {
    if (this.processedData.length === 0) return;

    // Get unique x and y values
    const xValues = [...new Set(this.processedData.map((d) => d.x))].sort();
    const yValues = [...new Set(this.processedData.map((d) => d.y))].sort();

    this.scales.x.domain(xValues);
    this.scales.y.domain(yValues);

    // Update color scale domain based on anomaly scores
    const scoreExtent = d3.extent(this.processedData, (d) => d.anomalyScore);
    this.scales.color.domain(scoreExtent.reverse()); // Higher scores = more red
  }

  render() {
    this.renderCells();
    this.renderAxes();
    this.renderLegend();
    this.renderStatistics();
  }

  renderCells() {
    const cells = this.cellsGroup
      .selectAll(".heatmap-cell")
      .data(this.processedData, (d) => `${d.x}-${d.y}`);

    const cellsEnter = cells
      .enter()
      .append("rect")
      .attr("class", "heatmap-cell")
      .attr("x", (d) => this.scales.x(d.x))
      .attr("y", (d) => this.scales.y(d.y))
      .attr("width", 0)
      .attr("height", 0)
      .attr("rx", 2)
      .attr("ry", 2)
      .attr("fill", (d) => this.scales.color(d.anomalyScore))
      .attr("opacity", 0)
      .attr("stroke", "#fff")
      .attr("stroke-width", 1);

    // Add interactions
    cellsEnter
      .on("mouseover", (event, d) => this.showTooltip(event, d))
      .on("mouseout", () => this.hideTooltip())
      .on("click", (event, d) => this.handleCellClick(event, d));

    // Update cells
    cellsEnter
      .merge(cells)
      .transition()
      .duration(this.options.animationDuration)
      .delay((d, i) => i * 2) // Staggered animation
      .attr("x", (d) => this.scales.x(d.x))
      .attr("y", (d) => this.scales.y(d.y))
      .attr(
        "width",
        (d) => this.scales.x.bandwidth() * this.scales.size(d.anomalyScore),
      )
      .attr(
        "height",
        (d) => this.scales.y.bandwidth() * this.scales.size(d.anomalyScore),
      )
      .attr("fill", (d) => this.scales.color(d.anomalyScore))
      .attr("opacity", (d) => 0.7 + d.anomalyScore * 0.3);

    // Remove old cells
    cells
      .exit()
      .transition()
      .duration(this.options.animationDuration / 2)
      .attr("width", 0)
      .attr("height", 0)
      .attr("opacity", 0)
      .remove();

    // Add anomaly indicators for high-score cells
    this.renderAnomalyIndicators();
  }

  renderAnomalyIndicators() {
    const highAnomalyThreshold = 0.8;
    const anomalyCells = this.processedData.filter(
      (d) => d.anomalyScore >= highAnomalyThreshold,
    );

    const indicators = this.cellsGroup
      .selectAll(".anomaly-indicator")
      .data(anomalyCells, (d) => `${d.x}-${d.y}`);

    const indicatorsEnter = indicators
      .enter()
      .append("g")
      .attr("class", "anomaly-indicator")
      .attr(
        "transform",
        (d) =>
          `translate(${this.scales.x(d.x) + this.scales.x.bandwidth() / 2}, ${this.scales.y(d.y) + this.scales.y.bandwidth() / 2})`,
      );

    // Add warning icon
    indicatorsEnter
      .append("circle")
      .attr("r", 6)
      .attr("fill", "#ef4444")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .attr("opacity", 0);

    indicatorsEnter
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("font-size", "8px")
      .attr("font-weight", "bold")
      .attr("fill", "white")
      .attr("opacity", 0)
      .text("!");

    // Update indicators
    indicatorsEnter
      .merge(indicators)
      .transition()
      .duration(this.options.animationDuration)
      .attr(
        "transform",
        (d) =>
          `translate(${this.scales.x(d.x) + this.scales.x.bandwidth() / 2}, ${this.scales.y(d.y) + this.scales.y.bandwidth() / 2})`,
      )
      .selectAll("circle, text")
      .attr("opacity", 1);

    indicators
      .exit()
      .transition()
      .duration(this.options.animationDuration / 2)
      .attr("opacity", 0)
      .remove();
  }

  renderAxes() {
    // X-axis
    this.xAxisGroup
      .attr(
        "transform",
        `translate(0, ${this.options.height - this.options.margin.top - this.options.margin.bottom})`,
      )
      .transition()
      .duration(this.options.animationDuration)
      .call(d3.axisBottom(this.scales.x))
      .selectAll("text")
      .style("text-anchor", "end")
      .attr("dx", "-.8em")
      .attr("dy", ".15em")
      .attr("transform", "rotate(-45)");

    // Y-axis
    this.yAxisGroup
      .transition()
      .duration(this.options.animationDuration)
      .call(d3.axisLeft(this.scales.y));

    // Axis labels
    this.svg.selectAll(".axis-label").remove();

    this.svg
      .append("text")
      .attr("class", "axis-label")
      .attr("text-anchor", "middle")
      .attr("x", this.options.width / 2)
      .attr("y", this.options.height - 10)
      .style("font-size", "14px")
      .style("font-weight", "600")
      .text("Features / Time Periods");

    this.svg
      .append("text")
      .attr("class", "axis-label")
      .attr("text-anchor", "middle")
      .attr("transform", "rotate(-90)")
      .attr("x", -this.options.height / 2)
      .attr("y", 20)
      .style("font-size", "14px")
      .style("font-weight", "600")
      .text("Data Points / Instances");
  }

  renderLegend() {
    if (!this.options.showLegend) return;

    this.legendGroup.selectAll("*").remove();

    const legendWidth = 200;
    const legendHeight = 20;
    const legendX = this.options.width - legendWidth - 20;
    const legendY = 20;

    // Create gradient definition
    const defs = this.svg.select("defs").empty()
      ? this.svg.append("defs")
      : this.svg.select("defs");

    const gradient = defs.selectAll("#heatmap-gradient").data([null]);
    const gradientEnter = gradient
      .enter()
      .append("linearGradient")
      .attr("id", "heatmap-gradient")
      .attr("x1", "0%")
      .attr("x2", "100%")
      .attr("y1", "0%")
      .attr("y2", "0%");

    const stops = [
      { offset: "0%", color: this.scales.color(this.scales.color.domain()[1]) },
      { offset: "50%", color: this.scales.color(0.5) },
      {
        offset: "100%",
        color: this.scales.color(this.scales.color.domain()[0]),
      },
    ];

    gradientEnter
      .selectAll("stop")
      .data(stops)
      .enter()
      .append("stop")
      .attr("offset", (d) => d.offset)
      .attr("stop-color", (d) => d.color);

    // Legend rectangle
    this.legendGroup
      .append("rect")
      .attr("x", legendX)
      .attr("y", legendY)
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .attr("fill", "url(#heatmap-gradient)")
      .attr("stroke", "#ccc")
      .attr("rx", 3);

    // Legend labels
    const legendScale = d3
      .scaleLinear()
      .domain(this.scales.color.domain())
      .range([legendX, legendX + legendWidth]);

    const legendAxis = d3
      .axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d3.format(".2f"));

    this.legendGroup
      .append("g")
      .attr("transform", `translate(0, ${legendY + legendHeight})`)
      .call(legendAxis);

    // Legend title
    this.legendGroup
      .append("text")
      .attr("x", legendX + legendWidth / 2)
      .attr("y", legendY - 5)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("font-weight", "600")
      .text("Anomaly Score");
  }

  renderStatistics() {
    const stats = this.calculateStatistics();

    this.svg.selectAll(".stats-panel").remove();

    const statsPanel = this.svg
      .append("g")
      .attr("class", "stats-panel")
      .attr("transform", `translate(20, 20)`);

    // Background
    statsPanel
      .append("rect")
      .attr("width", 180)
      .attr("height", 80)
      .attr("fill", "rgba(255, 255, 255, 0.95)")
      .attr("stroke", "#ddd")
      .attr("rx", 6);

    // Statistics text
    const statsText = [
      `Total Cells: ${stats.totalCells}`,
      `High Anomalies: ${stats.highAnomalies}`,
      `Average Score: ${stats.averageScore.toFixed(3)}`,
      `Max Score: ${stats.maxScore.toFixed(3)}`,
    ];

    statsPanel
      .selectAll(".stat-text")
      .data(statsText)
      .enter()
      .append("text")
      .attr("class", "stat-text")
      .attr("x", 10)
      .attr("y", (d, i) => 18 + i * 16)
      .style("font-size", "12px")
      .style("fill", "#333")
      .text((d) => d);
  }

  calculateStatistics() {
    const totalCells = this.processedData.length;
    const highAnomalies = this.processedData.filter(
      (d) => d.anomalyScore >= 0.8,
    ).length;
    const averageScore =
      d3.mean(this.processedData, (d) => d.anomalyScore) || 0;
    const maxScore = d3.max(this.processedData, (d) => d.anomalyScore) || 0;

    return {
      totalCells,
      highAnomalies,
      averageScore,
      maxScore,
    };
  }

  showTooltip(event, data) {
    if (!this.tooltip) return;

    const formatScore = d3.format(".3f");
    const formatValue = d3.format(".2f");

    const content = `
      <div style="margin-bottom: 8px;"><strong>Position</strong></div>
      <div>X: ${data.x}</div>
      <div>Y: ${data.y}</div>
      <div style="margin: 8px 0;"><strong>Metrics</strong></div>
      <div>Value: ${formatValue(data.value)}</div>
      <div>Anomaly Score: ${formatScore(data.anomalyScore)}</div>
      <div style="margin-top: 8px;">
        <strong>Status: </strong>
        <span style="color: ${data.anomalyScore >= 0.8 ? "#ef4444" : data.anomalyScore >= 0.5 ? "#f59e0b" : "#22c55e"};">
          ${data.anomalyScore >= 0.8 ? "High Anomaly" : data.anomalyScore >= 0.5 ? "Moderate" : "Normal"}
        </span>
      </div>
    `;

    this.tooltip
      .style("visibility", "visible")
      .html(content)
      .style("left", event.pageX + 15 + "px")
      .style("top", event.pageY - 10 + "px");
  }

  hideTooltip() {
    if (this.tooltip) {
      this.tooltip.style("visibility", "hidden");
    }
  }

  handleCellClick(event, data) {
    // Emit cell click event
    this.container.node().dispatchEvent(
      new CustomEvent("cellClicked", {
        detail: { data, event },
      }),
    );
  }

  handleZoom(event) {
    const { transform } = event;

    // Apply zoom transform to chart group
    this.chartGroup.attr(
      "transform",
      `translate(${this.options.margin.left + transform.x}, ${this.options.margin.top + transform.y}) scale(${transform.k})`,
    );
  }

  resize() {
    const containerRect = this.container.node().getBoundingClientRect();
    const newWidth = containerRect.width;
    const newHeight = Math.max(400, containerRect.height);

    if (newWidth === this.options.width && newHeight === this.options.height) {
      return;
    }

    this.options.width = newWidth;
    this.options.height = newHeight;

    // Update SVG dimensions
    this.svg.attr("width", newWidth).attr("height", newHeight);

    // Update scales
    this.createScales();
    this.updateScales();

    // Re-render
    this.render();
  }

  updateColorScheme(scheme) {
    this.options.colorScheme = scheme;

    // Update color scale
    switch (scheme) {
      case "viridis":
        this.scales.color = d3.scaleSequential(d3.interpolateViridis);
        break;
      case "plasma":
        this.scales.color = d3.scaleSequential(d3.interpolatePlasma);
        break;
      case "RdYlBu":
      default:
        this.scales.color = d3.scaleSequential(d3.interpolateRdYlBu);
        break;
    }

    this.scales.color.domain([1, 0]); // Reverse domain
    this.render();
  }

  filterByScore(minScore, maxScore) {
    const filteredData = this.data.filter(
      (d) => d.anomalyScore >= minScore && d.anomalyScore <= maxScore,
    );

    this.processedData = filteredData.map((d) => ({
      ...d,
      value: +d.value,
      anomalyScore: +d.anomalyScore || 0,
      x: String(d.x),
      y: String(d.y),
    }));

    this.updateScales();
    this.render();
  }

  destroy() {
    if (this.tooltip) {
      this.tooltip.remove();
    }

    this.container.selectAll("*").remove();
  }

  // Export methods
  exportAsPNG() {
    const svgElement = this.svg.node();
    const svgData = new XMLSerializer().serializeToString(svgElement);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();

    canvas.width = this.options.width;
    canvas.height = this.options.height;

    img.onload = () => {
      ctx.drawImage(img, 0, 0);
      const pngFile = canvas.toDataURL("image/png");

      const downloadLink = document.createElement("a");
      downloadLink.download = "anomaly-heatmap.png";
      downloadLink.href = pngFile;
      downloadLink.click();
    };

    img.src = "data:image/svg+xml;base64," + btoa(svgData);
  }
}

// Factory function for easy instantiation
export function createAnomalyHeatmap(container, options = {}) {
  return new AnomalyHeatmapChart(container, options);
}

// Example usage
export function initializeHeatmap() {
  const container = document.querySelector(
    '[data-component="anomaly-heatmap"]',
  );
  if (!container) return;

  const chart = new AnomalyHeatmapChart(container, {
    width: container.clientWidth,
    height: 500,
    interactive: true,
    showLegend: true,
  });

  // Example data
  const sampleData = generateSampleHeatmapData();
  chart.setData(sampleData);

  return chart;
}

function generateSampleHeatmapData() {
  const data = [];
  const features = [
    "Temperature",
    "Pressure",
    "Humidity",
    "Vibration",
    "Current",
    "Voltage",
  ];
  const timePoints = Array.from(
    { length: 24 },
    (_, i) => `${i.toString().padStart(2, "0")}:00`,
  );

  features.forEach((feature) => {
    timePoints.forEach((time) => {
      const baseScore = Math.random() * 0.3; // Base low anomaly score
      const isAnomaly = Math.random() < 0.15; // 15% chance of anomaly
      const anomalyScore = isAnomaly ? 0.7 + Math.random() * 0.3 : baseScore;
      const value = 50 + Math.random() * 100;

      data.push({
        x: time,
        y: feature,
        value,
        anomalyScore,
      });
    });
  });

  return data;
}
