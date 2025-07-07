/**
 * Advanced Anomaly Timeline Chart Component
 * Interactive D3.js-based visualization for anomaly detection data
 */

import * as d3 from "d3";

export class AnomalyTimelineChart {
  constructor(container, options = {}) {
    this.container = d3.select(container);
    this.options = {
      width: 800,
      height: 400,
      margin: { top: 20, right: 80, bottom: 40, left: 60 },
      animationDuration: 750,
      showLegend: true,
      showTooltip: true,
      interactive: true,
      realTimeUpdates: false,
      ...options,
    };

    this.data = [];
    this.scales = {};
    this.svg = null;
    this.tooltip = null;
    this.zoom = null;

    this.init();
  }

  init() {
    this.createSVG();
    this.createScales();
    this.createAxes();
    this.createTooltip();
    this.setupInteractions();
    this.setupResize();
  }

  createSVG() {
    // Clear existing content
    this.container.selectAll("*").remove();

    const { width, height, margin } = this.options;

    this.svg = this.container
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("class", "anomaly-timeline-chart")
      .attr("role", "img")
      .attr(
        "aria-label",
        "Anomaly detection timeline showing normal and anomalous data points over time",
      );

    // Create main chart group
    this.chartGroup = this.svg
      .append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // Create clipping path for data area
    this.svg
      .append("defs")
      .append("clipPath")
      .attr("id", "chart-clip")
      .append("rect")
      .attr("width", width - margin.left - margin.right)
      .attr("height", height - margin.top - margin.bottom);

    // Create data container with clipping
    this.dataGroup = this.chartGroup
      .append("g")
      .attr("clip-path", "url(#chart-clip)");
  }

  createScales() {
    const { width, height, margin } = this.options;
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    this.scales.x = d3.scaleTime().range([0, chartWidth]);

    this.scales.y = d3.scaleLinear().range([chartHeight, 0]);

    this.scales.color = d3
      .scaleOrdinal()
      .domain(["normal", "anomaly"])
      .range(["#0ea5e9", "#ef4444"]);

    this.scales.size = d3.scaleLinear().domain([0, 1]).range([3, 8]);
  }

  createAxes() {
    // X-axis
    this.xAxis = d3
      .axisBottom(this.scales.x)
      .tickFormat(d3.timeFormat("%m/%d %H:%M"));

    this.xAxisGroup = this.chartGroup
      .append("g")
      .attr("class", "x-axis")
      .attr(
        "transform",
        `translate(0, ${this.options.height - this.options.margin.top - this.options.margin.bottom})`,
      );

    // Y-axis
    this.yAxis = d3.axisLeft(this.scales.y);

    this.yAxisGroup = this.chartGroup.append("g").attr("class", "y-axis");

    // Axis labels
    this.svg
      .append("text")
      .attr("class", "axis-label")
      .attr("text-anchor", "middle")
      .attr("x", this.options.width / 2)
      .attr("y", this.options.height - 5)
      .text("Time");

    this.svg
      .append("text")
      .attr("class", "axis-label")
      .attr("text-anchor", "middle")
      .attr("transform", "rotate(-90)")
      .attr("x", -this.options.height / 2)
      .attr("y", 15)
      .text("Value");
  }

  createTooltip() {
    if (!this.options.showTooltip) return;

    this.tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "chart-tooltip")
      .style("position", "absolute")
      .style("visibility", "hidden")
      .style("background", "rgba(0, 0, 0, 0.8)")
      .style("color", "white")
      .style("padding", "8px 12px")
      .style("border-radius", "4px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("z-index", "1000");
  }

  setupInteractions() {
    if (!this.options.interactive) return;

    // Zoom and pan
    this.zoom = d3
      .zoom()
      .scaleExtent([0.5, 10])
      .on("zoom", (event) => this.handleZoom(event));

    this.svg.call(this.zoom);

    // Brush for selection
    this.brush = d3
      .brushX()
      .extent([
        [0, 0],
        [
          this.options.width -
            this.options.margin.left -
            this.options.margin.right,
          this.options.height -
            this.options.margin.top -
            this.options.margin.bottom,
        ],
      ])
      .on("end", (event) => this.handleBrush(event));

    this.brushGroup = this.chartGroup.append("g").attr("class", "brush");
  }

  setupResize() {
    const resizeObserver = new ResizeObserver(() => {
      this.resize();
    });

    resizeObserver.observe(this.container.node());
  }

  setData(data) {
    this.data = data.map((d) => ({
      ...d,
      timestamp: new Date(d.timestamp),
      value: +d.value,
      isAnomaly: d.isAnomaly || false,
      confidence: d.confidence || 0,
    }));

    this.updateScales();
    this.render();
  }

  updateScales() {
    if (this.data.length === 0) return;

    // Update domains based on data
    this.scales.x.domain(d3.extent(this.data, (d) => d.timestamp));
    this.scales.y.domain(d3.extent(this.data, (d) => d.value));
  }

  render() {
    this.renderAxes();
    this.renderData();
    this.renderLegend();
    this.renderInsights();
  }

  renderAxes() {
    this.xAxisGroup
      .transition()
      .duration(this.options.animationDuration)
      .call(this.xAxis);

    this.yAxisGroup
      .transition()
      .duration(this.options.animationDuration)
      .call(this.yAxis);
  }

  renderData() {
    // Create line generator
    const line = d3
      .line()
      .x((d) => this.scales.x(d.timestamp))
      .y((d) => this.scales.y(d.value))
      .curve(d3.curveMonotoneX);

    // Render baseline
    const baseline = this.dataGroup.selectAll(".baseline").data([this.data]);

    baseline
      .enter()
      .append("path")
      .attr("class", "baseline")
      .attr("fill", "none")
      .attr("stroke", "#64748b")
      .attr("stroke-width", 1)
      .attr("opacity", 0.5)
      .merge(baseline)
      .transition()
      .duration(this.options.animationDuration)
      .attr("d", line);

    // Render data points
    const points = this.dataGroup
      .selectAll(".data-point")
      .data(this.data, (d) => d.id || d.timestamp);

    const pointsEnter = points
      .enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("r", 0)
      .attr("cx", (d) => this.scales.x(d.timestamp))
      .attr("cy", (d) => this.scales.y(d.value))
      .attr("fill", (d) =>
        d.isAnomaly
          ? this.scales.color("anomaly")
          : this.scales.color("normal"),
      )
      .attr("opacity", 0);

    pointsEnter
      .merge(points)
      .on("mouseover", (event, d) => this.showTooltip(event, d))
      .on("mouseout", () => this.hideTooltip())
      .on("click", (event, d) => this.handlePointClick(event, d))
      .transition()
      .duration(this.options.animationDuration)
      .attr("cx", (d) => this.scales.x(d.timestamp))
      .attr("cy", (d) => this.scales.y(d.value))
      .attr("r", (d) => (d.isAnomaly ? this.scales.size(d.confidence) : 3))
      .attr("fill", (d) =>
        d.isAnomaly
          ? this.scales.color("anomaly")
          : this.scales.color("normal"),
      )
      .attr("opacity", (d) => (d.isAnomaly ? 0.9 : 0.7));

    points
      .exit()
      .transition()
      .duration(this.options.animationDuration / 2)
      .attr("r", 0)
      .attr("opacity", 0)
      .remove();

    // Render anomaly annotations
    this.renderAnomalyAnnotations();
  }

  renderAnomalyAnnotations() {
    const anomalies = this.data.filter((d) => d.isAnomaly);

    const annotations = this.dataGroup
      .selectAll(".anomaly-annotation")
      .data(anomalies, (d) => d.id || d.timestamp);

    const annotationsEnter = annotations
      .enter()
      .append("g")
      .attr("class", "anomaly-annotation")
      .attr(
        "transform",
        (d) =>
          `translate(${this.scales.x(d.timestamp)}, ${this.scales.y(d.value)})`,
      );

    // Add vertical line
    annotationsEnter
      .append("line")
      .attr("class", "annotation-line")
      .attr("x1", 0)
      .attr("x2", 0)
      .attr("y1", 0)
      .attr("y2", -20)
      .attr("stroke", this.scales.color("anomaly"))
      .attr("stroke-width", 2)
      .attr("opacity", 0);

    // Add confidence badge
    annotationsEnter
      .append("rect")
      .attr("class", "confidence-badge")
      .attr("x", -15)
      .attr("y", -35)
      .attr("width", 30)
      .attr("height", 15)
      .attr("rx", 3)
      .attr("fill", this.scales.color("anomaly"))
      .attr("opacity", 0);

    annotationsEnter
      .append("text")
      .attr("class", "confidence-text")
      .attr("text-anchor", "middle")
      .attr("x", 0)
      .attr("y", -25)
      .attr("fill", "white")
      .attr("font-size", "10px")
      .attr("opacity", 0)
      .text((d) => `${Math.round(d.confidence * 100)}%`);

    // Update positions
    annotationsEnter
      .merge(annotations)
      .transition()
      .duration(this.options.animationDuration)
      .attr(
        "transform",
        (d) =>
          `translate(${this.scales.x(d.timestamp)}, ${this.scales.y(d.value)})`,
      )
      .selectAll(".annotation-line, .confidence-badge, .confidence-text")
      .attr("opacity", 0.8);

    annotations
      .exit()
      .transition()
      .duration(this.options.animationDuration / 2)
      .attr("opacity", 0)
      .remove();
  }

  renderLegend() {
    if (!this.options.showLegend) return;

    const legend = this.svg.selectAll(".legend").data([
      { label: "Normal", color: this.scales.color("normal"), shape: "circle" },
      {
        label: "Anomaly",
        color: this.scales.color("anomaly"),
        shape: "circle",
      },
    ]);

    const legendEnter = legend
      .enter()
      .append("g")
      .attr("class", "legend")
      .attr(
        "transform",
        (d, i) => `translate(${this.options.width - 70}, ${30 + i * 20})`,
      );

    legendEnter
      .append("circle")
      .attr("r", 5)
      .attr("fill", (d) => d.color);

    legendEnter
      .append("text")
      .attr("x", 12)
      .attr("y", 0)
      .attr("dy", "0.35em")
      .attr("font-size", "12px")
      .text((d) => d.label);
  }

  renderInsights() {
    const anomalyCount = this.data.filter((d) => d.isAnomaly).length;
    const totalCount = this.data.length;
    const anomalyRate =
      totalCount > 0 ? ((anomalyCount / totalCount) * 100).toFixed(1) : 0;

    // Add insights text
    const insights = this.svg.selectAll(".insights").data([
      {
        text: `${anomalyCount} anomalies detected (${anomalyRate}% of data)`,
        x: this.options.margin.left,
        y: 15,
      },
    ]);

    insights
      .enter()
      .append("text")
      .attr("class", "insights")
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y)
      .attr("font-size", "12px")
      .attr("fill", "#64748b")
      .text((d) => d.text);

    insights.text((d) => d.text);
  }

  showTooltip(event, data) {
    if (!this.tooltip) return;

    const formatTime = d3.timeFormat("%Y-%m-%d %H:%M:%S");
    const formatValue = d3.format(".2f");

    let content = `
      <div><strong>Time:</strong> ${formatTime(data.timestamp)}</div>
      <div><strong>Value:</strong> ${formatValue(data.value)}</div>
      <div><strong>Status:</strong> ${data.isAnomaly ? "Anomaly" : "Normal"}</div>
    `;

    if (data.isAnomaly) {
      content += `<div><strong>Confidence:</strong> ${Math.round(data.confidence * 100)}%</div>`;
    }

    this.tooltip
      .style("visibility", "visible")
      .html(content)
      .style("left", event.pageX + 10 + "px")
      .style("top", event.pageY - 10 + "px");
  }

  hideTooltip() {
    if (this.tooltip) {
      this.tooltip.style("visibility", "hidden");
    }
  }

  handleZoom(event) {
    const { transform } = event;

    // Update x scale with zoom transform
    const newXScale = transform.rescaleX(this.scales.x);

    // Update axes
    this.xAxisGroup.call(this.xAxis.scale(newXScale));

    // Update data points
    this.dataGroup
      .selectAll(".data-point")
      .attr("cx", (d) => newXScale(d.timestamp));

    // Update baseline
    const line = d3
      .line()
      .x((d) => newXScale(d.timestamp))
      .y((d) => this.scales.y(d.value))
      .curve(d3.curveMonotoneX);

    this.dataGroup.selectAll(".baseline").attr("d", line);

    // Update annotations
    this.dataGroup
      .selectAll(".anomaly-annotation")
      .attr(
        "transform",
        (d) =>
          `translate(${newXScale(d.timestamp)}, ${this.scales.y(d.value)})`,
      );
  }

  handleBrush(event) {
    if (!event.selection) return;

    const [x0, x1] = event.selection.map(this.scales.x.invert);

    // Emit selection event
    this.container.node().dispatchEvent(
      new CustomEvent("timeRangeSelected", {
        detail: { startTime: x0, endTime: x1 },
      }),
    );
  }

  handlePointClick(event, data) {
    // Emit point click event
    this.container.node().dispatchEvent(
      new CustomEvent("pointClicked", {
        detail: { data, event },
      }),
    );
  }

  enableBrush() {
    this.brushGroup.call(this.brush);
  }

  disableBrush() {
    this.brushGroup.call(this.brush.clear);
  }

  resize() {
    const containerRect = this.container.node().getBoundingClientRect();
    const newWidth = containerRect.width;
    const newHeight = Math.max(300, containerRect.height);

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

  addDataPoint(dataPoint) {
    const formattedPoint = {
      ...dataPoint,
      timestamp: new Date(dataPoint.timestamp),
      value: +dataPoint.value,
      isAnomaly: dataPoint.isAnomaly || false,
      confidence: dataPoint.confidence || 0,
    };

    this.data.push(formattedPoint);

    // Keep only last N points for performance
    if (this.data.length > 1000) {
      this.data = this.data.slice(-1000);
    }

    this.updateScales();
    this.render();
  }

  updateData(newData) {
    this.setData(newData);
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

      // Create download link
      const downloadLink = document.createElement("a");
      downloadLink.download = "anomaly-timeline.png";
      downloadLink.href = pngFile;
      downloadLink.click();
    };

    img.src = "data:image/svg+xml;base64," + btoa(svgData);
  }

  exportAsSVG() {
    const svgElement = this.svg.node();
    const svgData = new XMLSerializer().serializeToString(svgElement);
    const blob = new Blob([svgData], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);

    const downloadLink = document.createElement("a");
    downloadLink.download = "anomaly-timeline.svg";
    downloadLink.href = url;
    downloadLink.click();

    URL.revokeObjectURL(url);
  }
}

// Factory function for easy instantiation
export function createAnomalyTimeline(container, options = {}) {
  return new AnomalyTimelineChart(container, options);
}

// Example usage
export function initializeChart() {
  const container = document.querySelector('[data-component="timeline-chart"]');
  if (!container) return;

  const chart = new AnomalyTimelineChart(container, {
    width: container.clientWidth,
    height: 400,
    interactive: true,
    showLegend: true,
    realTimeUpdates: true,
  });

  // Example data
  const sampleData = generateSampleData();
  chart.setData(sampleData);

  return chart;
}

function generateSampleData() {
  const data = [];
  const now = new Date();
  const oneDay = 24 * 60 * 60 * 1000;

  for (let i = 0; i < 100; i++) {
    const timestamp = new Date(now - (100 - i) * oneDay);
    const baseValue = 50 + Math.sin(i * 0.1) * 20;
    const noise = (Math.random() - 0.5) * 10;
    const value = baseValue + noise;

    // Randomly mark some points as anomalies
    const isAnomaly = Math.random() < 0.1; // 10% anomaly rate
    const confidence = isAnomaly ? 0.7 + Math.random() * 0.3 : 0;

    data.push({
      id: i,
      timestamp,
      value: isAnomaly ? value + (Math.random() > 0.5 ? 30 : -30) : value,
      isAnomaly,
      confidence,
    });
  }

  return data;
}
