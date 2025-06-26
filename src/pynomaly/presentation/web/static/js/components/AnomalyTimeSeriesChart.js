/**
 * Advanced Time Series Chart for Anomaly Detection
 * 
 * Features:
 * - Real-time data updates with smooth transitions
 * - Interactive brushing and zooming
 * - Multiple data series with anomaly highlighting
 * - Accessibility-first design with ARIA support
 * - Responsive design with container queries
 * - Performance optimized for large datasets
 */

class AnomalyTimeSeriesChart {
  constructor(container, options = {}) {
    this.container = d3.select(container);
    this.options = {
      margin: { top: 20, right: 80, bottom: 40, left: 60 },
      height: 400,
      animationDuration: 300,
      maxDataPoints: 1000,
      realTimeInterval: 5000,
      anomalyThreshold: 2.0,
      colors: {
        normal: 'var(--color-primary-500)',
        anomaly: 'var(--color-danger-500)',
        warning: 'var(--color-warning-500)',
        background: 'var(--color-bg-primary)',
        grid: 'var(--color-border-light)'
      },
      ...options
    };

    this.data = [];
    this.anomalies = [];
    this.isRealTime = false;
    this.realTimeTimer = null;
    
    this.init();
    this.setupAccessibility();
    this.bindEvents();
  }

  init() {
    // Calculate dimensions
    this.updateDimensions();
    
    // Create SVG
    this.svg = this.container
      .append('svg')
      .attr('class', 'anomaly-chart')
      .attr('role', 'img')
      .attr('aria-labelledby', 'chart-title')
      .attr('aria-describedby', 'chart-description');

    // Create chart groups
    this.chartGroup = this.svg.append('g')
      .attr('class', 'chart-group')
      .attr('transform', `translate(${this.options.margin.left}, ${this.options.margin.top})`);

    // Create scales
    this.xScale = d3.scaleTime()
      .range([0, this.width]);
    
    this.yScale = d3.scaleLinear()
      .range([this.height, 0]);

    // Create axes
    this.xAxis = d3.axisBottom(this.xScale)
      .tickFormat(d3.timeFormat('%H:%M'))
      .tickSizeOuter(0);
    
    this.yAxis = d3.axisLeft(this.yScale)
      .tickSizeOuter(0);

    // Create axis groups
    this.chartGroup.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${this.height})`);

    this.chartGroup.append('g')
      .attr('class', 'y-axis');

    // Create grid
    this.createGrid();

    // Create line generators
    this.line = d3.line()
      .x(d => this.xScale(d.timestamp))
      .y(d => this.yScale(d.value))
      .curve(d3.curveMonotoneX);

    // Create brush for zooming
    this.brush = d3.brushX()
      .extent([[0, 0], [this.width, this.height]])
      .on('brush end', this.handleBrush.bind(this));

    this.chartGroup.append('g')
      .attr('class', 'brush')
      .call(this.brush);

    // Create tooltip
    this.createTooltip();

    // Create legend
    this.createLegend();

    // Create zoom behavior
    this.zoom = d3.zoom()
      .scaleExtent([1, 10])
      .extent([[0, 0], [this.width, this.height]])
      .on('zoom', this.handleZoom.bind(this));

    this.svg.call(this.zoom);
  }

  updateDimensions() {
    const containerRect = this.container.node().getBoundingClientRect();
    this.fullWidth = containerRect.width || 800;
    this.fullHeight = this.options.height;
    this.width = this.fullWidth - this.options.margin.left - this.options.margin.right;
    this.height = this.fullHeight - this.options.margin.top - this.options.margin.bottom;

    if (this.svg) {
      this.svg
        .attr('width', this.fullWidth)
        .attr('height', this.fullHeight);
      
      this.xScale.range([0, this.width]);
      this.yScale.range([this.height, 0]);
    }
  }

  createGrid() {
    // Horizontal grid lines
    this.chartGroup.append('g')
      .attr('class', 'grid grid-horizontal')
      .attr('transform', `translate(0, ${this.height})`)
      .call(d3.axisBottom(this.xScale)
        .tickSize(-this.height)
        .tickFormat('')
      );

    // Vertical grid lines
    this.chartGroup.append('g')
      .attr('class', 'grid grid-vertical')
      .call(d3.axisLeft(this.yScale)
        .tickSize(-this.width)
        .tickFormat('')
      );
  }

  createTooltip() {
    this.tooltip = d3.select('body')
      .append('div')
      .attr('class', 'chart-tooltip')
      .attr('role', 'tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background', 'var(--color-bg-inverse)')
      .style('color', 'var(--color-text-inverse)')
      .style('padding', '8px 12px')
      .style('border-radius', 'var(--border-radius-md)')
      .style('font-size', 'var(--font-size-sm)')
      .style('box-shadow', 'var(--shadow-lg)')
      .style('pointer-events', 'none')
      .style('z-index', 'var(--z-index-tooltip)');
  }

  createLegend() {
    const legend = this.svg.append('g')
      .attr('class', 'chart-legend')
      .attr('transform', `translate(${this.fullWidth - this.options.margin.right + 10}, 30)`);

    const legendItems = [
      { label: 'Normal', color: this.options.colors.normal, type: 'line' },
      { label: 'Anomaly', color: this.options.colors.anomaly, type: 'circle' },
      { label: 'Warning', color: this.options.colors.warning, type: 'circle' }
    ];

    const legendItem = legend.selectAll('.legend-item')
      .data(legendItems)
      .enter()
      .append('g')
      .attr('class', 'legend-item')
      .attr('transform', (d, i) => `translate(0, ${i * 20})`);

    legendItem.append('circle')
      .attr('cx', 6)
      .attr('cy', 0)
      .attr('r', 4)
      .style('fill', d => d.color);

    legendItem.append('text')
      .attr('x', 16)
      .attr('y', 0)
      .attr('dy', '0.35em')
      .style('font-size', 'var(--font-size-sm)')
      .style('fill', 'var(--color-text-primary)')
      .text(d => d.label);
  }

  setupAccessibility() {
    // Add chart title and description
    this.container.insert('h3', ':first-child')
      .attr('id', 'chart-title')
      .attr('class', 'sr-only')
      .text('Anomaly Detection Time Series Chart');

    this.container.insert('div', ':first-child')
      .attr('id', 'chart-description')
      .attr('class', 'sr-only')
      .text('Interactive time series chart showing data points and detected anomalies over time. Use keyboard navigation to explore data points.');

    // Add keyboard navigation
    this.svg
      .attr('tabindex', '0')
      .attr('role', 'application')
      .attr('aria-label', 'Time series chart with anomaly detection')
      .on('keydown', this.handleKeydown.bind(this));
  }

  bindEvents() {
    // Resize observer for responsive design
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(() => {
        this.updateDimensions();
        this.render();
      });
      this.resizeObserver.observe(this.container.node());
    }

    // Window resize fallback
    window.addEventListener('resize', this.debounce(() => {
      this.updateDimensions();
      this.render();
    }, 250));
  }

  setData(data, anomalies = []) {
    this.data = data.map(d => ({
      ...d,
      timestamp: new Date(d.timestamp),
      value: +d.value
    }));
    
    this.anomalies = anomalies.map(a => ({
      ...a,
      timestamp: new Date(a.timestamp),
      value: +a.value
    }));

    this.updateScales();
    this.render();
    this.updateAccessibilityData();
  }

  updateScales() {
    if (this.data.length === 0) return;

    // Update x scale domain
    const timeExtent = d3.extent(this.data, d => d.timestamp);
    this.xScale.domain(timeExtent);

    // Update y scale domain with padding
    const valueExtent = d3.extent(this.data, d => d.value);
    const padding = (valueExtent[1] - valueExtent[0]) * 0.1;
    this.yScale.domain([
      valueExtent[0] - padding,
      valueExtent[1] + padding
    ]);
  }

  render() {
    if (this.data.length === 0) return;

    const t = d3.transition()
      .duration(this.options.animationDuration)
      .ease(d3.easeQuadOut);

    // Update axes
    this.chartGroup.select('.x-axis')
      .transition(t)
      .call(this.xAxis);

    this.chartGroup.select('.y-axis')
      .transition(t)
      .call(this.yAxis);

    // Update grid
    this.chartGroup.select('.grid-horizontal')
      .transition(t)
      .call(d3.axisBottom(this.xScale)
        .tickSize(-this.height)
        .tickFormat('')
      );

    this.chartGroup.select('.grid-vertical')
      .transition(t)
      .call(d3.axisLeft(this.yScale)
        .tickSize(-this.width)
        .tickFormat('')
      );

    // Render main data line
    this.renderDataLine(t);

    // Render anomaly points
    this.renderAnomalies(t);

    // Render data points for interaction
    this.renderDataPoints(t);
  }

  renderDataLine(transition) {
    const line = this.chartGroup.selectAll('.data-line')
      .data([this.data]);

    line.enter()
      .append('path')
      .attr('class', 'data-line')
      .style('fill', 'none')
      .style('stroke', this.options.colors.normal)
      .style('stroke-width', 2)
      .style('opacity', 0)
      .merge(line)
      .transition(transition)
      .style('opacity', 1)
      .attr('d', this.line);
  }

  renderAnomalies(transition) {
    const anomalyPoints = this.chartGroup.selectAll('.anomaly-point')
      .data(this.anomalies, d => d.timestamp);

    anomalyPoints.enter()
      .append('circle')
      .attr('class', 'anomaly-point')
      .attr('r', 0)
      .style('fill', d => this.getAnomalyColor(d))
      .style('stroke', 'var(--color-bg-primary)')
      .style('stroke-width', 2)
      .merge(anomalyPoints)
      .transition(transition)
      .attr('cx', d => this.xScale(d.timestamp))
      .attr('cy', d => this.yScale(d.value))
      .attr('r', d => this.getAnomalyRadius(d))
      .style('fill', d => this.getAnomalyColor(d));

    anomalyPoints.exit()
      .transition(transition)
      .attr('r', 0)
      .remove();
  }

  renderDataPoints(transition) {
    const dataPoints = this.chartGroup.selectAll('.data-point')
      .data(this.data, d => d.timestamp);

    const pointsEnter = dataPoints.enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('r', 3)
      .style('fill', 'transparent')
      .style('cursor', 'pointer')
      .on('mouseover', this.handlePointHover.bind(this))
      .on('mouseout', this.handlePointOut.bind(this))
      .on('click', this.handlePointClick.bind(this));

    pointsEnter.merge(dataPoints)
      .transition(transition)
      .attr('cx', d => this.xScale(d.timestamp))
      .attr('cy', d => this.yScale(d.value));

    dataPoints.exit()
      .transition(transition)
      .attr('r', 0)
      .remove();
  }

  getAnomalyColor(anomaly) {
    switch (anomaly.severity || 'medium') {
      case 'high':
        return this.options.colors.anomaly;
      case 'medium':
        return this.options.colors.warning;
      case 'low':
        return this.options.colors.warning;
      default:
        return this.options.colors.warning;
    }
  }

  getAnomalyRadius(anomaly) {
    switch (anomaly.severity || 'medium') {
      case 'high':
        return 8;
      case 'medium':
        return 6;
      case 'low':
        return 4;
      default:
        return 6;
    }
  }

  handlePointHover(event, d) {
    const [x, y] = d3.pointer(event, document.body);
    
    this.tooltip
      .style('opacity', 1)
      .style('left', (x + 10) + 'px')
      .style('top', (y - 10) + 'px')
      .html(`
        <div><strong>Time:</strong> ${d3.timeFormat('%Y-%m-%d %H:%M:%S')(d.timestamp)}</div>
        <div><strong>Value:</strong> ${d.value.toFixed(2)}</div>
        ${d.anomalyScore ? `<div><strong>Anomaly Score:</strong> ${d.anomalyScore.toFixed(2)}</div>` : ''}
      `);

    // Highlight point
    d3.select(event.target)
      .style('fill', this.options.colors.primary)
      .attr('r', 5);
  }

  handlePointOut(event, d) {
    this.tooltip.style('opacity', 0);
    
    d3.select(event.target)
      .style('fill', 'transparent')
      .attr('r', 3);
  }

  handlePointClick(event, d) {
    // Emit custom event for point selection
    this.container.node().dispatchEvent(new CustomEvent('pointSelected', {
      detail: { data: d, chart: this }
    }));
  }

  handleBrush(event) {
    if (!event.selection) return;

    const [x0, x1] = event.selection;
    const timeRange = [this.xScale.invert(x0), this.xScale.invert(x1)];
    
    // Emit brush event
    this.container.node().dispatchEvent(new CustomEvent('timeRangeSelected', {
      detail: { range: timeRange, chart: this }
    }));
  }

  handleZoom(event) {
    const newXScale = event.transform.rescaleX(this.xScale);
    const newYScale = event.transform.rescaleY(this.yScale);

    // Update scales
    this.chartGroup.select('.x-axis').call(d3.axisBottom(newXScale));
    this.chartGroup.select('.y-axis').call(d3.axisLeft(newYScale));

    // Update line
    const zoomedLine = d3.line()
      .x(d => newXScale(d.timestamp))
      .y(d => newYScale(d.value))
      .curve(d3.curveMonotoneX);

    this.chartGroup.select('.data-line')
      .attr('d', zoomedLine);

    // Update points
    this.chartGroup.selectAll('.data-point')
      .attr('cx', d => newXScale(d.timestamp))
      .attr('cy', d => newYScale(d.value));

    this.chartGroup.selectAll('.anomaly-point')
      .attr('cx', d => newXScale(d.timestamp))
      .attr('cy', d => newYScale(d.value));
  }

  handleKeydown(event) {
    switch (event.key) {
      case 'ArrowLeft':
        event.preventDefault();
        this.navigateDataPoint(-1);
        break;
      case 'ArrowRight':
        event.preventDefault();
        this.navigateDataPoint(1);
        break;
      case 'Home':
        event.preventDefault();
        this.focusDataPoint(0);
        break;
      case 'End':
        event.preventDefault();
        this.focusDataPoint(this.data.length - 1);
        break;
      case 'Enter':
      case ' ':
        event.preventDefault();
        if (this.focusedPointIndex >= 0) {
          this.handlePointClick(event, this.data[this.focusedPointIndex]);
        }
        break;
    }
  }

  navigateDataPoint(direction) {
    if (this.data.length === 0) return;

    this.focusedPointIndex = Math.max(0, Math.min(
      this.data.length - 1,
      (this.focusedPointIndex || 0) + direction
    ));

    this.focusDataPoint(this.focusedPointIndex);
  }

  focusDataPoint(index) {
    if (index < 0 || index >= this.data.length) return;

    this.focusedPointIndex = index;
    const data = this.data[index];

    // Update ARIA live region
    this.updateAriaLiveRegion(data);

    // Highlight focused point
    this.chartGroup.selectAll('.data-point')
      .style('stroke', (d, i) => i === index ? 'var(--color-primary-500)' : 'none')
      .style('stroke-width', (d, i) => i === index ? 2 : 0);
  }

  updateAriaLiveRegion(data) {
    let liveRegion = this.container.select('.chart-live-region');
    if (liveRegion.empty()) {
      liveRegion = this.container.append('div')
        .attr('class', 'chart-live-region sr-only')
        .attr('aria-live', 'polite')
        .attr('aria-atomic', 'true');
    }

    liveRegion.text(
      `Data point at ${d3.timeFormat('%Y-%m-%d %H:%M:%S')(data.timestamp)}, value ${data.value.toFixed(2)}${
        data.anomalyScore ? `, anomaly score ${data.anomalyScore.toFixed(2)}` : ''
      }`
    );
  }

  updateAccessibilityData() {
    // Update chart description with summary statistics
    const description = this.container.select('#chart-description');
    if (this.data.length > 0) {
      const values = this.data.map(d => d.value);
      const mean = d3.mean(values);
      const min = d3.min(values);
      const max = d3.max(values);
      
      description.text(
        `Time series chart with ${this.data.length} data points and ${this.anomalies.length} anomalies. ` +
        `Value range: ${min.toFixed(2)} to ${max.toFixed(2)}, average: ${mean.toFixed(2)}. ` +
        `Use arrow keys to navigate data points.`
      );
    }
  }

  startRealTime(dataCallback) {
    this.isRealTime = true;
    this.dataCallback = dataCallback;
    
    this.realTimeTimer = setInterval(() => {
      if (this.dataCallback) {
        this.dataCallback()
          .then(newData => {
            this.addRealTimeData(newData);
          })
          .catch(console.error);
      }
    }, this.options.realTimeInterval);
  }

  stopRealTime() {
    this.isRealTime = false;
    if (this.realTimeTimer) {
      clearInterval(this.realTimeTimer);
      this.realTimeTimer = null;
    }
  }

  addRealTimeData(newData) {
    // Add new data points
    const newPoints = newData.map(d => ({
      ...d,
      timestamp: new Date(d.timestamp),
      value: +d.value
    }));

    this.data.push(...newPoints);

    // Limit data points for performance
    if (this.data.length > this.options.maxDataPoints) {
      this.data = this.data.slice(-this.options.maxDataPoints);
    }

    // Update scales and render
    this.updateScales();
    this.render();
    this.updateAccessibilityData();
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
    
    if (this.tooltip) {
      this.tooltip.remove();
    }
    
    this.container.selectAll('*').remove();
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AnomalyTimeSeriesChart;
}