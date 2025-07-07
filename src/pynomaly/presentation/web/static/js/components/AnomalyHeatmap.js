/**
 * Advanced Anomaly Heatmap Component
 * 
 * Features:
 * - Interactive heatmap with hover and click interactions
 * - Multiple color scales for different anomaly types
 * - Zoom and pan capabilities for large datasets
 * - Accessibility support with keyboard navigation
 * - Real-time updates with smooth transitions
 * - Responsive design with container queries
 */

class AnomalyHeatmap {
  constructor(container, options = {}) {
    this.container = d3.select(container);
    this.options = {
      margin: { top: 60, right: 100, bottom: 60, left: 60 },
      cellSize: 20,
      padding: 1,
      animationDuration: 300,
      colorScale: 'viridis',
      thresholds: {
        low: 0.3,
        medium: 0.6,
        high: 0.8
      },
      colors: {
        normal: '#e8f5e8',
        lowAnomaly: '#fff3cd',
        mediumAnomaly: '#ffeaa7',
        highAnomaly: '#fab1a0',
        criticalAnomaly: '#e17055'
      },
      maxZoom: 5,
      minZoom: 0.5,
      ...options
    };

    this.data = [];
    this.xDomain = [];
    this.yDomain = [];
    this.focusedCell = null;
    
    this.init();
    this.setupAccessibility();
    this.bindEvents();
  }

  init() {
    this.updateDimensions();
    
    // Create SVG
    this.svg = this.container
      .append('svg')
      .attr('class', 'anomaly-heatmap')
      .attr('role', 'img')
      .attr('aria-labelledby', 'heatmap-title')
      .attr('aria-describedby', 'heatmap-description');

    // Create main group for zooming/panning
    this.mainGroup = this.svg.append('g')
      .attr('class', 'main-group')
      .attr('transform', `translate(${this.options.margin.left}, ${this.options.margin.top})`);

    // Create scales
    this.xScale = d3.scaleBand()
      .range([0, this.width])
      .padding(this.options.padding / this.options.cellSize);
    
    this.yScale = d3.scaleBand()
      .range([0, this.height])
      .padding(this.options.padding / this.options.cellSize);

    // Create color scale
    this.colorScale = this.createColorScale();

    // Create axes
    this.createAxes();

    // Create zoom behavior
    this.zoom = d3.zoom()
      .scaleExtent([this.options.minZoom, this.options.maxZoom])
      .on('zoom', this.handleZoom.bind(this));

    this.svg.call(this.zoom);

    // Create tooltip
    this.createTooltip();

    // Create legend
    this.createLegend();
  }

  updateDimensions() {
    const containerRect = this.container.node().getBoundingClientRect();
    this.fullWidth = containerRect.width || 800;
    this.fullHeight = containerRect.height || 600;
    this.width = this.fullWidth - this.options.margin.left - this.options.margin.right;
    this.height = this.fullHeight - this.options.margin.top - this.options.margin.bottom;

    if (this.svg) {
      this.svg
        .attr('width', this.fullWidth)
        .attr('height', this.fullHeight);
    }
  }

  createColorScale() {
    return d3.scaleSequential()
      .domain([0, 1])
      .interpolator(d3.interpolateViridis);
  }

  createAxes() {
    // X-axis
    this.xAxisGroup = this.svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(${this.options.margin.left}, ${this.options.margin.top + this.height})`);

    // Y-axis
    this.yAxisGroup = this.svg.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${this.options.margin.left}, ${this.options.margin.top})`);

    // X-axis label
    this.svg.append('text')
      .attr('class', 'axis-label x-axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', this.options.margin.left + this.width / 2)
      .attr('y', this.fullHeight - 10)
      .style('font-size', 'var(--font-size-sm)')
      .style('fill', 'var(--color-text-secondary)')
      .text('Features');

    // Y-axis label
    this.svg.append('text')
      .attr('class', 'axis-label y-axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(this.options.margin.top + this.height / 2))
      .attr('y', 15)
      .style('font-size', 'var(--font-size-sm)')
      .style('fill', 'var(--color-text-secondary)')
      .text('Time Periods');
  }

  createTooltip() {
    this.tooltip = d3.select('body')
      .append('div')
      .attr('class', 'heatmap-tooltip')
      .attr('role', 'tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background', 'var(--color-bg-inverse)')
      .style('color', 'var(--color-text-inverse)')
      .style('padding', '12px')
      .style('border-radius', 'var(--border-radius-md)')
      .style('font-size', 'var(--font-size-sm)')
      .style('box-shadow', 'var(--shadow-lg)')
      .style('pointer-events', 'none')
      .style('z-index', 'var(--z-index-tooltip)');
  }

  createLegend() {
    const legend = this.svg.append('g')
      .attr('class', 'heatmap-legend')
      .attr('transform', `translate(${this.fullWidth - this.options.margin.right + 20}, ${this.options.margin.top})`);

    // Create gradient for continuous legend
    const defs = this.svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'heatmap-gradient')
      .attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', 0).attr('y1', 0)
      .attr('x2', 0).attr('y2', 200);

    // Add color stops
    const stops = d3.range(0, 1.01, 0.1);
    gradient.selectAll('stop')
      .data(stops)
      .enter()
      .append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => this.colorScale(1 - d));

    // Legend rectangle
    legend.append('rect')
      .attr('width', 20)
      .attr('height', 200)
      .style('fill', 'url(#heatmap-gradient)')
      .style('stroke', 'var(--color-border-medium)')
      .style('stroke-width', 1);

    // Legend scale
    const legendScale = d3.scaleLinear()
      .domain([0, 1])
      .range([200, 0]);

    const legendAxis = d3.axisRight(legendScale)
      .tickSize(6)
      .ticks(5)
      .tickFormat(d3.format('.1f'));

    legend.append('g')
      .attr('class', 'legend-axis')
      .attr('transform', 'translate(20, 0)')
      .call(legendAxis);

    // Legend title
    legend.append('text')
      .attr('class', 'legend-title')
      .attr('x', 30)
      .attr('y', -10)
      .style('font-size', 'var(--font-size-sm)')
      .style('font-weight', 'var(--font-weight-medium)')
      .style('fill', 'var(--color-text-primary)')
      .text('Anomaly Score');
  }

  setupAccessibility() {
    // Add heatmap title and description
    this.container.insert('h3', ':first-child')
      .attr('id', 'heatmap-title')
      .attr('class', 'sr-only')
      .text('Anomaly Detection Heatmap');

    this.container.insert('div', ':first-child')
      .attr('id', 'heatmap-description')
      .attr('class', 'sr-only')
      .text('Interactive heatmap showing anomaly scores across features and time periods. Use keyboard navigation to explore cells.');

    // Add keyboard navigation
    this.svg
      .attr('tabindex', '0')
      .attr('role', 'application')
      .attr('aria-label', 'Anomaly detection heatmap with keyboard navigation')
      .on('keydown', this.handleKeydown.bind(this));

    // Add live region for announcements
    this.liveRegion = this.container.append('div')
      .attr('class', 'heatmap-live-region sr-only')
      .attr('aria-live', 'polite')
      .attr('aria-atomic', 'true');
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

  setData(data) {
    this.data = data;
    
    // Extract domains
    this.xDomain = [...new Set(data.map(d => d.feature))].sort();
    this.yDomain = [...new Set(data.map(d => d.timePeriod))].sort();

    // Update scales
    this.xScale.domain(this.xDomain);
    this.yScale.domain(this.yDomain);

    // Update color scale domain
    const scores = data.map(d => d.anomalyScore);
    this.colorScale.domain(d3.extent(scores));

    this.render();
    this.updateAccessibilityData();
  }

  render() {
    if (this.data.length === 0) return;

    const t = d3.transition()
      .duration(this.options.animationDuration)
      .ease(d3.easeQuadOut);

    // Update axes
    this.renderAxes(t);

    // Render heatmap cells
    this.renderCells(t);
  }

  renderAxes(transition) {
    const xAxis = d3.axisBottom(this.xScale)
      .tickSize(0)
      .tickPadding(8);

    const yAxis = d3.axisLeft(this.yScale)
      .tickSize(0)
      .tickPadding(8);

    this.xAxisGroup
      .transition(transition)
      .call(xAxis);

    this.yAxisGroup
      .transition(transition)
      .call(yAxis);

    // Style axis text
    this.xAxisGroup.selectAll('text')
      .style('font-size', 'var(--font-size-xs)')
      .style('fill', 'var(--color-text-secondary)')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');

    this.yAxisGroup.selectAll('text')
      .style('font-size', 'var(--font-size-xs)')
      .style('fill', 'var(--color-text-secondary)');

    // Remove axis lines
    this.xAxisGroup.select('.domain').remove();
    this.yAxisGroup.select('.domain').remove();
  }

  renderCells(transition) {
    const cells = this.mainGroup.selectAll('.heatmap-cell')
      .data(this.data, d => `${d.feature}-${d.timePeriod}`);

    const cellsEnter = cells.enter()
      .append('rect')
      .attr('class', 'heatmap-cell')
      .attr('x', d => this.xScale(d.feature))
      .attr('y', d => this.yScale(d.timePeriod))
      .attr('width', this.xScale.bandwidth())
      .attr('height', this.yScale.bandwidth())
      .style('fill', d => this.colorScale(d.anomalyScore))
      .style('stroke', 'var(--color-bg-primary)')
      .style('stroke-width', 1)
      .style('cursor', 'pointer')
      .style('opacity', 0)
      .on('mouseover', this.handleCellHover.bind(this))
      .on('mouseout', this.handleCellOut.bind(this))
      .on('click', this.handleCellClick.bind(this));

    cellsEnter.merge(cells)
      .transition(transition)
      .style('opacity', 1)
      .style('fill', d => this.colorScale(d.anomalyScore))
      .attr('x', d => this.xScale(d.feature))
      .attr('y', d => this.yScale(d.timePeriod))
      .attr('width', this.xScale.bandwidth())
      .attr('height', this.yScale.bandwidth());

    cells.exit()
      .transition(transition)
      .style('opacity', 0)
      .remove();
  }

  handleCellHover(event, d) {
    const [x, y] = d3.pointer(event, document.body);
    
    this.tooltip
      .style('opacity', 1)
      .style('left', (x + 10) + 'px')
      .style('top', (y - 10) + 'px')
      .html(`
        <div><strong>Feature:</strong> ${d.feature}</div>
        <div><strong>Time Period:</strong> ${d.timePeriod}</div>
        <div><strong>Anomaly Score:</strong> ${d.anomalyScore.toFixed(3)}</div>
        <div><strong>Severity:</strong> ${this.getAnomalySeverity(d.anomalyScore)}</div>
        ${d.value !== undefined ? `<div><strong>Value:</strong> ${d.value.toFixed(2)}</div>` : ''}
      `);

    // Highlight cell
    d3.select(event.target)
      .style('stroke', 'var(--color-primary-500)')
      .style('stroke-width', 2);

    // Highlight row and column
    this.highlightRowColumn(d.feature, d.timePeriod);
  }

  handleCellOut(event, d) {
    this.tooltip.style('opacity', 0);
    
    d3.select(event.target)
      .style('stroke', 'var(--color-bg-primary)')
      .style('stroke-width', 1);

    // Remove highlights
    this.removeHighlights();
  }

  handleCellClick(event, d) {
    // Set focus for keyboard navigation
    this.focusedCell = d;
    this.updateCellFocus();

    // Emit custom event
    this.container.node().dispatchEvent(new CustomEvent('cellSelected', {
      detail: { data: d, heatmap: this }
    }));
  }

  highlightRowColumn(feature, timePeriod) {
    // Highlight row
    this.mainGroup.selectAll('.heatmap-cell')
      .filter(d => d.timePeriod === timePeriod)
      .style('opacity', 0.7);

    // Highlight column
    this.mainGroup.selectAll('.heatmap-cell')
      .filter(d => d.feature === feature)
      .style('opacity', 0.7);
  }

  removeHighlights() {
    this.mainGroup.selectAll('.heatmap-cell')
      .style('opacity', 1);
  }

  getAnomalySeverity(score) {
    const thresholds = this.options.thresholds;
    if (score >= thresholds.high) return 'High';
    if (score >= thresholds.medium) return 'Medium';
    if (score >= thresholds.low) return 'Low';
    return 'Normal';
  }

  handleKeydown(event) {
    if (!this.focusedCell && this.data.length > 0) {
      this.focusedCell = this.data[0];
    }

    if (!this.focusedCell) return;

    const currentFeatureIndex = this.xDomain.indexOf(this.focusedCell.feature);
    const currentTimeIndex = this.yDomain.indexOf(this.focusedCell.timePeriod);

    switch (event.key) {
      case 'ArrowLeft':
        event.preventDefault();
        if (currentFeatureIndex > 0) {
          this.focusCell(this.xDomain[currentFeatureIndex - 1], this.focusedCell.timePeriod);
        }
        break;
      case 'ArrowRight':
        event.preventDefault();
        if (currentFeatureIndex < this.xDomain.length - 1) {
          this.focusCell(this.xDomain[currentFeatureIndex + 1], this.focusedCell.timePeriod);
        }
        break;
      case 'ArrowUp':
        event.preventDefault();
        if (currentTimeIndex > 0) {
          this.focusCell(this.focusedCell.feature, this.yDomain[currentTimeIndex - 1]);
        }
        break;
      case 'ArrowDown':
        event.preventDefault();
        if (currentTimeIndex < this.yDomain.length - 1) {
          this.focusCell(this.focusedCell.feature, this.yDomain[currentTimeIndex + 1]);
        }
        break;
      case 'Home':
        event.preventDefault();
        this.focusCell(this.xDomain[0], this.focusedCell.timePeriod);
        break;
      case 'End':
        event.preventDefault();
        this.focusCell(this.xDomain[this.xDomain.length - 1], this.focusedCell.timePeriod);
        break;
      case 'Enter':
      case ' ':
        event.preventDefault();
        this.handleCellClick(event, this.focusedCell);
        break;
    }
  }

  focusCell(feature, timePeriod) {
    const cellData = this.data.find(d => d.feature === feature && d.timePeriod === timePeriod);
    if (cellData) {
      this.focusedCell = cellData;
      this.updateCellFocus();
      this.announceCell(cellData);
    }
  }

  updateCellFocus() {
    if (!this.focusedCell) return;

    // Remove previous focus
    this.mainGroup.selectAll('.heatmap-cell')
      .style('stroke', 'var(--color-bg-primary)')
      .style('stroke-width', 1);

    // Add focus to current cell
    this.mainGroup.selectAll('.heatmap-cell')
      .filter(d => d.feature === this.focusedCell.feature && d.timePeriod === this.focusedCell.timePeriod)
      .style('stroke', 'var(--color-focus)')
      .style('stroke-width', 3);
  }

  announceCell(data) {
    this.liveRegion.text(
      `Cell at feature ${data.feature}, time period ${data.timePeriod}. ` +
      `Anomaly score ${data.anomalyScore.toFixed(3)}, severity ${this.getAnomalySeverity(data.anomalyScore)}.`
    );
  }

  updateAccessibilityData() {
    if (this.data.length === 0) return;

    const scores = this.data.map(d => d.anomalyScore);
    const avgScore = d3.mean(scores);
    const maxScore = d3.max(scores);
    const anomalyCells = scores.filter(s => s > this.options.thresholds.low).length;

    const description = this.container.select('#heatmap-description');
    description.text(
      `Heatmap with ${this.xDomain.length} features and ${this.yDomain.length} time periods. ` +
      `${anomalyCells} cells show anomalous behavior. Average anomaly score: ${avgScore.toFixed(3)}, ` +
      `maximum: ${maxScore.toFixed(3)}. Use arrow keys to navigate cells.`
    );
  }

  handleZoom(event) {
    const { transform } = event;
    
    // Apply transform to main group
    this.mainGroup.attr('transform', transform);
    
    // Update scales for correct mouse positioning
    const newXScale = transform.rescaleX(this.xScale);
    const newYScale = transform.rescaleY(this.yScale);
    
    // Update axes
    this.xAxisGroup.call(d3.axisBottom(newXScale).tickSize(0));
    this.yAxisGroup.call(d3.axisLeft(newYScale).tickSize(0));
  }

  resetZoom() {
    this.svg.transition()
      .duration(750)
      .call(this.zoom.transform, d3.zoomIdentity);
  }

  exportData() {
    return {
      data: this.data,
      summary: {
        features: this.xDomain.length,
        timePeriods: this.yDomain.length,
        totalCells: this.data.length,
        averageScore: d3.mean(this.data.map(d => d.anomalyScore)),
        maxScore: d3.max(this.data.map(d => d.anomalyScore)),
        anomalyCells: this.data.filter(d => d.anomalyScore > this.options.thresholds.low).length
      }
    };
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
  module.exports = AnomalyHeatmap;
}
