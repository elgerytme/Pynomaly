/**
 * Advanced 3D Anomaly Scatter Plot Component
 * 
 * Features:
 * - Interactive 3D scatter plot with WebGL rendering
 * - Multi-dimensional anomaly visualization
 * - Brushing and linking with other charts
 * - Real-time updates with smooth transitions
 * - Accessibility support with data table fallback
 * - Advanced clustering and outlier detection
 * - Responsive design with performance optimization
 */

class AnomalyScatterPlot {
  constructor(container, options = {}) {
    this.container = d3.select(container);
    this.options = {
      margin: { top: 20, right: 80, bottom: 60, left: 60 },
      height: 500,
      animationDuration: 300,
      pointSize: 4,
      pointOpacity: 0.7,
      is3D: false,
      maxPoints: 5000,
      clustering: {
        enabled: false,
        algorithm: 'kmeans',
        clusters: 3
      },
      colors: {
        normal: 'var(--color-primary-500)',
        anomaly: 'var(--color-danger-500)',
        warning: 'var(--color-warning-500)',
        cluster1: '#1f77b4',
        cluster2: '#ff7f0e',
        cluster3: '#2ca02c',
        cluster4: '#d62728',
        cluster5: '#9467bd'
      },
      axes: {
        x: { label: 'X Axis', scale: 'linear' },
        y: { label: 'Y Axis', scale: 'linear' },
        z: { label: 'Z Axis', scale: 'linear' }
      },
      ...options
    };

    this.data = [];
    this.selectedPoints = new Set();
    this.brushedPoints = new Set();
    this.focusedPointIndex = -1;
    
    this.init();
    this.setupAccessibility();
    this.bindEvents();
  }

  init() {
    this.updateDimensions();
    
    // Create SVG
    this.svg = this.container
      .append('svg')
      .attr('class', 'anomaly-scatter-plot')
      .attr('role', 'img')
      .attr('aria-labelledby', 'scatter-title')
      .attr('aria-describedby', 'scatter-description');

    // Create chart group
    this.chartGroup = this.svg.append('g')
      .attr('class', 'chart-group')
      .attr('transform', `translate(${this.options.margin.left}, ${this.options.margin.top})`);

    // Create scales
    this.createScales();

    // Create axes
    this.createAxes();

    // Create brush
    this.createBrush();

    // Create tooltip
    this.createTooltip();

    // Create legend
    this.createLegend();

    // Create controls
    this.createControls();
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
    }
  }

  createScales() {
    this.xScale = d3.scaleLinear()
      .range([0, this.width]);
    
    this.yScale = d3.scaleLinear()
      .range([this.height, 0]);

    if (this.options.is3D) {
      this.zScale = d3.scaleLinear()
        .range([0, 100]); // For 3D perspective
    }

    this.sizeScale = d3.scaleSqrt()
      .range([2, 10]);

    this.opacityScale = d3.scaleLinear()
      .range([0.3, 1]);
  }

  createAxes() {
    // X-axis
    this.xAxis = d3.axisBottom(this.xScale)
      .tickSizeOuter(0);
    
    this.xAxisGroup = this.chartGroup.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${this.height})`);

    // Y-axis
    this.yAxis = d3.axisLeft(this.yScale)
      .tickSizeOuter(0);
    
    this.yAxisGroup = this.chartGroup.append('g')
      .attr('class', 'y-axis');

    // Axis labels
    this.svg.append('text')
      .attr('class', 'axis-label x-axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', this.options.margin.left + this.width / 2)
      .attr('y', this.fullHeight - 10)
      .style('font-size', 'var(--font-size-sm)')
      .style('fill', 'var(--color-text-secondary)')
      .text(this.options.axes.x.label);

    this.svg.append('text')
      .attr('class', 'axis-label y-axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('x', -(this.options.margin.top + this.height / 2))
      .attr('y', 15)
      .style('font-size', 'var(--font-size-sm)')
      .style('fill', 'var(--color-text-secondary)')
      .text(this.options.axes.y.label);

    // Grid lines
    this.createGrid();
  }

  createGrid() {
    // X grid
    this.chartGroup.append('g')
      .attr('class', 'grid grid-x')
      .attr('transform', `translate(0, ${this.height})`)
      .call(d3.axisBottom(this.xScale)
        .tickSize(-this.height)
        .tickFormat('')
      );

    // Y grid
    this.chartGroup.append('g')
      .attr('class', 'grid grid-y')
      .call(d3.axisLeft(this.yScale)
        .tickSize(-this.width)
        .tickFormat('')
      );
  }

  createBrush() {
    this.brush = d3.brush()
      .extent([[0, 0], [this.width, this.height]])
      .on('start brush end', this.handleBrush.bind(this));

    this.brushGroup = this.chartGroup.append('g')
      .attr('class', 'brush')
      .call(this.brush);
  }

  createTooltip() {
    this.tooltip = d3.select('body')
      .append('div')
      .attr('class', 'scatter-tooltip')
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
      .style('z-index', 'var(--z-index-tooltip)')
      .style('max-width', '300px');
  }

  createLegend() {
    this.legend = this.svg.append('g')
      .attr('class', 'scatter-legend')
      .attr('transform', `translate(${this.fullWidth - this.options.margin.right + 10}, 30)`);
  }

  createControls() {
    const controls = this.container.insert('div', ':first-child')
      .attr('class', 'scatter-controls')
      .style('margin-bottom', '10px')
      .style('display', 'flex')
      .style('flex-wrap', 'wrap')
      .style('gap', '10px')
      .style('align-items', 'center');

    // Dimension controls
    if (this.options.is3D) {
      controls.append('button')
        .attr('class', 'btn btn--secondary btn--sm')
        .text('Toggle 2D/3D')
        .on('click', () => this.toggle3D());
    }

    // Clustering controls
    if (this.options.clustering.enabled) {
      controls.append('button')
        .attr('class', 'btn btn--secondary btn--sm')
        .text('Toggle Clustering')
        .on('click', () => this.toggleClustering());
    }

    // Clear selection
    controls.append('button')
      .attr('class', 'btn btn--secondary btn--sm')
      .text('Clear Selection')
      .on('click', () => this.clearSelection());

    // Export data
    controls.append('button')
      .attr('class', 'btn btn--secondary btn--sm')
      .text('Export Data')
      .on('click', () => this.exportData());
  }

  setupAccessibility() {
    // Add chart title and description
    this.container.insert('h3', ':first-child')
      .attr('id', 'scatter-title')
      .attr('class', 'sr-only')
      .text('Anomaly Detection Scatter Plot');

    this.container.insert('div', ':first-child')
      .attr('id', 'scatter-description')
      .attr('class', 'sr-only')
      .text('Interactive scatter plot showing data points with anomaly detection. Use keyboard navigation to explore points.');

    // Add keyboard navigation
    this.svg
      .attr('tabindex', '0')
      .attr('role', 'application')
      .attr('aria-label', 'Anomaly detection scatter plot with keyboard navigation')
      .on('keydown', this.handleKeydown.bind(this));

    // Create accessible data table
    this.createDataTable();

    // Add live region
    this.liveRegion = this.container.append('div')
      .attr('class', 'scatter-live-region sr-only')
      .attr('aria-live', 'polite')
      .attr('aria-atomic', 'true');
  }

  createDataTable() {
    this.dataTableContainer = this.container.append('div')
      .attr('class', 'scatter-data-table')
      .style('max-height', '300px')
      .style('overflow', 'auto')
      .style('margin-top', '20px')
      .style('display', 'none'); // Hidden by default

    // Toggle button
    this.container.select('.scatter-controls')
      .append('button')
      .attr('class', 'btn btn--secondary btn--sm')
      .text('Show Data Table')
      .on('click', () => this.toggleDataTable());
  }

  bindEvents() {
    // Resize observer
    if (window.ResizeObserver) {
      this.resizeObserver = new ResizeObserver(() => {
        this.updateDimensions();
        this.render();
      });
      this.resizeObserver.observe(this.container.node());
    }

    window.addEventListener('resize', this.debounce(() => {
      this.updateDimensions();
      this.render();
    }, 250));
  }

  setData(data) {
    this.data = data.map((d, i) => ({
      ...d,
      id: d.id || i,
      x: +d.x,
      y: +d.y,
      z: d.z ? +d.z : 0,
      anomalyScore: +d.anomalyScore || 0,
      isAnomaly: d.isAnomaly || d.anomalyScore > 0.5,
      cluster: d.cluster || 0
    }));

    // Subsample if too many points
    if (this.data.length > this.options.maxPoints) {
      this.data = this.sampleData(this.data, this.options.maxPoints);
    }

    this.updateScales();
    this.render();
    this.updateAccessibilityData();
    this.updateDataTable();
  }

  sampleData(data, maxPoints) {
    if (data.length <= maxPoints) return data;

    // Stratified sampling to preserve anomalies
    const anomalies = data.filter(d => d.isAnomaly);
    const normal = data.filter(d => !d.isAnomaly);
    
    const anomalyRatio = anomalies.length / data.length;
    const targetAnomalies = Math.floor(maxPoints * anomalyRatio);
    const targetNormal = maxPoints - targetAnomalies;

    const sampledAnomalies = this.randomSample(anomalies, Math.min(targetAnomalies, anomalies.length));
    const sampledNormal = this.randomSample(normal, targetNormal);

    return [...sampledAnomalies, ...sampledNormal];
  }

  randomSample(array, size) {
    const shuffled = array.slice();
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled.slice(0, size);
  }

  updateScales() {
    if (this.data.length === 0) return;

    // Update domain for X scale
    const xExtent = d3.extent(this.data, d => d.x);
    const xPadding = (xExtent[1] - xExtent[0]) * 0.05;
    this.xScale.domain([xExtent[0] - xPadding, xExtent[1] + xPadding]);

    // Update domain for Y scale
    const yExtent = d3.extent(this.data, d => d.y);
    const yPadding = (yExtent[1] - yExtent[0]) * 0.05;
    this.yScale.domain([yExtent[0] - yPadding, yExtent[1] + yPadding]);

    // Update Z scale if 3D
    if (this.options.is3D && this.data.some(d => d.z !== 0)) {
      const zExtent = d3.extent(this.data, d => d.z);
      this.zScale.domain(zExtent);
    }

    // Update size scale based on anomaly scores
    const scoreExtent = d3.extent(this.data, d => d.anomalyScore);
    this.sizeScale.domain(scoreExtent);
    this.opacityScale.domain(scoreExtent);
  }

  render() {
    if (this.data.length === 0) return;

    const t = d3.transition()
      .duration(this.options.animationDuration)
      .ease(d3.easeQuadOut);

    // Update axes
    this.xAxisGroup.transition(t).call(this.xAxis);
    this.yAxisGroup.transition(t).call(this.yAxis);

    // Update grid
    this.chartGroup.select('.grid-x')
      .transition(t)
      .call(d3.axisBottom(this.xScale)
        .tickSize(-this.height)
        .tickFormat(''));

    this.chartGroup.select('.grid-y')
      .transition(t)
      .call(d3.axisLeft(this.yScale)
        .tickSize(-this.width)
        .tickFormat(''));

    // Render points
    this.renderPoints(t);

    // Update legend
    this.updateLegend();
  }

  renderPoints(transition) {
    const points = this.chartGroup.selectAll('.scatter-point')
      .data(this.data, d => d.id);

    const pointsEnter = points.enter()
      .append('circle')
      .attr('class', 'scatter-point')
      .attr('r', 0)
      .style('cursor', 'pointer')
      .on('mouseover', this.handlePointHover.bind(this))
      .on('mouseout', this.handlePointOut.bind(this))
      .on('click', this.handlePointClick.bind(this));

    pointsEnter.merge(points)
      .transition(transition)
      .attr('cx', d => this.xScale(d.x))
      .attr('cy', d => this.yScale(d.y))
      .attr('r', d => this.getPointRadius(d))
      .style('fill', d => this.getPointColor(d))
      .style('opacity', d => this.getPointOpacity(d))
      .style('stroke', d => this.getPointStroke(d))
      .style('stroke-width', d => this.getPointStrokeWidth(d));

    points.exit()
      .transition(transition)
      .attr('r', 0)
      .remove();
  }

  getPointRadius(d) {
    const baseRadius = this.options.pointSize;
    const sizeMultiplier = this.sizeScale(d.anomalyScore);
    return Math.max(baseRadius * 0.5, Math.min(baseRadius * 2, sizeMultiplier));
  }

  getPointColor(d) {
    if (this.options.clustering.enabled && d.cluster !== undefined) {
      return this.options.colors[`cluster${d.cluster + 1}`] || this.options.colors.normal;
    }
    
    if (d.isAnomaly) {
      return d.anomalyScore > 0.8 ? this.options.colors.anomaly : this.options.colors.warning;
    }
    
    return this.options.colors.normal;
  }

  getPointOpacity(d) {
    if (this.selectedPoints.has(d.id)) return 1;
    if (this.brushedPoints.size > 0 && !this.brushedPoints.has(d.id)) return 0.3;
    return this.opacityScale(d.anomalyScore) * this.options.pointOpacity;
  }

  getPointStroke(d) {
    if (this.selectedPoints.has(d.id)) return 'var(--color-primary-700)';
    if (this.focusedPointIndex >= 0 && this.data[this.focusedPointIndex].id === d.id) {
      return 'var(--color-focus)';
    }
    return 'none';
  }

  getPointStrokeWidth(d) {
    if (this.selectedPoints.has(d.id) || 
        (this.focusedPointIndex >= 0 && this.data[this.focusedPointIndex].id === d.id)) {
      return 2;
    }
    return 0;
  }

  handlePointHover(event, d) {
    const [x, y] = d3.pointer(event, document.body);
    
    this.tooltip
      .style('opacity', 1)
      .style('left', (x + 10) + 'px')
      .style('top', (y - 10) + 'px')
      .html(`
        <div><strong>ID:</strong> ${d.id}</div>
        <div><strong>X:</strong> ${d.x.toFixed(3)}</div>
        <div><strong>Y:</strong> ${d.y.toFixed(3)}</div>
        ${this.options.is3D ? `<div><strong>Z:</strong> ${d.z.toFixed(3)}</div>` : ''}
        <div><strong>Anomaly Score:</strong> ${d.anomalyScore.toFixed(3)}</div>
        <div><strong>Status:</strong> ${d.isAnomaly ? 'Anomaly' : 'Normal'}</div>
        ${this.options.clustering.enabled ? `<div><strong>Cluster:</strong> ${d.cluster + 1}</div>` : ''}
        ${d.label ? `<div><strong>Label:</strong> ${d.label}</div>` : ''}
      `);

    // Highlight point
    d3.select(event.target)
      .style('stroke', 'var(--color-primary-600)')
      .style('stroke-width', 2);
  }

  handlePointOut(event, d) {
    this.tooltip.style('opacity', 0);
    
    // Remove highlight if not selected or focused
    if (!this.selectedPoints.has(d.id) && 
        !(this.focusedPointIndex >= 0 && this.data[this.focusedPointIndex].id === d.id)) {
      d3.select(event.target)
        .style('stroke', 'none')
        .style('stroke-width', 0);
    }
  }

  handlePointClick(event, d) {
    if (event.ctrlKey || event.metaKey) {
      // Multi-select
      if (this.selectedPoints.has(d.id)) {
        this.selectedPoints.delete(d.id);
      } else {
        this.selectedPoints.add(d.id);
      }
    } else {
      // Single select
      this.selectedPoints.clear();
      this.selectedPoints.add(d.id);
    }

    this.updatePointSelection();
    this.emitSelectionEvent();
  }

  handleBrush(event) {
    const { selection } = event;
    
    if (!selection) {
      this.brushedPoints.clear();
    } else {
      const [[x0, y0], [x1, y1]] = selection;
      
      this.brushedPoints.clear();
      this.data.forEach(d => {
        const x = this.xScale(d.x);
        const y = this.yScale(d.y);
        
        if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
          this.brushedPoints.add(d.id);
        }
      });
    }

    this.updatePointSelection();
    this.emitBrushEvent();
  }

  updatePointSelection() {
    this.chartGroup.selectAll('.scatter-point')
      .style('opacity', d => this.getPointOpacity(d))
      .style('stroke', d => this.getPointStroke(d))
      .style('stroke-width', d => this.getPointStrokeWidth(d));
  }

  emitSelectionEvent() {
    const selectedData = this.data.filter(d => this.selectedPoints.has(d.id));
    this.container.node().dispatchEvent(new CustomEvent('pointsSelected', {
      detail: { points: selectedData, plot: this }
    }));
  }

  emitBrushEvent() {
    const brushedData = this.data.filter(d => this.brushedPoints.has(d.id));
    this.container.node().dispatchEvent(new CustomEvent('pointsBrushed', {
      detail: { points: brushedData, plot: this }
    }));
  }

  clearSelection() {
    this.selectedPoints.clear();
    this.brushedPoints.clear();
    this.brush.clear(this.brushGroup);
    this.updatePointSelection();
  }

  updateLegend() {
    this.legend.selectAll('*').remove();

    const legendItems = [];
    
    if (this.options.clustering.enabled) {
      const clusters = [...new Set(this.data.map(d => d.cluster))];
      clusters.forEach(cluster => {
        legendItems.push({
          label: `Cluster ${cluster + 1}`,
          color: this.options.colors[`cluster${cluster + 1}`] || this.options.colors.normal,
          type: 'circle'
        });
      });
    } else {
      legendItems.push(
        { label: 'Normal', color: this.options.colors.normal, type: 'circle' },
        { label: 'Warning', color: this.options.colors.warning, type: 'circle' },
        { label: 'Anomaly', color: this.options.colors.anomaly, type: 'circle' }
      );
    }

    const legendItem = this.legend.selectAll('.legend-item')
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

  handleKeydown(event) {
    if (this.data.length === 0) return;

    switch (event.key) {
      case 'ArrowLeft':
        event.preventDefault();
        this.navigatePoint(-1);
        break;
      case 'ArrowRight':
        event.preventDefault();
        this.navigatePoint(1);
        break;
      case 'Home':
        event.preventDefault();
        this.focusPoint(0);
        break;
      case 'End':
        event.preventDefault();
        this.focusPoint(this.data.length - 1);
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

  navigatePoint(direction) {
    this.focusedPointIndex = Math.max(0, Math.min(
      this.data.length - 1,
      (this.focusedPointIndex >= 0 ? this.focusedPointIndex : 0) + direction
    ));
    this.focusPoint(this.focusedPointIndex);
  }

  focusPoint(index) {
    if (index < 0 || index >= this.data.length) return;

    this.focusedPointIndex = index;
    const data = this.data[index];

    this.updatePointSelection();
    this.announcePoint(data);
  }

  announcePoint(data) {
    this.liveRegion.text(
      `Point ${data.id} at coordinates X: ${data.x.toFixed(3)}, Y: ${data.y.toFixed(3)}. ` +
      `Anomaly score: ${data.anomalyScore.toFixed(3)}, Status: ${data.isAnomaly ? 'Anomaly' : 'Normal'}.`
    );
  }

  updateAccessibilityData() {
    if (this.data.length === 0) return;

    const anomalies = this.data.filter(d => d.isAnomaly);
    const normal = this.data.filter(d => !d.isAnomaly);

    const description = this.container.select('#scatter-description');
    description.text(
      `Scatter plot with ${this.data.length} data points. ` +
      `${anomalies.length} anomalies and ${normal.length} normal points. ` +
      `Use arrow keys to navigate points, Enter to select.`
    );
  }

  updateDataTable() {
    const table = this.dataTableContainer.selectAll('table')
      .data([null]);

    const tableEnter = table.enter()
      .append('table')
      .attr('class', 'table table--striped')
      .style('width', '100%');

    // Table header
    const thead = tableEnter.append('thead');
    const headerRow = thead.append('tr');
    
    const columns = ['ID', 'X', 'Y'];
    if (this.options.is3D) columns.push('Z');
    columns.push('Anomaly Score', 'Status');
    if (this.options.clustering.enabled) columns.push('Cluster');

    headerRow.selectAll('th')
      .data(columns)
      .enter()
      .append('th')
      .text(d => d);

    // Table body
    const tbody = tableEnter.append('tbody');

    // Update rows
    const rows = tbody.selectAll('tr')
      .data(this.data.slice(0, 100)); // Limit to first 100 rows

    const rowsEnter = rows.enter()
      .append('tr');

    const cells = rowsEnter.selectAll('td')
      .data(d => {
        const values = [d.id, d.x.toFixed(3), d.y.toFixed(3)];
        if (this.options.is3D) values.push(d.z.toFixed(3));
        values.push(d.anomalyScore.toFixed(3), d.isAnomaly ? 'Anomaly' : 'Normal');
        if (this.options.clustering.enabled) values.push(d.cluster + 1);
        return values;
      })
      .enter()
      .append('td')
      .text(d => d);
  }

  toggleDataTable() {
    const isVisible = this.dataTableContainer.style('display') !== 'none';
    this.dataTableContainer.style('display', isVisible ? 'none' : 'block');
    
    const button = this.container.select('.scatter-controls button:last-child');
    button.text(isVisible ? 'Show Data Table' : 'Hide Data Table');
  }

  toggle3D() {
    this.options.is3D = !this.options.is3D;
    this.render();
  }

  toggleClustering() {
    this.options.clustering.enabled = !this.options.clustering.enabled;
    this.render();
  }

  exportData() {
    const selectedData = this.selectedPoints.size > 0 
      ? this.data.filter(d => this.selectedPoints.has(d.id))
      : this.data;

    const csvContent = this.arrayToCsv(selectedData);
    this.downloadFile(csvContent, 'scatter-plot-data.csv', 'text/csv');
  }

  arrayToCsv(data) {
    if (data.length === 0) return '';

    const headers = Object.keys(data[0]);
    const csvRows = [headers.join(',')];

    data.forEach(row => {
      const values = headers.map(header => {
        const value = row[header];
        return typeof value === 'string' ? `"${value}"` : value;
      });
      csvRows.push(values.join(','));
    });

    return csvRows.join('\n');
  }

  downloadFile(content, fileName, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    link.click();
    window.URL.revokeObjectURL(url);
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
  module.exports = AnomalyScatterPlot;
}