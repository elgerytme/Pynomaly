/**
 * Advanced Interactive D3.js Visualizations for Pynomaly
 * Provides sophisticated anomaly investigation and drill-down capabilities
 */

class AdvancedInteractiveVisualizations {
  constructor() {
    this.charts = new Map();
    this.crossfilter = null;
    this.dimensions = new Map();
    this.brushSelections = new Map();
    this.annotationLayer = null;
    this.collaborationManager = null;

    this.init();
  }

  init() {
    this.setupCrossfilter();
    this.setupCollaboration();
    this.setupKeyboardShortcuts();
    this.setupContextMenus();
  }

  setupCrossfilter() {
    // Initialize crossfilter for multi-dimensional data filtering
    this.crossfilter = crossfilter();
  }

  setupCollaboration() {
    // Initialize collaboration features
    this.collaborationManager = new CollaborationManager();
  }

  setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'i':
            e.preventDefault();
            this.toggleInvestigationMode();
            break;
          case 'd':
            e.preventDefault();
            this.toggleDrillDownMode();
            break;
          case 'r':
            e.preventDefault();
            this.resetAllFilters();
            break;
          case 'f':
            e.preventDefault();
            this.showFilterPanel();
            break;
        }
      }
    });
  }

  setupContextMenus() {
    document.addEventListener('contextmenu', (e) => {
      if (e.target.closest('.chart-container')) {
        e.preventDefault();
        this.showChartContextMenu(e);
      }
    });
  }

  /**
   * Advanced Anomaly Timeline with Interactive Investigation
   */
  createAnomalyTimelineChart(container, data, options = {}) {
    const chartId = 'timeline-' + Date.now();
    const chart = new AnomalyTimelineChart(container, {
      ...options,
      chartId,
      enableDrillDown: true,
      enableAnnotations: true,
      enableCollaboration: true,
      crossfilter: this.crossfilter
    });

    this.charts.set(chartId, chart);
    chart.setData(data);

    return chart;
  }

  /**
   * Interactive Correlation Matrix with Feature Importance
   */
  createCorrelationMatrix(container, data, options = {}) {
    const chartId = 'correlation-' + Date.now();
    const chart = new InteractiveCorrelationMatrix(container, {
      ...options,
      chartId,
      enableFeatureImportance: true,
      enableClusterAnalysis: true
    });

    this.charts.set(chartId, chart);
    chart.setData(data);

    return chart;
  }

  /**
   * Advanced Anomaly Investigation Dashboard
   */
  createInvestigationDashboard(container, data, options = {}) {
    const dashboard = new AnomalyInvestigationDashboard(container, {
      ...options,
      visualizations: this
    });

    dashboard.setData(data);
    return dashboard;
  }

  toggleInvestigationMode() {
    document.body.classList.toggle('investigation-mode');
    this.charts.forEach(chart => {
      if (chart.toggleInvestigationMode) {
        chart.toggleInvestigationMode();
      }
    });
  }

  toggleDrillDownMode() {
    document.body.classList.toggle('drill-down-mode');
    this.charts.forEach(chart => {
      if (chart.toggleDrillDownMode) {
        chart.toggleDrillDownMode();
      }
    });
  }

  resetAllFilters() {
    this.dimensions.forEach(dim => dim.filterAll());
    this.updateAllCharts();
  }

  updateAllCharts() {
    this.charts.forEach(chart => {
      if (chart.redraw) {
        chart.redraw();
      }
    });
  }
}

/**
 * Advanced Anomaly Timeline Chart with Investigation Features
 */
class AnomalyTimelineChart extends BaseChart {
  constructor(container, options = {}) {
    super(container, {
      enableZoom: true,
      enableBrush: true,
      enableAnnotations: true,
      enableDrillDown: true,
      timeRanges: ['1h', '6h', '1d', '1w', '1m'],
      ...options
    });

    this.zoom = d3.zoom();
    this.brush = d3.brushX();
    this.investigationMode = false;
    this.selectedAnomalies = new Set();
    this.annotations = [];

    this.setupInvestigationFeatures();
  }

  setupInvestigationFeatures() {
    // Add investigation toolbar
    this.toolbar = this.container
      .append('div')
      .attr('class', 'chart-toolbar')
      .html(`
        <div class="toolbar-group">
          <button class="btn-tool" data-action="investigate" title="Investigate Selected">
            🔍 Investigate
          </button>
          <button class="btn-tool" data-action="compare" title="Compare Anomalies">
            📊 Compare
          </button>
          <button class="btn-tool" data-action="annotate" title="Add Annotation">
            📝 Annotate
          </button>
          <button class="btn-tool" data-action="export" title="Export Selection">
            💾 Export
          </button>
        </div>
        <div class="toolbar-group">
          <select class="time-range-selector">
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="1d" selected>Last Day</option>
            <option value="1w">Last Week</option>
            <option value="1m">Last Month</option>
          </select>
        </div>
      `);

    this.toolbar.selectAll('.btn-tool').on('click', (event) => {
      const action = event.target.dataset.action;
      this.handleToolbarAction(action);
    });
  }

  render() {
    if (!this.data || this.data.length === 0) return;

    super.render();

    // Create scales
    this.xScale = d3.scaleTime()
      .domain(d3.extent(this.data, d => d.timestamp))
      .range([0, this.innerWidth]);

    this.yScale = d3.scaleLinear()
      .domain(d3.extent(this.data, d => d.value))
      .nice()
      .range([this.innerHeight, 0]);

    // Add axes
    this.addAxes();

    // Add main timeline
    this.addTimeline();

    // Add anomaly markers with investigation features
    this.addAnomalyMarkers();

    // Add confidence bands
    this.addConfidenceBands();

    // Add zoom and brush
    this.addZoomAndBrush();

    // Add annotation layer
    this.addAnnotationLayer();

    // Add pattern detection overlay
    this.addPatternDetection();
  }

  addTimeline() {
    const line = d3.line()
      .x(d => this.xScale(d.timestamp))
      .y(d => this.yScale(d.value))
      .curve(d3.curveMonotoneX);

    // Main timeline
    this.chartGroup.append('path')
      .datum(this.data)
      .attr('class', 'timeline-main')
      .attr('fill', 'none')
      .attr('stroke', this.theme.primary)
      .attr('stroke-width', 2)
      .attr('d', line);

    // Threshold lines
    if (this.options.thresholds) {
      this.options.thresholds.forEach(threshold => {
        this.chartGroup.append('line')
          .attr('class', `threshold-line threshold-${threshold.type}`)
          .attr('x1', 0)
          .attr('x2', this.innerWidth)
          .attr('y1', this.yScale(threshold.value))
          .attr('y2', this.yScale(threshold.value))
          .attr('stroke', threshold.color || this.theme.warning)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.7);

        // Threshold label
        this.chartGroup.append('text')
          .attr('x', this.innerWidth - 5)
          .attr('y', this.yScale(threshold.value) - 5)
          .attr('text-anchor', 'end')
          .attr('class', 'threshold-label')
          .style('font-size', '12px')
          .style('fill', this.theme.text)
          .text(threshold.label || `Threshold: ${threshold.value}`);
      });
    }
  }

  addAnomalyMarkers() {
    const anomalies = this.data.filter(d => d.isAnomaly);

    const markers = this.chartGroup.selectAll('.anomaly-marker')
      .data(anomalies)
      .enter()
      .append('g')
      .attr('class', 'anomaly-marker')
      .attr('transform', d => `translate(${this.xScale(d.timestamp)}, ${this.yScale(d.value)})`);

    // Main anomaly circle
    markers.append('circle')
      .attr('r', d => this.getAnomalyRadius(d.severity))
      .attr('fill', d => this.getAnomalyColor(d.severity))
      .attr('stroke', this.theme.background)
      .attr('stroke-width', 2)
      .attr('opacity', 0.8)
      .style('cursor', 'pointer');

    // Severity indicator ring
    markers.append('circle')
      .attr('r', d => this.getAnomalyRadius(d.severity) + 3)
      .attr('fill', 'none')
      .attr('stroke', d => this.getAnomalyColor(d.severity))
      .attr('stroke-width', 1)
      .attr('opacity', d => d.severity === 'high' ? 1 : 0);

    // Investigation overlay
    const investigationOverlay = markers.append('g')
      .attr('class', 'investigation-overlay')
      .style('opacity', 0);

    investigationOverlay.append('circle')
      .attr('r', 15)
      .attr('fill', 'rgba(255, 255, 255, 0.9)')
      .attr('stroke', this.theme.primary)
      .attr('stroke-width', 2);

    investigationOverlay.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .style('font-size', '10px')
      .style('font-weight', 'bold')
      .text('🔍');

    // Event handlers
    markers
      .on('mouseover', (event, d) => this.handleAnomalyHover(event, d))
      .on('mouseout', (event, d) => this.handleAnomalyMouseOut(event, d))
      .on('click', (event, d) => this.handleAnomalyClick(event, d))
      .on('contextmenu', (event, d) => this.handleAnomalyContextMenu(event, d));

    // Selection highlighting
    markers.classed('selected', d => this.selectedAnomalies.has(d.id));
  }

  addConfidenceBands() {
    if (!this.options.showConfidence) return;

    const area = d3.area()
      .x(d => this.xScale(d.timestamp))
      .y0(d => this.yScale(d.value - d.confidence))
      .y1(d => this.yScale(d.value + d.confidence))
      .curve(d3.curveMonotoneX);

    this.chartGroup.append('path')
      .datum(this.data.filter(d => d.confidence))
      .attr('class', 'confidence-band')
      .attr('fill', this.theme.primary)
      .attr('fill-opacity', 0.1)
      .attr('d', area);
  }

  addZoomAndBrush() {
    // Zoom behavior
    this.zoom
      .scaleExtent([1, 50])
      .on('zoom', (event) => this.handleZoom(event));

    this.svg.call(this.zoom);

    // Brush for selection
    this.brush
      .extent([[0, 0], [this.innerWidth, this.innerHeight]])
      .on('brush end', (event) => this.handleBrush(event));

    this.brushGroup = this.chartGroup.append('g')
      .attr('class', 'brush')
      .call(this.brush);
  }

  addAnnotationLayer() {
    this.annotationGroup = this.chartGroup.append('g')
      .attr('class', 'annotations');

    // Render existing annotations
    this.renderAnnotations();
  }

  addPatternDetection() {
    // Add pattern recognition overlay
    this.patternGroup = this.chartGroup.append('g')
      .attr('class', 'patterns');

    // Detect and highlight patterns
    this.detectPatterns();
  }

  detectPatterns() {
    // Simple pattern detection - can be enhanced with ML
    const patterns = this.findTimeBasedPatterns();
    const anomalyPatterns = this.findAnomalyPatterns();

    this.renderPatterns([...patterns, ...anomalyPatterns]);
  }

  findTimeBasedPatterns() {
    // Detect daily, weekly patterns
    const patterns = [];
    const hourlyGroups = d3.group(this.data, d => d.timestamp.getHours());

    // Find peak hours
    const peakHours = Array.from(hourlyGroups.entries())
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, 3);

    peakHours.forEach(([hour, data]) => {
      patterns.push({
        type: 'temporal',
        subtype: 'peak_hour',
        hour: hour,
        count: data.length,
        description: `Peak activity at ${hour}:00`
      });
    });

    return patterns;
  }

  findAnomalyPatterns() {
    const patterns = [];
    const anomalies = this.data.filter(d => d.isAnomaly);

    // Cluster anomalies by proximity
    const clusters = this.clusterAnomaliesByTime(anomalies);

    clusters.forEach(cluster => {
      if (cluster.length >= 3) {
        patterns.push({
          type: 'anomaly_cluster',
          data: cluster,
          timespan: d3.extent(cluster, d => d.timestamp),
          severity: d3.max(cluster, d => this.getSeverityScore(d.severity)),
          description: `Anomaly cluster: ${cluster.length} anomalies`
        });
      }
    });

    return patterns;
  }

  clusterAnomaliesByTime(anomalies, threshold = 3600000) { // 1 hour
    const clusters = [];
    const sorted = anomalies.sort((a, b) => a.timestamp - b.timestamp);

    let currentCluster = [sorted[0]];

    for (let i = 1; i < sorted.length; i++) {
      const timeDiff = sorted[i].timestamp - sorted[i-1].timestamp;

      if (timeDiff <= threshold) {
        currentCluster.push(sorted[i]);
      } else {
        if (currentCluster.length > 1) {
          clusters.push(currentCluster);
        }
        currentCluster = [sorted[i]];
      }
    }

    if (currentCluster.length > 1) {
      clusters.push(currentCluster);
    }

    return clusters;
  }

  renderPatterns(patterns) {
    const patternElements = this.patternGroup.selectAll('.pattern')
      .data(patterns)
      .enter()
      .append('g')
      .attr('class', d => `pattern pattern-${d.type}`);

    // Render anomaly clusters
    patternElements.filter(d => d.type === 'anomaly_cluster')
      .each(function(d) {
        const g = d3.select(this);
        const xStart = this.xScale(d.timespan[0]);
        const xEnd = this.xScale(d.timespan[1]);

        g.append('rect')
          .attr('x', xStart)
          .attr('y', 0)
          .attr('width', xEnd - xStart)
          .attr('height', this.innerHeight)
          .attr('fill', this.theme.warning)
          .attr('fill-opacity', 0.1)
          .attr('stroke', this.theme.warning)
          .attr('stroke-dasharray', '3,3');
      }.bind(this));
  }

  handleAnomalyHover(event, d) {
    // Show detailed tooltip
    const content = `
      <div class="anomaly-tooltip">
        <div class="tooltip-header">
          <strong>Anomaly Detected</strong>
          <span class="severity-badge severity-${d.severity}">${d.severity}</span>
        </div>
        <div class="tooltip-body">
          <div class="metric-row">
            <span class="metric-label">Time:</span>
            <span class="metric-value">${d3.timeFormat('%Y-%m-%d %H:%M:%S')(d.timestamp)}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Value:</span>
            <span class="metric-value">${d.value.toFixed(3)}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Score:</span>
            <span class="metric-value">${d.anomalyScore.toFixed(3)}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Algorithm:</span>
            <span class="metric-value">${d.detectionMethod || 'Unknown'}</span>
          </div>
        </div>
        <div class="tooltip-actions">
          <button class="btn-sm" onclick="investigateAnomaly('${d.id}')">Investigate</button>
          <button class="btn-sm" onclick="compareAnomaly('${d.id}')">Compare</button>
        </div>
      </div>
    `;

    this.showTooltip(content, event);

    // Highlight in investigation mode
    if (this.investigationMode) {
      d3.select(event.target.parentNode)
        .select('.investigation-overlay')
        .transition()
        .duration(200)
        .style('opacity', 1);
    }
  }

  handleAnomalyClick(event, d) {
    event.stopPropagation();

    if (event.ctrlKey || event.metaKey) {
      // Multi-select
      if (this.selectedAnomalies.has(d.id)) {
        this.selectedAnomalies.delete(d.id);
      } else {
        this.selectedAnomalies.add(d.id);
      }
    } else {
      // Single select
      this.selectedAnomalies.clear();
      this.selectedAnomalies.add(d.id);
    }

    this.updateSelectionHighlights();
    this.emitSelectionChange();
  }

  handleAnomalyContextMenu(event, d) {
    event.preventDefault();

    const contextMenu = [
      { label: 'Investigate Anomaly', action: () => this.investigateAnomaly(d) },
      { label: 'Compare with Similar', action: () => this.compareAnomaly(d) },
      { label: 'Add Annotation', action: () => this.addAnnotation(d) },
      { label: 'Export Details', action: () => this.exportAnomalyDetails(d) },
      { separator: true },
      { label: 'Mark as False Positive', action: () => this.markFalsePositive(d) }
    ];

    this.showContextMenu(event, contextMenu);
  }

  investigateAnomaly(anomaly) {
    // Open investigation panel
    const investigationPanel = new AnomalyInvestigationPanel(anomaly, {
      timeline: this,
      data: this.data
    });

    investigationPanel.show();
  }

  compareAnomaly(anomaly) {
    // Find similar anomalies
    const similar = this.findSimilarAnomalies(anomaly);

    const comparisonView = new AnomalyComparisonView({
      primary: anomaly,
      similar: similar,
      timeline: this
    });

    comparisonView.show();
  }

  addAnnotation(anomaly) {
    const annotation = {
      id: 'annotation-' + Date.now(),
      x: this.xScale(anomaly.timestamp),
      y: this.yScale(anomaly.value),
      timestamp: anomaly.timestamp,
      text: '',
      author: window.currentUser?.name || 'Anonymous',
      created: new Date(),
      anomalyId: anomaly.id
    };

    this.showAnnotationEditor(annotation);
  }

  // Helper methods
  getAnomalyRadius(severity) {
    const sizes = { low: 4, medium: 6, high: 8 };
    return sizes[severity] || 5;
  }

  getAnomalyColor(severity) {
    const colors = {
      low: this.theme.warning,
      medium: this.theme.accent,
      high: this.theme.danger
    };
    return colors[severity] || this.theme.primary;
  }

  getSeverityScore(severity) {
    const scores = { low: 1, medium: 2, high: 3 };
    return scores[severity] || 1;
  }

  findSimilarAnomalies(target, maxResults = 5) {
    const anomalies = this.data.filter(d => d.isAnomaly && d.id !== target.id);

    return anomalies
      .map(anomaly => ({
        ...anomaly,
        similarity: this.calculateSimilarity(target, anomaly)
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, maxResults);
  }

  calculateSimilarity(a, b) {
    // Simple similarity calculation - can be enhanced
    const valueSim = 1 - Math.abs(a.value - b.value) / Math.max(a.value, b.value);
    const scoreSim = 1 - Math.abs(a.anomalyScore - b.anomalyScore);
    const timeSim = 1 - Math.abs(a.timestamp - b.timestamp) / (24 * 60 * 60 * 1000); // Day basis

    return (valueSim + scoreSim + timeSim) / 3;
  }

  updateSelectionHighlights() {
    this.chartGroup.selectAll('.anomaly-marker')
      .classed('selected', d => this.selectedAnomalies.has(d.id));
  }

  emitSelectionChange() {
    const selectedData = this.data.filter(d => this.selectedAnomalies.has(d.id));

    this.containerElement.dispatchEvent(new CustomEvent('anomaly-selection-changed', {
      detail: {
        selected: selectedData,
        count: selectedData.length
      }
    }));
  }

  handleToolbarAction(action) {
    switch (action) {
      case 'investigate':
        this.investigateSelectedAnomalies();
        break;
      case 'compare':
        this.compareSelectedAnomalies();
        break;
      case 'annotate':
        this.annotateSelection();
        break;
      case 'export':
        this.exportSelection();
        break;
    }
  }

  investigateSelectedAnomalies() {
    const selected = this.data.filter(d => this.selectedAnomalies.has(d.id));
    if (selected.length === 0) {
      this.showMessage('Please select anomalies to investigate');
      return;
    }

    const investigationDashboard = new MultiAnomalyInvestigationDashboard(selected, {
      timeline: this,
      data: this.data
    });

    investigationDashboard.show();
  }
}

/**
 * Interactive Correlation Matrix
 */
class InteractiveCorrelationMatrix extends BaseChart {
  constructor(container, options = {}) {
    super(container, {
      enableFeatureImportance: true,
      enableClusterAnalysis: true,
      colorScheme: d3.interpolateRdBu,
      ...options
    });

    this.correlationData = [];
    this.featureImportance = [];
    this.clusters = [];
    this.selectedFeatures = new Set();

    this.setupInteractions();
  }

  setupInteractions() {
    // Add feature selection controls
    this.controlPanel = this.container
      .append('div')
      .attr('class', 'correlation-controls')
      .html(`
        <div class="control-group">
          <label>Feature Selection:</label>
          <button class="btn-sm" id="select-all-features">Select All</button>
          <button class="btn-sm" id="clear-selection">Clear</button>
          <button class="btn-sm" id="select-important">Top Important</button>
        </div>
        <div class="control-group">
          <label>Analysis:</label>
          <button class="btn-sm" id="find-clusters">Find Clusters</button>
          <button class="btn-sm" id="show-importance">Feature Importance</button>
          <button class="btn-sm" id="export-correlations">Export</button>
        </div>
      `);

    this.setupControlHandlers();
  }

  render() {
    if (!this.data || this.data.length === 0) return;

    this.processCorrelationData();
    super.render();

    this.createCorrelationMatrix();
    this.addFeatureImportancePanel();
    this.addClusterOverlay();
  }

  processCorrelationData() {
    // Calculate correlation matrix from data
    const features = Object.keys(this.data[0]).filter(key =>
      typeof this.data[0][key] === 'number' && key !== 'timestamp'
    );

    this.correlationData = this.calculateCorrelationMatrix(features);
    this.featureImportance = this.calculateFeatureImportance(features);
  }

  calculateCorrelationMatrix(features) {
    const matrix = [];

    for (let i = 0; i < features.length; i++) {
      const row = [];
      for (let j = 0; j < features.length; j++) {
        const correlation = this.calculateCorrelation(
          this.data.map(d => d[features[i]]),
          this.data.map(d => d[features[j]])
        );

        row.push({
          x: features[j],
          y: features[i],
          value: correlation,
          xIndex: j,
          yIndex: i
        });
      }
      matrix.push(row);
    }

    return matrix.flat();
  }

  calculateCorrelation(x, y) {
    const n = x.length;
    const sumX = d3.sum(x);
    const sumY = d3.sum(y);
    const sumXY = d3.sum(x.map((xi, i) => xi * y[i]));
    const sumX2 = d3.sum(x.map(xi => xi * xi));
    const sumY2 = d3.sum(y.map(yi => yi * yi));

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  calculateFeatureImportance(features) {
    // Simple feature importance based on variance and correlation with target
    return features.map(feature => {
      const values = this.data.map(d => d[feature]);
      const variance = d3.variance(values);
      const importance = variance / d3.max(features.map(f =>
        d3.variance(this.data.map(d => d[f]))
      ));

      return {
        feature,
        importance,
        variance
      };
    }).sort((a, b) => b.importance - a.importance);
  }

  createCorrelationMatrix() {
    const features = [...new Set(this.correlationData.map(d => d.x))];
    const cellSize = Math.min(this.innerWidth, this.innerHeight) / features.length;

    this.xScale = d3.scaleBand()
      .domain(features)
      .range([0, cellSize * features.length])
      .padding(0.05);

    this.yScale = d3.scaleBand()
      .domain(features)
      .range([0, cellSize * features.length])
      .padding(0.05);

    this.colorScale = d3.scaleSequential(this.options.colorScheme)
      .domain([-1, 1]);

    // Add axes
    this.addAxes();

    // Add correlation cells
    this.addCorrelationCells();

    // Add color legend
    this.addColorLegend();
  }

  addCorrelationCells() {
    const cells = this.chartGroup.selectAll('.correlation-cell')
      .data(this.correlationData)
      .enter()
      .append('rect')
      .attr('class', 'correlation-cell')
      .attr('x', d => this.xScale(d.x))
      .attr('y', d => this.yScale(d.y))
      .attr('width', this.xScale.bandwidth())
      .attr('height', this.yScale.bandwidth())
      .attr('fill', d => this.colorScale(d.value))
      .attr('stroke', this.theme.background)
      .attr('stroke-width', 1)
      .style('cursor', 'pointer');

    // Add correlation values as text
    this.chartGroup.selectAll('.correlation-text')
      .data(this.correlationData)
      .enter()
      .append('text')
      .attr('class', 'correlation-text')
      .attr('x', d => this.xScale(d.x) + this.xScale.bandwidth() / 2)
      .attr('y', d => this.yScale(d.y) + this.yScale.bandwidth() / 2)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .style('fill', d => Math.abs(d.value) > 0.5 ? 'white' : 'black')
      .style('font-size', '10px')
      .style('font-weight', 'bold')
      .style('pointer-events', 'none')
      .text(d => d.value.toFixed(2));

    // Add interactions
    cells
      .on('mouseover', (event, d) => this.handleCellHover(event, d))
      .on('mouseout', (event, d) => this.handleCellMouseOut(event, d))
      .on('click', (event, d) => this.handleCellClick(event, d));
  }

  handleCellHover(event, d) {
    // Highlight row and column
    this.chartGroup.selectAll('.correlation-cell')
      .style('opacity', cell =>
        cell.x === d.x || cell.y === d.y ? 1 : 0.3
      );

    // Show detailed tooltip
    const content = `
      <div class="correlation-tooltip">
        <strong>Correlation Analysis</strong><br/>
        Features: ${d.x} ↔ ${d.y}<br/>
        Correlation: ${d.value.toFixed(3)}<br/>
        Strength: ${this.getCorrelationStrength(d.value)}<br/>
        Direction: ${d.value > 0 ? 'Positive' : 'Negative'}
      </div>
    `;

    this.showTooltip(content, event);
  }

  getCorrelationStrength(value) {
    const abs = Math.abs(value);
    if (abs >= 0.8) return 'Very Strong';
    if (abs >= 0.6) return 'Strong';
    if (abs >= 0.4) return 'Moderate';
    if (abs >= 0.2) return 'Weak';
    return 'Very Weak';
  }
}

/**
 * Anomaly Investigation Dashboard
 */
class AnomalyInvestigationDashboard {
  constructor(container, options = {}) {
    this.container = d3.select(container);
    this.options = options;
    this.data = null;
    this.selectedAnomaly = null;

    this.init();
  }

  init() {
    this.createLayout();
    this.setupEventHandlers();
  }

  createLayout() {
    this.container.html(`
      <div class="investigation-dashboard">
        <div class="dashboard-header">
          <h2>Anomaly Investigation Dashboard</h2>
          <div class="dashboard-controls">
            <button class="btn-primary" id="start-investigation">Start Investigation</button>
            <button class="btn-secondary" id="save-investigation">Save Report</button>
            <button class="btn-secondary" id="export-investigation">Export</button>
          </div>
        </div>

        <div class="dashboard-content">
          <div class="investigation-sidebar">
            <div class="anomaly-list">
              <h3>Detected Anomalies</h3>
              <div class="anomaly-filters">
                <select id="severity-filter">
                  <option value="">All Severities</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                <select id="time-filter">
                  <option value="">All Time</option>
                  <option value="1h">Last Hour</option>
                  <option value="1d">Last Day</option>
                  <option value="1w">Last Week</option>
                </select>
              </div>
              <div class="anomaly-items"></div>
            </div>

            <div class="investigation-tools">
              <h3>Investigation Tools</h3>
              <div class="tool-buttons">
                <button class="tool-btn" data-tool="timeline">Timeline Analysis</button>
                <button class="tool-btn" data-tool="correlation">Correlation Matrix</button>
                <button class="tool-btn" data-tool="features">Feature Analysis</button>
                <button class="tool-btn" data-tool="comparison">Anomaly Comparison</button>
                <button class="tool-btn" data-tool="patterns">Pattern Detection</button>
              </div>
            </div>
          </div>

          <div class="investigation-main">
            <div class="investigation-tabs">
              <button class="tab-btn active" data-tab="overview">Overview</button>
              <button class="tab-btn" data-tab="details">Details</button>
              <button class="tab-btn" data-tab="context">Context</button>
              <button class="tab-btn" data-tab="analysis">Analysis</button>
            </div>

            <div class="investigation-panels">
              <div class="investigation-panel active" id="overview-panel">
                <div class="overview-metrics"></div>
                <div class="overview-charts"></div>
              </div>

              <div class="investigation-panel" id="details-panel">
                <div class="anomaly-details"></div>
                <div class="feature-breakdown"></div>
              </div>

              <div class="investigation-panel" id="context-panel">
                <div class="temporal-context"></div>
                <div class="related-events"></div>
              </div>

              <div class="investigation-panel" id="analysis-panel">
                <div class="root-cause-analysis"></div>
                <div class="impact-assessment"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `);
  }

  setData(data) {
    this.data = data;
    this.renderAnomalyList();
    this.renderOverview();
  }

  renderAnomalyList() {
    const anomalies = this.data.filter(d => d.isAnomaly);
    const anomalyItems = this.container.select('.anomaly-items');

    const items = anomalyItems.selectAll('.anomaly-item')
      .data(anomalies)
      .enter()
      .append('div')
      .attr('class', 'anomaly-item')
      .on('click', (event, d) => this.selectAnomaly(d));

    items.html(d => `
      <div class="anomaly-summary">
        <div class="anomaly-header">
          <span class="anomaly-time">${d3.timeFormat('%H:%M:%S')(d.timestamp)}</span>
          <span class="severity-badge severity-${d.severity}">${d.severity}</span>
        </div>
        <div class="anomaly-value">Value: ${d.value.toFixed(3)}</div>
        <div class="anomaly-score">Score: ${d.anomalyScore.toFixed(3)}</div>
      </div>
    `);
  }

  selectAnomaly(anomaly) {
    this.selectedAnomaly = anomaly;
    this.container.selectAll('.anomaly-item').classed('selected', false);
    this.container.selectAll('.anomaly-item')
      .filter(d => d.id === anomaly.id)
      .classed('selected', true);

    this.renderAnomalyDetails();
  }

  renderAnomalyDetails() {
    if (!this.selectedAnomaly) return;

    const detailsPanel = this.container.select('#details-panel .anomaly-details');

    detailsPanel.html(`
      <div class="detail-section">
        <h4>Anomaly Information</h4>
        <div class="detail-grid">
          <div class="detail-item">
            <label>Timestamp:</label>
            <span>${d3.timeFormat('%Y-%m-%d %H:%M:%S')(this.selectedAnomaly.timestamp)}</span>
          </div>
          <div class="detail-item">
            <label>Value:</label>
            <span>${this.selectedAnomaly.value.toFixed(6)}</span>
          </div>
          <div class="detail-item">
            <label>Anomaly Score:</label>
            <span>${this.selectedAnomaly.anomalyScore.toFixed(6)}</span>
          </div>
          <div class="detail-item">
            <label>Severity:</label>
            <span class="severity-badge severity-${this.selectedAnomaly.severity}">${this.selectedAnomaly.severity}</span>
          </div>
          <div class="detail-item">
            <label>Detection Method:</label>
            <span>${this.selectedAnomaly.detectionMethod || 'Unknown'}</span>
          </div>
          <div class="detail-item">
            <label>Confidence:</label>
            <span>${((this.selectedAnomaly.confidence || 0) * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    `);

    this.renderFeatureBreakdown();
    this.renderContextualAnalysis();
  }

  renderFeatureBreakdown() {
    // Implementation for feature contribution analysis
    const features = Object.keys(this.selectedAnomaly)
      .filter(key => typeof this.selectedAnomaly[key] === 'number' &&
                     !['timestamp', 'anomalyScore'].includes(key));

    const breakdown = this.container.select('#details-panel .feature-breakdown');

    breakdown.html(`
      <div class="detail-section">
        <h4>Feature Contributions</h4>
        <div class="feature-chart-container"></div>
      </div>
    `);

    // Create feature importance chart
    this.createFeatureChart(breakdown.select('.feature-chart-container').node(), features);
  }

  createFeatureChart(container, features) {
    const margin = { top: 20, right: 30, bottom: 40, left: 100 };
    const width = 400 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Calculate feature deviations
    const featureData = features.map(feature => {
      const value = this.selectedAnomaly[feature];
      const values = this.data.map(d => d[feature]).filter(v => v != null);
      const mean = d3.mean(values);
      const stdDev = d3.deviation(values);
      const deviation = Math.abs(value - mean) / stdDev;

      return {
        feature,
        value,
        mean,
        deviation: isNaN(deviation) ? 0 : deviation
      };
    }).sort((a, b) => b.deviation - a.deviation);

    const xScale = d3.scaleLinear()
      .domain([0, d3.max(featureData, d => d.deviation)])
      .range([0, width]);

    const yScale = d3.scaleBand()
      .domain(featureData.map(d => d.feature))
      .range([0, height])
      .padding(0.1);

    // Add bars
    g.selectAll('.feature-bar')
      .data(featureData)
      .enter()
      .append('rect')
      .attr('class', 'feature-bar')
      .attr('x', 0)
      .attr('y', d => yScale(d.feature))
      .attr('width', d => xScale(d.deviation))
      .attr('height', yScale.bandwidth())
      .attr('fill', (d, i) => d3.schemeCategory10[i % 10]);

    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append('g')
      .call(d3.axisLeft(yScale));
  }
}

// Export classes
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    AdvancedInteractiveVisualizations,
    AnomalyTimelineChart,
    InteractiveCorrelationMatrix,
    AnomalyInvestigationDashboard
  };
} else {
  window.AdvancedInteractiveVisualizations = AdvancedInteractiveVisualizations;
  window.AnomalyTimelineChart = AnomalyTimelineChart;
  window.InteractiveCorrelationMatrix = InteractiveCorrelationMatrix;
  window.AnomalyInvestigationDashboard = AnomalyInvestigationDashboard;
}
