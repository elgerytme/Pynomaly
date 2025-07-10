/**
 * Interactive Data Visualization Components
 * Advanced visualizations for anomaly detection data analysis
 */

export class InteractiveDataVisualization {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      width: 800,
      height: 600,
      margin: { top: 20, right: 30, bottom: 40, left: 50 },
      enableZoom: true,
      enableBrush: true,
      enableTooltips: true,
      enableLegend: true,
      theme: 'light',
      ...options
    };

    this.svg = null;
    this.data = [];
    this.filteredData = [];
    this.currentView = 'scatter';
    this.scales = {};
    this.brushSelection = null;
    this.zoomTransform = null;

    this.init();
  }

  init() {
    this.createSVG();
    this.setupInteractionHandlers();
    this.createControls();
  }

  createSVG() {
    // Clear existing content
    this.container.innerHTML = '';

    // Create main SVG
    this.svg = d3.select(this.container)
      .append('svg')
      .attr('width', this.options.width)
      .attr('height', this.options.height)
      .attr('class', `visualization-svg ${this.options.theme}`);

    // Create main group for chart content
    this.chartGroup = this.svg.append('g')
      .attr('transform', `translate(${this.options.margin.left},${this.options.margin.top})`);

    // Create clip path for zooming
    this.svg.append('defs')
      .append('clipPath')
      .attr('id', 'clip')
      .append('rect')
      .attr('width', this.options.width - this.options.margin.left - this.options.margin.right)
      .attr('height', this.options.height - this.options.margin.top - this.options.margin.bottom);

    // Create groups for different chart elements
    this.dataGroup = this.chartGroup.append('g').attr('class', 'data-group');
    this.axesGroup = this.chartGroup.append('g').attr('class', 'axes-group');
    this.brushGroup = this.chartGroup.append('g').attr('class', 'brush-group');
    this.tooltipGroup = this.chartGroup.append('g').attr('class', 'tooltip-group');
  }

  createControls() {
    const controlsContainer = document.createElement('div');
    controlsContainer.className = 'visualization-controls';
    controlsContainer.innerHTML = `
      <div class="control-group">
        <label for="view-selector">View:</label>
        <select id="view-selector">
          <option value="scatter">Scatter Plot</option>
          <option value="timeline">Timeline</option>
          <option value="heatmap">Heatmap</option>
          <option value="parallel">Parallel Coordinates</option>
          <option value="box">Box Plot</option>
          <option value="violin">Violin Plot</option>
          <option value="network">Network Graph</option>
        </select>
      </div>

      <div class="control-group">
        <label for="x-axis-selector">X-Axis:</label>
        <select id="x-axis-selector">
          <option value="timestamp">Timestamp</option>
          <option value="value">Value</option>
          <option value="anomaly_score">Anomaly Score</option>
          <option value="feature_1">Feature 1</option>
          <option value="feature_2">Feature 2</option>
        </select>
      </div>

      <div class="control-group">
        <label for="y-axis-selector">Y-Axis:</label>
        <select id="y-axis-selector">
          <option value="anomaly_score">Anomaly Score</option>
          <option value="value">Value</option>
          <option value="feature_1">Feature 1</option>
          <option value="feature_2">Feature 2</option>
        </select>
      </div>

      <div class="control-group">
        <label for="color-selector">Color By:</label>
        <select id="color-selector">
          <option value="is_anomaly">Anomaly Status</option>
          <option value="algorithm">Algorithm</option>
          <option value="confidence">Confidence</option>
          <option value="cluster">Cluster</option>
        </select>
      </div>

      <div class="control-group">
        <label for="size-selector">Size By:</label>
        <select id="size-selector">
          <option value="none">None</option>
          <option value="anomaly_score">Anomaly Score</option>
          <option value="confidence">Confidence</option>
          <option value="importance">Importance</option>
        </select>
      </div>

      <div class="control-group">
        <label>
          <input type="checkbox" id="show-anomalies-only"> Show Anomalies Only
        </label>
      </div>

      <div class="control-group">
        <label>
          <input type="checkbox" id="enable-clustering"> Enable Clustering
        </label>
      </div>

      <div class="control-group">
        <button id="reset-view">Reset View</button>
        <button id="export-view">Export View</button>
        <button id="animate-view">Animate</button>
      </div>
    `;

    this.container.insertBefore(controlsContainer, this.container.firstChild);
    this.setupControlEvents();
  }

  setupControlEvents() {
    document.getElementById('view-selector').addEventListener('change', (e) => {
      this.currentView = e.target.value;
      this.updateVisualization();
    });

    document.getElementById('x-axis-selector').addEventListener('change', () => {
      this.updateVisualization();
    });

    document.getElementById('y-axis-selector').addEventListener('change', () => {
      this.updateVisualization();
    });

    document.getElementById('color-selector').addEventListener('change', () => {
      this.updateVisualization();
    });

    document.getElementById('size-selector').addEventListener('change', () => {
      this.updateVisualization();
    });

    document.getElementById('show-anomalies-only').addEventListener('change', () => {
      this.filterData();
      this.updateVisualization();
    });

    document.getElementById('enable-clustering').addEventListener('change', (e) => {
      if (e.target.checked) {
        this.performClustering();
      }
      this.updateVisualization();
    });

    document.getElementById('reset-view').addEventListener('click', () => {
      this.resetView();
    });

    document.getElementById('export-view').addEventListener('click', () => {
      this.exportView();
    });

    document.getElementById('animate-view').addEventListener('click', () => {
      this.animateView();
    });
  }

  setData(data) {
    this.data = data;
    this.filteredData = [...data];
    this.updateVisualization();
  }

  filterData() {
    const showAnomaliesOnly = document.getElementById('show-anomalies-only').checked;

    if (showAnomaliesOnly) {
      this.filteredData = this.data.filter(d => d.is_anomaly);
    } else {
      this.filteredData = [...this.data];
    }
  }

  updateVisualization() {
    this.filterData();

    switch (this.currentView) {
      case 'scatter':
        this.renderScatterPlot();
        break;
      case 'timeline':
        this.renderTimeline();
        break;
      case 'heatmap':
        this.renderHeatmap();
        break;
      case 'parallel':
        this.renderParallelCoordinates();
        break;
      case 'box':
        this.renderBoxPlot();
        break;
      case 'violin':
        this.renderViolinPlot();
        break;
      case 'network':
        this.renderNetworkGraph();
        break;
      default:
        this.renderScatterPlot();
    }
  }

  renderScatterPlot() {
    const xField = document.getElementById('x-axis-selector').value;
    const yField = document.getElementById('y-axis-selector').value;
    const colorField = document.getElementById('color-selector').value;
    const sizeField = document.getElementById('size-selector').value;

    // Clear previous content
    this.dataGroup.selectAll('*').remove();
    this.axesGroup.selectAll('*').remove();

    // Set up scales
    const width = this.options.width - this.options.margin.left - this.options.margin.right;
    const height = this.options.height - this.options.margin.top - this.options.margin.bottom;

    const xScale = d3.scaleLinear()
      .domain(d3.extent(this.filteredData, d => d[xField]))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(this.filteredData, d => d[yField]))
      .range([height, 0]);

    const colorScale = this.getColorScale(colorField);
    const sizeScale = this.getSizeScale(sizeField);

    this.scales = { x: xScale, y: yScale, color: colorScale, size: sizeScale };

    // Create axes
    this.axesGroup.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    this.axesGroup.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale));

    // Add axis labels
    this.axesGroup.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', width / 2)
      .attr('y', height + 35)
      .text(xField);

    this.axesGroup.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -35)
      .text(yField);

    // Create scatter points
    const points = this.dataGroup.selectAll('.data-point')
      .data(this.filteredData)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', d => xScale(d[xField]))
      .attr('cy', d => yScale(d[yField]))
      .attr('r', d => sizeScale(d[sizeField]))
      .attr('fill', d => colorScale(d[colorField]))
      .attr('opacity', 0.7)
      .attr('stroke', '#333')
      .attr('stroke-width', 0.5);

    // Add tooltips
    if (this.options.enableTooltips) {
      points.on('mouseover', (event, d) => {
        this.showTooltip(event, d);
      })
      .on('mouseout', () => {
        this.hideTooltip();
      });
    }

    // Add zoom and brush
    this.setupZoomAndBrush();
  }

  renderTimeline() {
    const yField = document.getElementById('y-axis-selector').value;
    const colorField = document.getElementById('color-selector').value;

    // Clear previous content
    this.dataGroup.selectAll('*').remove();
    this.axesGroup.selectAll('*').remove();

    const width = this.options.width - this.options.margin.left - this.options.margin.right;
    const height = this.options.height - this.options.margin.top - this.options.margin.bottom;

    // Sort data by timestamp
    const sortedData = this.filteredData.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    const xScale = d3.scaleTime()
      .domain(d3.extent(sortedData, d => new Date(d.timestamp)))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(sortedData, d => d[yField]))
      .range([height, 0]);

    const colorScale = this.getColorScale(colorField);

    this.scales = { x: xScale, y: yScale, color: colorScale };

    // Create axes
    this.axesGroup.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    this.axesGroup.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale));

    // Create line
    const line = d3.line()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d[yField]))
      .curve(d3.curveMonotoneX);

    this.dataGroup.append('path')
      .datum(sortedData)
      .attr('class', 'timeline-line')
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Add points for anomalies
    const anomalies = sortedData.filter(d => d.is_anomaly);
    this.dataGroup.selectAll('.anomaly-point')
      .data(anomalies)
      .enter()
      .append('circle')
      .attr('class', 'anomaly-point')
      .attr('cx', d => xScale(new Date(d.timestamp)))
      .attr('cy', d => yScale(d[yField]))
      .attr('r', 4)
      .attr('fill', '#ef4444')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    // Add tooltips
    if (this.options.enableTooltips) {
      this.dataGroup.selectAll('.anomaly-point')
        .on('mouseover', (event, d) => {
          this.showTooltip(event, d);
        })
        .on('mouseout', () => {
          this.hideTooltip();
        });
    }

    this.setupZoomAndBrush();
  }

  renderHeatmap() {
    const width = this.options.width - this.options.margin.left - this.options.margin.right;
    const height = this.options.height - this.options.margin.top - this.options.margin.bottom;

    // Clear previous content
    this.dataGroup.selectAll('*').remove();
    this.axesGroup.selectAll('*').remove();

    // Aggregate data for heatmap
    const aggregatedData = this.aggregateDataForHeatmap();

    const xScale = d3.scaleBand()
      .domain(aggregatedData.map(d => d.x))
      .range([0, width])
      .padding(0.1);

    const yScale = d3.scaleBand()
      .domain(aggregatedData.map(d => d.y))
      .range([height, 0])
      .padding(0.1);

    const colorScale = d3.scaleSequential(d3.interpolateYlOrRd)
      .domain([0, d3.max(aggregatedData, d => d.value)]);

    this.scales = { x: xScale, y: yScale, color: colorScale };

    // Create heatmap cells
    this.dataGroup.selectAll('.heatmap-cell')
      .data(aggregatedData)
      .enter()
      .append('rect')
      .attr('class', 'heatmap-cell')
      .attr('x', d => xScale(d.x))
      .attr('y', d => yScale(d.y))
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1);

    // Create axes
    this.axesGroup.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    this.axesGroup.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale));

    // Add color legend
    this.addColorLegend(colorScale);
  }

  renderParallelCoordinates() {
    const width = this.options.width - this.options.margin.left - this.options.margin.right;
    const height = this.options.height - this.options.margin.top - this.options.margin.bottom;

    // Clear previous content
    this.dataGroup.selectAll('*').remove();
    this.axesGroup.selectAll('*').remove();

    // Select numeric dimensions
    const dimensions = ['value', 'anomaly_score', 'feature_1', 'feature_2', 'feature_3'];
    const validDimensions = dimensions.filter(dim =>
      this.filteredData.some(d => d[dim] !== undefined && d[dim] !== null)
    );

    // Create scales for each dimension
    const xScale = d3.scalePoint()
      .domain(validDimensions)
      .range([0, width]);

    const yScales = {};
    validDimensions.forEach(dim => {
      yScales[dim] = d3.scaleLinear()
        .domain(d3.extent(this.filteredData, d => d[dim]))
        .range([height, 0]);
    });

    // Create line generator
    const line = d3.line()
      .x(d => xScale(d.dimension))
      .y(d => yScales[d.dimension](d.value));

    // Create parallel coordinate lines
    this.dataGroup.selectAll('.parallel-line')
      .data(this.filteredData)
      .enter()
      .append('path')
      .attr('class', 'parallel-line')
      .attr('d', d => {
        const lineData = validDimensions.map(dim => ({
          dimension: dim,
          value: d[dim]
        }));
        return line(lineData);
      })
      .attr('stroke', d => d.is_anomaly ? '#ef4444' : '#3b82f6')
      .attr('stroke-width', 1)
      .attr('fill', 'none')
      .attr('opacity', 0.6);

    // Create axes
    validDimensions.forEach(dim => {
      const axis = this.axesGroup.append('g')
        .attr('class', 'parallel-axis')
        .attr('transform', `translate(${xScale(dim)},0)`)
        .call(d3.axisLeft(yScales[dim]));

      axis.append('text')
        .attr('class', 'axis-label')
        .attr('text-anchor', 'middle')
        .attr('y', -10)
        .text(dim);
    });
  }

  renderBoxPlot() {
    const width = this.options.width - this.options.margin.left - this.options.margin.right;
    const height = this.options.height - this.options.margin.top - this.options.margin.bottom;

    // Clear previous content
    this.dataGroup.selectAll('*').remove();
    this.axesGroup.selectAll('*').remove();

    const yField = document.getElementById('y-axis-selector').value;
    const groupField = document.getElementById('color-selector').value;

    // Group data
    const groupedData = d3.group(this.filteredData, d => d[groupField]);
    const boxPlotData = [];

    groupedData.forEach((values, group) => {
      const sortedValues = values.map(d => d[yField]).sort(d3.ascending);
      const q1 = d3.quantile(sortedValues, 0.25);
      const median = d3.quantile(sortedValues, 0.5);
      const q3 = d3.quantile(sortedValues, 0.75);
      const iqr = q3 - q1;
      const min = Math.max(d3.min(sortedValues), q1 - 1.5 * iqr);
      const max = Math.min(d3.max(sortedValues), q3 + 1.5 * iqr);

      boxPlotData.push({
        group: group,
        min: min,
        q1: q1,
        median: median,
        q3: q3,
        max: max,
        outliers: sortedValues.filter(d => d < min || d > max)
      });
    });

    const xScale = d3.scaleBand()
      .domain(boxPlotData.map(d => d.group))
      .range([0, width])
      .padding(0.2);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(this.filteredData, d => d[yField]))
      .range([height, 0]);

    this.scales = { x: xScale, y: yScale };

    // Create box plots
    const boxWidth = xScale.bandwidth();

    boxPlotData.forEach(d => {
      const x = xScale(d.group);

      // Draw box
      this.dataGroup.append('rect')
        .attr('class', 'box-plot-box')
        .attr('x', x)
        .attr('y', yScale(d.q3))
        .attr('width', boxWidth)
        .attr('height', yScale(d.q1) - yScale(d.q3))
        .attr('fill', '#3b82f6')
        .attr('fill-opacity', 0.3)
        .attr('stroke', '#3b82f6')
        .attr('stroke-width', 2);

      // Draw median line
      this.dataGroup.append('line')
        .attr('class', 'box-plot-median')
        .attr('x1', x)
        .attr('x2', x + boxWidth)
        .attr('y1', yScale(d.median))
        .attr('y2', yScale(d.median))
        .attr('stroke', '#1e40af')
        .attr('stroke-width', 3);

      // Draw whiskers
      this.dataGroup.append('line')
        .attr('class', 'box-plot-whisker')
        .attr('x1', x + boxWidth/2)
        .attr('x2', x + boxWidth/2)
        .attr('y1', yScale(d.q1))
        .attr('y2', yScale(d.min))
        .attr('stroke', '#3b82f6')
        .attr('stroke-width', 2);

      this.dataGroup.append('line')
        .attr('class', 'box-plot-whisker')
        .attr('x1', x + boxWidth/2)
        .attr('x2', x + boxWidth/2)
        .attr('y1', yScale(d.q3))
        .attr('y2', yScale(d.max))
        .attr('stroke', '#3b82f6')
        .attr('stroke-width', 2);

      // Draw outliers
      d.outliers.forEach(outlier => {
        this.dataGroup.append('circle')
          .attr('class', 'box-plot-outlier')
          .attr('cx', x + boxWidth/2)
          .attr('cy', yScale(outlier))
          .attr('r', 3)
          .attr('fill', '#ef4444')
          .attr('stroke', '#fff')
          .attr('stroke-width', 1);
      });
    });

    // Create axes
    this.axesGroup.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    this.axesGroup.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale));
  }

  renderViolinPlot() {
    // Implementation for violin plot
    // This would create violin-shaped distributions for each group
    console.log('Violin plot rendering not implemented yet');
  }

  renderNetworkGraph() {
    // Implementation for network graph
    // This would show relationships between data points
    console.log('Network graph rendering not implemented yet');
  }

  getColorScale(field) {
    const values = this.filteredData.map(d => d[field]);

    if (field === 'is_anomaly') {
      return d3.scaleOrdinal()
        .domain([true, false])
        .range(['#ef4444', '#3b82f6']);
    } else if (field === 'algorithm') {
      const algorithms = [...new Set(values)];
      return d3.scaleOrdinal(d3.schemeCategory10)
        .domain(algorithms);
    } else {
      return d3.scaleSequential(d3.interpolateViridis)
        .domain(d3.extent(values));
    }
  }

  getSizeScale(field) {
    if (field === 'none') {
      return () => 4;
    }

    const values = this.filteredData.map(d => d[field]);
    return d3.scaleSqrt()
      .domain(d3.extent(values))
      .range([2, 10]);
  }

  aggregateDataForHeatmap() {
    // Simple aggregation for demonstration
    const aggregated = [];
    const hours = d3.timeHour.range(
      d3.timeHour.floor(d3.min(this.filteredData, d => new Date(d.timestamp))),
      d3.timeHour.ceil(d3.max(this.filteredData, d => new Date(d.timestamp)))
    );

    const algorithms = [...new Set(this.filteredData.map(d => d.algorithm))];

    hours.forEach(hour => {
      algorithms.forEach(algorithm => {
        const hourData = this.filteredData.filter(d =>
          d.algorithm === algorithm &&
          d3.timeHour.floor(new Date(d.timestamp)).getTime() === hour.getTime()
        );

        aggregated.push({
          x: d3.timeFormat('%H:%M')(hour),
          y: algorithm,
          value: hourData.filter(d => d.is_anomaly).length
        });
      });
    });

    return aggregated;
  }

  setupZoomAndBrush() {
    if (!this.options.enableZoom && !this.options.enableBrush) return;

    const width = this.options.width - this.options.margin.left - this.options.margin.right;
    const height = this.options.height - this.options.margin.top - this.options.margin.bottom;

    if (this.options.enableZoom) {
      const zoom = d3.zoom()
        .scaleExtent([1, 10])
        .on('zoom', (event) => {
          this.zoomTransform = event.transform;
          this.applyZoom();
        });

      this.svg.call(zoom);
    }

    if (this.options.enableBrush) {
      const brush = d3.brush()
        .extent([[0, 0], [width, height]])
        .on('brush', (event) => {
          this.brushSelection = event.selection;
          this.applyBrush();
        });

      this.brushGroup.call(brush);
    }
  }

  applyZoom() {
    // Apply zoom transform to data points
    this.dataGroup.selectAll('.data-point')
      .attr('transform', this.zoomTransform);
  }

  applyBrush() {
    if (!this.brushSelection) return;

    const [[x0, y0], [x1, y1]] = this.brushSelection;

    this.dataGroup.selectAll('.data-point')
      .classed('brushed', d => {
        const x = this.scales.x(d[document.getElementById('x-axis-selector').value]);
        const y = this.scales.y(d[document.getElementById('y-axis-selector').value]);
        return x0 <= x && x <= x1 && y0 <= y && y <= y1;
      });
  }

  showTooltip(event, data) {
    const tooltip = d3.select('body').append('div')
      .attr('class', 'visualization-tooltip')
      .style('opacity', 0);

    const content = `
      <div class="tooltip-content">
        <div class="tooltip-header">${data.is_anomaly ? 'Anomaly' : 'Normal'}</div>
        <div class="tooltip-body">
          <div>Timestamp: ${new Date(data.timestamp).toLocaleString()}</div>
          <div>Value: ${data.value.toFixed(3)}</div>
          <div>Anomaly Score: ${data.anomaly_score.toFixed(3)}</div>
          <div>Algorithm: ${data.algorithm}</div>
          ${data.confidence ? `<div>Confidence: ${(data.confidence * 100).toFixed(1)}%</div>` : ''}
        </div>
      </div>
    `;

    tooltip.html(content)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 10) + 'px');

    tooltip.transition()
      .duration(200)
      .style('opacity', 1);
  }

  hideTooltip() {
    d3.selectAll('.visualization-tooltip').remove();
  }

  addColorLegend(colorScale) {
    const legendWidth = 200;
    const legendHeight = 20;

    const legend = this.svg.append('g')
      .attr('class', 'color-legend')
      .attr('transform', `translate(${this.options.width - legendWidth - 20}, 20)`);

    const legendScale = d3.scaleLinear()
      .domain(colorScale.domain())
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5);

    legend.append('g')
      .attr('transform', `translate(0, ${legendHeight})`)
      .call(legendAxis);

    // Create gradient
    const gradient = this.svg.append('defs')
      .append('linearGradient')
      .attr('id', 'color-gradient')
      .attr('x1', '0%')
      .attr('x2', '100%')
      .attr('y1', '0%')
      .attr('y2', '0%');

    gradient.selectAll('stop')
      .data(d3.range(0, 1.01, 0.1))
      .enter()
      .append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => colorScale(colorScale.domain()[0] + d * (colorScale.domain()[1] - colorScale.domain()[0])));

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#color-gradient)');
  }

  performClustering() {
    // Simple k-means clustering implementation
    const k = 3;
    const maxIterations = 100;

    // Extract features for clustering
    const features = this.filteredData.map(d => [
      d.value,
      d.anomaly_score,
      d.feature_1 || 0,
      d.feature_2 || 0
    ]);

    // Initialize centroids randomly
    let centroids = [];
    for (let i = 0; i < k; i++) {
      centroids.push(features[Math.floor(Math.random() * features.length)]);
    }

    // K-means iterations
    for (let iter = 0; iter < maxIterations; iter++) {
      // Assign points to clusters
      const clusters = Array(k).fill().map(() => []);

      features.forEach((point, index) => {
        let minDistance = Infinity;
        let clusterIndex = 0;

        centroids.forEach((centroid, i) => {
          const distance = this.euclideanDistance(point, centroid);
          if (distance < minDistance) {
            minDistance = distance;
            clusterIndex = i;
          }
        });

        clusters[clusterIndex].push(index);
        this.filteredData[index].cluster = clusterIndex;
      });

      // Update centroids
      const newCentroids = clusters.map(cluster => {
        if (cluster.length === 0) return centroids[0];

        const sum = cluster.reduce((acc, pointIndex) => {
          const point = features[pointIndex];
          return acc.map((val, i) => val + point[i]);
        }, Array(features[0].length).fill(0));

        return sum.map(val => val / cluster.length);
      });

      // Check convergence
      let converged = true;
      for (let i = 0; i < k; i++) {
        if (this.euclideanDistance(centroids[i], newCentroids[i]) > 0.001) {
          converged = false;
          break;
        }
      }

      centroids = newCentroids;
      if (converged) break;
    }
  }

  euclideanDistance(point1, point2) {
    return Math.sqrt(
      point1.reduce((sum, val, i) => sum + Math.pow(val - point2[i], 2), 0)
    );
  }

  setupInteractionHandlers() {
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'r' && e.ctrlKey) {
        e.preventDefault();
        this.resetView();
      } else if (e.key === 'e' && e.ctrlKey) {
        e.preventDefault();
        this.exportView();
      } else if (e.key === 'a' && e.ctrlKey) {
        e.preventDefault();
        this.animateView();
      }
    });

    // Resize handler
    window.addEventListener('resize', () => {
      this.handleResize();
    });
  }

  handleResize() {
    // Update dimensions and re-render
    const rect = this.container.getBoundingClientRect();
    this.options.width = rect.width;
    this.options.height = rect.height;

    this.svg
      .attr('width', this.options.width)
      .attr('height', this.options.height);

    this.updateVisualization();
  }

  resetView() {
    this.zoomTransform = null;
    this.brushSelection = null;
    this.svg.call(d3.zoom().transform, d3.zoomIdentity);
    this.brushGroup.call(d3.brush().clear);
    this.updateVisualization();
  }

  exportView() {
    const svgData = new XMLSerializer().serializeToString(this.svg.node());
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      const link = document.createElement('a');
      link.download = `anomaly-visualization-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    };

    img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
  }

  animateView() {
    // Animate between different views
    const views = ['scatter', 'timeline', 'heatmap', 'parallel'];
    const currentIndex = views.indexOf(this.currentView);
    const nextIndex = (currentIndex + 1) % views.length;

    document.getElementById('view-selector').value = views[nextIndex];
    this.currentView = views[nextIndex];

    this.updateVisualization();
  }

  destroy() {
    // Clean up event listeners and resources
    this.container.innerHTML = '';
    document.removeEventListener('keydown', this.handleKeyDown);
    window.removeEventListener('resize', this.handleResize);
  }
}

// Export the visualization class
window.InteractiveDataVisualization = InteractiveDataVisualization;
