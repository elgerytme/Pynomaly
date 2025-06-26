/**
 * Advanced D3.js Chart Library for Pynomaly
 * Provides interactive, real-time data visualization components
 * with accessibility, responsive design, and performance optimization
 */

class D3ChartLibrary {
  constructor() {
    this.charts = new Map();
    this.themes = {
      light: {
        background: '#ffffff',
        text: '#1e293b',
        grid: '#e2e8f0',
        primary: '#0ea5e9',
        secondary: '#22c55e',
        danger: '#dc2626',
        warning: '#d97706'
      },
      dark: {
        background: '#0f172a',
        text: '#f1f5f9',
        grid: '#334155',
        primary: '#38bdf8',
        secondary: '#4ade80',
        danger: '#f87171',
        warning: '#fbbf24'
      }
    };
    this.currentTheme = 'light';
    this.setupEventListeners();
  }

  setupEventListeners() {
    // Theme change listener
    document.addEventListener('theme-changed', (event) => {
      this.currentTheme = event.detail.theme;
      this.updateAllChartsTheme();
    });

    // Resize listener for responsive charts
    window.addEventListener('resize', this.debounce(() => {
      this.resizeAllCharts();
    }, 250));
  }

  // Utility function for debouncing
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

  // Get current theme colors
  getTheme() {
    return this.themes[this.currentTheme];
  }

  // Register a chart instance
  registerChart(id, chart) {
    this.charts.set(id, chart);
  }

  // Remove a chart instance
  removeChart(id) {
    const chart = this.charts.get(id);
    if (chart && chart.destroy) {
      chart.destroy();
    }
    this.charts.delete(id);
  }

  // Update all charts theme
  updateAllChartsTheme() {
    this.charts.forEach(chart => {
      if (chart.updateTheme) {
        chart.updateTheme(this.getTheme());
      }
    });
  }

  // Resize all charts
  resizeAllCharts() {
    this.charts.forEach(chart => {
      if (chart.resize) {
        chart.resize();
      }
    });
  }
}

/**
 * Base Chart Class
 * Provides common functionality for all chart types
 */
class BaseChart {
  constructor(container, options = {}) {
    this.container = d3.select(container);
    this.containerElement = container;
    this.options = {
      width: 800,
      height: 400,
      margin: { top: 20, right: 30, bottom: 40, left: 50 },
      animated: true,
      responsive: true,
      accessibility: true,
      ...options
    };

    this.data = [];
    this.svg = null;
    this.tooltip = null;
    this.theme = chartLibrary.getTheme();
    
    this.setupChart();
    this.setupAccessibility();
    
    // Register with chart library
    const chartId = this.containerElement.id || `chart-${Date.now()}`;
    chartLibrary.registerChart(chartId, this);
  }

  setupChart() {
    // Clear existing content
    this.container.selectAll('*').remove();

    // Calculate dimensions
    this.updateDimensions();

    // Create main SVG
    this.svg = this.container
      .append('svg')
      .attr('width', this.width)
      .attr('height', this.height)
      .attr('role', 'img')
      .attr('aria-labelledby', `${this.containerElement.id}-title`)
      .attr('aria-describedby', `${this.containerElement.id}-desc`);

    // Create main group for chart content
    this.chartGroup = this.svg
      .append('g')
      .attr('transform', `translate(${this.options.margin.left},${this.options.margin.top})`);

    // Setup tooltip
    this.setupTooltip();
  }

  setupAccessibility() {
    if (!this.options.accessibility) return;

    // Add title and description elements
    const titleId = `${this.containerElement.id}-title`;
    const descId = `${this.containerElement.id}-desc`;

    if (!document.getElementById(titleId)) {
      const titleElement = document.createElement('h3');
      titleElement.id = titleId;
      titleElement.className = 'sr-only';
      titleElement.textContent = this.options.title || 'Data Visualization';
      this.containerElement.insertBefore(titleElement, this.containerElement.firstChild);
    }

    if (!document.getElementById(descId)) {
      const descElement = document.createElement('div');
      descElement.id = descId;
      descElement.className = 'sr-only';
      descElement.textContent = this.options.description || 'Interactive chart showing data trends and patterns';
      this.containerElement.appendChild(descElement);
    }
  }

  setupTooltip() {
    this.tooltip = d3.select('body')
      .append('div')
      .attr('class', 'chart-tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('padding', '8px 12px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('z-index', '1000');
  }

  updateDimensions() {
    if (this.options.responsive) {
      const containerWidth = this.containerElement.clientWidth || this.options.width;
      this.width = Math.max(300, containerWidth);
      this.height = Math.max(200, this.width * 0.5); // Maintain aspect ratio
    } else {
      this.width = this.options.width;
      this.height = this.options.height;
    }

    this.innerWidth = this.width - this.options.margin.left - this.options.margin.right;
    this.innerHeight = this.height - this.options.margin.top - this.options.margin.bottom;
  }

  showTooltip(content, event) {
    this.tooltip
      .html(content)
      .style('opacity', 1)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 10) + 'px');
  }

  hideTooltip() {
    this.tooltip.style('opacity', 0);
  }

  updateTheme(newTheme) {
    this.theme = newTheme;
    this.render();
  }

  resize() {
    this.updateDimensions();
    this.svg
      .attr('width', this.width)
      .attr('height', this.height);
    this.render();
  }

  setData(data) {
    this.data = data;
    this.render();
  }

  render() {
    // Override in subclasses
    throw new Error('render() method must be implemented by subclass');
  }

  destroy() {
    if (this.tooltip) {
      this.tooltip.remove();
    }
    this.container.selectAll('*').remove();
  }
}

/**
 * Time Series Chart
 * Interactive line chart for time-based anomaly detection data
 */
class TimeSeriesChart extends BaseChart {
  constructor(container, options = {}) {
    super(container, {
      xAccessor: d => d.timestamp,
      yAccessor: d => d.value,
      anomalyAccessor: d => d.isAnomaly,
      confidenceAccessor: d => d.confidence,
      showAnomalies: true,
      showConfidenceBands: false,
      interpolation: d3.curveMonotoneX,
      ...options
    });

    this.xScale = d3.scaleTime();
    this.yScale = d3.scaleLinear();
    this.line = d3.line()
      .x(d => this.xScale(this.options.xAccessor(d)))
      .y(d => this.yScale(this.options.yAccessor(d)))
      .curve(this.options.interpolation);
  }

  render() {
    if (!this.data || this.data.length === 0) return;

    // Clear previous content
    this.chartGroup.selectAll('*').remove();

    // Update scales
    this.xScale
      .domain(d3.extent(this.data, this.options.xAccessor))
      .range([0, this.innerWidth]);

    this.yScale
      .domain(d3.extent(this.data, this.options.yAccessor))
      .nice()
      .range([this.innerHeight, 0]);

    // Add axes
    this.addAxes();

    // Add main line
    this.addMainLine();

    // Add anomaly markers
    if (this.options.showAnomalies) {
      this.addAnomalyMarkers();
    }

    // Add confidence bands
    if (this.options.showConfidenceBands) {
      this.addConfidenceBands();
    }

    // Add interaction
    this.addInteraction();
  }

  addAxes() {
    // X-axis
    const xAxis = d3.axisBottom(this.xScale)
      .tickFormat(d3.timeFormat('%H:%M'));

    this.chartGroup
      .append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${this.innerHeight})`)
      .call(xAxis)
      .selectAll('text')
      .style('fill', this.theme.text);

    // Y-axis
    const yAxis = d3.axisLeft(this.yScale);

    this.chartGroup
      .append('g')
      .attr('class', 'y-axis')
      .call(yAxis)
      .selectAll('text')
      .style('fill', this.theme.text);

    // Style axes
    this.chartGroup.selectAll('.domain, .tick line')
      .style('stroke', this.theme.grid);
  }

  addMainLine() {
    const path = this.chartGroup
      .append('path')
      .datum(this.data)
      .attr('class', 'main-line')
      .attr('fill', 'none')
      .attr('stroke', this.theme.primary)
      .attr('stroke-width', 2)
      .attr('d', this.line);

    // Animate path drawing
    if (this.options.animated) {
      const totalLength = path.node().getTotalLength();
      path
        .attr('stroke-dasharray', totalLength + ' ' + totalLength)
        .attr('stroke-dashoffset', totalLength)
        .transition()
        .duration(1500)
        .ease(d3.easeLinear)
        .attr('stroke-dashoffset', 0);
    }
  }

  addAnomalyMarkers() {
    const anomalies = this.data.filter(this.options.anomalyAccessor);

    const markers = this.chartGroup
      .selectAll('.anomaly-marker')
      .data(anomalies)
      .enter()
      .append('circle')
      .attr('class', 'anomaly-marker')
      .attr('cx', d => this.xScale(this.options.xAccessor(d)))
      .attr('cy', d => this.yScale(this.options.yAccessor(d)))
      .attr('r', 0)
      .attr('fill', this.theme.danger)
      .attr('stroke', this.theme.background)
      .attr('stroke-width', 2)
      .attr('opacity', 0.8)
      .style('cursor', 'pointer');

    // Animate markers
    if (this.options.animated) {
      markers
        .transition()
        .delay((d, i) => i * 100)
        .duration(300)
        .attr('r', 5);
    } else {
      markers.attr('r', 5);
    }

    // Add tooltip interaction
    markers
      .on('mouseover', (event, d) => {
        const confidence = this.options.confidenceAccessor(d);
        const timestamp = d3.timeFormat('%Y-%m-%d %H:%M:%S')(this.options.xAccessor(d));
        const value = this.options.yAccessor(d).toFixed(2);
        
        const content = `
          <strong>Anomaly Detected</strong><br/>
          Time: ${timestamp}<br/>
          Value: ${value}<br/>
          Confidence: ${(confidence * 100).toFixed(1)}%
        `;
        
        this.showTooltip(content, event);
        
        // Enlarge marker on hover
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr('r', 7);
      })
      .on('mouseout', (event) => {
        this.hideTooltip();
        
        // Reset marker size
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr('r', 5);
      })
      .on('click', (event, d) => {
        // Emit custom event for anomaly selection
        this.containerElement.dispatchEvent(new CustomEvent('anomaly-selected', {
          detail: { data: d, chart: this }
        }));
      });

    // Add accessibility labels
    markers.attr('aria-label', d => {
      const timestamp = d3.timeFormat('%Y-%m-%d %H:%M:%S')(this.options.xAccessor(d));
      const value = this.options.yAccessor(d).toFixed(2);
      const confidence = this.options.confidenceAccessor(d);
      return `Anomaly at ${timestamp}, value ${value}, confidence ${(confidence * 100).toFixed(1)}%`;
    });
  }

  addConfidenceBands() {
    // Calculate confidence intervals (simplified implementation)
    const confidenceData = this.data.map(d => {
      const confidence = this.options.confidenceAccessor(d) || 0.5;
      const value = this.options.yAccessor(d);
      const margin = value * (1 - confidence) * 0.5;
      
      return {
        ...d,
        upper: value + margin,
        lower: value - margin
      };
    });

    // Create area generator for confidence band
    const area = d3.area()
      .x(d => this.xScale(this.options.xAccessor(d)))
      .y0(d => this.yScale(d.lower))
      .y1(d => this.yScale(d.upper))
      .curve(this.options.interpolation);

    // Add confidence band
    this.chartGroup
      .append('path')
      .datum(confidenceData)
      .attr('class', 'confidence-band')
      .attr('fill', this.theme.primary)
      .attr('fill-opacity', 0.2)
      .attr('d', area);
  }

  addInteraction() {
    // Add invisible overlay for mouse tracking
    const overlay = this.chartGroup
      .append('rect')
      .attr('class', 'overlay')
      .attr('width', this.innerWidth)
      .attr('height', this.innerHeight)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .style('cursor', 'crosshair');

    // Add vertical line for cursor tracking
    const verticalLine = this.chartGroup
      .append('line')
      .attr('class', 'cursor-line')
      .attr('stroke', this.theme.text)
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('opacity', 0);

    // Add mouse interaction
    overlay
      .on('mousemove', (event) => {
        const [mouseX] = d3.pointer(event);
        const xValue = this.xScale.invert(mouseX);
        
        // Find closest data point
        const bisector = d3.bisector(this.options.xAccessor).left;
        const index = bisector(this.data, xValue);
        const d0 = this.data[index - 1];
        const d1 = this.data[index];
        
        if (d0 && d1) {
          const d = xValue - this.options.xAccessor(d0) > this.options.xAccessor(d1) - xValue ? d1 : d0;
          
          // Update vertical line
          verticalLine
            .attr('x1', this.xScale(this.options.xAccessor(d)))
            .attr('x2', this.xScale(this.options.xAccessor(d)))
            .attr('y1', 0)
            .attr('y2', this.innerHeight)
            .attr('opacity', 0.5);

          // Show tooltip with data point info
          const timestamp = d3.timeFormat('%Y-%m-%d %H:%M:%S')(this.options.xAccessor(d));
          const value = this.options.yAccessor(d).toFixed(2);
          
          const content = `
            <strong>Data Point</strong><br/>
            Time: ${timestamp}<br/>
            Value: ${value}
          `;
          
          this.showTooltip(content, event);
        }
      })
      .on('mouseleave', () => {
        verticalLine.attr('opacity', 0);
        this.hideTooltip();
      });
  }

  // Real-time data update method
  updateData(newData, animate = true) {
    this.data = newData;
    
    if (animate) {
      // Animate the update
      this.render();
    } else {
      this.render();
    }

    // Announce update to screen readers
    if (this.options.accessibility) {
      const announcement = `Chart updated with ${newData.length} data points`;
      this.announceToScreenReader(announcement);
    }
  }

  // Add a single data point (for real-time streaming)
  addDataPoint(point, maxPoints = 1000) {
    this.data.push(point);
    
    // Keep only the last maxPoints
    if (this.data.length > maxPoints) {
      this.data = this.data.slice(-maxPoints);
    }
    
    this.render();
  }

  announceToScreenReader(message) {
    const announcer = document.getElementById('chart-announcer');
    if (announcer) {
      announcer.textContent = message;
    }
  }
}

/**
 * Scatter Plot Chart
 * Interactive scatter plot for anomaly detection in 2D space
 */
class ScatterPlotChart extends BaseChart {
  constructor(container, options = {}) {
    super(container, {
      xAccessor: d => d.x,
      yAccessor: d => d.y,
      colorAccessor: d => d.anomalyScore,
      sizeAccessor: d => d.confidence,
      anomalyAccessor: d => d.isAnomaly,
      showDensity: false,
      enableBrushing: true,
      enableZoom: true,
      ...options
    });

    this.xScale = d3.scaleLinear();
    this.yScale = d3.scaleLinear();
    this.colorScale = d3.scaleSequential(d3.interpolateViridis);
    this.sizeScale = d3.scaleSqrt().range([3, 12]);
    
    this.brush = null;
    this.zoom = null;
  }

  render() {
    if (!this.data || this.data.length === 0) return;

    // Clear previous content
    this.chartGroup.selectAll('*').remove();

    // Update scales
    this.updateScales();

    // Add axes
    this.addAxes();

    // Add density background if enabled
    if (this.options.showDensity) {
      this.addDensityBackground();
    }

    // Add scatter points
    this.addScatterPoints();

    // Add brushing if enabled
    if (this.options.enableBrushing) {
      this.addBrushing();
    }

    // Add zoom if enabled
    if (this.options.enableZoom) {
      this.addZoom();
    }

    // Add legend
    this.addLegend();
  }

  updateScales() {
    this.xScale
      .domain(d3.extent(this.data, this.options.xAccessor))
      .nice()
      .range([0, this.innerWidth]);

    this.yScale
      .domain(d3.extent(this.data, this.options.yAccessor))
      .nice()
      .range([this.innerHeight, 0]);

    this.colorScale
      .domain(d3.extent(this.data, this.options.colorAccessor));

    this.sizeScale
      .domain(d3.extent(this.data, this.options.sizeAccessor));
  }

  addAxes() {
    // X-axis
    this.chartGroup
      .append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${this.innerHeight})`)
      .call(d3.axisBottom(this.xScale))
      .selectAll('text')
      .style('fill', this.theme.text);

    // Y-axis
    this.chartGroup
      .append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(this.yScale))
      .selectAll('text')
      .style('fill', this.theme.text);

    // Axis labels
    this.chartGroup
      .append('text')
      .attr('class', 'x-label')
      .attr('text-anchor', 'middle')
      .attr('x', this.innerWidth / 2)
      .attr('y', this.innerHeight + 35)
      .style('fill', this.theme.text)
      .text(this.options.xLabel || 'X Value');

    this.chartGroup
      .append('text')
      .attr('class', 'y-label')
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('x', -this.innerHeight / 2)
      .attr('y', -35)
      .style('fill', this.theme.text)
      .text(this.options.yLabel || 'Y Value');

    // Style axes
    this.chartGroup.selectAll('.domain, .tick line')
      .style('stroke', this.theme.grid);
  }

  addScatterPoints() {
    const points = this.chartGroup
      .selectAll('.data-point')
      .data(this.data)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', d => this.xScale(this.options.xAccessor(d)))
      .attr('cy', d => this.yScale(this.options.yAccessor(d)))
      .attr('r', 0)
      .attr('fill', d => {
        if (this.options.anomalyAccessor(d)) {
          return this.theme.danger;
        }
        return this.colorScale(this.options.colorAccessor(d));
      })
      .attr('stroke', d => this.options.anomalyAccessor(d) ? this.theme.background : 'none')
      .attr('stroke-width', d => this.options.anomalyAccessor(d) ? 2 : 0)
      .attr('opacity', 0.7)
      .style('cursor', 'pointer');

    // Animate points
    if (this.options.animated) {
      points
        .transition()
        .delay((d, i) => i * 10)
        .duration(500)
        .attr('r', d => this.sizeScale(this.options.sizeAccessor(d)));
    } else {
      points.attr('r', d => this.sizeScale(this.options.sizeAccessor(d)));
    }

    // Add interaction
    points
      .on('mouseover', (event, d) => {
        const content = `
          <strong>${this.options.anomalyAccessor(d) ? 'Anomaly' : 'Normal Point'}</strong><br/>
          X: ${this.options.xAccessor(d).toFixed(2)}<br/>
          Y: ${this.options.yAccessor(d).toFixed(2)}<br/>
          Score: ${this.options.colorAccessor(d).toFixed(3)}<br/>
          Confidence: ${(this.options.sizeAccessor(d) * 100).toFixed(1)}%
        `;
        
        this.showTooltip(content, event);
        
        // Highlight point
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr('r', d => this.sizeScale(this.options.sizeAccessor(d)) * 1.5)
          .attr('opacity', 1);
      })
      .on('mouseout', (event, d) => {
        this.hideTooltip();
        
        // Reset point
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr('r', d => this.sizeScale(this.options.sizeAccessor(d)))
          .attr('opacity', 0.7);
      })
      .on('click', (event, d) => {
        // Emit selection event
        this.containerElement.dispatchEvent(new CustomEvent('point-selected', {
          detail: { data: d, chart: this }
        }));
      });
  }

  addLegend() {
    const legendGroup = this.svg
      .append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${this.width - 120}, 20)`);

    // Color legend
    const colorLegend = legendGroup
      .append('g')
      .attr('class', 'color-legend');

    const colorGradient = this.svg
      .append('defs')
      .append('linearGradient')
      .attr('id', 'color-gradient')
      .attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', 0).attr('y1', 0)
      .attr('x2', 0).attr('y2', 100);

    colorGradient.selectAll('stop')
      .data(d3.range(0, 1.1, 0.1))
      .enter()
      .append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => this.colorScale(d3.quantile(this.data.map(this.options.colorAccessor), d)));

    colorLegend
      .append('rect')
      .attr('width', 20)
      .attr('height', 100)
      .style('fill', 'url(#color-gradient)');

    colorLegend
      .append('text')
      .attr('x', 25)
      .attr('y', 10)
      .style('fill', this.theme.text)
      .style('font-size', '12px')
      .text('Anomaly Score');

    // Size legend
    const sizeLegend = legendGroup
      .append('g')
      .attr('class', 'size-legend')
      .attr('transform', 'translate(0, 120)');

    const sizeValues = [0.2, 0.5, 0.8];
    sizeLegend.selectAll('.size-legend-item')
      .data(sizeValues)
      .enter()
      .append('g')
      .attr('class', 'size-legend-item')
      .attr('transform', (d, i) => `translate(0, ${i * 25})`)
      .each(function(d) {
        const group = d3.select(this);
        
        group.append('circle')
          .attr('cx', 10)
          .attr('cy', 0)
          .attr('r', d => this.sizeScale(d))
          .attr('fill', this.theme.primary)
          .attr('opacity', 0.7);
        
        group.append('text')
          .attr('x', 25)
          .attr('y', 4)
          .style('fill', this.theme.text)
          .style('font-size', '12px')
          .text(`${(d * 100).toFixed(0)}%`);
      });

    sizeLegend
      .append('text')
      .attr('x', 0)
      .attr('y', -10)
      .style('fill', this.theme.text)
      .style('font-size', '12px')
      .text('Confidence');
  }

  addBrushing() {
    this.brush = d3.brush()
      .extent([[0, 0], [this.innerWidth, this.innerHeight]])
      .on('end', (event) => {
        const selection = event.selection;
        if (!selection) {
          // Clear selection
          this.chartGroup.selectAll('.data-point')
            .classed('selected', false)
            .attr('opacity', 0.7);
          return;
        }

        const [[x0, y0], [x1, y1]] = selection;
        
        // Select points within brush
        const selectedData = [];
        this.chartGroup.selectAll('.data-point')
          .classed('selected', d => {
            const x = this.xScale(this.options.xAccessor(d));
            const y = this.yScale(this.options.yAccessor(d));
            const isSelected = x >= x0 && x <= x1 && y >= y0 && y <= y1;
            
            if (isSelected) {
              selectedData.push(d);
            }
            
            return isSelected;
          })
          .attr('opacity', d => {
            const x = this.xScale(this.options.xAccessor(d));
            const y = this.yScale(this.options.yAccessor(d));
            return (x >= x0 && x <= x1 && y >= y0 && y <= y1) ? 1 : 0.3;
          });

        // Emit selection event
        this.containerElement.dispatchEvent(new CustomEvent('points-selected', {
          detail: { data: selectedData, chart: this }
        }));
      });

    this.chartGroup
      .append('g')
      .attr('class', 'brush')
      .call(this.brush);
  }

  addZoom() {
    this.zoom = d3.zoom()
      .scaleExtent([1, 10])
      .on('zoom', (event) => {
        const { transform } = event;
        
        // Create new scales based on zoom transform
        const newXScale = transform.rescaleX(this.xScale);
        const newYScale = transform.rescaleY(this.yScale);
        
        // Update axes
        this.chartGroup.select('.x-axis')
          .call(d3.axisBottom(newXScale));
        
        this.chartGroup.select('.y-axis')
          .call(d3.axisLeft(newYScale));
        
        // Update points
        this.chartGroup.selectAll('.data-point')
          .attr('cx', d => newXScale(this.options.xAccessor(d)))
          .attr('cy', d => newYScale(this.options.yAccessor(d)));
      });

    this.svg.call(this.zoom);
  }
}

/**
 * Heatmap Chart
 * Interactive heatmap for correlation and anomaly density visualization
 */
class HeatmapChart extends BaseChart {
  constructor(container, options = {}) {
    super(container, {
      xAccessor: d => d.x,
      yAccessor: d => d.y,
      valueAccessor: d => d.value,
      gridSize: 20,
      colorScheme: d3.interpolateViridis,
      showLabels: true,
      enableZoom: false,
      ...options
    });

    this.xScale = d3.scaleBand();
    this.yScale = d3.scaleBand();
    this.colorScale = d3.scaleSequential();
  }

  render() {
    if (!this.data || this.data.length === 0) return;

    // Clear previous content
    this.chartGroup.selectAll('*').remove();

    // Process data into grid format
    this.processData();

    // Update scales
    this.updateScales();

    // Add axes
    this.addAxes();

    // Add heatmap cells
    this.addHeatmapCells();

    // Add color legend
    this.addColorLegend();
  }

  processData() {
    // Create grid from data
    const xValues = [...new Set(this.data.map(this.options.xAccessor))].sort();
    const yValues = [...new Set(this.data.map(this.options.yAccessor))].sort();
    
    this.gridData = [];
    
    for (let y of yValues) {
      for (let x of xValues) {
        const dataPoint = this.data.find(d => 
          this.options.xAccessor(d) === x && this.options.yAccessor(d) === y
        );
        
        this.gridData.push({
          x: x,
          y: y,
          value: dataPoint ? this.options.valueAccessor(dataPoint) : 0,
          original: dataPoint
        });
      }
    }
    
    this.xValues = xValues;
    this.yValues = yValues;
  }

  updateScales() {
    this.xScale
      .domain(this.xValues)
      .range([0, this.innerWidth])
      .padding(0.1);

    this.yScale
      .domain(this.yValues)
      .range([this.innerHeight, 0])
      .padding(0.1);

    this.colorScale
      .interpolator(this.options.colorScheme)
      .domain(d3.extent(this.gridData, d => d.value));
  }

  addAxes() {
    // X-axis
    this.chartGroup
      .append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${this.innerHeight})`)
      .call(d3.axisBottom(this.xScale))
      .selectAll('text')
      .style('fill', this.theme.text)
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em')
      .attr('transform', 'rotate(-65)');

    // Y-axis
    this.chartGroup
      .append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(this.yScale))
      .selectAll('text')
      .style('fill', this.theme.text);

    // Style axes
    this.chartGroup.selectAll('.domain, .tick line')
      .style('stroke', this.theme.grid);
  }

  addHeatmapCells() {
    const cells = this.chartGroup
      .selectAll('.heatmap-cell')
      .data(this.gridData)
      .enter()
      .append('rect')
      .attr('class', 'heatmap-cell')
      .attr('x', d => this.xScale(d.x))
      .attr('y', d => this.yScale(d.y))
      .attr('width', this.xScale.bandwidth())
      .attr('height', this.yScale.bandwidth())
      .attr('fill', d => this.colorScale(d.value))
      .attr('stroke', this.theme.background)
      .attr('stroke-width', 1)
      .attr('opacity', 0)
      .style('cursor', 'pointer');

    // Animate cells
    if (this.options.animated) {
      cells
        .transition()
        .delay((d, i) => i * 20)
        .duration(500)
        .attr('opacity', 1);
    } else {
      cells.attr('opacity', 1);
    }

    // Add interaction
    cells
      .on('mouseover', (event, d) => {
        const content = `
          <strong>Cell (${d.x}, ${d.y})</strong><br/>
          Value: ${d.value.toFixed(3)}
        `;
        
        this.showTooltip(content, event);
        
        // Highlight cell
        d3.select(event.target)
          .attr('stroke-width', 3)
          .attr('stroke', this.theme.text);
      })
      .on('mouseout', (event) => {
        this.hideTooltip();
        
        // Reset cell
        d3.select(event.target)
          .attr('stroke-width', 1)
          .attr('stroke', this.theme.background);
      })
      .on('click', (event, d) => {
        // Emit selection event
        this.containerElement.dispatchEvent(new CustomEvent('cell-selected', {
          detail: { data: d, chart: this }
        }));
      });

    // Add labels if enabled
    if (this.options.showLabels) {
      this.chartGroup
        .selectAll('.cell-label')
        .data(this.gridData.filter(d => d.value > 0))
        .enter()
        .append('text')
        .attr('class', 'cell-label')
        .attr('x', d => this.xScale(d.x) + this.xScale.bandwidth() / 2)
        .attr('y', d => this.yScale(d.y) + this.yScale.bandwidth() / 2)
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .style('fill', d => d.value > this.colorScale.domain()[1] * 0.6 ? 'white' : 'black')
        .style('font-size', '10px')
        .style('pointer-events', 'none')
        .text(d => d.value.toFixed(2));
    }
  }

  addColorLegend() {
    const legendGroup = this.svg
      .append('g')
      .attr('class', 'color-legend')
      .attr('transform', `translate(${this.width - 100}, 20)`);

    // Create gradient
    const gradient = this.svg
      .append('defs')
      .append('linearGradient')
      .attr('id', 'heatmap-gradient')
      .attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', 0).attr('y1', 100)
      .attr('x2', 0).attr('y2', 0);

    gradient.selectAll('stop')
      .data(d3.range(0, 1.1, 0.1))
      .enter()
      .append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => this.colorScale(d3.quantile(this.colorScale.domain(), d)));

    // Legend rectangle
    legendGroup
      .append('rect')
      .attr('width', 20)
      .attr('height', 100)
      .style('fill', 'url(#heatmap-gradient)')
      .style('stroke', this.theme.grid);

    // Legend scale
    const legendScale = d3.scaleLinear()
      .domain(this.colorScale.domain())
      .range([100, 0]);

    const legendAxis = d3.axisRight(legendScale)
      .tickSize(6)
      .tickFormat(d3.format('.2f'));

    legendGroup
      .append('g')
      .attr('class', 'legend-axis')
      .attr('transform', 'translate(20, 0)')
      .call(legendAxis)
      .selectAll('text')
      .style('fill', this.theme.text)
      .style('font-size', '10px');

    // Legend title
    legendGroup
      .append('text')
      .attr('x', 0)
      .attr('y', -5)
      .style('fill', this.theme.text)
      .style('font-size', '12px')
      .text('Value');
  }
}

// Initialize the chart library
const chartLibrary = new D3ChartLibrary();

// Export classes for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    D3ChartLibrary,
    BaseChart,
    TimeSeriesChart,
    ScatterPlotChart,
    HeatmapChart
  };
} else {
  // Browser environment
  window.D3ChartLibrary = D3ChartLibrary;
  window.BaseChart = BaseChart;
  window.TimeSeriesChart = TimeSeriesChart;
  window.ScatterPlotChart = ScatterPlotChart;
  window.HeatmapChart = HeatmapChart;
  window.chartLibrary = chartLibrary;
}