/**
 * D3.js Chart Library - Core visualization components
 * Provides reusable D3.js charts for anomaly detection visualization
 */

class D3ChartLibrary {
  constructor() {
    this.charts = new Map();
    this.defaultConfig = {
      margin: { top: 20, right: 20, bottom: 30, left: 40 },
      width: 800,
      height: 400,
      colors: {
        normal: '#2563eb',
        anomaly: '#dc2626',
        warning: '#f59e0b',
        background: '#f8fafc'
      },
      animations: {
        duration: 750,
        easing: 'ease-in-out'
      }
    };
  }

  /**
   * Creates an anomaly scatter plot with interactive features
   */
  createAnomalyScatterPlot(containerId, data, config = {}) {
    const finalConfig = { ...this.defaultConfig, ...config };
    const container = d3.select(`#${containerId}`);
    
    // Clear existing content
    container.selectAll("*").remove();
    
    const { margin, width, height, colors } = finalConfig;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG
    const svg = container
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("class", "anomaly-scatter-plot");

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create scales
    const xScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.x))
      .range([0, innerWidth])
      .nice();

    const yScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.y))
      .range([innerHeight, 0])
      .nice();

    const colorScale = d3.scaleOrdinal()
      .domain(['normal', 'anomaly', 'warning'])
      .range([colors.normal, colors.anomaly, colors.warning]);

    // Create axes
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d3.format('.2f'));
    
    const yAxis = d3.axisLeft(yScale)
      .tickFormat(d3.format('.2f'));

    g.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis);

    g.append("g")
      .attr("class", "y-axis")
      .call(yAxis);

    // Add axis labels
    g.append("text")
      .attr("class", "axis-label")
      .attr("text-anchor", "middle")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + margin.bottom - 5)
      .text(config.xLabel || "Feature 1");

    g.append("text")
      .attr("class", "axis-label")
      .attr("text-anchor", "middle")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -margin.left + 15)
      .text(config.yLabel || "Feature 2");

    // Create tooltip
    const tooltip = d3.select("body").append("div")
      .attr("class", "d3-tooltip")
      .style("position", "absolute")
      .style("padding", "8px")
      .style("background", "rgba(0, 0, 0, 0.8)")
      .style("color", "white")
      .style("border-radius", "4px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("opacity", 0);

    // Add points
    const circles = g.selectAll(".data-point")
      .data(data)
      .enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 0)
      .attr("fill", d => colorScale(d.type))
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr("r", 8)
          .attr("stroke-width", 2);

        tooltip.transition()
          .duration(200)
          .style("opacity", 1);

        tooltip.html(`
          <strong>Sample ID:</strong> ${d.id}<br/>
          <strong>X:</strong> ${d.x.toFixed(3)}<br/>
          <strong>Y:</strong> ${d.y.toFixed(3)}<br/>
          <strong>Type:</strong> ${d.type}<br/>
          <strong>Score:</strong> ${d.score?.toFixed(3) || 'N/A'}
        `)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 10) + "px");
      })
      .on("mouseout", function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr("r", d.type === 'anomaly' ? 6 : 4)
          .attr("stroke-width", 1);

        tooltip.transition()
          .duration(200)
          .style("opacity", 0);
      })
      .on("click", function(event, d) {
        // Emit custom event for drill-down
        const customEvent = new CustomEvent('pointClicked', {
          detail: { data: d, chart: 'scatter' }
        });
        document.dispatchEvent(customEvent);
      });

    // Animate points in
    circles.transition()
      .duration(finalConfig.animations.duration)
      .delay((d, i) => i * 10)
      .attr("r", d => d.type === 'anomaly' ? 6 : 4);

    // Add legend
    const legend = g.append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${innerWidth - 100}, 20)`);

    const legendData = [
      { type: 'normal', label: 'Normal' },
      { type: 'anomaly', label: 'Anomaly' },
      { type: 'warning', label: 'Warning' }
    ];

    const legendItems = legend.selectAll(".legend-item")
      .data(legendData)
      .enter()
      .append("g")
      .attr("class", "legend-item")
      .attr("transform", (d, i) => `translate(0, ${i * 20})`);

    legendItems.append("circle")
      .attr("cx", 8)
      .attr("cy", 8)
      .attr("r", 4)
      .attr("fill", d => colorScale(d.type));

    legendItems.append("text")
      .attr("x", 20)
      .attr("y", 8)
      .attr("dy", "0.35em")
      .style("font-size", "12px")
      .text(d => d.label);

    // Store chart reference
    this.charts.set(containerId, {
      svg,
      xScale,
      yScale,
      colorScale,
      data,
      config: finalConfig
    });

    return this;
  }

  /**
   * Creates an interactive time series chart
   */
  createTimeSeriesChart(containerId, data, config = {}) {
    const finalConfig = { ...this.defaultConfig, ...config };
    const container = d3.select(`#${containerId}`);
    
    container.selectAll("*").remove();
    
    const { margin, width, height, colors } = finalConfig;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = container
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("class", "time-series-chart");

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Parse dates
    const parseTime = d3.timeParse("%Y-%m-%d %H:%M:%S");
    data.forEach(d => {
      d.date = parseTime(d.timestamp) || new Date(d.timestamp);
    });

    // Create scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(data, d => d.date))
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.value))
      .range([innerHeight, 0])
      .nice();

    // Create line generator
    const line = d3.line()
      .x(d => xScale(d.date))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    // Create axes
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d3.timeFormat("%H:%M"));
    
    const yAxis = d3.axisLeft(yScale);

    g.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis);

    g.append("g")
      .attr("class", "y-axis")
      .call(yAxis);

    // Add grid lines
    g.append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .tickSize(-innerHeight)
        .tickFormat("")
      );

    g.append("g")
      .attr("class", "grid")
      .call(d3.axisLeft(yScale)
        .tickSize(-innerWidth)
        .tickFormat("")
      );

    // Add line path
    const path = g.append("path")
      .datum(data)
      .attr("class", "line")
      .attr("fill", "none")
      .attr("stroke", colors.normal)
      .attr("stroke-width", 2)
      .attr("d", line);

    // Animate line drawing
    const totalLength = path.node().getTotalLength();
    path
      .attr("stroke-dasharray", totalLength + " " + totalLength)
      .attr("stroke-dashoffset", totalLength)
      .transition()
      .duration(finalConfig.animations.duration)
      .attr("stroke-dashoffset", 0);

    // Add anomaly points
    const anomalies = data.filter(d => d.is_anomaly);
    g.selectAll(".anomaly-point")
      .data(anomalies)
      .enter()
      .append("circle")
      .attr("class", "anomaly-point")
      .attr("cx", d => xScale(d.date))
      .attr("cy", d => yScale(d.value))
      .attr("r", 0)
      .attr("fill", colors.anomaly)
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .transition()
      .delay(finalConfig.animations.duration)
      .duration(300)
      .attr("r", 5);

    // Add brush for zooming
    const brush = d3.brushX()
      .extent([[0, 0], [innerWidth, innerHeight]])
      .on("brush end", function(event) {
        if (!event.selection) return;
        
        const [x0, x1] = event.selection;
        const newDomain = [xScale.invert(x0), xScale.invert(x1)];
        
        // Emit zoom event
        const customEvent = new CustomEvent('chartZoomed', {
          detail: { domain: newDomain, chart: 'timeseries' }
        });
        document.dispatchEvent(customEvent);
      });

    g.append("g")
      .attr("class", "brush")
      .call(brush);

    this.charts.set(containerId, {
      svg,
      xScale,
      yScale,
      data,
      config: finalConfig
    });

    return this;
  }

  /**
   * Creates a feature importance bar chart
   */
  createFeatureImportanceChart(containerId, data, config = {}) {
    const finalConfig = { ...this.defaultConfig, ...config };
    const container = d3.select(`#${containerId}`);
    
    container.selectAll("*").remove();
    
    const { margin, width, height, colors } = finalConfig;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = container
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("class", "feature-importance-chart");

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Sort data by importance
    data.sort((a, b) => b.importance - a.importance);

    // Create scales
    const xScale = d3.scaleBand()
      .domain(data.map(d => d.feature))
      .range([0, innerWidth])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.importance)])
      .range([innerHeight, 0])
      .nice();

    // Create color scale based on importance
    const colorScale = d3.scaleSequential()
      .domain([0, d3.max(data, d => d.importance)])
      .interpolator(d3.interpolateBlues);

    // Create axes
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);

    g.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll("text")
      .style("text-anchor", "end")
      .attr("dx", "-.8em")
      .attr("dy", ".15em")
      .attr("transform", "rotate(-45)");

    g.append("g")
      .attr("class", "y-axis")
      .call(yAxis);

    // Add bars
    g.selectAll(".bar")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", d => xScale(d.feature))
      .attr("width", xScale.bandwidth())
      .attr("y", innerHeight)
      .attr("height", 0)
      .attr("fill", d => colorScale(d.importance))
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr("opacity", 0.8);
      })
      .on("mouseout", function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr("opacity", 1);
      })
      .transition()
      .duration(finalConfig.animations.duration)
      .delay((d, i) => i * 50)
      .attr("y", d => yScale(d.importance))
      .attr("height", d => innerHeight - yScale(d.importance));

    this.charts.set(containerId, {
      svg,
      xScale,
      yScale,
      data,
      config: finalConfig
    });

    return this;
  }

  /**
   * Creates a correlation heatmap
   */
  createCorrelationHeatmap(containerId, data, config = {}) {
    const finalConfig = { ...this.defaultConfig, ...config };
    const container = d3.select(`#${containerId}`);
    
    container.selectAll("*").remove();
    
    const { margin, width, height } = finalConfig;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = container
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("class", "correlation-heatmap");

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const features = data.features;
    const matrix = data.correlations;

    // Create scales
    const xScale = d3.scaleBand()
      .domain(features)
      .range([0, innerWidth])
      .padding(0.01);

    const yScale = d3.scaleBand()
      .domain(features)
      .range([0, innerHeight])
      .padding(0.01);

    const colorScale = d3.scaleSequential()
      .domain([-1, 1])
      .interpolator(d3.interpolateRdBu);

    // Create cells
    const cells = g.selectAll(".cell")
      .data(matrix.flat())
      .enter()
      .append("rect")
      .attr("class", "cell")
      .attr("x", d => xScale(d.x))
      .attr("y", d => yScale(d.y))
      .attr("width", xScale.bandwidth())
      .attr("height", yScale.bandwidth())
      .attr("fill", d => colorScale(d.value))
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {
        // Highlight row and column
        g.selectAll(".cell")
          .attr("opacity", 0.3);
        
        g.selectAll(".cell")
          .filter(cell => cell.x === d.x || cell.y === d.y)
          .attr("opacity", 1);
      })
      .on("mouseout", function() {
        g.selectAll(".cell")
          .attr("opacity", 1);
      });

    // Add axes
    g.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll("text")
      .style("text-anchor", "end")
      .attr("dx", "-.8em")
      .attr("dy", ".15em")
      .attr("transform", "rotate(-45)");

    g.append("g")
      .attr("class", "y-axis")
      .call(d3.axisLeft(yScale));

    this.charts.set(containerId, {
      svg,
      xScale,
      yScale,
      data,
      config: finalConfig
    });

    return this;
  }

  /**
   * Updates chart data with animation
   */
  updateChart(containerId, newData) {
    const chart = this.charts.get(containerId);
    if (!chart) return;

    // Implementation depends on chart type
    // This is a placeholder for dynamic updates
    console.log(`Updating chart ${containerId} with new data`, newData);
  }

  /**
   * Exports chart as image
   */
  exportChart(containerId, format = 'png') {
    const chart = this.charts.get(containerId);
    if (!chart) return null;

    const svgElement = chart.svg.node();
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svgElement);
    
    if (format === 'svg') {
      return svgString;
    }

    // For PNG/JPEG, create canvas and convert
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    return new Promise((resolve) => {
      img.onload = function() {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        resolve(canvas.toDataURL(`image/${format}`));
      };
      
      img.src = 'data:image/svg+xml;base64,' + btoa(svgString);
    });
  }

  /**
   * Destroys chart and cleans up resources
   */
  destroyChart(containerId) {
    const chart = this.charts.get(containerId);
    if (chart) {
      chart.svg.remove();
      this.charts.delete(containerId);
    }
  }
}

// Export for use in other modules
window.D3ChartLibrary = D3ChartLibrary;

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.d3Charts = new D3ChartLibrary();
  });
} else {
  window.d3Charts = new D3ChartLibrary();
}