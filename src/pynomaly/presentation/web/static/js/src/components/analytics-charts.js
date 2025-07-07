/**
 * Advanced Analytics Charts Component
 * 
 * Interactive data exploration and analysis charts using D3.js and ECharts
 * for comprehensive anomaly detection analytics and insights
 */

import * as d3 from 'd3';

export class AnalyticsCharts {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        this.options = {
            width: 800,
            height: 400,
            theme: 'light',
            enableInteraction: true,
            enableZoom: true,
            enableBrush: true,
            enableLegend: true,
            enableTooltip: true,
            animationDuration: 750,
            margin: { top: 20, right: 80, bottom: 40, left: 60 },
            ...options
        };
        
        this.data = [];
        this.charts = new Map();
        this.scales = {};
        this.brushes = new Map();
        this.zoom = null;
        
        this.init();
    }
    
    init() {
        this.setupContainer();
        this.setupScales();
        this.bindEvents();
    }
    
    setupContainer() {
        this.container.classList.add('analytics-charts');
        this.container.innerHTML = '';
        
        // Create SVG
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height)
            .style('background', this.options.theme === 'dark' ? '#1a1a1a' : '#ffffff');
        
        // Create main group
        this.mainGroup = this.svg.append('g')
            .attr('transform', `translate(${this.options.margin.left}, ${this.options.margin.top})`);
        
        // Calculate dimensions
        this.innerWidth = this.options.width - this.options.margin.left - this.options.margin.right;
        this.innerHeight = this.options.height - this.options.margin.top - this.options.margin.bottom;
        
        // Create chart groups
        this.chartArea = this.mainGroup.append('g').attr('class', 'chart-area');
        this.axesGroup = this.mainGroup.append('g').attr('class', 'axes');
        this.legendGroup = this.mainGroup.append('g').attr('class', 'legend');
        this.tooltipGroup = this.mainGroup.append('g').attr('class', 'tooltip-group');
        
        // Create tooltip
        this.tooltip = d3.select('body').append('div')
            .attr('class', 'chart-tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('background', 'rgba(0, 0, 0, 0.8)')
            .style('color', 'white')
            .style('padding', '8px 12px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('pointer-events', 'none')
            .style('z-index', 1000);
    }
    
    setupScales() {
        this.scales.x = d3.scaleTime().range([0, this.innerWidth]);
        this.scales.y = d3.scaleLinear().range([this.innerHeight, 0]);
        this.scales.color = d3.scaleOrdinal(d3.schemeCategory10);
        this.scales.size = d3.scaleSqrt().range([2, 10]);
    }
    
    bindEvents() {
        // Setup zoom
        if (this.options.enableZoom) {
            this.zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {
                    this.onZoom(event);
                });
            
            this.svg.call(this.zoom);
        }
        
        // Setup brush
        if (this.options.enableBrush) {
            this.brush = d3.brushX()
                .extent([[0, 0], [this.innerWidth, this.innerHeight]])
                .on('brush end', (event) => {
                    this.onBrush(event);
                });
        }
    }
    
    // Chart creation methods
    createScatterPlot(data, config = {}) {
        const chartConfig = {
            xField: 'x',
            yField: 'y',
            colorField: 'category',
            sizeField: 'value',
            showTrendline: false,
            enableClustering: false,
            ...config
        };
        
        this.clearChart();
        this.data = data;
        
        // Update scales
        this.scales.x.domain(d3.extent(data, d => d[chartConfig.xField]));
        this.scales.y.domain(d3.extent(data, d => d[chartConfig.yField]));
        
        if (chartConfig.sizeField) {
            this.scales.size.domain(d3.extent(data, d => d[chartConfig.sizeField]));
        }
        
        // Draw axes
        this.drawAxes();
        
        // Draw points
        const circles = this.chartArea.selectAll('.data-point')
            .data(data)
            .enter()
            .append('circle')
            .attr('class', 'data-point')
            .attr('cx', d => this.scales.x(d[chartConfig.xField]))
            .attr('cy', d => this.scales.y(d[chartConfig.yField]))
            .attr('r', d => chartConfig.sizeField ? this.scales.size(d[chartConfig.sizeField]) : 4)
            .attr('fill', d => this.scales.color(d[chartConfig.colorField] || 'default'))
            .attr('opacity', 0.7)
            .style('cursor', 'pointer');
        
        // Add interactions
        if (this.options.enableTooltip) {
            circles
                .on('mouseover', (event, d) => {
                    this.showTooltip(event, d, chartConfig);
                })
                .on('mouseout', () => {
                    this.hideTooltip();
                });
        }
        
        // Add trendline if requested
        if (chartConfig.showTrendline) {
            this.addTrendline(data, chartConfig);
        }
        
        // Add legend
        if (this.options.enableLegend && chartConfig.colorField) {
            this.drawLegend(data, chartConfig.colorField);
        }
        
        this.charts.set('scatter', { data, config: chartConfig });
        return this;
    }
    
    createTimeSeries(data, config = {}) {
        const chartConfig = {
            timeField: 'timestamp',
            valueField: 'value',
            categoryField: null,
            lineType: 'line', // line, area, step
            showPoints: false,
            showConfidenceBand: false,
            aggregation: null, // sum, avg, max, min
            ...config
        };
        
        this.clearChart();
        this.data = data;
        
        // Process data for time series
        const processedData = this.processTimeSeriesData(data, chartConfig);
        
        // Update scales
        this.scales.x.domain(d3.extent(processedData, d => d.time));
        this.scales.y.domain(d3.extent(processedData, d => d.value));
        
        // Draw axes
        this.drawAxes(true); // time axis
        
        // Create line generator
        const line = d3.line()
            .x(d => this.scales.x(d.time))
            .y(d => this.scales.y(d.value))
            .curve(chartConfig.lineType === 'step' ? d3.curveStepAfter : d3.curveMonotoneX);
        
        // Group by category if specified
        const dataGroups = chartConfig.categoryField
            ? d3.group(processedData, d => d[chartConfig.categoryField])
            : new Map([['default', processedData]]);
        
        // Draw lines
        dataGroups.forEach((groupData, category) => {
            const path = this.chartArea.append('path')
                .datum(groupData)
                .attr('class', `time-series-line category-${category}`)
                .attr('fill', chartConfig.lineType === 'area' ? this.scales.color(category) : 'none')
                .attr('stroke', this.scales.color(category))
                .attr('stroke-width', 2)
                .attr('opacity', chartConfig.lineType === 'area' ? 0.6 : 0.8)
                .attr('d', chartConfig.lineType === 'area' ? 
                    d3.area()
                        .x(d => this.scales.x(d.time))
                        .y0(this.scales.y(0))
                        .y1(d => this.scales.y(d.value))
                        .curve(d3.curveMonotoneX)
                    : line);
            
            // Animate path drawing
            const totalLength = path.node().getTotalLength();
            path
                .attr('stroke-dasharray', totalLength + ' ' + totalLength)
                .attr('stroke-dashoffset', totalLength)
                .transition()
                .duration(this.options.animationDuration)
                .attr('stroke-dashoffset', 0);
            
            // Add points if requested
            if (chartConfig.showPoints) {
                this.chartArea.selectAll(`.points-${category}`)
                    .data(groupData)
                    .enter()
                    .append('circle')
                    .attr('class', `data-point points-${category}`)
                    .attr('cx', d => this.scales.x(d.time))
                    .attr('cy', d => this.scales.y(d.value))
                    .attr('r', 3)
                    .attr('fill', this.scales.color(category))
                    .style('cursor', 'pointer')
                    .on('mouseover', (event, d) => {
                        this.showTooltip(event, d, chartConfig);
                    })
                    .on('mouseout', () => {
                        this.hideTooltip();
                    });
            }
        });
        
        // Add confidence band if requested
        if (chartConfig.showConfidenceBand) {
            this.addConfidenceBand(processedData, chartConfig);
        }
        
        // Add brush for time selection
        if (this.options.enableBrush) {
            this.chartArea.append('g')
                .attr('class', 'brush')
                .call(this.brush);
        }
        
        // Add legend
        if (this.options.enableLegend && chartConfig.categoryField) {
            this.drawLegend(Array.from(dataGroups.keys()).map(key => ({ [chartConfig.categoryField]: key })), chartConfig.categoryField);
        }
        
        this.charts.set('timeSeries', { data: processedData, config: chartConfig });
        return this;
    }
    
    createHeatmap(data, config = {}) {
        const chartConfig = {
            xField: 'x',
            yField: 'y',
            valueField: 'value',
            colorScheme: 'interpolateViridis',
            showValues: false,
            cellPadding: 1,
            ...config
        };
        
        this.clearChart();
        this.data = data;
        
        // Get unique x and y values
        const xValues = [...new Set(data.map(d => d[chartConfig.xField]))].sort();
        const yValues = [...new Set(data.map(d => d[chartConfig.yField]))].sort();
        
        // Update scales
        this.scales.x = d3.scaleBand().domain(xValues).range([0, this.innerWidth]).padding(0.1);
        this.scales.y = d3.scaleBand().domain(yValues).range([0, this.innerHeight]).padding(0.1);
        
        // Color scale
        const colorScale = d3.scaleSequential(d3[chartConfig.colorScheme])
            .domain(d3.extent(data, d => d[chartConfig.valueField]));
        
        // Draw axes
        this.drawAxes(false, true); // categorical axes
        
        // Draw heatmap cells
        const cells = this.chartArea.selectAll('.heatmap-cell')
            .data(data)
            .enter()
            .append('rect')
            .attr('class', 'heatmap-cell')
            .attr('x', d => this.scales.x(d[chartConfig.xField]))
            .attr('y', d => this.scales.y(d[chartConfig.yField]))
            .attr('width', this.scales.x.bandwidth())
            .attr('height', this.scales.y.bandwidth())
            .attr('fill', d => colorScale(d[chartConfig.valueField]))
            .attr('opacity', 0)
            .style('cursor', 'pointer');
        
        // Animate cells
        cells.transition()
            .duration(this.options.animationDuration)
            .delay((d, i) => i * 10)
            .attr('opacity', 0.8);
        
        // Add value labels if requested
        if (chartConfig.showValues) {
            this.chartArea.selectAll('.cell-label')
                .data(data)
                .enter()
                .append('text')
                .attr('class', 'cell-label')
                .attr('x', d => this.scales.x(d[chartConfig.xField]) + this.scales.x.bandwidth() / 2)
                .attr('y', d => this.scales.y(d[chartConfig.yField]) + this.scales.y.bandwidth() / 2)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'central')
                .attr('fill', d => colorScale(d[chartConfig.valueField]) > 0.5 ? 'white' : 'black')
                .attr('font-size', '10px')
                .text(d => d[chartConfig.valueField].toFixed(2));
        }
        
        // Add interactions
        if (this.options.enableTooltip) {
            cells
                .on('mouseover', (event, d) => {
                    this.showTooltip(event, d, chartConfig);
                })
                .on('mouseout', () => {
                    this.hideTooltip();
                });
        }
        
        // Add color legend
        this.drawColorLegend(colorScale, chartConfig.valueField);
        
        this.charts.set('heatmap', { data, config: chartConfig });
        return this;
    }
    
    createHistogram(data, config = {}) {
        const chartConfig = {
            valueField: 'value',
            bins: 20,
            showDensity: false,
            showStats: true,
            overlayDistribution: null, // normal, exponential, etc.
            ...config
        };
        
        this.clearChart();
        this.data = data;
        
        const values = data.map(d => d[chartConfig.valueField]);
        
        // Create histogram
        const histogram = d3.histogram()
            .domain(d3.extent(values))
            .thresholds(chartConfig.bins);
        
        const bins = histogram(values);
        
        // Update scales
        this.scales.x.domain(d3.extent(values));
        this.scales.y.domain([0, d3.max(bins, d => d.length)]);
        
        // Draw axes
        this.drawAxes();
        
        // Draw bars
        const bars = this.chartArea.selectAll('.histogram-bar')
            .data(bins)
            .enter()
            .append('rect')
            .attr('class', 'histogram-bar')
            .attr('x', d => this.scales.x(d.x0))
            .attr('y', this.innerHeight)
            .attr('width', d => Math.max(0, this.scales.x(d.x1) - this.scales.x(d.x0) - 1))
            .attr('height', 0)
            .attr('fill', this.scales.color('histogram'))
            .attr('opacity', 0.7)
            .style('cursor', 'pointer');
        
        // Animate bars
        bars.transition()
            .duration(this.options.animationDuration)
            .attr('y', d => this.scales.y(d.length))
            .attr('height', d => this.innerHeight - this.scales.y(d.length));
        
        // Add interactions
        if (this.options.enableTooltip) {
            bars
                .on('mouseover', (event, d) => {
                    this.showTooltip(event, {
                        range: `${d.x0.toFixed(2)} - ${d.x1.toFixed(2)}`,
                        count: d.length,
                        percentage: ((d.length / values.length) * 100).toFixed(1) + '%'
                    }, { type: 'histogram' });
                })
                .on('mouseout', () => {
                    this.hideTooltip();
                });
        }
        
        // Add statistics
        if (chartConfig.showStats) {
            this.addStatistics(values);
        }
        
        // Add overlay distribution
        if (chartConfig.overlayDistribution) {
            this.addDistributionOverlay(values, chartConfig.overlayDistribution);
        }
        
        this.charts.set('histogram', { data: bins, config: chartConfig });
        return this;
    }
    
    // Utility methods
    processTimeSeriesData(data, config) {
        return data.map(d => ({
            time: new Date(d[config.timeField]),
            value: +d[config.valueField],
            ...d
        })).sort((a, b) => a.time - b.time);
    }
    
    drawAxes(isTimeAxis = false, isCategorical = false) {
        this.axesGroup.selectAll('*').remove();
        
        // X axis
        const xAxis = isCategorical
            ? d3.axisBottom(this.scales.x)
            : isTimeAxis
                ? d3.axisBottom(this.scales.x).tickFormat(d3.timeFormat('%H:%M'))
                : d3.axisBottom(this.scales.x);
        
        this.axesGroup.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0, ${this.innerHeight})`)
            .call(xAxis);
        
        // Y axis
        const yAxis = isCategorical
            ? d3.axisLeft(this.scales.y)
            : d3.axisLeft(this.scales.y);
        
        this.axesGroup.append('g')
            .attr('class', 'y-axis')
            .call(yAxis);
    }
    
    drawLegend(data, field) {
        this.legendGroup.selectAll('*').remove();
        
        const categories = [...new Set(data.map(d => d[field]))];
        const legendItems = this.legendGroup.selectAll('.legend-item')
            .data(categories)
            .enter()
            .append('g')
            .attr('class', 'legend-item')
            .attr('transform', (d, i) => `translate(${this.innerWidth + 10}, ${i * 20})`);
        
        legendItems.append('rect')
            .attr('width', 12)
            .attr('height', 12)
            .attr('fill', d => this.scales.color(d));
        
        legendItems.append('text')
            .attr('x', 16)
            .attr('y', 9)
            .attr('font-size', '12px')
            .text(d => d);
    }
    
    drawColorLegend(colorScale, label) {
        const legendWidth = 200;
        const legendHeight = 20;
        
        const legend = this.legendGroup.append('g')
            .attr('class', 'color-legend')
            .attr('transform', `translate(${(this.innerWidth - legendWidth) / 2}, ${this.innerHeight + 40})`);
        
        // Create gradient
        const defs = this.svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'legend-gradient');
        
        const domain = colorScale.domain();
        const steps = 10;
        for (let i = 0; i <= steps; i++) {
            const value = domain[0] + (domain[1] - domain[0]) * i / steps;
            gradient.append('stop')
                .attr('offset', `${(i / steps) * 100}%`)
                .attr('stop-color', colorScale(value));
        }
        
        // Draw legend rectangle
        legend.append('rect')
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .attr('fill', 'url(#legend-gradient)');
        
        // Add scale
        const legendScale = d3.scaleLinear()
            .domain(domain)
            .range([0, legendWidth]);
        
        legend.append('g')
            .attr('transform', `translate(0, ${legendHeight})`)
            .call(d3.axisBottom(legendScale).ticks(5));
        
        // Add label
        legend.append('text')
            .attr('x', legendWidth / 2)
            .attr('y', -5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .text(label);
    }
    
    addTrendline(data, config) {
        const xValues = data.map(d => d[config.xField]);
        const yValues = data.map(d => d[config.yField]);
        
        // Calculate linear regression
        const n = data.length;
        const sumX = d3.sum(xValues);
        const sumY = d3.sum(yValues);
        const sumXY = d3.sum(data, d => d[config.xField] * d[config.yField]);
        const sumXX = d3.sum(xValues, d => d * d);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        // Draw trendline
        const xExtent = d3.extent(xValues);
        const trendData = [
            { x: xExtent[0], y: slope * xExtent[0] + intercept },
            { x: xExtent[1], y: slope * xExtent[1] + intercept }
        ];
        
        const line = d3.line()
            .x(d => this.scales.x(d.x))
            .y(d => this.scales.y(d.y));
        
        this.chartArea.append('path')
            .datum(trendData)
            .attr('class', 'trendline')
            .attr('fill', 'none')
            .attr('stroke', 'red')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '5,5')
            .attr('d', line);
    }
    
    addConfidenceBand(data, config) {
        // Simple confidence band implementation
        const windowSize = Math.floor(data.length / 10);
        const confidenceData = [];
        
        for (let i = windowSize; i < data.length - windowSize; i++) {
            const window = data.slice(i - windowSize, i + windowSize);
            const values = window.map(d => d.value);
            const mean = d3.mean(values);
            const std = d3.deviation(values);
            
            confidenceData.push({
                time: data[i].time,
                lower: mean - 1.96 * std,
                upper: mean + 1.96 * std
            });
        }
        
        const area = d3.area()
            .x(d => this.scales.x(d.time))
            .y0(d => this.scales.y(d.lower))
            .y1(d => this.scales.y(d.upper))
            .curve(d3.curveMonotoneX);
        
        this.chartArea.append('path')
            .datum(confidenceData)
            .attr('class', 'confidence-band')
            .attr('fill', 'steelblue')
            .attr('opacity', 0.2)
            .attr('d', area);
    }
    
    addStatistics(values) {
        const stats = {
            mean: d3.mean(values),
            median: d3.median(values),
            std: d3.deviation(values),
            min: d3.min(values),
            max: d3.max(values)
        };
        
        // Add vertical lines for mean and median
        this.chartArea.append('line')
            .attr('class', 'stat-line mean')
            .attr('x1', this.scales.x(stats.mean))
            .attr('x2', this.scales.x(stats.mean))
            .attr('y1', 0)
            .attr('y2', this.innerHeight)
            .attr('stroke', 'red')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '3,3');
        
        this.chartArea.append('line')
            .attr('class', 'stat-line median')
            .attr('x1', this.scales.x(stats.median))
            .attr('x2', this.scales.x(stats.median))
            .attr('y1', 0)
            .attr('y2', this.innerHeight)
            .attr('stroke', 'blue')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '3,3');
        
        // Add statistics text
        const statsText = this.chartArea.append('g')
            .attr('class', 'statistics')
            .attr('transform', `translate(10, 20)`);
        
        Object.entries(stats).forEach(([key, value], i) => {
            statsText.append('text')
                .attr('x', 0)
                .attr('y', i * 15)
                .attr('font-size', '12px')
                .attr('fill', this.options.theme === 'dark' ? 'white' : 'black')
                .text(`${key}: ${value.toFixed(3)}`);
        });
    }
    
    addDistributionOverlay(values, distributionType) {
        const mean = d3.mean(values);
        const std = d3.deviation(values);
        
        if (distributionType === 'normal') {
            const normalData = [];
            const xExtent = d3.extent(values);
            const step = (xExtent[1] - xExtent[0]) / 100;
            
            for (let x = xExtent[0]; x <= xExtent[1]; x += step) {
                const y = (1 / (std * Math.sqrt(2 * Math.PI))) * 
                         Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
                normalData.push({ x, y: y * values.length * step });
            }
            
            const line = d3.line()
                .x(d => this.scales.x(d.x))
                .y(d => this.scales.y(d.y))
                .curve(d3.curveMonotoneX);
            
            this.chartArea.append('path')
                .datum(normalData)
                .attr('class', 'distribution-overlay')
                .attr('fill', 'none')
                .attr('stroke', 'orange')
                .attr('stroke-width', 2)
                .attr('d', line);
        }
    }
    
    showTooltip(event, data, config) {
        if (!this.options.enableTooltip) return;
        
        let content = '';
        if (config.type === 'histogram') {
            content = `Range: ${data.range}<br>Count: ${data.count}<br>Percentage: ${data.percentage}`;
        } else {
            content = Object.entries(data)
                .filter(([key, value]) => key !== 'index')
                .map(([key, value]) => `${key}: ${typeof value === 'number' ? value.toFixed(3) : value}`)
                .join('<br>');
        }
        
        this.tooltip
            .style('opacity', 1)
            .html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    }
    
    hideTooltip() {
        this.tooltip.style('opacity', 0);
    }
    
    onZoom(event) {
        const newScaleX = event.transform.rescaleX(this.scales.x);
        
        // Update chart elements based on zoom
        this.chartArea.selectAll('.data-point')
            .attr('cx', d => newScaleX(d.timestamp || d.x || d.time));
        
        this.chartArea.selectAll('.time-series-line')
            .attr('transform', event.transform);
        
        // Update x-axis
        this.axesGroup.select('.x-axis').call(d3.axisBottom(newScaleX));
    }
    
    onBrush(event) {
        if (!event.selection) return;
        
        const [x0, x1] = event.selection.map(this.scales.x.invert);
        
        // Emit brush event
        this.container.dispatchEvent(new CustomEvent('brush', {
            detail: { selection: [x0, x1] }
        }));
    }
    
    clearChart() {
        this.chartArea.selectAll('*').remove();
        this.axesGroup.selectAll('*').remove();
        this.legendGroup.selectAll('*').remove();
    }
    
    resize(width, height) {
        this.options.width = width;
        this.options.height = height;
        
        this.svg
            .attr('width', width)
            .attr('height', height);
        
        this.innerWidth = width - this.options.margin.left - this.options.margin.right;
        this.innerHeight = height - this.options.margin.top - this.options.margin.bottom;
        
        this.scales.x.range([0, this.innerWidth]);
        this.scales.y.range([this.innerHeight, 0]);
        
        // Redraw current chart
        const currentChart = this.charts.values().next().value;
        if (currentChart) {
            // Re-render with current data and config
            this.redraw();
        }
    }
    
    redraw() {
        const currentChart = this.charts.values().next().value;
        if (!currentChart) return;
        
        // Re-render based on chart type
        const chartType = Array.from(this.charts.keys())[0];
        switch (chartType) {
            case 'scatter':
                this.createScatterPlot(currentChart.data, currentChart.config);
                break;
            case 'timeSeries':
                this.createTimeSeries(this.data, currentChart.config);
                break;
            case 'heatmap':
                this.createHeatmap(currentChart.data, currentChart.config);
                break;
            case 'histogram':
                this.createHistogram(this.data, currentChart.config);
                break;
        }
    }
    
    exportChart(format = 'png') {
        const svgElement = this.svg.node();
        
        if (format === 'svg') {
            const serializer = new XMLSerializer();
            return serializer.serializeToString(svgElement);
        }
        
        // For raster formats, we'd need to use canvas conversion
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        const svgData = new XMLSerializer().serializeToString(svgElement);
        const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(svgBlob);
        
        return new Promise((resolve) => {
            img.onload = () => {
                canvas.width = this.options.width;
                canvas.height = this.options.height;
                ctx.drawImage(img, 0, 0);
                URL.revokeObjectURL(url);
                
                if (format === 'png') {
                    resolve(canvas.toDataURL('image/png'));
                } else if (format === 'jpeg') {
                    resolve(canvas.toDataURL('image/jpeg'));
                }
            };
            img.src = url;
        });
    }
    
    destroy() {
        if (this.tooltip) {
            this.tooltip.remove();
        }
        
        this.charts.clear();
        this.brushes.clear();
        
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

export default AnalyticsCharts;
