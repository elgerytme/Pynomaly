/**
 * Advanced Anomaly Timeline Component
 * 
 * Interactive timeline visualization for anomaly detection events with D3.js
 * Features real-time updates, zooming, filtering, and detailed event inspection
 */

import * as d3 from 'd3';

export class AnomalyTimeline {
    constructor(container, options = {}) {
        this.container = d3.select(container);
        this.options = {
            width: 800,
            height: 400,
            margin: { top: 20, right: 30, bottom: 40, left: 50 },
            animationDuration: 500,
            showTooltip: true,
            enableZoom: true,
            enableBrush: true,
            colorScheme: 'category10',
            severityColors: {
                low: '#28a745',
                medium: '#ffc107', 
                high: '#fd7e14',
                critical: '#dc3545'
            },
            ...options
        };
        
        this.data = [];
        this.filteredData = [];
        this.selectedTimeRange = null;
        this.brushSelection = null;
        
        this.init();
    }
    
    init() {
        this.setupDimensions();
        this.createSVG();
        this.createScales();
        this.createAxes();
        this.createTooltip();
        this.createBrush();
        this.createZoom();
        this.setupEventListeners();
    }
    
    setupDimensions() {
        this.width = this.options.width - this.options.margin.left - this.options.margin.right;
        this.height = this.options.height - this.options.margin.top - this.options.margin.bottom;
    }
    
    createSVG() {
        // Clear existing content
        this.container.selectAll('*').remove();
        
        this.svg = this.container
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height)
            .attr('class', 'anomaly-timeline');
            
        this.g = this.svg
            .append('g')
            .attr('transform', `translate(${this.options.margin.left},${this.options.margin.top})`);
            
        // Create clipping path for chart area
        this.svg.append('defs')
            .append('clipPath')
            .attr('id', 'timeline-clip')
            .append('rect')
            .attr('width', this.width)
            .attr('height', this.height);
            
        this.chartArea = this.g.append('g')
            .attr('clip-path', 'url(#timeline-clip)');
    }
    
    createScales() {
        this.xScale = d3.scaleTime()
            .range([0, this.width]);
            
        this.yScale = d3.scaleLinear()
            .range([this.height, 0]);
            
        this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        
        this.sizeScale = d3.scaleSqrt()
            .range([3, 15]);
    }
    
    createAxes() {
        this.xAxis = d3.axisBottom(this.xScale)
            .tickFormat(d3.timeFormat('%H:%M'));
            
        this.yAxis = d3.axisLeft(this.yScale)
            .tickFormat(d3.format('.2f'));
            
        this.g.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${this.height})`);
            
        this.g.append('g')
            .attr('class', 'y-axis');
            
        // Add axis labels
        this.g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - this.options.margin.left)
            .attr('x', 0 - (this.height / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .text('Anomaly Score');
            
        this.g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', `translate(${this.width / 2}, ${this.height + this.options.margin.bottom})`)
            .style('text-anchor', 'middle')
            .text('Time');
    }
    
    createTooltip() {
        if (!this.options.showTooltip) return;
        
        this.tooltip = d3.select('body')
            .append('div')
            .attr('class', 'anomaly-timeline-tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('background', 'rgba(0, 0, 0, 0.8)')
            .style('color', 'white')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('pointer-events', 'none')
            .style('font-size', '12px')
            .style('z-index', '1000');
    }
    
    createBrush() {
        if (!this.options.enableBrush) return;
        
        this.brush = d3.brushX()
            .extent([[0, 0], [this.width, this.height]])
            .on('start brush end', this.onBrush.bind(this));
            
        this.brushGroup = this.g.append('g')
            .attr('class', 'brush')
            .call(this.brush);
    }
    
    createZoom() {
        if (!this.options.enableZoom) return;
        
        this.zoom = d3.zoom()
            .scaleExtent([1, 10])
            .translateExtent([[0, 0], [this.width, this.height]])
            .on('zoom', this.onZoom.bind(this));
            
        this.svg.call(this.zoom);
    }
    
    setupEventListeners() {
        // Resize listener
        window.addEventListener('resize', this.debounce(this.resize.bind(this), 250));
    }
    
    setData(data) {
        this.data = data.map(d => ({
            ...d,
            timestamp: new Date(d.timestamp),
            score: +d.score,
            severity: d.severity || this.getSeverityFromScore(d.score)
        }));
        
        this.filteredData = [...this.data];
        this.updateScales();
        this.render();
    }
    
    getSeverityFromScore(score) {
        if (score >= 0.8) return 'critical';
        if (score >= 0.6) return 'high';
        if (score >= 0.4) return 'medium';
        return 'low';
    }
    
    updateScales() {
        if (this.filteredData.length === 0) return;
        
        const timeExtent = d3.extent(this.filteredData, d => d.timestamp);
        const scoreExtent = d3.extent(this.filteredData, d => d.score);
        
        this.xScale.domain(timeExtent);
        this.yScale.domain([0, Math.max(1, scoreExtent[1])]);
        this.sizeScale.domain([0, scoreExtent[1]]);
    }
    
    render() {
        this.renderAxes();
        this.renderAnomalies();
        this.renderTrend();
        this.renderLegend();
    }
    
    renderAxes() {
        this.g.select('.x-axis')
            .transition()
            .duration(this.options.animationDuration)
            .call(this.xAxis);
            
        this.g.select('.y-axis')
            .transition()
            .duration(this.options.animationDuration)
            .call(this.yAxis);
    }
    
    renderAnomalies() {
        const circles = this.chartArea
            .selectAll('.anomaly-point')
            .data(this.filteredData, d => d.id || d.timestamp);
            
        // Enter
        const enter = circles.enter()
            .append('circle')
            .attr('class', 'anomaly-point')
            .attr('cx', d => this.xScale(d.timestamp))
            .attr('cy', this.height)
            .attr('r', 0)
            .style('fill', d => this.options.severityColors[d.severity])
            .style('opacity', 0.7)
            .style('cursor', 'pointer');
            
        // Update
        enter.merge(circles)
            .transition()
            .duration(this.options.animationDuration)
            .attr('cx', d => this.xScale(d.timestamp))
            .attr('cy', d => this.yScale(d.score))
            .attr('r', d => this.sizeScale(d.score))
            .style('fill', d => this.options.severityColors[d.severity]);
            
        // Exit
        circles.exit()
            .transition()
            .duration(this.options.animationDuration)
            .attr('r', 0)
            .style('opacity', 0)
            .remove();
            
        // Add event listeners
        this.chartArea.selectAll('.anomaly-point')
            .on('mouseover', this.showTooltip.bind(this))
            .on('mouseout', this.hideTooltip.bind(this))
            .on('click', this.onAnomalyClick.bind(this));
    }
    
    renderTrend() {
        if (this.filteredData.length < 2) return;
        
        const line = d3.line()
            .x(d => this.xScale(d.timestamp))
            .y(d => this.yScale(d.movingAverage || d.score))
            .curve(d3.curveMonotoneX);
            
        // Calculate moving average
        const windowSize = Math.max(3, Math.floor(this.filteredData.length / 10));
        const dataWithMA = this.calculateMovingAverage(this.filteredData, windowSize);
        
        const trendPath = this.chartArea
            .selectAll('.trend-line')
            .data([dataWithMA]);
            
        trendPath.enter()
            .append('path')
            .attr('class', 'trend-line')
            .style('fill', 'none')
            .style('stroke', '#666')
            .style('stroke-width', 2)
            .style('opacity', 0.5)
            .merge(trendPath)
            .transition()
            .duration(this.options.animationDuration)
            .attr('d', line);
            
        trendPath.exit().remove();
    }
    
    renderLegend() {
        const legend = this.g.selectAll('.legend')
            .data([null]);
            
        const legendEnter = legend.enter()
            .append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${this.width - 120}, 20)`);
            
        const severities = Object.keys(this.options.severityColors);
        const legendItems = legendEnter.selectAll('.legend-item')
            .data(severities);
            
        const legendItem = legendItems.enter()
            .append('g')
            .attr('class', 'legend-item')
            .attr('transform', (d, i) => `translate(0, ${i * 20})`);
            
        legendItem.append('circle')
            .attr('r', 6)
            .style('fill', d => this.options.severityColors[d]);
            
        legendItem.append('text')
            .attr('x', 12)
            .attr('y', 5)
            .style('font-size', '12px')
            .text(d => d.charAt(0).toUpperCase() + d.slice(1));
    }
    
    calculateMovingAverage(data, windowSize) {
        return data.map((item, index) => {
            const start = Math.max(0, index - windowSize + 1);
            const window = data.slice(start, index + 1);
            const average = window.reduce((sum, d) => sum + d.score, 0) / window.length;
            return { ...item, movingAverage: average };
        });
    }
    
    showTooltip(event, d) {
        if (!this.tooltip) return;
        
        this.tooltip.transition()
            .duration(200)
            .style('opacity', 0.9);
            
        this.tooltip.html(`
            <strong>Anomaly Details</strong><br/>
            Time: ${d3.timeFormat('%Y-%m-%d %H:%M:%S')(d.timestamp)}<br/>
            Score: ${d.score.toFixed(3)}<br/>
            Severity: ${d.severity}<br/>
            ${d.description ? `Description: ${d.description}<br/>` : ''}
            ${d.features ? `Features: ${d.features.join(', ')}<br/>` : ''}
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 28) + 'px');
    }
    
    hideTooltip() {
        if (!this.tooltip) return;
        
        this.tooltip.transition()
            .duration(500)
            .style('opacity', 0);
    }
    
    onAnomalyClick(event, d) {
        // Emit custom event for anomaly selection
        this.container.node().dispatchEvent(new CustomEvent('anomalySelected', {
            detail: { anomaly: d, timeline: this }
        }));
    }
    
    onBrush(event) {
        if (!event.selection) {
            this.brushSelection = null;
            this.filteredData = [...this.data];
        } else {
            this.brushSelection = event.selection.map(this.xScale.invert);
            this.filteredData = this.data.filter(d => 
                d.timestamp >= this.brushSelection[0] && d.timestamp <= this.brushSelection[1]
            );
        }
        
        this.updateScales();
        this.renderAnomalies();
        this.renderTrend();
        
        // Emit filter event
        this.container.node().dispatchEvent(new CustomEvent('timeRangeFiltered', {
            detail: { range: this.brushSelection, data: this.filteredData }
        }));
    }
    
    onZoom(event) {
        const newXScale = event.transform.rescaleX(this.xScale);
        
        this.g.select('.x-axis').call(this.xAxis.scale(newXScale));
        
        this.chartArea.selectAll('.anomaly-point')
            .attr('cx', d => newXScale(d.timestamp));
            
        this.chartArea.selectAll('.trend-line')
            .attr('d', d3.line()
                .x(d => newXScale(d.timestamp))
                .y(d => this.yScale(d.movingAverage || d.score))
                .curve(d3.curveMonotoneX)
            );
    }
    
    resize() {
        const containerNode = this.container.node();
        const newWidth = containerNode.getBoundingClientRect().width;
        
        if (newWidth !== this.options.width) {
            this.options.width = newWidth;
            this.setupDimensions();
            this.createSVG();
            this.createScales();
            this.createAxes();
            this.updateScales();
            this.render();
        }
    }
    
    // Public methods for external control
    filterByTimeRange(startTime, endTime) {
        this.filteredData = this.data.filter(d => 
            d.timestamp >= startTime && d.timestamp <= endTime
        );
        this.updateScales();
        this.render();
    }
    
    filterBySeverity(severities) {
        this.filteredData = this.data.filter(d => 
            severities.includes(d.severity)
        );
        this.updateScales();
        this.render();
    }
    
    highlightAnomalies(anomalyIds) {
        this.chartArea.selectAll('.anomaly-point')
            .style('stroke', d => anomalyIds.includes(d.id) ? '#000' : 'none')
            .style('stroke-width', d => anomalyIds.includes(d.id) ? 2 : 0);
    }
    
    addRealTimeData(newData) {
        const processedData = newData.map(d => ({
            ...d,
            timestamp: new Date(d.timestamp),
            score: +d.score,
            severity: d.severity || this.getSeverityFromScore(d.score)
        }));
        
        this.data.push(...processedData);
        
        // Keep only recent data if too many points
        const maxPoints = 1000;
        if (this.data.length > maxPoints) {
            this.data = this.data.slice(-maxPoints);
        }
        
        this.filteredData = [...this.data];
        this.updateScales();
        this.render();
    }
    
    exportData() {
        return {
            data: this.data,
            filteredData: this.filteredData,
            timeRange: this.brushSelection,
            config: this.options
        };
    }
    
    // Utility methods
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
        if (this.tooltip) {
            this.tooltip.remove();
        }
        window.removeEventListener('resize', this.resize);
        this.container.selectAll('*').remove();
    }
}

// CSS styles (to be included in stylesheet)
export const timelineStyles = `
.anomaly-timeline {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.anomaly-timeline .axis-label {
    font-size: 12px;
    font-weight: 500;
}

.anomaly-timeline .legend-item text {
    font-size: 11px;
    fill: #666;
}

.anomaly-timeline .anomaly-point {
    transition: all 0.2s ease;
}

.anomaly-timeline .anomaly-point:hover {
    stroke: #000;
    stroke-width: 2px;
    transform: scale(1.1);
}

.anomaly-timeline .trend-line {
    filter: drop-shadow(0 0 2px rgba(102, 102, 102, 0.3));
}

.anomaly-timeline .brush .selection {
    fill: rgba(59, 130, 246, 0.3);
    stroke: #3b82f6;
}

.anomaly-timeline-tooltip {
    max-width: 300px;
    word-wrap: break-word;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
`;