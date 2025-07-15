/**
 * Advanced Drill-Down Navigation System
 * 
 * Features:
 * - Multi-level data exploration
 * - Breadcrumb navigation
 * - Context-aware filtering
 * - Hierarchical data visualization
 * - Dynamic chart type switching
 * - Interactive filtering and grouping
 * - Time-based drill-down
 * - Anomaly investigation workflows
 */

class AdvancedDrillDownNavigation {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            maxDepth: 5,
            enableBreadcrumbs: true,
            enableFiltering: true,
            enableGrouping: true,
            enableTimeNavigation: true,
            animationDuration: 300,
            ...options
        };
        
        this.navigationStack = [];
        this.currentLevel = 0;
        this.filters = new Map();
        this.groupings = new Map();
        this.timeRange = { start: null, end: null };
        this.originalData = null;
        this.currentData = null;
        this.chartInstance = null;
        
        this.init();
    }

    init() {
        this.setupNavigationUI();
        this.setupEventListeners();
        this.loadInitialData();
    }

    setupNavigationUI() {
        this.container.innerHTML = `
            <div class="drill-down-container">
                <!-- Navigation Header -->
                <div class="navigation-header">
                    <div class="breadcrumb-container">
                        <nav class="breadcrumb" id="breadcrumb-nav">
                            <span class="breadcrumb-item active" data-level="0">
                                <i class="icon-home"></i>
                                Overview
                            </span>
                        </nav>
                    </div>
                    <div class="navigation-controls">
                        <button class="nav-button" id="back-button" disabled>
                            <i class="icon-arrow-left"></i>
                            Back
                        </button>
                        <button class="nav-button" id="forward-button" disabled>
                            <i class="icon-arrow-right"></i>
                            Forward
                        </button>
                        <button class="nav-button" id="home-button">
                            <i class="icon-home"></i>
                            Home
                        </button>
                    </div>
                </div>

                <!-- Filter and Grouping Controls -->
                <div class="navigation-filters">
                    <div class="filter-section">
                        <h4>Filters</h4>
                        <div class="filter-controls" id="filter-controls">
                            <div class="filter-group">
                                <label>Time Range:</label>
                                <select id="time-range-filter">
                                    <option value="all">All Time</option>
                                    <option value="1h">Last Hour</option>
                                    <option value="24h">Last 24 Hours</option>
                                    <option value="7d">Last 7 Days</option>
                                    <option value="30d">Last 30 Days</option>
                                    <option value="custom">Custom Range</option>
                                </select>
                            </div>
                            <div class="filter-group">
                                <label>Anomaly Score:</label>
                                <div class="range-slider">
                                    <input type="range" id="anomaly-min" min="0" max="1" step="0.1" value="0">
                                    <input type="range" id="anomaly-max" min="0" max="1" step="0.1" value="1">
                                </div>
                                <span class="range-values">
                                    <span id="anomaly-min-value">0</span> - 
                                    <span id="anomaly-max-value">1</span>
                                </span>
                            </div>
                            <div class="filter-group">
                                <label>Category:</label>
                                <select id="category-filter" multiple>
                                    <option value="all">All Categories</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <div class="grouping-section">
                        <h4>Grouping</h4>
                        <div class="grouping-controls">
                            <div class="grouping-option">
                                <label>Group By:</label>
                                <select id="group-by-field">
                                    <option value="none">No Grouping</option>
                                    <option value="category">Category</option>
                                    <option value="time">Time Period</option>
                                    <option value="severity">Severity Level</option>
                                    <option value="source">Data Source</option>
                                </select>
                            </div>
                            <div class="grouping-option">
                                <label>Aggregation:</label>
                                <select id="aggregation-method">
                                    <option value="count">Count</option>
                                    <option value="sum">Sum</option>
                                    <option value="avg">Average</option>
                                    <option value="max">Maximum</option>
                                    <option value="min">Minimum</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Main Visualization Area -->
                <div class="visualization-area">
                    <div class="chart-container" id="drill-down-chart">
                        <!-- Chart will be rendered here -->
                    </div>
                    
                    <!-- Context Panel -->
                    <div class="context-panel" id="context-panel">
                        <div class="context-header">
                            <h4>Context Information</h4>
                            <button class="panel-toggle" id="toggle-context">
                                <i class="icon-chevron-right"></i>
                            </button>
                        </div>
                        <div class="context-content">
                            <div class="context-section">
                                <h5>Current Selection</h5>
                                <div class="selection-info" id="selection-info">
                                    <p>No selection</p>
                                </div>
                            </div>
                            <div class="context-section">
                                <h5>Data Summary</h5>
                                <div class="data-summary" id="data-summary">
                                    <div class="summary-item">
                                        <span class="label">Total Points:</span>
                                        <span class="value" id="total-points">0</span>
                                    </div>
                                    <div class="summary-item">
                                        <span class="label">Anomalies:</span>
                                        <span class="value" id="anomaly-count">0</span>
                                    </div>
                                    <div class="summary-item">
                                        <span class="label">Avg Score:</span>
                                        <span class="value" id="avg-score">0.0</span>
                                    </div>
                                </div>
                            </div>
                            <div class="context-section">
                                <h5>Available Actions</h5>
                                <div class="action-buttons" id="action-buttons">
                                    <button class="action-btn" id="drill-down-btn" disabled>
                                        <i class="icon-search"></i>
                                        Drill Down
                                    </button>
                                    <button class="action-btn" id="investigate-btn" disabled>
                                        <i class="icon-target"></i>
                                        Investigate
                                    </button>
                                    <button class="action-btn" id="compare-btn" disabled>
                                        <i class="icon-compare"></i>
                                        Compare
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Detail Modal -->
                <div class="detail-modal" id="detail-modal">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3>Detailed Analysis</h3>
                            <button class="modal-close" id="close-detail-modal">×</button>
                        </div>
                        <div class="modal-body">
                            <div class="detail-tabs">
                                <button class="tab-button active" data-tab="overview">Overview</button>
                                <button class="tab-button" data-tab="timeseries">Time Series</button>
                                <button class="tab-button" data-tab="distribution">Distribution</button>
                                <button class="tab-button" data-tab="correlations">Correlations</button>
                            </div>
                            <div class="detail-content" id="detail-content">
                                <!-- Detail content will be populated here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // Navigation controls
        document.getElementById('back-button').addEventListener('click', () => this.navigateBack());
        document.getElementById('forward-button').addEventListener('click', () => this.navigateForward());
        document.getElementById('home-button').addEventListener('click', () => this.navigateHome());
        
        // Filter controls
        document.getElementById('time-range-filter').addEventListener('change', (e) => this.applyTimeFilter(e.target.value));
        document.getElementById('anomaly-min').addEventListener('input', (e) => this.updateAnomalyFilter());
        document.getElementById('anomaly-max').addEventListener('input', (e) => this.updateAnomalyFilter());
        document.getElementById('category-filter').addEventListener('change', (e) => this.applyCategoryFilter(e.target.value));
        
        // Grouping controls
        document.getElementById('group-by-field').addEventListener('change', (e) => this.applyGrouping(e.target.value));
        document.getElementById('aggregation-method').addEventListener('change', (e) => this.updateAggregation(e.target.value));
        
        // Context panel
        document.getElementById('toggle-context').addEventListener('click', () => this.toggleContextPanel());
        document.getElementById('drill-down-btn').addEventListener('click', () => this.performDrillDown());
        document.getElementById('investigate-btn').addEventListener('click', () => this.investigateSelection());
        document.getElementById('compare-btn').addEventListener('click', () => this.compareSelection());
        
        // Detail modal
        document.getElementById('close-detail-modal').addEventListener('click', () => this.closeDetailModal());
        
        // Breadcrumb navigation
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('breadcrumb-item')) {
                const level = parseInt(e.target.dataset.level);
                this.navigateToLevel(level);
            }
        });
    }

    loadInitialData() {
        // Generate sample hierarchical data
        this.originalData = this.generateSampleData();
        this.currentData = this.originalData;
        this.renderChart();
        this.updateUI();
    }

    generateSampleData() {
        const categories = ['Network', 'Performance', 'Security', 'Application', 'Infrastructure'];
        const sources = ['Server-A', 'Server-B', 'Database', 'API', 'Frontend'];
        const data = [];
        
        const now = new Date();
        
        for (let i = 0; i < 1000; i++) {
            const timestamp = new Date(now.getTime() - Math.random() * 7 * 24 * 60 * 60 * 1000);
            const category = categories[Math.floor(Math.random() * categories.length)];
            const source = sources[Math.floor(Math.random() * sources.length)];
            const isAnomaly = Math.random() > 0.85;
            
            data.push({
                id: i,
                timestamp: timestamp,
                category: category,
                source: source,
                value: Math.random() * 100,
                anomaly_score: isAnomaly ? 0.7 + Math.random() * 0.3 : Math.random() * 0.3,
                is_anomaly: isAnomaly,
                severity: isAnomaly ? ['high', 'medium', 'low'][Math.floor(Math.random() * 3)] : 'normal',
                details: {
                    cpu_usage: Math.random() * 100,
                    memory_usage: Math.random() * 100,
                    disk_usage: Math.random() * 100,
                    network_latency: Math.random() * 1000
                }
            });
        }
        
        return data;
    }

    renderChart() {
        const chartContainer = document.getElementById('drill-down-chart');
        chartContainer.innerHTML = '';
        
        // Create chart based on current navigation level
        const chartType = this.getChartTypeForLevel(this.currentLevel);
        
        if (chartType === 'timeline') {
            this.renderTimelineChart(chartContainer);
        } else if (chartType === 'distribution') {
            this.renderDistributionChart(chartContainer);
        } else if (chartType === 'heatmap') {
            this.renderHeatmapChart(chartContainer);
        } else if (chartType === 'network') {
            this.renderNetworkChart(chartContainer);
        }
        
        this.updateDataSummary();
    }

    getChartTypeForLevel(level) {
        const chartTypes = ['timeline', 'distribution', 'heatmap', 'network', 'detail'];
        return chartTypes[Math.min(level, chartTypes.length - 1)];
    }

    renderTimelineChart(container) {
        if (!window.d3) return;
        
        const width = container.clientWidth;
        const height = 400;
        const margin = { top: 20, right: 30, bottom: 40, left: 50 };
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        const xScale = d3.scaleTime()
            .domain(d3.extent(this.currentData, d => d.timestamp))
            .range([0, width - margin.left - margin.right]);
        
        const yScale = d3.scaleLinear()
            .domain(d3.extent(this.currentData, d => d.anomaly_score))
            .range([height - margin.top - margin.bottom, 0]);
        
        // Add axes
        g.append('g')
            .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
            .call(d3.axisBottom(xScale));
        
        g.append('g')
            .call(d3.axisLeft(yScale));
        
        // Add data points
        const points = g.selectAll('.data-point')
            .data(this.currentData)
            .enter()
            .append('circle')
            .attr('class', 'data-point')
            .attr('cx', d => xScale(d.timestamp))
            .attr('cy', d => yScale(d.anomaly_score))
            .attr('r', 3)
            .attr('fill', d => d.is_anomaly ? '#ef4444' : '#3b82f6')
            .attr('opacity', 0.7);
        
        // Add interaction
        points.on('click', (event, d) => {
            this.selectDataPoint(d);
        });
        
        // Add brush for selection
        const brush = d3.brushX()
            .extent([[0, 0], [width - margin.left - margin.right, height - margin.top - margin.bottom]])
            .on('end', (event) => {
                if (event.selection) {
                    const [x0, x1] = event.selection;
                    const timeRange = [xScale.invert(x0), xScale.invert(x1)];
                    this.applyTimeRangeFilter(timeRange);
                }
            });
        
        g.append('g')
            .attr('class', 'brush')
            .call(brush);
        
        this.chartInstance = { svg, xScale, yScale, points };
    }

    renderDistributionChart(container) {
        if (!window.echarts) return;
        
        const chart = echarts.init(container);
        
        // Group data by category
        const categories = {};
        this.currentData.forEach(d => {
            if (!categories[d.category]) {
                categories[d.category] = { count: 0, anomalies: 0 };
            }
            categories[d.category].count++;
            if (d.is_anomaly) {
                categories[d.category].anomalies++;
            }
        });
        
        const option = {
            title: {
                text: 'Anomaly Distribution by Category',
                left: 'center'
            },
            tooltip: {
                trigger: 'item',
                formatter: '{a} <br/>{b}: {c} ({d}%)'
            },
            legend: {
                orient: 'vertical',
                left: 'left'
            },
            series: [
                {
                    name: 'Categories',
                    type: 'pie',
                    radius: '50%',
                    data: Object.entries(categories).map(([name, data]) => ({
                        value: data.count,
                        name: name,
                        itemStyle: {
                            color: data.anomalies > data.count * 0.1 ? '#ef4444' : '#3b82f6'
                        }
                    })),
                    emphasis: {
                        itemStyle: {
                            shadowBlur: 10,
                            shadowOffsetX: 0,
                            shadowColor: 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                }
            ]
        };
        
        chart.setOption(option);
        
        // Add click handler for drill-down
        chart.on('click', (params) => {
            if (params.data) {
                this.drillDownByCategory(params.data.name);
            }
        });
        
        this.chartInstance = chart;
    }

    renderHeatmapChart(container) {
        if (!window.d3) return;
        
        const width = container.clientWidth;
        const height = 400;
        const margin = { top: 20, right: 30, bottom: 40, left: 80 };
        
        // Create correlation matrix
        const features = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_latency'];
        const matrix = [];
        
        for (let i = 0; i < features.length; i++) {
            for (let j = 0; j < features.length; j++) {
                const correlation = this.calculateCorrelation(features[i], features[j]);
                matrix.push({
                    x: i,
                    y: j,
                    value: correlation,
                    xLabel: features[i],
                    yLabel: features[j]
                });
            }
        }
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        const cellSize = Math.min(
            (width - margin.left - margin.right) / features.length,
            (height - margin.top - margin.bottom) / features.length
        );
        
        const colorScale = d3.scaleSequential(d3.interpolateRdYlBu)
            .domain([-1, 1]);
        
        // Draw heatmap cells
        g.selectAll('.heatmap-cell')
            .data(matrix)
            .enter()
            .append('rect')
            .attr('class', 'heatmap-cell')
            .attr('x', d => d.x * cellSize)
            .attr('y', d => d.y * cellSize)
            .attr('width', cellSize)
            .attr('height', cellSize)
            .attr('fill', d => colorScale(d.value))
            .attr('stroke', '#fff')
            .attr('stroke-width', 1);
        
        // Add labels
        g.selectAll('.x-label')
            .data(features)
            .enter()
            .append('text')
            .attr('class', 'x-label')
            .attr('x', (d, i) => i * cellSize + cellSize / 2)
            .attr('y', -5)
            .attr('text-anchor', 'middle')
            .text(d => d);
        
        g.selectAll('.y-label')
            .data(features)
            .enter()
            .append('text')
            .attr('class', 'y-label')
            .attr('x', -5)
            .attr('y', (d, i) => i * cellSize + cellSize / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .text(d => d);
        
        this.chartInstance = { svg, matrix, features };
    }

    calculateCorrelation(feature1, feature2) {
        const data = this.currentData.filter(d => d.details && d.details[feature1] && d.details[feature2]);
        if (data.length === 0) return 0;
        
        const x = data.map(d => d.details[feature1]);
        const y = data.map(d => d.details[feature2]);
        
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
        const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
        const sumY2 = y.reduce((acc, yi) => acc + yi * yi, 0);
        
        const correlation = (n * sumXY - sumX * sumY) / 
            Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return isNaN(correlation) ? 0 : correlation;
    }

    selectDataPoint(dataPoint) {
        this.selectedPoint = dataPoint;
        this.updateSelectionInfo();
        this.enableActionButtons();
    }

    updateSelectionInfo() {
        const selectionInfo = document.getElementById('selection-info');
        if (this.selectedPoint) {
            selectionInfo.innerHTML = `
                <div class="selection-detail">
                    <strong>Selected Point:</strong><br>
                    <span class="detail-item">Time: ${this.selectedPoint.timestamp.toLocaleString()}</span><br>
                    <span class="detail-item">Category: ${this.selectedPoint.category}</span><br>
                    <span class="detail-item">Score: ${this.selectedPoint.anomaly_score.toFixed(3)}</span><br>
                    <span class="detail-item">Anomaly: ${this.selectedPoint.is_anomaly ? 'Yes' : 'No'}</span>
                </div>
            `;
        } else {
            selectionInfo.innerHTML = '<p>No selection</p>';
        }
    }

    enableActionButtons() {
        document.getElementById('drill-down-btn').disabled = !this.selectedPoint;
        document.getElementById('investigate-btn').disabled = !this.selectedPoint;
        document.getElementById('compare-btn').disabled = !this.selectedPoint;
    }

    performDrillDown() {
        if (!this.selectedPoint) return;
        
        const nextLevel = this.currentLevel + 1;
        if (nextLevel >= this.options.maxDepth) {
            alert('Maximum drill-down depth reached');
            return;
        }
        
        // Filter data based on selection
        const filteredData = this.currentData.filter(d => {
            if (this.currentLevel === 0) {
                return d.category === this.selectedPoint.category;
            } else if (this.currentLevel === 1) {
                return d.source === this.selectedPoint.source;
            } else {
                return d.severity === this.selectedPoint.severity;
            }
        });
        
        // Save current state
        this.navigationStack.push({
            level: this.currentLevel,
            data: this.currentData,
            filters: new Map(this.filters),
            selection: this.selectedPoint
        });
        
        // Navigate to next level
        this.currentLevel = nextLevel;
        this.currentData = filteredData;
        this.selectedPoint = null;
        
        this.updateBreadcrumb();
        this.renderChart();
        this.updateUI();
    }

    navigateBack() {
        if (this.navigationStack.length === 0) return;
        
        const previousState = this.navigationStack.pop();
        this.currentLevel = previousState.level;
        this.currentData = previousState.data;
        this.filters = previousState.filters;
        this.selectedPoint = previousState.selection;
        
        this.updateBreadcrumb();
        this.renderChart();
        this.updateUI();
    }

    navigateHome() {
        this.navigationStack = [];
        this.currentLevel = 0;
        this.currentData = this.originalData;
        this.filters.clear();
        this.selectedPoint = null;
        
        this.updateBreadcrumb();
        this.renderChart();
        this.updateUI();
    }

    updateBreadcrumb() {
        const breadcrumb = document.getElementById('breadcrumb-nav');
        breadcrumb.innerHTML = '';
        
        const levels = ['Overview', 'Category', 'Source', 'Details', 'Investigation'];
        
        for (let i = 0; i <= this.currentLevel; i++) {
            const item = document.createElement('span');
            item.className = `breadcrumb-item ${i === this.currentLevel ? 'active' : ''}`;
            item.dataset.level = i;
            item.innerHTML = `<i class="icon-${i === 0 ? 'home' : 'chevron-right'}"></i> ${levels[i]}`;
            breadcrumb.appendChild(item);
        }
    }

    updateUI() {
        // Update navigation buttons
        document.getElementById('back-button').disabled = this.navigationStack.length === 0;
        document.getElementById('forward-button').disabled = true; // Forward not implemented yet
        
        // Update filter controls
        this.populateFilterOptions();
        
        // Update action buttons
        this.enableActionButtons();
    }

    populateFilterOptions() {
        const categoryFilter = document.getElementById('category-filter');
        const categories = [...new Set(this.currentData.map(d => d.category))];
        
        categoryFilter.innerHTML = '<option value="all">All Categories</option>';
        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            categoryFilter.appendChild(option);
        });
    }

    updateDataSummary() {
        document.getElementById('total-points').textContent = this.currentData.length;
        document.getElementById('anomaly-count').textContent = this.currentData.filter(d => d.is_anomaly).length;
        
        const avgScore = this.currentData.reduce((sum, d) => sum + d.anomaly_score, 0) / this.currentData.length;
        document.getElementById('avg-score').textContent = avgScore.toFixed(3);
    }

    applyTimeFilter(range) {
        const now = new Date();
        let startTime;
        
        switch (range) {
            case '1h':
                startTime = new Date(now.getTime() - 60 * 60 * 1000);
                break;
            case '24h':
                startTime = new Date(now.getTime() - 24 * 60 * 60 * 1000);
                break;
            case '7d':
                startTime = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                break;
            case '30d':
                startTime = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                break;
            case 'all':
            default:
                startTime = null;
                break;
        }
        
        if (startTime) {
            this.currentData = this.originalData.filter(d => d.timestamp >= startTime);
        } else {
            this.currentData = this.originalData;
        }
        
        this.renderChart();
    }

    updateAnomalyFilter() {
        const minScore = parseFloat(document.getElementById('anomaly-min').value);
        const maxScore = parseFloat(document.getElementById('anomaly-max').value);
        
        document.getElementById('anomaly-min-value').textContent = minScore.toFixed(1);
        document.getElementById('anomaly-max-value').textContent = maxScore.toFixed(1);
        
        this.currentData = this.originalData.filter(d => 
            d.anomaly_score >= minScore && d.anomaly_score <= maxScore
        );
        
        this.renderChart();
    }

    investigateSelection() {
        if (!this.selectedPoint) return;
        
        document.getElementById('detail-modal').style.display = 'block';
        this.populateDetailModal();
    }

    populateDetailModal() {
        const content = document.getElementById('detail-content');
        content.innerHTML = `
            <div class="detail-overview">
                <h4>Anomaly Details</h4>
                <div class="detail-grid">
                    <div class="detail-item">
                        <label>Timestamp:</label>
                        <span>${this.selectedPoint.timestamp.toLocaleString()}</span>
                    </div>
                    <div class="detail-item">
                        <label>Category:</label>
                        <span>${this.selectedPoint.category}</span>
                    </div>
                    <div class="detail-item">
                        <label>Source:</label>
                        <span>${this.selectedPoint.source}</span>
                    </div>
                    <div class="detail-item">
                        <label>Anomaly Score:</label>
                        <span>${this.selectedPoint.anomaly_score.toFixed(3)}</span>
                    </div>
                    <div class="detail-item">
                        <label>Severity:</label>
                        <span class="severity-${this.selectedPoint.severity}">${this.selectedPoint.severity}</span>
                    </div>
                </div>
                
                <h4>System Metrics</h4>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <label>CPU Usage:</label>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${this.selectedPoint.details.cpu_usage}%"></div>
                        </div>
                        <span>${this.selectedPoint.details.cpu_usage.toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <label>Memory Usage:</label>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${this.selectedPoint.details.memory_usage}%"></div>
                        </div>
                        <span>${this.selectedPoint.details.memory_usage.toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <label>Disk Usage:</label>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${this.selectedPoint.details.disk_usage}%"></div>
                        </div>
                        <span>${this.selectedPoint.details.disk_usage.toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <label>Network Latency:</label>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: ${Math.min(this.selectedPoint.details.network_latency / 10, 100)}%"></div>
                        </div>
                        <span>${this.selectedPoint.details.network_latency.toFixed(1)}ms</span>
                    </div>
                </div>
            </div>
        `;
    }

    closeDetailModal() {
        document.getElementById('detail-modal').style.display = 'none';
    }

    toggleContextPanel() {
        const panel = document.getElementById('context-panel');
        const toggle = document.getElementById('toggle-context');
        
        panel.classList.toggle('collapsed');
        toggle.innerHTML = panel.classList.contains('collapsed') ? 
            '<i class="icon-chevron-left"></i>' : 
            '<i class="icon-chevron-right"></i>';
    }
}

// Initialize the drill-down navigation
window.AdvancedDrillDownNavigation = AdvancedDrillDownNavigation;

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedDrillDownNavigation;
}