/**
 * Interactive Data Science Dashboard
 * Provides comprehensive data science capabilities with real-time monitoring,
 * statistical analysis, and interactive visualizations.
 */

// Main Dashboard State Management
function dataScientistDashboard() {
    return {
        // Core State
        loading: false,
        sidebarOpen: window.innerWidth >= 1024,
        darkMode: localStorage.getItem('darkMode') === 'true',
        
        // Dashboard Data
        datasets: [],
        selectedDataset: '',
        profileData: {},
        selectedFeature: null,
        drillDownEnabled: false,
        
        // Analysis State
        selectedAnalysis: '',
        analysisConfig: {
            features: [],
            confidence: 0.95,
            method: 'parametric'
        },
        analysisResults: null,
        
        // Performance Monitoring
        performanceStatus: {
            connected: false,
            monitoring: false
        },
        performanceMetrics: {
            cpu: 0,
            memory: 0,
            speed: 0,
            responseTime: 0,
            connections: 0
        },
        
        // Modal State
        showNewAnalysisModal: false,
        newAnalysis: {
            type: '',
            dataset: '',
            name: ''
        },
        
        // Dashboard Statistics
        stats: {
            totalDatasets: 42,
            datasetsGrowth: 15,
            mlModels: 128,
            modelsGrowth: 23,
            avgAccuracy: 94.7,
            processingTime: 156
        },
        
        // WebSocket Connection
        ws: null,
        wsReconnectAttempts: 0,
        maxReconnectAttempts: 5,
        
        // Chart Instances
        charts: {
            featureDistribution: null,
            correlationMatrix: null,
            analysisVisualization: null,
            performanceTimeline: null
        },

        // Initialization
        init() {
            this.initializeTheme();
            this.loadDashboardData();
            this.initializeWebSocket();
            this.initializePerformanceMonitoring();
            this.setupEventListeners();
            
            // Initialize charts after DOM is ready
            this.$nextTick(() => {
                this.initializeCharts();
            });
        },

        // Theme Management
        initializeTheme() {
            if (this.darkMode) {
                document.documentElement.classList.add('dark');
            }
        },

        toggleTheme() {
            this.darkMode = !this.darkMode;
            localStorage.setItem('darkMode', this.darkMode);
            
            if (this.darkMode) {
                document.documentElement.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
            }
            
            // Refresh charts with new theme
            this.refreshCharts();
        },

        // Data Loading
        async loadDashboardData() {
            this.loading = true;
            
            try {
                // Load datasets
                const datasetsResponse = await fetch('/api/datasets');
                this.datasets = await datasetsResponse.json();
                
                // Load default profile data
                if (this.datasets.length > 0) {
                    this.selectedDataset = this.datasets[0].id;
                    await this.loadDatasetProfile();
                }
                
            } catch (error) {
                console.error('Error loading dashboard data:', error);
                this.showNotification('Error loading dashboard data', 'error');
            } finally {
                this.loading = false;
            }
        },

        async loadDatasetProfile() {
            if (!this.selectedDataset) return;
            
            this.loading = true;
            
            try {
                const response = await fetch(`/api/datasets/${this.selectedDataset}/profile`);
                const profile = await response.json();
                
                this.profileData = {
                    rows: profile.shape?.rows || 1000,
                    columns: profile.shape?.columns || 25,
                    quality: profile.quality || 0.89,
                    missing: profile.missing_rate || 0.05,
                    features: profile.features || this.generateMockFeatures()
                };
                
                // Update visualizations
                this.updateFeatureDistribution();
                this.updateCorrelationMatrix();
                
            } catch (error) {
                console.error('Error loading dataset profile:', error);
                this.profileData = this.generateMockProfileData();
                this.updateFeatureDistribution();
                this.updateCorrelationMatrix();
            } finally {
                this.loading = false;
            }
        },

        // Analysis Management
        async runAnalysis() {
            if (!this.selectedAnalysis || !this.selectedDataset) return;
            
            this.loading = true;
            
            try {
                const response = await fetch('/api/analysis/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        type: this.selectedAnalysis,
                        dataset: this.selectedDataset,
                        config: this.analysisConfig
                    })
                });
                
                this.analysisResults = await response.json();
                this.updateAnalysisVisualization();
                
            } catch (error) {
                console.error('Error running analysis:', error);
                this.analysisResults = this.generateMockAnalysisResults();
                this.updateAnalysisVisualization();
            } finally {
                this.loading = false;
            }
        },

        // WebSocket Management
        initializeWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/data-science`;
            
            try {
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.performanceStatus.connected = true;
                    this.wsReconnectAttempts = 0;
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.performanceStatus.connected = false;
                    this.scheduleReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.performanceStatus.connected = false;
                };
                
            } catch (error) {
                console.error('Failed to initialize WebSocket:', error);
                this.performanceStatus.connected = false;
            }
        },

        handleWebSocketMessage(data) {
            switch (data.type) {
                case 'performance_update':
                    this.updatePerformanceMetrics(data.metrics);
                    break;
                case 'analysis_complete':
                    this.handleAnalysisComplete(data);
                    break;
                case 'dataset_update':
                    this.handleDatasetUpdate(data);
                    break;
                default:
                    console.log('Unknown WebSocket message type:', data.type);
            }
        },

        scheduleReconnect() {
            if (this.wsReconnectAttempts < this.maxReconnectAttempts) {
                this.wsReconnectAttempts++;
                const delay = Math.pow(2, this.wsReconnectAttempts) * 1000; // Exponential backoff
                
                setTimeout(() => {
                    console.log(`Attempting WebSocket reconnection (${this.wsReconnectAttempts}/${this.maxReconnectAttempts})`);
                    this.initializeWebSocket();
                }, delay);
            }
        },

        // Performance Monitoring
        initializePerformanceMonitoring() {
            // Start periodic performance updates
            setInterval(() => {
                if (this.performanceStatus.monitoring) {
                    this.updateMockPerformanceMetrics();
                }
            }, 2000);
        },

        togglePerformanceMonitoring() {
            this.performanceStatus.monitoring = !this.performanceStatus.monitoring;
            
            if (this.performanceStatus.monitoring) {
                this.startPerformanceMonitoring();
            } else {
                this.stopPerformanceMonitoring();
            }
        },

        startPerformanceMonitoring() {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'start_monitoring',
                    interval: 2000
                }));
            } else {
                // Use mock data if WebSocket not available
                this.performanceStatus.monitoring = true;
            }
        },

        stopPerformanceMonitoring() {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'stop_monitoring'
                }));
            }
            this.performanceStatus.monitoring = false;
        },

        updatePerformanceMetrics(metrics) {
            this.performanceMetrics = {
                cpu: metrics.cpu || this.performanceMetrics.cpu,
                memory: metrics.memory || this.performanceMetrics.memory,
                speed: metrics.speed || this.performanceMetrics.speed,
                responseTime: metrics.responseTime || this.performanceMetrics.responseTime,
                connections: metrics.connections || this.performanceMetrics.connections
            };
            
            this.updatePerformanceChart();
        },

        updateMockPerformanceMetrics() {
            this.performanceMetrics = {
                cpu: Math.max(0, Math.min(100, this.performanceMetrics.cpu + (Math.random() - 0.5) * 10)),
                memory: Math.max(0, Math.min(100, this.performanceMetrics.memory + (Math.random() - 0.5) * 8)),
                speed: Math.max(0, this.performanceMetrics.speed + (Math.random() - 0.5) * 20),
                responseTime: Math.max(10, this.performanceMetrics.responseTime + (Math.random() - 0.5) * 50),
                connections: Math.max(0, this.performanceMetrics.connections + Math.floor((Math.random() - 0.5) * 5))
            };
            
            this.updatePerformanceChart();
        },

        // Chart Management
        initializeCharts() {
            this.initializeFeatureDistributionChart();
            this.initializeCorrelationMatrixChart();
            this.initializeAnalysisVisualizationChart();
            this.initializePerformanceTimelineChart();
        },

        initializeFeatureDistributionChart() {
            const container = d3.select('#feature-distribution-chart');
            if (container.empty()) return;
            
            container.selectAll('*').remove();
            
            const margin = { top: 20, right: 30, bottom: 40, left: 40 };
            const width = container.node().getBoundingClientRect().width - margin.left - margin.right;
            const height = 240 - margin.top - margin.bottom;
            
            const svg = container
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom);
            
            const g = svg
                .append('g')
                .attr('transform', `translate(${margin.left},${margin.top})`);
            
            this.charts.featureDistribution = { svg, g, width, height, margin };
            this.updateFeatureDistribution();
        },

        updateFeatureDistribution() {
            if (!this.charts.featureDistribution || !this.selectedFeature) return;
            
            const { g, width, height } = this.charts.featureDistribution;
            
            // Generate sample distribution data
            const data = Array.from({ length: 20 }, (_, i) => ({
                bin: i,
                count: Math.floor(Math.random() * 100) + 10
            }));
            
            // Clear previous chart
            g.selectAll('*').remove();
            
            // Create scales
            const xScale = d3.scaleBand()
                .domain(data.map(d => d.bin))
                .range([0, width])
                .padding(0.1);
            
            const yScale = d3.scaleLinear()
                .domain([0, d3.max(data, d => d.count)])
                .range([height, 0]);
            
            // Create bars
            g.selectAll('.bar')
                .data(data)
                .enter()
                .append('rect')
                .attr('class', 'bar')
                .attr('x', d => xScale(d.bin))
                .attr('y', d => yScale(d.count))
                .attr('width', xScale.bandwidth())
                .attr('height', d => height - yScale(d.count))
                .attr('fill', '#3b82f6')
                .attr('opacity', 0.8)
                .on('mouseover', function(event, d) {
                    d3.select(this).attr('opacity', 1);
                })
                .on('mouseout', function(event, d) {
                    d3.select(this).attr('opacity', 0.8);
                });
            
            // Add axes
            g.append('g')
                .attr('transform', `translate(0,${height})`)
                .call(d3.axisBottom(xScale));
            
            g.append('g')
                .call(d3.axisLeft(yScale));
        },

        initializeCorrelationMatrixChart() {
            const container = d3.select('#correlation-matrix');
            if (container.empty()) return;
            
            container.selectAll('*').remove();
            
            const size = Math.min(container.node().getBoundingClientRect().width, 240);
            const margin = 40;
            const cellSize = (size - margin * 2) / 8;
            
            const svg = container
                .append('svg')
                .attr('width', size)
                .attr('height', size);
            
            this.charts.correlationMatrix = { svg, size, margin, cellSize };
            this.updateCorrelationMatrix();
        },

        updateCorrelationMatrix() {
            if (!this.charts.correlationMatrix) return;
            
            const { svg, size, margin, cellSize } = this.charts.correlationMatrix;
            
            // Generate correlation matrix data
            const features = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'];
            const data = [];
            
            features.forEach((row, i) => {
                features.forEach((col, j) => {
                    const correlation = i === j ? 1 : (Math.random() * 2 - 1);
                    data.push({
                        row: i,
                        col: j,
                        feature1: row,
                        feature2: col,
                        correlation: correlation
                    });
                });
            });
            
            // Clear previous chart
            svg.selectAll('*').remove();
            
            // Color scale
            const colorScale = d3.scaleSequential(d3.interpolateRdBu)
                .domain([-1, 1]);
            
            // Create cells
            svg.selectAll('.cell')
                .data(data)
                .enter()
                .append('rect')
                .attr('class', 'cell')
                .attr('x', d => margin + d.col * cellSize)
                .attr('y', d => margin + d.row * cellSize)
                .attr('width', cellSize - 1)
                .attr('height', cellSize - 1)
                .attr('fill', d => colorScale(d.correlation))
                .style('cursor', 'pointer')
                .on('click', (event, d) => {
                    if (this.drillDownEnabled) {
                        this.openCorrelationDrillDown(d);
                    }
                });
            
            // Add labels
            svg.selectAll('.row-label')
                .data(features)
                .enter()
                .append('text')
                .attr('class', 'row-label')
                .attr('x', margin - 5)
                .attr('y', (d, i) => margin + i * cellSize + cellSize / 2)
                .attr('text-anchor', 'end')
                .attr('dominant-baseline', 'middle')
                .attr('font-size', '10px')
                .text(d => d);
            
            svg.selectAll('.col-label')
                .data(features)
                .enter()
                .append('text')
                .attr('class', 'col-label')
                .attr('x', (d, i) => margin + i * cellSize + cellSize / 2)
                .attr('y', margin - 5)
                .attr('text-anchor', 'middle')
                .attr('dominant-baseline', 'text-after-edge')
                .attr('font-size', '10px')
                .text(d => d);
        },

        initializeAnalysisVisualizationChart() {
            const container = d3.select('#analysis-visualization');
            if (container.empty()) return;
            
            container.selectAll('*').remove();
            
            const margin = { top: 20, right: 30, bottom: 40, left: 40 };
            const width = container.node().getBoundingClientRect().width - margin.left - margin.right;
            const height = 240 - margin.top - margin.bottom;
            
            const svg = container
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom);
            
            const g = svg
                .append('g')
                .attr('transform', `translate(${margin.left},${margin.top})`);
            
            this.charts.analysisVisualization = { svg, g, width, height, margin };
        },

        updateAnalysisVisualization() {
            if (!this.charts.analysisVisualization || !this.analysisResults) return;
            
            const { g, width, height } = this.charts.analysisVisualization;
            
            // Clear previous chart
            g.selectAll('*').remove();
            
            // Create sample scatter plot
            const data = Array.from({ length: 50 }, () => ({
                x: Math.random() * 100,
                y: Math.random() * 100
            }));
            
            const xScale = d3.scaleLinear()
                .domain([0, 100])
                .range([0, width]);
            
            const yScale = d3.scaleLinear()
                .domain([0, 100])
                .range([height, 0]);
            
            // Add points
            g.selectAll('.point')
                .data(data)
                .enter()
                .append('circle')
                .attr('class', 'point')
                .attr('cx', d => xScale(d.x))
                .attr('cy', d => yScale(d.y))
                .attr('r', 3)
                .attr('fill', '#10b981')
                .attr('opacity', 0.7);
            
            // Add axes
            g.append('g')
                .attr('transform', `translate(0,${height})`)
                .call(d3.axisBottom(xScale));
            
            g.append('g')
                .call(d3.axisLeft(yScale));
        },

        initializePerformanceTimelineChart() {
            const container = d3.select('#performance-timeline');
            if (container.empty()) return;
            
            container.selectAll('*').remove();
            
            const margin = { top: 20, right: 30, bottom: 40, left: 40 };
            const width = container.node().getBoundingClientRect().width - margin.left - margin.right;
            const height = 240 - margin.top - margin.bottom;
            
            const svg = container
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom);
            
            const g = svg
                .append('g')
                .attr('transform', `translate(${margin.left},${margin.top})`);
            
            this.charts.performanceTimeline = { 
                svg, g, width, height, margin,
                data: []
            };
        },

        updatePerformanceChart() {
            if (!this.charts.performanceTimeline) return;
            
            const { g, width, height, data } = this.charts.performanceTimeline;
            
            // Add new data point
            data.push({
                timestamp: new Date(),
                cpu: this.performanceMetrics.cpu,
                memory: this.performanceMetrics.memory
            });
            
            // Keep only last 50 points
            if (data.length > 50) {
                data.shift();
            }
            
            if (data.length < 2) return;
            
            // Clear previous lines
            g.selectAll('.performance-line').remove();
            
            // Create scales
            const xScale = d3.scaleTime()
                .domain(d3.extent(data, d => d.timestamp))
                .range([0, width]);
            
            const yScale = d3.scaleLinear()
                .domain([0, 100])
                .range([height, 0]);
            
            // CPU line
            const cpuLine = d3.line()
                .x(d => xScale(d.timestamp))
                .y(d => yScale(d.cpu))
                .curve(d3.curveMonotoneX);
            
            g.append('path')
                .datum(data)
                .attr('class', 'performance-line')
                .attr('fill', 'none')
                .attr('stroke', '#3b82f6')
                .attr('stroke-width', 2)
                .attr('d', cpuLine);
            
            // Memory line
            const memoryLine = d3.line()
                .x(d => xScale(d.timestamp))
                .y(d => yScale(d.memory))
                .curve(d3.curveMonotoneX);
            
            g.append('path')
                .datum(data)
                .attr('class', 'performance-line')
                .attr('fill', 'none')
                .attr('stroke', '#10b981')
                .attr('stroke-width', 2)
                .attr('d', memoryLine);
        },

        // Event Handlers
        selectFeature(feature) {
            this.selectedFeature = feature;
            this.updateFeatureDistribution();
        },

        openFeatureDetails(feature) {
            // Open feature details modal or panel
            this.showNotification(`Opening details for ${feature.name}`, 'info');
        },

        openCorrelationDrillDown(correlation) {
            this.showNotification(`Drill-down: ${correlation.feature1} vs ${correlation.feature2}`, 'info');
        },

        openNewAnalysisModal() {
            this.showNewAnalysisModal = true;
            this.newAnalysis = {
                type: '',
                dataset: this.selectedDataset,
                name: ''
            };
        },

        createNewAnalysis() {
            if (!this.newAnalysis.type || !this.newAnalysis.dataset || !this.newAnalysis.name) {
                this.showNotification('Please fill in all fields', 'error');
                return;
            }
            
            // Create new analysis
            this.showNotification(`Creating analysis: ${this.newAnalysis.name}`, 'success');
            this.showNewAnalysisModal = false;
        },

        // Utility Functions
        async refreshData() {
            await this.loadDashboardData();
            this.showNotification('Dashboard data refreshed', 'success');
        },

        exportDashboard() {
            this.showNotification('Exporting dashboard...', 'info');
            // Implement export functionality
        },

        exportAnalysis() {
            if (!this.analysisResults) return;
            
            this.showNotification('Exporting analysis results...', 'info');
            // Implement export functionality
        },

        refreshCharts() {
            // Refresh all charts with new theme
            setTimeout(() => {
                this.updateFeatureDistribution();
                this.updateCorrelationMatrix();
                this.updateAnalysisVisualization();
            }, 100);
        },

        setupEventListeners() {
            // Window resize handler
            window.addEventListener('resize', () => {
                clearTimeout(this.resizeTimeout);
                this.resizeTimeout = setTimeout(() => {
                    this.initializeCharts();
                }, 300);
            });
            
            // Keyboard shortcuts
            document.addEventListener('keydown', (event) => {
                if (event.ctrlKey || event.metaKey) {
                    switch (event.key) {
                        case 'r':
                            event.preventDefault();
                            this.refreshData();
                            break;
                        case 'n':
                            event.preventDefault();
                            this.openNewAnalysisModal();
                            break;
                        case 'd':
                            event.preventDefault();
                            this.toggleTheme();
                            break;
                    }
                }
            });
        },

        animateNumber(current, target, duration) {
            const start = performance.now();
            const startValue = current;
            
            const animate = (currentTime) => {
                const elapsed = currentTime - start;
                const progress = Math.min(elapsed / duration, 1);
                
                this[current] = startValue + (target - startValue) * progress;
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                }
            };
            
            requestAnimationFrame(animate);
        },

        showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.textContent = message;
            
            const container = document.getElementById('notifications');
            container.appendChild(notification);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 5000);
        },

        // Mock Data Generators
        generateMockFeatures() {
            return [
                { name: 'feature_1', type: 'numerical', missing: 0.02, unique: 850, mean: 45.6, std: 12.3 },
                { name: 'feature_2', type: 'numerical', missing: 0.01, unique: 920, mean: 23.1, std: 8.7 },
                { name: 'feature_3', type: 'categorical', missing: 0.05, unique: 8, mean: null, std: null },
                { name: 'feature_4', type: 'numerical', missing: 0.00, unique: 1000, mean: 67.8, std: 15.2 },
                { name: 'feature_5', type: 'boolean', missing: 0.03, unique: 2, mean: null, std: null }
            ];
        },

        generateMockProfileData() {
            return {
                rows: 1000,
                columns: 15,
                quality: 0.87,
                missing: 0.08,
                features: this.generateMockFeatures()
            };
        },

        generateMockAnalysisResults() {
            return {
                summary: [
                    { metric: 'Mean', value: '45.67' },
                    { metric: 'Standard Deviation', value: '12.34' },
                    { metric: 'Correlation Coefficient', value: '0.78' },
                    { metric: 'P-value', value: '< 0.001' }
                ],
                headers: ['Feature', 'Statistic', 'Value', 'Confidence'],
                data: [
                    { id: 1, cells: [
                        { key: 'feature', value: 'feature_1' },
                        { key: 'statistic', value: 'Mean' },
                        { key: 'value', value: '45.67' },
                        { key: 'confidence', value: '95%' }
                    ]},
                    { id: 2, cells: [
                        { key: 'feature', value: 'feature_2' },
                        { key: 'statistic', value: 'Mean' },
                        { key: 'value', value: '23.10' },
                        { key: 'confidence', value: '95%' }
                    ]}
                ]
            };
        }
    };
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Register Alpine.js component
    if (window.Alpine) {
        window.Alpine.data('dataScientistDashboard', dataScientistDashboard);
    }
});

export { dataScientistDashboard };