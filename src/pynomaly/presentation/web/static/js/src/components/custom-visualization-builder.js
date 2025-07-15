/**
 * Custom Visualization Builder - Advanced Interactive Chart Builder
 * 
 * Features:
 * - Drag-and-drop visualization creation
 * - Real-time preview and editing
 * - Multiple chart type support
 * - Custom styling and theming
 * - Export and sharing capabilities
 * - Template gallery and presets
 */

class CustomVisualizationBuilder {
    constructor(container) {
        this.container = container;
        this.currentChart = null;
        this.previewContainer = null;
        this.templates = new Map();
        this.currentConfig = {
            type: 'line',
            data: [],
            style: {
                theme: 'default',
                colors: ['#3b82f6', '#ef4444', '#10b981'],
                width: 800,
                height: 400
            },
            interactions: {
                zoom: true,
                pan: true,
                tooltip: true,
                selection: true
            }
        };
        this.init();
    }

    init() {
        this.setupUI();
        this.loadTemplates();
        this.bindEvents();
        this.createPreview();
    }

    setupUI() {
        this.container.innerHTML = `
            <div class="visualization-builder">
                <!-- Header -->
                <div class="builder-header">
                    <h2 class="text-2xl font-bold text-gray-800">Custom Visualization Builder</h2>
                    <div class="header-actions">
                        <button id="save-visualization" class="btn btn-primary">Save</button>
                        <button id="export-visualization" class="btn btn-secondary">Export</button>
                        <button id="share-visualization" class="btn btn-outline">Share</button>
                    </div>
                </div>

                <!-- Builder Content -->
                <div class="builder-content">
                    <!-- Sidebar -->
                    <div class="builder-sidebar">
                        <!-- Chart Type Selector -->
                        <div class="panel chart-types">
                            <h3>Chart Types</h3>
                            <div class="chart-type-grid">
                                <div class="chart-type-item active" data-type="line">
                                    <div class="chart-icon">📈</div>
                                    <span>Line Chart</span>
                                </div>
                                <div class="chart-type-item" data-type="bar">
                                    <div class="chart-icon">📊</div>
                                    <span>Bar Chart</span>
                                </div>
                                <div class="chart-type-item" data-type="scatter">
                                    <div class="chart-icon">🔘</div>
                                    <span>Scatter Plot</span>
                                </div>
                                <div class="chart-type-item" data-type="heatmap">
                                    <div class="chart-icon">🔥</div>
                                    <span>Heatmap</span>
                                </div>
                                <div class="chart-type-item" data-type="network">
                                    <div class="chart-icon">🕸️</div>
                                    <span>Network Graph</span>
                                </div>
                                <div class="chart-type-item" data-type="3d">
                                    <div class="chart-icon">🎯</div>
                                    <span>3D Plot</span>
                                </div>
                            </div>
                        </div>

                        <!-- Data Configuration -->
                        <div class="panel data-config">
                            <h3>Data Configuration</h3>
                            <div class="data-source">
                                <label>Data Source:</label>
                                <select id="data-source">
                                    <option value="sample">Sample Data</option>
                                    <option value="api">API Endpoint</option>
                                    <option value="upload">Upload CSV</option>
                                    <option value="realtime">Real-time Stream</option>
                                </select>
                            </div>
                            <div class="data-fields">
                                <label>X-Axis:</label>
                                <select id="x-axis-field">
                                    <option value="timestamp">Timestamp</option>
                                    <option value="value">Value</option>
                                    <option value="category">Category</option>
                                </select>
                                <label>Y-Axis:</label>
                                <select id="y-axis-field">
                                    <option value="score">Anomaly Score</option>
                                    <option value="value">Value</option>
                                    <option value="count">Count</option>
                                </select>
                            </div>
                        </div>

                        <!-- Style Configuration -->
                        <div class="panel style-config">
                            <h3>Style Configuration</h3>
                            <div class="style-options">
                                <div class="option-group">
                                    <label>Theme:</label>
                                    <select id="theme-selector">
                                        <option value="default">Default</option>
                                        <option value="dark">Dark</option>
                                        <option value="light">Light</option>
                                        <option value="minimal">Minimal</option>
                                    </select>
                                </div>
                                <div class="option-group">
                                    <label>Color Scheme:</label>
                                    <div class="color-palette">
                                        <input type="color" id="color-1" value="#3b82f6">
                                        <input type="color" id="color-2" value="#ef4444">
                                        <input type="color" id="color-3" value="#10b981">
                                    </div>
                                </div>
                                <div class="option-group">
                                    <label>Width:</label>
                                    <input type="range" id="width-slider" min="400" max="1200" value="800">
                                    <span id="width-value">800px</span>
                                </div>
                                <div class="option-group">
                                    <label>Height:</label>
                                    <input type="range" id="height-slider" min="300" max="800" value="400">
                                    <span id="height-value">400px</span>
                                </div>
                            </div>
                        </div>

                        <!-- Interaction Configuration -->
                        <div class="panel interaction-config">
                            <h3>Interactions</h3>
                            <div class="interaction-options">
                                <label class="checkbox-label">
                                    <input type="checkbox" id="enable-zoom" checked>
                                    Enable Zoom
                                </label>
                                <label class="checkbox-label">
                                    <input type="checkbox" id="enable-pan" checked>
                                    Enable Pan
                                </label>
                                <label class="checkbox-label">
                                    <input type="checkbox" id="enable-tooltip" checked>
                                    Show Tooltips
                                </label>
                                <label class="checkbox-label">
                                    <input type="checkbox" id="enable-selection" checked>
                                    Enable Selection
                                </label>
                                <label class="checkbox-label">
                                    <input type="checkbox" id="enable-brush" checked>
                                    Enable Brush
                                </label>
                            </div>
                        </div>

                        <!-- Templates -->
                        <div class="panel templates">
                            <h3>Templates</h3>
                            <div class="template-gallery" id="template-gallery">
                                <!-- Templates will be populated here -->
                            </div>
                        </div>
                    </div>

                    <!-- Main Preview Area -->
                    <div class="builder-main">
                        <div class="preview-header">
                            <h3>Preview</h3>
                            <div class="preview-actions">
                                <button id="refresh-preview" class="btn btn-sm">Refresh</button>
                                <button id="fullscreen-preview" class="btn btn-sm">Fullscreen</button>
                            </div>
                        </div>
                        <div class="preview-container" id="preview-container">
                            <!-- Chart preview will be rendered here -->
                        </div>
                        <div class="preview-footer">
                            <div class="chart-info">
                                <span id="chart-type-info">Line Chart</span>
                                <span id="data-points-info">0 data points</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    loadTemplates() {
        const templates = [
            {
                id: 'anomaly-timeline',
                name: 'Anomaly Timeline',
                type: 'line',
                config: {
                    type: 'line',
                    style: {
                        theme: 'default',
                        colors: ['#ef4444', '#f59e0b'],
                        showPoints: true,
                        lineWidth: 2
                    },
                    interactions: {
                        zoom: true,
                        brush: true,
                        tooltip: true
                    }
                }
            },
            {
                id: 'correlation-matrix',
                name: 'Correlation Matrix',
                type: 'heatmap',
                config: {
                    type: 'heatmap',
                    style: {
                        theme: 'default',
                        colorScale: 'RdYlBu',
                        showLabels: true
                    },
                    interactions: {
                        tooltip: true,
                        selection: true
                    }
                }
            },
            {
                id: 'feature-importance',
                name: 'Feature Importance',
                type: 'bar',
                config: {
                    type: 'bar',
                    style: {
                        theme: 'default',
                        colors: ['#10b981'],
                        orientation: 'horizontal'
                    },
                    interactions: {
                        tooltip: true,
                        selection: true
                    }
                }
            },
            {
                id: 'network-analysis',
                name: 'Network Analysis',
                type: 'network',
                config: {
                    type: 'network',
                    style: {
                        theme: 'default',
                        nodeSize: 'degree',
                        linkWidth: 'weight',
                        colorBy: 'community'
                    },
                    interactions: {
                        drag: true,
                        zoom: true,
                        tooltip: true
                    }
                }
            }
        ];

        templates.forEach(template => {
            this.templates.set(template.id, template);
        });

        this.renderTemplateGallery();
    }

    renderTemplateGallery() {
        const gallery = document.getElementById('template-gallery');
        gallery.innerHTML = '';

        this.templates.forEach(template => {
            const templateElement = document.createElement('div');
            templateElement.className = 'template-item';
            templateElement.innerHTML = `
                <div class="template-preview">
                    <div class="template-icon">${this.getTemplateIcon(template.type)}</div>
                </div>
                <div class="template-name">${template.name}</div>
            `;
            templateElement.addEventListener('click', () => this.loadTemplate(template.id));
            gallery.appendChild(templateElement);
        });
    }

    getTemplateIcon(type) {
        const icons = {
            line: '📈',
            bar: '📊',
            scatter: '🔘',
            heatmap: '🔥',
            network: '🕸️',
            '3d': '🎯'
        };
        return icons[type] || '📊';
    }

    bindEvents() {
        // Chart type selection
        document.querySelectorAll('.chart-type-item').forEach(item => {
            item.addEventListener('click', (e) => {
                document.querySelectorAll('.chart-type-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                this.currentConfig.type = item.dataset.type;
                this.updatePreview();
            });
        });

        // Style configuration
        document.getElementById('theme-selector').addEventListener('change', (e) => {
            this.currentConfig.style.theme = e.target.value;
            this.updatePreview();
        });

        document.getElementById('width-slider').addEventListener('input', (e) => {
            this.currentConfig.style.width = parseInt(e.target.value);
            document.getElementById('width-value').textContent = `${e.target.value}px`;
            this.updatePreview();
        });

        document.getElementById('height-slider').addEventListener('input', (e) => {
            this.currentConfig.style.height = parseInt(e.target.value);
            document.getElementById('height-value').textContent = `${e.target.value}px`;
            this.updatePreview();
        });

        // Color palette
        ['color-1', 'color-2', 'color-3'].forEach((id, index) => {
            document.getElementById(id).addEventListener('change', (e) => {
                this.currentConfig.style.colors[index] = e.target.value;
                this.updatePreview();
            });
        });

        // Interaction toggles
        ['zoom', 'pan', 'tooltip', 'selection', 'brush'].forEach(interaction => {
            document.getElementById(`enable-${interaction}`).addEventListener('change', (e) => {
                this.currentConfig.interactions[interaction] = e.target.checked;
                this.updatePreview();
            });
        });

        // Data source
        document.getElementById('data-source').addEventListener('change', (e) => {
            this.loadDataSource(e.target.value);
        });

        // Actions
        document.getElementById('save-visualization').addEventListener('click', () => this.saveVisualization());
        document.getElementById('export-visualization').addEventListener('click', () => this.exportVisualization());
        document.getElementById('share-visualization').addEventListener('click', () => this.shareVisualization());
        document.getElementById('refresh-preview').addEventListener('click', () => this.updatePreview());
        document.getElementById('fullscreen-preview').addEventListener('click', () => this.fullscreenPreview());
    }

    createPreview() {
        this.previewContainer = document.getElementById('preview-container');
        this.loadSampleData();
        this.updatePreview();
    }

    loadSampleData() {
        // Generate sample anomaly data
        const now = new Date();
        const data = [];
        
        for (let i = 0; i < 100; i++) {
            const timestamp = new Date(now.getTime() - (100 - i) * 60000);
            const baseValue = Math.sin(i * 0.1) * 50 + 100;
            const anomaly = Math.random() > 0.9 ? Math.random() * 100 : 0;
            
            data.push({
                timestamp: timestamp,
                value: baseValue + anomaly,
                anomaly_score: anomaly > 0 ? 0.8 + Math.random() * 0.2 : Math.random() * 0.2,
                category: `Category ${(i % 5) + 1}`,
                is_anomaly: anomaly > 0
            });
        }
        
        this.currentConfig.data = data;
    }

    updatePreview() {
        if (!this.previewContainer) return;

        // Clear previous chart
        this.previewContainer.innerHTML = '';

        // Create new chart based on configuration
        const chartContainer = document.createElement('div');
        chartContainer.style.width = `${this.currentConfig.style.width}px`;
        chartContainer.style.height = `${this.currentConfig.style.height}px`;
        this.previewContainer.appendChild(chartContainer);

        // Render chart based on type
        switch (this.currentConfig.type) {
            case 'line':
                this.renderLineChart(chartContainer);
                break;
            case 'bar':
                this.renderBarChart(chartContainer);
                break;
            case 'scatter':
                this.renderScatterPlot(chartContainer);
                break;
            case 'heatmap':
                this.renderHeatmap(chartContainer);
                break;
            case 'network':
                this.renderNetworkGraph(chartContainer);
                break;
            case '3d':
                this.render3DPlot(chartContainer);
                break;
        }

        // Update info
        document.getElementById('chart-type-info').textContent = 
            this.currentConfig.type.charAt(0).toUpperCase() + this.currentConfig.type.slice(1) + ' Chart';
        document.getElementById('data-points-info').textContent = 
            `${this.currentConfig.data.length} data points`;
    }

    renderLineChart(container) {
        // Use existing D3 chart library
        if (window.D3ChartLibrary) {
            const chart = new window.D3ChartLibrary.TimeSeriesChart(container, {
                data: this.currentConfig.data,
                xAccessor: d => d.timestamp,
                yAccessor: d => d.value,
                colorAccessor: d => d.is_anomaly ? this.currentConfig.style.colors[1] : this.currentConfig.style.colors[0],
                theme: this.currentConfig.style.theme,
                interactions: this.currentConfig.interactions
            });
            this.currentChart = chart;
        }
    }

    renderBarChart(container) {
        // Aggregate data by category
        const aggregated = this.currentConfig.data.reduce((acc, item) => {
            if (!acc[item.category]) {
                acc[item.category] = { category: item.category, value: 0, count: 0 };
            }
            acc[item.category].value += item.value;
            acc[item.category].count += 1;
            return acc;
        }, {});

        const chartData = Object.values(aggregated);
        
        // Create ECharts bar chart
        if (window.echarts) {
            const chart = window.echarts.init(container);
            const option = {
                theme: this.currentConfig.style.theme,
                xAxis: {
                    type: 'category',
                    data: chartData.map(d => d.category)
                },
                yAxis: {
                    type: 'value'
                },
                series: [{
                    data: chartData.map(d => d.value),
                    type: 'bar',
                    itemStyle: {
                        color: this.currentConfig.style.colors[0]
                    }
                }],
                tooltip: {
                    trigger: 'axis'
                }
            };
            chart.setOption(option);
            this.currentChart = chart;
        }
    }

    renderScatterPlot(container) {
        if (window.D3ChartLibrary) {
            const chart = new window.D3ChartLibrary.ScatterPlotChart(container, {
                data: this.currentConfig.data,
                xAccessor: d => d.value,
                yAccessor: d => d.anomaly_score,
                colorAccessor: d => d.is_anomaly ? this.currentConfig.style.colors[1] : this.currentConfig.style.colors[0],
                theme: this.currentConfig.style.theme,
                interactions: this.currentConfig.interactions
            });
            this.currentChart = chart;
        }
    }

    renderHeatmap(container) {
        // Generate correlation matrix data
        const features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'];
        const correlationData = [];
        
        for (let i = 0; i < features.length; i++) {
            for (let j = 0; j < features.length; j++) {
                correlationData.push({
                    x: i,
                    y: j,
                    value: i === j ? 1 : (Math.random() - 0.5) * 2,
                    xLabel: features[i],
                    yLabel: features[j]
                });
            }
        }

        if (window.D3ChartLibrary) {
            const chart = new window.D3ChartLibrary.HeatmapChart(container, {
                data: correlationData,
                theme: this.currentConfig.style.theme,
                interactions: this.currentConfig.interactions
            });
            this.currentChart = chart;
        }
    }

    renderNetworkGraph(container) {
        // Generate network data
        const nodes = [];
        const links = [];
        
        for (let i = 0; i < 20; i++) {
            nodes.push({
                id: i,
                name: `Node ${i}`,
                group: Math.floor(i / 5),
                size: Math.random() * 10 + 5
            });
        }
        
        for (let i = 0; i < 30; i++) {
            const source = Math.floor(Math.random() * nodes.length);
            const target = Math.floor(Math.random() * nodes.length);
            if (source !== target) {
                links.push({
                    source: source,
                    target: target,
                    value: Math.random()
                });
            }
        }

        this.renderD3Network(container, { nodes, links });
    }

    renderD3Network(container, data) {
        if (!window.d3) return;

        const width = this.currentConfig.style.width;
        const height = this.currentConfig.style.height;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        const simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links).id(d => d.id))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2));

        const link = svg.append('g')
            .selectAll('line')
            .data(data.links)
            .enter().append('line')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', d => Math.sqrt(d.value) * 2);

        const node = svg.append('g')
            .selectAll('circle')
            .data(data.nodes)
            .enter().append('circle')
            .attr('r', d => d.size)
            .attr('fill', d => this.currentConfig.style.colors[d.group % this.currentConfig.style.colors.length])
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        node.append('title')
            .text(d => d.name);

        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
        });

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    }

    render3DPlot(container) {
        // Placeholder for 3D visualization
        container.innerHTML = `
            <div class="placeholder-3d">
                <div class="placeholder-content">
                    <h3>3D Visualization</h3>
                    <p>3D plotting capabilities coming soon...</p>
                    <p>Will support:</p>
                    <ul>
                        <li>3D scatter plots</li>
                        <li>3D surface plots</li>
                        <li>3D network graphs</li>
                        <li>WebGL acceleration</li>
                    </ul>
                </div>
            </div>
        `;
    }

    loadTemplate(templateId) {
        const template = this.templates.get(templateId);
        if (!template) return;

        this.currentConfig = { ...this.currentConfig, ...template.config };
        this.updateUI();
        this.updatePreview();
    }

    updateUI() {
        // Update chart type selection
        document.querySelectorAll('.chart-type-item').forEach(item => {
            item.classList.toggle('active', item.dataset.type === this.currentConfig.type);
        });

        // Update theme selector
        document.getElementById('theme-selector').value = this.currentConfig.style.theme;

        // Update color palette
        this.currentConfig.style.colors.forEach((color, index) => {
            const colorInput = document.getElementById(`color-${index + 1}`);
            if (colorInput) colorInput.value = color;
        });

        // Update sliders
        document.getElementById('width-slider').value = this.currentConfig.style.width;
        document.getElementById('height-slider').value = this.currentConfig.style.height;
        document.getElementById('width-value').textContent = `${this.currentConfig.style.width}px`;
        document.getElementById('height-value').textContent = `${this.currentConfig.style.height}px`;

        // Update interaction checkboxes
        Object.entries(this.currentConfig.interactions).forEach(([key, value]) => {
            const checkbox = document.getElementById(`enable-${key}`);
            if (checkbox) checkbox.checked = value;
        });
    }

    saveVisualization() {
        const name = prompt('Enter visualization name:');
        if (!name) return;

        const visualization = {
            id: Date.now().toString(),
            name: name,
            config: this.currentConfig,
            created: new Date().toISOString()
        };

        // Save to localStorage for now
        const saved = JSON.parse(localStorage.getItem('custom-visualizations') || '[]');
        saved.push(visualization);
        localStorage.setItem('custom-visualizations', JSON.stringify(saved));

        alert('Visualization saved successfully!');
    }

    exportVisualization() {
        const format = prompt('Export format (png, svg, pdf):', 'png');
        if (!format) return;

        // Implementation depends on chart library
        if (this.currentChart) {
            if (format === 'png') {
                this.exportAsPNG();
            } else if (format === 'svg') {
                this.exportAsSVG();
            } else if (format === 'pdf') {
                this.exportAsPDF();
            }
        }
    }

    exportAsPNG() {
        // Convert chart to PNG
        const canvas = document.createElement('canvas');
        canvas.width = this.currentConfig.style.width;
        canvas.height = this.currentConfig.style.height;
        
        // This is a simplified version - real implementation would depend on the chart library
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Download the image
        const link = document.createElement('a');
        link.download = 'visualization.png';
        link.href = canvas.toDataURL();
        link.click();
    }

    exportAsSVG() {
        // Export as SVG
        const svgElement = this.previewContainer.querySelector('svg');
        if (svgElement) {
            const serializer = new XMLSerializer();
            const svgString = serializer.serializeToString(svgElement);
            const blob = new Blob([svgString], { type: 'image/svg+xml' });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.download = 'visualization.svg';
            link.href = url;
            link.click();
            URL.revokeObjectURL(url);
        }
    }

    exportAsPDF() {
        // Export as PDF - would need jsPDF library
        alert('PDF export requires additional libraries. Feature coming soon!');
    }

    shareVisualization() {
        const shareData = {
            title: 'Custom Visualization',
            config: this.currentConfig,
            url: window.location.href
        };

        // Generate shareable link
        const encodedData = btoa(JSON.stringify(shareData));
        const shareUrl = `${window.location.origin}/shared-visualization?data=${encodedData}`;

        // Copy to clipboard
        navigator.clipboard.writeText(shareUrl).then(() => {
            alert('Shareable link copied to clipboard!');
        });
    }

    fullscreenPreview() {
        if (this.previewContainer.requestFullscreen) {
            this.previewContainer.requestFullscreen();
        }
    }

    loadDataSource(source) {
        switch (source) {
            case 'sample':
                this.loadSampleData();
                break;
            case 'api':
                this.loadFromAPI();
                break;
            case 'upload':
                this.loadFromUpload();
                break;
            case 'realtime':
                this.loadRealTimeData();
                break;
        }
        this.updatePreview();
    }

    loadFromAPI() {
        // Load data from API endpoint
        fetch('/api/visualization-data')
            .then(response => response.json())
            .then(data => {
                this.currentConfig.data = data;
                this.updatePreview();
            })
            .catch(error => {
                console.error('Error loading API data:', error);
                alert('Failed to load data from API');
            });
    }

    loadFromUpload() {
        // Create file input for CSV upload
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.csv';
        input.onchange = (event) => {
            const file = event.target.files[0];
            if (file) {
                this.parseCSV(file);
            }
        };
        input.click();
    }

    parseCSV(file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const csv = event.target.result;
            const lines = csv.split('\n');
            const headers = lines[0].split(',');
            const data = [];

            for (let i = 1; i < lines.length; i++) {
                const values = lines[i].split(',');
                if (values.length === headers.length) {
                    const row = {};
                    headers.forEach((header, index) => {
                        row[header.trim()] = values[index].trim();
                    });
                    data.push(row);
                }
            }

            this.currentConfig.data = data;
            this.updatePreview();
        };
        reader.readAsText(file);
    }

    loadRealTimeData() {
        // Set up WebSocket for real-time data
        const ws = new WebSocket('ws://localhost:8000/ws/visualization-data');
        
        ws.onmessage = (event) => {
            const newData = JSON.parse(event.data);
            this.currentConfig.data = newData;
            this.updatePreview();
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            alert('Failed to connect to real-time data stream');
        };
    }
}

// Initialize the builder when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const builderContainer = document.getElementById('visualization-builder');
    if (builderContainer) {
        new CustomVisualizationBuilder(builderContainer);
    }
});

// Export for use in other modules
window.CustomVisualizationBuilder = CustomVisualizationBuilder;