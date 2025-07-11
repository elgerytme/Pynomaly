/**
 * Advanced Interactive Dashboard - Issue #114 Implementation
 * Comprehensive real-time analytics with drill-down capabilities
 */

export class AdvancedInteractiveDashboard {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      enableRealTime: true,
      enableDrillDown: true,
      enableCollaboration: true,
      enableAdvancedFiltering: true,
      refreshInterval: 5000,
      maxDataPoints: 10000,
      enableWebGL: true,
      ...options
    };

    this.state = {
      filters: {},
      selectedTimeRange: '24h',
      realTimeEnabled: true,
      drillDownStack: [],
      collaborationMode: false,
      currentView: 'overview'
    };

    this.websocket = null;
    this.charts = new Map();
    this.interactions = new Map();
    this.realTimeData = new Map();
    this.eventListeners = new Map();

    this.init();
  }

  async init() {
    await this.setupLayout();
    await this.initializeCharts();
    await this.setupInteractions();
    await this.connectRealTime();
    await this.loadInitialData();
    this.setupKeyboardShortcuts();
    this.setupCollaboration();
  }

  async setupLayout() {
    this.container.innerHTML = `
      <div class="advanced-dashboard" id="advanced-dashboard">
        <!-- Dashboard Header -->
        <div class="dashboard-header">
          <div class="dashboard-title">
            <h1 class="text-2xl font-bold">Advanced Analytics Dashboard</h1>
            <div class="real-time-indicator ${this.state.realTimeEnabled ? 'active' : 'inactive'}">
              <div class="pulse-dot"></div>
              <span>Real-time ${this.state.realTimeEnabled ? 'ON' : 'OFF'}</span>
            </div>
          </div>
          
          <div class="dashboard-controls">
            <div class="time-range-selector">
              <label class="sr-only">Time Range</label>
              <select id="time-range-select" class="form-select">
                <option value="1h">Last Hour</option>
                <option value="6h">Last 6 Hours</option>
                <option value="24h" selected>Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
                <option value="custom">Custom Range</option>
              </select>
            </div>
            
            <div class="view-controls">
              <button id="toggle-realtime" class="btn-base btn-secondary" aria-label="Toggle real-time updates">
                <span class="realtime-icon">📡</span>
              </button>
              <button id="fullscreen-toggle" class="btn-base btn-secondary" aria-label="Toggle fullscreen">
                <span class="fullscreen-icon">⛶</span>
              </button>
              <button id="export-dashboard" class="btn-base btn-secondary" aria-label="Export dashboard">
                <span class="export-icon">📊</span>
              </button>
            </div>
          </div>
        </div>

        <!-- Advanced Filters Panel -->
        <div class="filters-panel" id="filters-panel">
          <div class="filters-header">
            <h3>Advanced Filters</h3>
            <button id="toggle-filters" class="btn-base btn-ghost">
              <span class="filter-toggle-icon">⚙️</span>
            </button>
          </div>
          <div class="filters-content">
            <div class="filter-group">
              <label>Algorithm Type</label>
              <div class="multi-select" id="algorithm-filter">
                <div class="select-all">
                  <input type="checkbox" id="select-all-algorithms" checked>
                  <label for="select-all-algorithms">All Algorithms</label>
                </div>
              </div>
            </div>
            
            <div class="filter-group">
              <label>Anomaly Score Range</label>
              <div class="range-slider" id="score-range-slider">
                <input type="range" min="0" max="1" step="0.01" value="0" id="score-min">
                <input type="range" min="0" max="1" step="0.01" value="1" id="score-max">
                <div class="range-display">
                  <span id="score-range-text">0.00 - 1.00</span>
                </div>
              </div>
            </div>
            
            <div class="filter-group">
              <label>Dataset Source</label>
              <select id="dataset-filter" class="form-select" multiple>
                <option value="all" selected>All Sources</option>
              </select>
            </div>
            
            <div class="filter-actions">
              <button id="apply-filters" class="btn-base btn-primary">Apply Filters</button>
              <button id="reset-filters" class="btn-base btn-secondary">Reset</button>
            </div>
          </div>
        </div>

        <!-- Main Dashboard Grid -->
        <div class="dashboard-grid" id="dashboard-grid">
          <!-- Overview Cards -->
          <div class="dashboard-section overview-cards">
            <div class="metric-card" id="total-detections-card">
              <div class="metric-header">
                <h3>Total Detections</h3>
                <span class="trend-indicator"></span>
              </div>
              <div class="metric-value">
                <span class="value">-</span>
                <span class="change">-</span>
              </div>
            </div>
            
            <div class="metric-card" id="anomaly-rate-card">
              <div class="metric-header">
                <h3>Anomaly Rate</h3>
                <span class="trend-indicator"></span>
              </div>
              <div class="metric-value">
                <span class="value">-</span>
                <span class="change">-</span>
              </div>
            </div>
            
            <div class="metric-card" id="avg-processing-time-card">
              <div class="metric-header">
                <h3>Avg Processing Time</h3>
                <span class="trend-indicator"></span>
              </div>
              <div class="metric-value">
                <span class="value">-</span>
                <span class="change">-</span>
              </div>
            </div>
            
            <div class="metric-card" id="system-health-card">
              <div class="metric-header">
                <h3>System Health</h3>
                <span class="status-indicator"></span>
              </div>
              <div class="metric-value">
                <span class="value">-</span>
                <span class="status">-</span>
              </div>
            </div>
          </div>

          <!-- Interactive Charts -->
          <div class="chart-section large-chart">
            <div class="chart-header">
              <h3>Anomaly Timeline - Interactive Drill-down</h3>
              <div class="chart-controls">
                <button class="chart-control" data-chart="timeline" data-action="zoom-in" aria-label="Zoom in">🔍+</button>
                <button class="chart-control" data-chart="timeline" data-action="zoom-out" aria-label="Zoom out">🔍-</button>
                <button class="chart-control" data-chart="timeline" data-action="reset" aria-label="Reset view">🔄</button>
                <button class="chart-control" data-chart="timeline" data-action="drill-down" aria-label="Drill down">⬇️</button>
              </div>
            </div>
            <div class="chart-container" id="anomaly-timeline-chart"></div>
          </div>

          <div class="chart-section medium-chart">
            <div class="chart-header">
              <h3>Algorithm Performance Matrix</h3>
              <div class="chart-controls">
                <button class="chart-control" data-chart="performance" data-action="3d-view" aria-label="3D view">🎯</button>
                <button class="chart-control" data-chart="performance" data-action="heatmap" aria-label="Heatmap view">🔥</button>
              </div>
            </div>
            <div class="chart-container" id="performance-matrix-chart"></div>
          </div>

          <div class="chart-section medium-chart">
            <div class="chart-header">
              <h3>Feature Importance Network</h3>
              <div class="chart-controls">
                <button class="chart-control" data-chart="network" data-action="cluster" aria-label="Cluster view">🕸️</button>
                <button class="chart-control" data-chart="network" data-action="hierarchy" aria-label="Hierarchy view">🌳</button>
              </div>
            </div>
            <div class="chart-container" id="feature-network-chart"></div>
          </div>

          <div class="chart-section large-chart">
            <div class="chart-header">
              <h3>Multi-dimensional Anomaly Explorer</h3>
              <div class="chart-controls">
                <button class="chart-control" data-chart="explorer" data-action="pca" aria-label="PCA view">📊</button>
                <button class="chart-control" data-chart="explorer" data-action="tsne" aria-label="t-SNE view">🎨</button>
                <button class="chart-control" data-chart="explorer" data-action="parallel" aria-label="Parallel coordinates">📈</button>
              </div>
            </div>
            <div class="chart-container" id="multidim-explorer-chart"></div>
          </div>

          <!-- Real-time Stream -->
          <div class="chart-section full-width">
            <div class="chart-header">
              <h3>Real-time Anomaly Stream</h3>
              <div class="stream-controls">
                <button id="pause-stream" class="btn-base btn-secondary">⏸️ Pause</button>
                <button id="clear-stream" class="btn-base btn-secondary">🗑️ Clear</button>
                <div class="stream-speed">
                  <label>Speed:</label>
                  <input type="range" id="stream-speed" min="1" max="10" value="5">
                  <span id="speed-value">5x</span>
                </div>
              </div>
            </div>
            <div class="chart-container stream-container" id="realtime-stream-chart"></div>
          </div>
        </div>

        <!-- Drill-down Modal -->
        <div class="drill-down-modal" id="drill-down-modal" style="display: none;">
          <div class="modal-content">
            <div class="modal-header">
              <h3>Detailed Analysis</h3>
              <button class="modal-close" id="close-drill-down">&times;</button>
            </div>
            <div class="modal-body">
              <div class="drill-down-navigation">
                <button id="drill-back" class="btn-base btn-secondary">← Back</button>
                <div class="breadcrumb"></div>
              </div>
              <div class="drill-down-content" id="drill-down-content"></div>
            </div>
          </div>
        </div>

        <!-- Collaboration Panel -->
        <div class="collaboration-panel" id="collaboration-panel" style="display: none;">
          <div class="panel-header">
            <h3>Collaboration</h3>
            <button id="toggle-collaboration" class="btn-base btn-ghost">👥</button>
          </div>
          <div class="panel-content">
            <div class="active-users"></div>
            <div class="shared-annotations"></div>
            <div class="collaboration-chat"></div>
          </div>
        </div>
      </div>
    `;

    this.attachLayoutEvents();
  }

  attachLayoutEvents() {
    // Time range selector
    document.getElementById('time-range-select').addEventListener('change', (e) => {
      this.state.selectedTimeRange = e.target.value;
      this.updateAllCharts();
    });

    // Real-time toggle
    document.getElementById('toggle-realtime').addEventListener('click', () => {
      this.toggleRealTime();
    });

    // Fullscreen toggle
    document.getElementById('fullscreen-toggle').addEventListener('click', () => {
      this.toggleFullscreen();
    });

    // Export dashboard
    document.getElementById('export-dashboard').addEventListener('click', () => {
      this.exportDashboard();
    });

    // Filter controls
    document.getElementById('toggle-filters').addEventListener('click', () => {
      this.toggleFiltersPanel();
    });

    document.getElementById('apply-filters').addEventListener('click', () => {
      this.applyFilters();
    });

    document.getElementById('reset-filters').addEventListener('click', () => {
      this.resetFilters();
    });

    // Chart controls
    document.querySelectorAll('.chart-control').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const chart = e.target.dataset.chart;
        const action = e.target.dataset.action;
        this.handleChartAction(chart, action);
      });
    });

    // Stream controls
    document.getElementById('pause-stream').addEventListener('click', () => {
      this.toggleStreamPause();
    });

    document.getElementById('clear-stream').addEventListener('click', () => {
      this.clearStream();
    });

    document.getElementById('stream-speed').addEventListener('input', (e) => {
      this.setStreamSpeed(e.target.value);
    });

    // Modal controls
    document.getElementById('close-drill-down').addEventListener('click', () => {
      this.closeDrillDown();
    });

    document.getElementById('drill-back').addEventListener('click', () => {
      this.drillBack();
    });
  }

  async initializeCharts() {
    // Initialize all interactive charts
    await Promise.all([
      this.initAnomalyTimelineChart(),
      this.initPerformanceMatrixChart(),
      this.initFeatureNetworkChart(),
      this.initMultidimExplorerChart(),
      this.initRealTimeStreamChart()
    ]);
  }

  async initAnomalyTimelineChart() {
    const container = document.getElementById('anomaly-timeline-chart');
    const width = container.clientWidth;
    const height = 400;

    // Create SVG with advanced interactions
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'timeline-chart');

    // Add zoom and brush capabilities
    const zoom = d3.zoom()
      .scaleExtent([1, 20])
      .on('zoom', (event) => this.handleTimelineZoom(event));

    svg.call(zoom);

    // Create timeline with drill-down points
    this.charts.set('timeline', {
      svg,
      container,
      zoom,
      width,
      height,
      data: []
    });

    this.setupTimelineInteractions();
  }

  async initPerformanceMatrixChart() {
    const container = document.getElementById('performance-matrix-chart');
    
    // Initialize ECharts with 3D capabilities
    const chart = echarts.init(container, null, {
      renderer: this.options.enableWebGL ? 'canvas' : 'svg'
    });

    const option = {
      title: {
        text: 'Algorithm Performance Matrix',
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: (params) => this.formatPerformanceTooltip(params)
      },
      visualMap: {
        min: 0,
        max: 1,
        calculable: true,
        realtime: false,
        inRange: {
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffcc', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        }
      },
      series: [{
        type: 'heatmap',
        data: this.generatePerformanceData(),
        emphasis: {
          itemStyle: {
            borderColor: '#333',
            borderWidth: 1
          }
        },
        progressive: 1000,
        animation: true
      }]
    };

    chart.setOption(option);
    
    // Add click handler for drill-down
    chart.on('click', (params) => {
      this.handlePerformanceDrillDown(params);
    });

    this.charts.set('performance', {
      chart,
      container,
      option
    });
  }

  async initFeatureNetworkChart() {
    const container = document.getElementById('feature-network-chart');
    
    // Create D3 force-directed network
    const width = container.clientWidth;
    const height = 300;

    const svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    // Generate network data
    const networkData = this.generateNetworkData();

    // Create force simulation
    const simulation = d3.forceSimulation(networkData.nodes)
      .force('link', d3.forceLink(networkData.links).id(d => d.id).distance(50))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2));

    // Add links
    const link = svg.append('g')
      .selectAll('line')
      .data(networkData.links)
      .enter().append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => Math.sqrt(d.value));

    // Add nodes
    const node = svg.append('g')
      .selectAll('circle')
      .data(networkData.nodes)
      .enter().append('circle')
      .attr('r', d => d.importance * 10)
      .attr('fill', d => this.getNodeColor(d.type))
      .call(this.createDragHandler(simulation));

    // Add labels
    const label = svg.append('g')
      .selectAll('text')
      .data(networkData.nodes)
      .enter().append('text')
      .text(d => d.name)
      .attr('font-size', 10)
      .attr('dx', 12)
      .attr('dy', 4);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);

      label
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    });

    this.charts.set('network', {
      svg,
      simulation,
      nodes: networkData.nodes,
      links: networkData.links,
      width,
      height
    });
  }

  async initMultidimExplorerChart() {
    const container = document.getElementById('multidim-explorer-chart');
    
    // Create interactive visualization component
    const explorer = new InteractiveDataVisualization(container, {
      width: container.clientWidth,
      height: 400,
      enableZoom: true,
      enableBrush: true,
      enableTooltips: true,
      theme: document.body.classList.contains('theme-dark') ? 'dark' : 'light'
    });

    // Load sample multidimensional data
    const multidimData = this.generateMultidimensionalData();
    explorer.setData(multidimData);

    this.charts.set('explorer', {
      component: explorer,
      container,
      data: multidimData
    });
  }

  async initRealTimeStreamChart() {
    const container = document.getElementById('realtime-stream-chart');
    
    // Create streaming chart with WebGL for performance
    const chart = echarts.init(container, null, {
      renderer: 'canvas'
    });

    const option = {
      title: {
        text: 'Real-time Anomaly Detection Stream'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['Normal', 'Anomaly', 'Alert Threshold']
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'time',
        splitLine: {
          show: false
        }
      },
      yAxis: {
        type: 'value',
        boundaryGap: [0, '100%'],
        splitLine: {
          show: false
        }
      },
      series: [
        {
          name: 'Normal',
          type: 'line',
          showSymbol: false,
          data: [],
          lineStyle: {
            color: '#10B981'
          }
        },
        {
          name: 'Anomaly',
          type: 'scatter',
          data: [],
          symbolSize: 8,
          itemStyle: {
            color: '#EF4444'
          }
        },
        {
          name: 'Alert Threshold',
          type: 'line',
          data: [],
          lineStyle: {
            color: '#F59E0B',
            type: 'dashed'
          },
          markLine: {
            data: [{
              yAxis: 0.8,
              lineStyle: { color: '#F59E0B' }
            }]
          }
        }
      ]
    };

    chart.setOption(option);
    
    this.charts.set('stream', {
      chart,
      container,
      option,
      data: {
        normal: [],
        anomaly: [],
        threshold: []
      },
      paused: false
    });

    // Start real-time updates
    this.startStreamingUpdates();
  }

  setupTimelineInteractions() {
    const timelineChart = this.charts.get('timeline');
    
    // Add brush for time selection
    const brush = d3.brushX()
      .extent([[0, 0], [timelineChart.width, timelineChart.height]])
      .on('end', (event) => {
        if (event.selection) {
          const [start, end] = event.selection;
          this.handleTimeRangeSelection(start, end);
        }
      });

    timelineChart.svg.append('g')
      .attr('class', 'brush')
      .call(brush);

    // Add double-click for drill-down
    timelineChart.svg.on('dblclick', (event) => {
      const coordinates = d3.pointer(event);
      this.handleTimelineDrillDown(coordinates);
    });
  }

  async setupInteractions() {
    // Cross-chart interactions
    this.setupCrossChartInteractions();
    
    // Advanced filtering
    this.setupAdvancedFiltering();
    
    // Collaborative features
    this.setupCollaborativeInteractions();
  }

  setupCrossChartInteractions() {
    // When timeline selection changes, update other charts
    this.interactions.set('timeline-selection', (timeRange) => {
      this.updateChartsForTimeRange(timeRange);
    });

    // When performance matrix is clicked, highlight related features
    this.interactions.set('performance-highlight', (algorithmId) => {
      this.highlightRelatedFeatures(algorithmId);
    });

    // When network node is selected, filter other visualizations
    this.interactions.set('network-filter', (featureId) => {
      this.filterByFeature(featureId);
    });
  }

  setupAdvancedFiltering() {
    // Multi-dimensional filter system
    const scoreMinSlider = document.getElementById('score-min');
    const scoreMaxSlider = document.getElementById('score-max');
    const scoreRangeText = document.getElementById('score-range-text');

    [scoreMinSlider, scoreMaxSlider].forEach(slider => {
      slider.addEventListener('input', () => {
        const min = parseFloat(scoreMinSlider.value);
        const max = parseFloat(scoreMaxSlider.value);
        
        // Ensure min doesn't exceed max
        if (min > max) {
          if (slider === scoreMinSlider) {
            scoreMaxSlider.value = min;
          } else {
            scoreMinSlider.value = max;
          }
        }
        
        scoreRangeText.textContent = `${scoreMinSlider.value} - ${scoreMaxSlider.value}`;
        this.updateFilters();
      });
    });
  }

  setupCollaborativeInteractions() {
    if (!this.options.enableCollaboration) return;

    // Real-time cursor sharing
    document.addEventListener('mousemove', (e) => {
      if (this.state.collaborationMode) {
        this.broadcastCursorPosition(e.clientX, e.clientY);
      }
    });

    // Annotation system
    document.addEventListener('dblclick', (e) => {
      if (this.state.collaborationMode && e.ctrlKey) {
        this.createAnnotation(e.clientX, e.clientY);
      }
    });
  }

  async connectRealTime() {
    if (!this.options.enableRealTime) return;

    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`;
      
      this.websocket = new WebSocket(wsUrl);
      
      this.websocket.onopen = () => {
        console.log('[Dashboard] WebSocket connected');
        this.updateConnectionStatus(true);
      };
      
      this.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleRealTimeData(data);
      };
      
      this.websocket.onclose = () => {
        console.log('[Dashboard] WebSocket disconnected');
        this.updateConnectionStatus(false);
        // Attempt reconnection
        setTimeout(() => this.connectRealTime(), 5000);
      };
      
      this.websocket.onerror = (error) => {
        console.error('[Dashboard] WebSocket error:', error);
      };
    } catch (error) {
      console.error('[Dashboard] Failed to connect WebSocket:', error);
    }
  }

  handleRealTimeData(data) {
    switch (data.type) {
      case 'anomaly_detection':
        this.addAnomalyToStream(data.payload);
        this.updateMetricCards(data.payload);
        break;
      case 'system_metrics':
        this.updateSystemHealth(data.payload);
        break;
      case 'user_cursor':
        this.updateCollaboratorCursor(data.payload);
        break;
      case 'annotation':
        this.addCollaborativeAnnotation(data.payload);
        break;
    }
  }

  addAnomalyToStream(anomalyData) {
    const streamChart = this.charts.get('stream');
    if (!streamChart || streamChart.paused) return;

    const timestamp = new Date(anomalyData.timestamp);
    const isAnomaly = anomalyData.is_anomaly;
    const score = anomalyData.anomaly_score;

    if (isAnomaly) {
      streamChart.data.anomaly.push([timestamp, score]);
    } else {
      streamChart.data.normal.push([timestamp, score]);
    }

    // Keep only recent data points
    const cutoff = Date.now() - (24 * 60 * 60 * 1000); // 24 hours
    streamChart.data.normal = streamChart.data.normal.filter(([time]) => time.getTime() > cutoff);
    streamChart.data.anomaly = streamChart.data.anomaly.filter(([time]) => time.getTime() > cutoff);

    // Update chart
    streamChart.option.series[0].data = streamChart.data.normal;
    streamChart.option.series[1].data = streamChart.data.anomaly;
    streamChart.chart.setOption(streamChart.option);
  }

  updateMetricCards(data) {
    // Update total detections
    const totalCard = document.getElementById('total-detections-card');
    const totalValue = totalCard.querySelector('.value');
    const totalChange = totalCard.querySelector('.change');
    
    // Update anomaly rate
    const rateCard = document.getElementById('anomaly-rate-card');
    const rateValue = rateCard.querySelector('.value');
    const rateChange = rateCard.querySelector('.change');
    
    // Update processing time
    const timeCard = document.getElementById('avg-processing-time-card');
    const timeValue = timeCard.querySelector('.value');
    const timeChange = timeCard.querySelector('.change');

    // Calculate and display metrics with animations
    this.animateMetricUpdate(totalValue, data.total_detections);
    this.animateMetricUpdate(rateValue, `${(data.anomaly_rate * 100).toFixed(1)}%`);
    this.animateMetricUpdate(timeValue, `${data.avg_processing_time}ms`);
  }

  animateMetricUpdate(element, newValue) {
    element.style.transform = 'scale(1.1)';
    element.style.transition = 'transform 0.3s ease';
    
    setTimeout(() => {
      element.textContent = newValue;
      element.style.transform = 'scale(1)';
    }, 150);
  }

  handleChartAction(chartType, action) {
    const chart = this.charts.get(chartType);
    if (!chart) return;

    switch (action) {
      case 'zoom-in':
        this.zoomChart(chart, 1.5);
        break;
      case 'zoom-out':
        this.zoomChart(chart, 0.67);
        break;
      case 'reset':
        this.resetChart(chart);
        break;
      case 'drill-down':
        this.initiateDrillDown(chartType);
        break;
      case '3d-view':
        this.toggle3DView(chart);
        break;
      case 'heatmap':
        this.toggleHeatmapView(chart);
        break;
      case 'cluster':
        this.toggleClusterView(chart);
        break;
      case 'hierarchy':
        this.toggleHierarchyView(chart);
        break;
      case 'pca':
        this.switchToPCAView(chart);
        break;
      case 'tsne':
        this.switchToTSNEView(chart);
        break;
      case 'parallel':
        this.switchToParallelView(chart);
        break;
    }
  }

  initiateDrillDown(chartType) {
    const modal = document.getElementById('drill-down-modal');
    const content = document.getElementById('drill-down-content');
    
    // Add to drill-down stack
    this.state.drillDownStack.push({
      chartType,
      timestamp: Date.now(),
      filters: { ...this.state.filters }
    });

    // Show detailed analysis
    this.showDrillDownAnalysis(chartType, content);
    modal.style.display = 'block';
  }

  showDrillDownAnalysis(chartType, container) {
    switch (chartType) {
      case 'timeline':
        this.renderTimelineDrillDown(container);
        break;
      case 'performance':
        this.renderPerformanceDrillDown(container);
        break;
      case 'network':
        this.renderNetworkDrillDown(container);
        break;
      case 'explorer':
        this.renderExplorerDrillDown(container);
        break;
    }
  }

  renderTimelineDrillDown(container) {
    container.innerHTML = `
      <div class="drill-down-timeline">
        <div class="timeline-detail-controls">
          <label>Granularity:</label>
          <select id="timeline-granularity">
            <option value="minute">Per Minute</option>
            <option value="hour">Per Hour</option>
            <option value="day">Per Day</option>
          </select>
        </div>
        <div class="timeline-detail-chart" id="timeline-detail-chart"></div>
        <div class="timeline-detail-stats">
          <div class="stat-card">
            <h4>Peak Anomaly Period</h4>
            <p id="peak-period">-</p>
          </div>
          <div class="stat-card">
            <h4>Most Affected Features</h4>
            <ul id="affected-features"></ul>
          </div>
          <div class="stat-card">
            <h4>Algorithm Performance</h4>
            <div id="algorithm-performance"></div>
          </div>
        </div>
      </div>
    `;

    // Initialize detailed timeline chart
    this.initDetailedTimelineChart();
  }

  setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'f':
            e.preventDefault();
            this.toggleFiltersPanel();
            break;
          case 'r':
            e.preventDefault();
            this.toggleRealTime();
            break;
          case 'e':
            e.preventDefault();
            this.exportDashboard();
            break;
          case 'd':
            e.preventDefault();
            this.resetDashboard();
            break;
        }
      }
      
      if (e.key === 'Escape') {
        this.closeDrillDown();
        this.closeModalDialogs();
      }
    });
  }

  async loadInitialData() {
    try {
      // Load data from API or offline storage
      const response = await fetch('/api/dashboard/initial-data');
      if (response.ok) {
        const data = await response.json();
        this.populateChartsWithData(data);
      } else {
        // Fallback to offline data
        await this.loadOfflineData();
      }
    } catch (error) {
      console.error('[Dashboard] Failed to load initial data:', error);
      await this.loadOfflineData();
    }
  }

  async loadOfflineData() {
    if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'GET_OFFLINE_DASHBOARD_DATA'
      });
    }
  }

  // Utility methods for data generation (for demo purposes)
  generatePerformanceData() {
    const algorithms = ['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM', 'ECOD', 'COPOD'];
    const metrics = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Processing Time'];
    const data = [];

    algorithms.forEach((algo, i) => {
      metrics.forEach((metric, j) => {
        data.push([i, j, Math.random()]);
      });
    });

    return data;
  }

  generateNetworkData() {
    const nodes = [];
    const links = [];
    
    // Generate feature nodes
    for (let i = 0; i < 20; i++) {
      nodes.push({
        id: `feature_${i}`,
        name: `Feature ${i}`,
        type: 'feature',
        importance: Math.random(),
        anomaly_score: Math.random()
      });
    }

    // Generate algorithm nodes
    const algorithms = ['IsolationForest', 'LOF', 'OneClassSVM'];
    algorithms.forEach(algo => {
      nodes.push({
        id: algo,
        name: algo,
        type: 'algorithm',
        importance: 0.8 + Math.random() * 0.2
      });
    });

    // Generate links
    nodes.forEach(node => {
      if (node.type === 'feature') {
        algorithms.forEach(algo => {
          if (Math.random() > 0.6) {
            links.push({
              source: node.id,
              target: algo,
              value: Math.random()
            });
          }
        });
      }
    });

    return { nodes, links };
  }

  generateMultidimensionalData() {
    const data = [];
    for (let i = 0; i < 1000; i++) {
      data.push({
        id: i,
        timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000),
        value: Math.random() * 100,
        anomaly_score: Math.random(),
        feature_1: Math.random() * 50,
        feature_2: Math.random() * 50,
        feature_3: Math.random() * 50,
        is_anomaly: Math.random() > 0.9,
        algorithm: ['IsolationForest', 'LOF', 'OneClassSVM'][Math.floor(Math.random() * 3)],
        confidence: Math.random()
      });
    }
    return data;
  }

  // Event handlers and utility methods
  toggleRealTime() {
    this.state.realTimeEnabled = !this.state.realTimeEnabled;
    
    const indicator = document.querySelector('.real-time-indicator');
    const button = document.getElementById('toggle-realtime');
    
    if (this.state.realTimeEnabled) {
      indicator.classList.add('active');
      indicator.classList.remove('inactive');
      button.innerHTML = '<span class="realtime-icon">📡</span>';
      this.connectRealTime();
    } else {
      indicator.classList.add('inactive');
      indicator.classList.remove('active');
      button.innerHTML = '<span class="realtime-icon">⏸️</span>';
      if (this.websocket) {
        this.websocket.close();
      }
    }
  }

  toggleFiltersPanel() {
    const panel = document.getElementById('filters-panel');
    panel.classList.toggle('expanded');
  }

  async exportDashboard() {
    const dashboardData = {
      timestamp: new Date().toISOString(),
      charts: {},
      state: this.state,
      filters: this.state.filters
    };

    // Export chart data
    this.charts.forEach((chart, key) => {
      dashboardData.charts[key] = this.serializeChart(chart);
    });

    // Create and download JSON file
    const blob = new Blob([JSON.stringify(dashboardData, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `pynomaly-dashboard-${Date.now()}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
  }

  serializeChart(chart) {
    // Serialize chart configuration and data for export
    if (chart.chart && chart.chart.getOption) {
      return {
        type: 'echarts',
        option: chart.chart.getOption()
      };
    } else if (chart.svg) {
      return {
        type: 'd3',
        // Serialize D3 chart data
        data: chart.data || {}
      };
    }
    return {};
  }

  destroy() {
    // Clean up all resources
    if (this.websocket) {
      this.websocket.close();
    }
    
    this.charts.forEach(chart => {
      if (chart.chart && chart.chart.dispose) {
        chart.chart.dispose();
      }
    });
    
    this.charts.clear();
    this.interactions.clear();
    this.realTimeData.clear();
    this.eventListeners.clear();
  }
}

// Initialize when DOM is ready
if (typeof window !== 'undefined') {
  window.AdvancedInteractiveDashboard = AdvancedInteractiveDashboard;
}

export default AdvancedInteractiveDashboard;