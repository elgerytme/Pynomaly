/**
 * Advanced Interactive Dashboard
 * Combines D3.js and ECharts with real-time data integration and drill-down capabilities
 */

class AdvancedInteractiveDashboard {
  constructor(config = {}) {
    this.config = {
      apiBaseUrl: '/api/visualization',
      updateInterval: 30000, // 30 seconds
      maxDataPoints: 1000,
      enableRealtime: true,
      enableDrillDown: true,
      enableExport: true,
      ...config
    };
    
    this.charts = new Map();
    this.dataCache = new Map();
    this.updateTimers = new Map();
    this.interactions = new Map();
    this.selectedData = new Set();
    
    this.init();
  }

  async init() {
    await this.loadLibraries();
    this.setupEventListeners();
    this.setupRealTimeUpdates();
    this.createDashboardLayout();
  }

  async loadLibraries() {
    // Ensure D3.js and ECharts are loaded
    if (typeof d3 === 'undefined') {
      await this.loadScript('https://d3js.org/d3.v7.min.js');
    }
    
    if (typeof echarts === 'undefined') {
      await this.loadScript('https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js');
    }
  }

  loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = src;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  createDashboardLayout() {
    const container = document.getElementById('advanced-dashboard') || this.createDashboardContainer();
    
    container.innerHTML = `
      <div class="dashboard-header">
        <div class="dashboard-controls">
          <button id="toggle-realtime" class="btn btn-primary">
            <i class="fas fa-play"></i> Real-time Updates
          </button>
          <button id="export-dashboard" class="btn btn-secondary">
            <i class="fas fa-download"></i> Export
          </button>
          <button id="customize-dashboard" class="btn btn-secondary">
            <i class="fas fa-cog"></i> Customize
          </button>
          <div class="data-selector">
            <select id="detector-selector" class="form-select">
              <option value="">Select Detector...</option>
            </select>
            <select id="dataset-selector" class="form-select">
              <option value="">Select Dataset...</option>
            </select>
          </div>
        </div>
      </div>
      
      <div class="dashboard-grid">
        <!-- Main Time Series Chart -->
        <div class="chart-container large" id="main-timeseries-container">
          <div class="chart-header">
            <h3>Real-time Anomaly Detection</h3>
            <div class="chart-controls">
              <button class="btn-small" onclick="dashboard.zoomChart('main-timeseries')">
                <i class="fas fa-search-plus"></i>
              </button>
              <button class="btn-small" onclick="dashboard.exportChart('main-timeseries')">
                <i class="fas fa-download"></i>
              </button>
            </div>
          </div>
          <div id="main-timeseries" class="chart-content"></div>
        </div>

        <!-- Scatter Plot with Drill-down -->
        <div class="chart-container medium" id="scatter-container">
          <div class="chart-header">
            <h3>Feature Space Analysis</h3>
            <div class="feature-selectors">
              <select id="feature-x-selector" class="form-select-small">
                <option value="feature1">Feature 1</option>
              </select>
              <select id="feature-y-selector" class="form-select-small">
                <option value="feature2">Feature 2</option>
              </select>
            </div>
          </div>
          <div id="feature-scatter" class="chart-content"></div>
        </div>

        <!-- Correlation Heatmap -->
        <div class="chart-container medium" id="correlation-container">
          <div class="chart-header">
            <h3>Feature Correlations</h3>
          </div>
          <div id="correlation-heatmap" class="chart-content"></div>
        </div>

        <!-- Feature Importance -->
        <div class="chart-container small" id="importance-container">
          <div class="chart-header">
            <h3>Feature Importance</h3>
          </div>
          <div id="feature-importance" class="chart-content"></div>
        </div>

        <!-- Detector Comparison -->
        <div class="chart-container small" id="comparison-container">
          <div class="chart-header">
            <h3>Detector Performance</h3>
          </div>
          <div id="detector-comparison" class="chart-content"></div>
        </div>

        <!-- Distribution Chart -->
        <div class="chart-container small" id="distribution-container">
          <div class="chart-header">
            <h3>Score Distribution</h3>
          </div>
          <div id="score-distribution" class="chart-content"></div>
        </div>
      </div>

      <!-- Drill-down Modal -->
      <div id="drilldown-modal" class="modal">
        <div class="modal-content">
          <div class="modal-header">
            <h3>Detailed Analysis</h3>
            <button class="modal-close">&times;</button>
          </div>
          <div class="modal-body">
            <div id="drilldown-chart"></div>
            <div id="drilldown-details"></div>
          </div>
        </div>
      </div>

      <!-- Customization Panel -->
      <div id="customization-panel" class="side-panel">
        <div class="panel-header">
          <h3>Dashboard Customization</h3>
          <button class="panel-close">&times;</button>
        </div>
        <div class="panel-body">
          <div class="customization-options">
            <div class="option-group">
              <h4>Chart Types</h4>
              <div class="chart-type-selector">
                <!-- Chart type options will be populated here -->
              </div>
            </div>
            <div class="option-group">
              <h4>Data Sources</h4>
              <div class="data-source-selector">
                <!-- Data source options will be populated here -->
              </div>
            </div>
            <div class="option-group">
              <h4>Update Frequency</h4>
              <select id="update-frequency">
                <option value="5000">5 seconds</option>
                <option value="15000">15 seconds</option>
                <option value="30000" selected>30 seconds</option>
                <option value="60000">1 minute</option>
                <option value="300000">5 minutes</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    `;

    this.setupDashboardControls();
    this.loadInitialData();
  }

  createDashboardContainer() {
    const container = document.createElement('div');
    container.id = 'advanced-dashboard';
    container.className = 'advanced-dashboard';
    document.body.appendChild(container);
    return container;
  }

  setupEventListeners() {
    // Global event listeners for chart interactions
    document.addEventListener('pointClicked', (event) => {
      this.handlePointClick(event.detail);
    });

    document.addEventListener('chartZoomed', (event) => {
      this.handleChartZoom(event.detail);
    });

    document.addEventListener('dataSelected', (event) => {
      this.handleDataSelection(event.detail);
    });

    document.addEventListener('anomalyPointClicked', (event) => {
      this.handleAnomalyClick(event.detail);
    });

    // Chart linking for cross-filtering
    document.addEventListener('chartBrushed', (event) => {
      this.handleChartBrush(event.detail);
    });
  }

  setupDashboardControls() {
    // Real-time toggle
    document.getElementById('toggle-realtime')?.addEventListener('click', () => {
      this.toggleRealTimeUpdates();
    });

    // Export functionality
    document.getElementById('export-dashboard')?.addEventListener('click', () => {
      this.exportDashboard();
    });

    // Customization panel
    document.getElementById('customize-dashboard')?.addEventListener('click', () => {
      this.showCustomizationPanel();
    });

    // Data selectors
    document.getElementById('detector-selector')?.addEventListener('change', (e) => {
      this.onDetectorChange(e.target.value);
    });

    document.getElementById('dataset-selector')?.addEventListener('change', (e) => {
      this.onDatasetChange(e.target.value);
    });

    // Feature selectors
    document.getElementById('feature-x-selector')?.addEventListener('change', (e) => {
      this.updateScatterPlot();
    });

    document.getElementById('feature-y-selector')?.addEventListener('change', (e) => {
      this.updateScatterPlot();
    });
  }

  async loadInitialData() {
    try {
      // Load available detectors and datasets
      await this.loadDetectorsList();
      await this.loadDatasetsList();
      
      // Load default charts
      if (this.config.enableRealtime) {
        await this.createRealTimeChart();
      }
      
      await this.createFeatureScatterPlot();
      await this.createCorrelationHeatmap();
      await this.createFeatureImportanceChart();
      await this.createDetectorComparison();
      await this.createDistributionChart();
      
    } catch (error) {
      console.error('Error loading initial data:', error);
      this.showErrorMessage('Failed to load dashboard data. Using sample data.');
      this.loadSampleData();
    }
  }

  async loadDetectorsList() {
    try {
      const response = await fetch('/api/detectors');
      const detectors = await response.json();
      
      const selector = document.getElementById('detector-selector');
      selector.innerHTML = '<option value="">Select Detector...</option>';
      
      detectors.forEach(detector => {
        const option = document.createElement('option');
        option.value = detector.id;
        option.textContent = `${detector.name} (${detector.algorithm_name})`;
        selector.appendChild(option);
      });
      
      // Auto-select first detector if available
      if (detectors.length > 0) {
        selector.value = detectors[0].id;
        this.selectedDetectorId = detectors[0].id;
      }
    } catch (error) {
      console.error('Error loading detectors list:', error);
    }
  }

  async loadDatasetsList() {
    try {
      const response = await fetch('/api/datasets');
      const datasets = await response.json();
      
      const selector = document.getElementById('dataset-selector');
      selector.innerHTML = '<option value="">Select Dataset...</option>';
      
      datasets.forEach(dataset => {
        const option = document.createElement('option');
        option.value = dataset.id;
        option.textContent = dataset.name;
        selector.appendChild(option);
      });
      
      // Auto-select first dataset if available
      if (datasets.length > 0) {
        selector.value = datasets[0].id;
        this.selectedDatasetId = datasets[0].id;
      }
    } catch (error) {
      console.error('Error loading datasets list:', error);
    }
  }

  async createRealTimeChart() {
    if (!this.selectedDetectorId) return;

    try {
      const response = await fetch(`${this.config.apiBaseUrl}/timeseries/${this.selectedDetectorId}`);
      const data = await response.json();

      // Use ECharts for real-time streaming
      if (window.echartsManager) {
        const chart = window.echartsManager.createStreamingChart('main-timeseries');
        this.charts.set('main-timeseries', chart);
        
        // Setup real-time updates
        if (this.config.enableRealtime) {
          this.setupRealTimeUpdates('main-timeseries');
        }
      }
    } catch (error) {
      console.error('Error creating real-time chart:', error);
    }
  }

  async createFeatureScatterPlot() {
    if (!this.selectedDatasetId) return;

    try {
      const featureX = document.getElementById('feature-x-selector')?.value || 'feature1';
      const featureY = document.getElementById('feature-y-selector')?.value || 'feature2';
      
      const response = await fetch(
        `${this.config.apiBaseUrl}/scatter/${this.selectedDatasetId}?feature_x=${featureX}&feature_y=${featureY}`
      );
      const data = await response.json();

      // Use D3.js for interactive scatter plot
      if (window.d3Charts) {
        const scatterData = data.points.map(point => ({
          x: point.x,
          y: point.y,
          type: point.type,
          id: point.id,
          score: point.score
        }));

        window.d3Charts.createAnomalyScatterPlot('feature-scatter', scatterData, {
          xLabel: featureX,
          yLabel: featureY,
          enableDrillDown: this.config.enableDrillDown
        });

        this.charts.set('feature-scatter', { type: 'd3', data: scatterData });
      }
    } catch (error) {
      console.error('Error creating scatter plot:', error);
    }
  }

  async createCorrelationHeatmap() {
    if (!this.selectedDatasetId) return;

    try {
      const response = await fetch(`${this.config.apiBaseUrl}/correlation/${this.selectedDatasetId}`);
      const data = await response.json();

      // Use ECharts for correlation heatmap
      if (window.echartsManager) {
        const chart = window.echartsManager.createCorrelationHeatmap('correlation-heatmap', data);
        this.charts.set('correlation-heatmap', chart);
      }
    } catch (error) {
      console.error('Error creating correlation heatmap:', error);
    }
  }

  async createFeatureImportanceChart() {
    if (!this.selectedDetectorId) return;

    try {
      const response = await fetch(`${this.config.apiBaseUrl}/feature-importance/${this.selectedDetectorId}`);
      const data = await response.json();

      // Use D3.js for feature importance
      if (window.d3Charts) {
        const importanceData = data.features.map((feature, index) => ({
          feature: feature,
          importance: data.importance_scores[index]
        }));

        window.d3Charts.createFeatureImportanceChart('feature-importance', importanceData);
        this.charts.set('feature-importance', { type: 'd3', data: importanceData });
      }
    } catch (error) {
      console.error('Error creating feature importance chart:', error);
    }
  }

  async createDetectorComparison() {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/detector-comparison`);
      const data = await response.json();

      // Use ECharts for comparison
      if (window.echartsManager) {
        const chart = window.echartsManager.createDetectorComparison('detector-comparison', data);
        this.charts.set('detector-comparison', chart);
      }
    } catch (error) {
      console.error('Error creating detector comparison:', error);
    }
  }

  async createDistributionChart() {
    if (!this.selectedDetectorId) return;

    try {
      const response = await fetch(`${this.config.apiBaseUrl}/anomaly-distribution/${this.selectedDetectorId}`);
      const data = await response.json();

      // Use ECharts for distribution histogram
      const chart = echarts.init(document.getElementById('score-distribution'));
      
      const option = {
        title: {
          text: 'Anomaly Score Distribution',
          left: 'center',
          textStyle: { fontSize: 14 }
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' }
        },
        legend: {
          data: ['Normal', 'Anomalies'],
          top: '10%'
        },
        xAxis: {
          type: 'category',
          data: data.bins.map(bin => bin.toFixed(2))
        },
        yAxis: {
          type: 'value'
        },
        series: [
          {
            name: 'Normal',
            type: 'bar',
            data: data.normal_counts,
            itemStyle: { color: '#2563eb' }
          },
          {
            name: 'Anomalies',
            type: 'bar',
            data: data.anomaly_counts,
            itemStyle: { color: '#dc2626' }
          }
        ]
      };

      chart.setOption(option);
      this.charts.set('score-distribution', chart);
    } catch (error) {
      console.error('Error creating distribution chart:', error);
    }
  }

  setupRealTimeUpdates(chartId) {
    if (this.updateTimers.has(chartId)) {
      clearInterval(this.updateTimers.get(chartId));
    }

    const timer = setInterval(() => {
      this.updateRealTimeChart(chartId);
    }, this.config.updateInterval);

    this.updateTimers.set(chartId, timer);
  }

  async updateRealTimeChart(chartId) {
    if (chartId === 'main-timeseries' && window.echartsManager) {
      // Simulate new data point (in production, this would fetch from real-time endpoint)
      const newValue = Math.random() * 100;
      const isAnomaly = Math.random() < 0.05; // 5% chance of anomaly
      
      window.echartsManager.updateStreamingChart('main-timeseries', newValue, isAnomaly);
    }
  }

  handlePointClick(detail) {
    if (this.config.enableDrillDown) {
      this.showDrillDownModal(detail);
    }
  }

  handleChartZoom(detail) {
    // Sync zoom across linked charts
    console.log('Chart zoomed:', detail);
    
    // Update other time-based charts with the same domain
    this.charts.forEach((chart, chartId) => {
      if (chartId !== detail.chart && this.isTimeBased(chartId)) {
        this.syncChartZoom(chartId, detail.domain);
      }
    });
  }

  handleDataSelection(detail) {
    // Cross-filter other charts based on selection
    console.log('Data selected:', detail);
    
    // Update selected data set
    this.selectedData.clear();
    if (detail.areas) {
      detail.areas.forEach(area => {
        // Add selected data points to set
        this.selectedData.add(area);
      });
    }
    
    // Update other charts to highlight selected data
    this.updateCrossFilter();
  }

  handleAnomalyClick(detail) {
    // Show detailed anomaly information
    this.showAnomalyDetails(detail);
  }

  handleChartBrush(detail) {
    // Handle brush selection for cross-filtering
    this.updateCrossFilter(detail);
  }

  showDrillDownModal(detail) {
    const modal = document.getElementById('drilldown-modal');
    const chartContainer = document.getElementById('drilldown-chart');
    const detailsContainer = document.getElementById('drilldown-details');
    
    // Clear previous content
    chartContainer.innerHTML = '';
    detailsContainer.innerHTML = '';
    
    // Create detailed view based on the clicked point
    this.createDrillDownChart(chartContainer, detail);
    this.createDrillDownDetails(detailsContainer, detail);
    
    // Show modal
    modal.style.display = 'flex';
    
    // Setup close functionality
    modal.querySelector('.modal-close').onclick = () => {
      modal.style.display = 'none';
    };
  }

  createDrillDownChart(container, detail) {
    // Create a detailed chart for the selected data point
    if (detail.chart === 'scatter') {
      // Show time series for this specific data point
      this.createPointTimeSeriesChart(container, detail.data);
    } else if (detail.chart === 'timeseries') {
      // Show feature breakdown for this time point
      this.createFeatureBreakdownChart(container, detail.data);
    }
  }

  createDrillDownDetails(container, detail) {
    container.innerHTML = `
      <div class="detail-section">
        <h4>Data Point Details</h4>
        <table class="detail-table">
          <tr><td><strong>ID:</strong></td><td>${detail.data.id}</td></tr>
          <tr><td><strong>Type:</strong></td><td>${detail.data.type}</td></tr>
          <tr><td><strong>Score:</strong></td><td>${detail.data.score?.toFixed(4) || 'N/A'}</td></tr>
          <tr><td><strong>Confidence:</strong></td><td>${detail.data.confidence?.toFixed(4) || 'N/A'}</td></tr>
        </table>
      </div>
      
      <div class="detail-section">
        <h4>Actions</h4>
        <button class="btn btn-primary" onclick="dashboard.investigateAnomaly('${detail.data.id}')">
          Investigate Further
        </button>
        <button class="btn btn-secondary" onclick="dashboard.flagAsNormal('${detail.data.id}')">
          Flag as Normal
        </button>
      </div>
    `;
  }

  showAnomalyDetails(detail) {
    // Enhanced anomaly details with context
    this.showDrillDownModal({
      chart: 'anomaly',
      data: detail
    });
  }

  updateCrossFilter(filterDetail) {
    // Update all charts to reflect cross-filtering
    this.charts.forEach((chart, chartId) => {
      if (chart.updateFilter) {
        chart.updateFilter(this.selectedData);
      }
    });
  }

  toggleRealTimeUpdates() {
    const button = document.getElementById('toggle-realtime');
    const isActive = button.classList.contains('active');
    
    if (isActive) {
      // Stop real-time updates
      this.updateTimers.forEach(timer => clearInterval(timer));
      this.updateTimers.clear();
      button.innerHTML = '<i class="fas fa-play"></i> Start Real-time';
      button.classList.remove('active');
    } else {
      // Start real-time updates
      this.charts.forEach((chart, chartId) => {
        if (this.isRealTimeChart(chartId)) {
          this.setupRealTimeUpdates(chartId);
        }
      });
      button.innerHTML = '<i class="fas fa-pause"></i> Stop Real-time';
      button.classList.add('active');
    }
  }

  async exportDashboard() {
    const exportData = {
      timestamp: new Date().toISOString(),
      charts: {},
      metadata: {
        selectedDetector: this.selectedDetectorId,
        selectedDataset: this.selectedDatasetId,
        config: this.config
      }
    };

    // Export each chart
    for (const [chartId, chart] of this.charts) {
      try {
        if (window.d3Charts && chart.type === 'd3') {
          exportData.charts[chartId] = await window.d3Charts.exportChart(chartId, 'png');
        } else if (window.echartsManager) {
          exportData.charts[chartId] = window.echartsManager.exportChart(chartId, 'png');
        }
      } catch (error) {
        console.error(`Error exporting chart ${chartId}:`, error);
      }
    }

    // Download as ZIP file
    this.downloadExport(exportData);
  }

  downloadExport(exportData) {
    // Create a downloadable export package
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pynomaly-dashboard-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  showCustomizationPanel() {
    const panel = document.getElementById('customization-panel');
    panel.classList.add('open');
    
    // Setup close functionality
    panel.querySelector('.panel-close').onclick = () => {
      panel.classList.remove('open');
    };
  }

  onDetectorChange(detectorId) {
    this.selectedDetectorId = detectorId;
    this.refreshDetectorDependentCharts();
  }

  onDatasetChange(datasetId) {
    this.selectedDatasetId = datasetId;
    this.refreshDatasetDependentCharts();
  }

  async refreshDetectorDependentCharts() {
    // Refresh charts that depend on detector selection
    await this.createRealTimeChart();
    await this.createFeatureImportanceChart();
    await this.createDistributionChart();
  }

  async refreshDatasetDependentCharts() {
    // Refresh charts that depend on dataset selection
    await this.createFeatureScatterPlot();
    await this.createCorrelationHeatmap();
  }

  updateScatterPlot() {
    this.createFeatureScatterPlot();
  }

  isTimeBased(chartId) {
    return ['main-timeseries'].includes(chartId);
  }

  isRealTimeChart(chartId) {
    return ['main-timeseries'].includes(chartId);
  }

  syncChartZoom(chartId, domain) {
    // Sync zoom across charts
    const chart = this.charts.get(chartId);
    if (chart && chart.updateDomain) {
      chart.updateDomain(domain);
    }
  }

  showErrorMessage(message) {
    // Show error toast or notification
    console.error(message);
    
    // Create error notification
    const notification = document.createElement('div');
    notification.className = 'error-notification';
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      document.body.removeChild(notification);
    }, 5000);
  }

  loadSampleData() {
    // Load sample data when real data is unavailable
    console.log('Loading sample data for dashboard...');
    
    // Create sample charts with mock data
    this.createSampleCharts();
  }

  createSampleCharts() {
    // Implementation for sample charts
    console.log('Creating sample charts...');
  }

  // Cleanup method
  destroy() {
    // Clear all timers
    this.updateTimers.forEach(timer => clearInterval(timer));
    this.updateTimers.clear();
    
    // Dispose all charts
    this.charts.forEach((chart, chartId) => {
      if (chart.dispose) {
        chart.dispose();
      } else if (window.d3Charts) {
        window.d3Charts.destroyChart(chartId);
      }
    });
    
    this.charts.clear();
  }
}

// Global dashboard instance
window.dashboard = null;

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new AdvancedInteractiveDashboard();
  });
} else {
  window.dashboard = new AdvancedInteractiveDashboard();
}

// Handle page unload
window.addEventListener('beforeunload', () => {
  if (window.dashboard) {
    window.dashboard.destroy();
  }
});