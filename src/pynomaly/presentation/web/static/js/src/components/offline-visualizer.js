/**
 * Offline Data Visualizer Component
 * Provides advanced data visualization capabilities using cached data
 */
export class OfflineVisualizer {
  constructor() {
    this.charts = new Map();
    this.datasets = new Map();
    this.results = new Map();
    this.currentDataset = null;
    this.currentResult = null;
    this.isInitialized = false;
    
    this.init();
  }

  async init() {
    await this.loadCachedData();
    this.setupEventListeners();
    this.isInitialized = true;
  }

  /**
   * Load cached data from IndexedDB
   */
  async loadCachedData() {
    try {
      if ('serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          registration.active.postMessage({ type: 'GET_OFFLINE_DATASETS' });
          registration.active.postMessage({ type: 'GET_OFFLINE_RESULTS' });
          
          return new Promise((resolve) => {
            let datasetsLoaded = false;
            let resultsLoaded = false;
            
            navigator.serviceWorker.addEventListener('message', (event) => {
              if (event.data.type === 'OFFLINE_DATASETS') {
                event.data.datasets.forEach(dataset => {
                  this.datasets.set(dataset.id, dataset);
                });
                datasetsLoaded = true;
              } else if (event.data.type === 'OFFLINE_RESULTS') {
                event.data.results.forEach(result => {
                  this.results.set(result.id, result);
                });
                resultsLoaded = true;
              }
              
              if (datasetsLoaded && resultsLoaded) {
                resolve();
              }
            });
          });
        }
      }
    } catch (error) {
      console.error('[OfflineVisualizer] Failed to load cached data:', error);
    }
  }

  /**
   * Setup event listeners
   */
  setupEventListeners() {
    // Dataset selection
    document.addEventListener('change', (event) => {
      if (event.target.matches('.dataset-selector')) {
        this.selectDataset(event.target.value);
      }
    });

    // Result selection
    document.addEventListener('change', (event) => {
      if (event.target.matches('.result-selector')) {
        this.selectResult(event.target.value);
      }
    });

    // Visualization type selection
    document.addEventListener('change', (event) => {
      if (event.target.matches('.viz-type-selector')) {
        this.changeVisualizationType(event.target.value);
      }
    });

    // Export functionality
    document.addEventListener('click', (event) => {
      if (event.target.matches('.export-viz')) {
        this.exportVisualization(event.target.dataset.format);
      }
    });
  }

  /**
   * Select and load a dataset for visualization
   */
  async selectDataset(datasetId) {
    const dataset = this.datasets.get(datasetId);
    if (!dataset) return;

    this.currentDataset = dataset;
    await this.renderDatasetVisualization();
    this.updateResultSelector();
  }

  /**
   * Select and load a result for visualization
   */
  async selectResult(resultId) {
    const result = this.results.get(resultId);
    if (!result) return;

    this.currentResult = result;
    await this.renderResultVisualization();
  }

  /**
   * Render dataset visualization
   */
  async renderDatasetVisualization() {
    if (!this.currentDataset) return;

    const container = document.getElementById('dataset-visualization');
    if (!container) return;

    const data = this.currentDataset.data;
    const features = this.extractFeatures(data);

    // Clear existing charts
    this.clearCharts();

    // Create visualization container
    container.innerHTML = `
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Data Distribution</h3>
            <select class="form-select form-select-sm viz-type-selector" data-target="distribution">
              <option value="histogram">Histogram</option>
              <option value="boxplot">Box Plot</option>
              <option value="violin">Violin Plot</option>
            </select>
          </div>
          <div class="card-body">
            <div id="distribution-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Feature Correlation</h3>
            <select class="form-select form-select-sm viz-type-selector" data-target="correlation">
              <option value="heatmap">Heatmap</option>
              <option value="scatter">Scatter Matrix</option>
            </select>
          </div>
          <div class="card-body">
            <div id="correlation-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Feature Statistics</h3>
          </div>
          <div class="card-body">
            <div id="statistics-table"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Data Quality</h3>
          </div>
          <div class="card-body">
            <div id="quality-chart" style="height: 300px;"></div>
          </div>
        </div>
      </div>
    `;

    // Render individual charts
    await Promise.all([
      this.renderDistributionChart(features),
      this.renderCorrelationChart(features),
      this.renderStatisticsTable(features),
      this.renderQualityChart(features)
    ]);
  }

  /**
   * Render result visualization (anomaly detection results)
   */
  async renderResultVisualization() {
    if (!this.currentResult) return;

    const container = document.getElementById('result-visualization');
    if (!container) return;

    container.innerHTML = `
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Anomaly Distribution</h3>
          </div>
          <div class="card-body">
            <div id="anomaly-distribution-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Score Distribution</h3>
          </div>
          <div class="card-body">
            <div id="score-distribution-chart" style="height: 300px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Anomaly Scatter Plot</h3>
          </div>
          <div class="card-body">
            <div id="anomaly-scatter-chart" style="height: 400px;"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3 class="card-title">Detection Summary</h3>
          </div>
          <div class="card-body">
            <div id="detection-summary"></div>
          </div>
        </div>
      </div>
    `;

    await Promise.all([
      this.renderAnomalyDistributionChart(),
      this.renderScoreDistributionChart(),
      this.renderAnomalyScatterPlot(),
      this.renderDetectionSummary()
    ]);
  }

  /**
   * Render distribution chart (histogram/boxplot)
   */
  async renderDistributionChart(features) {
    const container = document.getElementById('distribution-chart');
    if (!container || !features.length) return;

    const chart = echarts.init(container);
    
    // Use first numeric feature for histogram
    const feature = features.find(f => f.type === 'numeric');
    if (!feature) return;

    const values = feature.values;
    const bins = this.calculateHistogramBins(values, 20);

    const option = {
      title: {
        text: `Distribution: ${feature.name}`,
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      xAxis: {
        type: 'category',
        data: bins.map(bin => bin.range)
      },
      yAxis: {
        type: 'value',
        name: 'Frequency'
      },
      series: [{
        name: 'Frequency',
        type: 'bar',
        data: bins.map(bin => bin.count),
        itemStyle: {
          color: '#3b82f6'
        }
      }]
    };

    chart.setOption(option);
    this.charts.set('distribution-chart', chart);
  }

  /**
   * Render correlation heatmap
   */
  async renderCorrelationChart(features) {
    const container = document.getElementById('correlation-chart');
    if (!container) return;

    const numericFeatures = features.filter(f => f.type === 'numeric');
    if (numericFeatures.length < 2) return;

    const correlationMatrix = this.calculateCorrelationMatrix(numericFeatures);
    const chart = echarts.init(container);

    const option = {
      title: {
        text: 'Feature Correlation Matrix',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        position: 'top',
        formatter: (params) => {
          return `${params.name}<br/>Correlation: ${params.value[2].toFixed(3)}`;
        }
      },
      grid: {
        height: '50%',
        top: '10%'
      },
      xAxis: {
        type: 'category',
        data: numericFeatures.map(f => f.name),
        splitArea: { show: true }
      },
      yAxis: {
        type: 'category',
        data: numericFeatures.map(f => f.name),
        splitArea: { show: true }
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '15%',
        inRange: {
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', 
                  '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        }
      },
      series: [{
        name: 'Correlation',
        type: 'heatmap',
        data: correlationMatrix,
        label: {
          show: true,
          formatter: (params) => params.value[2].toFixed(2)
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }]
    };

    chart.setOption(option);
    this.charts.set('correlation-chart', chart);
  }

  /**
   * Render statistics table
   */
  renderStatisticsTable(features) {
    const container = document.getElementById('statistics-table');
    if (!container) return;

    const numericFeatures = features.filter(f => f.type === 'numeric');
    
    const tableRows = numericFeatures.map(feature => {
      const stats = this.calculateBasicStats(feature.values);
      return `
        <tr>
          <td class="font-medium">${feature.name}</td>
          <td>${stats.mean.toFixed(3)}</td>
          <td>${stats.std.toFixed(3)}</td>
          <td>${stats.min.toFixed(3)}</td>
          <td>${stats.max.toFixed(3)}</td>
          <td>${stats.median.toFixed(3)}</td>
        </tr>
      `;
    }).join('');

    container.innerHTML = `
      <div class="overflow-x-auto">
        <table class="table table-striped">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Mean</th>
              <th>Std Dev</th>
              <th>Min</th>
              <th>Max</th>
              <th>Median</th>
            </tr>
          </thead>
          <tbody>
            ${tableRows}
          </tbody>
        </table>
      </div>
    `;
  }

  /**
   * Render data quality chart
   */
  async renderQualityChart(features) {
    const container = document.getElementById('quality-chart');
    if (!container) return;

    const qualityMetrics = features.map(feature => {
      const totalValues = feature.values.length;
      const missingValues = feature.values.filter(v => v === null || v === undefined || v === '').length;
      const completeness = ((totalValues - missingValues) / totalValues) * 100;
      
      return {
        name: feature.name,
        completeness: completeness,
        uniqueness: this.calculateUniqueness(feature.values),
        validity: this.calculateValidity(feature.values, feature.type)
      };
    });

    const chart = echarts.init(container);
    const option = {
      title: {
        text: 'Data Quality Metrics',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      legend: {
        data: ['Completeness', 'Uniqueness', 'Validity'],
        bottom: 0
      },
      xAxis: {
        type: 'category',
        data: qualityMetrics.map(m => m.name),
        axisLabel: {
          rotate: 45
        }
      },
      yAxis: {
        type: 'value',
        name: 'Percentage',
        max: 100
      },
      series: [
        {
          name: 'Completeness',
          type: 'bar',
          data: qualityMetrics.map(m => m.completeness),
          itemStyle: { color: '#10b981' }
        },
        {
          name: 'Uniqueness',
          type: 'bar',
          data: qualityMetrics.map(m => m.uniqueness),
          itemStyle: { color: '#3b82f6' }
        },
        {
          name: 'Validity',
          type: 'bar',
          data: qualityMetrics.map(m => m.validity),
          itemStyle: { color: '#f59e0b' }
        }
      ]
    };

    chart.setOption(option);
    this.charts.set('quality-chart', chart);
  }

  /**
   * Render anomaly distribution chart
   */
  async renderAnomalyDistributionChart() {
    const container = document.getElementById('anomaly-distribution-chart');
    if (!container || !this.currentResult) return;

    const result = this.currentResult;
    const totalSamples = result.statistics?.totalSamples || 0;
    const totalAnomalies = result.statistics?.totalAnomalies || 0;
    const normalSamples = totalSamples - totalAnomalies;

    const chart = echarts.init(container);
    const option = {
      title: {
        text: 'Normal vs Anomalous Data',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c} ({d}%)'
      },
      series: [{
        name: 'Data Distribution',
        type: 'pie',
        radius: ['40%', '70%'],
        data: [
          {
            value: normalSamples,
            name: 'Normal',
            itemStyle: { color: '#10b981' }
          },
          {
            value: totalAnomalies,
            name: 'Anomalous',
            itemStyle: { color: '#ef4444' }
          }
        ],
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }]
    };

    chart.setOption(option);
    this.charts.set('anomaly-distribution-chart', chart);
  }

  /**
   * Render score distribution chart
   */
  async renderScoreDistributionChart() {
    const container = document.getElementById('score-distribution-chart');
    if (!container || !this.currentResult) return;

    const scores = this.currentResult.scores || [];
    if (!scores.length) return;

    const bins = this.calculateHistogramBins(scores, 30);
    const chart = echarts.init(container);

    const option = {
      title: {
        text: 'Anomaly Score Distribution',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      xAxis: {
        type: 'category',
        data: bins.map(bin => bin.range),
        name: 'Anomaly Score'
      },
      yAxis: {
        type: 'value',
        name: 'Frequency'
      },
      series: [{
        name: 'Frequency',
        type: 'bar',
        data: bins.map(bin => bin.count),
        itemStyle: {
          color: '#8b5cf6'
        }
      }]
    };

    chart.setOption(option);
    this.charts.set('score-distribution-chart', chart);
  }

  /**
   * Render anomaly scatter plot
   */
  async renderAnomalyScatterPlot() {
    const container = document.getElementById('anomaly-scatter-chart');
    if (!container || !this.currentResult || !this.currentDataset) return;

    const data = this.currentDataset.data;
    const anomalies = this.currentResult.anomalies || [];
    const scores = this.currentResult.scores || [];

    // Use first two numeric features for scatter plot
    const features = this.extractFeatures(data);
    const numericFeatures = features.filter(f => f.type === 'numeric').slice(0, 2);
    
    if (numericFeatures.length < 2) return;

    const normalData = [];
    const anomalyData = [];
    const anomalyIndices = new Set(anomalies.map(a => a.index));

    data.forEach((row, index) => {
      const point = [
        row[numericFeatures[0].name] || 0,
        row[numericFeatures[1].name] || 0,
        scores[index] || 0
      ];

      if (anomalyIndices.has(index)) {
        anomalyData.push(point);
      } else {
        normalData.push(point);
      }
    });

    const chart = echarts.init(container);
    const option = {
      title: {
        text: `${numericFeatures[0].name} vs ${numericFeatures[1].name}`,
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'item',
        formatter: (params) => {
          const [x, y, score] = params.data;
          return `${params.seriesName}<br/>
                  ${numericFeatures[0].name}: ${x.toFixed(3)}<br/>
                  ${numericFeatures[1].name}: ${y.toFixed(3)}<br/>
                  Score: ${score.toFixed(3)}`;
        }
      },
      legend: {
        data: ['Normal', 'Anomaly'],
        bottom: 0
      },
      xAxis: {
        type: 'value',
        name: numericFeatures[0].name,
        scale: true
      },
      yAxis: {
        type: 'value',
        name: numericFeatures[1].name,
        scale: true
      },
      series: [
        {
          name: 'Normal',
          type: 'scatter',
          data: normalData,
          itemStyle: {
            color: '#10b981',
            opacity: 0.7
          },
          symbolSize: 6
        },
        {
          name: 'Anomaly',
          type: 'scatter',
          data: anomalyData,
          itemStyle: {
            color: '#ef4444',
            opacity: 0.9
          },
          symbolSize: 10
        }
      ]
    };

    chart.setOption(option);
    this.charts.set('anomaly-scatter-chart', chart);
  }

  /**
   * Render detection summary
   */
  renderDetectionSummary() {
    const container = document.getElementById('detection-summary');
    if (!container || !this.currentResult) return;

    const result = this.currentResult;
    const stats = result.statistics || {};

    container.innerHTML = `
      <div class="space-y-4">
        <div class="grid grid-cols-2 gap-4">
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Algorithm</div>
            <div class="font-semibold">${result.algorithmId || 'Unknown'}</div>
          </div>
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Processing Time</div>
            <div class="font-semibold">${result.processingTimeMs || 0}ms</div>
          </div>
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Total Samples</div>
            <div class="font-semibold">${stats.totalSamples || 0}</div>
          </div>
          <div class="bg-gray-50 p-3 rounded">
            <div class="text-sm text-gray-600">Anomaly Rate</div>
            <div class="font-semibold">${((stats.anomalyRate || 0) * 100).toFixed(2)}%</div>
          </div>
        </div>

        <div class="bg-blue-50 p-4 rounded border border-blue-200">
          <h4 class="font-medium text-blue-900 mb-2">Detection Parameters</h4>
          <div class="text-sm text-blue-800">
            ${Object.entries(result.parameters || {}).map(([key, value]) => 
              `<div><strong>${key}:</strong> ${value}</div>`
            ).join('')}
          </div>
        </div>

        <div class="flex gap-2">
          <button class="btn-base btn-sm btn-primary export-viz" data-format="png">
            Export as PNG
          </button>
          <button class="btn-base btn-sm btn-secondary export-viz" data-format="pdf">
            Export as PDF
          </button>
        </div>
      </div>
    `;
  }

  // Helper methods...

  /**
   * Extract features from dataset
   */
  extractFeatures(data) {
    if (!data || !data.length) return [];

    const features = [];
    const firstRow = data[0];

    Object.keys(firstRow).forEach(key => {
      const values = data.map(row => row[key]);
      const type = this.inferDataType(values);
      
      features.push({
        name: key,
        type,
        values: values.filter(v => v !== null && v !== undefined)
      });
    });

    return features;
  }

  /**
   * Infer data type from values
   */
  inferDataType(values) {
    const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
    if (!nonNullValues.length) return 'unknown';

    const numericCount = nonNullValues.filter(v => !isNaN(parseFloat(v))).length;
    const ratio = numericCount / nonNullValues.length;

    return ratio > 0.8 ? 'numeric' : 'categorical';
  }

  /**
   * Calculate histogram bins
   */
  calculateHistogramBins(values, numBins) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binWidth = (max - min) / numBins;

    const bins = [];
    for (let i = 0; i < numBins; i++) {
      const start = min + i * binWidth;
      const end = min + (i + 1) * binWidth;
      const count = values.filter(v => v >= start && (i === numBins - 1 ? v <= end : v < end)).length;
      
      bins.push({
        range: `${start.toFixed(2)}-${end.toFixed(2)}`,
        count
      });
    }

    return bins;
  }

  /**
   * Calculate correlation matrix
   */
  calculateCorrelationMatrix(features) {
    const matrix = [];
    
    for (let i = 0; i < features.length; i++) {
      for (let j = 0; j < features.length; j++) {
        const correlation = this.calculateCorrelation(features[i].values, features[j].values);
        matrix.push([i, j, correlation]);
      }
    }

    return matrix;
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  calculateCorrelation(x, y) {
    const n = Math.min(x.length, y.length);
    if (n < 2) return 0;

    const sumX = x.slice(0, n).reduce((a, b) => a + b, 0);
    const sumY = y.slice(0, n).reduce((a, b) => a + b, 0);
    const sumXY = x.slice(0, n).reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.slice(0, n).reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.slice(0, n).reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Calculate basic statistics
   */
  calculateBasicStats(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    
    return {
      mean,
      std: Math.sqrt(variance),
      min: Math.min(...values),
      max: Math.max(...values),
      median: sorted[Math.floor(sorted.length / 2)]
    };
  }

  /**
   * Calculate data uniqueness percentage
   */
  calculateUniqueness(values) {
    const unique = new Set(values).size;
    return (unique / values.length) * 100;
  }

  /**
   * Calculate data validity percentage
   */
  calculateValidity(values, type) {
    if (type === 'numeric') {
      const validCount = values.filter(v => !isNaN(parseFloat(v))).length;
      return (validCount / values.length) * 100;
    }
    // For categorical, assume all non-null values are valid
    const validCount = values.filter(v => v !== null && v !== undefined && v !== '').length;
    return (validCount / values.length) * 100;
  }

  /**
   * Update result selector based on current dataset
   */
  updateResultSelector() {
    const selector = document.querySelector('.result-selector');
    if (!selector || !this.currentDataset) return;

    const relevantResults = Array.from(this.results.values())
      .filter(result => result.datasetId === this.currentDataset.id);

    selector.innerHTML = `
      <option value="">Select a result...</option>
      ${relevantResults.map(result => `
        <option value="${result.id}">
          ${result.algorithmId} - ${new Date(result.timestamp).toLocaleDateString()}
        </option>
      `).join('')}
    `;
  }

  /**
   * Change visualization type
   */
  changeVisualizationType(type) {
    // Implementation for different visualization types
    console.log('Changing visualization type to:', type);
  }

  /**
   * Export visualization
   */
  exportVisualization(format) {
    // Implementation for exporting visualizations
    console.log('Exporting visualization as:', format);
  }

  /**
   * Clear all charts
   */
  clearCharts() {
    this.charts.forEach(chart => {
      chart.dispose();
    });
    this.charts.clear();
  }

  /**
   * Get available datasets for selection
   */
  getAvailableDatasets() {
    return Array.from(this.datasets.values());
  }

  /**
   * Get available results for selection
   */
  getAvailableResults() {
    return Array.from(this.results.values());
  }
}

// Initialize and expose globally
if (typeof window !== 'undefined') {
  window.OfflineVisualizer = new OfflineVisualizer();
}
