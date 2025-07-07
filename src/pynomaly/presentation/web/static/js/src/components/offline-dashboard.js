/**
 * Offline Dashboard Component
 * Provides interactive dashboard functionality using cached data
 */
export class OfflineDashboard {
  constructor() {
    this.charts = new Map();
    this.cachedData = {
      datasets: [],
      results: [],
      stats: {},
      algorithms: []
    };
    this.isInitialized = false;
    
    this.init();
  }

  async init() {
    await this.loadCachedData();
    this.setupEventListeners();
    this.renderDashboard();
    this.isInitialized = true;
  }

  /**
   * Load cached data from IndexedDB via service worker
   */
  async loadCachedData() {
    try {
      if ('serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          // Request all cached data
          registration.active.postMessage({ type: 'GET_OFFLINE_DASHBOARD_DATA' });
          
          return new Promise((resolve) => {
            navigator.serviceWorker.addEventListener('message', function handler(event) {
              if (event.data.type === 'OFFLINE_DASHBOARD_DATA') {
                navigator.serviceWorker.removeEventListener('message', handler);
                this.cachedData = {
                  ...this.cachedData,
                  ...event.data.data
                };
                resolve(event.data.data);
              }
            }.bind(this));
          });
        }
      }
    } catch (error) {
      console.error('[OfflineDashboard] Failed to load cached data:', error);
    }
  }

  /**
   * Setup event listeners for user interactions
   */
  setupEventListeners() {
    // Dataset selection
    document.addEventListener('change', (event) => {
      if (event.target.matches('.dataset-selector')) {
        this.onDatasetChange(event.target.value);
      }
    });

    // Algorithm selection
    document.addEventListener('change', (event) => {
      if (event.target.matches('.algorithm-selector')) {
        this.onAlgorithmChange(event.target.value);
      }
    });

    // Refresh button
    document.addEventListener('click', (event) => {
      if (event.target.matches('.refresh-dashboard')) {
        this.refreshDashboard();
      }
    });

    // Export buttons
    document.addEventListener('click', (event) => {
      if (event.target.matches('.export-chart')) {
        this.exportChart(event.target.dataset.chartId);
      }
    });
  }

  /**
   * Render the complete dashboard
   */
  renderDashboard() {
    this.renderOverviewCards();
    this.renderDatasetChart();
    this.renderAlgorithmPerformanceChart();
    this.renderAnomalyTimelineChart();
    this.renderRecentActivity();
  }

  /**
   * Render overview statistic cards
   */
  renderOverviewCards() {
    const container = document.getElementById('overview-cards');
    if (!container) return;

    const stats = this.calculateStats();
    
    container.innerHTML = `
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Total Datasets</p>
                <p class="text-2xl font-bold">${stats.totalDatasets}</p>
              </div>
              <div class="text-3xl">📊</div>
            </div>
            <div class="mt-2 text-sm text-green-600">
              ${stats.datasetsLastWeek} added this week
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Detections Run</p>
                <p class="text-2xl font-bold">${stats.totalDetections}</p>
              </div>
              <div class="text-3xl">🔍</div>
            </div>
            <div class="mt-2 text-sm text-blue-600">
              ${stats.detectionsToday} today
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Anomalies Found</p>
                <p class="text-2xl font-bold">${stats.totalAnomalies}</p>
              </div>
              <div class="text-3xl">⚠️</div>
            </div>
            <div class="mt-2 text-sm ${stats.anomalyRate > 0.1 ? 'text-red-600' : 'text-gray-600'}">
              ${(stats.anomalyRate * 100).toFixed(1)}% anomaly rate
            </div>
          </div>
        </div>
        
        <div class="card">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-text-secondary text-sm">Cached Data</p>
                <p class="text-2xl font-bold">${this.formatBytes(stats.cacheSize)}</p>
              </div>
              <div class="text-3xl">💾</div>
            </div>
            <div class="mt-2 text-sm text-purple-600">
              Available offline
            </div>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Render dataset distribution chart
   */
  renderDatasetChart() {
    const container = document.getElementById('dataset-chart');
    if (!container) return;

    const datasets = this.cachedData.datasets || [];
    const typeDistribution = datasets.reduce((acc, dataset) => {
      const type = dataset.type || 'unknown';
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});

    const chartData = Object.entries(typeDistribution).map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value
    }));

    // Use ECharts for visualization
    const chart = echarts.init(container);
    const option = {
      title: {
        text: 'Dataset Distribution',
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
          name: 'Datasets',
          type: 'pie',
          radius: '50%',
          data: chartData,
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
    this.charts.set('dataset-chart', chart);

    // Make chart responsive
    window.addEventListener('resize', () => chart.resize());
  }

  /**
   * Render algorithm performance comparison chart
   */
  renderAlgorithmPerformanceChart() {
    const container = document.getElementById('algorithm-performance-chart');
    if (!container) return;

    const results = this.cachedData.results || [];
    const algorithmStats = results.reduce((acc, result) => {
      const algo = result.algorithm || 'unknown';
      if (!acc[algo]) {
        acc[algo] = { count: 0, totalTime: 0, totalAnomalies: 0 };
      }
      acc[algo].count++;
      acc[algo].totalTime += result.processingTime || 0;
      acc[algo].totalAnomalies += result.anomalies?.length || 0;
      return acc;
    }, {});

    const algorithms = Object.keys(algorithmStats);
    const avgTimes = algorithms.map(algo => 
      algorithmStats[algo].totalTime / algorithmStats[algo].count
    );
    const totalAnomalies = algorithms.map(algo => 
      algorithmStats[algo].totalAnomalies
    );

    const chart = echarts.init(container);
    const option = {
      title: {
        text: 'Algorithm Performance',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['Average Processing Time (ms)', 'Total Anomalies Found'],
        bottom: 0
      },
      xAxis: {
        type: 'category',
        data: algorithms,
        axisPointer: {
          type: 'shadow'
        }
      },
      yAxis: [
        {
          type: 'value',
          name: 'Time (ms)',
          position: 'left'
        },
        {
          type: 'value',
          name: 'Anomalies',
          position: 'right'
        }
      ],
      series: [
        {
          name: 'Average Processing Time (ms)',
          type: 'bar',
          yAxisIndex: 0,
          data: avgTimes,
          itemStyle: {
            color: '#3b82f6'
          }
        },
        {
          name: 'Total Anomalies Found',
          type: 'line',
          yAxisIndex: 1,
          data: totalAnomalies,
          itemStyle: {
            color: '#ef4444'
          }
        }
      ]
    };

    chart.setOption(option);
    this.charts.set('algorithm-performance-chart', chart);

    window.addEventListener('resize', () => chart.resize());
  }

  /**
   * Render anomaly detection timeline chart
   */
  renderAnomalyTimelineChart() {
    const container = document.getElementById('anomaly-timeline-chart');
    if (!container) return;

    const results = this.cachedData.results || [];
    
    // Group results by day
    const timelineData = results.reduce((acc, result) => {
      const date = new Date(result.timestamp).toISOString().split('T')[0];
      if (!acc[date]) {
        acc[date] = { detections: 0, anomalies: 0 };
      }
      acc[date].detections++;
      acc[date].anomalies += result.anomalies?.length || 0;
      return acc;
    }, {});

    const dates = Object.keys(timelineData).sort();
    const detections = dates.map(date => timelineData[date].detections);
    const anomalies = dates.map(date => timelineData[date].anomalies);

    const chart = echarts.init(container);
    const option = {
      title: {
        text: 'Detection Activity Timeline',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['Detections Run', 'Anomalies Found'],
        bottom: 0
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: dates
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          name: 'Detections Run',
          type: 'line',
          stack: 'Total',
          areaStyle: {},
          data: detections,
          itemStyle: {
            color: '#10b981'
          }
        },
        {
          name: 'Anomalies Found',
          type: 'line',
          stack: 'Total',
          areaStyle: {},
          data: anomalies,
          itemStyle: {
            color: '#f59e0b'
          }
        }
      ]
    };

    chart.setOption(option);
    this.charts.set('anomaly-timeline-chart', chart);

    window.addEventListener('resize', () => chart.resize());
  }

  /**
   * Render recent activity feed
   */
  renderRecentActivity() {
    const container = document.getElementById('recent-activity');
    if (!container) return;

    const results = this.cachedData.results || [];
    const recentResults = results
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      .slice(0, 10);

    const activityHtml = recentResults.map(result => {
      const timeAgo = this.timeAgo(new Date(result.timestamp));
      const anomalyCount = result.anomalies?.length || 0;
      const statusColor = anomalyCount > 0 ? 'text-orange-600' : 'text-green-600';
      const statusIcon = anomalyCount > 0 ? '⚠️' : '✅';

      return `
        <div class="flex items-start gap-3 p-3 border-b border-border last:border-b-0">
          <div class="text-xl">${statusIcon}</div>
          <div class="flex-grow">
            <div class="flex items-center justify-between">
              <h4 class="font-medium">${result.dataset || 'Unknown Dataset'}</h4>
              <span class="text-sm text-text-secondary">${timeAgo}</span>
            </div>
            <p class="text-sm text-text-secondary">
              Algorithm: ${result.algorithm || 'Unknown'}
            </p>
            <p class="text-sm ${statusColor}">
              ${anomalyCount} anomalies detected
            </p>
          </div>
        </div>
      `;
    }).join('');

    container.innerHTML = `
      <div class="card">
        <div class="card-header">
          <h3 class="card-title">Recent Activity</h3>
          <button class="btn-base btn-sm refresh-dashboard">
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"></path>
            </svg>
            Refresh
          </button>
        </div>
        <div class="card-body p-0">
          ${activityHtml || '<div class="p-4 text-center text-text-secondary">No recent activity</div>'}
        </div>
      </div>
    `;
  }

  /**
   * Handle dataset selection change
   */
  onDatasetChange(datasetId) {
    const dataset = this.cachedData.datasets.find(d => d.id === datasetId);
    if (dataset) {
      this.renderDatasetDetails(dataset);
    }
  }

  /**
   * Handle algorithm selection change
   */
  onAlgorithmChange(algorithmId) {
    const algorithm = this.cachedData.algorithms.find(a => a.id === algorithmId);
    if (algorithm) {
      this.renderAlgorithmDetails(algorithm);
    }
  }

  /**
   * Refresh dashboard data
   */
  async refreshDashboard() {
    const refreshButton = document.querySelector('.refresh-dashboard');
    if (refreshButton) {
      refreshButton.disabled = true;
      refreshButton.innerHTML = 'Refreshing...';
    }

    try {
      await this.loadCachedData();
      this.renderDashboard();
    } catch (error) {
      console.error('[OfflineDashboard] Failed to refresh:', error);
    } finally {
      if (refreshButton) {
        refreshButton.disabled = false;
        refreshButton.innerHTML = `
          <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"></path>
          </svg>
          Refresh
        `;
      }
    }
  }

  /**
   * Export chart as image
   */
  exportChart(chartId) {
    const chart = this.charts.get(chartId);
    if (chart) {
      const dataURL = chart.getDataURL({
        type: 'png',
        pixelRatio: 2,
        backgroundColor: '#fff'
      });
      
      const link = document.createElement('a');
      link.download = `${chartId}-${Date.now()}.png`;
      link.href = dataURL;
      link.click();
    }
  }

  /**
   * Calculate dashboard statistics
   */
  calculateStats() {
    const datasets = this.cachedData.datasets || [];
    const results = this.cachedData.results || [];
    
    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const dayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);

    const datasetsLastWeek = datasets.filter(d => 
      new Date(d.timestamp) > weekAgo
    ).length;

    const detectionsToday = results.filter(r => 
      new Date(r.timestamp) > dayAgo
    ).length;

    const totalAnomalies = results.reduce((sum, r) => 
      sum + (r.anomalies?.length || 0), 0
    );

    const totalSamples = results.reduce((sum, r) => 
      sum + (r.totalSamples || 0), 0
    );

    return {
      totalDatasets: datasets.length,
      datasetsLastWeek,
      totalDetections: results.length,
      detectionsToday,
      totalAnomalies,
      anomalyRate: totalSamples > 0 ? totalAnomalies / totalSamples : 0,
      cacheSize: this.estimateCacheSize()
    };
  }

  /**
   * Estimate cache size
   */
  estimateCacheSize() {
    const jsonSize = JSON.stringify(this.cachedData).length;
    return jsonSize * 2; // Rough estimate including IndexedDB overhead
  }

  /**
   * Format bytes to human readable
   */
  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Format time ago
   */
  timeAgo(date) {
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);
    
    if (diffInSeconds < 60) return 'Just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    return `${Math.floor(diffInSeconds / 86400)}d ago`;
  }
}

// Initialize and expose globally
if (typeof window !== 'undefined') {
  window.OfflineDashboard = new OfflineDashboard();
}
