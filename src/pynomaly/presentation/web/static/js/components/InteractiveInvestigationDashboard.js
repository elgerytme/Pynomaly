/**
 * Interactive Investigation Dashboard
 * 
 * Opens when user clicks on an anomaly to provide detailed context,
 * feature analysis, and related anomaly information.
 */

class InteractiveInvestigationDashboard {
  constructor(options = {}) {
    this.options = {
      title: 'Anomaly Investigation Dashboard',
      width: '90%',
      height: '90%',
      position: 'center',
      enableContextualAnalysis: true,
      enableFeatureImportance: true,
      enableSimilarAnomalies: true,
      enableExport: true,
      ...options
    };

    this.anomalyData = null;
    this.contextData = null;
    this.charts = new Map();
    this.overlay = null;
    this.dashboard = null;
    this.isVisible = false;
  }

  /**
   * Open the investigation dashboard with anomaly data
   * @param {Object} anomalyData - The anomaly data to investigate
   */
  open(anomalyData) {
    this.anomalyData = anomalyData;
    this.createOverlay();
    this.createDashboard();
    this.loadContextualData();
    this.show();
  }

  createOverlay() {
    this.overlay = document.createElement('div');
    this.overlay.className = 'investigation-overlay';
    this.overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      z-index: 9999;
      display: flex;
      justify-content: center;
      align-items: center;
      opacity: 0;
      transition: opacity 0.3s ease;
    `;

    // Close on overlay click
    this.overlay.addEventListener('click', (e) => {
      if (e.target === this.overlay) {
        this.close();
      }
    });

    // ESC key to close
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isVisible) {
        this.close();
      }
    });

    document.body.appendChild(this.overlay);
  }

  createDashboard() {
    this.dashboard = document.createElement('div');
    this.dashboard.className = 'investigation-dashboard';
    this.dashboard.style.cssText = `
      background: white;
      border-radius: 12px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
      width: ${this.options.width};
      height: ${this.options.height};
      max-width: 1200px;
      max-height: 800px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transform: scale(0.9);
      transition: transform 0.3s ease;
    `;

    this.createHeader();
    this.createContent();
    this.createFooter();

    this.overlay.appendChild(this.dashboard);
  }

  createHeader() {
    const header = document.createElement('div');
    header.className = 'investigation-header';
    header.style.cssText = `
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 30px;
      border-bottom: 1px solid #e5e7eb;
      background: #f9fafb;
    `;

    const title = document.createElement('h2');
    title.textContent = this.options.title;
    title.style.cssText = `
      margin: 0;
      font-size: 24px;
      font-weight: 600;
      color: #1f2937;
    `;

    const closeButton = document.createElement('button');
    closeButton.innerHTML = '×';
    closeButton.className = 'investigation-close-btn';
    closeButton.style.cssText = `
      background: none;
      border: none;
      font-size: 32px;
      cursor: pointer;
      color: #6b7280;
      width: 40px;
      height: 40px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background-color 0.2s ease;
    `;

    closeButton.addEventListener('click', () => this.close());
    closeButton.addEventListener('mouseenter', () => {
      closeButton.style.backgroundColor = '#f3f4f6';
    });
    closeButton.addEventListener('mouseleave', () => {
      closeButton.style.backgroundColor = 'transparent';
    });

    header.appendChild(title);
    header.appendChild(closeButton);
    this.dashboard.appendChild(header);
  }

  createContent() {
    const content = document.createElement('div');
    content.className = 'investigation-content';
    content.style.cssText = `
      flex: 1;
      overflow-y: auto;
      padding: 0;
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: auto 1fr 1fr;
      gap: 20px;
      padding: 20px;
    `;

    this.createAnomalyOverview(content);
    this.createTimeSeriesContext(content);
    this.createFeatureAnalysis(content);
    this.createSimilarAnomalies(content);
    this.createContextualMetrics(content);

    this.dashboard.appendChild(content);
  }

  createAnomalyOverview(parent) {
    const overview = document.createElement('div');
    overview.className = 'anomaly-overview';
    overview.style.cssText = `
      grid-column: 1 / -1;
      background: #f8fafc;
      border-radius: 8px;
      padding: 20px;
      border: 1px solid #e2e8f0;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Anomaly Overview';
    title.style.cssText = `
      margin: 0 0 15px 0;
      font-size: 18px;
      font-weight: 600;
      color: #1f2937;
    `;

    const details = document.createElement('div');
    details.className = 'anomaly-details';
    details.style.cssText = `
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
    `;

    // Add anomaly details
    const detailItems = [
      { label: 'Timestamp', value: this.anomalyData ? new Date(this.anomalyData[0]).toLocaleString() : 'N/A' },
      { label: 'Value', value: this.anomalyData ? this.anomalyData[1] : 'N/A' },
      { label: 'Anomaly Score', value: this.anomalyData ? this.anomalyData[2]?.toFixed(3) : 'N/A' },
      { label: 'Severity', value: this.getSeverityLevel(this.anomalyData?.[2]) },
      { label: 'Detection Method', value: 'Isolation Forest' },
      { label: 'Confidence', value: this.anomalyData ? (this.anomalyData[2] * 100).toFixed(1) + '%' : 'N/A' }
    ];

    detailItems.forEach(item => {
      const detailItem = document.createElement('div');
      detailItem.className = 'detail-item';
      detailItem.innerHTML = `
        <div style="font-weight: 500; color: #374151; margin-bottom: 5px;">${item.label}</div>
        <div style="color: #6b7280; font-size: 14px;">${item.value}</div>
      `;
      details.appendChild(detailItem);
    });

    overview.appendChild(title);
    overview.appendChild(details);
    parent.appendChild(overview);
  }

  createTimeSeriesContext(parent) {
    const container = document.createElement('div');
    container.className = 'time-series-context';
    container.style.cssText = `
      background: white;
      border-radius: 8px;
      padding: 20px;
      border: 1px solid #e2e8f0;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Time Series Context';
    title.style.cssText = `
      margin: 0 0 15px 0;
      font-size: 16px;
      font-weight: 600;
      color: #1f2937;
    `;

    const chartContainer = document.createElement('div');
    chartContainer.style.cssText = `
      width: 100%;
      height: 250px;
    `;

    container.appendChild(title);
    container.appendChild(chartContainer);
    parent.appendChild(container);

    // Initialize ECharts
    const chart = echarts.init(chartContainer);
    this.charts.set('timeSeries', chart);
    this.renderTimeSeriesContext(chart);
  }

  createFeatureAnalysis(parent) {
    const container = document.createElement('div');
    container.className = 'feature-analysis';
    container.style.cssText = `
      background: white;
      border-radius: 8px;
      padding: 20px;
      border: 1px solid #e2e8f0;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Feature Importance';
    title.style.cssText = `
      margin: 0 0 15px 0;
      font-size: 16px;
      font-weight: 600;
      color: #1f2937;
    `;

    const chartContainer = document.createElement('div');
    chartContainer.style.cssText = `
      width: 100%;
      height: 250px;
    `;

    container.appendChild(title);
    container.appendChild(chartContainer);
    parent.appendChild(container);

    // Initialize ECharts
    const chart = echarts.init(chartContainer);
    this.charts.set('features', chart);
    this.renderFeatureAnalysis(chart);
  }

  createSimilarAnomalies(parent) {
    const container = document.createElement('div');
    container.className = 'similar-anomalies';
    container.style.cssText = `
      background: white;
      border-radius: 8px;
      padding: 20px;
      border: 1px solid #e2e8f0;
      grid-column: 1 / -1;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Similar Anomalies';
    title.style.cssText = `
      margin: 0 0 15px 0;
      font-size: 16px;
      font-weight: 600;
      color: #1f2937;
    `;

    const list = document.createElement('div');
    list.className = 'similar-anomalies-list';
    list.style.cssText = `
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 15px;
      max-height: 200px;
      overflow-y: auto;
    `;

    // Mock similar anomalies
    const similarAnomalies = [
      { timestamp: new Date(Date.now() - 3600000), score: 0.89, similarity: 0.95 },
      { timestamp: new Date(Date.now() - 7200000), score: 0.92, similarity: 0.87 },
      { timestamp: new Date(Date.now() - 14400000), score: 0.85, similarity: 0.82 }
    ];

    similarAnomalies.forEach(anomaly => {
      const item = document.createElement('div');
      item.className = 'similar-anomaly-item';
      item.style.cssText = `
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 15px;
        cursor: pointer;
        transition: all 0.2s ease;
      `;

      item.innerHTML = `
        <div style="font-weight: 500; color: #374151; margin-bottom: 8px;">
          ${anomaly.timestamp.toLocaleString()}
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 14px; color: #6b7280;">
          <span>Score: ${anomaly.score.toFixed(3)}</span>
          <span>Similarity: ${(anomaly.similarity * 100).toFixed(1)}%</span>
        </div>
      `;

      item.addEventListener('mouseenter', () => {
        item.style.backgroundColor = '#f1f5f9';
        item.style.borderColor = '#cbd5e1';
      });

      item.addEventListener('mouseleave', () => {
        item.style.backgroundColor = '#f8fafc';
        item.style.borderColor = '#e2e8f0';
      });

      item.addEventListener('click', () => {
        // Navigate to similar anomaly
        this.navigateToSimilarAnomaly(anomaly);
      });

      list.appendChild(item);
    });

    container.appendChild(title);
    container.appendChild(list);
    parent.appendChild(container);
  }

  createContextualMetrics(parent) {
    const container = document.createElement('div');
    container.className = 'contextual-metrics';
    container.style.cssText = `
      background: white;
      border-radius: 8px;
      padding: 20px;
      border: 1px solid #e2e8f0;
    `;

    const title = document.createElement('h3');
    title.textContent = 'Contextual Metrics';
    title.style.cssText = `
      margin: 0 0 15px 0;
      font-size: 16px;
      font-weight: 600;
      color: #1f2937;
    `;

    const metrics = document.createElement('div');
    metrics.className = 'metrics-grid';
    metrics.style.cssText = `
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 15px;
    `;

    const metricItems = [
      { label: 'Local Outlier Factor', value: '2.34' },
      { label: 'Isolation Depth', value: '4.2' },
      { label: 'Nearest Neighbors', value: '8' },
      { label: 'Cluster Distance', value: '15.7' }
    ];

    metricItems.forEach(item => {
      const metricItem = document.createElement('div');
      metricItem.className = 'metric-item';
      metricItem.style.cssText = `
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
      `;

      metricItem.innerHTML = `
        <div style="font-size: 24px; font-weight: 600; color: #1f2937; margin-bottom: 5px;">
          ${item.value}
        </div>
        <div style="font-size: 12px; color: #6b7280;">
          ${item.label}
        </div>
      `;

      metrics.appendChild(metricItem);
    });

    container.appendChild(title);
    container.appendChild(metrics);
    parent.appendChild(container);
  }

  createFooter() {
    const footer = document.createElement('div');
    footer.className = 'investigation-footer';
    footer.style.cssText = `
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 30px;
      border-top: 1px solid #e5e7eb;
      background: #f9fafb;
    `;

    const actions = document.createElement('div');
    actions.className = 'investigation-actions';
    actions.style.cssText = `
      display: flex;
      gap: 10px;
    `;

    const exportBtn = document.createElement('button');
    exportBtn.textContent = 'Export Analysis';
    exportBtn.className = 'btn btn-secondary';
    exportBtn.style.cssText = `
      padding: 8px 16px;
      border: 1px solid #d1d5db;
      background: white;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      transition: all 0.2s ease;
    `;

    exportBtn.addEventListener('click', () => this.exportAnalysis());

    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.className = 'btn btn-primary';
    closeBtn.style.cssText = `
      padding: 8px 16px;
      border: none;
      background: #3b82f6;
      color: white;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      transition: all 0.2s ease;
    `;

    closeBtn.addEventListener('click', () => this.close());

    actions.appendChild(exportBtn);
    actions.appendChild(closeBtn);
    footer.appendChild(actions);
    this.dashboard.appendChild(footer);
  }

  renderTimeSeriesContext(chart) {
    // Generate mock time series data around the anomaly
    const anomalyTime = this.anomalyData ? new Date(this.anomalyData[0]).getTime() : Date.now();
    const timeRange = 3600000; // 1 hour
    const data = [];

    for (let i = -30; i <= 30; i++) {
      const time = anomalyTime + (i * timeRange / 30);
      const value = Math.sin(i * 0.1) * 50 + 100 + Math.random() * 20;
      data.push([time, value]);
    }

    const option = {
      title: {
        show: false
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      xAxis: {
        type: 'time',
        boundaryGap: false
      },
      yAxis: {
        type: 'value'
      },
      series: [
        {
          name: 'Value',
          type: 'line',
          data: data,
          smooth: true,
          symbol: 'none',
          lineStyle: {
            color: '#3b82f6'
          },
          areaStyle: {
            color: 'rgba(59, 130, 246, 0.1)'
          }
        },
        {
          name: 'Anomaly Point',
          type: 'scatter',
          data: this.anomalyData ? [[this.anomalyData[0], this.anomalyData[1]]] : [],
          symbolSize: 12,
          itemStyle: {
            color: '#ef4444'
          },
          emphasis: {
            scale: 1.5
          }
        }
      ]
    };

    chart.setOption(option);
  }

  renderFeatureAnalysis(chart) {
    // Mock feature importance data
    const features = [
      { name: 'Feature 1', importance: 0.85 },
      { name: 'Feature 2', importance: 0.72 },
      { name: 'Feature 3', importance: 0.68 },
      { name: 'Feature 4', importance: 0.54 },
      { name: 'Feature 5', importance: 0.42 }
    ];

    const option = {
      title: {
        show: false
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'value',
        max: 1
      },
      yAxis: {
        type: 'category',
        data: features.map(f => f.name)
      },
      series: [
        {
          name: 'Importance',
          type: 'bar',
          data: features.map(f => f.importance),
          itemStyle: {
            color: '#f59e0b'
          }
        }
      ]
    };

    chart.setOption(option);
  }

  getSeverityLevel(score) {
    if (!score) return 'Unknown';
    if (score >= 0.9) return 'Critical';
    if (score >= 0.7) return 'High';
    if (score >= 0.5) return 'Medium';
    return 'Low';
  }

  navigateToSimilarAnomaly(anomaly) {
    // Close current dashboard and open new one
    this.close();
    const newDashboard = new InteractiveInvestigationDashboard();
    newDashboard.open([anomaly.timestamp, Math.random() * 100, anomaly.score]);
  }

  exportAnalysis() {
    const analysis = {
      anomaly: this.anomalyData,
      timestamp: new Date().toISOString(),
      context: this.contextData,
      severity: this.getSeverityLevel(this.anomalyData?.[2]),
      features: ['Feature 1', 'Feature 2', 'Feature 3'],
      similarAnomalies: 3
    };

    const dataStr = JSON.stringify(analysis, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `anomaly-analysis-${Date.now()}.json`;
    link.click();

    URL.revokeObjectURL(url);
  }

  loadContextualData() {
    // Mock loading contextual data
    this.contextData = {
      timeRange: '1 hour',
      relatedFeatures: ['Feature 1', 'Feature 2'],
      detectionAlgorithm: 'Isolation Forest',
      confidence: this.anomalyData?.[2] || 0.8
    };
  }

  show() {
    this.isVisible = true;
    document.body.style.overflow = 'hidden';
    
    // Fade in overlay
    requestAnimationFrame(() => {
      this.overlay.style.opacity = '1';
      this.dashboard.style.transform = 'scale(1)';
    });

    // Resize charts after animation
    setTimeout(() => {
      this.charts.forEach(chart => {
        chart.resize();
      });
    }, 300);
  }

  close() {
    this.isVisible = false;
    document.body.style.overflow = '';
    
    // Fade out
    this.overlay.style.opacity = '0';
    this.dashboard.style.transform = 'scale(0.9)';
    
    setTimeout(() => {
      if (this.overlay && this.overlay.parentNode) {
        this.overlay.parentNode.removeChild(this.overlay);
      }
      this.cleanup();
    }, 300);
  }

  cleanup() {
    // Dispose of charts
    this.charts.forEach(chart => {
      chart.dispose();
    });
    this.charts.clear();
    
    // Remove event listeners
    document.removeEventListener('keydown', this.handleKeyDown);
    
    // Clear references
    this.overlay = null;
    this.dashboard = null;
    this.anomalyData = null;
    this.contextData = null;
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = InteractiveInvestigationDashboard;
} else {
  window.InteractiveInvestigationDashboard = InteractiveInvestigationDashboard;
}
