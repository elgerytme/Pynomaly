/**
 * Enhanced Real-Time Anomaly Detection Dashboard
 * Advanced features: AI-powered insights, multi-algorithm comparison, intelligent alerts
 */

import { AnomalyTimelineChart } from "../charts/anomaly-timeline.js";
import { AnomalyHeatmapChart } from "../charts/anomaly-heatmap.js";

export class EnhancedRealTimeDashboard {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      websocketUrl: "ws://localhost:8000/ws/anomalies",
      updateInterval: 1000,
      maxDataPoints: 5000,
      alertThreshold: 0.8,
      enableAIInsights: true,
      enableMultiAlgorithmComparison: true,
      enableIntelligentAlerting: true,
      enablePredictiveAnalytics: true,
      bufferSize: 1000,
      refreshRate: 60,
      ...options,
    };

    this.websocket = null;
    this.charts = new Map();
    this.dataBuffers = new Map();
    this.algorithmResults = new Map();
    this.aiInsights = [];
    this.alertHistory = [];
    this.performanceMetrics = new Map();
    this.isConnected = false;
    this.isPaused = false;
    this.alertCount = 0;
    this.totalDataPoints = 0;
    this.processingRate = 0;
    this.lastProcessingTime = Date.now();

    // AI and analytics components
    this.aiInsightEngine = null;
    this.anomalyPredictor = null;
    this.alertOptimizer = null;

    this.init();
  }

  init() {
    this.createAdvancedLayout();
    this.initializeCharts();
    this.initializeAIComponents();
    this.setupWebSocket();
    this.setupEventHandlers();
    this.startHeartbeat();
    this.initializePerformanceMonitoring();
  }

  createAdvancedLayout() {
    this.container.innerHTML = `
      <div class="enhanced-dashboard">
        <div class="dashboard-header">
          <div class="dashboard-title">
            <h2>AI-Powered Real-Time Anomaly Detection</h2>
            <div class="connection-status" id="connection-status">
              <span class="status-indicator disconnected"></span>
              <span class="status-text">Connecting...</span>
              <span class="latency-info" id="latency-info">0ms</span>
            </div>
          </div>
          <div class="dashboard-controls">
            <div class="control-group">
              <label for="algorithm-selector">Algorithm:</label>
              <select id="algorithm-selector" class="form-control">
                <option value="all">All Algorithms</option>
                <option value="isolation_forest">Isolation Forest</option>
                <option value="lof">Local Outlier Factor</option>
                <option value="copod">COPOD</option>
                <option value="hbos">HBOS</option>
                <option value="ensemble">Ensemble</option>
              </select>
            </div>
            <div class="control-group">
              <label for="sensitivity-slider">Sensitivity:</label>
              <input type="range" id="sensitivity-slider" min="0.1" max="1.0" step="0.1" value="0.8" class="slider">
              <span id="sensitivity-value">0.8</span>
            </div>
            <div class="control-group">
              <button id="pause-btn" class="btn-secondary btn-sm">
                <i class="fas fa-pause"></i> Pause
              </button>
              <button id="clear-btn" class="btn-secondary btn-sm">
                <i class="fas fa-trash"></i> Clear
              </button>
              <button id="export-btn" class="btn-primary btn-sm">
                <i class="fas fa-download"></i> Export
              </button>
              <button id="ai-insights-btn" class="btn-info btn-sm">
                <i class="fas fa-brain"></i> AI Insights
              </button>
            </div>
          </div>
        </div>

        <div class="dashboard-metrics">
          <div class="metric-card" id="total-points">
            <div class="metric-icon">📊</div>
            <div class="metric-content">
              <div class="metric-value">0</div>
              <div class="metric-label">Total Data Points</div>
              <div class="metric-trend" id="points-trend"></div>
            </div>
          </div>
          <div class="metric-card" id="anomaly-count">
            <div class="metric-icon">⚠️</div>
            <div class="metric-content">
              <div class="metric-value">0</div>
              <div class="metric-label">Anomalies Detected</div>
              <div class="metric-trend" id="anomaly-trend"></div>
            </div>
          </div>
          <div class="metric-card" id="anomaly-rate">
            <div class="metric-icon">📈</div>
            <div class="metric-content">
              <div class="metric-value">0.0%</div>
              <div class="metric-label">Anomaly Rate</div>
              <div class="metric-trend" id="rate-trend"></div>
            </div>
          </div>
          <div class="metric-card" id="processing-rate">
            <div class="metric-icon">⚡</div>
            <div class="metric-content">
              <div class="metric-value">0</div>
              <div class="metric-label">Processing Rate (pts/sec)</div>
              <div class="metric-trend" id="processing-trend"></div>
            </div>
          </div>
          <div class="metric-card" id="ai-confidence">
            <div class="metric-icon">🤖</div>
            <div class="metric-content">
              <div class="metric-value">0%</div>
              <div class="metric-label">AI Confidence</div>
              <div class="metric-trend" id="confidence-trend"></div>
            </div>
          </div>
        </div>

        <div class="dashboard-content">
          <div class="main-charts">
            <div class="chart-section">
              <div class="chart-header">
                <h3>Multi-Algorithm Anomaly Timeline</h3>
                <div class="chart-controls">
                  <select id="timeline-interval">
                    <option value="1">Last 1 hour</option>
                    <option value="6">Last 6 hours</option>
                    <option value="24" selected>Last 24 hours</option>
                    <option value="168">Last week</option>
                  </select>
                  <button id="zoom-reset" class="btn-sm btn-secondary">Reset Zoom</button>
                </div>
              </div>
              <div id="timeline-chart" class="chart-container"></div>
            </div>

            <div class="chart-section">
              <div class="chart-header">
                <h3>Real-Time Anomaly Heatmap</h3>
                <div class="chart-controls">
                  <select id="heatmap-aggregation">
                    <option value="minute">Per Minute</option>
                    <option value="hour" selected>Per Hour</option>
                    <option value="day">Per Day</option>
                  </select>
                </div>
              </div>
              <div id="heatmap-chart" class="chart-container"></div>
            </div>

            <div class="chart-section">
              <div class="chart-header">
                <h3>Algorithm Performance Comparison</h3>
                <div class="chart-controls">
                  <select id="performance-metric">
                    <option value="accuracy">Accuracy</option>
                    <option value="precision">Precision</option>
                    <option value="recall">Recall</option>
                    <option value="f1">F1 Score</option>
                    <option value="processing_time">Processing Time</option>
                  </select>
                </div>
              </div>
              <div id="performance-chart" class="chart-container"></div>
            </div>
          </div>

          <div class="sidebar-panels">
            <div class="panel" id="ai-insights-panel">
              <div class="panel-header">
                <h4>🤖 AI Insights</h4>
                <button id="refresh-insights" class="btn-sm btn-secondary">
                  <i class="fas fa-sync"></i>
                </button>
              </div>
              <div class="panel-content">
                <div id="insights-list" class="insights-container">
                  <div class="insight-item loading">
                    <div class="insight-icon">🔍</div>
                    <div class="insight-text">Analyzing patterns...</div>
                  </div>
                </div>
              </div>
            </div>

            <div class="panel" id="alerts-panel">
              <div class="panel-header">
                <h4>🚨 Intelligent Alerts</h4>
                <div class="alert-controls">
                  <button id="alert-settings" class="btn-sm btn-secondary">
                    <i class="fas fa-cog"></i>
                  </button>
                  <button id="clear-alerts" class="btn-sm btn-danger">
                    <i class="fas fa-times"></i>
                  </button>
                </div>
              </div>
              <div class="panel-content">
                <div id="alerts-list" class="alerts-container">
                  <div class="no-alerts">No alerts to display</div>
                </div>
              </div>
            </div>

            <div class="panel" id="predictive-panel">
              <div class="panel-header">
                <h4>🔮 Predictive Analytics</h4>
                <button id="update-predictions" class="btn-sm btn-info">
                  <i class="fas fa-crystal-ball"></i>
                </button>
              </div>
              <div class="panel-content">
                <div id="predictions-container">
                  <div class="prediction-item">
                    <div class="prediction-label">Next 1 hour:</div>
                    <div class="prediction-value" id="prediction-1h">Calculating...</div>
                  </div>
                  <div class="prediction-item">
                    <div class="prediction-label">Next 6 hours:</div>
                    <div class="prediction-value" id="prediction-6h">Calculating...</div>
                  </div>
                  <div class="prediction-item">
                    <div class="prediction-label">Next 24 hours:</div>
                    <div class="prediction-value" id="prediction-24h">Calculating...</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="dashboard-footer">
          <div class="footer-stats">
            <span>System Status: <span id="system-status" class="status-healthy">Healthy</span></span>
            <span>•</span>
            <span>Uptime: <span id="uptime">0m</span></span>
            <span>•</span>
            <span>Memory Usage: <span id="memory-usage">0%</span></span>
            <span>•</span>
            <span>CPU Usage: <span id="cpu-usage">0%</span></span>
          </div>
          <div class="footer-actions">
            <button id="fullscreen-btn" class="btn-link">
              <i class="fas fa-expand"></i> Fullscreen
            </button>
            <button id="settings-btn" class="btn-link">
              <i class="fas fa-cog"></i> Settings
            </button>
          </div>
        </div>
      </div>

      <!-- AI Insights Modal -->
      <div id="ai-insights-modal" class="modal" style="display: none;">
        <div class="modal-content">
          <div class="modal-header">
            <h3>AI-Powered Insights</h3>
            <button id="close-insights-modal" class="btn-close">&times;</button>
          </div>
          <div class="modal-body">
            <div id="detailed-insights"></div>
          </div>
        </div>
      </div>

      <!-- Alert Settings Modal -->
      <div id="alert-settings-modal" class="modal" style="display: none;">
        <div class="modal-content">
          <div class="modal-header">
            <h3>Alert Settings</h3>
            <button id="close-alert-settings" class="btn-close">&times;</button>
          </div>
          <div class="modal-body">
            <div class="form-group">
              <label for="alert-threshold">Alert Threshold:</label>
              <input type="range" id="alert-threshold" min="0.1" max="1.0" step="0.1" value="0.8">
              <span id="alert-threshold-value">0.8</span>
            </div>
            <div class="form-group">
              <label for="alert-cooldown">Alert Cooldown (seconds):</label>
              <input type="number" id="alert-cooldown" min="1" max="3600" value="60">
            </div>
            <div class="form-group">
              <label>
                <input type="checkbox" id="enable-email-alerts" checked>
                Enable Email Alerts
              </label>
            </div>
            <div class="form-group">
              <label>
                <input type="checkbox" id="enable-slack-alerts">
                Enable Slack Alerts
              </label>
            </div>
            <div class="form-group">
              <label>
                <input type="checkbox" id="enable-ai-filtering" checked>
                Enable AI Alert Filtering
              </label>
            </div>
          </div>
          <div class="modal-footer">
            <button id="save-alert-settings" class="btn-primary">Save Settings</button>
            <button id="cancel-alert-settings" class="btn-secondary">Cancel</button>
          </div>
        </div>
      </div>
    `;
  }

  initializeCharts() {
    // Initialize timeline chart with multi-algorithm support
    this.charts.set('timeline', new AnomalyTimelineChart(
      document.getElementById('timeline-chart'),
      {
        multiAlgorithm: true,
        enableZoom: true,
        enableBrush: true,
        showConfidenceIntervals: true,
        maxDataPoints: this.options.maxDataPoints
      }
    ));

    // Initialize heatmap chart
    this.charts.set('heatmap', new AnomalyHeatmapChart(
      document.getElementById('heatmap-chart'),
      {
        enableInteraction: true,
        showTooltips: true,
        enableDrilldown: true
      }
    ));

    // Initialize performance comparison chart
    this.initializePerformanceChart();
  }

  initializePerformanceChart() {
    const performanceChart = document.getElementById('performance-chart');

    // Create performance comparison visualization
    this.charts.set('performance', {
      container: performanceChart,
      update: (data) => {
        this.updatePerformanceChart(data);
      }
    });
  }

  initializeAIComponents() {
    if (this.options.enableAIInsights) {
      this.aiInsightEngine = new AIInsightEngine({
        updateInterval: 30000, // 30 seconds
        analysisDepth: 'deep',
        enablePatternRecognition: true,
        enableAnomalyClassification: true
      });
    }

    if (this.options.enablePredictiveAnalytics) {
      this.anomalyPredictor = new AnomalyPredictor({
        predictionHorizon: [1, 6, 24], // hours
        modelType: 'lstm',
        enableUncertaintyQuantification: true
      });
    }

    if (this.options.enableIntelligentAlerting) {
      this.alertOptimizer = new IntelligentAlertOptimizer({
        enableNoiseReduction: true,
        enableContextualFiltering: true,
        enableSeverityScoring: true
      });
    }
  }

  setupWebSocket() {
    this.websocket = new WebSocket(this.options.websocketUrl);

    this.websocket.onopen = () => {
      this.isConnected = true;
      this.updateConnectionStatus('connected');
      this.startLatencyMonitoring();
    };

    this.websocket.onmessage = (event) => {
      if (!this.isPaused) {
        const data = JSON.parse(event.data);
        this.processIncomingData(data);
      }
    };

    this.websocket.onclose = () => {
      this.isConnected = false;
      this.updateConnectionStatus('disconnected');
      this.reconnectWebSocket();
    };

    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.updateConnectionStatus('error');
    };
  }

  processIncomingData(data) {
    const now = Date.now();

    // Update processing rate
    this.updateProcessingRate(now);

    // Store data in buffers
    this.updateDataBuffers(data);

    // Update algorithm results
    if (data.algorithm_results) {
      this.updateAlgorithmResults(data.algorithm_results);
    }

    // Update charts
    this.updateCharts(data);

    // Update metrics
    this.updateMetrics(data);

    // Generate AI insights
    if (this.aiInsightEngine) {
      this.generateAIInsights(data);
    }

    // Update predictions
    if (this.anomalyPredictor) {
      this.updatePredictions(data);
    }

    // Process alerts
    if (this.alertOptimizer) {
      this.processIntelligentAlerts(data);
    }
  }

  updateDataBuffers(data) {
    // Update main data buffer
    if (!this.dataBuffers.has('main')) {
      this.dataBuffers.set('main', []);
    }

    const mainBuffer = this.dataBuffers.get('main');
    mainBuffer.push({
      timestamp: data.timestamp,
      value: data.value,
      anomaly_score: data.anomaly_score,
      is_anomaly: data.is_anomaly,
      algorithm: data.algorithm || 'default'
    });

    // Keep buffer size manageable
    if (mainBuffer.length > this.options.bufferSize) {
      mainBuffer.shift();
    }

    // Update algorithm-specific buffers
    if (data.algorithm_results) {
      Object.keys(data.algorithm_results).forEach(algorithm => {
        if (!this.dataBuffers.has(algorithm)) {
          this.dataBuffers.set(algorithm, []);
        }

        const algoBuffer = this.dataBuffers.get(algorithm);
        algoBuffer.push({
          timestamp: data.timestamp,
          ...data.algorithm_results[algorithm]
        });

        if (algoBuffer.length > this.options.bufferSize) {
          algoBuffer.shift();
        }
      });
    }
  }

  updateAlgorithmResults(results) {
    Object.keys(results).forEach(algorithm => {
      if (!this.algorithmResults.has(algorithm)) {
        this.algorithmResults.set(algorithm, {
          total_predictions: 0,
          anomaly_count: 0,
          accuracy: 0,
          precision: 0,
          recall: 0,
          f1_score: 0,
          processing_time: 0
        });
      }

      const algoStats = this.algorithmResults.get(algorithm);
      const result = results[algorithm];

      // Update statistics
      algoStats.total_predictions++;
      if (result.is_anomaly) {
        algoStats.anomaly_count++;
      }

      // Update performance metrics (if available)
      if (result.performance_metrics) {
        Object.assign(algoStats, result.performance_metrics);
      }

      algoStats.processing_time = result.processing_time || 0;
    });
  }

  updateCharts(data) {
    // Update timeline chart
    const timelineChart = this.charts.get('timeline');
    if (timelineChart) {
      timelineChart.addDataPoint(data);
    }

    // Update heatmap chart
    const heatmapChart = this.charts.get('heatmap');
    if (heatmapChart) {
      heatmapChart.updateHeatmap(data);
    }

    // Update performance chart
    this.updatePerformanceChart(this.algorithmResults);
  }

  updatePerformanceChart(algorithmResults) {
    const performanceChart = this.charts.get('performance');
    if (!performanceChart) return;

    const selectedMetric = document.getElementById('performance-metric').value;
    const chartData = [];

    algorithmResults.forEach((stats, algorithm) => {
      chartData.push({
        algorithm: algorithm,
        value: stats[selectedMetric] || 0,
        total_predictions: stats.total_predictions
      });
    });

    // Update the performance chart visualization
    this.renderPerformanceChart(chartData, selectedMetric);
  }

  renderPerformanceChart(data, metric) {
    const container = document.getElementById('performance-chart');
    container.innerHTML = '';

    // Create simple bar chart for performance comparison
    const chart = d3.select(container)
      .append('svg')
      .attr('width', container.clientWidth)
      .attr('height', 300);

    const margin = { top: 20, right: 30, bottom: 40, left: 60 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const g = chart.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleBand()
      .domain(data.map(d => d.algorithm))
      .range([0, width])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.value)])
      .range([height, 0]);

    // Bars
    g.selectAll('.bar')
      .data(data)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.algorithm))
      .attr('y', d => yScale(d.value))
      .attr('width', xScale.bandwidth())
      .attr('height', d => height - yScale(d.value))
      .attr('fill', '#3b82f6')
      .on('mouseover', function(event, d) {
        // Show tooltip
        const tooltip = d3.select('body').append('div')
          .attr('class', 'tooltip')
          .style('opacity', 0);

        tooltip.transition()
          .duration(200)
          .style('opacity', .9);

        tooltip.html(`${d.algorithm}<br/>${metric}: ${d.value.toFixed(3)}<br/>Predictions: ${d.total_predictions}`)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', function() {
        d3.selectAll('.tooltip').remove();
      });

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append('g')
      .call(d3.axisLeft(yScale));
  }

  updateMetrics(data) {
    this.totalDataPoints++;
    if (data.is_anomaly) {
      this.alertCount++;
    }

    // Update metric displays
    document.getElementById('total-points').querySelector('.metric-value').textContent =
      this.totalDataPoints.toLocaleString();

    document.getElementById('anomaly-count').querySelector('.metric-value').textContent =
      this.alertCount.toLocaleString();

    const anomalyRate = (this.alertCount / this.totalDataPoints * 100).toFixed(1);
    document.getElementById('anomaly-rate').querySelector('.metric-value').textContent =
      `${anomalyRate}%`;

    document.getElementById('processing-rate').querySelector('.metric-value').textContent =
      Math.round(this.processingRate);

    // Update AI confidence if available
    if (data.ai_confidence) {
      document.getElementById('ai-confidence').querySelector('.metric-value').textContent =
        `${Math.round(data.ai_confidence * 100)}%`;
    }
  }

  updateProcessingRate(now) {
    const timeDiff = now - this.lastProcessingTime;
    this.processingRate = 1000 / timeDiff; // points per second
    this.lastProcessingTime = now;
  }

  generateAIInsights(data) {
    if (!this.aiInsightEngine) return;

    // Generate insights based on current data patterns
    this.aiInsightEngine.analyze(data, this.dataBuffers.get('main'))
      .then(insights => {
        this.displayAIInsights(insights);
      })
      .catch(error => {
        console.error('AI Insights generation failed:', error);
      });
  }

  displayAIInsights(insights) {
    const insightsContainer = document.getElementById('insights-list');
    insightsContainer.innerHTML = '';

    insights.forEach(insight => {
      const insightElement = document.createElement('div');
      insightElement.className = `insight-item ${insight.severity}`;
      insightElement.innerHTML = `
        <div class="insight-icon">${insight.icon}</div>
        <div class="insight-content">
          <div class="insight-text">${insight.message}</div>
          <div class="insight-confidence">Confidence: ${Math.round(insight.confidence * 100)}%</div>
          <div class="insight-time">${new Date(insight.timestamp).toLocaleTimeString()}</div>
        </div>
      `;
      insightsContainer.appendChild(insightElement);
    });
  }

  updatePredictions(data) {
    if (!this.anomalyPredictor) return;

    this.anomalyPredictor.predict(data, this.dataBuffers.get('main'))
      .then(predictions => {
        this.displayPredictions(predictions);
      })
      .catch(error => {
        console.error('Prediction generation failed:', error);
      });
  }

  displayPredictions(predictions) {
    predictions.forEach(prediction => {
      const element = document.getElementById(`prediction-${prediction.horizon}`);
      if (element) {
        element.innerHTML = `
          <div class="prediction-probability">${Math.round(prediction.probability * 100)}%</div>
          <div class="prediction-confidence">±${Math.round(prediction.uncertainty * 100)}%</div>
        `;
        element.className = `prediction-value ${prediction.severity}`;
      }
    });
  }

  processIntelligentAlerts(data) {
    if (!this.alertOptimizer) return;

    this.alertOptimizer.processAlert(data, this.alertHistory)
      .then(alert => {
        if (alert.shouldShow) {
          this.displayAlert(alert);
        }
      })
      .catch(error => {
        console.error('Alert processing failed:', error);
      });
  }

  displayAlert(alert) {
    const alertsContainer = document.getElementById('alerts-list');

    // Remove "no alerts" message if present
    const noAlerts = alertsContainer.querySelector('.no-alerts');
    if (noAlerts) {
      noAlerts.remove();
    }

    const alertElement = document.createElement('div');
    alertElement.className = `alert-item ${alert.severity}`;
    alertElement.innerHTML = `
      <div class="alert-icon">${alert.icon}</div>
      <div class="alert-content">
        <div class="alert-title">${alert.title}</div>
        <div class="alert-message">${alert.message}</div>
        <div class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</div>
      </div>
      <div class="alert-actions">
        <button class="btn-sm btn-secondary" onclick="this.parentElement.parentElement.remove()">
          <i class="fas fa-times"></i>
        </button>
      </div>
    `;

    alertsContainer.insertBefore(alertElement, alertsContainer.firstChild);

    // Keep only last 20 alerts
    const alerts = alertsContainer.querySelectorAll('.alert-item');
    if (alerts.length > 20) {
      alerts[alerts.length - 1].remove();
    }

    // Store in history
    this.alertHistory.push(alert);
    if (this.alertHistory.length > 100) {
      this.alertHistory.shift();
    }
  }

  setupEventHandlers() {
    // Pause/Resume button
    document.getElementById('pause-btn').addEventListener('click', () => {
      this.isPaused = !this.isPaused;
      const btn = document.getElementById('pause-btn');
      btn.innerHTML = this.isPaused ?
        '<i class="fas fa-play"></i> Resume' :
        '<i class="fas fa-pause"></i> Pause';
    });

    // Clear data button
    document.getElementById('clear-btn').addEventListener('click', () => {
      this.clearAllData();
    });

    // Export button
    document.getElementById('export-btn').addEventListener('click', () => {
      this.exportData();
    });

    // AI Insights button
    document.getElementById('ai-insights-btn').addEventListener('click', () => {
      this.showDetailedInsights();
    });

    // Algorithm selector
    document.getElementById('algorithm-selector').addEventListener('change', (e) => {
      this.filterByAlgorithm(e.target.value);
    });

    // Sensitivity slider
    document.getElementById('sensitivity-slider').addEventListener('input', (e) => {
      document.getElementById('sensitivity-value').textContent = e.target.value;
      this.updateSensitivity(parseFloat(e.target.value));
    });

    // Performance metric selector
    document.getElementById('performance-metric').addEventListener('change', () => {
      this.updatePerformanceChart(this.algorithmResults);
    });

    // Modal event handlers
    this.setupModalHandlers();
  }

  setupModalHandlers() {
    // AI Insights modal
    document.getElementById('close-insights-modal').addEventListener('click', () => {
      document.getElementById('ai-insights-modal').style.display = 'none';
    });

    // Alert settings modal
    document.getElementById('alert-settings').addEventListener('click', () => {
      document.getElementById('alert-settings-modal').style.display = 'block';
    });

    document.getElementById('close-alert-settings').addEventListener('click', () => {
      document.getElementById('alert-settings-modal').style.display = 'none';
    });

    document.getElementById('save-alert-settings').addEventListener('click', () => {
      this.saveAlertSettings();
    });

    document.getElementById('cancel-alert-settings').addEventListener('click', () => {
      document.getElementById('alert-settings-modal').style.display = 'none';
    });

    // Clear alerts
    document.getElementById('clear-alerts').addEventListener('click', () => {
      document.getElementById('alerts-list').innerHTML = '<div class="no-alerts">No alerts to display</div>';
      this.alertHistory = [];
    });
  }

  clearAllData() {
    // Clear all data buffers
    this.dataBuffers.clear();
    this.algorithmResults.clear();
    this.aiInsights = [];
    this.alertHistory = [];

    // Reset counters
    this.totalDataPoints = 0;
    this.alertCount = 0;
    this.processingRate = 0;

    // Clear charts
    this.charts.forEach(chart => {
      if (chart.clear) {
        chart.clear();
      }
    });

    // Reset metrics display
    this.updateMetrics({ is_anomaly: false });
  }

  exportData() {
    const exportData = {
      timestamp: new Date().toISOString(),
      totalDataPoints: this.totalDataPoints,
      alertCount: this.alertCount,
      algorithmResults: Object.fromEntries(this.algorithmResults),
      dataBuffers: Object.fromEntries(this.dataBuffers),
      aiInsights: this.aiInsights,
      alertHistory: this.alertHistory
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `anomaly-detection-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  showDetailedInsights() {
    const modal = document.getElementById('ai-insights-modal');
    const detailedInsights = document.getElementById('detailed-insights');

    // Generate detailed insights report
    detailedInsights.innerHTML = `
      <div class="insights-report">
        <h4>Pattern Analysis</h4>
        <div class="insight-section">
          <p>Analyzing ${this.totalDataPoints} data points across ${this.algorithmResults.size} algorithms...</p>
          <!-- More detailed insights would be generated here -->
        </div>

        <h4>Anomaly Classification</h4>
        <div class="insight-section">
          <p>Detected ${this.alertCount} anomalies with ${((this.alertCount / this.totalDataPoints) * 100).toFixed(1)}% anomaly rate.</p>
          <!-- Classification details would be shown here -->
        </div>

        <h4>Recommendations</h4>
        <div class="insight-section">
          <ul>
            <li>Consider adjusting sensitivity based on current patterns</li>
            <li>Monitor algorithm performance variations</li>
            <li>Review alert thresholds for optimization</li>
          </ul>
        </div>
      </div>
    `;

    modal.style.display = 'block';
  }

  saveAlertSettings() {
    const settings = {
      threshold: parseFloat(document.getElementById('alert-threshold').value),
      cooldown: parseInt(document.getElementById('alert-cooldown').value),
      enableEmail: document.getElementById('enable-email-alerts').checked,
      enableSlack: document.getElementById('enable-slack-alerts').checked,
      enableAIFiltering: document.getElementById('enable-ai-filtering').checked
    };

    // Save settings (would typically be sent to server)
    console.log('Saving alert settings:', settings);

    // Update alert optimizer if available
    if (this.alertOptimizer) {
      this.alertOptimizer.updateSettings(settings);
    }

    document.getElementById('alert-settings-modal').style.display = 'none';
  }

  updateConnectionStatus(status) {
    const statusElement = document.getElementById('connection-status');
    const indicator = statusElement.querySelector('.status-indicator');
    const text = statusElement.querySelector('.status-text');

    indicator.className = `status-indicator ${status}`;

    switch (status) {
      case 'connected':
        text.textContent = 'Connected';
        break;
      case 'disconnected':
        text.textContent = 'Disconnected';
        break;
      case 'error':
        text.textContent = 'Connection Error';
        break;
      default:
        text.textContent = 'Connecting...';
    }
  }

  startLatencyMonitoring() {
    setInterval(() => {
      if (this.isConnected) {
        const start = Date.now();
        this.websocket.send(JSON.stringify({ type: 'ping', timestamp: start }));

        // Listen for pong response (would need server implementation)
        // For now, simulate latency
        const simulatedLatency = Math.random() * 50 + 10; // 10-60ms
        document.getElementById('latency-info').textContent = `${Math.round(simulatedLatency)}ms`;
      }
    }, 5000);
  }

  startHeartbeat() {
    setInterval(() => {
      if (this.isConnected) {
        this.websocket.send(JSON.stringify({ type: 'heartbeat' }));
      }
    }, 30000);
  }

  initializePerformanceMonitoring() {
    // Monitor system performance
    setInterval(() => {
      this.updateSystemMetrics();
    }, 5000);
  }

  updateSystemMetrics() {
    // Simulate system metrics (would typically come from server)
    const memoryUsage = Math.random() * 30 + 20; // 20-50%
    const cpuUsage = Math.random() * 40 + 10; // 10-50%
    const uptime = Math.floor((Date.now() - this.startTime) / 1000 / 60); // minutes

    document.getElementById('memory-usage').textContent = `${Math.round(memoryUsage)}%`;
    document.getElementById('cpu-usage').textContent = `${Math.round(cpuUsage)}%`;
    document.getElementById('uptime').textContent = `${uptime}m`;

    // Update system status based on metrics
    const status = (memoryUsage > 80 || cpuUsage > 80) ? 'degraded' : 'healthy';
    const statusElement = document.getElementById('system-status');
    statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    statusElement.className = `status-${status}`;
  }

  filterByAlgorithm(algorithm) {
    // Filter charts and data by selected algorithm
    this.charts.forEach(chart => {
      if (chart.filterByAlgorithm) {
        chart.filterByAlgorithm(algorithm);
      }
    });
  }

  updateSensitivity(sensitivity) {
    // Update sensitivity for all algorithms
    this.options.alertThreshold = sensitivity;

    // Send sensitivity update to server
    if (this.isConnected) {
      this.websocket.send(JSON.stringify({
        type: 'update_sensitivity',
        sensitivity: sensitivity
      }));
    }
  }

  reconnectWebSocket() {
    setTimeout(() => {
      if (!this.isConnected) {
        this.setupWebSocket();
      }
    }, 5000);
  }

  destroy() {
    // Clean up resources
    if (this.websocket) {
      this.websocket.close();
    }

    this.charts.forEach(chart => {
      if (chart.destroy) {
        chart.destroy();
      }
    });

    this.dataBuffers.clear();
    this.algorithmResults.clear();
  }
}

// AI Insight Engine (mock implementation)
class AIInsightEngine {
  constructor(options) {
    this.options = options;
  }

  async analyze(data, dataBuffer) {
    // Simulate AI analysis
    await new Promise(resolve => setTimeout(resolve, 100));

    const insights = [];

    // Pattern recognition insights
    if (dataBuffer.length > 100) {
      const anomalyRate = dataBuffer.filter(d => d.is_anomaly).length / dataBuffer.length;

      if (anomalyRate > 0.1) {
        insights.push({
          icon: '🚨',
          message: 'Anomaly rate is significantly higher than normal',
          severity: 'high',
          confidence: 0.85,
          timestamp: Date.now()
        });
      }

      if (anomalyRate < 0.01) {
        insights.push({
          icon: '✅',
          message: 'System operating within normal parameters',
          severity: 'info',
          confidence: 0.92,
          timestamp: Date.now()
        });
      }
    }

    return insights;
  }
}

// Anomaly Predictor (mock implementation)
class AnomalyPredictor {
  constructor(options) {
    this.options = options;
  }

  async predict(data, dataBuffer) {
    // Simulate prediction
    await new Promise(resolve => setTimeout(resolve, 150));

    const predictions = [];

    this.options.predictionHorizon.forEach(horizon => {
      const probability = Math.random() * 0.3 + 0.1; // 10-40% probability
      const uncertainty = Math.random() * 0.2 + 0.05; // 5-25% uncertainty

      predictions.push({
        horizon: `${horizon}h`,
        probability: probability,
        uncertainty: uncertainty,
        severity: probability > 0.25 ? 'high' : probability > 0.15 ? 'medium' : 'low'
      });
    });

    return predictions;
  }
}

// Intelligent Alert Optimizer (mock implementation)
class IntelligentAlertOptimizer {
  constructor(options) {
    this.options = options;
    this.lastAlertTime = 0;
    this.alertCooldown = 60000; // 1 minute
  }

  async processAlert(data, alertHistory) {
    // Simulate intelligent alert processing
    await new Promise(resolve => setTimeout(resolve, 50));

    const now = Date.now();
    const shouldShow = data.is_anomaly &&
                      data.anomaly_score > 0.8 &&
                      (now - this.lastAlertTime) > this.alertCooldown;

    if (shouldShow) {
      this.lastAlertTime = now;

      return {
        shouldShow: true,
        icon: '⚠️',
        title: 'Anomaly Detected',
        message: `High anomaly score detected: ${data.anomaly_score.toFixed(3)}`,
        severity: data.anomaly_score > 0.9 ? 'critical' : 'high',
        timestamp: now
      };
    }

    return { shouldShow: false };
  }

  updateSettings(settings) {
    this.alertCooldown = settings.cooldown * 1000;
    // Update other settings
  }
}

// Export the enhanced dashboard
window.EnhancedRealTimeDashboard = EnhancedRealTimeDashboard;
