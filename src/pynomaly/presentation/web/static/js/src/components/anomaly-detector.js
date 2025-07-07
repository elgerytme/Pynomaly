// Anomaly Detector Component - Production-ready anomaly detection interface
import { ChartComponent } from './chart-components.js';

export class AnomalyDetector {
  constructor(element) {
    this.element = element;
    this.config = this.getConfig();
    this.state = {
      isProcessing: false,
      currentDataset: null,
      selectedAlgorithm: null,
      results: null,
      realTimeMode: false
    };
    
    this.charts = new Map();
    this.websocket = null;
    
    this.init();
  }

  // Initialize the detector component
  init() {
    console.log('🔍 Initializing Anomaly Detector component');
    
    this.createInterface();
    this.bindEvents();
    this.loadAlgorithms();
    this.setupRealTimeCapabilities();
    
    // Announce component ready
    this.element.dispatchEvent(new CustomEvent('component:ready', {
      detail: { component: 'anomaly-detector', element: this.element }
    }));
  }

  // Get component configuration from data attributes
  getConfig() {
    const element = this.element;
    return {
      apiEndpoint: element.dataset.apiEndpoint || '/api/anomaly-detection',
      websocketUrl: element.dataset.websocketUrl || '/ws/anomaly-detection',
      autoDetect: element.dataset.autoDetect === 'true',
      realTime: element.dataset.realTime === 'true',
      maxFileSize: parseInt(element.dataset.maxFileSize) || 10 * 1024 * 1024, // 10MB
      allowedFormats: (element.dataset.allowedFormats || 'csv,json,parquet').split(','),
      algorithms: element.dataset.algorithms ? JSON.parse(element.dataset.algorithms) : null
    };
  }

  // Create the detector interface
  createInterface() {
    this.element.innerHTML = `
      <div class="anomaly-detector-container">
        <!-- Header -->
        <div class="detector-header card-header">
          <h3 class="text-lg font-semibold text-neutral-900">Anomaly Detection</h3>
          <div class="detector-controls flex items-center space-x-2">
            <button class="btn btn-sm btn-outline" data-action="toggle-realtime">
              <span class="realtime-icon">📡</span>
              <span class="realtime-text">Real-time</span>
            </button>
            <button class="btn btn-sm btn-ghost" data-action="toggle-settings">
              <span>⚙️</span>
            </button>
          </div>
        </div>

        <!-- Main Content -->
        <div class="detector-content">
          <!-- Step 1: Data Input -->
          <div class="detection-step" data-step="data-input">
            <div class="step-header">
              <h4 class="text-md font-medium">1. Select Data Source</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>
            
            <div class="data-input-section">
              <div class="input-tabs">
                <button class="tab-button active" data-tab="upload">Upload File</button>
                <button class="tab-button" data-tab="dataset">Existing Dataset</button>
                <button class="tab-button" data-tab="stream">Real-time Stream</button>
              </div>
              
              <!-- File Upload Tab -->
              <div class="tab-content active" data-tab-content="upload">
                <div class="file-upload-area" data-drop-zone>
                  <div class="upload-icon">📁</div>
                  <div class="upload-text">
                    <p class="text-sm font-medium">Drop your data file here or click to browse</p>
                    <p class="text-xs text-neutral-500">Supported formats: CSV, JSON, Parquet (max ${this.config.maxFileSize / 1024 / 1024}MB)</p>
                  </div>
                  <input type="file" class="file-input" accept=".csv,.json,.parquet" hidden>
                </div>
                <div class="file-info hidden" data-file-info></div>
              </div>
              
              <!-- Existing Dataset Tab -->
              <div class="tab-content" data-tab-content="dataset">
                <div class="dataset-selector">
                  <select class="form-select" data-dataset-select>
                    <option value="">Select a dataset...</option>
                  </select>
                  <button class="btn btn-sm btn-outline" data-action="refresh-datasets">
                    <span>🔄</span> Refresh
                  </button>
                </div>
                <div class="dataset-info hidden" data-dataset-info></div>
              </div>
              
              <!-- Real-time Stream Tab -->
              <div class="tab-content" data-tab-content="stream">
                <div class="stream-config">
                  <div class="form-group">
                    <label class="form-label">Stream Source</label>
                    <select class="form-select" data-stream-source>
                      <option value="">Select stream source...</option>
                      <option value="kafka">Kafka Topic</option>
                      <option value="mqtt">MQTT</option>
                      <option value="websocket">WebSocket</option>
                      <option value="api">REST API Polling</option>
                    </select>
                  </div>
                  <div class="stream-connection-config hidden" data-stream-config></div>
                  <button class="btn btn-primary" data-action="connect-stream">Connect Stream</button>
                </div>
              </div>
            </div>
          </div>

          <!-- Step 2: Algorithm Selection -->
          <div class="detection-step" data-step="algorithm-selection">
            <div class="step-header">
              <h4 class="text-md font-medium">2. Choose Detection Algorithm</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>
            
            <div class="algorithm-selection">
              <div class="algorithm-tabs">
                <button class="tab-button active" data-algo-tab="recommended">Recommended</button>
                <button class="tab-button" data-algo-tab="all">All Algorithms</button>
                <button class="tab-button" data-algo-tab="ensemble">Ensemble</button>
                <button class="tab-button" data-algo-tab="custom">Custom</button>
              </div>
              
              <div class="algorithm-grid" data-algorithm-grid></div>
              
              <div class="algorithm-params hidden" data-algorithm-params>
                <h5 class="text-sm font-medium mb-2">Algorithm Parameters</h5>
                <div class="params-form" data-params-form></div>
              </div>
            </div>
          </div>

          <!-- Step 3: Detection Execution -->
          <div class="detection-step" data-step="execution">
            <div class="step-header">
              <h4 class="text-md font-medium">3. Run Detection</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>
            
            <div class="execution-controls">
              <button class="btn btn-primary btn-lg" data-action="start-detection" disabled>
                <span class="btn-icon">🚀</span>
                <span class="btn-text">Start Detection</span>
              </button>
              
              <div class="execution-options">
                <label class="checkbox-label">
                  <input type="checkbox" data-option="explain-results">
                  <span class="checkmark"></span>
                  Generate explanations
                </label>
                <label class="checkbox-label">
                  <input type="checkbox" data-option="save-model">
                  <span class="checkmark"></span>
                  Save trained model
                </label>
                <label class="checkbox-label">
                  <input type="checkbox" data-option="auto-threshold">
                  <span class="checkmark"></span>
                  Auto-optimize threshold
                </label>
              </div>
            </div>
            
            <div class="execution-progress hidden" data-execution-progress>
              <div class="progress-bar">
                <div class="progress-fill" data-progress-fill></div>
              </div>
              <div class="progress-text" data-progress-text>Initializing...</div>
              <button class="btn btn-sm btn-accent" data-action="cancel-detection">Cancel</button>
            </div>
          </div>

          <!-- Step 4: Results Visualization -->
          <div class="detection-step" data-step="results">
            <div class="step-header">
              <h4 class="text-md font-medium">4. Detection Results</h4>
              <div class="step-status status-normal" data-status="pending">Pending</div>
            </div>
            
            <div class="results-container hidden" data-results-container>
              <!-- Results Summary -->
              <div class="results-summary">
                <div class="metric-cards">
                  <div class="metric-card">
                    <div class="metric-value" data-metric="total-samples">-</div>
                    <div class="metric-label">Total Samples</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value text-accent-600" data-metric="anomalies-detected">-</div>
                    <div class="metric-label">Anomalies Detected</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value" data-metric="anomaly-rate">-</div>
                    <div class="metric-label">Anomaly Rate</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-value" data-metric="confidence-score">-</div>
                    <div class="metric-label">Confidence Score</div>
                  </div>
                </div>
              </div>
              
              <!-- Results Visualization -->
              <div class="results-visualization">
                <div class="chart-tabs">
                  <button class="tab-button active" data-chart-tab="scatter">Scatter Plot</button>
                  <button class="tab-button" data-chart-tab="timeline">Timeline</button>
                  <button class="tab-button" data-chart-tab="distribution">Distribution</button>
                  <button class="tab-button" data-chart-tab="heatmap">Feature Heatmap</button>
                </div>
                
                <div class="chart-container" data-chart-container>
                  <div class="chart-loading skeleton h-64"></div>
                </div>
                
                <div class="chart-controls">
                  <div class="threshold-control">
                    <label class="form-label">Anomaly Threshold</label>
                    <input type="range" class="threshold-slider" data-threshold-slider min="0" max="1" step="0.01" value="0.5">
                    <span class="threshold-value" data-threshold-value>0.5</span>
                  </div>
                  
                  <div class="filter-controls">
                    <button class="btn btn-sm btn-outline" data-filter="show-all">Show All</button>
                    <button class="btn btn-sm btn-outline" data-filter="show-anomalies">Anomalies Only</button>
                    <button class="btn btn-sm btn-outline" data-filter="show-normal">Normal Only</button>
                  </div>
                </div>
              </div>
              
              <!-- Results Actions -->
              <div class="results-actions">
                <button class="btn btn-primary" data-action="export-results">
                  <span>📊</span> Export Results
                </button>
                <button class="btn btn-secondary" data-action="save-model">
                  <span>💾</span> Save Model
                </button>
                <button class="btn btn-outline" data-action="generate-report">
                  <span>📄</span> Generate Report
                </button>
                <button class="btn btn-ghost" data-action="explain-results">
                  <span>🔍</span> Explain Results
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Settings Panel -->
        <div class="settings-panel hidden" data-settings-panel>
          <div class="settings-header">
            <h4 class="text-md font-medium">Detection Settings</h4>
            <button class="btn btn-sm btn-ghost" data-action="close-settings">✕</button>
          </div>
          
          <div class="settings-content">
            <div class="setting-group">
              <label class="form-label">Default Algorithm</label>
              <select class="form-select" data-setting="default-algorithm">
                <option value="auto">Auto-select</option>
                <option value="isolation-forest">Isolation Forest</option>
                <option value="one-class-svm">One-Class SVM</option>
                <option value="lof">Local Outlier Factor</option>
              </select>
            </div>
            
            <div class="setting-group">
              <label class="form-label">Contamination Rate</label>
              <input type="number" class="form-input" data-setting="contamination-rate" min="0" max="1" step="0.01" value="0.1">
            </div>
            
            <div class="setting-group">
              <label class="checkbox-label">
                <input type="checkbox" data-setting="auto-preprocess">
                <span class="checkmark"></span>
                Auto-preprocess data
              </label>
            </div>
            
            <div class="setting-group">
              <label class="checkbox-label">
                <input type="checkbox" data-setting="enable-notifications">
                <span class="checkmark"></span>
                Enable notifications
              </label>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // Bind event listeners
  bindEvents() {
    const element = this.element;
    
    // Tab switching
    element.addEventListener('click', (e) => {
      if (e.target.matches('.tab-button')) {
        this.switchTab(e.target);
      }
    });
    
    // File upload handling
    const fileInput = element.querySelector('.file-input');
    const dropZone = element.querySelector('[data-drop-zone]');
    
    fileInput?.addEventListener('change', (e) => {
      this.handleFileUpload(e.target.files[0]);
    });
    
    dropZone?.addEventListener('click', () => fileInput?.click());
    dropZone?.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    
    dropZone?.addEventListener('dragleave', () => {
      dropZone.classList.remove('drag-over');
    });
    
    dropZone?.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      this.handleFileUpload(e.dataTransfer.files[0]);
    });
    
    // Action button handlers
    element.addEventListener('click', (e) => {
      const action = e.target.closest('[data-action]')?.dataset.action;
      if (action) {
        this.handleAction(action, e.target);
      }
    });
    
    // Algorithm selection
    element.addEventListener('click', (e) => {
      if (e.target.matches('.algorithm-card')) {
        this.selectAlgorithm(e.target);
      }
    });
    
    // Real-time threshold adjustment
    const thresholdSlider = element.querySelector('[data-threshold-slider]');
    thresholdSlider?.addEventListener('input', (e) => {
      this.updateThreshold(parseFloat(e.target.value));
    });
    
    // Keyboard shortcuts
    element.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'Enter':
            e.preventDefault();
            this.startDetection();
            break;
          case 'r':
            e.preventDefault();
            this.resetDetector();
            break;
        }
      }
    });
  }

  // Load available algorithms
  async loadAlgorithms() {
    try {
      const response = await fetch('/api/algorithms');
      const algorithms = await response.json();
      
      this.renderAlgorithmGrid(algorithms);
      this.updateAlgorithmRecommendations(algorithms);
    } catch (error) {
      console.error('Failed to load algorithms:', error);
      this.showError('Failed to load detection algorithms');
    }
  }

  // Render algorithm selection grid
  renderAlgorithmGrid(algorithms) {
    const grid = this.element.querySelector('[data-algorithm-grid]');
    if (!grid) return;
    
    grid.innerHTML = algorithms.map(algo => `
      <div class="algorithm-card" data-algorithm="${algo.id}">
        <div class="algorithm-header">
          <h5 class="algorithm-name">${algo.name}</h5>
          <div class="algorithm-type">${algo.type}</div>
        </div>
        <div class="algorithm-description">${algo.description}</div>
        <div class="algorithm-metrics">
          <div class="metric">
            <span class="metric-label">Accuracy</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: ${algo.accuracy * 100}%"></div>
            </div>
          </div>
          <div class="metric">
            <span class="metric-label">Speed</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: ${algo.speed * 100}%"></div>
            </div>
          </div>
        </div>
        <div class="algorithm-tags">
          ${algo.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
        </div>
      </div>
    `).join('');
  }

  // Handle file upload
  handleFileUpload(file) {
    if (!file) return;
    
    // Validate file type
    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (!this.config.allowedFormats.includes(fileExtension)) {
      this.showError(`Unsupported file format: ${fileExtension}`);
      return;
    }
    
    // Validate file size
    if (file.size > this.config.maxFileSize) {
      this.showError(`File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB (max: ${this.config.maxFileSize / 1024 / 1024}MB)`);
      return;
    }
    
    // Show file info
    this.showFileInfo(file);
    
    // Upload file
    this.uploadFile(file);
  }

  // Upload file to server
  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      this.updateStepStatus('data-input', 'processing');
      
      const response = await fetch('/api/datasets/upload', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      this.state.currentDataset = result.dataset;
      
      this.updateStepStatus('data-input', 'completed');
      this.enableStep('algorithm-selection');
      
      this.announceToScreenReader(`File uploaded successfully: ${file.name}`);
    } catch (error) {
      console.error('Upload failed:', error);
      this.updateStepStatus('data-input', 'error');
      this.showError(`Upload failed: ${error.message}`);
    }
  }

  // Start anomaly detection
  async startDetection() {
    if (!this.state.currentDataset || !this.state.selectedAlgorithm) {
      this.showError('Please select data and algorithm first');
      return;
    }
    
    this.state.isProcessing = true;
    this.updateStepStatus('execution', 'processing');
    
    const progressContainer = this.element.querySelector('[data-execution-progress]');
    progressContainer?.classList.remove('hidden');
    
    try {
      const params = this.getAlgorithmParams();
      const options = this.getDetectionOptions();
      
      const response = await fetch('/api/anomaly-detection/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          dataset_id: this.state.currentDataset.id,
          algorithm: this.state.selectedAlgorithm,
          parameters: params,
          options: options
        })
      });
      
      if (!response.ok) {
        throw new Error(`Detection failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      this.state.results = result;
      
      this.updateStepStatus('execution', 'completed');
      this.updateStepStatus('results', 'completed');
      
      this.showResults(result);
      this.announceToScreenReader(`Detection completed. Found ${result.anomaly_count} anomalies.`);
      
    } catch (error) {
      console.error('Detection failed:', error);
      this.updateStepStatus('execution', 'error');
      this.showError(`Detection failed: ${error.message}`);
    } finally {
      this.state.isProcessing = false;
      progressContainer?.classList.add('hidden');
    }
  }

  // Show detection results
  showResults(results) {
    const container = this.element.querySelector('[data-results-container]');
    container?.classList.remove('hidden');
    
    // Update metrics
    this.updateMetric('total-samples', results.total_samples.toLocaleString());
    this.updateMetric('anomalies-detected', results.anomaly_count.toLocaleString());
    this.updateMetric('anomaly-rate', `${(results.anomaly_rate * 100).toFixed(2)}%`);
    this.updateMetric('confidence-score', results.confidence_score.toFixed(3));
    
    // Create visualizations
    this.createResultsCharts(results);
  }

  // Create results charts
  createResultsCharts(results) {
    const chartContainer = this.element.querySelector('[data-chart-container]');
    if (!chartContainer) return;
    
    // Create scatter plot
    const scatterChart = new ChartComponent(chartContainer, {
      type: 'scatter',
      data: results.visualization_data.scatter,
      options: {
        title: 'Anomaly Detection Results',
        xAxis: { title: 'Feature 1' },
        yAxis: { title: 'Feature 2' },
        colorScale: {
          normal: '#22c55e',
          anomaly: '#ef4444'
        }
      }
    });
    
    this.charts.set('scatter', scatterChart);
  }

  // Utility methods
  switchTab(button) {
    const tabGroup = button.closest('.input-tabs, .algorithm-tabs, .chart-tabs');
    const tabName = button.dataset.tab || button.dataset.algoTab || button.dataset.chartTab;
    
    // Update button states
    tabGroup.querySelectorAll('.tab-button').forEach(btn => {
      btn.classList.remove('active');
    });
    button.classList.add('active');
    
    // Update content visibility
    const contentContainer = tabGroup.nextElementSibling;
    contentContainer.querySelectorAll('.tab-content').forEach(content => {
      content.classList.remove('active');
    });
    
    const targetContent = contentContainer.querySelector(`[data-tab-content="${tabName}"]`);
    targetContent?.classList.add('active');
  }

  updateStepStatus(step, status) {
    const stepElement = this.element.querySelector(`[data-step="${step}"]`);
    const statusElement = stepElement?.querySelector('[data-status]');
    
    if (statusElement) {
      statusElement.className = `step-status status-${status}`;
      statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
      statusElement.dataset.status = status;
    }
  }

  enableStep(step) {
    const stepElement = this.element.querySelector(`[data-step="${step}"]`);
    stepElement?.classList.remove('disabled');
  }

  updateMetric(metric, value) {
    const metricElement = this.element.querySelector(`[data-metric="${metric}"]`);
    if (metricElement) {
      metricElement.textContent = value;
    }
  }

  showError(message) {
    this.element.dispatchEvent(new CustomEvent('component:error', {
      detail: { component: 'anomaly-detector', message }
    }));
  }

  announceToScreenReader(message) {
    // Use the global app's announce method if available
    if (window.PynomalyApp) {
      window.PynomalyApp.announceToScreenReader(message);
    }
  }

  // Setup real-time capabilities
  setupRealTimeCapabilities() {
    if (this.config.realTime) {
      this.initWebSocket();
    }
  }

  // Initialize WebSocket for real-time updates
  initWebSocket() {
    if (!this.config.websocketUrl) return;
    
    try {
      this.websocket = new WebSocket(this.config.websocketUrl);
      
      this.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleRealTimeUpdate(data);
      };
      
      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  }

  // Handle real-time updates
  handleRealTimeUpdate(data) {
    if (data.type === 'anomaly_detected') {
      this.showAnomalyAlert(data.anomaly);
    } else if (data.type === 'model_updated') {
      this.refreshResults();
    }
  }

  // Handle action button clicks
  handleAction(action, button) {
    switch (action) {
      case 'start-detection':
        this.startDetection();
        break;
      case 'toggle-realtime':
        this.toggleRealTimeMode();
        break;
      case 'export-results':
        this.exportResults();
        break;
      case 'generate-report':
        this.generateReport();
        break;
      default:
        console.warn(`Unknown action: ${action}`);
    }
  }

  // Get algorithm parameters from form
  getAlgorithmParams() {
    const paramsForm = this.element.querySelector('[data-params-form]');
    if (!paramsForm) return {};
    
    const formData = new FormData(paramsForm);
    const params = {};
    
    for (const [key, value] of formData.entries()) {
      params[key] = value;
    }
    
    return params;
  }

  // Get detection options
  getDetectionOptions() {
    return {
      explain_results: this.element.querySelector('[data-option="explain-results"]')?.checked || false,
      save_model: this.element.querySelector('[data-option="save-model"]')?.checked || false,
      auto_threshold: this.element.querySelector('[data-option="auto-threshold"]')?.checked || false
    };
  }

  // Cleanup
  destroy() {
    if (this.websocket) {
      this.websocket.close();
    }
    
    this.charts.forEach(chart => {
      if (chart.destroy) {
        chart.destroy();
      }
    });
    
    this.charts.clear();
  }
}

// Auto-initialize components
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-component="anomaly-detector"]').forEach(element => {
    new AnomalyDetector(element);
  });
});
