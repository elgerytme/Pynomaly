/**
 * Advanced Dashboard Layout System with Enhanced Real-Time Capabilities
 * Combines drag-and-drop layout with real-time data streaming and WebSocket integration
 */

import { dashboardState } from '../state/dashboard-state.js';
import { DashboardLayoutEngine } from './dashboard-layout.js';
import { RealTimeDashboard } from '../charts/real-time-dashboard.js';

export class AdvancedDashboard {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      websocketUrl: 'ws://localhost:8000/ws/dashboard',
      enableDragDrop: true,
      enableRealTime: true,
      autoSave: true,
      collaborationMode: false,
      maxDataPoints: 10000,
      updateInterval: 1000,
      gridSize: 20,
      maxColumns: 12,
      ...options
    };
    
    this.layoutEngine = null;
    this.realTimeDashboard = null;
    this.websocket = null;
    this.isConnected = false;
    this.isPaused = false;
    this.dataStreams = new Map();
    this.subscribers = new Set();
    this.performanceMetrics = new Map();
    this.collaborationState = new Map();
    
    this.init();
  }
  
  async init() {
    try {
      console.log('🚀 Initializing Advanced Dashboard...');
      
      // Create main layout structure
      this.createDashboardStructure();
      
      // Initialize layout engine
      this.initializeLayoutEngine();
      
      // Initialize real-time capabilities
      if (this.options.enableRealTime) {
        await this.initializeRealTime();
      }
      
      // Setup WebSocket connection
      this.setupWebSocket();
      
      // Initialize performance monitoring
      this.initializePerformanceMonitoring();
      
      // Setup event handlers
      this.setupEventHandlers();
      
      // Load saved configuration
      await this.loadDashboardConfiguration();
      
      console.log('✅ Advanced Dashboard initialized successfully');
      
    } catch (error) {
      console.error('❌ Failed to initialize Advanced Dashboard:', error);
      this.showErrorState(error);
    }
  }
  
  createDashboardStructure() {
    this.container.className = `${this.container.className} advanced-dashboard`.trim();
    this.container.innerHTML = `
      <!-- Dashboard Header -->
      <div class="dashboard-header" role="banner">
        <div class="header-left">
          <h1 class="dashboard-title">Pynomaly Advanced Dashboard</h1>
          <div class="dashboard-status">
            <div class="status-indicator" id="connection-status">
              <span class="status-dot disconnected"></span>
              <span class="status-text">Initializing...</span>
            </div>
            <div class="performance-indicator" id="performance-status">
              <span class="perf-icon">⚡</span>
              <span class="perf-text">--ms</span>
            </div>
          </div>
        </div>
        
        <div class="header-center">
          <div class="dashboard-controls" role="group" aria-label="Dashboard controls">
            <button class="btn btn-secondary btn-sm" id="add-widget-btn" aria-label="Add new widget">
              <svg class="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z"/>
              </svg>
              Add Widget
            </button>
            
            <button class="btn btn-secondary btn-sm" id="layout-menu-btn" aria-label="Layout options">
              <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM11 13a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/>
              </svg>
            </button>
            
            <div class="btn-group" role="group" aria-label="Real-time controls">
              <button class="btn btn-secondary btn-sm" id="pause-btn" aria-label="Pause real-time updates">
                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zM11 8a1 1 0 112 0v4a1 1 0 11-2 0V8z"/>
                </svg>
                Pause
              </button>
              
              <button class="btn btn-secondary btn-sm" id="clear-btn" aria-label="Clear all data">
                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" clip-rule="evenodd"/>
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414L9.586 12l-3.293 3.293a1 1 0 101.414 1.414L10 13.414l2.293 2.293a1 1 0 001.414-1.414L10.414 12l3.293-3.293z" clip-rule="evenodd"/>
                </svg>
                Clear
              </button>
            </div>
          </div>
        </div>
        
        <div class="header-right">
          <div class="dashboard-metrics-summary">
            <div class="metric-summary" id="data-rate-summary">
              <span class="metric-label">Data Rate:</span>
              <span class="metric-value">0 pts/sec</span>
            </div>
            <div class="metric-summary" id="anomaly-rate-summary">
              <span class="metric-label">Anomalies:</span>
              <span class="metric-value">0.0%</span>
            </div>
          </div>
          
          <div class="dashboard-actions">
            <button class="btn btn-primary btn-sm" id="export-btn" aria-label="Export dashboard data">
              <svg class="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd"/>
              </svg>
              Export
            </button>
            
            <button class="btn btn-secondary btn-sm" id="settings-btn" aria-label="Dashboard settings">
              <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
      
      <!-- Dashboard Content -->
      <div class="dashboard-content" role="main">
        <!-- Sidebar -->
        <aside class="dashboard-sidebar" id="dashboard-sidebar" role="complementary">
          <div class="sidebar-header">
            <h3>Dashboard Tools</h3>
            <button class="sidebar-toggle" id="sidebar-toggle" aria-label="Toggle sidebar">
              <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clip-rule="evenodd"/>
              </svg>
            </button>
          </div>
          
          <div class="sidebar-content">
            <!-- Widget Library -->
            <div class="widget-library">
              <h4>Available Widgets</h4>
              <div class="widget-list">
                <div class="widget-item" data-widget-type="anomaly-timeline" draggable="true">
                  <div class="widget-icon">📈</div>
                  <div class="widget-info">
                    <div class="widget-name">Timeline Chart</div>
                    <div class="widget-description">Anomalies over time</div>
                  </div>
                </div>
                
                <div class="widget-item" data-widget-type="anomaly-heatmap" draggable="true">
                  <div class="widget-icon">🔥</div>
                  <div class="widget-info">
                    <div class="widget-name">Heatmap</div>
                    <div class="widget-description">Feature correlation</div>
                  </div>
                </div>
                
                <div class="widget-item" data-widget-type="metrics-summary" draggable="true">
                  <div class="widget-icon">📊</div>
                  <div class="widget-info">
                    <div class="widget-name">Metrics</div>
                    <div class="widget-description">Key statistics</div>
                  </div>
                </div>
                
                <div class="widget-item" data-widget-type="alert-list" draggable="true">
                  <div class="widget-icon">🚨</div>
                  <div class="widget-info">
                    <div class="widget-name">Alerts</div>
                    <div class="widget-description">Recent notifications</div>
                  </div>
                </div>
                
                <div class="widget-item" data-widget-type="data-stream" draggable="true">
                  <div class="widget-icon">⚡</div>
                  <div class="widget-info">
                    <div class="widget-name">Live Stream</div>
                    <div class="widget-description">Real-time data</div>
                  </div>
                </div>
                
                <div class="widget-item" data-widget-type="model-performance" draggable="true">
                  <div class="widget-icon">🎯</div>
                  <div class="widget-info">
                    <div class="widget-name">Model Stats</div>
                    <div class="widget-description">ML performance</div>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- Data Sources -->
            <div class="data-sources">
              <h4>Data Sources</h4>
              <div class="source-list">
                <div class="source-item active" data-source="real-time">
                  <div class="source-indicator connected"></div>
                  <div class="source-name">Real-time Stream</div>
                  <div class="source-status">Connected</div>
                </div>
                
                <div class="source-item" data-source="historical">
                  <div class="source-indicator disconnected"></div>
                  <div class="source-name">Historical Data</div>
                  <div class="source-status">Available</div>
                </div>
                
                <div class="source-item" data-source="simulation">
                  <div class="source-indicator warning"></div>
                  <div class="source-name">Simulation Mode</div>
                  <div class="source-status">Standby</div>
                </div>
              </div>
            </div>
            
            <!-- Dashboard Settings -->
            <div class="dashboard-settings">
              <h4>Settings</h4>
              <div class="setting-group">
                <label class="setting-item">
                  <input type="checkbox" id="auto-refresh" checked>
                  <span class="setting-label">Auto Refresh</span>
                </label>
                
                <label class="setting-item">
                  <input type="checkbox" id="sound-alerts">
                  <span class="setting-label">Sound Alerts</span>
                </label>
                
                <label class="setting-item">
                  <input type="checkbox" id="compact-mode">
                  <span class="setting-label">Compact Mode</span>
                </label>
                
                <div class="setting-item">
                  <label class="setting-label">Update Rate</label>
                  <select id="update-rate">
                    <option value="500">500ms</option>
                    <option value="1000" selected>1s</option>
                    <option value="2000">2s</option>
                    <option value="5000">5s</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        </aside>
        
        <!-- Main Dashboard Area -->
        <main class="dashboard-main" id="dashboard-main">
          <div class="layout-container" id="layout-container" data-component="dashboard-layout">
            <!-- Drag-and-drop layout area -->
            <div class="empty-dashboard">
              <div class="empty-icon">📊</div>
              <h3>Welcome to Advanced Dashboard</h3>
              <p>Drag widgets from the sidebar to get started, or click "Add Widget" to begin building your custom dashboard.</p>
              <button class="btn btn-primary" id="quick-start-btn">Quick Start</button>
            </div>
          </div>
        </main>
      </div>
      
      <!-- Dashboard Overlays -->
      <div class="dashboard-overlays">
        <!-- Drop Zone Indicator -->
        <div class="drop-zone-indicator" id="drop-zone-indicator" style="display: none;">
          <div class="drop-zone-content">
            <div class="drop-zone-icon">📍</div>
            <div class="drop-zone-text">Drop widget here</div>
          </div>
        </div>
        
        <!-- Performance Overlay -->
        <div class="performance-overlay" id="performance-overlay" style="display: none;">
          <div class="performance-content">
            <h4>Performance Monitor</h4>
            <div class="performance-metrics">
              <div class="perf-metric">
                <span class="perf-label">FPS:</span>
                <span class="perf-value" id="fps-value">60</span>
              </div>
              <div class="perf-metric">
                <span class="perf-label">Memory:</span>
                <span class="perf-value" id="memory-value">--MB</span>
              </div>
              <div class="perf-metric">
                <span class="perf-label">Render:</span>
                <span class="perf-value" id="render-time">--ms</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Loading Overlay -->
      <div class="loading-overlay" id="loading-overlay" style="display: none;">
        <div class="loading-content">
          <div class="loading-spinner"></div>
          <div class="loading-text">Initializing dashboard...</div>
        </div>
      </div>
    `;
  }
  
  initializeLayoutEngine() {
    const layoutContainer = this.container.querySelector('#layout-container');
    
    this.layoutEngine = new DashboardLayoutEngine(layoutContainer, {
      gridSize: this.options.gridSize,
      maxColumns: this.options.maxColumns,
      persistLayout: this.options.autoSave,
      enableDragDrop: this.options.enableDragDrop,
      animationDuration: 300
    });
    
    // Subscribe to layout events
    this.layoutEngine.container.addEventListener('widgetAdded', (e) => {
      this.handleWidgetAdded(e.detail);
    });
    
    this.layoutEngine.container.addEventListener('widgetRemoved', (e) => {
      this.handleWidgetRemoved(e.detail);
    });
    
    this.layoutEngine.container.addEventListener('widgetMoved', (e) => {
      this.handleWidgetMoved(e.detail);
    });
    
    this.layoutEngine.container.addEventListener('widgetResized', (e) => {
      this.handleWidgetResized(e.detail);
    });
  }
  
  async initializeRealTime() {
    try {
      // Initialize real-time data processing
      this.setupDataStreams();
      
      // Initialize real-time dashboard if needed
      const realTimeContainer = document.createElement('div');
      realTimeContainer.id = 'real-time-dashboard';
      realTimeContainer.style.display = 'none';
      this.container.appendChild(realTimeContainer);
      
      this.realTimeDashboard = new RealTimeDashboard(realTimeContainer, {
        websocketUrl: this.options.websocketUrl,
        updateInterval: this.options.updateInterval,
        maxDataPoints: this.options.maxDataPoints
      });
      
    } catch (error) {
      console.error('Failed to initialize real-time capabilities:', error);
    }
  }
  
  setupDataStreams() {
    // Create data stream handlers for different types
    const streamTypes = [
      'anomaly-detection',
      'system-metrics', 
      'model-performance',
      'user-activity',
      'error-logs'
    ];
    
    streamTypes.forEach(streamType => {
      this.dataStreams.set(streamType, {
        buffer: new CircularBuffer(this.options.maxDataPoints),
        subscribers: new Set(),
        lastUpdate: null,
        isActive: false,
        metrics: {
          dataRate: 0,
          errorRate: 0,
          latency: 0
        }
      });
    });
  }
  
  setupWebSocket() {
    if (!this.options.enableRealTime) return;
    
    try {
      console.log(`🔌 Connecting to WebSocket: ${this.options.websocketUrl}`);
      
      this.websocket = new WebSocket(this.options.websocketUrl);
      
      this.websocket.onopen = () => {
        console.log('✅ WebSocket connected');
        this.isConnected = true;
        this.updateConnectionStatus('connected');
        
        // Send initial configuration
        this.sendWebSocketMessage({
          type: 'configure',
          config: {
            streams: Array.from(this.dataStreams.keys()),
            updateInterval: this.options.updateInterval,
            compression: true
          }
        });
      };
      
      this.websocket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleWebSocketMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      this.websocket.onclose = (event) => {
        console.warn('🔌 WebSocket disconnected:', event.code, event.reason);
        this.isConnected = false;
        this.updateConnectionStatus('disconnected');
        
        // Attempt reconnection
        if (!event.wasClean) {
          setTimeout(() => this.setupWebSocket(), 5000);
        }
      };
      
      this.websocket.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        this.updateConnectionStatus('error');
      };
      
    } catch (error) {
      console.error('Failed to setup WebSocket:', error);
      this.updateConnectionStatus('error');
    }
  }
  
  initializePerformanceMonitoring() {
    // Track FPS
    let frameCount = 0;
    let lastTime = performance.now();
    
    const measureFPS = () => {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime - lastTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
        this.updatePerformanceMetric('fps', fps);
        
        frameCount = 0;
        lastTime = currentTime;
      }
      
      requestAnimationFrame(measureFPS);
    };
    
    requestAnimationFrame(measureFPS);
    
    // Track memory usage
    if (performance.memory) {
      setInterval(() => {
        const memoryMB = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
        this.updatePerformanceMetric('memory', memoryMB);
      }, 2000);
    }
    
    // Track render times
    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        if (entry.name.includes('dashboard')) {
          this.updatePerformanceMetric('renderTime', Math.round(entry.duration));
        }
      });
    });
    
    observer.observe({ entryTypes: ['measure'] });
  }
  
  setupEventHandlers() {
    // Add widget button
    this.container.querySelector('#add-widget-btn').addEventListener('click', () => {
      this.showWidgetSelector();
    });
    
    // Layout menu button
    this.container.querySelector('#layout-menu-btn').addEventListener('click', () => {
      this.showLayoutMenu();
    });
    
    // Pause button
    this.container.querySelector('#pause-btn').addEventListener('click', () => {
      this.togglePause();
    });
    
    // Clear button
    this.container.querySelector('#clear-btn').addEventListener('click', () => {
      this.clearAllData();
    });
    
    // Export button
    this.container.querySelector('#export-btn').addEventListener('click', () => {
      this.exportDashboard();
    });
    
    // Settings button
    this.container.querySelector('#settings-btn').addEventListener('click', () => {
      this.showSettings();
    });
    
    // Sidebar toggle
    this.container.querySelector('#sidebar-toggle').addEventListener('click', () => {
      this.toggleSidebar();
    });
    
    // Quick start button
    this.container.querySelector('#quick-start-btn')?.addEventListener('click', () => {
      this.quickStart();
    });
    
    // Widget drag and drop from sidebar
    this.setupWidgetDragDrop();
    
    // Settings controls
    this.setupSettingsControls();
    
    // Keyboard shortcuts
    this.setupKeyboardShortcuts();
  }
  
  setupWidgetDragDrop() {
    const widgetItems = this.container.querySelectorAll('.widget-item');
    const layoutContainer = this.container.querySelector('#layout-container');
    
    widgetItems.forEach(item => {
      item.addEventListener('dragstart', (e) => {
        const widgetType = item.dataset.widgetType;
        e.dataTransfer.setData('text/plain', widgetType);
        e.dataTransfer.effectAllowed = 'copy';
        
        // Show drop zone indicator
        this.showDropZoneIndicator();
      });
      
      item.addEventListener('dragend', () => {
        this.hideDropZoneIndicator();
      });
    });
    
    layoutContainer.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    });
    
    layoutContainer.addEventListener('drop', (e) => {
      e.preventDefault();
      const widgetType = e.dataTransfer.getData('text/plain');
      
      if (widgetType) {
        this.addWidgetFromDrop(widgetType, e.clientX, e.clientY);
      }
      
      this.hideDropZoneIndicator();
    });
  }
  
  setupSettingsControls() {
    // Auto refresh toggle
    const autoRefreshToggle = this.container.querySelector('#auto-refresh');
    autoRefreshToggle?.addEventListener('change', (e) => {
      dashboardState.dispatch(dashboardState.actions.setPreference('autoRefresh', e.target.checked));
    });
    
    // Sound alerts toggle
    const soundAlertsToggle = this.container.querySelector('#sound-alerts');
    soundAlertsToggle?.addEventListener('change', (e) => {
      dashboardState.dispatch(dashboardState.actions.setPreference('soundEnabled', e.target.checked));
    });
    
    // Compact mode toggle
    const compactModeToggle = this.container.querySelector('#compact-mode');
    compactModeToggle?.addEventListener('change', (e) => {
      dashboardState.dispatch(dashboardState.actions.setPreference('compactMode', e.target.checked));
      this.updateCompactMode(e.target.checked);
    });
    
    // Update rate selector
    const updateRateSelect = this.container.querySelector('#update-rate');
    updateRateSelect?.addEventListener('change', (e) => {
      const newRate = parseInt(e.target.value);
      this.options.updateInterval = newRate;
      dashboardState.dispatch(dashboardState.actions.setUpdateRate(newRate));
      this.updateWebSocketConfiguration();
    });
  }
  
  setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Only handle shortcuts when dashboard has focus
      if (!this.container.contains(e.target)) return;
      
      const shortcuts = {
        'KeyA': (e) => { // Ctrl+A - Add widget
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            this.showWidgetSelector();
          }
        },
        'KeyP': (e) => { // Ctrl+P - Pause/Resume
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            this.togglePause();
          }
        },
        'KeyE': (e) => { // Ctrl+E - Export
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            this.exportDashboard();
          }
        },
        'KeyS': (e) => { // Ctrl+S - Settings
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            this.showSettings();
          }
        },
        'KeyT': (e) => { // Ctrl+T - Toggle sidebar
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            this.toggleSidebar();
          }
        },
        'Escape': () => { // Escape - Close overlays
          this.closeAllOverlays();
        }
      };
      
      const handler = shortcuts[e.code];
      if (handler) {
        handler(e);
      }
    });
  }
  
  // WebSocket message handling
  handleWebSocketMessage(message) {
    switch (message.type) {
      case 'data':
        this.handleDataMessage(message);
        break;
      case 'alert':
        this.handleAlertMessage(message);
        break;
      case 'status':
        this.handleStatusMessage(message);
        break;
      case 'configuration':
        this.handleConfigurationMessage(message);
        break;
      case 'collaboration':
        this.handleCollaborationMessage(message);
        break;
      default:
        console.warn('Unknown WebSocket message type:', message.type);
    }
  }
  
  handleDataMessage(message) {
    const { stream, data, timestamp } = message;
    
    if (this.isPaused) return;
    
    const streamInfo = this.dataStreams.get(stream);
    if (!streamInfo) return;
    
    // Add data to buffer
    streamInfo.buffer.push({
      ...data,
      timestamp: timestamp || Date.now()
    });
    
    // Update stream metrics
    streamInfo.lastUpdate = Date.now();
    streamInfo.isActive = true;
    
    // Notify subscribers
    streamInfo.subscribers.forEach(callback => {
      try {
        callback(data, stream);
      } catch (error) {
        console.error('Error in stream subscriber:', error);
      }
    });
    
    // Update dashboard metrics
    this.updateDashboardMetrics();
  }
  
  handleAlertMessage(message) {
    const { severity, title, description, timestamp, data } = message;
    
    // Add to dashboard state
    dashboardState.dispatch(dashboardState.actions.addAlert({
      severity,
      title,
      description,
      timestamp: timestamp || Date.now(),
      data
    }));
    
    // Show notification
    this.showNotification({
      type: severity,
      title,
      message: description,
      duration: severity === 'critical' ? 0 : 5000 // Critical alerts don't auto-dismiss
    });
    
    // Play sound if enabled
    if (dashboardState.getStateSlice('preferences.soundEnabled')) {
      this.playAlertSound(severity);
    }
  }
  
  handleStatusMessage(message) {
    const { component, status, details } = message;
    
    console.log(`Status update for ${component}:`, status, details);
    
    // Update UI status indicators
    this.updateComponentStatus(component, status, details);
  }
  
  handleConfigurationMessage(message) {
    const { config } = message;
    
    // Update dashboard configuration
    Object.assign(this.options, config);
    
    // Apply configuration changes
    this.applyConfiguration(config);
  }
  
  handleCollaborationMessage(message) {
    if (!this.options.collaborationMode) return;
    
    const { action, user, data } = message;
    
    // Handle collaborative actions
    switch (action) {
      case 'widget_added':
        this.handleCollaborativeWidgetAdd(user, data);
        break;
      case 'widget_moved':
        this.handleCollaborativeWidgetMove(user, data);
        break;
      case 'user_joined':
        this.handleUserJoined(user);
        break;
      case 'user_left':
        this.handleUserLeft(user);
        break;
    }
  }
  
  // Widget management
  addWidgetFromDrop(widgetType, x, y) {
    const layoutContainer = this.container.querySelector('#layout-container');
    const rect = layoutContainer.getBoundingClientRect();
    
    // Calculate grid position
    const gridX = Math.floor((x - rect.left) / (this.options.gridSize * 2));
    const gridY = Math.floor((y - rect.top) / this.options.gridSize);
    
    this.addWidget(widgetType, { x: gridX, y: gridY });
  }
  
  addWidget(widgetType, position = null) {
    if (!this.layoutEngine) {
      console.error('Layout engine not initialized');
      return;
    }
    
    const widgetConfig = this.getWidgetConfig(widgetType);
    if (position) {
      widgetConfig.x = position.x;
      widgetConfig.y = position.y;
    }
    
    const widget = this.layoutEngine.addWidget(widgetConfig);
    
    // Hide empty state if this is the first widget
    const emptyDashboard = this.container.querySelector('.empty-dashboard');
    if (emptyDashboard) {
      emptyDashboard.style.display = 'none';
    }
    
    return widget;
  }
  
  getWidgetConfig(widgetType) {
    const configs = {
      'anomaly-timeline': {
        type: 'anomaly-timeline',
        title: 'Anomaly Timeline',
        width: 6,
        height: 4,
        options: {
          showLegend: true,
          interactive: true,
          realTime: true
        }
      },
      'anomaly-heatmap': {
        type: 'anomaly-heatmap',
        title: 'Anomaly Heatmap',
        width: 6,
        height: 5,
        options: {
          colorScheme: 'RdYlBu',
          showValues: false,
          aggregation: '15min'
        }
      },
      'metrics-summary': {
        type: 'metrics-summary',
        title: 'Key Metrics',
        width: 4,
        height: 3,
        options: {
          showTrends: true,
          compactMode: false
        }
      },
      'alert-list': {
        type: 'alert-list',
        title: 'Recent Alerts',
        width: 4,
        height: 4,
        options: {
          maxItems: 10,
          showTimestamps: true
        }
      },
      'data-stream': {
        type: 'data-stream',
        title: 'Live Data Stream',
        width: 8,
        height: 3,
        options: {
          streamType: 'anomaly-detection',
          bufferSize: 100
        }
      },
      'model-performance': {
        type: 'model-performance',
        title: 'Model Performance',
        width: 5,
        height: 4,
        options: {
          showAccuracy: true,
          showLatency: true,
          showThroughput: true
        }
      }
    };
    
    return configs[widgetType] || configs['metrics-summary'];
  }
  
  // Event handlers
  handleWidgetAdded(detail) {
    const { widget, layout } = detail;
    
    // Subscribe widget to relevant data streams
    this.subscribeWidgetToStreams(widget);
    
    // Send collaboration message if enabled
    if (this.options.collaborationMode && this.isConnected) {
      this.sendWebSocketMessage({
        type: 'collaboration',
        action: 'widget_added',
        data: { widgetType: widget.type, layout }
      });
    }
    
    console.log('Widget added:', widget.type, layout);
  }
  
  handleWidgetRemoved(detail) {
    const { widgetId } = detail;
    
    // Unsubscribe from data streams
    this.unsubscribeWidgetFromStreams(widgetId);
    
    console.log('Widget removed:', widgetId);
  }
  
  handleWidgetMoved(detail) {
    const { widget, layout } = detail;
    
    // Send collaboration message if enabled
    if (this.options.collaborationMode && this.isConnected) {
      this.sendWebSocketMessage({
        type: 'collaboration',
        action: 'widget_moved',
        data: { widgetId: widget.id, layout }
      });
    }
    
    console.log('Widget moved:', widget.id, layout);
  }
  
  handleWidgetResized(detail) {
    const { widget, layout } = detail;
    
    // Update widget content to fit new size
    if (widget.component && typeof widget.component.resize === 'function') {
      widget.component.resize();
    }
    
    console.log('Widget resized:', widget.id, layout);
  }
  
  // Stream subscription management
  subscribeWidgetToStreams(widget) {
    const streamMappings = {
      'anomaly-timeline': ['anomaly-detection'],
      'anomaly-heatmap': ['anomaly-detection', 'system-metrics'],
      'metrics-summary': ['anomaly-detection', 'system-metrics'],
      'alert-list': ['anomaly-detection'],
      'data-stream': [widget.options?.streamType || 'anomaly-detection'],
      'model-performance': ['model-performance']
    };
    
    const streams = streamMappings[widget.type] || [];
    
    streams.forEach(streamType => {
      const streamInfo = this.dataStreams.get(streamType);
      if (streamInfo) {
        const callback = (data) => this.updateWidgetData(widget, data, streamType);
        streamInfo.subscribers.add(callback);
        
        // Store callback reference for cleanup
        if (!widget.streamCallbacks) {
          widget.streamCallbacks = new Map();
        }
        widget.streamCallbacks.set(streamType, callback);
      }
    });
  }
  
  unsubscribeWidgetFromStreams(widgetId) {
    // Find widget and remove all stream subscriptions
    const widget = this.layoutEngine.widgets.get(widgetId);
    if (widget && widget.streamCallbacks) {
      widget.streamCallbacks.forEach((callback, streamType) => {
        const streamInfo = this.dataStreams.get(streamType);
        if (streamInfo) {
          streamInfo.subscribers.delete(callback);
        }
      });
      widget.streamCallbacks.clear();
    }
  }
  
  updateWidgetData(widget, data, streamType) {
    if (widget.component && typeof widget.component.updateData === 'function') {
      widget.component.updateData(data, streamType);
    }
  }
  
  // Dashboard controls
  togglePause() {
    this.isPaused = !this.isPaused;
    
    const pauseBtn = this.container.querySelector('#pause-btn');
    pauseBtn.textContent = this.isPaused ? 'Resume' : 'Pause';
    pauseBtn.setAttribute('aria-label', this.isPaused ? 'Resume real-time updates' : 'Pause real-time updates');
    
    // Update dashboard state
    dashboardState.dispatch(dashboardState.actions.toggleRealTimePause());
    
    console.log(this.isPaused ? 'Dashboard paused' : 'Dashboard resumed');
  }
  
  clearAllData() {
    // Clear all data streams
    this.dataStreams.forEach(streamInfo => {
      streamInfo.buffer.clear();
    });
    
    // Clear dashboard state
    dashboardState.dispatch(dashboardState.actions.clearAnomalies());
    dashboardState.dispatch(dashboardState.actions.clearAlerts());
    
    // Update all widgets
    this.layoutEngine.widgets.forEach(widget => {
      if (widget.component && typeof widget.component.clear === 'function') {
        widget.component.clear();
      }
    });
    
    console.log('All dashboard data cleared');
  }
  
  toggleSidebar() {
    const sidebar = this.container.querySelector('#dashboard-sidebar');
    sidebar.classList.toggle('collapsed');
    
    const isCollapsed = sidebar.classList.contains('collapsed');
    dashboardState.dispatch(dashboardState.actions.toggleSidebar());
    
    console.log(isCollapsed ? 'Sidebar collapsed' : 'Sidebar expanded');
  }
  
  quickStart() {
    // Add a few default widgets to get started
    const defaultWidgets = [
      { type: 'metrics-summary', x: 0, y: 0 },
      { type: 'anomaly-timeline', x: 4, y: 0 },
      { type: 'alert-list', x: 0, y: 3 },
      { type: 'data-stream', x: 4, y: 4 }
    ];
    
    defaultWidgets.forEach(widget => {
      this.addWidget(widget.type, { x: widget.x, y: widget.y });
    });
    
    console.log('Quick start completed - default widgets added');
  }
  
  // UI updates
  updateConnectionStatus(status) {
    const statusIndicator = this.container.querySelector('#connection-status');
    const statusDot = statusIndicator.querySelector('.status-dot');
    const statusText = statusIndicator.querySelector('.status-text');
    
    statusDot.className = `status-dot ${status}`;
    
    const statusMessages = {
      'connected': 'Connected',
      'disconnected': 'Disconnected',
      'connecting': 'Connecting...',
      'error': 'Connection Error'
    };
    
    statusText.textContent = statusMessages[status] || 'Unknown';
  }
  
  updatePerformanceMetric(metric, value) {
    this.performanceMetrics.set(metric, value);
    
    // Update UI
    switch (metric) {
      case 'fps':
        const fpsElement = this.container.querySelector('#fps-value');
        if (fpsElement) fpsElement.textContent = value;
        break;
      case 'memory':
        const memoryElement = this.container.querySelector('#memory-value');
        if (memoryElement) memoryElement.textContent = `${value}MB`;
        break;
      case 'renderTime':
        const renderElement = this.container.querySelector('#render-time');
        if (renderElement) renderElement.textContent = `${value}ms`;
        
        // Update header performance indicator
        const perfText = this.container.querySelector('.perf-text');
        if (perfText) perfText.textContent = `${value}ms`;
        break;
    }
  }
  
  updateDashboardMetrics() {
    // Calculate metrics across all streams
    let totalDataPoints = 0;
    let totalAnomalies = 0;
    let dataRate = 0;
    
    this.dataStreams.forEach((streamInfo, streamType) => {
      const buffer = streamInfo.buffer;
      totalDataPoints += buffer.size();
      
      if (streamType === 'anomaly-detection') {
        const anomalies = buffer.toArray().filter(item => item.isAnomaly);
        totalAnomalies += anomalies.length;
      }
      
      // Calculate data rate (last minute)
      const oneMinuteAgo = Date.now() - 60000;
      const recentData = buffer.toArray().filter(item => item.timestamp > oneMinuteAgo);
      dataRate += recentData.length / 60; // per second
    });
    
    const anomalyRate = totalDataPoints > 0 ? (totalAnomalies / totalDataPoints) * 100 : 0;
    
    // Update header metrics
    const dataRateSummary = this.container.querySelector('#data-rate-summary .metric-value');
    if (dataRateSummary) dataRateSummary.textContent = `${Math.round(dataRate)} pts/sec`;
    
    const anomalyRateSummary = this.container.querySelector('#anomaly-rate-summary .metric-value');
    if (anomalyRateSummary) anomalyRateSummary.textContent = `${anomalyRate.toFixed(1)}%`;
    
    // Update dashboard state
    dashboardState.dispatch(dashboardState.actions.updateMetrics({
      totalDataPoints,
      anomalyCount: totalAnomalies,
      anomalyRate: anomalyRate / 100,
      dataRate
    }));
  }
  
  updateCompactMode(enabled) {
    this.container.classList.toggle('compact-mode', enabled);
    
    // Update all widgets to use compact mode
    this.layoutEngine.widgets.forEach(widget => {
      if (widget.component && typeof widget.component.setCompactMode === 'function') {
        widget.component.setCompactMode(enabled);
      }
    });
  }
  
  // Overlay management
  showDropZoneIndicator() {
    const indicator = this.container.querySelector('#drop-zone-indicator');
    indicator.style.display = 'flex';
  }
  
  hideDropZoneIndicator() {
    const indicator = this.container.querySelector('#drop-zone-indicator');
    indicator.style.display = 'none';
  }
  
  showLoadingOverlay(text = 'Loading...') {
    const overlay = this.container.querySelector('#loading-overlay');
    const loadingText = overlay.querySelector('.loading-text');
    loadingText.textContent = text;
    overlay.style.display = 'flex';
  }
  
  hideLoadingOverlay() {
    const overlay = this.container.querySelector('#loading-overlay');
    overlay.style.display = 'none';
  }
  
  closeAllOverlays() {
    const overlays = this.container.querySelectorAll('.dashboard-overlays > div');
    overlays.forEach(overlay => {
      overlay.style.display = 'none';
    });
  }
  
  // Dialog/modal management
  showWidgetSelector() {
    // Create and show widget selector modal
    const modal = this.createWidgetSelectorModal();
    document.body.appendChild(modal);
  }
  
  showLayoutMenu() {
    // Create and show layout menu modal
    const modal = this.createLayoutMenuModal();
    document.body.appendChild(modal);
  }
  
  showSettings() {
    // Create and show settings modal
    const modal = this.createSettingsModal();
    document.body.appendChild(modal);
  }
  
  createWidgetSelectorModal() {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
      <div class="modal-dialog modal-lg">
        <div class="modal-header">
          <h3>Add Widget</h3>
          <button class="modal-close">×</button>
        </div>
        <div class="modal-body">
          <div class="widget-gallery">
            ${this.renderWidgetGallery()}
          </div>
        </div>
      </div>
    `;
    
    // Setup event handlers
    modal.querySelector('.modal-close').addEventListener('click', () => {
      modal.remove();
    });
    
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.remove();
    });
    
    modal.querySelectorAll('.widget-gallery-item').forEach(item => {
      item.addEventListener('click', () => {
        const widgetType = item.dataset.widgetType;
        this.addWidget(widgetType);
        modal.remove();
      });
    });
    
    return modal;
  }
  
  renderWidgetGallery() {
    const widgets = [
      { type: 'anomaly-timeline', icon: '📈', name: 'Timeline Chart', description: 'Anomaly detection over time' },
      { type: 'anomaly-heatmap', icon: '🔥', name: 'Heatmap', description: 'Feature correlation analysis' },
      { type: 'metrics-summary', icon: '📊', name: 'Metrics Summary', description: 'Key performance indicators' },
      { type: 'alert-list', icon: '🚨', name: 'Alert List', description: 'Recent alerts and notifications' },
      { type: 'data-stream', icon: '⚡', name: 'Data Stream', description: 'Live data visualization' },
      { type: 'model-performance', icon: '🎯', name: 'Model Performance', description: 'ML model statistics' }
    ];
    
    return widgets.map(widget => `
      <div class="widget-gallery-item" data-widget-type="${widget.type}">
        <div class="widget-gallery-icon">${widget.icon}</div>
        <div class="widget-gallery-info">
          <div class="widget-gallery-name">${widget.name}</div>
          <div class="widget-gallery-description">${widget.description}</div>
        </div>
      </div>
    `).join('');
  }
  
  createLayoutMenuModal() {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
      <div class="modal-dialog">
        <div class="modal-header">
          <h3>Layout Options</h3>
          <button class="modal-close">×</button>
        </div>
        <div class="modal-body">
          <div class="layout-options">
            <button class="btn btn-secondary" id="save-layout">Save Layout</button>
            <button class="btn btn-secondary" id="load-layout">Load Layout</button>
            <button class="btn btn-secondary" id="reset-layout">Reset Layout</button>
            <button class="btn btn-secondary" id="export-layout">Export Layout</button>
          </div>
        </div>
      </div>
    `;
    
    // Setup event handlers
    modal.querySelector('.modal-close').addEventListener('click', () => {
      modal.remove();
    });
    
    modal.querySelector('#save-layout').addEventListener('click', () => {
      this.saveLayoutConfiguration();
      modal.remove();
    });
    
    modal.querySelector('#load-layout').addEventListener('click', () => {
      this.loadLayoutConfiguration();
      modal.remove();
    });
    
    modal.querySelector('#reset-layout').addEventListener('click', () => {
      this.resetLayout();
      modal.remove();
    });
    
    modal.querySelector('#export-layout').addEventListener('click', () => {
      this.exportLayoutConfiguration();
      modal.remove();
    });
    
    return modal;
  }
  
  createSettingsModal() {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
      <div class="modal-dialog modal-lg">
        <div class="modal-header">
          <h3>Dashboard Settings</h3>
          <button class="modal-close">×</button>
        </div>
        <div class="modal-body">
          <div class="settings-form">
            ${this.renderSettingsForm()}
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-secondary" id="cancel-settings">Cancel</button>
          <button class="btn btn-primary" id="save-settings">Save Settings</button>
        </div>
      </div>
    `;
    
    // Setup event handlers
    modal.querySelector('.modal-close').addEventListener('click', () => {
      modal.remove();
    });
    
    modal.querySelector('#cancel-settings').addEventListener('click', () => {
      modal.remove();
    });
    
    modal.querySelector('#save-settings').addEventListener('click', () => {
      this.saveSettings(modal);
      modal.remove();
    });
    
    return modal;
  }
  
  renderSettingsForm() {
    const currentSettings = {
      updateInterval: this.options.updateInterval,
      maxDataPoints: this.options.maxDataPoints,
      enableRealTime: this.options.enableRealTime,
      enableDragDrop: this.options.enableDragDrop,
      autoSave: this.options.autoSave,
      collaborationMode: this.options.collaborationMode
    };
    
    return `
      <div class="setting-group">
        <h4>Real-Time Settings</h4>
        
        <div class="setting-item">
          <label for="setting-update-interval">Update Interval (ms)</label>
          <input type="number" id="setting-update-interval" value="${currentSettings.updateInterval}" min="100" max="10000" step="100">
        </div>
        
        <div class="setting-item">
          <label for="setting-max-data-points">Max Data Points</label>
          <input type="number" id="setting-max-data-points" value="${currentSettings.maxDataPoints}" min="100" max="100000" step="100">
        </div>
        
        <div class="setting-item">
          <label>
            <input type="checkbox" id="setting-enable-realtime" ${currentSettings.enableRealTime ? 'checked' : ''}>
            Enable Real-Time Updates
          </label>
        </div>
      </div>
      
      <div class="setting-group">
        <h4>Layout Settings</h4>
        
        <div class="setting-item">
          <label>
            <input type="checkbox" id="setting-enable-dragdrop" ${currentSettings.enableDragDrop ? 'checked' : ''}>
            Enable Drag & Drop
          </label>
        </div>
        
        <div class="setting-item">
          <label>
            <input type="checkbox" id="setting-auto-save" ${currentSettings.autoSave ? 'checked' : ''}>
            Auto-Save Layout
          </label>
        </div>
      </div>
      
      <div class="setting-group">
        <h4>Collaboration</h4>
        
        <div class="setting-item">
          <label>
            <input type="checkbox" id="setting-collaboration-mode" ${currentSettings.collaborationMode ? 'checked' : ''}>
            Enable Collaboration Mode
          </label>
        </div>
      </div>
    `;
  }
  
  saveSettings(modal) {
    const formData = new FormData(modal.querySelector('.settings-form'));
    
    // Update options
    this.options.updateInterval = parseInt(modal.querySelector('#setting-update-interval').value);
    this.options.maxDataPoints = parseInt(modal.querySelector('#setting-max-data-points').value);
    this.options.enableRealTime = modal.querySelector('#setting-enable-realtime').checked;
    this.options.enableDragDrop = modal.querySelector('#setting-enable-dragdrop').checked;
    this.options.autoSave = modal.querySelector('#setting-auto-save').checked;
    this.options.collaborationMode = modal.querySelector('#setting-collaboration-mode').checked;
    
    // Apply changes
    this.applyConfiguration(this.options);
    
    // Save to local storage
    localStorage.setItem('dashboard-settings', JSON.stringify(this.options));
    
    console.log('Settings saved:', this.options);
  }
  
  // Configuration management
  async loadDashboardConfiguration() {
    try {
      const savedSettings = localStorage.getItem('dashboard-settings');
      if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        Object.assign(this.options, settings);
        this.applyConfiguration(settings);
      }
      
      const savedLayout = localStorage.getItem('dashboard-layout');
      if (savedLayout && this.layoutEngine) {
        const layout = JSON.parse(savedLayout);
        this.layoutEngine.importLayout(layout);
      }
    } catch (error) {
      console.error('Failed to load dashboard configuration:', error);
    }
  }
  
  saveLayoutConfiguration() {
    if (this.layoutEngine) {
      const layout = this.layoutEngine.exportLayout();
      localStorage.setItem('dashboard-layout', JSON.stringify(layout));
      console.log('Layout configuration saved');
    }
  }
  
  loadLayoutConfiguration() {
    try {
      const savedLayout = localStorage.getItem('dashboard-layout');
      if (savedLayout && this.layoutEngine) {
        const layout = JSON.parse(savedLayout);
        this.layoutEngine.importLayout(layout);
        console.log('Layout configuration loaded');
      }
    } catch (error) {
      console.error('Failed to load layout configuration:', error);
    }
  }
  
  resetLayout() {
    if (this.layoutEngine) {
      this.layoutEngine.clearLayout();
      localStorage.removeItem('dashboard-layout');
      
      // Show empty state
      const emptyDashboard = this.container.querySelector('.empty-dashboard');
      if (emptyDashboard) {
        emptyDashboard.style.display = 'block';
      }
      
      console.log('Layout reset');
    }
  }
  
  exportLayoutConfiguration() {
    if (this.layoutEngine) {
      const layout = this.layoutEngine.exportLayout();
      const blob = new Blob([JSON.stringify(layout, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = `dashboard-layout-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      console.log('Layout configuration exported');
    }
  }
  
  applyConfiguration(config) {
    // Apply real-time settings
    if (config.updateInterval !== undefined) {
      this.updateWebSocketConfiguration();
    }
    
    // Apply layout settings
    if (this.layoutEngine) {
      if (config.enableDragDrop !== undefined) {
        this.layoutEngine.options.enableDragDrop = config.enableDragDrop;
      }
      if (config.autoSave !== undefined) {
        this.layoutEngine.options.persistLayout = config.autoSave;
      }
    }
    
    console.log('Configuration applied:', config);
  }
  
  updateWebSocketConfiguration() {
    if (this.isConnected) {
      this.sendWebSocketMessage({
        type: 'configure',
        config: {
          updateInterval: this.options.updateInterval,
          maxDataPoints: this.options.maxDataPoints
        }
      });
    }
  }
  
  // Utility methods
  sendWebSocketMessage(message) {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(message));
    }
  }
  
  exportDashboard() {
    const exportData = {
      settings: this.options,
      layout: this.layoutEngine ? this.layoutEngine.exportLayout() : null,
      timestamp: new Date().toISOString(),
      version: '1.0'
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `pynomaly-dashboard-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    console.log('Dashboard exported');
  }
  
  showNotification(notification) {
    // Add notification to dashboard state
    dashboardState.dispatch(dashboardState.actions.addNotification(notification));
  }
  
  playAlertSound(severity) {
    // Play alert sound based on severity
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    // Different frequencies for different severities
    const frequencies = {
      'low': 300,
      'medium': 500,
      'high': 700,
      'critical': 900
    };
    
    oscillator.frequency.setValueAtTime(frequencies[severity] || 500, audioContext.currentTime);
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.5);
  }
  
  showErrorState(error) {
    this.container.innerHTML = `
      <div class="error-state">
        <div class="error-icon">⚠️</div>
        <h3>Dashboard Error</h3>
        <p>Failed to initialize dashboard: ${error.message}</p>
        <button class="btn btn-primary" onclick="location.reload()">Reload</button>
      </div>
    `;
  }
  
  destroy() {
    // Cleanup WebSocket
    if (this.websocket) {
      this.websocket.close();
    }
    
    // Cleanup layout engine
    if (this.layoutEngine) {
      this.layoutEngine.destroy();
    }
    
    // Cleanup real-time dashboard
    if (this.realTimeDashboard) {
      this.realTimeDashboard.destroy();
    }
    
    // Clear data streams
    this.dataStreams.clear();
    this.subscribers.clear();
    
    console.log('Advanced Dashboard destroyed');
  }
}

// Circular buffer implementation for efficient data storage
class CircularBuffer {
  constructor(capacity) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
    this.size_ = 0;
    this.head = 0;
  }
  
  push(item) {
    this.buffer[this.head] = item;
    this.head = (this.head + 1) % this.capacity;
    
    if (this.size_ < this.capacity) {
      this.size_++;
    }
  }
  
  toArray() {
    const result = new Array(this.size_);
    for (let i = 0; i < this.size_; i++) {
      result[i] = this.buffer[(this.head - this.size_ + i + this.capacity) % this.capacity];
    }
    return result;
  }
  
  size() {
    return this.size_;
  }
  
  clear() {
    this.size_ = 0;
    this.head = 0;
  }
}

// Factory function
export function createAdvancedDashboard(container, options = {}) {
  return new AdvancedDashboard(container, options);
}

// Auto-initialize advanced dashboards
export function initializeAdvancedDashboards() {
  document.querySelectorAll('[data-component="advanced-dashboard"]').forEach(container => {
    new AdvancedDashboard(container);
  });
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeAdvancedDashboards);
} else {
  initializeAdvancedDashboards();
}

export default AdvancedDashboard;
