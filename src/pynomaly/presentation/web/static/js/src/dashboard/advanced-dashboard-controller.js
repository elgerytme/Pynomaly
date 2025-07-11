/**
 * Advanced Dashboard Controller
 * Orchestrates all interactive components for the advanced analytics dashboard
 */

class AdvancedDashboardController {
  constructor(options = {}) {
    this.options = {
      enableRealTime: true,
      enableCollaboration: true,
      enableOfflineMode: true,
      autoRefreshInterval: 30000,
      ...options
    };

    this.components = new Map();
    this.eventBus = new EventTarget();
    this.isInitialized = false;
    this.currentUser = null;
    this.dashboardState = {
      selectedTimeRange: '24h',
      activeFilters: new Map(),
      currentView: 'overview',
      collaborativeMode: false
    };

    this.init();
  }

  async init() {
    try {
      console.log('[Dashboard] Initializing advanced dashboard controller...');

      await this.initializeCore();
      await this.initializeComponents();
      await this.initializeServices();
      await this.setupEventHandlers();
      await this.loadInitialData();

      this.isInitialized = true;
      this.emit('dashboard-ready');

      console.log('[Dashboard] Advanced dashboard controller initialized successfully');
    } catch (error) {
      console.error('[Dashboard] Initialization failed:', error);
      this.handleInitializationError(error);
    }
  }

  async initializeCore() {
    // Initialize accessibility manager
    if (window.AccessibilityManager) {
      this.components.set('accessibility', new window.AccessibilityManager());
    }

    // Initialize PWA manager
    if (window.EnhancedPWAManager) {
      this.components.set('pwa', new window.EnhancedPWAManager({
        enableOfflineDetection: true,
        enableDataSync: true,
        enableNotifications: true
      }));
    }

    // Initialize real-time analytics engine
    if (window.RealTimeAnalyticsEngine) {
      this.components.set('analytics', new window.RealTimeAnalyticsEngine({
        bufferSize: 10000,
        updateInterval: 100,
        enablePredictiveAnalytics: true
      }));
    }
  }

  async initializeComponents() {
    // Initialize advanced visualizations
    if (window.AdvancedInteractiveVisualizations) {
      this.components.set('visualizations', new window.AdvancedInteractiveVisualizations());
    }

    // Initialize WebSocket service
    if (window.EnhancedWebSocketService) {
      this.components.set('websocket', window.EnhancedWebSocketService.getInstance({
        enableCompression: true,
        enableBinarySupport: true
      }));
    }

    // Initialize theme manager
    this.components.set('theme', new ThemeManager());

    // Initialize notification manager
    this.components.set('notifications', new NotificationManager());
  }

  async initializeServices() {
    const analytics = this.components.get('analytics');
    const websocket = this.components.get('websocket');
    const visualizations = this.components.get('visualizations');

    if (analytics && this.options.enableRealTime) {
      analytics.start();

      // Connect analytics to visualizations
      analytics.on('anomaly-detected', (anomaly) => {
        this.handleAnomalyDetected(anomaly);
      });

      analytics.on('metrics-updated', (metrics) => {
        this.handleMetricsUpdate(metrics);
      });
    }

    if (websocket) {
      // Subscribe to real-time channels
      websocket.subscribe('anomaly_detection', (data) => {
        this.handleRealTimeAnomaly(data);
      });

      websocket.subscribe('system_metrics', (data) => {
        this.handleSystemMetrics(data);
      });

      if (this.options.enableCollaboration) {
        this.setupCollaboration(websocket);
      }
    }
  }

  setupEventHandlers() {
    // Dashboard navigation
    this.setupNavigation();

    // Chart interactions
    this.setupChartInteractions();

    // Filter interactions
    this.setupFilterHandlers();

    // Keyboard shortcuts
    this.setupKeyboardShortcuts();

    // Window events
    window.addEventListener('resize', this.debounce(() => {
      this.handleWindowResize();
    }, 250));

    // Visibility change
    document.addEventListener('visibilitychange', () => {
      this.handleVisibilityChange();
    });
  }

  setupNavigation() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
      tab.addEventListener('click', (e) => {
        const targetView = e.target.dataset.view;
        this.switchView(targetView);
      });
    });

    // Investigation tools
    const toolButtons = document.querySelectorAll('.tool-btn');
    toolButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const tool = e.target.dataset.tool;
        this.activateInvestigationTool(tool);
      });
    });
  }

  setupChartInteractions() {
    // Anomaly selection
    this.eventBus.addEventListener('anomaly-selected', (event) => {
      this.handleAnomalySelection(event.detail);
    });

    // Chart zoom/pan
    this.eventBus.addEventListener('chart-zoom', (event) => {
      this.handleChartZoom(event.detail);
    });

    // Brush selection
    this.eventBus.addEventListener('brush-selection', (event) => {
      this.handleBrushSelection(event.detail);
    });
  }

  setupFilterHandlers() {
    const filterControls = document.querySelectorAll('.filter-control');
    filterControls.forEach(control => {
      control.addEventListener('change', (e) => {
        this.updateFilter(e.target.name, e.target.value);
      });
    });

    // Time range selector
    const timeRangeSelector = document.getElementById('time-range-selector');
    if (timeRangeSelector) {
      timeRangeSelector.addEventListener('change', (e) => {
        this.updateTimeRange(e.target.value);
      });
    }
  }

  setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'f':
            e.preventDefault();
            this.focusSearch();
            break;
          case 'r':
            e.preventDefault();
            this.refreshData();
            break;
          case 'h':
            e.preventDefault();
            this.showKeyboardShortcuts();
            break;
        }
      }

      // Escape key
      if (e.key === 'Escape') {
        this.clearSelections();
      }

      // Arrow keys for navigation
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        if (e.target.tagName !== 'INPUT') {
          this.navigateWithArrows(e.key);
        }
      }
    });
  }

  setupCollaboration(websocket) {
    this.dashboardState.collaborativeMode = true;

    // Cursor tracking
    document.addEventListener('mousemove', this.throttle((e) => {
      websocket.broadcastCursorPosition(e.clientX, e.clientY);
    }, 100));

    // Chart interaction sharing
    this.eventBus.addEventListener('chart-interaction', (event) => {
      websocket.broadcastChartInteraction(
        event.detail.chartId,
        event.detail.interaction
      );
    });

    // Listen for collaborative events
    websocket.subscribe('user_cursor', (data) => {
      this.updateCollaboratorCursor(data);
    });

    websocket.subscribe('annotation', (data) => {
      this.addCollaborativeAnnotation(data);
    });
  }

  async loadInitialData() {
    try {
      this.showLoadingState();

      // Load dashboard configuration
      const config = await this.loadDashboardConfig();
      this.applyConfiguration(config);

      // Load initial datasets
      const datasets = await this.loadDatasets();
      this.processInitialDatasets(datasets);

      // Load user preferences
      const preferences = await this.loadUserPreferences();
      this.applyUserPreferences(preferences);

      this.hideLoadingState();
    } catch (error) {
      console.error('[Dashboard] Failed to load initial data:', error);
      this.showErrorState(error);
    }
  }

  // Data Management
  async loadDashboardConfig() {
    try {
      const response = await fetch('/api/dashboard/config');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.warn('[Dashboard] Using default config due to error:', error);
      return this.getDefaultConfig();
    }
  }

  async loadDatasets() {
    try {
      const response = await fetch('/api/datasets?active=true');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.warn('[Dashboard] Failed to load datasets:', error);
      return [];
    }
  }

  async loadUserPreferences() {
    try {
      const response = await fetch('/api/user/preferences');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.warn('[Dashboard] Using default preferences:', error);
      return this.getDefaultPreferences();
    }
  }

  // Event Handlers
  handleAnomalyDetected(anomaly) {
    console.log('[Dashboard] Anomaly detected:', anomaly);

    // Update visualizations
    const visualizations = this.components.get('visualizations');
    if (visualizations) {
      visualizations.addAnomalyToStream(anomaly);
    }

    // Update metrics
    this.updateAnomalyMetrics(anomaly);

    // Trigger notifications for high-severity anomalies
    if (anomaly.severity === 'high') {
      this.showAnomalyNotification(anomaly);
    }

    // Emit event for other components
    this.emit('anomaly-detected', anomaly);
  }

  handleRealTimeAnomaly(data) {
    const analytics = this.components.get('analytics');
    if (analytics) {
      analytics.addDataPoint({
        ...data,
        timestamp: Date.now(),
        source: 'websocket'
      });
    }
  }

  handleSystemMetrics(data) {
    // Update system health indicators
    this.updateSystemHealth(data);

    // Store for offline access
    const pwa = this.components.get('pwa');
    if (pwa) {
      pwa.performAction('cache-metrics', data);
    }
  }

  handleMetricsUpdate(metrics) {
    // Update dashboard metrics display
    this.updateMetricsDisplay(metrics);

    // Emit for other components
    this.emit('metrics-updated', metrics);
  }

  handleAnomalySelection(anomaly) {
    console.log('[Dashboard] Anomaly selected:', anomaly);

    // Update investigation panel
    this.showAnomalyDetails(anomaly);

    // Highlight in visualizations
    const visualizations = this.components.get('visualizations');
    if (visualizations) {
      visualizations.highlightAnomaly(anomaly.id);
    }

    // Update URL for deep linking
    this.updateURL({ selectedAnomaly: anomaly.id });
  }

  handleWindowResize() {
    // Notify all components of resize
    this.emit('window-resize', {
      width: window.innerWidth,
      height: window.innerHeight
    });

    // Trigger chart redraws
    const visualizations = this.components.get('visualizations');
    if (visualizations) {
      visualizations.handleResize();
    }
  }

  handleVisibilityChange() {
    const analytics = this.components.get('analytics');

    if (document.hidden) {
      // Page hidden - reduce processing
      if (analytics) {
        analytics.stop();
      }
    } else {
      // Page visible - resume processing
      if (analytics && this.options.enableRealTime) {
        analytics.start();
      }

      // Refresh data
      this.refreshData();
    }
  }

  // Dashboard Operations
  switchView(viewName) {
    if (this.dashboardState.currentView === viewName) return;

    console.log(`[Dashboard] Switching to view: ${viewName}`);

    // Update UI
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.view === viewName);
    });

    document.querySelectorAll('.investigation-panel').forEach(panel => {
      panel.classList.toggle('active', panel.dataset.view === viewName);
    });

    // Update state
    this.dashboardState.currentView = viewName;

    // Trigger view-specific actions
    this.onViewChanged(viewName);

    // Update URL
    this.updateURL({ view: viewName });
  }

  activateInvestigationTool(toolName) {
    console.log(`[Dashboard] Activating tool: ${toolName}`);

    // Update tool buttons
    document.querySelectorAll('.tool-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tool === toolName);
    });

    // Apply tool-specific behavior
    const visualizations = this.components.get('visualizations');
    if (visualizations) {
      visualizations.setInvestigationMode(toolName);
    }

    // Update body class for CSS styling
    document.body.className = document.body.className.replace(/investigation-\w+/g, '');
    document.body.classList.add(`investigation-${toolName}`);
  }

  updateFilter(filterName, filterValue) {
    this.dashboardState.activeFilters.set(filterName, filterValue);

    console.log(`[Dashboard] Filter updated: ${filterName} = ${filterValue}`);

    // Apply filters to data
    this.applyFilters();

    // Update visualizations
    this.refreshVisualizations();
  }

  updateTimeRange(timeRange) {
    this.dashboardState.selectedTimeRange = timeRange;

    console.log(`[Dashboard] Time range updated: ${timeRange}`);

    // Reload data for new time range
    this.loadDataForTimeRange(timeRange);

    // Update URL
    this.updateURL({ timeRange });
  }

  async refreshData() {
    console.log('[Dashboard] Refreshing data...');

    try {
      // Reload datasets
      const datasets = await this.loadDatasets();
      this.processInitialDatasets(datasets);

      // Refresh visualizations
      this.refreshVisualizations();

      // Update last refresh time
      this.updateLastRefreshTime();

    } catch (error) {
      console.error('[Dashboard] Data refresh failed:', error);
      this.showErrorNotification('Failed to refresh data');
    }
  }

  // UI Updates
  showLoadingState() {
    const loadingScreen = document.querySelector('.loading-screen');
    if (loadingScreen) {
      loadingScreen.classList.remove('hidden');
    }
  }

  hideLoadingState() {
    const loadingScreen = document.querySelector('.loading-screen');
    if (loadingScreen) {
      loadingScreen.classList.add('hidden');
    }
  }

  showErrorState(error) {
    const notifications = this.components.get('notifications');
    if (notifications) {
      notifications.show({
        type: 'error',
        title: 'Dashboard Error',
        message: error.message || 'An error occurred while loading the dashboard',
        persistent: true
      });
    }
  }

  updateMetricsDisplay(metrics) {
    // Update overview metrics cards
    Object.entries(metrics).forEach(([key, value]) => {
      const element = document.querySelector(`[data-metric="${key}"]`);
      if (element) {
        element.textContent = this.formatMetricValue(value);
      }
    });
  }

  showAnomalyDetails(anomaly) {
    const detailPanel = document.getElementById('anomaly-details');
    if (!detailPanel) return;

    detailPanel.innerHTML = this.renderAnomalyDetails(anomaly);
    detailPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Utility Methods
  emit(eventName, detail = null) {
    this.eventBus.dispatchEvent(new CustomEvent(eventName, { detail }));
  }

  on(eventName, handler) {
    this.eventBus.addEventListener(eventName, handler);
    return () => this.eventBus.removeEventListener(eventName, handler);
  }

  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  throttle(func, limit) {
    let inThrottle;
    return function() {
      const args = arguments;
      const context = this;
      if (!inThrottle) {
        func.apply(context, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }

  formatMetricValue(value) {
    if (typeof value === 'number') {
      if (value > 1000000) {
        return (value / 1000000).toFixed(1) + 'M';
      } else if (value > 1000) {
        return (value / 1000).toFixed(1) + 'K';
      } else {
        return value.toFixed(1);
      }
    }
    return value;
  }

  updateURL(params) {
    const url = new URL(window.location);
    Object.entries(params).forEach(([key, value]) => {
      if (value) {
        url.searchParams.set(key, value);
      } else {
        url.searchParams.delete(key);
      }
    });

    window.history.replaceState({}, '', url);
  }

  getDefaultConfig() {
    return {
      refreshInterval: 30000,
      maxAnomalies: 1000,
      enableNotifications: true,
      theme: 'light'
    };
  }

  getDefaultPreferences() {
    return {
      timeRange: '24h',
      autoRefresh: true,
      notifications: true,
      collaborativeMode: false
    };
  }

  renderAnomalyDetails(anomaly) {
    return `
      <div class="detail-section">
        <h4>Anomaly Details</h4>
        <div class="detail-grid">
          <div class="detail-item">
            <label>Timestamp</label>
            <span>${new Date(anomaly.timestamp).toLocaleString()}</span>
          </div>
          <div class="detail-item">
            <label>Severity</label>
            <span class="severity-badge severity-${anomaly.severity}">${anomaly.severity}</span>
          </div>
          <div class="detail-item">
            <label>Score</label>
            <span>${anomaly.score.toFixed(3)}</span>
          </div>
          <div class="detail-item">
            <label>Confidence</label>
            <span>${(anomaly.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    `;
  }

  // Cleanup
  destroy() {
    console.log('[Dashboard] Destroying dashboard controller...');

    // Stop analytics engine
    const analytics = this.components.get('analytics');
    if (analytics) {
      analytics.destroy();
    }

    // Disconnect WebSocket
    const websocket = this.components.get('websocket');
    if (websocket) {
      websocket.disconnect();
    }

    // Clear all components
    this.components.clear();

    // Clear event listeners
    this.eventBus = null;

    this.isInitialized = false;
  }
}

// Theme Manager
class ThemeManager {
  constructor() {
    this.currentTheme = this.getStoredTheme() || 'light';
    this.applyTheme(this.currentTheme);
  }

  getStoredTheme() {
    return localStorage.getItem('dashboard-theme');
  }

  setTheme(theme) {
    this.currentTheme = theme;
    localStorage.setItem('dashboard-theme', theme);
    this.applyTheme(theme);
  }

  applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    document.body.classList.toggle('dark', theme === 'dark');
  }

  toggleTheme() {
    const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
    this.setTheme(newTheme);
    return newTheme;
  }
}

// Notification Manager
class NotificationManager {
  constructor() {
    this.notifications = new Map();
    this.container = this.createContainer();
  }

  createContainer() {
    let container = document.getElementById('notification-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'notification-container';
      container.className = 'notification-container';
      document.body.appendChild(container);
    }
    return container;
  }

  show(options) {
    const id = 'notification_' + Date.now();
    const notification = this.createNotification(id, options);

    this.notifications.set(id, notification);
    this.container.appendChild(notification.element);

    // Auto-remove if not persistent
    if (!options.persistent) {
      setTimeout(() => this.remove(id), options.duration || 5000);
    }

    return id;
  }

  createNotification(id, options) {
    const element = document.createElement('div');
    element.className = `notification notification-${options.type || 'info'}`;
    element.innerHTML = `
      <div class="notification-content">
        <div class="notification-title">${options.title}</div>
        <div class="notification-message">${options.message}</div>
      </div>
      <button class="notification-close" aria-label="Close notification">&times;</button>
    `;

    element.querySelector('.notification-close').addEventListener('click', () => {
      this.remove(id);
    });

    return { element, options };
  }

  remove(id) {
    const notification = this.notifications.get(id);
    if (notification) {
      notification.element.remove();
      this.notifications.delete(id);
    }
  }

  clear() {
    this.notifications.forEach((_, id) => this.remove(id));
  }
}

// Global instance
window.AdvancedDashboardController = AdvancedDashboardController;

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.dashboardInstance = new AdvancedDashboardController();
  });
} else {
  window.dashboardInstance = new AdvancedDashboardController();
}

export default AdvancedDashboardController;
