/**
 * Main Application Entry Point
 * Orchestrates all components and initializes the Pynomaly web application
 */

import { dashboardState } from './state/dashboard-state.js';
import { createRealTimeDashboard } from './charts/real-time-dashboard.js';
import { createAnomalyTimeline } from './charts/anomaly-timeline.js';
import { createAnomalyHeatmap } from './charts/anomaly-heatmap.js';
import { initializeInteractiveForms } from './components/interactive-forms.js';

class PynomAlyApp {
  constructor() {
    this.components = new Map();
    this.initialized = false;
    this.version = '1.0.0';
    
    this.init();
  }
  
  async init() {
    try {
      console.log(`🚀 Initializing Pynomaly Web Application v${this.version}`);
      
      // Initialize core components
      await this.initializeCore();
      
      // Initialize state management
      this.initializeStateManagement();
      
      // Initialize components
      await this.initializeComponents();
      
      // Setup global event handlers
      this.setupGlobalEventHandlers();
      
      // Setup keyboard shortcuts
      this.setupKeyboardShortcuts();
      
      // Initialize theme
      this.initializeTheme();
      
      // Setup periodic tasks
      this.setupPeriodicTasks();
      
      this.initialized = true;
      console.log('✅ Pynomaly application initialized successfully');
      
      // Emit ready event
      document.dispatchEvent(new CustomEvent('pynomaly:ready'));
      
    } catch (error) {
      console.error('❌ Failed to initialize Pynomaly application:', error);
      this.handleInitializationError(error);
    }
  }
  
  async initializeCore() {
    // Check browser compatibility
    this.checkBrowserCompatibility();
    
    // Initialize service worker for PWA features
    await this.initializeServiceWorker();
    
    // Setup error handling
    this.setupErrorHandling();
    
    // Initialize performance monitoring
    this.initializePerformanceMonitoring();
  }
  
  checkBrowserCompatibility() {
    const features = [
      'fetch',
      'WebSocket',
      'localStorage',
      'sessionStorage',
      'Promise',
      'Map',
      'Set'
    ];
    
    const missingFeatures = features.filter(feature => !(feature in window));
    
    if (missingFeatures.length > 0) {
      const message = `Your browser is missing required features: ${missingFeatures.join(', ')}. Please update your browser.`;
      this.showCriticalError(message);
      throw new Error(message);
    }
  }
  
  async initializeServiceWorker() {
    if ('serviceWorker' in navigator) {
      try {
        const registration = await navigator.serviceWorker.register('/sw.js');
        console.log('Service Worker registered:', registration);
        
        // Update state
        dashboardState.dispatch(dashboardState.actions.setPreference('serviceWorkerEnabled', true));
      } catch (error) {
        console.warn('Service Worker registration failed:', error);
      }
    }
  }
  
  setupErrorHandling() {
    // Global error handler
    window.addEventListener('error', (event) => {
      this.handleError(event.error, 'Global Error');
    });
    
    // Unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.handleError(event.reason, 'Unhandled Promise Rejection');
    });
  }
  
  initializePerformanceMonitoring() {
    // Web Vitals monitoring
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          this.recordPerformanceMetric(entry);
        });
      });
      
      observer.observe({ entryTypes: ['navigation', 'paint', 'largest-contentful-paint'] });
    }
  }
  
  initializeStateManagement() {
    // Subscribe to state changes for global updates
    dashboardState.subscribe((action, prevState, newState) => {
      this.handleStateChange(action, prevState, newState);
    });
    
    // Initialize with default data if needed
    this.loadInitialData();
  }
  
  async loadInitialData() {
    try {
      dashboardState.dispatch(dashboardState.actions.setLoading(true));
      
      // Load datasets
      const datasets = await this.fetchDatasets();
      dashboardState.dispatch(dashboardState.actions.setDatasets(datasets));
      
      // Load recent anomalies
      const anomalies = await this.fetchRecentAnomalies();
      dashboardState.dispatch(dashboardState.actions.addAnomalies(anomalies));
      
      // Update metrics
      this.updateMetrics();
      
    } catch (error) {
      this.handleError(error, 'Failed to load initial data');
    } finally {
      dashboardState.dispatch(dashboardState.actions.setLoading(false));
    }
  }
  
  async fetchDatasets() {
    try {
      const response = await fetch('/api/datasets');
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      return await response.json();
    } catch (error) {
      console.warn('Failed to fetch datasets:', error);
      return [];
    }
  }
  
  async fetchRecentAnomalies() {
    try {
      const response = await fetch('/api/anomalies/recent');
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      return await response.json();
    } catch (error) {
      console.warn('Failed to fetch recent anomalies:', error);
      return [];
    }
  }
  
  async initializeComponents() {
    // Initialize real-time dashboard if container exists
    const dashboardContainer = document.querySelector('[data-component="real-time-dashboard"]');
    if (dashboardContainer) {
      const dashboard = createRealTimeDashboard(dashboardContainer, {
        websocketUrl: this.getWebSocketUrl(),
        updateInterval: dashboardState.getStateSlice('realTime.updateRate'),
        alertThreshold: 0.8
      });
      this.components.set('realTimeDashboard', dashboard);
    }
    
    // Initialize timeline charts
    document.querySelectorAll('[data-component="anomaly-timeline"]').forEach(container => {
      const timeline = createAnomalyTimeline(container, {
        width: container.clientWidth,
        height: 400,
        interactive: true,
        showLegend: true
      });
      this.components.set(`timeline-${container.id}`, timeline);
    });
    
    // Initialize heatmap charts
    document.querySelectorAll('[data-component="anomaly-heatmap"]').forEach(container => {
      const heatmap = createAnomalyHeatmap(container, {
        width: container.clientWidth,
        height: 500,
        interactive: true,
        showLegend: true
      });
      this.components.set(`heatmap-${container.id}`, heatmap);
    });
    
    // Initialize interactive forms
    initializeInteractiveForms();
    
    // Initialize other UI components
    this.initializeUIComponents();
  }
  
  initializeUIComponents() {
    // Theme toggle
    this.initializeThemeToggle();
    
    // Sidebar toggle
    this.initializeSidebarToggle();
    
    // Layout controls
    this.initializeLayoutControls();
    
    // Search functionality
    this.initializeSearch();
    
    // Notification system
    this.initializeNotifications();
    
    // Filter controls
    this.initializeFilters();
  }
  
  initializeThemeToggle() {
    const themeToggle = document.querySelector('[data-action="toggle-theme"]');
    if (themeToggle) {
      themeToggle.addEventListener('click', () => {
        const currentTheme = dashboardState.getStateSlice('ui.theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        dashboardState.dispatch(dashboardState.actions.setTheme(newTheme));
      });
    }
  }
  
  initializeSidebarToggle() {
    const sidebarToggle = document.querySelector('[data-action="toggle-sidebar"]');
    if (sidebarToggle) {
      sidebarToggle.addEventListener('click', () => {
        dashboardState.dispatch(dashboardState.actions.toggleSidebar());
      });
    }
  }
  
  initializeLayoutControls() {
    document.querySelectorAll('[data-layout]').forEach(control => {
      control.addEventListener('click', () => {
        const layout = control.dataset.layout;
        dashboardState.dispatch(dashboardState.actions.setLayout(layout));
      });
    });
  }
  
  initializeSearch() {
    const searchInput = document.querySelector('[data-component="global-search"]');
    if (searchInput) {
      let searchTimeout;
      
      searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
          this.performSearch(e.target.value);
        }, 300);
      });
    }
  }
  
  initializeNotifications() {
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
    
    // Setup notification container
    if (!document.querySelector('.notification-container')) {
      const container = document.createElement('div');
      container.className = 'notification-container';
      container.setAttribute('role', 'region');
      container.setAttribute('aria-live', 'polite');
      container.setAttribute('aria-label', 'Notifications');
      document.body.appendChild(container);
    }
  }
  
  initializeFilters() {
    // Date range filter
    const dateRangeInputs = document.querySelectorAll('[data-filter="date-range"]');
    dateRangeInputs.forEach(input => {
      input.addEventListener('change', () => {
        const startInput = document.querySelector('[data-filter="date-start"]');
        const endInput = document.querySelector('[data-filter="date-end"]');
        
        if (startInput && endInput) {
          const start = startInput.value ? new Date(startInput.value) : null;
          const end = endInput.value ? new Date(endInput.value) : null;
          dashboardState.dispatch(dashboardState.actions.setDateRange(start, end));
        }
      });
    });
    
    // Confidence range filter
    const confidenceRange = document.querySelector('[data-filter="confidence-range"]');
    if (confidenceRange) {
      confidenceRange.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        dashboardState.dispatch(dashboardState.actions.setConfidenceRange(value, 1));
      });
    }
    
    // Status filter
    const statusFilter = document.querySelector('[data-filter="status"]');
    if (statusFilter) {
      statusFilter.addEventListener('change', (e) => {
        dashboardState.dispatch(dashboardState.actions.setStatusFilter(e.target.value));
      });
    }
  }
  
  setupGlobalEventHandlers() {
    // Handle component interactions
    document.addEventListener('pointClicked', (e) => {
      this.handlePointClick(e.detail);
    });
    
    document.addEventListener('cellClicked', (e) => {
      this.handleCellClick(e.detail);
    });
    
    document.addEventListener('timeRangeSelected', (e) => {
      this.handleTimeRangeSelection(e.detail);
    });
    
    // Handle form submissions
    document.addEventListener('formSubmit', (e) => {
      this.handleFormSubmit(e.detail);
    });
    
    // Handle file uploads
    document.addEventListener('fileUploaded', (e) => {
      this.handleFileUpload(e.detail);
    });
    
    // Handle window resize
    let resizeTimeout;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        this.handleWindowResize();
      }, 250);
    });
    
    // Handle visibility change
    document.addEventListener('visibilitychange', () => {
      this.handleVisibilityChange();
    });
  }
  
  setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Only handle shortcuts when not in input fields
      if (e.target.matches('input, textarea, select')) return;
      
      const shortcuts = {
        'KeyT': () => dashboardState.dispatch(dashboardState.actions.toggleSidebar()),
        'KeyD': () => this.toggleTheme(),
        'KeyF': () => this.focusSearch(),
        'KeyR': () => this.refreshData(),
        'Escape': () => this.clearSelection(),
        'KeyH': () => this.showHelp()
      };
      
      // Handle Ctrl/Cmd + key combinations
      if (e.ctrlKey || e.metaKey) {
        const handler = shortcuts[e.code];
        if (handler) {
          e.preventDefault();
          handler();
        }
      }
    });
  }
  
  initializeTheme() {
    const theme = dashboardState.getStateSlice('ui.theme');
    this.applyTheme(theme);
  }
  
  applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    document.body.className = document.body.className
      .replace(/theme-\w+/g, '')
      .concat(` theme-${theme}`);
    
    // Update theme color meta tag for mobile browsers
    const themeColorMeta = document.querySelector('meta[name="theme-color"]');
    if (themeColorMeta) {
      themeColorMeta.content = theme === 'dark' ? '#0f172a' : '#ffffff';
    }
  }
  
  setupPeriodicTasks() {
    // Auto-refresh data if enabled
    setInterval(() => {
      const autoRefresh = dashboardState.getStateSlice('preferences.autoRefresh');
      const interval = dashboardState.getStateSlice('preferences.refreshInterval');
      
      if (autoRefresh && !document.hidden) {
        this.refreshData();
      }
    }, dashboardState.getStateSlice('preferences.refreshInterval') || 30000);
    
    // Update metrics periodically
    setInterval(() => {
      this.updateMetrics();
    }, 5000);
    
    // Clean up old notifications
    setInterval(() => {
      this.cleanupNotifications();
    }, 60000);
  }
  
  handleStateChange(action, prevState, newState) {
    // Handle theme changes
    if (action.type === 'SET_THEME') {
      this.applyTheme(newState.ui.theme);
    }
    
    // Handle sidebar changes
    if (action.type === 'TOGGLE_SIDEBAR') {
      const sidebar = document.querySelector('.sidebar');
      if (sidebar) {
        sidebar.classList.toggle('collapsed', newState.ui.sidebarCollapsed);
      }
    }
    
    // Handle layout changes
    if (action.type === 'SET_LAYOUT') {
      document.body.setAttribute('data-layout', newState.ui.layout);
    }
    
    // Handle notification additions
    if (action.type === 'ADD_NOTIFICATION') {
      this.showNotification(action.payload);
    }
    
    // Handle loading state changes
    if (action.type === 'SET_LOADING') {
      this.updateLoadingState(newState.ui.loading);
    }
    
    // Handle filter changes - update charts
    if (action.type.includes('FILTER') || action.type === 'RESET_FILTERS') {
      this.updateChartsWithFilters();
    }
  }
  
  handlePointClick(detail) {
    const { data } = detail;
    
    // Show anomaly details modal or sidebar
    this.showAnomalyDetails(data);
    
    // Update selection in state
    dashboardState.dispatch(dashboardState.actions.setTimelineSelection({
      point: data,
      timestamp: data.timestamp
    }));
  }
  
  handleCellClick(detail) {
    const { data } = detail;
    
    // Filter to show only this feature/time combination
    dashboardState.dispatch(dashboardState.actions.setFeatureFilter([data.y]));
    
    // Update date range to focus on this time period
    const timeRange = this.getTimeRangeForCell(data);
    dashboardState.dispatch(dashboardState.actions.setDateRange(timeRange.start, timeRange.end));
  }
  
  handleTimeRangeSelection(detail) {
    const { startTime, endTime } = detail;
    dashboardState.dispatch(dashboardState.actions.setDateRange(startTime, endTime));
  }
  
  async handleFormSubmit(detail) {
    const { formData } = detail;
    
    try {
      // Process form submission
      const response = await this.submitAnalysisRequest(formData);
      
      // Show success notification
      dashboardState.dispatch(dashboardState.actions.addNotification({
        type: 'success',
        title: 'Analysis Started',
        message: 'Your anomaly detection analysis has been queued for processing.'
      }));
      
      // Refresh data to show new analysis
      this.refreshData();
      
    } catch (error) {
      this.handleError(error, 'Form submission failed');
    }
  }
  
  handleFileUpload(detail) {
    const { file, response } = detail;
    
    // Add uploaded dataset to state
    dashboardState.dispatch(dashboardState.actions.setDatasets([
      ...dashboardState.getStateSlice('data.datasets'),
      response.dataset
    ]));
    
    // Show success notification
    dashboardState.dispatch(dashboardState.actions.addNotification({
      type: 'success',
      title: 'File Uploaded',
      message: `Successfully uploaded ${file.name}`
    }));
  }
  
  handleWindowResize() {
    // Resize all charts
    this.components.forEach((component, key) => {
      if (component.resize && typeof component.resize === 'function') {
        component.resize();
      }
    });
  }
  
  handleVisibilityChange() {
    if (document.hidden) {
      // Pause real-time updates when tab is hidden
      const dashboard = this.components.get('realTimeDashboard');
      if (dashboard && dashboard.pauseUpdates) {
        dashboard.pauseUpdates();
      }
    } else {
      // Resume updates when tab becomes visible
      const dashboard = this.components.get('realTimeDashboard');
      if (dashboard && dashboard.resumeUpdates) {
        dashboard.resumeUpdates();
      }
      
      // Refresh data to catch up
      this.refreshData();
    }
  }
  
  async refreshData() {
    try {
      // Refresh datasets
      const datasets = await this.fetchDatasets();
      dashboardState.dispatch(dashboardState.actions.setDatasets(datasets));
      
      // Refresh recent anomalies
      const anomalies = await this.fetchRecentAnomalies();
      dashboardState.dispatch(dashboardState.actions.clearAnomalies());
      dashboardState.dispatch(dashboardState.actions.addAnomalies(anomalies));
      
      // Update metrics
      this.updateMetrics();
      
    } catch (error) {
      this.handleError(error, 'Failed to refresh data');
    }
  }
  
  updateMetrics() {
    const anomalies = dashboardState.getStateSlice('data.anomalies');
    const totalDataPoints = anomalies.length;
    const anomalyCount = anomalies.filter(a => a.isAnomaly).length;
    const anomalyRate = totalDataPoints > 0 ? (anomalyCount / totalDataPoints) : 0;
    
    dashboardState.dispatch(dashboardState.actions.updateMetrics({
      totalDataPoints,
      anomalyCount,
      anomalyRate
    }));
  }
  
  updateChartsWithFilters() {
    const filteredAnomalies = dashboardState.getters.getFilteredAnomalies();
    
    // Update timeline chart
    const timeline = this.components.get('timeline-main');
    if (timeline) {
      timeline.setData(filteredAnomalies);
    }
    
    // Update heatmap chart
    const heatmap = this.components.get('heatmap-main');
    if (heatmap) {
      const heatmapData = this.convertToHeatmapData(filteredAnomalies);
      heatmap.setData(heatmapData);
    }
  }
  
  showNotification(notification) {
    const container = document.querySelector('.notification-container');
    if (!container) return;
    
    const notificationElement = document.createElement('div');
    notificationElement.className = `notification notification-${notification.type}`;
    notificationElement.innerHTML = `
      <div class="notification-icon">
        ${this.getNotificationIcon(notification.type)}
      </div>
      <div class="notification-content">
        <div class="notification-title">${notification.title}</div>
        <div class="notification-message">${notification.message}</div>
      </div>
      <button class="notification-close" data-id="${notification.id}">×</button>
    `;
    
    // Add click handler for close button
    notificationElement.querySelector('.notification-close').addEventListener('click', () => {
      dashboardState.dispatch(dashboardState.actions.removeNotification(notification.id));
      notificationElement.remove();
    });
    
    container.appendChild(notificationElement);
    
    // Auto-remove after delay
    setTimeout(() => {
      if (notificationElement.parentElement) {
        notificationElement.remove();
        dashboardState.dispatch(dashboardState.actions.removeNotification(notification.id));
      }
    }, notification.duration || 5000);
    
    // Show browser notification if supported and enabled
    this.showBrowserNotification(notification);
  }
  
  showBrowserNotification(notification) {
    if ('Notification' in window && 
        Notification.permission === 'granted' && 
        dashboardState.getStateSlice('preferences.alertsEnabled')) {
      
      new Notification(notification.title, {
        body: notification.message,
        icon: '/static/images/pynomaly-icon-192.png',
        tag: notification.id
      });
    }
  }
  
  getNotificationIcon(type) {
    const icons = {
      success: '✅',
      error: '❌',
      warning: '⚠️',
      info: 'ℹ️'
    };
    return icons[type] || 'ℹ️';
  }
  
  handleError(error, context = 'Application Error') {
    console.error(`${context}:`, error);
    
    // Add error to state
    dashboardState.dispatch(dashboardState.actions.addError({
      message: error.message || 'An unexpected error occurred',
      context,
      timestamp: new Date(),
      stack: error.stack
    }));
    
    // Show error notification
    dashboardState.dispatch(dashboardState.actions.addNotification({
      type: 'error',
      title: context,
      message: error.message || 'An unexpected error occurred',
      duration: 8000
    }));
  }
  
  showCriticalError(message) {
    document.body.innerHTML = `
      <div class="critical-error">
        <div class="error-container">
          <h1>Critical Error</h1>
          <p>${message}</p>
          <button onclick="location.reload()">Reload Page</button>
        </div>
      </div>
    `;
  }
  
  handleInitializationError(error) {
    this.showCriticalError(`Failed to initialize application: ${error.message}`);
  }
  
  // Utility methods
  getWebSocketUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws/anomalies`;
  }
  
  performSearch(query) {
    // Implement global search functionality
    console.log('Performing search:', query);
  }
  
  toggleTheme() {
    const currentTheme = dashboardState.getStateSlice('ui.theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    dashboardState.dispatch(dashboardState.actions.setTheme(newTheme));
  }
  
  focusSearch() {
    const searchInput = document.querySelector('[data-component="global-search"]');
    if (searchInput) searchInput.focus();
  }
  
  clearSelection() {
    dashboardState.dispatch(dashboardState.actions.setTimelineSelection(null));
  }
  
  showHelp() {
    // Show help modal or navigate to help page
    console.log('Showing help');
  }
  
  cleanupNotifications() {
    const notifications = dashboardState.getStateSlice('ui.notifications');
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
    
    notifications.forEach(notification => {
      if (notification.timestamp < fiveMinutesAgo) {
        dashboardState.dispatch(dashboardState.actions.removeNotification(notification.id));
      }
    });
  }
  
  updateLoadingState(loading) {
    document.body.classList.toggle('loading', loading);
    
    const loadingIndicator = document.querySelector('.global-loading');
    if (loadingIndicator) {
      loadingIndicator.style.display = loading ? 'block' : 'none';
    }
  }
  
  showAnomalyDetails(anomaly) {
    // Implementation for showing anomaly details
    console.log('Showing anomaly details:', anomaly);
  }
  
  getTimeRangeForCell(cellData) {
    // Convert cell time to range
    const time = new Date(cellData.x);
    const start = new Date(time);
    const end = new Date(time);
    
    const aggregation = dashboardState.getStateSlice('charts.heatmap.aggregation');
    switch (aggregation) {
      case '5min':
        end.setMinutes(end.getMinutes() + 5);
        break;
      case '15min':
        end.setMinutes(end.getMinutes() + 15);
        break;
      case 'hour':
      default:
        end.setHours(end.getHours() + 1);
        break;
    }
    
    return { start, end };
  }
  
  async submitAnalysisRequest(formData) {
    const response = await fetch('/api/analysis/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
  }
  
  convertToHeatmapData(anomalies) {
    // Convert anomaly data to heatmap format
    const heatmapData = [];
    const aggregation = dashboardState.getStateSlice('charts.heatmap.aggregation');
    
    // Group anomalies by feature and time
    const grouped = new Map();
    
    anomalies.forEach(anomaly => {
      const timeKey = this.getTimeKey(anomaly.timestamp, aggregation);
      const key = `${anomaly.feature}-${timeKey}`;
      
      if (!grouped.has(key)) {
        grouped.set(key, {
          x: timeKey,
          y: anomaly.feature,
          values: [],
          maxScore: 0
        });
      }
      
      const group = grouped.get(key);
      group.values.push(anomaly);
      group.maxScore = Math.max(group.maxScore, anomaly.anomalyScore || 0);
    });
    
    // Convert to heatmap format
    grouped.forEach(group => {
      heatmapData.push({
        x: group.x,
        y: group.y,
        value: group.values.length,
        anomalyScore: group.maxScore
      });
    });
    
    return heatmapData;
  }
  
  getTimeKey(timestamp, aggregation) {
    const date = new Date(timestamp);
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes();
    
    switch (aggregation) {
      case '5min':
        const min5 = Math.floor(minutes / 5) * 5;
        return `${hours}:${min5.toString().padStart(2, '0')}`;
      case '15min':
        const min15 = Math.floor(minutes / 15) * 15;
        return `${hours}:${min15.toString().padStart(2, '0')}`;
      case 'hour':
      default:
        return `${hours}:00`;
    }
  }
  
  recordPerformanceMetric(entry) {
    // Record performance metrics for monitoring
    console.log('Performance metric:', entry.name, entry.duration || entry.startTime);
  }
  
  // Public API
  getComponent(name) {
    return this.components.get(name);
  }
  
  getAllComponents() {
    return new Map(this.components);
  }
  
  getState() {
    return dashboardState.getState();
  }
  
  dispatch(action) {
    return dashboardState.dispatch(action);
  }
  
  destroy() {
    // Clean up components
    this.components.forEach(component => {
      if (component.destroy) {
        component.destroy();
      }
    });
    
    this.components.clear();
    this.initialized = false;
  }
}

// Create and initialize the application
const app = new PynomAlyApp();

// Export for global access
window.PynomAly = app;
export default app;