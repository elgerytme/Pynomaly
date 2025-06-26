/**
 * Pynomaly Advanced Anomaly Detection Components Integration
 * 
 * This module provides a unified interface for all advanced anomaly detection
 * visualization components with centralized configuration, event handling,
 * and performance optimization.
 */

class PynomaryComponents {
  constructor(options = {}) {
    this.options = {
      theme: 'light',
      accessibility: {
        enabled: true,
        announcements: true,
        keyboardNavigation: true,
        highContrast: false,
        reducedMotion: false
      },
      performance: {
        enableGPUAcceleration: true,
        maxDataPoints: 5000,
        animationOptimization: true,
        memoryManagement: true
      },
      realTime: {
        enabled: false,
        interval: 5000,
        bufferSize: 1000,
        compression: true
      },
      ...options
    };

    this.components = new Map();
    this.eventBus = new EventTarget();
    this.dataCache = new Map();
    this.performanceMonitor = new PerformanceMonitor();
    
    this.init();
  }

  init() {
    this.setupGlobalStyles();
    this.setupEventListeners();
    this.setupPerformanceOptimizations();
    this.setupAccessibility();
  }

  setupGlobalStyles() {
    // Inject design tokens and component styles
    if (!document.querySelector('#pynomaly-styles')) {
      const styleSheet = document.createElement('style');
      styleSheet.id = 'pynomaly-styles';
      styleSheet.textContent = this.getGlobalStyles();
      document.head.appendChild(styleSheet);
    }

    // Apply theme
    this.setTheme(this.options.theme);
  }

  getGlobalStyles() {
    return `
      /* Pynomaly Component Styles */
      .pynomaly-component {
        font-family: var(--font-family-sans, -apple-system, BlinkMacSystemFont, sans-serif);
        position: relative;
        overflow: hidden;
      }
      
      .pynomaly-component.gpu-accelerated {
        will-change: transform;
        transform: translateZ(0);
      }
      
      .pynomaly-component.high-contrast {
        --color-primary-500: #0066cc;
        --color-danger-500: #cc0000;
        --color-warning-500: #cc6600;
        --color-text-primary: #000000;
        --color-bg-primary: #ffffff;
      }
      
      .pynomaly-component.reduced-motion * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
      
      @media (prefers-reduced-motion: reduce) {
        .pynomaly-component * {
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 0.01ms !important;
        }
      }
    `;
  }

  setupEventListeners() {
    // Global resize handler
    window.addEventListener('resize', this.debounce(() => {
      this.handleGlobalResize();
    }, 250));

    // Theme detection
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (this.options.theme === 'auto') {
          this.setTheme(e.matches ? 'dark' : 'light');
        }
      });

      window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', (e) => {
        this.setReducedMotion(e.matches);
      });

      window.matchMedia('(prefers-contrast: high)').addEventListener('change', (e) => {
        this.setHighContrast(e.matches);
      });
    }

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      this.handleGlobalKeyboard(e);
    });
  }

  setupPerformanceOptimizations() {
    if (this.options.performance.enableGPUAcceleration) {
      this.enableGPUAcceleration();
    }

    if (this.options.performance.memoryManagement) {
      this.setupMemoryManagement();
    }

    // Performance monitoring
    this.performanceMonitor.start();
  }

  setupAccessibility() {
    if (!this.options.accessibility.enabled) return;

    // Add ARIA live region for announcements
    if (!document.querySelector('#pynomaly-announcements')) {
      const liveRegion = document.createElement('div');
      liveRegion.id = 'pynomaly-announcements';
      liveRegion.className = 'sr-only';
      liveRegion.setAttribute('aria-live', 'polite');
      liveRegion.setAttribute('aria-atomic', 'true');
      document.body.appendChild(liveRegion);
    }

    // Detect and apply user preferences
    this.detectAccessibilityPreferences();
  }

  // Component Management
  createTimeSeriesChart(container, options = {}) {
    const mergedOptions = this.mergeOptions(options);
    const component = new AnomalyTimeSeriesChart(container, mergedOptions);
    
    this.registerComponent('timeSeries', component, container);
    this.setupComponentIntegration(component, 'timeSeries');
    
    return component;
  }

  createHeatmap(container, options = {}) {
    const mergedOptions = this.mergeOptions(options);
    const component = new AnomalyHeatmap(container, mergedOptions);
    
    this.registerComponent('heatmap', component, container);
    this.setupComponentIntegration(component, 'heatmap');
    
    return component;
  }

  createScatterPlot(container, options = {}) {
    const mergedOptions = this.mergeOptions(options);
    const component = new AnomalyScatterPlot(container, mergedOptions);
    
    this.registerComponent('scatterPlot', component, container);
    this.setupComponentIntegration(component, 'scatterPlot');
    
    return component;
  }

  createDashboard(container, options = {}) {
    const mergedOptions = this.mergeOptions(options);
    const component = new AnomalyDashboard(container, mergedOptions);
    
    this.registerComponent('dashboard', component, container);
    this.setupComponentIntegration(component, 'dashboard');
    
    return component;
  }

  registerComponent(id, component, container) {
    this.components.set(id, {
      instance: component,
      container: container,
      type: component.constructor.name,
      created: Date.now(),
      lastUpdate: Date.now()
    });

    // Apply global styles
    container.classList.add('pynomaly-component');
    
    if (this.options.performance.enableGPUAcceleration) {
      container.classList.add('gpu-accelerated');
    }

    if (this.options.accessibility.highContrast) {
      container.classList.add('high-contrast');
    }

    if (this.options.accessibility.reducedMotion) {
      container.classList.add('reduced-motion');
    }

    this.announceToUser(`${component.constructor.name} component created`);
  }

  setupComponentIntegration(component, id) {
    // Wrap component methods for monitoring
    this.wrapComponentMethods(component, id);

    // Setup data caching
    this.setupDataCaching(component, id);

    // Setup event forwarding
    this.setupEventForwarding(component, id);
  }

  wrapComponentMethods(component, id) {
    const originalSetData = component.setData?.bind(component);
    const originalRender = component.render?.bind(component);

    if (originalSetData) {
      component.setData = (...args) => {
        const startTime = performance.now();
        
        try {
          const result = originalSetData(...args);
          this.performanceMonitor.recordDataUpdate(id, performance.now() - startTime);
          this.updateComponentTimestamp(id);
          return result;
        } catch (error) {
          this.handleComponentError(id, error);
          throw error;
        }
      };
    }

    if (originalRender) {
      component.render = (...args) => {
        const startTime = performance.now();
        
        try {
          const result = originalRender(...args);
          this.performanceMonitor.recordRender(id, performance.now() - startTime);
          return result;
        } catch (error) {
          this.handleComponentError(id, error);
          throw error;
        }
      };
    }
  }

  setupDataCaching(component, id) {
    if (!this.options.performance.memoryManagement) return;

    const originalSetData = component.setData?.bind(component);
    if (originalSetData) {
      component.setData = (data, ...args) => {
        // Cache data with compression if enabled
        const cacheKey = `${id}_${Date.now()}`;
        let cachedData = data;
        
        if (this.options.realTime.compression && data.length > 100) {
          cachedData = this.compressData(data);
        }
        
        this.dataCache.set(cacheKey, {
          data: cachedData,
          timestamp: Date.now(),
          compressed: this.options.realTime.compression
        });

        // Limit cache size
        if (this.dataCache.size > 10) {
          const oldestKey = Array.from(this.dataCache.keys())[0];
          this.dataCache.delete(oldestKey);
        }

        return originalSetData(data, ...args);
      };
    }
  }

  setupEventForwarding(component, id) {
    const container = this.components.get(id)?.container;
    if (!container) return;

    // Forward component events to global event bus
    const eventTypes = [
      'pointSelected', 'pointsSelected', 'cellSelected', 
      'timeRangeSelected', 'pointsBrushed', 'chartError'
    ];

    eventTypes.forEach(eventType => {
      container.addEventListener(eventType, (event) => {
        this.eventBus.dispatchEvent(new CustomEvent(`component:${eventType}`, {
          detail: { ...event.detail, componentId: id, componentType: component.constructor.name }
        }));
      });
    });
  }

  // Data Management
  setGlobalData(componentId, data) {
    const component = this.components.get(componentId);
    if (component?.instance.setData) {
      component.instance.setData(data);
    }
  }

  getComponentData(componentId) {
    const cacheEntries = Array.from(this.dataCache.entries())
      .filter(([key]) => key.startsWith(componentId))
      .sort(([, a], [, b]) => b.timestamp - a.timestamp);

    if (cacheEntries.length > 0) {
      const [, cachedData] = cacheEntries[0];
      return cachedData.compressed ? this.decompressData(cachedData.data) : cachedData.data;
    }

    return null;
  }

  // Real-time Data Management
  startRealTimeUpdates(dataCallback) {
    if (this.realTimeTimer) {
      this.stopRealTimeUpdates();
    }

    this.options.realTime.enabled = true;
    this.realTimeTimer = setInterval(async () => {
      try {
        const newData = await dataCallback();
        this.distributeRealTimeData(newData);
      } catch (error) {
        console.error('Real-time data error:', error);
        this.announceToUser('Real-time data update failed');
      }
    }, this.options.realTime.interval);

    this.announceToUser('Real-time updates started');
  }

  stopRealTimeUpdates() {
    if (this.realTimeTimer) {
      clearInterval(this.realTimeTimer);
      this.realTimeTimer = null;
    }

    this.options.realTime.enabled = false;
    this.announceToUser('Real-time updates stopped');
  }

  distributeRealTimeData(data) {
    this.components.forEach((component, id) => {
      if (component.instance.addRealTimeData) {
        component.instance.addRealTimeData(data);
      }
    });
  }

  // Theme and Accessibility
  setTheme(theme) {
    this.options.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    
    this.components.forEach((component) => {
      if (component.instance.updateTheme) {
        component.instance.updateTheme(theme);
      }
    });

    this.announceToUser(`Theme changed to ${theme}`);
  }

  setHighContrast(enabled) {
    this.options.accessibility.highContrast = enabled;
    
    this.components.forEach((component) => {
      if (enabled) {
        component.container.classList.add('high-contrast');
      } else {
        component.container.classList.remove('high-contrast');
      }
    });

    this.announceToUser(`High contrast ${enabled ? 'enabled' : 'disabled'}`);
  }

  setReducedMotion(enabled) {
    this.options.accessibility.reducedMotion = enabled;
    
    this.components.forEach((component) => {
      if (enabled) {
        component.container.classList.add('reduced-motion');
      } else {
        component.container.classList.remove('reduced-motion');
      }
    });

    this.announceToUser(`Reduced motion ${enabled ? 'enabled' : 'disabled'}`);
  }

  detectAccessibilityPreferences() {
    if (window.matchMedia) {
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        this.setReducedMotion(true);
      }

      if (window.matchMedia('(prefers-contrast: high)').matches) {
        this.setHighContrast(true);
      }

      if (window.matchMedia('(prefers-color-scheme: dark)').matches && this.options.theme === 'auto') {
        this.setTheme('dark');
      }
    }
  }

  // Event Handling
  on(eventType, callback) {
    this.eventBus.addEventListener(eventType, callback);
  }

  off(eventType, callback) {
    this.eventBus.removeEventListener(eventType, callback);
  }

  emit(eventType, detail) {
    this.eventBus.dispatchEvent(new CustomEvent(eventType, { detail }));
  }

  handleGlobalResize() {
    this.components.forEach((component) => {
      if (component.instance.updateDimensions) {
        component.instance.updateDimensions();
      }
      if (component.instance.render) {
        component.instance.render();
      }
    });

    this.performanceMonitor.recordResize();
  }

  handleGlobalKeyboard(event) {
    if (!this.options.accessibility.keyboardNavigation) return;

    // Global keyboard shortcuts
    if (event.ctrlKey || event.metaKey) {
      switch (event.key.toLowerCase()) {
        case 'r':
          event.preventDefault();
          this.toggleRealTime();
          break;
        case 'e':
          event.preventDefault();
          this.exportAllData();
          break;
        case 'h':
          event.preventDefault();
          this.showKeyboardHelp();
          break;
      }
    }

    if (event.altKey) {
      switch (event.key.toLowerCase()) {
        case 't':
          event.preventDefault();
          this.toggleTheme();
          break;
        case 'c':
          event.preventDefault();
          this.setHighContrast(!this.options.accessibility.highContrast);
          break;
        case 'm':
          event.preventDefault();
          this.setReducedMotion(!this.options.accessibility.reducedMotion);
          break;
      }
    }
  }

  // Utility Methods
  toggleRealTime() {
    if (this.options.realTime.enabled) {
      this.stopRealTimeUpdates();
    } else {
      this.startRealTimeUpdates(() => this.generateMockData());
    }
  }

  toggleTheme() {
    const newTheme = this.options.theme === 'dark' ? 'light' : 'dark';
    this.setTheme(newTheme);
  }

  exportAllData() {
    const exportData = {};
    
    this.components.forEach((component, id) => {
      if (component.instance.exportData) {
        exportData[id] = component.instance.exportData();
      } else {
        exportData[id] = this.getComponentData(id);
      }
    });

    const dataStr = JSON.stringify({
      exported: new Date().toISOString(),
      components: exportData,
      options: this.options,
      performance: this.performanceMonitor.getReport()
    }, null, 2);

    this.downloadFile(dataStr, 'pynomaly-dashboard-export.json', 'application/json');
    this.announceToUser('Data exported successfully');
  }

  showKeyboardHelp() {
    const helpText = `
      Pynomaly Keyboard Shortcuts:
      
      Global:
      • Ctrl+R: Toggle real-time updates
      • Ctrl+E: Export all data
      • Ctrl+H: Show this help
      • Alt+T: Toggle theme
      • Alt+C: Toggle high contrast
      • Alt+M: Toggle reduced motion
      
      Chart Navigation:
      • Arrow Keys: Navigate data points
      • Home/End: First/Last point
      • Enter/Space: Select point
      • Escape: Clear selection
      
      Mouse & Touch:
      • Click: Select points/cells
      • Drag: Brush selection
      • Scroll: Zoom (where applicable)
      • Right-click: Context menu
    `;

    if (window.alert) {
      alert(helpText);
    } else {
      console.log(helpText);
    }

    this.announceToUser('Keyboard help displayed');
  }

  mergeOptions(componentOptions) {
    return {
      ...componentOptions,
      colors: {
        ...this.getThemeColors(),
        ...componentOptions.colors
      },
      accessibility: {
        ...this.options.accessibility,
        ...componentOptions.accessibility
      },
      performance: {
        ...this.options.performance,
        ...componentOptions.performance
      }
    };
  }

  getThemeColors() {
    const isDark = this.options.theme === 'dark';
    return {
      normal: isDark ? '#60a5fa' : '#3b82f6',
      anomaly: isDark ? '#f87171' : '#ef4444',
      warning: isDark ? '#fbbf24' : '#f59e0b',
      background: isDark ? '#111827' : '#ffffff',
      text: isDark ? '#f9fafb' : '#1f2937'
    };
  }

  enableGPUAcceleration() {
    const style = document.createElement('style');
    style.textContent = `
      .pynomaly-component.gpu-accelerated {
        transform: translate3d(0, 0, 0);
        backface-visibility: hidden;
        perspective: 1000px;
      }
      
      .pynomaly-component.gpu-accelerated * {
        transform-style: preserve-3d;
      }
    `;
    document.head.appendChild(style);
  }

  setupMemoryManagement() {
    // Monitor memory usage
    setInterval(() => {
      if (performance.memory) {
        const memoryInfo = performance.memory;
        const usageRatio = memoryInfo.usedJSHeapSize / memoryInfo.totalJSHeapSize;
        
        if (usageRatio > 0.9) {
          this.performGarbageCollection();
        }
      }
    }, 30000); // Check every 30 seconds
  }

  performGarbageCollection() {
    // Clean up old cached data
    const cutoffTime = Date.now() - (5 * 60 * 1000); // 5 minutes ago
    
    for (const [key, value] of this.dataCache.entries()) {
      if (value.timestamp < cutoffTime) {
        this.dataCache.delete(key);
      }
    }

    // Force garbage collection if available
    if (window.gc) {
      window.gc();
    }

    this.announceToUser('Memory optimization performed');
  }

  compressData(data) {
    // Simple compression for demonstration
    // In production, you might use a library like pako for gzip compression
    if (Array.isArray(data) && data.length > 1000) {
      // Downsample large datasets
      const step = Math.ceil(data.length / 1000);
      return data.filter((_, index) => index % step === 0);
    }
    return data;
  }

  decompressData(data) {
    // Decompression would reverse the compression process
    return data;
  }

  updateComponentTimestamp(id) {
    const component = this.components.get(id);
    if (component) {
      component.lastUpdate = Date.now();
    }
  }

  handleComponentError(id, error) {
    console.error(`Component ${id} error:`, error);
    
    const component = this.components.get(id);
    if (component?.container) {
      component.container.dispatchEvent(new CustomEvent('chartError', {
        detail: { error: error.message, componentId: id }
      }));
    }

    this.announceToUser(`Error in ${id} component: ${error.message}`);
  }

  announceToUser(message) {
    if (!this.options.accessibility.announcements) return;

    const liveRegion = document.getElementById('pynomaly-announcements');
    if (liveRegion) {
      liveRegion.textContent = message;
    }
  }

  downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  generateMockData() {
    return [{
      timestamp: new Date(),
      value: 50 + Math.random() * 20,
      anomalyScore: Math.random(),
      isAnomaly: Math.random() > 0.9
    }];
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

  // Cleanup
  destroy() {
    this.stopRealTimeUpdates();
    
    this.components.forEach((component) => {
      if (component.instance.destroy) {
        component.instance.destroy();
      }
    });
    
    this.components.clear();
    this.dataCache.clear();
    this.performanceMonitor.stop();
  }

  // Static factory method
  static create(options = {}) {
    return new PynomaryComponents(options);
  }
}

// Performance monitoring helper
class PerformanceMonitor {
  constructor() {
    this.metrics = {
      dataUpdates: [],
      renders: [],
      resizes: [],
      errors: []
    };
    this.startTime = performance.now();
  }

  start() {
    this.startTime = performance.now();
  }

  recordDataUpdate(componentId, duration) {
    this.metrics.dataUpdates.push({
      componentId,
      duration,
      timestamp: performance.now()
    });
  }

  recordRender(componentId, duration) {
    this.metrics.renders.push({
      componentId,
      duration,
      timestamp: performance.now()
    });
  }

  recordResize() {
    this.metrics.resizes.push({
      timestamp: performance.now()
    });
  }

  recordError(componentId, error) {
    this.metrics.errors.push({
      componentId,
      error: error.message,
      timestamp: performance.now()
    });
  }

  getReport() {
    const now = performance.now();
    const uptime = now - this.startTime;

    return {
      uptime,
      dataUpdates: {
        count: this.metrics.dataUpdates.length,
        averageDuration: this.getAverageDuration(this.metrics.dataUpdates),
        lastUpdate: this.getLastTimestamp(this.metrics.dataUpdates)
      },
      renders: {
        count: this.metrics.renders.length,
        averageDuration: this.getAverageDuration(this.metrics.renders),
        lastRender: this.getLastTimestamp(this.metrics.renders)
      },
      resizes: {
        count: this.metrics.resizes.length,
        lastResize: this.getLastTimestamp(this.metrics.resizes)
      },
      errors: {
        count: this.metrics.errors.length,
        lastError: this.getLastTimestamp(this.metrics.errors)
      }
    };
  }

  getAverageDuration(metrics) {
    if (metrics.length === 0) return 0;
    const total = metrics.reduce((sum, metric) => sum + metric.duration, 0);
    return total / metrics.length;
  }

  getLastTimestamp(metrics) {
    if (metrics.length === 0) return null;
    return Math.max(...metrics.map(m => m.timestamp));
  }

  stop() {
    this.metrics = {
      dataUpdates: [],
      renders: [],
      resizes: [],
      errors: []
    };
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { PynomaryComponents, PerformanceMonitor };
}

// Global access
window.PynomaryComponents = PynomaryComponents;