/**
 * Optimized Main Application Entry Point
 * Integrates performance optimizations, lazy loading, and intelligent resource management
 */

import { performanceOptimizer } from './utils/performance-optimizer.js';
import { chartPerformanceManager } from './utils/chart-performance.js';
import { dataCacheManager } from './utils/data-cache.js';
import { dashboardState } from './state/dashboard-state.js';

class OptimizedPynomAlyApp {
  constructor() {
    this.components = new Map();
    this.lazyModules = new Map();
    this.initialized = false;
    this.version = "1.2.0";

    this.performanceMetrics = {
      initTime: 0,
      firstContentfulPaint: 0,
      timeToInteractive: 0,
      componentLoadTimes: new Map(),
    };

    // Bind critical performance measurement
    this.startTime = performance.now();
    this.measureCriticalPath();

    this.init();
  }

  measureCriticalPath() {
    // Measure critical rendering path metrics
    new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        if (entry.name === 'first-contentful-paint') {
          this.performanceMetrics.firstContentfulPaint = entry.startTime;
        }
      });
    }).observe({ entryTypes: ['paint'] });

    // Measure Time to Interactive
    document.addEventListener('DOMContentLoaded', () => {
      this.measureTimeToInteractive();
    });
  }

  async measureTimeToInteractive() {
    // Simple TTI measurement
    const startTime = performance.now();

    await new Promise(resolve => {
      const checkInteractive = () => {
        if (this.isPageInteractive()) {
          this.performanceMetrics.timeToInteractive = performance.now() - startTime;
          resolve();
        } else {
          setTimeout(checkInteractive, 100);
        }
      };
      checkInteractive();
    });
  }

  isPageInteractive() {
    // Check if main thread is not blocked and critical resources are loaded
    return !document.querySelector('.loading') &&
           document.readyState === 'complete' &&
           this.initialized;
  }

  async init() {
    try {
      console.log(`🚀 Initializing Optimized Pynomaly v${this.version}`);

      // Initialize performance systems first
      await this.initializePerformanceSystems();

      // Preload critical resources
      await this.preloadCriticalResources();

      // Initialize core functionality
      await this.initializeCore();

      // Setup lazy loading for non-critical components
      this.setupLazyLoading();

      // Initialize progressive enhancement
      await this.initializeProgressively();

      this.performanceMetrics.initTime = performance.now() - this.startTime;
      this.initialized = true;

      console.log(`✅ App initialized in ${this.performanceMetrics.initTime.toFixed(2)}ms`);

      // Emit ready event
      document.dispatchEvent(new CustomEvent('pynomaly:optimized-ready', {
        detail: { metrics: this.performanceMetrics }
      }));

    } catch (error) {
      console.error('❌ Failed to initialize optimized application:', error);
      this.handleInitializationError(error);
    }
  }

  async initializePerformanceSystems() {
    // Initialize performance optimizer
    performanceOptimizer.init();

    // Initialize chart performance manager
    chartPerformanceManager.init();

    // Initialize data cache with custom options
    await dataCacheManager.init();

    // Setup performance monitoring
    this.setupAdvancedPerformanceMonitoring();
  }

  setupAdvancedPerformanceMonitoring() {
    // Enhanced performance monitoring beyond the basic systems

    // Monitor frame rate
    let frameCount = 0;
    let lastFrameTime = performance.now();

    const measureFPS = () => {
      frameCount++;
      const currentTime = performance.now();

      if (currentTime - lastFrameTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastFrameTime));

        if (fps < 30) {
          console.warn(`Low FPS detected: ${fps}fps`);
          this.handleLowPerformance();
        }

        frameCount = 0;
        lastFrameTime = currentTime;
      }

      requestAnimationFrame(measureFPS);
    };

    requestAnimationFrame(measureFPS);

    // Monitor memory usage
    setInterval(() => {
      if ('memory' in performance) {
        const memoryInfo = performance.memory;
        const usageRatio = memoryInfo.usedJSHeapSize / memoryInfo.jsHeapSizeLimit;

        if (usageRatio > 0.9) {
          console.warn('High memory usage detected:', usageRatio);
          this.handleHighMemoryUsage();
        }
      }
    }, 10000);

    // Monitor network quality
    this.monitorNetworkQuality();
  }

  monitorNetworkQuality() {
    if ('connection' in navigator) {
      const connection = navigator.connection;

      const checkConnection = () => {
        const effectiveType = connection.effectiveType;
        const downlink = connection.downlink;

        // Adjust performance settings based on connection
        if (effectiveType === '2g' || downlink < 1) {
          this.enableLowBandwidthMode();
        } else if (effectiveType === '4g' && downlink > 10) {
          this.enableHighPerformanceMode();
        }
      };

      connection.addEventListener('change', checkConnection);
      checkConnection();
    }
  }

  async preloadCriticalResources() {
    const criticalResources = [
      '/static/css/design-system.css',
      '/static/js/state/dashboard-state.js',
      // Add other critical resources
    ];

    // Preload with high priority
    const preloadPromises = criticalResources.map(resource => {
      return new Promise((resolve) => {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.href = resource;
        link.as = resource.endsWith('.css') ? 'style' : 'script';
        link.onload = resolve;
        link.onerror = resolve; // Don't fail on preload errors
        document.head.appendChild(link);

        // Timeout after 5 seconds
        setTimeout(resolve, 5000);
      });
    });

    await Promise.allSettled(preloadPromises);
  }

  async initializeCore() {
    // Initialize only essential core functionality
    await this.initializeState();
    await this.initializeServiceWorker();
    this.setupCriticalEventHandlers();
    this.initializeTheme();
  }

  async initializeState() {
    // Initialize state management with cached data
    try {
      const cachedState = await dataCacheManager.get('dashboard-state');
      if (cachedState) {
        dashboardState.hydrate(cachedState);
      }
    } catch (error) {
      console.warn('Failed to load cached state:', error);
    }

    // Subscribe to state changes for caching
    dashboardState.subscribe((action, prevState, newState) => {
      // Cache state changes with debouncing
      performanceOptimizer.debounce(() => {
        dataCacheManager.set('dashboard-state', newState, { ttl: 30 * 60 * 1000 });
      }, 'cache-state', 1000);
    });
  }

  async initializeServiceWorker() {
    if ('serviceWorker' in navigator) {
      try {
        const registration = await navigator.serviceWorker.register('/sw.js');
        console.log('Service Worker registered');

        // Handle SW updates
        registration.addEventListener('updatefound', () => {
          this.handleServiceWorkerUpdate(registration);
        });

      } catch (error) {
        console.warn('Service Worker registration failed:', error);
      }
    }
  }

  setupLazyLoading() {
    // Define lazy-loadable modules
    this.lazyModules.set('charts', () => import('./charts/anomaly-timeline.js'));
    this.lazyModules.set('heatmap', () => import('./charts/anomaly-heatmap.js'));
    this.lazyModules.set('realtime', () => import('./charts/real-time-dashboard.js'));
    this.lazyModules.set('forms', () => import('./components/interactive-forms.js'));
    this.lazyModules.set('advanced-dashboard', () => import('./components/advanced-dashboard.js'));
    this.lazyModules.set('collaboration', () => import('./collaboration.js'));

    // Setup intersection observers for lazy loading
    performanceOptimizer.enableLazyLoading();

    // Listen for component load requests
    document.addEventListener('componentRequest', (event) => {
      this.loadComponentOnDemand(event.detail.componentName, event.detail.element);
    });
  }

  async loadComponentOnDemand(componentName, element) {
    const startTime = performance.now();

    try {
      // Check if component is already loaded
      if (this.components.has(componentName)) {
        return this.components.get(componentName);
      }

      // Check if we have a lazy loader for this component
      const moduleLoader = this.lazyModules.get(componentName);
      if (!moduleLoader) {
        console.warn(`No lazy loader found for component: ${componentName}`);
        return null;
      }

      // Load the module
      element?.classList.add('loading');
      const module = await moduleLoader();
      const Component = module.default || module[componentName];

      if (Component) {
        const instance = new Component(element);
        this.components.set(componentName, instance);

        // Apply performance optimizations
        performanceOptimizer.optimizeComponent(instance);

        const loadTime = performance.now() - startTime;
        this.performanceMetrics.componentLoadTimes.set(componentName, loadTime);

        console.log(`Component ${componentName} loaded in ${loadTime.toFixed(2)}ms`);

        element?.classList.remove('loading');
        element?.classList.add('loaded');

        return instance;
      }

    } catch (error) {
      console.error(`Failed to load component ${componentName}:`, error);
      element?.classList.remove('loading');
      element?.classList.add('error');
    }
  }

  async initializeProgressively() {
    // Progressive enhancement - load features based on importance and visibility

    // Phase 1: Load visible components immediately
    await this.loadVisibleComponents();

    // Phase 2: Load above-the-fold components with slight delay
    setTimeout(() => this.loadAboveFoldComponents(), 100);

    // Phase 3: Load below-the-fold components when idle
    this.scheduleIdleWork(() => this.loadBelowFoldComponents());

    // Phase 4: Load enhancement features
    this.scheduleIdleWork(() => this.loadEnhancementFeatures(), 2000);
  }

  async loadVisibleComponents() {
    // Load components that are immediately visible
    const visibleContainers = document.querySelectorAll('[data-component]:not([data-lazy])');

    for (const container of visibleContainers) {
      const componentName = container.dataset.component;
      if (componentName) {
        await this.loadComponentOnDemand(componentName, container);
      }
    }
  }

  async loadAboveFoldComponents() {
    // Load components in the viewport or slightly below
    const containers = document.querySelectorAll('[data-component][data-priority="high"]');

    for (const container of containers) {
      const rect = container.getBoundingClientRect();
      if (rect.top < window.innerHeight * 1.5) { // 1.5x viewport height
        const componentName = container.dataset.component;
        await this.loadComponentOnDemand(componentName, container);
      }
    }
  }

  loadBelowFoldComponents() {
    // Load remaining components that are below the fold
    const containers = document.querySelectorAll('[data-component][data-lazy="true"]');

    containers.forEach(container => {
      performanceOptimizer.observers.get('lazyComponents')?.observe(container);
    });
  }

  loadEnhancementFeatures() {
    // Load progressive enhancement features
    this.loadFeatureIfSupported('webgl', () => this.enableWebGLAcceleration());
    this.loadFeatureIfSupported('indexeddb', () => this.enableOfflineSupport());
    this.loadFeatureIfSupported('serviceworker', () => this.enablePushNotifications());
    this.loadFeatureIfSupported('intersection-observer', () => this.enableAdvancedLazyLoading());
  }

  loadFeatureIfSupported(feature, loader) {
    const supported = this.checkFeatureSupport(feature);
    if (supported) {
      this.scheduleIdleWork(loader);
    }
  }

  checkFeatureSupport(feature) {
    const features = {
      webgl: () => {
        const canvas = document.createElement('canvas');
        return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
      },
      indexeddb: () => 'indexedDB' in window,
      serviceworker: () => 'serviceWorker' in navigator,
      'intersection-observer': () => 'IntersectionObserver' in window,
      webassembly: () => 'WebAssembly' in window,
    };

    return features[feature] ? features[feature]() : false;
  }

  scheduleIdleWork(callback, timeout = 5000) {
    if ('requestIdleCallback' in window) {
      requestIdleCallback(callback, { timeout });
    } else {
      setTimeout(callback, 1);
    }
  }

  setupCriticalEventHandlers() {
    // Only setup absolutely essential event handlers

    // Handle page visibility changes for performance
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.pauseNonEssentialOperations();
      } else {
        this.resumeOperations();
      }
    });

    // Handle resize events with throttling
    window.addEventListener('resize', () => {
      performanceOptimizer.throttle(() => {
        this.handleResize();
      }, 'window-resize');
    });

    // Handle critical errors
    window.addEventListener('error', (event) => {
      this.handleCriticalError(event.error);
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.handleCriticalError(event.reason);
    });
  }

  initializeTheme() {
    // Initialize theme with minimal DOM manipulation
    const savedTheme = localStorage.getItem('pynomaly-theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    // Cache theme preference
    dataCacheManager.set('theme-preference', savedTheme, { ttl: 24 * 60 * 60 * 1000 });
  }

  // === PERFORMANCE OPTIMIZATION HANDLERS ===

  handleLowPerformance() {
    console.log('Switching to performance mode');

    // Reduce animation quality
    document.documentElement.classList.add('reduced-motion');

    // Reduce chart update frequency
    this.components.forEach(component => {
      if (component.setUpdateInterval) {
        component.setUpdateInterval(2000); // Reduce to 2 seconds
      }
    });

    // Reduce data polling frequency
    dashboardState.dispatch(dashboardState.actions.setRealTimeUpdateRate(5000));
  }

  handleHighMemoryUsage() {
    console.log('Optimizing memory usage');

    // Force garbage collection if available
    if (window.gc) {
      window.gc();
    }

    // Clear data caches
    dataCacheManager.clear();

    // Reduce chart data retention
    this.components.forEach(component => {
      if (component.reduceDataRetention) {
        component.reduceDataRetention();
      }
    });

    // Request components to free resources
    document.dispatchEvent(new CustomEvent('memoryPressure', {
      detail: { severity: 'high' }
    }));
  }

  enableLowBandwidthMode() {
    console.log('Enabling low bandwidth mode');

    // Reduce image quality
    document.documentElement.classList.add('low-bandwidth');

    // Disable auto-refresh
    dashboardState.dispatch(dashboardState.actions.setPreference('autoRefresh', false));

    // Use compressed data transfers
    this.enableDataCompression = true;
  }

  enableHighPerformanceMode() {
    console.log('Enabling high performance mode');

    // Enable WebGL acceleration
    this.enableWebGLAcceleration();

    // Increase chart update frequency
    dashboardState.dispatch(dashboardState.actions.setRealTimeUpdateRate(500));

    // Enable advanced features
    this.enableAdvancedFeatures();
  }

  enableWebGLAcceleration() {
    // Enable WebGL for chart rendering where supported
    chartPerformanceManager.options.enableWebGL = true;

    document.documentElement.classList.add('webgl-enabled');
  }

  enableAdvancedFeatures() {
    // Enable resource-intensive features
    this.scheduleIdleWork(() => {
      this.loadComponentOnDemand('collaboration', null);
    });
  }

  pauseNonEssentialOperations() {
    // Pause animations and timers
    this.components.forEach(component => {
      if (component.pause) component.pause();
    });

    // Pause data polling
    dashboardState.dispatch(dashboardState.actions.setPaused(true));
  }

  resumeOperations() {
    // Resume operations
    this.components.forEach(component => {
      if (component.resume) component.resume();
    });

    // Resume data polling
    dashboardState.dispatch(dashboardState.actions.setPaused(false));

    // Refresh data to catch up
    this.scheduleIdleWork(() => this.refreshData());
  }

  handleResize() {
    // Resize charts efficiently
    this.components.forEach(component => {
      if (component.resize) {
        performanceOptimizer.throttle(() => {
          component.resize();
        }, `resize-${component.id || Math.random()}`);
      }
    });
  }

  async refreshData() {
    try {
      // Batch data refresh operations
      const refreshPromises = [
        this.fetchDatasets(),
        this.fetchRecentAnomalies(),
        this.fetchMetrics(),
      ];

      const [datasets, anomalies, metrics] = await Promise.allSettled(refreshPromises);

      // Update state with successful results
      if (datasets.status === 'fulfilled') {
        dashboardState.dispatch(dashboardState.actions.setDatasets(datasets.value));
      }

      if (anomalies.status === 'fulfilled') {
        dashboardState.dispatch(dashboardState.actions.addAnomalies(anomalies.value));
      }

      if (metrics.status === 'fulfilled') {
        dashboardState.dispatch(dashboardState.actions.updateMetrics(metrics.value));
      }

    } catch (error) {
      console.error('Data refresh failed:', error);
    }
  }

  async fetchDatasets() {
    // Check cache first
    const cached = await dataCacheManager.get('datasets');
    if (cached) return cached;

    // Fetch from API
    const response = await fetch('/api/datasets');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();

    // Cache result
    await dataCacheManager.set('datasets', data, { ttl: 5 * 60 * 1000 });

    return data;
  }

  async fetchRecentAnomalies() {
    const cached = await dataCacheManager.get('recent-anomalies');
    if (cached) return cached;

    const response = await fetch('/api/anomalies/recent');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    await dataCacheManager.set('recent-anomalies', data, { ttl: 2 * 60 * 1000 });

    return data;
  }

  async fetchMetrics() {
    const response = await fetch('/api/metrics/summary');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    return response.json();
  }

  // === ERROR HANDLING ===

  handleCriticalError(error) {
    console.error('Critical error:', error);

    // Try to recover gracefully
    this.attemptGracefulRecovery(error);
  }

  attemptGracefulRecovery(error) {
    // Clear problematic caches
    dataCacheManager.clear();

    // Reset components that might be in a bad state
    this.components.forEach((component, name) => {
      try {
        if (component.reset) {
          component.reset();
        }
      } catch (resetError) {
        console.warn(`Failed to reset component ${name}:`, resetError);
        this.components.delete(name);
      }
    });

    // Show user-friendly error message
    this.showRecoveryNotification();
  }

  showRecoveryNotification() {
    dashboardState.dispatch(dashboardState.actions.addNotification({
      type: 'warning',
      title: 'Application Recovered',
      message: 'The application encountered an issue but has recovered automatically.',
      duration: 5000,
    }));
  }

  handleInitializationError(error) {
    console.error('Initialization failed:', error);

    // Show fallback UI
    document.body.innerHTML = `
      <div class="error-container">
        <div class="error-content">
          <h1>Application Failed to Load</h1>
          <p>We're sorry, but the application couldn't start properly.</p>
          <button onclick="location.reload()" class="btn-primary">
            Reload Application
          </button>
        </div>
      </div>
    `;
  }

  handleServiceWorkerUpdate(registration) {
    const newWorker = registration.installing;

    newWorker.addEventListener('statechange', () => {
      if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
        // New version available
        dashboardState.dispatch(dashboardState.actions.addNotification({
          type: 'info',
          title: 'Update Available',
          message: 'A new version is available. Refresh to update.',
          actions: [
            {
              label: 'Refresh',
              action: () => location.reload(),
            },
          ],
        }));
      }
    });
  }

  // === PUBLIC API ===

  getComponent(name) {
    return this.components.get(name);
  }

  async loadComponent(name, element) {
    return this.loadComponentOnDemand(name, element);
  }

  getPerformanceMetrics() {
    return {
      ...this.performanceMetrics,
      optimizer: performanceOptimizer.getPerformanceReport(),
      charts: chartPerformanceManager.getPerformanceStats(),
      cache: dataCacheManager.getStats(),
    };
  }

  destroy() {
    // Clean up all components
    this.components.forEach(component => {
      if (component.destroy) component.destroy();
    });

    this.components.clear();

    // Clean up performance systems
    performanceOptimizer.destroy();
    chartPerformanceManager.destroy();
    dataCacheManager.destroy();
  }
}

// Create and initialize the optimized application
const app = new OptimizedPynomAlyApp();

// Export for global access
window.PynomAlyOptimized = app;

// Also maintain backwards compatibility
window.PynomAly = app;

export default app;
