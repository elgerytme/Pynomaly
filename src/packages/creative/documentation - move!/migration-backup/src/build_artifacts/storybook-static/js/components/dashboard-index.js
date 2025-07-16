/**
 * Dashboard Components Entry Point - Code Splitting and Lazy Loading
 * Provides efficient loading of dashboard components based on user navigation
 */

/**
 * Dashboard Component Loader with intelligent caching and preloading
 */
export class DashboardComponentLoader {
  constructor() {
    this.loadedComponents = new Map();
    this.loadingPromises = new Map();
    this.componentRegistry = new Map();
    this.preloadQueue = [];

    this._initializeRegistry();
  }

  /**
   * Load advanced dashboard with drag-drop functionality
   */
  async loadAdvancedDashboard(container, config = {}) {
    return this._loadComponent('advanced-dashboard', container, config, () =>
      import(/* webpackChunkName: "dashboard-advanced" */ './advanced-dashboard.js')
    );
  }

  /**
   * Load anomaly detection dashboard
   */
  async loadAnomalyDashboard(container, config = {}) {
    return this._loadComponent('anomaly-dashboard', container, config, () =>
      import(/* webpackChunkName: "dashboard-anomaly" */ './AnomalyDashboard.js')
    );
  }

  /**
   * Load real-time monitoring dashboard
   */
  async loadRealtimeMonitoring(container, config = {}) {
    return this._loadComponent('realtime-monitoring', container, config, () =>
      import(/* webpackChunkName: "dashboard-realtime" */ './realtime-monitoring.js')
    );
  }

  /**
   * Load dashboard layout system
   */
  async loadDashboardLayout(container, config = {}) {
    return this._loadComponent('dashboard-layout', container, config, () =>
      import(/* webpackChunkName: "dashboard-layout" */ './dashboard-layout.js')
    );
  }

  /**
   * Load interactive forms with validation
   */
  async loadInteractiveForms(container, config = {}) {
    return this._loadComponent('interactive-forms', container, config, () =>
      import(/* webpackChunkName: "forms-interactive" */ './interactive-forms.js')
    );
  }

  /**
   * Load data uploader component
   */
  async loadDataUploader(container, config = {}) {
    return this._loadComponent('data-uploader', container, config, () =>
      import(/* webpackChunkName: "uploader" */ './data-uploader.js')
    );
  }

  /**
   * Load model training interface
   */
  async loadTrainingInterface(container, config = {}) {
    return this._loadComponent('training-interface', container, config, () =>
      import(/* webpackChunkName: "training" */ './training-interface.js')
    );
  }

  /**
   * Load explainability dashboard
   */
  async loadExplainabilityDashboard(container, config = {}) {
    return this._loadComponent('explainability-dashboard', container, config, () =>
      import(/* webpackChunkName: "explainability" */ './explainability-dashboard.js')
    );
  }

  /**
   * Load AutoML interface
   */
  async loadAutoMLInterface(container, config = {}) {
    return this._loadComponent('automl-interface', container, config, () =>
      import(/* webpackChunkName: "automl" */ './automl-interface.js')
    );
  }

  /**
   * Load ensemble management interface
   */
  async loadEnsembleInterface(container, config = {}) {
    return this._loadComponent('ensemble-interface', container, config, () =>
      import(/* webpackChunkName: "ensemble" */ './ensemble-interface.js')
    );
  }

  /**
   * Load component based on data attributes
   */
  async loadComponentFromElement(element) {
    const componentType = element.dataset.component;
    const config = this._parseConfigFromElement(element);

    const loader = this.componentRegistry.get(componentType);
    if (!loader) {
      throw new Error(`Unknown component type: ${componentType}`);
    }

    return loader.call(this, element, config);
  }

  /**
   * Preload components based on user behavior and page context
   */
  async preloadComponents(context = 'dashboard') {
    const preloadStrategies = {
      'dashboard': ['dashboard-layout', 'anomaly-dashboard'],
      'detection': ['training-interface', 'data-uploader'],
      'analysis': ['explainability-dashboard', 'realtime-monitoring'],
      'automl': ['automl-interface', 'training-interface'],
      'ensemble': ['ensemble-interface', 'anomaly-dashboard']
    };

    const componentsToPreload = preloadStrategies[context] || [];

    // Add to preload queue
    this.preloadQueue.push(...componentsToPreload);

    // Process preload queue with debouncing
    this._processPreloadQueue();
  }

  /**
   * Preload components on idle time
   */
  preloadOnIdle() {
    if ('requestIdleCallback' in window) {
      requestIdleCallback(() => {
        this._processPreloadQueue();
      });
    } else {
      // Fallback for browsers without requestIdleCallback
      setTimeout(() => this._processPreloadQueue(), 1000);
    }
  }

  /**
   * Get component performance metrics
   */
  getComponentMetrics() {
    const metrics = {
      loadedComponents: this.loadedComponents.size,
      loadingComponents: this.loadingPromises.size,
      totalMemoryUsage: this._estimateMemoryUsage(),
      loadTimes: this._getLoadTimes()
    };

    return metrics;
  }

  /**
   * Cleanup components and free memory
   */
  cleanup(componentType = null) {
    if (componentType) {
      const component = this.loadedComponents.get(componentType);
      if (component && typeof component.destroy === 'function') {
        component.destroy();
      }
      this.loadedComponents.delete(componentType);
    } else {
      // Cleanup all components
      this.loadedComponents.forEach((component, type) => {
        if (component && typeof component.destroy === 'function') {
          component.destroy();
        }
      });
      this.loadedComponents.clear();
      this.loadingPromises.clear();
    }
  }

  // Private methods

  _initializeRegistry() {
    this.componentRegistry.set('advanced-dashboard', this.loadAdvancedDashboard);
    this.componentRegistry.set('anomaly-dashboard', this.loadAnomalyDashboard);
    this.componentRegistry.set('realtime-monitoring', this.loadRealtimeMonitoring);
    this.componentRegistry.set('dashboard-layout', this.loadDashboardLayout);
    this.componentRegistry.set('interactive-forms', this.loadInteractiveForms);
    this.componentRegistry.set('data-uploader', this.loadDataUploader);
    this.componentRegistry.set('training-interface', this.loadTrainingInterface);
    this.componentRegistry.set('explainability-dashboard', this.loadExplainabilityDashboard);
    this.componentRegistry.set('automl-interface', this.loadAutoMLInterface);
    this.componentRegistry.set('ensemble-interface', this.loadEnsembleInterface);
  }

  async _loadComponent(componentType, container, config, importFn) {
    // Check if already loaded
    if (this.loadedComponents.has(componentType)) {
      return this.loadedComponents.get(componentType);
    }

    // Check if currently loading
    if (this.loadingPromises.has(componentType)) {
      return this.loadingPromises.get(componentType);
    }

    // Record load start time for metrics
    const loadStartTime = performance.now();

    // Create loading promise
    const loadPromise = importFn()
      .then(module => {
        const component = this._instantiateComponent(module, container, config);

        // Record load time
        const loadTime = performance.now() - loadStartTime;
        console.log(`✅ Component loaded: ${componentType} (${loadTime.toFixed(2)}ms)`);

        this.loadedComponents.set(componentType, {
          instance: component,
          loadTime,
          timestamp: Date.now()
        });

        return component;
      })
      .catch(error => {
        console.error(`❌ Failed to load component: ${componentType}`, error);
        throw error;
      })
      .finally(() => {
        this.loadingPromises.delete(componentType);
      });

    this.loadingPromises.set(componentType, loadPromise);
    return loadPromise;
  }

  _instantiateComponent(module, container, config) {
    // Try different instantiation patterns
    const patterns = [
      () => new module.default(container, config),
      () => module.default(container, config),
      () => new module[Object.keys(module)[0]](container, config),
      () => module[Object.keys(module)[0]](container, config)
    ];

    for (const pattern of patterns) {
      try {
        return pattern();
      } catch (error) {
        // Continue to next pattern
      }
    }

    throw new Error('Unable to instantiate component with any known pattern');
  }

  _parseConfigFromElement(element) {
    const config = {};

    // Parse data attributes
    Object.keys(element.dataset).forEach(key => {
      if (key.startsWith('config')) {
        const configKey = key.replace('config', '').toLowerCase();
        let value = element.dataset[key];

        // Try to parse as JSON
        try {
          value = JSON.parse(value);
        } catch (e) {
          // Keep as string
        }

        config[configKey] = value;
      }
    });

    return {
      width: element.clientWidth,
      height: element.clientHeight,
      responsive: true,
      ...config
    };
  }

  _processPreloadQueue() {
    if (this.preloadQueue.length === 0) return;

    // Remove duplicates
    const uniqueComponents = [...new Set(this.preloadQueue)];
    this.preloadQueue = [];

    // Preload components one by one to avoid overwhelming the browser
    uniqueComponents.forEach((componentType, index) => {
      setTimeout(() => {
        this._preloadComponent(componentType);
      }, index * 100); // Stagger preloading
    });
  }

  async _preloadComponent(componentType) {
    if (this.loadedComponents.has(componentType) || this.loadingPromises.has(componentType)) {
      return; // Already loaded or loading
    }

    const loader = this.componentRegistry.get(componentType);
    if (!loader) {
      console.warn(`Cannot preload unknown component: ${componentType}`);
      return;
    }

    try {
      // Create temporary container for preloading
      const tempContainer = document.createElement('div');
      tempContainer.style.display = 'none';
      document.body.appendChild(tempContainer);

      await loader.call(this, tempContainer, { preload: true });

      // Remove temporary container
      document.body.removeChild(tempContainer);

      console.log(`📦 Component preloaded: ${componentType}`);
    } catch (error) {
      console.warn(`Failed to preload component: ${componentType}`, error);
    }
  }

  _estimateMemoryUsage() {
    // Rough estimation based on loaded components
    return this.loadedComponents.size * 50; // KB per component (rough estimate)
  }

  _getLoadTimes() {
    const loadTimes = {};
    this.loadedComponents.forEach((data, type) => {
      loadTimes[type] = data.loadTime;
    });
    return loadTimes;
  }
}

/**
 * Global dashboard component loader instance
 */
export const dashboardLoader = new DashboardComponentLoader();

/**
 * Initialize dashboard components with intelligent loading
 */
export function initializeDashboardComponents() {
  const componentElements = document.querySelectorAll('[data-component]');

  if (!componentElements.length) return;

  // Determine page context for intelligent preloading
  const pageContext = document.body.dataset.page || 'dashboard';

  // Use intersection observer for lazy loading
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const element = entry.target;

        // Add loading indicator
        element.innerHTML = `
          <div class="component-loading">
            <div class="loading-spinner animate-spin"></div>
            <p class="text-sm text-gray-600">Loading component...</p>
          </div>
        `;

        // Load component
        dashboardLoader.loadComponentFromElement(element)
          .then((component) => {
            console.log(`✅ Dashboard component loaded: ${element.dataset.component}`);
          })
          .catch(error => {
            console.error(`❌ Failed to load component: ${element.dataset.component}`, error);
            element.innerHTML = `
              <div class="component-error bg-red-50 border border-red-200 rounded p-4">
                <p class="text-red-800">Failed to load component.</p>
                <button onclick="location.reload()" class="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700">
                  Retry
                </button>
              </div>
            `;
          });

        // Stop observing this element
        observer.unobserve(element);
      }
    });
  }, {
    rootMargin: '100px 0px', // Start loading 100px before visible
    threshold: 0.1
  });

  // Observe all component elements
  componentElements.forEach(element => observer.observe(element));

  // Intelligent preloading based on page context
  dashboardLoader.preloadComponents(pageContext);
  dashboardLoader.preloadOnIdle();
}

/**
 * Performance monitoring and cleanup
 */
export function monitorDashboardPerformance() {
  // Log performance metrics every 30 seconds
  setInterval(() => {
    const metrics = dashboardLoader.getComponentMetrics();
    console.log('📊 Dashboard Performance:', metrics);
  }, 30000);

  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    dashboardLoader.cleanup();
  });

  // Cleanup inactive components every 5 minutes
  setInterval(() => {
    const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;

    dashboardLoader.loadedComponents.forEach((data, type) => {
      if (data.timestamp < fiveMinutesAgo && !document.querySelector(`[data-component="${type}"]`)) {
        console.log(`🧹 Cleaning up inactive component: ${type}`);
        dashboardLoader.cleanup(type);
      }
    });
  }, 5 * 60 * 1000);
}

// Auto-initialize when module is loaded
if (typeof window !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initializeDashboardComponents();
      monitorDashboardPerformance();
    });
  } else {
    initializeDashboardComponents();
    monitorDashboardPerformance();
  }
}
