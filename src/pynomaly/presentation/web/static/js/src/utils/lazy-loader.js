/**
 * Advanced Lazy Loading Utility with Performance Optimization
 * Implements dynamic imports, intersection observer, and intelligent caching
 */

class LazyLoader {
  constructor(options = {}) {
    this.options = {
      rootMargin: '50px 0px',
      threshold: 0.1,
      preloadDistance: 2,
      enablePrefetch: true,
      enablePreconnect: true,
      loadingClass: 'lazy-loading',
      loadedClass: 'lazy-loaded',
      errorClass: 'lazy-error',
      ...options
    };
    
    this.loadedComponents = new Set();
    this.loadingComponents = new Set();
    this.componentCache = new Map();
    this.observer = null;
    this.loadQueue = [];
    this.isProcessingQueue = false;
    this.performanceMetrics = {
      totalLoads: 0,
      successfulLoads: 0,
      failedLoads: 0,
      averageLoadTime: 0,
      cacheHits: 0
    };
    
    this.initIntersectionObserver();
    this.initPerformanceMonitoring();
  }

  initIntersectionObserver() {
    if ('IntersectionObserver' in window) {
      this.observer = new IntersectionObserver(
        (entries) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              this.queueComponentLoad(entry.target);
              this.observer.unobserve(entry.target);
            }
          });
        },
        {
          rootMargin: this.options.rootMargin,
          threshold: this.options.threshold
        }
      );
    }
  }

  initPerformanceMonitoring() {
    // Monitor component loading performance
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach(entry => {
          if (entry.name.includes('component-load')) {
            this.updatePerformanceMetrics(entry.duration);
          }
        });
      });
      observer.observe({ entryTypes: ['measure'] });
    }
  }

  // Queue component loading to avoid blocking the main thread
  queueComponentLoad(element) {
    const componentName = element.dataset.component;
    
    if (this.loadingComponents.has(componentName) || this.loadedComponents.has(componentName)) {
      return;
    }
    
    this.loadQueue.push({ element, componentName, timestamp: Date.now() });
    this.processLoadQueue();
  }

  async processLoadQueue() {
    if (this.isProcessingQueue || this.loadQueue.length === 0) {
      return;
    }
    
    this.isProcessingQueue = true;
    
    // Process up to 2 components concurrently
    const batchSize = 2;
    const batch = this.loadQueue.splice(0, batchSize);
    
    const loadPromises = batch.map(({ element, componentName }) => 
      this.loadComponent(element, componentName)
    );
    
    await Promise.allSettled(loadPromises);
    
    this.isProcessingQueue = false;
    
    // Process remaining queue
    if (this.loadQueue.length > 0) {
      setTimeout(() => this.processLoadQueue(), 10);
    }
  }

  async loadComponent(element, componentName) {
    if (this.loadingComponents.has(componentName) || this.loadedComponents.has(componentName)) {
      return;
    }

    this.loadingComponents.add(componentName);
    const startTime = performance.now();
    
    // Mark performance measurement start
    performance.mark(`component-load-${componentName}-start`);

    try {
      // Show loading state
      this.showLoadingState(element);

      // Check cache first
      let component = this.componentCache.get(componentName);
      let isCacheHit = false;
      
      if (component) {
        this.performanceMetrics.cacheHits++;
        isCacheHit = true;
      } else {
        // Dynamic import based on component type
        component = await this.loadComponentModule(componentName);
        
        // Cache the component module
        this.componentCache.set(componentName, component);
      }

      // Initialize component
      if (component.default) {
        await component.default.init(element);
      } else if (component.init) {
        await component.init(element);
      }

      this.loadedComponents.add(componentName);
      this.loadingComponents.delete(componentName);
      this.hideLoadingState(element);
      
      // Add loaded class
      element.classList.add(this.options.loadedClass);

      // Mark performance measurement end
      performance.mark(`component-load-${componentName}-end`);
      performance.measure(
        `component-load-${componentName}`,
        `component-load-${componentName}-start`,
        `component-load-${componentName}-end`
      );

      // Update metrics
      const loadTime = performance.now() - startTime;
      this.updatePerformanceMetrics(loadTime);
      this.performanceMetrics.successfulLoads++;

      // Dispatch loaded event
      element.dispatchEvent(new CustomEvent('component-loaded', {
        detail: { 
          componentName, 
          loadTime, 
          isCacheHit,
          performance: this.getPerformanceMetrics()
        }
      }));

      // Prefetch related components
      if (this.options.enablePrefetch) {
        this.prefetchRelatedComponents(componentName);
      }

    } catch (error) {
      console.error(`Failed to load component ${componentName}:`, error);
      this.showErrorState(element);
      this.loadingComponents.delete(componentName);
      this.performanceMetrics.failedLoads++;
      
      // Add error class
      element.classList.add(this.options.errorClass);
      
      // Dispatch error event
      element.dispatchEvent(new CustomEvent('component-error', {
        detail: { componentName, error: error.message }
      }));
    }
    
    this.performanceMetrics.totalLoads++;
  }

  updatePerformanceMetrics(loadTime) {
    const { totalLoads, averageLoadTime } = this.performanceMetrics;
    this.performanceMetrics.averageLoadTime = 
      (averageLoadTime * totalLoads + loadTime) / (totalLoads + 1);
  }

  getPerformanceMetrics() {
    return {
      ...this.performanceMetrics,
      cacheHitRate: this.performanceMetrics.cacheHits / this.performanceMetrics.totalLoads,
      successRate: this.performanceMetrics.successfulLoads / this.performanceMetrics.totalLoads
    };
  }

  showLoadingState(element) {
    element.innerHTML = `
      <div class="flex items-center justify-center p-8">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span class="ml-2 text-gray-600">Loading...</span>
      </div>
    `;
  }

  hideLoadingState(element) {
    const loadingElement = element.querySelector('.animate-spin');
    if (loadingElement) {
      loadingElement.parentElement.remove();
    }
  }

  showErrorState(element) {
    element.innerHTML = `
      <div class="flex items-center justify-center p-8 text-red-600">
        <svg class="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>Failed to load component</span>
      </div>
    `;
  }

  observe(element) {
    if (this.observer) {
      this.observer.observe(element);
    } else {
      // Fallback for browsers without IntersectionObserver
      this.loadComponent(element);
    }
  }

  // Prefetch related components based on usage patterns
  async prefetchRelatedComponents(componentName) {
    const relatedComponents = this.getRelatedComponents(componentName);
    
    for (const relatedComponent of relatedComponents) {
      if (!this.componentCache.has(relatedComponent) && !this.loadingComponents.has(relatedComponent)) {
        try {
          // Load in background with low priority
          setTimeout(async () => {
            const module = await this.loadComponentModule(relatedComponent);
            this.componentCache.set(relatedComponent, module);
          }, 100);
        } catch (error) {
          console.warn(`Failed to prefetch ${relatedComponent}:`, error);
        }
      }
    }
  }

  getRelatedComponents(componentName) {
    const relationships = {
      'dashboard': ['chart', 'real-time'],
      'chart': ['visualization', 'echarts'],
      'visualization': ['chart', 'echarts'],
      'echarts': ['chart', 'visualization'],
      'real-time': ['dashboard', 'chart'],
      'data-uploader': ['dashboard', 'chart'],
      'anomaly-detector': ['dashboard', 'chart', 'real-time']
    };
    
    return relationships[componentName] || [];
  }

  // Preload critical components
  async preloadCritical(componentNames) {
    const loadPromises = componentNames.map(async (componentName) => {
      if (!this.componentCache.has(componentName)) {
        try {
          const module = await this.loadComponentModule(componentName);
          this.componentCache.set(componentName, module);
        } catch (error) {
          console.warn(`Failed to preload ${componentName}:`, error);
        }
      }
    });

    await Promise.all(loadPromises);
  }

  // Intelligent preloading based on user behavior
  async preloadBasedOnBehavior() {
    const userBehavior = this.getUserBehaviorData();
    const likelyComponents = this.predictLikelyComponents(userBehavior);
    
    for (const componentName of likelyComponents) {
      if (!this.componentCache.has(componentName)) {
        try {
          const module = await this.loadComponentModule(componentName);
          this.componentCache.set(componentName, module);
        } catch (error) {
          console.warn(`Failed to preload ${componentName}:`, error);
        }
      }
    }
  }

  getUserBehaviorData() {
    // Get user behavior from localStorage or analytics
    const behavior = localStorage.getItem('component-usage-patterns');
    return behavior ? JSON.parse(behavior) : {};
  }

  predictLikelyComponents(behavior) {
    // Simple prediction based on usage frequency
    const usage = behavior.usage || {};
    return Object.keys(usage)
      .sort((a, b) => usage[b] - usage[a])
      .slice(0, 3); // Top 3 most used components
  }

  async loadComponentModule(componentName) {
    switch (componentName) {
      case 'chart':
        return import('../components/chart-components.js');
      case 'dashboard':
        return import('../components/dashboard-layout.js');
      case 'visualization':
        return import('../components/d3-charts-demo.js');
      case 'echarts':
        return import('../components/echarts-dashboard.js');
      case 'real-time':
        return import('../components/real-time-dashboard.js');
      case 'data-uploader':
        return import('../components/data-uploader.js');
      case 'anomaly-detector':
        return import('../components/anomaly-detector.js');
      case 'settings':
        return import('../components/settings-components.js');
      default:
        throw new Error(`Unknown component: ${componentName}`);
    }
  }

  // Performance optimization: cleanup unused components
  cleanupUnusedComponents() {
    const now = Date.now();
    const maxAge = 5 * 60 * 1000; // 5 minutes
    
    this.componentCache.forEach((value, key) => {
      if (value._lastUsed && now - value._lastUsed > maxAge) {
        this.componentCache.delete(key);
      }
    });
  }

  // Resource hints for better loading performance
  addResourceHints() {
    if (!this.options.enablePreconnect) return;
    
    const head = document.head;
    
    // Add preconnect for CDN resources
    const preconnectLinks = [
      'https://cdn.jsdelivr.net',
      'https://unpkg.com',
      'https://cdnjs.cloudflare.com'
    ];
    
    preconnectLinks.forEach(href => {
      if (!document.querySelector(`link[href="${href}"]`)) {
        const link = document.createElement('link');
        link.rel = 'preconnect';
        link.href = href;
        head.appendChild(link);
      }
    });
  }
}

// Global lazy loader instance
const lazyLoader = new LazyLoader();

// Initialize lazy loading on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  // Add resource hints
  lazyLoader.addResourceHints();
  
  // Find all lazy-load elements
  const lazyElements = document.querySelectorAll('[data-component]');

  lazyElements.forEach(element => {
    lazyLoader.observe(element);
  });

  // Preload critical components
  lazyLoader.preloadCritical(['dashboard', 'chart']);
  
  // Preload based on user behavior after a delay
  setTimeout(() => {
    lazyLoader.preloadBasedOnBehavior();
  }, 2000);
  
  // Cleanup unused components periodically
  setInterval(() => {
    lazyLoader.cleanupUnusedComponents();
  }, 5 * 60 * 1000); // Every 5 minutes
});

// Handle HTMX dynamic content
if (typeof htmx !== 'undefined') {
  htmx.on('htmx:afterSwap', () => {
    const lazyElements = document.querySelectorAll('[data-component]:not(.lazy-loaded)');
    lazyElements.forEach(element => {
      lazyLoader.observe(element);
    });
  });
}

// Export for module usage
export default lazyLoader;
