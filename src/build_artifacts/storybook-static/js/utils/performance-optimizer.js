/**
 * Advanced Performance Optimizer for Pynomaly Web UI
 * Implements lazy loading, debouncing, virtual scrolling, and resource management
 */

export class PerformanceOptimizer {
  constructor(options = {}) {
    this.options = {
      debounceDelay: 300,
      throttleDelay: 16, // 60fps
      virtualScrollThreshold: 100,
      lazyLoadThreshold: 0.1,
      resourceCacheSize: 50,
      performanceMetricsInterval: 5000,
      memoryThreshold: 0.8, // 80% of available memory
      ...options,
    };

    this.observers = new Map();
    this.debouncedFunctions = new Map();
    this.throttledFunctions = new Map();
    this.resourceCache = new Map();
    this.performanceMetrics = {
      renderTimes: [],
      memoryUsage: [],
      networkRequests: [],
      userInteractions: [],
    };

    this.init();
  }

  init() {
    this.initializeIntersectionObserver();
    this.setupPerformanceMonitoring();
    this.setupMemoryManagement();
    this.optimizeAnimations();
    this.setupNetworkOptimization();
    this.initializeWorkers();
  }

  // === LAZY LOADING OPTIMIZATION ===

  initializeIntersectionObserver() {
    if (!('IntersectionObserver' in window)) {
      console.warn('IntersectionObserver not supported, falling back to scroll events');
      return;
    }

    const lazyImageObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            this.loadImage(entry.target);
            lazyImageObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: this.options.lazyLoadThreshold }
    );

    const lazyComponentObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            this.loadComponent(entry.target);
            lazyComponentObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: this.options.lazyLoadThreshold }
    );

    this.observers.set('lazyImages', lazyImageObserver);
    this.observers.set('lazyComponents', lazyComponentObserver);

    this.enableLazyLoading();
  }

  enableLazyLoading() {
    // Lazy load images
    document.querySelectorAll('img[data-src]').forEach(img => {
      this.observers.get('lazyImages').observe(img);
    });

    // Lazy load components
    document.querySelectorAll('[data-lazy-component]').forEach(component => {
      this.observers.get('lazyComponents').observe(component);
    });
  }

  loadImage(img) {
    const src = img.dataset.src;
    if (src) {
      img.src = src;
      img.classList.remove('lazy');
      img.classList.add('loaded');
    }
  }

  async loadComponent(element) {
    const componentName = element.dataset.lazyComponent;
    if (!componentName) return;

    try {
      const module = await import(`../components/${componentName}.js`);
      const Component = module.default || module[componentName];

      if (Component) {
        const instance = new Component(element);
        element.dataset.componentLoaded = 'true';
        element.dispatchEvent(new CustomEvent('componentLoaded', { detail: { componentName, instance } }));
      }
    } catch (error) {
      console.error(`Failed to load component ${componentName}:`, error);
      element.classList.add('component-error');
    }
  }

  // === DEBOUNCING AND THROTTLING ===

  debounce(func, key, delay = this.options.debounceDelay) {
    if (this.debouncedFunctions.has(key)) {
      clearTimeout(this.debouncedFunctions.get(key));
    }

    const timeoutId = setTimeout(() => {
      func();
      this.debouncedFunctions.delete(key);
    }, delay);

    this.debouncedFunctions.set(key, timeoutId);
  }

  throttle(func, key, delay = this.options.throttleDelay) {
    if (this.throttledFunctions.has(key)) {
      return;
    }

    this.throttledFunctions.set(key, true);

    requestAnimationFrame(() => {
      func();
      setTimeout(() => {
        this.throttledFunctions.delete(key);
      }, delay);
    });
  }

  // === VIRTUAL SCROLLING ===

  createVirtualScrollContainer(container, items, renderItem, itemHeight = 50) {
    const virtualContainer = document.createElement('div');
    virtualContainer.className = 'virtual-scroll-container';
    virtualContainer.style.position = 'relative';
    virtualContainer.style.height = `${items.length * itemHeight}px`;

    const viewport = document.createElement('div');
    viewport.className = 'virtual-scroll-viewport';
    viewport.style.height = `${container.clientHeight}px`;
    viewport.style.overflow = 'auto';

    const content = document.createElement('div');
    content.className = 'virtual-scroll-content';
    content.style.position = 'absolute';
    content.style.width = '100%';

    viewport.appendChild(virtualContainer);
    virtualContainer.appendChild(content);
    container.appendChild(viewport);

    let startIndex = 0;
    let endIndex = Math.min(items.length, Math.ceil(container.clientHeight / itemHeight) + 5);

    const updateVisibleItems = () => {
      const scrollTop = viewport.scrollTop;
      const newStartIndex = Math.floor(scrollTop / itemHeight);
      const newEndIndex = Math.min(
        items.length,
        newStartIndex + Math.ceil(container.clientHeight / itemHeight) + 5
      );

      if (newStartIndex !== startIndex || newEndIndex !== endIndex) {
        startIndex = newStartIndex;
        endIndex = newEndIndex;

        content.style.transform = `translateY(${startIndex * itemHeight}px)`;
        content.innerHTML = '';

        for (let i = startIndex; i < endIndex; i++) {
          const item = renderItem(items[i], i);
          content.appendChild(item);
        }
      }
    };

    viewport.addEventListener('scroll', () => {
      this.throttle(updateVisibleItems, `virtual-scroll-${container.id}`);
    });

    // Initial render
    updateVisibleItems();

    return {
      updateItems: (newItems) => {
        items = newItems;
        virtualContainer.style.height = `${items.length * itemHeight}px`;
        updateVisibleItems();
      },
      destroy: () => {
        viewport.removeEventListener('scroll', updateVisibleItems);
        container.removeChild(viewport);
      }
    };
  }

  // === PERFORMANCE MONITORING ===

  setupPerformanceMonitoring() {
    if (!('PerformanceObserver' in window)) return;

    // Monitor Long Tasks
    try {
      const longTaskObserver = new PerformanceObserver(entries => {
        entries.getEntries().forEach(entry => {
          if (entry.duration > 50) {
            console.warn(`Long task detected: ${entry.duration}ms`);
            this.optimizeAfterLongTask();
          }
        });
      });
      longTaskObserver.observe({ entryTypes: ['longtask'] });
      this.observers.set('longTasks', longTaskObserver);
    } catch (e) {
      console.warn('Long task monitoring not supported');
    }

    // Monitor Layout Shifts
    try {
      const layoutShiftObserver = new PerformanceObserver(entries => {
        entries.getEntries().forEach(entry => {
          if (entry.value > 0.1) {
            console.warn(`Significant layout shift: ${entry.value}`);
          }
        });
      });
      layoutShiftObserver.observe({ entryTypes: ['layout-shift'] });
      this.observers.set('layoutShift', layoutShiftObserver);
    } catch (e) {
      console.warn('Layout shift monitoring not supported');
    }

    // Performance metrics collection
    setInterval(() => {
      this.collectPerformanceMetrics();
    }, this.options.performanceMetricsInterval);
  }

  collectPerformanceMetrics() {
    const metrics = {
      timestamp: Date.now(),
      memory: this.getMemoryUsage(),
      navigation: this.getNavigationTiming(),
      resources: this.getResourceTiming(),
    };

    this.performanceMetrics.memoryUsage.push(metrics.memory);
    this.performanceMetrics.networkRequests.push(...metrics.resources);

    // Keep only recent metrics
    const maxEntries = 100;
    Object.keys(this.performanceMetrics).forEach(key => {
      if (this.performanceMetrics[key].length > maxEntries) {
        this.performanceMetrics[key] = this.performanceMetrics[key].slice(-maxEntries);
      }
    });

    this.analyzePerformance(metrics);
  }

  getMemoryUsage() {
    if ('memory' in performance) {
      return {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit,
        ratio: performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit
      };
    }
    return null;
  }

  getNavigationTiming() {
    const navigation = performance.getEntriesByType('navigation')[0];
    if (!navigation) return null;

    return {
      dns: navigation.domainLookupEnd - navigation.domainLookupStart,
      connect: navigation.connectEnd - navigation.connectStart,
      request: navigation.responseStart - navigation.requestStart,
      response: navigation.responseEnd - navigation.responseStart,
      domLoad: navigation.domContentLoadedEventEnd - navigation.navigationStart,
      pageLoad: navigation.loadEventEnd - navigation.navigationStart,
    };
  }

  getResourceTiming() {
    return performance.getEntriesByType('resource')
      .filter(entry => entry.duration > 100) // Only slow resources
      .map(entry => ({
        name: entry.name,
        duration: entry.duration,
        size: entry.transferSize || 0,
        type: entry.initiatorType,
      }));
  }

  analyzePerformance(metrics) {
    // Memory pressure detection
    if (metrics.memory && metrics.memory.ratio > this.options.memoryThreshold) {
      this.handleMemoryPressure();
    }

    // Slow resource detection
    if (metrics.resources.length > 0) {
      const avgDuration = metrics.resources.reduce((sum, r) => sum + r.duration, 0) / metrics.resources.length;
      if (avgDuration > 1000) {
        console.warn('Slow network resources detected, optimizing...');
        this.optimizeNetworkRequests();
      }
    }
  }

  // === MEMORY MANAGEMENT ===

  setupMemoryManagement() {
    // Cleanup intervals
    setInterval(() => {
      this.cleanupUnusedResources();
    }, 30000); // Every 30 seconds

    // Page visibility handling
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.pauseNonEssentialOperations();
      } else {
        this.resumeOperations();
      }
    });
  }

  handleMemoryPressure() {
    console.warn('Memory pressure detected, optimizing...');

    // Clear caches
    this.resourceCache.clear();

    // Reduce data retention
    Object.keys(this.performanceMetrics).forEach(key => {
      this.performanceMetrics[key] = this.performanceMetrics[key].slice(-20);
    });

    // Request garbage collection if available
    if (window.gc) {
      window.gc();
    }

    // Emit memory pressure event
    document.dispatchEvent(new CustomEvent('memoryPressure', {
      detail: { severity: 'high' }
    }));
  }

  cleanupUnusedResources() {
    // Remove unused cached resources
    const now = Date.now();
    for (const [key, resource] of this.resourceCache.entries()) {
      if (now - resource.lastAccessed > 300000) { // 5 minutes
        this.resourceCache.delete(key);
      }
    }

    // Clean up old performance entries
    if (performance.clearResourceTimings) {
      performance.clearResourceTimings();
    }
  }

  pauseNonEssentialOperations() {
    // Pause animations
    document.querySelectorAll('.animated, .chart-container').forEach(el => {
      el.style.animationPlayState = 'paused';
    });

    // Pause timers for non-critical components
    document.dispatchEvent(new CustomEvent('pauseOperations'));
  }

  resumeOperations() {
    // Resume animations
    document.querySelectorAll('.animated, .chart-container').forEach(el => {
      el.style.animationPlayState = 'running';
    });

    // Resume timers
    document.dispatchEvent(new CustomEvent('resumeOperations'));
  }

  // === ANIMATION OPTIMIZATION ===

  optimizeAnimations() {
    // Use CSS transforms for better performance
    const animationElements = document.querySelectorAll('[data-animate]');
    animationElements.forEach(el => {
      el.style.willChange = 'transform, opacity';
      el.style.transform = 'translateZ(0)'; // Force hardware acceleration
    });

    // Reduce animations on low-end devices
    if (this.isLowEndDevice()) {
      document.documentElement.classList.add('reduce-motion');
    }
  }

  isLowEndDevice() {
    // Heuristics for low-end device detection
    const memory = navigator.deviceMemory || 4;
    const cores = navigator.hardwareConcurrency || 4;

    return memory < 4 || cores < 4;
  }

  optimizeAfterLongTask() {
    // Break up long tasks
    this.scheduleIdleWork(() => {
      // Perform cleanup or optimization work
      this.cleanupUnusedResources();
    });
  }

  scheduleIdleWork(callback) {
    if ('requestIdleCallback' in window) {
      requestIdleCallback(callback, { timeout: 5000 });
    } else {
      setTimeout(callback, 1);
    }
  }

  // === NETWORK OPTIMIZATION ===

  setupNetworkOptimization() {
    // Resource preloading
    this.preloadCriticalResources();

    // Request coalescing
    this.requestQueue = [];
    this.requestProcessor = this.debounce(() => {
      this.processBatchedRequests();
    }, 'batchRequests', 100);
  }

  preloadCriticalResources() {
    const criticalResources = [
      '/static/css/design-system.css',
      '/static/js/charts/anomaly-timeline.js',
      '/static/js/charts/anomaly-heatmap.js'
    ];

    criticalResources.forEach(resource => {
      const link = document.createElement('link');
      link.rel = 'preload';
      link.href = resource;
      link.as = resource.endsWith('.css') ? 'style' : 'script';
      document.head.appendChild(link);
    });
  }

  optimizeNetworkRequests() {
    // Implement request deduplication
    const pendingRequests = new Map();

    const originalFetch = window.fetch;
    window.fetch = async (url, options = {}) => {
      const key = `${options.method || 'GET'}:${url}`;

      if (pendingRequests.has(key)) {
        return pendingRequests.get(key);
      }

      const promise = originalFetch(url, options);
      pendingRequests.set(key, promise);

      try {
        const response = await promise;
        pendingRequests.delete(key);
        return response;
      } catch (error) {
        pendingRequests.delete(key);
        throw error;
      }
    };
  }

  batchRequest(request) {
    this.requestQueue.push(request);
    this.requestProcessor();
  }

  async processBatchedRequests() {
    if (this.requestQueue.length === 0) return;

    const requests = [...this.requestQueue];
    this.requestQueue = [];

    // Group similar requests
    const grouped = new Map();
    requests.forEach(req => {
      const key = req.endpoint;
      if (!grouped.has(key)) {
        grouped.set(key, []);
      }
      grouped.get(key).push(req);
    });

    // Process grouped requests
    for (const [endpoint, reqs] of grouped.entries()) {
      try {
        await this.processBatchedEndpoint(endpoint, reqs);
      } catch (error) {
        console.error(`Batch request failed for ${endpoint}:`, error);
      }
    }
  }

  async processBatchedEndpoint(endpoint, requests) {
    // Implementation depends on API design
    const ids = requests.map(req => req.id);
    const response = await fetch(`${endpoint}/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids })
    });

    const results = await response.json();

    // Distribute results back to original requesters
    requests.forEach((req, index) => {
      req.resolve(results[index]);
    });
  }

  // === WEB WORKERS ===

  initializeWorkers() {
    if ('Worker' in window) {
      this.workers = {
        dataProcessor: this.createDataProcessorWorker(),
        chartRenderer: this.createChartRendererWorker()
      };
    }
  }

  createDataProcessorWorker() {
    const workerScript = `
      self.onmessage = function(e) {
        const { type, data } = e.data;

        switch(type) {
          case 'processAnomalies':
            const processed = processAnomalies(data);
            self.postMessage({ type: 'anomaliesProcessed', result: processed });
            break;

          case 'aggregateMetrics':
            const aggregated = aggregateMetrics(data);
            self.postMessage({ type: 'metricsAggregated', result: aggregated });
            break;
        }
      };

      function processAnomalies(anomalies) {
        return anomalies.map(anomaly => ({
          ...anomaly,
          severity: anomaly.anomalyScore > 0.8 ? 'high' :
                   anomaly.anomalyScore > 0.5 ? 'medium' : 'low',
          timestamp: new Date(anomaly.timestamp).getTime()
        }));
      }

      function aggregateMetrics(data) {
        const aggregated = {};
        data.forEach(item => {
          const key = new Date(item.timestamp).toISOString().slice(0, 13);
          if (!aggregated[key]) {
            aggregated[key] = { count: 0, totalScore: 0, maxScore: 0 };
          }
          aggregated[key].count++;
          aggregated[key].totalScore += item.anomalyScore;
          aggregated[key].maxScore = Math.max(aggregated[key].maxScore, item.anomalyScore);
        });
        return aggregated;
      }
    `;

    const blob = new Blob([workerScript], { type: 'application/javascript' });
    return new Worker(URL.createObjectURL(blob));
  }

  createChartRendererWorker() {
    // Similar worker for chart data processing
    const workerScript = `
      self.onmessage = function(e) {
        const { type, data } = e.data;

        if (type === 'generateChartData') {
          const chartData = generateChartData(data);
          self.postMessage({ type: 'chartDataReady', result: chartData });
        }
      };

      function generateChartData(rawData) {
        return rawData.map((point, index) => ({
          x: index,
          y: point.value,
          anomalyScore: point.anomalyScore,
          color: point.anomalyScore > 0.7 ? 'red' : 'blue'
        }));
      }
    `;

    const blob = new Blob([workerScript], { type: 'application/javascript' });
    return new Worker(URL.createObjectURL(blob));
  }

  // === PUBLIC API ===

  async processDataInWorker(type, data) {
    return new Promise((resolve) => {
      const worker = this.workers.dataProcessor;

      const messageHandler = (e) => {
        worker.removeEventListener('message', messageHandler);
        resolve(e.data.result);
      };

      worker.addEventListener('message', messageHandler);
      worker.postMessage({ type, data });
    });
  }

  enableVirtualScrolling(container, items, renderItem, itemHeight) {
    if (items.length > this.options.virtualScrollThreshold) {
      return this.createVirtualScrollContainer(container, items, renderItem, itemHeight);
    }
    return null;
  }

  optimizeComponent(component) {
    // Apply optimization strategies to a component
    this.enableLazyLoading();

    if (component.container) {
      component.container.style.willChange = 'transform';
      component.container.style.contain = 'layout style paint';
    }

    return component;
  }

  getPerformanceReport() {
    return {
      metrics: this.performanceMetrics,
      cacheSize: this.resourceCache.size,
      activeObservers: this.observers.size,
      debouncedFunctions: this.debouncedFunctions.size,
      throttledFunctions: this.throttledFunctions.size,
      memory: this.getMemoryUsage(),
    };
  }

  destroy() {
    // Clean up observers
    this.observers.forEach(observer => observer.disconnect());
    this.observers.clear();

    // Clear timers
    this.debouncedFunctions.forEach(clearTimeout);
    this.debouncedFunctions.clear();
    this.throttledFunctions.clear();

    // Terminate workers
    if (this.workers) {
      Object.values(this.workers).forEach(worker => worker.terminate());
    }

    // Clear caches
    this.resourceCache.clear();
  }
}

// Export singleton instance
export const performanceOptimizer = new PerformanceOptimizer();
export default PerformanceOptimizer;
