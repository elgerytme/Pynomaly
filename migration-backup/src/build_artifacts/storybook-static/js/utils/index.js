/**
 * Utilities Module Entry Point - Performance Optimized Utilities
 * Provides shared utilities with lazy loading and intelligent caching
 */

/**
 * Performance monitoring utilities
 */
export class PerformanceMonitor {
  constructor() {
    this.metrics = new Map();
    this.observers = new Map();
    this.thresholds = {
      loadTime: 3000, // 3 seconds
      renderTime: 100, // 100ms
      memoryUsage: 50 * 1024 * 1024 // 50MB
    };
  }

  /**
   * Measure function execution time
   */
  measureExecution(fn, name) {
    return (...args) => {
      const start = performance.now();
      const result = fn.apply(this, args);
      const duration = performance.now() - start;

      this.recordMetric('execution', name, duration);

      if (duration > this.thresholds.renderTime) {
        console.warn(`⚠️ Slow execution detected: ${name} took ${duration.toFixed(2)}ms`);
      }

      return result;
    };
  }

  /**
   * Measure async function execution time
   */
  measureAsyncExecution(fn, name) {
    return async (...args) => {
      const start = performance.now();
      const result = await fn.apply(this, args);
      const duration = performance.now() - start;

      this.recordMetric('async-execution', name, duration);

      if (duration > this.thresholds.loadTime) {
        console.warn(`⚠️ Slow async operation: ${name} took ${duration.toFixed(2)}ms`);
      }

      return result;
    };
  }

  /**
   * Monitor Web Vitals
   */
  monitorWebVitals() {
    if ('PerformanceObserver' in window) {
      // Largest Contentful Paint
      const lcpObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          this.recordMetric('web-vitals', 'lcp', entry.startTime);
        });
      });
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

      // First Input Delay
      const fidObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          this.recordMetric('web-vitals', 'fid', entry.processingStart - entry.startTime);
        });
      });
      fidObserver.observe({ entryTypes: ['first-input'] });

      // Cumulative Layout Shift
      let clsScore = 0;
      const clsObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (!entry.hadRecentInput) {
            clsScore += entry.value;
          }
        });
        this.recordMetric('web-vitals', 'cls', clsScore);
      });
      clsObserver.observe({ entryTypes: ['layout-shift'] });
    }
  }

  /**
   * Monitor memory usage
   */
  monitorMemoryUsage() {
    if ('memory' in performance) {
      const memInfo = performance.memory;
      this.recordMetric('memory', 'used', memInfo.usedJSHeapSize);
      this.recordMetric('memory', 'total', memInfo.totalJSHeapSize);

      if (memInfo.usedJSHeapSize > this.thresholds.memoryUsage) {
        console.warn(`⚠️ High memory usage: ${(memInfo.usedJSHeapSize / 1024 / 1024).toFixed(2)}MB`);
      }
    }
  }

  recordMetric(category, name, value) {
    const key = `${category}.${name}`;
    if (!this.metrics.has(key)) {
      this.metrics.set(key, []);
    }

    this.metrics.get(key).push({
      value,
      timestamp: Date.now()
    });

    // Keep only last 100 measurements
    const measurements = this.metrics.get(key);
    if (measurements.length > 100) {
      this.metrics.set(key, measurements.slice(-100));
    }
  }

  getMetrics() {
    const summary = {};
    this.metrics.forEach((measurements, key) => {
      const values = measurements.map(m => m.value);
      summary[key] = {
        count: values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        avg: values.reduce((a, b) => a + b, 0) / values.length,
        latest: values[values.length - 1]
      };
    });
    return summary;
  }
}

/**
 * Intelligent data loader with caching and batching
 */
export class DataLoader {
  constructor() {
    this.cache = new Map();
    this.loadingQueue = new Map();
    this.batchQueue = [];
    this.batchTimeout = null;
  }

  /**
   * Load data with intelligent caching
   */
  async loadData(url, options = {}) {
    const cacheKey = this._getCacheKey(url, options);

    // Check cache first
    if (this.cache.has(cacheKey) && !options.refresh) {
      const cached = this.cache.get(cacheKey);
      if (Date.now() - cached.timestamp < (options.maxAge || 300000)) { // 5 min default
        return cached.data;
      }
    }

    // Check if already loading
    if (this.loadingQueue.has(cacheKey)) {
      return this.loadingQueue.get(cacheKey);
    }

    // Create loading promise
    const loadPromise = this._fetchData(url, options)
      .then(data => {
        this.cache.set(cacheKey, {
          data,
          timestamp: Date.now()
        });
        return data;
      })
      .finally(() => {
        this.loadingQueue.delete(cacheKey);
      });

    this.loadingQueue.set(cacheKey, loadPromise);
    return loadPromise;
  }

  /**
   * Batch multiple requests together
   */
  batchLoad(requests) {
    return new Promise((resolve) => {
      this.batchQueue.push({ requests, resolve });

      if (this.batchTimeout) {
        clearTimeout(this.batchTimeout);
      }

      this.batchTimeout = setTimeout(() => {
        this._processBatchQueue();
      }, 50); // 50ms batch window
    });
  }

  async _fetchData(url, options) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout || 10000);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  _processBatchQueue() {
    const batch = this.batchQueue.splice(0);

    // Group requests by domain for efficient batching
    const grouped = new Map();
    batch.forEach(({ requests, resolve }) => {
      requests.forEach(req => {
        const domain = new URL(req.url).origin;
        if (!grouped.has(domain)) {
          grouped.set(domain, []);
        }
        grouped.get(domain).push({ ...req, resolve });
      });
    });

    // Process each domain group
    grouped.forEach(async (requests, domain) => {
      const results = await Promise.allSettled(
        requests.map(req => this.loadData(req.url, req.options))
      );

      requests.forEach((req, index) => {
        req.resolve(results[index]);
      });
    });
  }

  _getCacheKey(url, options) {
    return `${url}_${JSON.stringify(options)}`;
  }

  clearCache() {
    this.cache.clear();
  }
}

/**
 * Advanced visualization loader with library-specific optimizations
 */
export class VisualizationLoader {
  constructor() {
    this.libraryLoaders = new Map();
    this.loadedLibraries = new Set();
  }

  /**
   * Load D3.js with optimized modules
   */
  async loadD3(modules = ['d3-selection', 'd3-scale', 'd3-axis']) {
    if (this.loadedLibraries.has('d3')) {
      return window.d3;
    }

    const d3Modules = await Promise.all(
      modules.map(module =>
        import(/* webpackChunkName: "d3-modules" */ `d3/${module}`)
      )
    );

    // Combine D3 modules
    const d3 = Object.assign({}, ...d3Modules);
    window.d3 = d3;
    this.loadedLibraries.add('d3');

    return d3;
  }

  /**
   * Load ECharts with tree shaking
   */
  async loadECharts(charts = ['line', 'bar', 'scatter']) {
    if (this.loadedLibraries.has('echarts')) {
      return window.echarts;
    }

    const { init, use } = await import(
      /* webpackChunkName: "echarts-core" */
      'echarts/core'
    );

    const rendererModule = await import(
      /* webpackChunkName: "echarts-renderer" */
      'echarts/renderers'
    );

    const chartModules = await Promise.all(
      charts.map(chart =>
        import(/* webpackChunkName: "echarts-charts" */ `echarts/charts/${chart}Chart`)
      )
    );

    const componentModules = await Promise.all([
      import(/* webpackChunkName: "echarts-components" */ 'echarts/components/GridComponent'),
      import(/* webpackChunkName: "echarts-components" */ 'echarts/components/TooltipComponent'),
      import(/* webpackChunkName: "echarts-components" */ 'echarts/components/LegendComponent')
    ]);

    // Register components with ECharts
    use([
      rendererModule.CanvasRenderer,
      ...chartModules.map(m => Object.values(m)[0]),
      ...componentModules.map(m => Object.values(m)[0])
    ]);

    const echarts = { init };
    window.echarts = echarts;
    this.loadedLibraries.add('echarts');

    return echarts;
  }

  /**
   * Load Plotly.js with selective imports
   */
  async loadPlotly(plotTypes = ['scatter', 'scatter3d']) {
    if (this.loadedLibraries.has('plotly')) {
      return window.Plotly;
    }

    const plotlyModules = await Promise.all([
      import(/* webpackChunkName: "plotly-core" */ 'plotly.js/lib/core'),
      ...plotTypes.map(type =>
        import(/* webpackChunkName: "plotly-traces" */ `plotly.js/lib/${type}`)
      )
    ]);

    const [Plotly, ...traces] = plotlyModules;

    // Register trace types
    traces.forEach(trace => {
      Plotly.register(trace.default || trace);
    });

    window.Plotly = Plotly;
    this.loadedLibraries.add('plotly');

    return Plotly;
  }

  /**
   * Intelligent library loading based on chart requirements
   */
  async loadLibraryForChart(chartType, config = {}) {
    const libraryMap = {
      'timeline': () => this.loadD3(['d3-selection', 'd3-scale', 'd3-axis', 'd3-time']),
      'heatmap': () => this.loadD3(['d3-selection', 'd3-scale', 'd3-color']),
      'realtime': () => this.loadECharts(['line', 'bar']),
      'scatter3d': () => this.loadPlotly(['scatter3d']),
      'correlation': () => this.loadD3(['d3-selection', 'd3-scale', 'd3-color']),
      'distribution': () => this.loadECharts(['line', 'bar', 'boxplot'])
    };

    const loader = libraryMap[chartType];
    if (!loader) {
      throw new Error(`No library loader found for chart type: ${chartType}`);
    }

    return loader();
  }

  /**
   * Preload commonly used visualization libraries
   */
  async preloadCommonLibraries() {
    const preloadTasks = [
      this.loadD3(['d3-selection', 'd3-scale']),
      this.loadECharts(['line', 'bar'])
    ];

    try {
      await Promise.all(preloadTasks);
      console.log('✅ Common visualization libraries preloaded');
    } catch (error) {
      console.warn('⚠️ Failed to preload some visualization libraries:', error);
    }
  }
}

/**
 * Image optimization utilities
 */
export class ImageOptimizer {
  constructor() {
    this.cache = new Map();
    this.observers = new Map();
  }

  /**
   * Lazy load images with intersection observer
   */
  setupLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');

    if (!images.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          this.loadImage(img);
          observer.unobserve(img);
        }
      });
    }, {
      rootMargin: '50px 0px'
    });

    images.forEach(img => observer.observe(img));
  }

  /**
   * Load image with fallback and optimization
   */
  async loadImage(img) {
    const src = img.dataset.src;
    const fallback = img.dataset.fallback;

    try {
      // Check if WebP is supported
      const format = await this.getSupportedFormat();
      const optimizedSrc = this.getOptimizedUrl(src, format);

      await this.preloadImage(optimizedSrc);
      img.src = optimizedSrc;
      img.classList.add('loaded');
    } catch (error) {
      if (fallback) {
        img.src = fallback;
      }
      img.classList.add('error');
    }
  }

  async getSupportedFormat() {
    if (this.cache.has('supportedFormat')) {
      return this.cache.get('supportedFormat');
    }

    const formats = ['webp', 'avif'];
    for (const format of formats) {
      if (await this.canUseFormat(format)) {
        this.cache.set('supportedFormat', format);
        return format;
      }
    }

    this.cache.set('supportedFormat', 'original');
    return 'original';
  }

  canUseFormat(format) {
    return new Promise((resolve) => {
      const testImages = {
        webp: 'data:image/webp;base64,UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==',
        avif: 'data:image/avif;base64,AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADybWV0YQAAAAAAAAAoaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAGxpYmF2aWYAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABGgAAAB0AAAAoaWluZgAAAAAAAQAAABppbmZlAgAAAAABAABhdjAxQ29sb3IAAAAAamlwcnAAAABLaXBjbwAAABRpc3BlAAAAAAAAAAIAAAACAAAAEHBpeGkAAAAAAwgICAAAAAxhdjFDgQ0MAAAAABNjb2xybmNseAACAAIAAYAAAAAXaXBtYQAAAAAAAAABAAEEAQKDBAAAACVtZGF0EgAKCBgABogQEAwgMg8f8D///8WfhwB8+ErK42A='
      };

      const img = new Image();
      img.onload = () => resolve(true);
      img.onerror = () => resolve(false);
      img.src = testImages[format];
    });
  }

  getOptimizedUrl(src, format) {
    if (format === 'original') return src;

    // Assume your backend supports format conversion
    const url = new URL(src, window.location.origin);
    url.searchParams.set('format', format);
    return url.toString();
  }

  preloadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = resolve;
      img.onerror = reject;
      img.src = src;
    });
  }
}

// Export instances for global use
export const performanceMonitor = new PerformanceMonitor();
export const dataLoader = new DataLoader();
export const visualizationLoader = new VisualizationLoader();
export const imageOptimizer = new ImageOptimizer();

/**
 * Initialize performance utilities
 */
export function initializePerformanceUtilities() {
  performanceMonitor.monitorWebVitals();

  // Monitor memory usage every 30 seconds
  setInterval(() => {
    performanceMonitor.monitorMemoryUsage();
  }, 30000);

  // Setup image optimization
  imageOptimizer.setupLazyLoading();

  // Preload common visualization libraries on idle
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      visualizationLoader.preloadCommonLibraries();
    });
  }

  // Log performance summary every 2 minutes
  setInterval(() => {
    const metrics = performanceMonitor.getMetrics();
    console.log('📊 Performance Summary:', metrics);
  }, 120000);
}

// Auto-initialize utilities
if (typeof window !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePerformanceUtilities);
  } else {
    initializePerformanceUtilities();
  }
}
