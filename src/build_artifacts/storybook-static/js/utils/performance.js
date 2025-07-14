/**
 * Performance Utilities for Pynomaly Web UI
 * Provides performance monitoring, optimization helpers, and lazy loading utilities
 */

/**
 * Lazy Loading Utilities
 */
export class LazyLoader {
  constructor() {
    this.observers = new Map();
    this.loadingPromises = new Map();
  }

  /**
   * Lazy load component when it becomes visible
   */
  observeComponent(selector, moduleLoader, options = {}) {
    const {
      threshold = 0.1,
      rootMargin = '50px',
      once = true
    } = options;

    const elements = document.querySelectorAll(selector);
    if (elements.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(async (entry) => {
          if (entry.isIntersecting) {
            if (once) {
              observer.unobserve(entry.target);
            }

            try {
              const module = await this.loadModule(moduleLoader);
              if (module.default && typeof module.default === 'function') {
                module.default(entry.target);
              }
            } catch (error) {
              console.error('Failed to lazy load component:', error);
            }
          }
        });
      },
      { threshold, rootMargin }
    );

    elements.forEach(element => observer.observe(element));
    this.observers.set(selector, observer);
  }

  /**
   * Load module with caching and deduplication
   */
  async loadModule(moduleLoader) {
    const moduleId = moduleLoader.toString();

    if (this.loadingPromises.has(moduleId)) {
      return this.loadingPromises.get(moduleId);
    }

    const promise = moduleLoader();
    this.loadingPromises.set(moduleId, promise);

    try {
      const module = await promise;
      return module;
    } catch (error) {
      this.loadingPromises.delete(moduleId);
      throw error;
    }
  }

  /**
   * Preload modules for better performance
   */
  preload(moduleLoaders) {
    moduleLoaders.forEach(loader => {
      this.loadModule(loader).catch(console.warn);
    });
  }

  /**
   * Clean up observers
   */
  destroy() {
    this.observers.forEach(observer => observer.disconnect());
    this.observers.clear();
    this.loadingPromises.clear();
  }
}

/**
 * Performance Monitor Utilities
 */
export class PerformanceUtils {
  /**
   * Measure function execution time
   */
  static time(name, fn) {
    const startTime = performance.now();
    const result = fn();
    const duration = performance.now() - startTime;

    console.log(`⏱️ ${name}: ${duration.toFixed(2)}ms`);

    if (window.performanceMonitor) {
      window.performanceMonitor.recordMetric(name, duration, {
        type: 'custom-timing'
      });
    }

    return result;
  }

  /**
   * Measure async function execution time
   */
  static async timeAsync(name, fn) {
    const startTime = performance.now();
    const result = await fn();
    const duration = performance.now() - startTime;

    console.log(`⏱️ ${name}: ${duration.toFixed(2)}ms`);

    if (window.performanceMonitor) {
      window.performanceMonitor.recordMetric(name, duration, {
        type: 'custom-timing-async'
      });
    }

    return result;
  }

  /**
   * Debounce function calls
   */
  static debounce(func, wait, immediate = false) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        timeout = null;
        if (!immediate) func(...args);
      };
      const callNow = immediate && !timeout;
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
      if (callNow) func(...args);
    };
  }

  /**
   * Throttle function calls
   */
  static throttle(func, limit) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }

  /**
   * RAF-based throttling for smooth animations
   */
  static rafThrottle(func) {
    let ticking = false;
    return function(...args) {
      if (!ticking) {
        requestAnimationFrame(() => {
          func.apply(this, args);
          ticking = false;
        });
        ticking = true;
      }
    };
  }

  /**
   * Idle callback wrapper
   */
  static onIdle(callback, options = {}) {
    if ('requestIdleCallback' in window) {
      return requestIdleCallback(callback, options);
    } else {
      // Fallback for browsers without requestIdleCallback
      return setTimeout(callback, 1);
    }
  }

  /**
   * Check if user prefers reduced motion
   */
  static prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /**
   * Get connection type and speed
   */
  static getConnectionInfo() {
    if ('connection' in navigator) {
      const connection = navigator.connection;
      return {
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt,
        saveData: connection.saveData
      };
    }
    return null;
  }

  /**
   * Check if device has limited resources
   */
  static isLowEndDevice() {
    const connection = this.getConnectionInfo();
    const memory = navigator.deviceMemory;
    const cores = navigator.hardwareConcurrency;

    return (
      (connection && connection.saveData) ||
      (memory && memory < 4) ||
      (cores && cores < 4) ||
      (connection && ['slow-2g', '2g'].includes(connection.effectiveType))
    );
  }
}

/**
 * Critical Resource Loader
 */
export class CriticalResourceLoader {
  constructor() {
    this.loadedResources = new Set();
    this.loadingPromises = new Map();
  }

  /**
   * Preload critical CSS
   */
  preloadCSS(href) {
    if (this.loadedResources.has(href)) return Promise.resolve();

    if (this.loadingPromises.has(href)) {
      return this.loadingPromises.get(href);
    }

    const promise = new Promise((resolve, reject) => {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = href;
      link.onload = () => {
        this.loadedResources.add(href);
        resolve();
      };
      link.onerror = reject;

      // Insert before first stylesheet or in head
      const firstStylesheet = document.querySelector('link[rel="stylesheet"]');
      if (firstStylesheet) {
        firstStylesheet.parentNode.insertBefore(link, firstStylesheet);
      } else {
        document.head.appendChild(link);
      }
    });

    this.loadingPromises.set(href, promise);
    return promise;
  }

  /**
   * Preload JavaScript module
   */
  preloadJS(src) {
    if (this.loadedResources.has(src)) return Promise.resolve();

    if (this.loadingPromises.has(src)) {
      return this.loadingPromises.get(src);
    }

    const promise = new Promise((resolve, reject) => {
      const link = document.createElement('link');
      link.rel = 'modulepreload';
      link.href = src;
      link.onload = () => {
        this.loadedResources.add(src);
        resolve();
      };
      link.onerror = reject;
      document.head.appendChild(link);
    });

    this.loadingPromises.set(src, promise);
    return promise;
  }

  /**
   * Preload image
   */
  preloadImage(src) {
    if (this.loadedResources.has(src)) return Promise.resolve();

    if (this.loadingPromises.has(src)) {
      return this.loadingPromises.get(src);
    }

    const promise = new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.loadedResources.add(src);
        resolve();
      };
      img.onerror = reject;
      img.src = src;
    });

    this.loadingPromises.set(src, promise);
    return promise;
  }
}

/**
 * Bundle Size Monitor
 */
export class BundleSizeMonitor {
  constructor() {
    this.sizes = new Map();
    this.monitor();
  }

  monitor() {
    // Monitor script tags
    const scripts = document.querySelectorAll('script[src]');
    scripts.forEach(script => this.measureScript(script.src));

    // Monitor stylesheets
    const stylesheets = document.querySelectorAll('link[rel="stylesheet"]');
    stylesheets.forEach(link => this.measureStylesheet(link.href));
  }

  async measureScript(src) {
    try {
      const response = await fetch(src, { method: 'HEAD' });
      const size = parseInt(response.headers.get('content-length') || '0');

      this.sizes.set(src, {
        type: 'script',
        size,
        compressed: response.headers.get('content-encoding') !== null
      });

      console.log(`📦 Script: ${src} - ${(size / 1024).toFixed(1)}KB`);
    } catch (error) {
      console.warn('Failed to measure script size:', src);
    }
  }

  async measureStylesheet(href) {
    try {
      const response = await fetch(href, { method: 'HEAD' });
      const size = parseInt(response.headers.get('content-length') || '0');

      this.sizes.set(href, {
        type: 'stylesheet',
        size,
        compressed: response.headers.get('content-encoding') !== null
      });

      console.log(`🎨 Stylesheet: ${href} - ${(size / 1024).toFixed(1)}KB`);
    } catch (error) {
      console.warn('Failed to measure stylesheet size:', href);
    }
  }

  getTotalSize() {
    let total = 0;
    this.sizes.forEach(({ size }) => total += size);
    return total;
  }

  getReport() {
    const report = {
      total: this.getTotalSize(),
      scripts: 0,
      stylesheets: 0,
      resources: []
    };

    this.sizes.forEach((data, url) => {
      if (data.type === 'script') {
        report.scripts += data.size;
      } else if (data.type === 'stylesheet') {
        report.stylesheets += data.size;
      }

      report.resources.push({
        url,
        ...data,
        sizeKB: (data.size / 1024).toFixed(1)
      });
    });

    return report;
  }
}

/**
 * Performance Budget Checker
 */
export class PerformanceBudget {
  constructor(budgets = {}) {
    this.budgets = {
      totalJS: 500 * 1024,        // 500KB
      totalCSS: 50 * 1024,        // 50KB
      totalImages: 1000 * 1024,   // 1MB
      totalFonts: 100 * 1024,     // 100KB
      ...budgets
    };
  }

  check() {
    const violations = [];
    const monitor = new BundleSizeMonitor();

    setTimeout(() => {
      const report = monitor.getReport();

      if (report.scripts > this.budgets.totalJS) {
        violations.push({
          type: 'JavaScript',
          actual: report.scripts,
          budget: this.budgets.totalJS,
          overage: report.scripts - this.budgets.totalJS
        });
      }

      if (report.stylesheets > this.budgets.totalCSS) {
        violations.push({
          type: 'CSS',
          actual: report.stylesheets,
          budget: this.budgets.totalCSS,
          overage: report.stylesheets - this.budgets.totalCSS
        });
      }

      if (violations.length > 0) {
        console.warn('⚠️ Performance budget violations:', violations);
        this.reportViolations(violations);
      } else {
        console.log('✅ Performance budgets met');
      }
    }, 2000); // Wait for resources to load
  }

  reportViolations(violations) {
    violations.forEach(violation => {
      if (window.gtag) {
        window.gtag('event', 'performance_budget_violation', {
          resource_type: violation.type,
          actual_size: Math.round(violation.actual / 1024),
          budget_size: Math.round(violation.budget / 1024),
          overage_kb: Math.round(violation.overage / 1024)
        });
      }
    });
  }
}

// Initialize global instances
export const lazyLoader = new LazyLoader();
export const criticalResourceLoader = new CriticalResourceLoader();

// Auto-check performance budget on load
document.addEventListener('DOMContentLoaded', () => {
  const budget = new PerformanceBudget();
  budget.check();
});

// Export utilities for global use
window.PerformanceUtils = PerformanceUtils;
