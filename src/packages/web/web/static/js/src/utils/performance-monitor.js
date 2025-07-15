/**
 * Performance Monitor for Web UI
 * Tracks Core Web Vitals and custom metrics
 */

class PerformanceMonitor {
  constructor() {
    this.metrics = new Map();
    this.observers = new Map();
    this.initObservers();
  }

  initObservers() {
    // Performance Observer for Core Web Vitals
    if ('PerformanceObserver' in window) {
      this.initCoreWebVitals();
      this.initResourceTiming();
      this.initNavigationTiming();
    }

    // Custom metrics tracking
    this.trackCustomMetrics();
  }

  initCoreWebVitals() {
    // Largest Contentful Paint (LCP)
    const lcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      this.recordMetric('LCP', lastEntry.startTime);
    });

    try {
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
      this.observers.set('lcp', lcpObserver);
    } catch (e) {
      console.warn('LCP observer not supported');
    }

    // First Input Delay (FID)
    const fidObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach(entry => {
        this.recordMetric('FID', entry.processingStart - entry.startTime);
      });
    });

    try {
      fidObserver.observe({ entryTypes: ['first-input'] });
      this.observers.set('fid', fidObserver);
    } catch (e) {
      console.warn('FID observer not supported');
    }

    // Cumulative Layout Shift (CLS)
    let clsValue = 0;
    const clsObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach(entry => {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
        }
      });
      this.recordMetric('CLS', clsValue);
    });

    try {
      clsObserver.observe({ entryTypes: ['layout-shift'] });
      this.observers.set('cls', clsObserver);
    } catch (e) {
      console.warn('CLS observer not supported');
    }

    // First Contentful Paint (FCP)
    const fcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const fcpEntry = entries.find(entry => entry.name === 'first-contentful-paint');
      if (fcpEntry) {
        this.recordMetric('FCP', fcpEntry.startTime);
      }
    });

    try {
      fcpObserver.observe({ entryTypes: ['paint'] });
      this.observers.set('fcp', fcpObserver);
    } catch (e) {
      console.warn('FCP observer not supported');
    }
  }

  initResourceTiming() {
    const resourceObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach(entry => {
        this.analyzeResourceTiming(entry);
      });
    });

    try {
      resourceObserver.observe({ entryTypes: ['resource'] });
      this.observers.set('resource', resourceObserver);
    } catch (e) {
      console.warn('Resource timing observer not supported');
    }
  }

  initNavigationTiming() {
    const navigationObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach(entry => {
        this.analyzeNavigationTiming(entry);
      });
    });

    try {
      navigationObserver.observe({ entryTypes: ['navigation'] });
      this.observers.set('navigation', navigationObserver);
    } catch (e) {
      console.warn('Navigation timing observer not supported');
    }
  }

  recordMetric(name, value) {
    const timestamp = Date.now();

    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }

    this.metrics.get(name).push({
      value,
      timestamp
    });

    // Keep only last 100 entries per metric
    const entries = this.metrics.get(name);
    if (entries.length > 100) {
      entries.shift();
    }

    // Emit performance event
    this.emitPerformanceEvent(name, value);
  }

  emitPerformanceEvent(name, value) {
    const event = new CustomEvent('performance-metric', {
      detail: { name, value, timestamp: Date.now() }
    });

    document.dispatchEvent(event);

    // Log critical issues
    if (this.isCriticalMetric(name, value)) {
      console.warn(`Performance issue detected: ${name} = ${value}`);
    }
  }

  isCriticalMetric(name, value) {
    const thresholds = {
      'LCP': 2500, // 2.5 seconds
      'FID': 100,  // 100ms
      'CLS': 0.1,  // 0.1
      'FCP': 1800, // 1.8 seconds
      'TTFB': 600  // 600ms
    };

    return thresholds[name] && value > thresholds[name];
  }

  analyzeResourceTiming(entry) {
    // Track slow resources
    const loadTime = entry.responseEnd - entry.startTime;
    if (loadTime > 1000) { // > 1 second
      this.recordMetric('slow-resource', {
        name: entry.name,
        loadTime,
        size: entry.transferSize
      });
    }

    // Track resource types
    const resourceType = this.getResourceType(entry.name);
    this.recordMetric(`${resourceType}-load-time`, loadTime);
  }

  analyzeNavigationTiming(entry) {
    // Time to First Byte (TTFB)
    const ttfb = entry.responseStart - entry.requestStart;
    this.recordMetric('TTFB', ttfb);

    // DOM Content Loaded
    const domContentLoaded = entry.domContentLoadedEventEnd - entry.domContentLoadedEventStart;
    this.recordMetric('DOM-Content-Loaded', domContentLoaded);

    // Load Complete
    const loadComplete = entry.loadEventEnd - entry.loadEventStart;
    this.recordMetric('Load-Complete', loadComplete);
  }

  getResourceType(url) {
    if (url.includes('.js')) return 'script';
    if (url.includes('.css')) return 'style';
    if (url.includes('.png') || url.includes('.jpg') || url.includes('.svg')) return 'image';
    if (url.includes('/api/')) return 'api';
    return 'other';
  }

  trackCustomMetrics() {
    // Track memory usage
    if ('memory' in performance) {
      setInterval(() => {
        this.recordMetric('memory-used', performance.memory.usedJSHeapSize);
        this.recordMetric('memory-total', performance.memory.totalJSHeapSize);
      }, 30000); // Every 30 seconds
    }
  }

  getPerformanceReport() {
    const metrics = {};

    this.metrics.forEach((entries, name) => {
      const values = entries.map(entry => entry.value);
      metrics[name] = {
        current: values[values.length - 1],
        average: values.reduce((a, b) => a + b, 0) / values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        count: values.length
      };
    });

    return {
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      metrics
    };
  }

  disconnect() {
    this.observers.forEach(observer => {
      observer.disconnect();
    });
    this.observers.clear();
  }
}

// Global performance monitor instance
const performanceMonitor = new PerformanceMonitor();

export default performanceMonitor;
