/**
 * Web Performance Monitoring System for Pynomaly
 * Tracks Core Web Vitals, custom metrics, and performance budgets
 */

/**
 * Performance Configuration
 */
export const PERFORMANCE_CONFIG = {
  // Core Web Vitals thresholds (Lighthouse recommendations)
  vitals: {
    fcp: { good: 1800, poor: 3000 },      // First Contentful Paint
    lcp: { good: 2500, poor: 4000 },      // Largest Contentful Paint
    fid: { good: 100, poor: 300 },        // First Input Delay
    cls: { good: 0.1, poor: 0.25 },       // Cumulative Layout Shift
    ttfb: { good: 800, poor: 1800 },      // Time to First Byte
    tti: { good: 3800, poor: 7300 },      // Time to Interactive
  },

  // Custom metrics thresholds
  custom: {
    bundleSize: { good: 250 * 1024, poor: 500 * 1024 },
    apiResponse: { good: 500, poor: 2000 },
    imageLoad: { good: 1000, poor: 3000 },
    interactionDelay: { good: 50, poor: 200 },
  },

  // Sampling rates
  sampling: {
    production: 0.1,   // 10% in production
    staging: 0.5,      // 50% in staging
    development: 1.0,  // 100% in development
  }
};

/**
 * Core Performance Monitor
 */
export const createPerformanceMonitor = () => `
/**
 * Comprehensive Performance Monitor
 */
class PerformanceMonitor {
  constructor() {
    this.metrics = new Map();
    this.observers = new Map();
    this.config = ${JSON.stringify(PERFORMANCE_CONFIG)};
    this.sessionId = this.generateSessionId();
    this.startTime = performance.now();

    this.init();
  }

  init() {
    // Initialize Core Web Vitals monitoring
    this.initCoreWebVitals();

    // Initialize custom metrics
    this.initCustomMetrics();

    // Initialize resource monitoring
    this.initResourceMonitoring();

    // Initialize user interaction monitoring
    this.initInteractionMonitoring();

    // Set up periodic reporting
    this.setupReporting();

    console.log('📊 Performance monitoring initialized');
  }

  /**
   * Core Web Vitals Monitoring
   */
  initCoreWebVitals() {
    // First Contentful Paint
    this.observePerformanceEntry('paint', (entries) => {
      entries.forEach(entry => {
        if (entry.name === 'first-contentful-paint') {
          this.recordMetric('FCP', entry.startTime, {
            threshold: this.config.vitals.fcp,
            type: 'core-web-vital'
          });
        }
      });
    });

    // Largest Contentful Paint
    this.observePerformanceEntry('largest-contentful-paint', (entries) => {
      const lastEntry = entries[entries.length - 1];
      this.recordMetric('LCP', lastEntry.startTime, {
        threshold: this.config.vitals.lcp,
        type: 'core-web-vital',
        element: lastEntry.element
      });
    });

    // Cumulative Layout Shift
    this.observePerformanceEntry('layout-shift', (entries) => {
      let cumulativeScore = 0;
      entries.forEach(entry => {
        if (!entry.hadRecentInput) {
          cumulativeScore += entry.value;
        }
      });

      this.recordMetric('CLS', cumulativeScore, {
        threshold: this.config.vitals.cls,
        type: 'core-web-vital'
      });
    });

    // First Input Delay
    this.observePerformanceEntry('first-input', (entries) => {
      const firstInput = entries[0];
      const fid = firstInput.processingStart - firstInput.startTime;

      this.recordMetric('FID', fid, {
        threshold: this.config.vitals.fid,
        type: 'core-web-vital',
        eventType: firstInput.name
      });
    });

    // Navigation timing
    this.measureNavigationTiming();
  }

  measureNavigationTiming() {
    window.addEventListener('load', () => {
      const navigation = performance.getEntriesByType('navigation')[0];

      if (navigation) {
        const ttfb = navigation.responseStart - navigation.requestStart;
        this.recordMetric('TTFB', ttfb, {
          threshold: this.config.vitals.ttfb,
          type: 'navigation'
        });

        // Time to Interactive (approximation)
        const tti = navigation.loadEventEnd - navigation.fetchStart;
        this.recordMetric('TTI', tti, {
          threshold: this.config.vitals.tti,
          type: 'navigation'
        });
      }
    });
  }

  /**
   * Custom Metrics Monitoring
   */
  initCustomMetrics() {
    // Bundle size monitoring
    this.measureBundleSize();

    // API response time monitoring
    this.monitorAPIResponses();

    // Memory usage monitoring
    this.monitorMemoryUsage();

    // Error monitoring
    this.monitorErrors();
  }

  measureBundleSize() {
    const scriptTags = document.querySelectorAll('script[src]');
    let totalSize = 0;

    scriptTags.forEach(script => {
      fetch(script.src, { method: 'HEAD' })
        .then(response => {
          const size = parseInt(response.headers.get('content-length') || '0');
          totalSize += size;

          this.recordMetric('BundleSize', totalSize, {
            threshold: this.config.custom.bundleSize,
            type: 'bundle',
            file: script.src
          });
        })
        .catch(() => {}); // Ignore errors
    });
  }

  monitorAPIResponses() {
    const originalFetch = window.fetch;

    window.fetch = function(...args) {
      const startTime = performance.now();
      const url = args[0];

      return originalFetch.apply(this, args)
        .then(response => {
          const endTime = performance.now();
          const duration = endTime - startTime;

          if (url.includes('/api/')) {
            window.performanceMonitor.recordMetric('APIResponse', duration, {
              threshold: window.performanceMonitor.config.custom.apiResponse,
              type: 'api',
              url,
              status: response.status,
              cached: response.headers.get('x-cache') === 'HIT'
            });
          }

          return response;
        });
    };
  }

  monitorMemoryUsage() {
    if ('memory' in performance) {
      setInterval(() => {
        const memory = performance.memory;
        this.recordMetric('MemoryUsage', memory.usedJSHeapSize, {
          type: 'memory',
          total: memory.totalJSHeapSize,
          limit: memory.jsHeapSizeLimit
        });
      }, 30000); // Every 30 seconds
    }
  }

  monitorErrors() {
    window.addEventListener('error', (event) => {
      this.recordMetric('JavaScriptError', 1, {
        type: 'error',
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
      });
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.recordMetric('UnhandledPromiseRejection', 1, {
        type: 'error',
        reason: event.reason
      });
    });
  }

  /**
   * Resource Monitoring
   */
  initResourceMonitoring() {
    // Monitor resource loading
    this.observePerformanceEntry('resource', (entries) => {
      entries.forEach(entry => {
        const duration = entry.responseEnd - entry.startTime;

        if (entry.initiatorType === 'img') {
          this.recordMetric('ImageLoad', duration, {
            threshold: this.config.custom.imageLoad,
            type: 'resource',
            url: entry.name,
            size: entry.transferSize
          });
        } else if (entry.name.includes('.js')) {
          this.recordMetric('ScriptLoad', duration, {
            type: 'resource',
            url: entry.name,
            size: entry.transferSize,
            cached: entry.transferSize === 0
          });
        } else if (entry.name.includes('.css')) {
          this.recordMetric('StyleLoad', duration, {
            type: 'resource',
            url: entry.name,
            size: entry.transferSize
          });
        }
      });
    });
  }

  /**
   * User Interaction Monitoring
   */
  initInteractionMonitoring() {
    // Click response time
    document.addEventListener('click', (event) => {
      const startTime = performance.now();

      requestAnimationFrame(() => {
        const responseTime = performance.now() - startTime;
        this.recordMetric('ClickResponse', responseTime, {
          threshold: this.config.custom.interactionDelay,
          type: 'interaction',
          target: event.target.tagName
        });
      });
    });

    // Form interaction timing
    document.addEventListener('submit', (event) => {
      const form = event.target;
      const startTime = performance.now();

      // Measure time until form processing starts
      setTimeout(() => {
        const processingTime = performance.now() - startTime;
        this.recordMetric('FormSubmission', processingTime, {
          type: 'interaction',
          formId: form.id || 'anonymous'
        });
      }, 0);
    });
  }

  /**
   * Performance Entry Observer
   */
  observePerformanceEntry(type, callback) {
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        callback(list.getEntries());
      });

      observer.observe({ type, buffered: true });
      this.observers.set(type, observer);
    }
  }

  /**
   * Record Metric
   */
  recordMetric(name, value, metadata = {}) {
    const timestamp = Date.now();
    const metric = {
      name,
      value,
      timestamp,
      sessionId: this.sessionId,
      url: window.location.href,
      userAgent: navigator.userAgent,
      connectionType: this.getConnectionType(),
      ...metadata
    };

    // Store in metrics map
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name).push(metric);

    // Evaluate threshold if provided
    if (metadata.threshold) {
      this.evaluateThreshold(metric);
    }

    // Send to analytics if sampling allows
    this.sendToAnalytics(metric);

    console.log(\`📊 \${name}: \${value}ms\`, metadata);
  }

  evaluateThreshold(metric) {
    const { threshold, value, name } = metric;

    let status = 'good';
    if (value > threshold.poor) {
      status = 'poor';
    } else if (value > threshold.good) {
      status = 'needs-improvement';
    }

    metric.status = status;

    if (status === 'poor') {
      console.warn(\`⚠️ Poor \${name}: \${value}ms (threshold: \${threshold.poor}ms)\`);
      this.triggerAlert(metric);
    }
  }

  triggerAlert(metric) {
    // Trigger performance alert
    document.dispatchEvent(new CustomEvent('performance-alert', {
      detail: metric
    }));
  }

  /**
   * Analytics Integration
   */
  sendToAnalytics(metric) {
    const sampleRate = this.getSampleRate();

    if (Math.random() > sampleRate) {
      return; // Skip due to sampling
    }

    // Send to Google Analytics 4
    if (window.gtag) {
      window.gtag('event', 'performance_metric', {
        metric_name: metric.name,
        metric_value: Math.round(metric.value),
        metric_status: metric.status || 'unknown',
        custom_parameter_1: metric.type || 'custom'
      });
    }

    // Send to custom analytics endpoint
    if (window.location.hostname !== 'localhost') {
      fetch('/api/analytics/performance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metric)
      }).catch(() => {}); // Ignore errors
    }
  }

  getSampleRate() {
    const hostname = window.location.hostname;

    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return this.config.sampling.development;
    } else if (hostname.includes('staging')) {
      return this.config.sampling.staging;
    } else {
      return this.config.sampling.production;
    }
  }

  /**
   * Reporting
   */
  setupReporting() {
    // Report metrics when page is hidden
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.generateReport();
      }
    });

    // Report metrics on page unload
    window.addEventListener('beforeunload', () => {
      this.generateReport();
    });

    // Periodic reporting
    setInterval(() => {
      this.generateReport();
    }, 60000); // Every minute
  }

  generateReport() {
    const report = {
      sessionId: this.sessionId,
      timestamp: Date.now(),
      sessionDuration: Date.now() - this.startTime,
      url: window.location.href,
      metrics: this.getMetricsSummary(),
      vitals: this.getCoreWebVitalsSummary(),
      resources: this.getResourcesSummary(),
      performance: this.getPerformanceScore()
    };

    // Store report locally
    this.storeReport(report);

    // Send to server if needed
    this.sendReport(report);

    return report;
  }

  getMetricsSummary() {
    const summary = {};

    this.metrics.forEach((entries, name) => {
      const values = entries.map(e => e.value);
      summary[name] = {
        count: values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        avg: values.reduce((a, b) => a + b, 0) / values.length,
        p50: this.percentile(values, 50),
        p95: this.percentile(values, 95),
        p99: this.percentile(values, 99)
      };
    });

    return summary;
  }

  getCoreWebVitalsSummary() {
    const vitals = {};

    ['FCP', 'LCP', 'CLS', 'FID', 'TTFB', 'TTI'].forEach(vital => {
      const entries = this.metrics.get(vital);
      if (entries && entries.length > 0) {
        const latest = entries[entries.length - 1];
        vitals[vital] = {
          value: latest.value,
          status: latest.status,
          timestamp: latest.timestamp
        };
      }
    });

    return vitals;
  }

  getResourcesSummary() {
    const resources = {
      total: 0,
      images: 0,
      scripts: 0,
      styles: 0,
      totalSize: 0
    };

    const resourceEntries = performance.getEntriesByType('resource');
    resourceEntries.forEach(entry => {
      resources.total++;
      resources.totalSize += entry.transferSize || 0;

      if (entry.initiatorType === 'img') resources.images++;
      else if (entry.name.includes('.js')) resources.scripts++;
      else if (entry.name.includes('.css')) resources.styles++;
    });

    return resources;
  }

  getPerformanceScore() {
    // Calculate overall performance score based on Core Web Vitals
    const vitals = this.getCoreWebVitalsSummary();
    let score = 100;

    Object.values(vitals).forEach(vital => {
      if (vital.status === 'poor') score -= 20;
      else if (vital.status === 'needs-improvement') score -= 10;
    });

    return Math.max(0, score);
  }

  storeReport(report) {
    if ('localStorage' in window) {
      try {
        const reports = JSON.parse(localStorage.getItem('performance-reports') || '[]');
        reports.push(report);

        // Keep only last 10 reports
        if (reports.length > 10) {
          reports.splice(0, reports.length - 10);
        }

        localStorage.setItem('performance-reports', JSON.stringify(reports));
      } catch (error) {
        console.warn('Failed to store performance report:', error);
      }
    }
  }

  sendReport(report) {
    // Send to server for aggregation
    if (window.navigator.sendBeacon) {
      const blob = new Blob([JSON.stringify(report)], {
        type: 'application/json'
      });
      navigator.sendBeacon('/api/analytics/performance-report', blob);
    }
  }

  /**
   * Utility Methods
   */
  percentile(values, p) {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[index] || 0;
  }

  generateSessionId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }

  getConnectionType() {
    if ('connection' in navigator) {
      return navigator.connection.effectiveType || 'unknown';
    }
    return 'unknown';
  }

  /**
   * Public API
   */
  mark(name) {
    performance.mark(name);
  }

  measure(name, startMark, endMark) {
    performance.measure(name, startMark, endMark);
    const measure = performance.getEntriesByName(name, 'measure')[0];

    this.recordMetric(name, measure.duration, {
      type: 'custom-measure',
      startMark,
      endMark
    });
  }

  time(name, fn) {
    const startTime = performance.now();
    const result = fn();
    const duration = performance.now() - startTime;

    this.recordMetric(name, duration, {
      type: 'custom-timing'
    });

    return result;
  }

  async timeAsync(name, fn) {
    const startTime = performance.now();
    const result = await fn();
    const duration = performance.now() - startTime;

    this.recordMetric(name, duration, {
      type: 'custom-timing-async'
    });

    return result;
  }

  getMetrics() {
    return Object.fromEntries(this.metrics);
  }

  clearMetrics() {
    this.metrics.clear();
  }

  destroy() {
    this.observers.forEach(observer => observer.disconnect());
    this.observers.clear();
    this.metrics.clear();
  }
}

// Initialize global performance monitor
window.performanceMonitor = new PerformanceMonitor();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = window.performanceMonitor;
}
`;

/**
 * Performance Budget Checker
 */
export const createPerformanceBudgetChecker = () => `
/**
 * Performance Budget Checker
 */
class PerformanceBudgetChecker {
  constructor() {
    this.budgets = ${JSON.stringify(PERFORMANCE_CONFIG)};
    this.violations = [];
  }

  checkBudgets() {
    const violations = [];
    const metrics = window.performanceMonitor.getMetrics();

    // Check Core Web Vitals
    Object.entries(this.budgets.vitals).forEach(([metric, threshold]) => {
      const entries = metrics[metric.toUpperCase()];
      if (entries && entries.length > 0) {
        const latest = entries[entries.length - 1];
        if (latest.value > threshold.poor) {
          violations.push({
            type: 'vital',
            metric: metric.toUpperCase(),
            value: latest.value,
            threshold: threshold.poor,
            severity: 'high'
          });
        }
      }
    });

    // Check custom metrics
    Object.entries(this.budgets.custom).forEach(([metric, threshold]) => {
      const entries = metrics[metric];
      if (entries && entries.length > 0) {
        const latest = entries[entries.length - 1];
        if (latest.value > threshold.poor) {
          violations.push({
            type: 'custom',
            metric,
            value: latest.value,
            threshold: threshold.poor,
            severity: 'medium'
          });
        }
      }
    });

    this.violations = violations;

    if (violations.length > 0) {
      console.warn('⚠️ Performance budget violations:', violations);
      this.reportViolations(violations);
    }

    return violations;
  }

  reportViolations(violations) {
    violations.forEach(violation => {
      // Send to monitoring service
      if (window.gtag) {
        window.gtag('event', 'performance_budget_violation', {
          metric_name: violation.metric,
          metric_value: Math.round(violation.value),
          threshold: violation.threshold,
          severity: violation.severity
        });
      }
    });
  }

  getScore() {
    const totalChecks = Object.keys(this.budgets.vitals).length +
                       Object.keys(this.budgets.custom).length;
    const violations = this.violations.length;

    return Math.max(0, ((totalChecks - violations) / totalChecks) * 100);
  }
}

// Check budgets periodically
setInterval(() => {
  const checker = new PerformanceBudgetChecker();
  checker.checkBudgets();
}, 30000); // Every 30 seconds
`;

export { PERFORMANCE_CONFIG };
