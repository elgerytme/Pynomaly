/**
 * Real User Monitoring (RUM) for Pynomaly Web Interface
 * Tracks Core Web Vitals and custom performance metrics
 */

class PerformanceMonitor {
  constructor() {
    this.metrics = {};
    this.startTime = performance.now();
    this.apiEndpoint = '/api/metrics/performance';
    this.batchSize = 10;
    this.metricsQueue = [];
    
    this.initializeMonitoring();
  }

  initializeMonitoring() {
    // Initialize Core Web Vitals monitoring
    this.initCoreWebVitals();
    
    // Monitor custom metrics
    this.initCustomMetrics();
    
    // Set up periodic reporting
    this.initPeriodicReporting();
    
    // Monitor page visibility changes
    this.initVisibilityTracking();
  }

  initCoreWebVitals() {
    // LCP (Largest Contentful Paint)
    this.observeLCP();
    
    // FID (First Input Delay)
    this.observeFID();
    
    // CLS (Cumulative Layout Shift)
    this.observeCLS();
    
    // FCP (First Contentful Paint)
    this.observeFCP();
  }

  observeLCP() {
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1];
          
          this.metrics.lcp = {
            value: Math.round(lastEntry.startTime),
            element: lastEntry.element ? lastEntry.element.tagName : 'unknown',
            timestamp: Date.now()
          };
          
          this.queueMetric('lcp', this.metrics.lcp);
        });
        
        observer.observe({ entryTypes: ['largest-contentful-paint'] });
      } catch (e) {
        console.warn('LCP monitoring not supported:', e);
      }
    }
  }

  observeFID() {
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            this.metrics.fid = {
              value: Math.round(entry.processingStart - entry.startTime),
              timestamp: Date.now()
            };
            
            this.queueMetric('fid', this.metrics.fid);
          }
        });
        
        observer.observe({ entryTypes: ['first-input'] });
      } catch (e) {
        console.warn('FID monitoring not supported:', e);
      }
    }
  }

  observeCLS() {
    if ('PerformanceObserver' in window) {
      let clsValue = 0;
      
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (!entry.hadRecentInput) {
              clsValue += entry.value;
            }
          }
          
          this.metrics.cls = {
            value: parseFloat(clsValue.toFixed(4)),
            timestamp: Date.now()
          };
        });
        
        observer.observe({ entryTypes: ['layout-shift'] });
        
        // Report CLS when page becomes hidden
        document.addEventListener('visibilitychange', () => {
          if (document.visibilityState === 'hidden' && this.metrics.cls) {
            this.queueMetric('cls', this.metrics.cls);
          }
        });
      } catch (e) {
        console.warn('CLS monitoring not supported:', e);
      }
    }
  }

  observeFCP() {
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.name === 'first-contentful-paint') {
              this.metrics.fcp = {
                value: Math.round(entry.startTime),
                timestamp: Date.now()
              };
              
              this.queueMetric('fcp', this.metrics.fcp);
            }
          }
        });
        
        observer.observe({ entryTypes: ['paint'] });
      } catch (e) {
        console.warn('FCP monitoring not supported:', e);
      }
    }
  }

  initCustomMetrics() {
    // Chart rendering performance
    this.monitorChartRendering();
    
    // API response times
    this.monitorAPIResponses();
    
    // User interaction responsiveness
    this.monitorInteractions();
    
    // Resource loading times
    this.monitorResourceLoading();
  }

  monitorChartRendering() {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === 1) { // Element node
            const charts = node.querySelectorAll('.chart-container, [data-testid="chart"]');
            charts.forEach((chart) => {
              this.measureChartRenderTime(chart);
            });
          }
        });
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  measureChartRenderTime(chartElement) {
    const startTime = performance.now();
    
    const checkRendered = () => {
      const canvas = chartElement.querySelector('canvas');
      const svg = chartElement.querySelector('svg');
      
      if (canvas || svg) {
        const renderTime = Math.round(performance.now() - startTime);
        
        this.queueMetric('chart_render', {
          value: renderTime,
          chart_type: chartElement.dataset.chartType || 'unknown',
          timestamp: Date.now()
        });
      } else {
        // Check again in 50ms if not rendered yet
        setTimeout(checkRendered, 50);
      }
    };
    
    setTimeout(checkRendered, 0);
  }

  monitorAPIResponses() {
    // Intercept fetch requests
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const [url, options] = args;
      
      if (url.includes('/api/')) {
        const startTime = performance.now();
        
        try {
          const response = await originalFetch(...args);
          const duration = Math.round(performance.now() - startTime);
          
          this.queueMetric('api_response', {
            url: url.replace(/\?.*/g, ''), // Remove query params
            duration,
            status: response.status,
            method: options?.method || 'GET',
            timestamp: Date.now()
          });
          
          return response;
        } catch (error) {
          const duration = Math.round(performance.now() - startTime);
          
          this.queueMetric('api_error', {
            url: url.replace(/\?.*/g, ''),
            duration,
            error: error.message,
            timestamp: Date.now()
          });
          
          throw error;
        }
      }
      
      return originalFetch(...args);
    };
  }

  monitorInteractions() {
    let interactionStart = null;
    
    // Monitor click responsiveness
    document.addEventListener('mousedown', () => {
      interactionStart = performance.now();
    });
    
    document.addEventListener('click', (event) => {
      if (interactionStart) {
        const responseTime = Math.round(performance.now() - interactionStart);
        
        this.queueMetric('interaction_response', {
          type: 'click',
          duration: responseTime,
          element: event.target.tagName,
          timestamp: Date.now()
        });
        
        interactionStart = null;
      }
    });
    
    // Monitor keyboard responsiveness
    document.addEventListener('keydown', () => {
      interactionStart = performance.now();
    });
    
    document.addEventListener('keyup', (event) => {
      if (interactionStart) {
        const responseTime = Math.round(performance.now() - interactionStart);
        
        this.queueMetric('interaction_response', {
          type: 'keyboard',
          duration: responseTime,
          key: event.key,
          timestamp: Date.now()
        });
        
        interactionStart = null;
      }
    });
  }

  monitorResourceLoading() {
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.duration > 1000) { // Only track slow resources
              this.queueMetric('slow_resource', {
                name: entry.name.split('/').pop(),
                duration: Math.round(entry.duration),
                size: entry.transferSize || 0,
                type: entry.initiatorType,
                timestamp: Date.now()
              });
            }
          }
        });
        
        observer.observe({ entryTypes: ['resource'] });
      } catch (e) {
        console.warn('Resource monitoring not supported:', e);
      }
    }
  }

  initPeriodicReporting() {
    // Send metrics every 30 seconds
    setInterval(() => {
      this.sendQueuedMetrics();
    }, 30000);
    
    // Send metrics when page becomes hidden
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.sendQueuedMetrics();
      }
    });
    
    // Send metrics before page unload
    window.addEventListener('beforeunload', () => {
      this.sendQueuedMetrics(true);
    });
  }

  initVisibilityTracking() {
    let visibilityStart = Date.now();
    
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        const visibilityDuration = Date.now() - visibilityStart;
        
        this.queueMetric('page_visibility', {
          duration: visibilityDuration,
          timestamp: Date.now()
        });
      } else {
        visibilityStart = Date.now();
      }
    });
  }

  queueMetric(type, data) {
    this.metricsQueue.push({
      type,
      data,
      page: window.location.pathname,
      userAgent: navigator.userAgent,
      timestamp: Date.now()
    });
    
    // Send immediately if queue is full
    if (this.metricsQueue.length >= this.batchSize) {
      this.sendQueuedMetrics();
    }
  }

  async sendQueuedMetrics(useBeacon = false) {
    if (this.metricsQueue.length === 0) return;
    
    const payload = {
      metrics: [...this.metricsQueue],
      session: this.getSessionId(),
      page_load_time: Date.now() - this.startTime
    };
    
    // Clear queue immediately to prevent duplicate sends
    this.metricsQueue = [];
    
    try {
      if (useBeacon && navigator.sendBeacon) {
        // Use sendBeacon for page unload to ensure delivery
        navigator.sendBeacon(
          this.apiEndpoint,
          JSON.stringify(payload)
        );
      } else {
        // Use fetch for normal reporting
        await fetch(this.apiEndpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload)
        });
      }
    } catch (error) {
      console.warn('Failed to send performance metrics:', error);
      
      // Re-queue metrics on failure (except for beacon sends)
      if (!useBeacon) {
        this.metricsQueue.unshift(...payload.metrics);
      }
    }
  }

  getSessionId() {
    let sessionId = sessionStorage.getItem('pynomaly_session_id');
    
    if (!sessionId) {
      sessionId = 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      sessionStorage.setItem('pynomaly_session_id', sessionId);
    }
    
    return sessionId;
  }

  // Public API for manual metric tracking
  trackCustomMetric(name, value, metadata = {}) {
    this.queueMetric('custom', {
      name,
      value,
      metadata,
      timestamp: Date.now()
    });
  }

  // Get current metrics summary
  getMetricsSummary() {
    return {
      coreWebVitals: {
        lcp: this.metrics.lcp?.value || null,
        fid: this.metrics.fid?.value || null,
        cls: this.metrics.cls?.value || null,
        fcp: this.metrics.fcp?.value || null
      },
      queueSize: this.metricsQueue.length,
      sessionId: this.getSessionId()
    };
  }
}

// Initialize performance monitoring when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.PynormalyPerformanceMonitor = new PerformanceMonitor();
  });
} else {
  window.PynormalyPerformanceMonitor = new PerformanceMonitor();
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = PerformanceMonitor;
}
