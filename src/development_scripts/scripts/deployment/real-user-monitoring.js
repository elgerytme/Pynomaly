/**
 * Real User Monitoring (RUM) for Pynomaly Web UI
 * Collects performance metrics from actual users in production
 */

class RealUserMonitoring {
  constructor(options = {}) {
    this.apiEndpoint = options.apiEndpoint || '/api/metrics/rum';
    this.sampleRate = options.sampleRate || 0.1; // 10% sampling
    this.batchSize = options.batchSize || 10;
    this.flushInterval = options.flushInterval || 30000; // 30 seconds
    this.enabled = options.enabled !== false;

    this.metrics = [];
    this.sessionId = this.generateSessionId();
    this.startTime = performance.now();

    if (this.enabled && this.shouldSample()) {
      this.initialize();
    }
  }

  initialize() {
    console.log('[RUM] Real User Monitoring initialized');

    // Collect Core Web Vitals
    this.collectCoreWebVitals();

    // Monitor navigation timing
    this.collectNavigationTiming();

    // Monitor resource timing
    this.collectResourceTiming();

    // Monitor user interactions
    this.collectUserInteractions();

    // Monitor JavaScript errors
    this.collectErrorMetrics();

    // Monitor memory usage
    this.collectMemoryMetrics();

    // Monitor network conditions
    this.collectNetworkMetrics();

    // Start periodic flushing
    this.startPeriodicFlush();

    // Flush on page unload
    this.setupUnloadHandler();
  }

  collectCoreWebVitals() {
    // Largest Contentful Paint (LCP)
    if ('PerformanceObserver' in window) {
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1];

        this.addMetric({
          type: 'core-web-vitals',
          name: 'largest-contentful-paint',
          value: lastEntry.startTime,
          timestamp: Date.now(),
          url: window.location.href,
          sessionId: this.sessionId
        });
      });

      try {
        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
      } catch (e) {
        console.warn('[RUM] LCP observation not supported');
      }
    }

    // First Input Delay (FID)
    if ('PerformanceObserver' in window) {
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(entry => {
          this.addMetric({
            type: 'core-web-vitals',
            name: 'first-input-delay',
            value: entry.processingStart - entry.startTime,
            timestamp: Date.now(),
            url: window.location.href,
            sessionId: this.sessionId,
            inputType: entry.name
          });
        });
      });

      try {
        fidObserver.observe({ entryTypes: ['first-input'] });
      } catch (e) {
        console.warn('[RUM] FID observation not supported');
      }
    }

    // Cumulative Layout Shift (CLS)
    if ('PerformanceObserver' in window) {
      let clsValue = 0;

      const clsObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(entry => {
          if (!entry.hadRecentInput) {
            clsValue += entry.value;
          }
        });

        this.addMetric({
          type: 'core-web-vitals',
          name: 'cumulative-layout-shift',
          value: clsValue,
          timestamp: Date.now(),
          url: window.location.href,
          sessionId: this.sessionId
        });
      });

      try {
        clsObserver.observe({ entryTypes: ['layout-shift'] });
      } catch (e) {
        console.warn('[RUM] CLS observation not supported');
      }
    }
  }

  collectNavigationTiming() {
    if ('performance' in window && 'getEntriesByType' in performance) {
      window.addEventListener('load', () => {
        setTimeout(() => {
          const navigation = performance.getEntriesByType('navigation')[0];
          if (navigation) {
            this.addMetric({
              type: 'navigation',
              name: 'page-load-timing',
              metrics: {
                dns: navigation.domainLookupEnd - navigation.domainLookupStart,
                tcp: navigation.connectEnd - navigation.connectStart,
                ssl: navigation.connectEnd - navigation.secureConnectionStart,
                ttfb: navigation.responseStart - navigation.requestStart,
                download: navigation.responseEnd - navigation.responseStart,
                domParse: navigation.domContentLoadedEventEnd - navigation.responseEnd,
                domReady: navigation.domContentLoadedEventEnd - navigation.navigationStart,
                pageLoad: navigation.loadEventEnd - navigation.navigationStart
              },
              timestamp: Date.now(),
              url: window.location.href,
              sessionId: this.sessionId
            });
          }
        }, 0);
      });
    }
  }

  collectResourceTiming() {
    if ('PerformanceObserver' in window) {
      const resourceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(entry => {
          // Only track significant resources
          if (entry.transferSize > 10000 || entry.duration > 100) {
            this.addMetric({
              type: 'resource',
              name: 'resource-timing',
              resource: {
                name: entry.name,
                type: entry.initiatorType,
                size: entry.transferSize,
                duration: entry.duration,
                startTime: entry.startTime
              },
              timestamp: Date.now(),
              url: window.location.href,
              sessionId: this.sessionId
            });
          }
        });
      });

      try {
        resourceObserver.observe({ entryTypes: ['resource'] });
      } catch (e) {
        console.warn('[RUM] Resource timing observation not supported');
      }
    }
  }

  collectUserInteractions() {
    // Track click interactions
    let clickTimeout;
    document.addEventListener('click', (event) => {
      clearTimeout(clickTimeout);
      clickTimeout = setTimeout(() => {
        this.addMetric({
          type: 'interaction',
          name: 'click',
          element: {
            tagName: event.target.tagName,
            className: event.target.className,
            id: event.target.id,
            text: event.target.textContent?.substring(0, 50)
          },
          timestamp: Date.now(),
          url: window.location.href,
          sessionId: this.sessionId
        });
      }, 100);
    });

    // Track scroll depth
    let maxScrollDepth = 0;
    let scrollTimeout;

    window.addEventListener('scroll', () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        const scrollDepth = Math.round(
          (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100
        );

        if (scrollDepth > maxScrollDepth) {
          maxScrollDepth = scrollDepth;

          // Report at 25%, 50%, 75%, 100% milestones
          if ([25, 50, 75, 100].includes(Math.floor(scrollDepth / 25) * 25)) {
            this.addMetric({
              type: 'engagement',
              name: 'scroll-depth',
              value: scrollDepth,
              timestamp: Date.now(),
              url: window.location.href,
              sessionId: this.sessionId
            });
          }
        }
      }, 100);
    });

    // Track time on page
    window.addEventListener('beforeunload', () => {
      this.addMetric({
        type: 'engagement',
        name: 'time-on-page',
        value: performance.now() - this.startTime,
        timestamp: Date.now(),
        url: window.location.href,
        sessionId: this.sessionId
      });
    });
  }

  collectErrorMetrics() {
    // JavaScript errors
    window.addEventListener('error', (event) => {
      this.addMetric({
        type: 'error',
        name: 'javascript-error',
        error: {
          message: event.message,
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno,
          stack: event.error?.stack
        },
        timestamp: Date.now(),
        url: window.location.href,
        sessionId: this.sessionId
      });
    });

    // Unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.addMetric({
        type: 'error',
        name: 'unhandled-promise-rejection',
        error: {
          reason: event.reason?.toString(),
          stack: event.reason?.stack
        },
        timestamp: Date.now(),
        url: window.location.href,
        sessionId: this.sessionId
      });
    });
  }

  collectMemoryMetrics() {
    if ('memory' in performance) {
      setInterval(() => {
        this.addMetric({
          type: 'performance',
          name: 'memory-usage',
          memory: {
            used: performance.memory.usedJSHeapSize,
            total: performance.memory.totalJSHeapSize,
            limit: performance.memory.jsHeapSizeLimit
          },
          timestamp: Date.now(),
          url: window.location.href,
          sessionId: this.sessionId
        });
      }, 60000); // Every minute
    }
  }

  collectNetworkMetrics() {
    if ('connection' in navigator) {
      const connection = navigator.connection;

      this.addMetric({
        type: 'network',
        name: 'connection-info',
        connection: {
          effectiveType: connection.effectiveType,
          downlink: connection.downlink,
          rtt: connection.rtt,
          saveData: connection.saveData
        },
        timestamp: Date.now(),
        url: window.location.href,
        sessionId: this.sessionId
      });

      // Monitor connection changes
      connection.addEventListener('change', () => {
        this.addMetric({
          type: 'network',
          name: 'connection-change',
          connection: {
            effectiveType: connection.effectiveType,
            downlink: connection.downlink,
            rtt: connection.rtt,
            saveData: connection.saveData
          },
          timestamp: Date.now(),
          url: window.location.href,
          sessionId: this.sessionId
        });
      });
    }
  }

  addMetric(metric) {
    this.metrics.push(metric);

    if (this.metrics.length >= this.batchSize) {
      this.flush();
    }
  }

  startPeriodicFlush() {
    setInterval(() => {
      if (this.metrics.length > 0) {
        this.flush();
      }
    }, this.flushInterval);
  }

  setupUnloadHandler() {
    window.addEventListener('beforeunload', () => {
      if (this.metrics.length > 0) {
        this.flush(true); // Synchronous flush on unload
      }
    });

    // Use Page Visibility API for better unload detection
    if ('visibilityState' in document) {
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden' && this.metrics.length > 0) {
          this.flush(true);
        }
      });
    }
  }

  flush(synchronous = false) {
    if (this.metrics.length === 0) return;

    const payload = {
      metrics: [...this.metrics],
      metadata: {
        userAgent: navigator.userAgent,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight
        },
        screen: {
          width: screen.width,
          height: screen.height,
          pixelRatio: window.devicePixelRatio
        },
        timestamp: Date.now()
      }
    };

    this.metrics = []; // Clear metrics after copying

    if (synchronous) {
      // Use sendBeacon for synchronous sends (on page unload)
      if ('sendBeacon' in navigator) {
        navigator.sendBeacon(
          this.apiEndpoint,
          JSON.stringify(payload)
        );
      }
    } else {
      // Use fetch for asynchronous sends
      fetch(this.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      }).catch(error => {
        console.warn('[RUM] Failed to send metrics:', error);
        // Re-add metrics on failure for retry
        this.metrics.unshift(...payload.metrics);
      });
    }
  }

  shouldSample() {
    return Math.random() < this.sampleRate;
  }

  generateSessionId() {
    return `rum_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Public API methods
  trackCustomEvent(eventName, eventData = {}) {
    this.addMetric({
      type: 'custom',
      name: eventName,
      data: eventData,
      timestamp: Date.now(),
      url: window.location.href,
      sessionId: this.sessionId
    });
  }

  trackTiming(name, startTime, endTime) {
    this.addMetric({
      type: 'timing',
      name: name,
      value: endTime - startTime,
      timestamp: Date.now(),
      url: window.location.href,
      sessionId: this.sessionId
    });
  }

  updateSampleRate(newRate) {
    this.sampleRate = newRate;
  }

  getSessionId() {
    return this.sessionId;
  }
}

// Auto-initialize if in browser environment
if (typeof window !== 'undefined') {
  // Initialize RUM with default settings
  window.PynomalyRUM = new RealUserMonitoring({
    sampleRate: 0.1, // 10% sampling
    enabled: true
  });

  // Expose global methods for custom tracking
  window.trackCustomEvent = (eventName, eventData) => {
    if (window.PynomalyRUM) {
      window.PynomalyRUM.trackCustomEvent(eventName, eventData);
    }
  };

  window.trackTiming = (name, startTime, endTime) => {
    if (window.PynomalyRUM) {
      window.PynomalyRUM.trackTiming(name, startTime, endTime);
    }
  };
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = RealUserMonitoring;
}
