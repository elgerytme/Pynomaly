// Performance Monitor - Real-time performance tracking
export class PerformanceMonitor {
  constructor() {
    this.metrics = {};
    this.observers = new Map();

    this.init();
  }

  init() {
    this.setupCoreWebVitals();
    this.setupResourceMonitoring();
    this.setupLongTaskMonitoring();
  }

  setupCoreWebVitals() {
    // LCP Observer
    if ("PerformanceObserver" in window) {
      const lcpObserver = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
          this.recordMetric("lcp", entry.startTime);
        }
      });
      lcpObserver.observe({ type: "largest-contentful-paint", buffered: true });
      this.observers.set("lcp", lcpObserver);

      // FID Observer
      const fidObserver = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
          const fid = entry.processingStart - entry.startTime;
          this.recordMetric("fid", fid);
        }
      });
      fidObserver.observe({ type: "first-input", buffered: true });
      this.observers.set("fid", fidObserver);

      // CLS Observer
      let clsValue = 0;
      const clsObserver = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
          if (!entry.hadRecentInput) {
            clsValue += entry.value;
          }
        }
        this.recordMetric("cls", clsValue);
      });
      clsObserver.observe({ type: "layout-shift", buffered: true });
      this.observers.set("cls", clsObserver);
    }
  }

  setupResourceMonitoring() {
    if ("PerformanceObserver" in window) {
      const resourceObserver = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
          if (entry.duration > 1000) {
            console.warn("Slow resource:", entry.name, `${entry.duration}ms`);
          }
        }
      });
      resourceObserver.observe({ type: "resource", buffered: true });
      this.observers.set("resource", resourceObserver);
    }
  }

  setupLongTaskMonitoring() {
    if ("PerformanceObserver" in window) {
      const longTaskObserver = new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
          console.warn("Long task detected:", `${entry.duration}ms`);
          this.recordMetric("long-task", entry.duration);
        }
      });
      longTaskObserver.observe({ type: "longtask", buffered: true });
      this.observers.set("longtask", longTaskObserver);
    }
  }

  recordMetric(name, value) {
    this.metrics[name] = value;

    // Send to analytics
    if (navigator.sendBeacon) {
      const data = JSON.stringify({
        metric: name,
        value: value,
        timestamp: Date.now(),
        url: window.location.href,
      });
      navigator.sendBeacon("/api/analytics/performance", data);
    }
  }

  getMetrics() {
    return { ...this.metrics };
  }

  destroy() {
    this.observers.forEach((observer) => {
      observer.disconnect();
    });
    this.observers.clear();
  }
}

// Initialize Performance Monitor
if (typeof window !== "undefined") {
  window.PerformanceMonitor = new PerformanceMonitor();
}
