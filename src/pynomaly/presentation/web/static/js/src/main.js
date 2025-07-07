// Pynomaly Web UI - Main JavaScript Entry Point
import "./components/anomaly-detector.js";
import "./components/data-uploader.js";
import "./components/chart-components.js";
import "./utils/htmx-extensions.js";
import "./utils/pwa-manager.js";
import "./utils/accessibility.js";
import "./utils/performance-monitor.js";

// Core Application Class
class PynomalyApp {
  constructor() {
    this.version = "1.0.0";
    this.isOnline = navigator.onLine;
    this.theme = localStorage.getItem("theme") || "light";
    this.components = new Map();

    this.init();
  }

  // Initialize the application
  async init() {
    console.log("🚀 Pynomaly App initializing...");

    try {
      // Initialize core features
      await this.initServiceWorker();
      this.initTheme();
      this.initHTMX();
      this.initAccessibility();
      this.initPerformanceMonitoring();
      this.bindEventListeners();

      // Initialize components
      this.initComponents();

      // Setup offline handling
      this.setupOfflineHandling();

      console.log("✅ Pynomaly App initialized successfully");
    } catch (error) {
      console.error("❌ Failed to initialize Pynomaly App:", error);
      this.showErrorMessage("Failed to initialize application");
    }
  }

  // Initialize Service Worker for PWA functionality
  async initServiceWorker() {
    if ("serviceWorker" in navigator) {
      try {
        const registration = await navigator.serviceWorker.register("/sw.js");
        console.log("Service Worker registered:", registration);

        // Handle service worker updates
        registration.addEventListener("updatefound", () => {
          const newWorker = registration.installing;
          newWorker.addEventListener("statechange", () => {
            if (
              newWorker.state === "installed" &&
              navigator.serviceWorker.controller
            ) {
              this.showUpdateNotification();
            }
          });
        });
      } catch (error) {
        console.warn("Service Worker registration failed:", error);
      }
    }
  }

  // Initialize theme system
  initTheme() {
    // Apply saved theme
    document.documentElement.classList.toggle("dark", this.theme === "dark");

    // Update theme toggle button if it exists
    const themeToggle = document.querySelector("[data-theme-toggle]");
    if (themeToggle) {
      themeToggle.setAttribute("aria-pressed", this.theme === "dark");
    }

    // Listen for system theme changes
    if (window.matchMedia) {
      const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
      mediaQuery.addEventListener("change", (e) => {
        if (!localStorage.getItem("theme")) {
          this.setTheme(e.matches ? "dark" : "light");
        }
      });
    }
  }

  // Toggle theme
  toggleTheme() {
    const newTheme = this.theme === "light" ? "dark" : "light";
    this.setTheme(newTheme);
  }

  // Set theme
  setTheme(theme) {
    this.theme = theme;
    localStorage.setItem("theme", theme);
    document.documentElement.classList.toggle("dark", theme === "dark");

    // Update theme toggle button
    const themeToggle = document.querySelector("[data-theme-toggle]");
    if (themeToggle) {
      themeToggle.setAttribute("aria-pressed", theme === "dark");
    }

    // Dispatch theme change event
    document.dispatchEvent(
      new CustomEvent("themechange", {
        detail: { theme },
      }),
    );
  }

  // Initialize HTMX with custom extensions
  initHTMX() {
    if (typeof htmx !== "undefined") {
      // Configure HTMX
      htmx.config.responseHandling = [
        { code: "204", swap: false },
        { code: "[23].+", swap: true },
        { code: "[45].+", swap: false, error: true },
      ];

      // Add loading indicators
      htmx.on("htmx:beforeRequest", (event) => {
        this.showLoadingIndicator(event.target);
      });

      htmx.on("htmx:afterRequest", (event) => {
        this.hideLoadingIndicator(event.target);
      });

      // Handle HTMX errors
      htmx.on("htmx:responseError", (event) => {
        console.error("HTMX error:", event.detail);
        this.showErrorMessage("Request failed. Please try again.");
      });

      // Handle network errors
      htmx.on("htmx:sendError", (event) => {
        console.error("HTMX network error:", event.detail);
        this.showErrorMessage("Network error. Please check your connection.");
      });
    }
  }

  // Initialize accessibility features
  initAccessibility() {
    // Skip link focus management
    this.setupSkipLinks();

    // Focus management for dynamic content
    this.setupFocusManagement();

    // Keyboard navigation enhancement
    this.setupKeyboardNavigation();

    // Announce dynamic content changes
    this.setupLiveRegions();
  }

  // Initialize performance monitoring
  initPerformanceMonitoring() {
    if ("PerformanceObserver" in window) {
      // Monitor Core Web Vitals
      this.monitorCoreWebVitals();

      // Monitor resource loading
      this.monitorResourceLoading();

      // Monitor long tasks
      this.monitorLongTasks();
    }
  }

  // Monitor Core Web Vitals
  monitorCoreWebVitals() {
    // LCP (Largest Contentful Paint)
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        console.log("LCP:", entry.startTime);
        this.reportMetric("lcp", entry.startTime);
      }
    }).observe({ type: "largest-contentful-paint", buffered: true });

    // FID (First Input Delay)
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        const fid = entry.processingStart - entry.startTime;
        console.log("FID:", fid);
        this.reportMetric("fid", fid);
      }
    }).observe({ type: "first-input", buffered: true });

    // CLS (Cumulative Layout Shift)
    let clsValue = 0;
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
        }
      }
      console.log("CLS:", clsValue);
      this.reportMetric("cls", clsValue);
    }).observe({ type: "layout-shift", buffered: true });
  }

  // Monitor resource loading performance
  monitorResourceLoading() {
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        if (entry.duration > 1000) {
          // Log slow resources
          console.warn("Slow resource:", entry.name, entry.duration);
        }
      }
    }).observe({ type: "resource", buffered: true });
  }

  // Monitor long tasks that block the main thread
  monitorLongTasks() {
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        console.warn("Long task detected:", entry.duration);
        this.reportMetric("long-task", entry.duration);
      }
    }).observe({ type: "longtask", buffered: true });
  }

  // Report performance metrics
  reportMetric(name, value) {
    // Send to analytics or monitoring service
    if (navigator.sendBeacon) {
      const data = JSON.stringify({
        metric: name,
        value,
        timestamp: Date.now(),
      });
      navigator.sendBeacon("/api/metrics", data);
    }
  }

  // Initialize components
  initComponents() {
    // Auto-initialize components with data attributes
    document.querySelectorAll("[data-component]").forEach((element) => {
      const componentName = element.dataset.component;
      this.initializeComponent(componentName, element);
    });
  }

  // Initialize a specific component
  initializeComponent(name, element) {
    try {
      switch (name) {
        case "anomaly-detector":
          import("./components/anomaly-detector.js").then((module) => {
            this.components.set(element, new module.AnomalyDetector(element));
          });
          break;
        case "data-uploader":
          import("./components/data-uploader.js").then((module) => {
            this.components.set(element, new module.DataUploader(element));
          });
          break;
        case "chart":
          import("./components/chart-components.js").then((module) => {
            this.components.set(element, new module.ChartComponent(element));
          });
          break;
        default:
          console.warn(`Unknown component: ${name}`);
      }
    } catch (error) {
      console.error(`Failed to initialize component ${name}:`, error);
    }
  }

  // Setup offline handling
  setupOfflineHandling() {
    window.addEventListener("online", () => {
      this.isOnline = true;
      this.hideOfflineNotification();
      console.log("✅ Back online");
    });

    window.addEventListener("offline", () => {
      this.isOnline = false;
      this.showOfflineNotification();
      console.log("📡 Gone offline");
    });
  }

  // Bind global event listeners
  bindEventListeners() {
    // Theme toggle
    document.addEventListener("click", (e) => {
      if (e.target.matches("[data-theme-toggle]")) {
        e.preventDefault();
        this.toggleTheme();
      }
    });

    // Skip links
    document.addEventListener("click", (e) => {
      if (e.target.matches(".skip-link")) {
        e.preventDefault();
        const target = document.querySelector(e.target.getAttribute("href"));
        if (target) {
          target.focus();
          target.scrollIntoView();
        }
      }
    });

    // Global keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      // Alt + T: Toggle theme
      if (e.altKey && e.key === "t") {
        e.preventDefault();
        this.toggleTheme();
      }

      // Alt + D: Go to dashboard
      if (e.altKey && e.key === "d") {
        e.preventDefault();
        window.location.href = "/dashboard";
      }
    });

    // Handle component events
    document.addEventListener("component:ready", (e) => {
      console.log("Component ready:", e.detail);
    });

    document.addEventListener("component:error", (e) => {
      console.error("Component error:", e.detail);
      this.showErrorMessage(e.detail.message);
    });
  }

  // Setup skip links for accessibility
  setupSkipLinks() {
    const skipLinks = document.querySelectorAll(".skip-link");
    skipLinks.forEach((link) => {
      link.addEventListener("focus", () => {
        link.style.position = "absolute";
        link.style.left = "6px";
        link.style.top = "7px";
        link.style.zIndex = "999999";
      });

      link.addEventListener("blur", () => {
        link.style.position = "absolute";
        link.style.left = "-10000px";
        link.style.top = "auto";
      });
    });
  }

  // Setup focus management for dynamic content
  setupFocusManagement() {
    // Restore focus after HTMX requests
    htmx.on("htmx:afterSwap", (event) => {
      const newContent = event.target;
      const focusTarget =
        newContent.querySelector("[autofocus]") ||
        newContent.querySelector("h1, h2, h3") ||
        newContent;

      if (focusTarget && focusTarget.focus) {
        focusTarget.focus();
      }
    });
  }

  // Setup keyboard navigation
  setupKeyboardNavigation() {
    // Enhanced tab navigation
    document.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        document.body.classList.add("keyboard-navigation");
      }
    });

    document.addEventListener("mousedown", () => {
      document.body.classList.remove("keyboard-navigation");
    });
  }

  // Setup live regions for screen readers
  setupLiveRegions() {
    // Create live region if it doesn't exist
    if (!document.getElementById("live-region")) {
      const liveRegion = document.createElement("div");
      liveRegion.id = "live-region";
      liveRegion.setAttribute("aria-live", "polite");
      liveRegion.setAttribute("aria-atomic", "true");
      liveRegion.className = "sr-only";
      document.body.appendChild(liveRegion);
    }
  }

  // Announce message to screen readers
  announceToScreenReader(message) {
    const liveRegion = document.getElementById("live-region");
    if (liveRegion) {
      liveRegion.textContent = message;
      setTimeout(() => {
        liveRegion.textContent = "";
      }, 1000);
    }
  }

  // Show loading indicator
  showLoadingIndicator(element) {
    const indicator = document.createElement("div");
    indicator.className = "loading-indicator";
    indicator.innerHTML = '<div class="loading-spinner w-4 h-4"></div>';
    indicator.setAttribute("aria-label", "Loading");

    element.style.position = "relative";
    element.appendChild(indicator);
  }

  // Hide loading indicator
  hideLoadingIndicator(element) {
    const indicator = element.querySelector(".loading-indicator");
    if (indicator) {
      indicator.remove();
    }
  }

  // Show error message
  showErrorMessage(message) {
    this.showNotification(message, "error");
    this.announceToScreenReader(`Error: ${message}`);
  }

  // Show success message
  showSuccessMessage(message) {
    this.showNotification(message, "success");
    this.announceToScreenReader(message);
  }

  // Show notification
  showNotification(message, type = "info") {
    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
      <div class="notification-content">
        <span class="notification-message">${message}</span>
        <button class="notification-close" aria-label="Close notification">&times;</button>
      </div>
    `;

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove();
      }
    }, 5000);

    // Close button
    notification
      .querySelector(".notification-close")
      .addEventListener("click", () => {
        notification.remove();
      });

    // Add to page
    const container = document.getElementById("notifications") || document.body;
    container.appendChild(notification);
  }

  // Show offline notification
  showOfflineNotification() {
    this.showNotification(
      "You are currently offline. Some features may not be available.",
      "warning",
    );
  }

  // Hide offline notification
  hideOfflineNotification() {
    this.showNotification("You are back online!", "success");
  }

  // Show update notification
  showUpdateNotification() {
    const notification = document.createElement("div");
    notification.className = "notification notification-info";
    notification.innerHTML = `
      <div class="notification-content">
        <span class="notification-message">A new version is available!</span>
        <button class="btn btn-sm btn-primary" onclick="window.location.reload()">Update</button>
        <button class="notification-close" aria-label="Close notification">&times;</button>
      </div>
    `;

    notification
      .querySelector(".notification-close")
      .addEventListener("click", () => {
        notification.remove();
      });

    const container = document.getElementById("notifications") || document.body;
    container.appendChild(notification);
  }

  // Get component instance
  getComponent(element) {
    return this.components.get(element);
  }

  // Destroy component
  destroyComponent(element) {
    const component = this.components.get(element);
    if (component && typeof component.destroy === "function") {
      component.destroy();
    }
    this.components.delete(element);
  }

  // Cleanup
  destroy() {
    this.components.forEach((component, element) => {
      this.destroyComponent(element);
    });
    this.components.clear();
  }
}

// Initialize the application when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.PynomalyApp = new PynomalyApp();
});

// Export for module usage
export default PynomalyApp;
