/**
 * Pynomaly Dashboard JavaScript
 * Enhanced UI interactions and real-time updates
 */

class PynomalyDashboard {
  constructor() {
    this.init();
  }

  init() {
    this.setupRealTimeUpdates();
    this.setupNavigationDropdowns();
    this.setupToastNotifications();
    this.updateLastRefreshTime();
  }

  setupRealTimeUpdates() {
    // Update last refresh time every minute
    setInterval(() => {
      this.updateLastRefreshTime();
    }, 60000);

    // Auto-refresh dashboard data every 5 minutes
    setInterval(() => {
      this.refreshDashboardData();
    }, 300000);
  }

  setupNavigationDropdowns() {
    // Enhanced dropdown behavior for mobile
    document.addEventListener("click", (e) => {
      const dropdown = e.target.closest(".group");
      if (dropdown && window.innerWidth <= 768) {
        const menu = dropdown.querySelector(".nav-dropdown");
        if (menu) {
          menu.classList.toggle("show");
        }
      }
    });
  }

  setupToastNotifications() {
    // Create toast container if it doesn't exist
    if (!document.getElementById("toast-container")) {
      const container = document.createElement("div");
      container.id = "toast-container";
      container.className = "fixed top-4 right-4 z-50 space-y-2";
      document.body.appendChild(container);
    }
  }

  showToast(message, type = "info", duration = 5000) {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");

    const colors = {
      success: "bg-green-500",
      error: "bg-red-500",
      warning: "bg-yellow-500",
      info: "bg-blue-500",
    };

    toast.className = `${colors[type]} text-white px-4 py-2 rounded-lg shadow-lg transform transition-all duration-300 translate-x-full opacity-0`;
    toast.textContent = message;

    container.appendChild(toast);

    // Animate in
    setTimeout(() => {
      toast.classList.remove("translate-x-full", "opacity-0");
    }, 100);

    // Animate out and remove
    setTimeout(() => {
      toast.classList.add("translate-x-full", "opacity-0");
      setTimeout(() => {
        if (container.contains(toast)) {
          container.removeChild(toast);
        }
      }, 300);
    }, duration);
  }

  updateLastRefreshTime() {
    const element = document.getElementById("last-updated");
    if (element) {
      const now = new Date();
      const timeString = now.toLocaleTimeString();
      element.textContent = timeString;
    }
  }

  refreshDashboardData() {
    // Use HTMX to refresh dashboard components
    if (window.htmx) {
      console.log("Refreshing dashboard data...");
      htmx.trigger("#results-table", "htmx:trigger");
      this.showToast("Dashboard refreshed", "success", 3000);
    }
  }

  // Enhanced metric card animations
  animateMetricCard(element, newValue) {
    const valueElement = element.querySelector(".text-2xl");
    if (valueElement && valueElement.textContent !== newValue) {
      // Add pulse animation
      element.classList.add("animate-pulse");
      setTimeout(() => {
        valueElement.textContent = newValue;
        element.classList.remove("animate-pulse");
      }, 500);
    }
  }

  // Theme toggle functionality
  toggleTheme() {
    const body = document.body;
    const isDark = body.classList.contains("dark");

    if (isDark) {
      body.classList.remove("dark");
      localStorage.setItem("theme", "light");
      this.showToast("Switched to light mode", "info");
    } else {
      body.classList.add("dark");
      localStorage.setItem("theme", "dark");
      this.showToast("Switched to dark mode", "info");
    }
  }

  // Initialize theme from localStorage
  initTheme() {
    const savedTheme = localStorage.getItem("theme");
    const prefersDark = window.matchMedia(
      "(prefers-color-scheme: dark)",
    ).matches;

    if (savedTheme === "dark" || (!savedTheme && prefersDark)) {
      document.body.classList.add("dark");
    }
  }

  // Enhanced error handling for HTMX requests
  setupHTMXErrorHandling() {
    document.addEventListener("htmx:responseError", (e) => {
      this.showToast("Network error occurred", "error");
    });

    document.addEventListener("htmx:timeout", (e) => {
      this.showToast("Request timed out", "warning");
    });

    document.addEventListener("htmx:afterSwap", (e) => {
      // Re-initialize any components in the swapped content
      this.reinitializeComponents(e.detail.target);
    });
  }

  reinitializeComponents(container) {
    // Re-setup any event listeners or components in the new content
    const metricCards = container.querySelectorAll(".metric-card");
    metricCards.forEach((card) => {
      card.addEventListener("mouseenter", () => {
        card.style.transform = "translateY(-2px)";
      });
      card.addEventListener("mouseleave", () => {
        card.style.transform = "translateY(0)";
      });
    });
  }
}

// Initialize dashboard when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  const dashboard = new PynomalyDashboard();

  // Make dashboard instance globally accessible
  window.PynomalyDashboard = dashboard;

  // Setup theme toggle button if it exists
  const themeToggle = document.querySelector("[data-theme-toggle]");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      dashboard.toggleTheme();
    });
  }

  // Initialize theme
  dashboard.initTheme();

  // Setup HTMX error handling
  dashboard.setupHTMXErrorHandling();

  console.log("Pynomaly Dashboard initialized successfully");
});

// Export for module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = PynomalyDashboard;
}
