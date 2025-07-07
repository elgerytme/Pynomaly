/**
 * Advanced PWA Manager - Enhanced Progressive Web App functionality
 * Features: Install prompts, offline data management, sync status, update handling
 */
export class PWAManager {
  constructor() {
    this.deferredPrompt = null;
    this.isInstalled = false;
    this.isOnline = navigator.onLine;
    this.syncQueue = [];
    this.offlineData = {
      datasets: [],
      results: [],
      preferences: {},
    };

    this.init();
  }

  async init() {
    await this.setupServiceWorker();
    this.setupInstallPrompt();
    this.setupUpdateHandling();
    this.setupOfflineHandling();
    this.checkInstallStatus();
    this.setupConnectionMonitoring();
    await this.loadOfflineData();
  }

  /**
   * Service Worker Setup and Communication
   */
  async setupServiceWorker() {
    if ("serviceWorker" in navigator) {
      try {
        const registration =
          await navigator.serviceWorker.register("/static/sw.js");
        console.log("[PWA] Service Worker registered:", registration.scope);

        // Listen for service worker messages
        navigator.serviceWorker.addEventListener("message", (event) => {
          this.handleServiceWorkerMessage(event.data);
        });

        // Handle updates
        registration.addEventListener("updatefound", () => {
          this.handleServiceWorkerUpdate(registration.installing);
        });

        this.registration = registration;
      } catch (error) {
        console.error("[PWA] Service Worker registration failed:", error);
      }
    }
  }

  handleServiceWorkerMessage(data) {
    const { type, payload } = data;

    switch (type) {
      case "DETECTION_COMPLETE":
        this.onDetectionComplete(payload);
        break;
      case "SYNC_QUEUE_STATUS":
        this.updateSyncStatus(payload);
        break;
      case "CACHE_STATUS":
        this.updateCacheStatus(payload);
        break;
      case "OFFLINE_DATA_UPDATED":
        this.loadOfflineData();
        break;
    }
  }

  handleServiceWorkerUpdate(worker) {
    worker.addEventListener("statechange", () => {
      if (worker.state === "installed") {
        this.showUpdateNotification();
      }
    });
  }

  /**
   * Install Prompt Management
   */
  setupInstallPrompt() {
    window.addEventListener("beforeinstallprompt", (e) => {
      e.preventDefault();
      this.deferredPrompt = e;
      this.showInstallButton();
    });

    // Handle successful installation
    window.addEventListener("appinstalled", () => {
      this.isInstalled = true;
      this.hideInstallButton();
      this.showInstallSuccessMessage();
      this.deferredPrompt = null;
    });
  }

  showInstallButton() {
    // Create install button if it doesn't exist
    if (!document.querySelector(".pwa-install-button")) {
      const installButton = this.createInstallButton();
      this.addInstallButtonToPage(installButton);
    }
  }

  createInstallButton() {
    const button = document.createElement("button");
    button.className =
      "pwa-install-button fixed bottom-4 right-4 bg-primary text-white px-4 py-2 rounded-lg shadow-lg hover:bg-blue-700 transition-colors z-50 flex items-center gap-2";
    button.innerHTML = `
      <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd"></path>
      </svg>
      Install App
    `;
    button.addEventListener("click", () => this.installPWA());
    return button;
  }

  addInstallButtonToPage(button) {
    document.body.appendChild(button);

    // Animate in
    setTimeout(() => {
      button.style.transform = "translateY(0)";
      button.style.opacity = "1";
    }, 100);
  }

  hideInstallButton() {
    const button = document.querySelector(".pwa-install-button");
    if (button) {
      button.style.transform = "translateY(100px)";
      button.style.opacity = "0";
      setTimeout(() => button.remove(), 300);
    }
  }

  async installPWA() {
    if (this.deferredPrompt) {
      this.deferredPrompt.prompt();
      const result = await this.deferredPrompt.userChoice;

      if (result.outcome === "accepted") {
        console.log("[PWA] User accepted the install prompt");
        this.trackEvent("pwa_install_accepted");
      } else {
        console.log("[PWA] User dismissed the install prompt");
        this.trackEvent("pwa_install_dismissed");
      }

      this.deferredPrompt = null;
    }
  }

  showInstallSuccessMessage() {
    this.showNotification(
      "📱 App installed successfully! You can now use Pynomaly offline.",
      "success",
    );
  }

  /**
   * Update Handling
   */
  setupUpdateHandling() {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (this.refreshing) return;
        this.refreshing = true;
        window.location.reload();
      });
    }
  }

  showUpdateNotification() {
    const notification = this.createNotification(
      "🔄 New version available! Click to update.",
      "info",
      [
        { text: "Update Now", action: () => this.updateApp() },
        { text: "Later", action: () => this.dismissUpdate() },
      ],
    );
    document.body.appendChild(notification);
  }

  async updateApp() {
    if (this.registration && this.registration.waiting) {
      this.registration.waiting.postMessage({ type: "SKIP_WAITING" });
    }
  }

  dismissUpdate() {
    const notification = document.querySelector(".pwa-update-notification");
    if (notification) notification.remove();
  }

  /**
   * Offline Data Management
   */
  setupOfflineHandling() {
    window.addEventListener("online", () => {
      this.isOnline = true;
      this.onConnectionRestore();
    });

    window.addEventListener("offline", () => {
      this.isOnline = false;
      this.onConnectionLost();
    });
  }

  async loadOfflineData() {
    try {
      if (this.registration && this.registration.active) {
        // Request data from service worker
        this.registration.active.postMessage({ type: "GET_OFFLINE_DATA" });
      }
    } catch (error) {
      console.error("[PWA] Failed to load offline data:", error);
    }
  }

  async saveDataOffline(type, data) {
    try {
      if (this.registration && this.registration.active) {
        this.registration.active.postMessage({
          type: "SAVE_OFFLINE_DATA",
          payload: { type, data },
        });
      }
    } catch (error) {
      console.error("[PWA] Failed to save offline data:", error);
    }
  }

  /**
   * Connection Monitoring
   */
  setupConnectionMonitoring() {
    // Update UI connection status
    this.updateConnectionUI();

    // Periodic connectivity check
    setInterval(() => {
      this.checkConnectivity();
    }, 30000); // Check every 30 seconds
  }

  updateConnectionUI() {
    const indicators = document.querySelectorAll(".connection-indicator");
    indicators.forEach((indicator) => {
      indicator.className = `connection-indicator ${this.isOnline ? "online" : "offline"}`;
    });

    const statusTexts = document.querySelectorAll(".connection-status");
    statusTexts.forEach((text) => {
      text.textContent = this.isOnline ? "Online" : "Offline";
      text.className = `connection-status ${this.isOnline ? "text-green-600" : "text-orange-600"}`;
    });
  }

  async checkConnectivity() {
    try {
      const response = await fetch("/api/health/ping", {
        method: "HEAD",
        mode: "no-cors",
        cache: "no-cache",
      });

      if (!this.isOnline && navigator.onLine) {
        this.isOnline = true;
        this.onConnectionRestore();
      }
    } catch (error) {
      if (this.isOnline && !navigator.onLine) {
        this.isOnline = false;
        this.onConnectionLost();
      }
    }
  }

  onConnectionRestore() {
    console.log("[PWA] Connection restored");
    this.updateConnectionUI();
    this.syncPendingData();
    this.showNotification("🌐 Connection restored! Syncing data...", "success");
  }

  onConnectionLost() {
    console.log("[PWA] Connection lost");
    this.updateConnectionUI();
    this.showNotification(
      "📡 You're now offline. Changes will sync when connection returns.",
      "warning",
    );
  }

  /**
   * Data Synchronization
   */
  async syncPendingData() {
    if (this.registration && this.registration.active) {
      // Trigger background sync for all queued requests
      this.registration.active.postMessage({ type: "SYNC_ALL_QUEUES" });

      // Request sync status update
      setTimeout(() => {
        this.registration.active.postMessage({ type: "GET_SYNC_STATUS" });
      }, 1000);
    }
  }

  updateSyncStatus(status) {
    const pendingCount = status.pending || 0;

    // Update sync indicators in UI
    const syncIndicators = document.querySelectorAll(".sync-indicator");
    syncIndicators.forEach((indicator) => {
      indicator.textContent =
        pendingCount > 0 ? `${pendingCount} pending` : "Up to date";
      indicator.className = `sync-indicator ${pendingCount > 0 ? "text-orange-600" : "text-green-600"}`;
    });
  }

  /**
   * Cache Management
   */
  async getCacheInfo() {
    if (this.registration && this.registration.active) {
      return new Promise((resolve) => {
        const channel = new MessageChannel();
        channel.port1.onmessage = (event) => resolve(event.data);

        this.registration.active.postMessage({ type: "GET_CACHE_STATUS" }, [
          channel.port2,
        ]);
      });
    }
    return null;
  }

  async clearCache(cacheName = null) {
    if (this.registration && this.registration.active) {
      this.registration.active.postMessage({
        type: "CLEAR_CACHE",
        payload: { cacheName },
      });
    }
  }

  /**
   * Installation Status
   */
  checkInstallStatus() {
    // Check if running in standalone mode
    if (window.matchMedia("(display-mode: standalone)").matches) {
      this.isInstalled = true;
    }

    // iOS Safari check
    if (window.navigator.standalone === true) {
      this.isInstalled = true;
    }

    // Android TWA check
    if (document.referrer.includes("android-app://")) {
      this.isInstalled = true;
    }

    if (this.isInstalled) {
      this.onInstallDetected();
    }
  }

  onInstallDetected() {
    document.body.classList.add("pwa-installed");
    this.trackEvent("pwa_running_installed");
  }

  /**
   * Event Handlers
   */
  onDetectionComplete(payload) {
    this.showNotification(
      `✅ Detection completed: ${payload.result.n_anomalies} anomalies found`,
      "success",
    );
  }

  /**
   * UI Utilities
   */
  showNotification(message, type = "info", actions = []) {
    const notification = this.createNotification(message, type, actions);
    document.body.appendChild(notification);

    // Auto-dismiss after 5 seconds if no actions
    if (actions.length === 0) {
      setTimeout(() => {
        if (notification.parentNode) {
          notification.remove();
        }
      }, 5000);
    }
  }

  createNotification(message, type, actions = []) {
    const notification = document.createElement("div");
    notification.className = `pwa-notification fixed top-4 right-4 max-w-sm bg-white border rounded-lg shadow-lg z-50 transform transition-all duration-300`;

    const colors = {
      success: "border-green-200 bg-green-50",
      warning: "border-orange-200 bg-orange-50",
      error: "border-red-200 bg-red-50",
      info: "border-blue-200 bg-blue-50",
    };

    notification.className += ` ${colors[type] || colors.info}`;

    const actionsHtml =
      actions.length > 0
        ? `
      <div class="mt-3 flex gap-2">
        ${actions
          .map(
            (action) => `
          <button class="px-3 py-1 text-sm bg-white border rounded hover:bg-gray-50 transition-colors"
                  onclick="this.closest('.pwa-notification').remove(); (${action.action.toString()})()">
            ${action.text}
          </button>
        `,
          )
          .join("")}
      </div>
    `
        : "";

    notification.innerHTML = `
      <div class="p-4">
        <div class="flex justify-between items-start">
          <div class="flex-1">
            <p class="text-sm font-medium text-gray-900">${message}</p>
            ${actionsHtml}
          </div>
          <button onclick="this.closest('.pwa-notification').remove()" 
                  class="ml-3 text-gray-400 hover:text-gray-600">
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
            </svg>
          </button>
        </div>
      </div>
    `;

    // Animate in
    setTimeout(() => {
      notification.style.transform = "translateX(0)";
    }, 100);

    return notification;
  }

  /**
   * Analytics and Tracking
   */
  trackEvent(eventName, data = {}) {
    // Send analytics event
    if (typeof gtag !== "undefined") {
      gtag("event", eventName, data);
    }

    console.log(`[PWA] Event: ${eventName}`, data);
  }

  /**
   * Public API Methods
   */
  isAppInstalled() {
    return this.isInstalled;
  }

  isAppOnline() {
    return this.isOnline;
  }

  async getAppStatus() {
    const cacheInfo = await this.getCacheInfo();
    return {
      installed: this.isInstalled,
      online: this.isOnline,
      serviceWorkerActive: !!this.registration?.active,
      cacheInfo: cacheInfo,
    };
  }
}

// Initialize PWA Manager
if (typeof window !== "undefined") {
  window.PWAManager = new PWAManager();

  // Expose useful methods globally
  window.PWA = {
    install: () => window.PWAManager.installPWA(),
    getStatus: () => window.PWAManager.getAppStatus(),
    clearCache: (name) => window.PWAManager.clearCache(name),
    sync: () => window.PWAManager.syncPendingData(),
  };
}
