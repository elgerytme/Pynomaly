/**
 * PWA Installation and Update Manager
 * Handles installability, service worker updates, and offline management
 */

class PWAManager {
  constructor() {
    this.deferredPrompt = null;
    this.isInstalled = false;
    this.isOnline = navigator.onLine;
    this.installButton = null;
    this.updateButton = null;
    this.offlineIndicator = null;

    this.init();
  }

  /**
   * Initialize PWA functionality
   */
  async init() {
    await this.checkInstallation();
    this.setupServiceWorker();
    this.setupInstallPrompt();
    this.setupOfflineHandling();
    this.setupUpdateHandling();
    this.createInstallBanner();
    this.setupNotifications();
  }

  /**
   * Check if app is already installed
   */
  async checkInstallation() {
    // Check if running in standalone mode (installed as PWA)
    this.isInstalled =
      window.matchMedia("(display-mode: standalone)").matches ||
      window.navigator.standalone ||
      document.referrer.includes("android-app://");

    console.log(
      "[PWA] Installation status:",
      this.isInstalled ? "Installed" : "Not installed",
    );
  }

  /**
   * Setup service worker registration and handling
   */
  async setupServiceWorker() {
    if ("serviceWorker" in navigator) {
      try {
        const registration = await navigator.serviceWorker.register("/sw.js", {
          scope: "/",
        });

        console.log("[PWA] Service Worker registered:", registration.scope);

        // Handle service worker updates
        registration.addEventListener("updatefound", () => {
          const newWorker = registration.installing;
          newWorker.addEventListener("statechange", () => {
            if (
              newWorker.state === "installed" &&
              navigator.serviceWorker.controller
            ) {
              this.showUpdateAvailable();
            }
          });
        });

        // Listen for messages from service worker
        navigator.serviceWorker.addEventListener("message", (event) => {
          this.handleServiceWorkerMessage(event.data);
        });
      } catch (error) {
        console.error("[PWA] Service Worker registration failed:", error);
      }
    }
  }

  /**
   * Setup install prompt handling
   */
  setupInstallPrompt() {
    // Listen for beforeinstallprompt event
    window.addEventListener("beforeinstallprompt", (event) => {
      console.log("[PWA] Install prompt available");
      event.preventDefault();
      this.deferredPrompt = event;
      this.showInstallBanner();
    });

    // Handle successful installation
    window.addEventListener("appinstalled", () => {
      console.log("[PWA] App installed successfully");
      this.isInstalled = true;
      this.hideInstallBanner();
      this.showInstallSuccessMessage();
    });
  }

  /**
   * Setup offline/online handling
   */
  setupOfflineHandling() {
    window.addEventListener("online", () => {
      this.isOnline = true;
      this.updateOfflineIndicator();
      this.syncQueuedRequests();
    });

    window.addEventListener("offline", () => {
      this.isOnline = false;
      this.updateOfflineIndicator();
      this.showOfflineMessage();
    });

    this.updateOfflineIndicator();
  }

  /**
   * Setup update handling
   */
  setupUpdateHandling() {
    // Check for updates periodically
    setInterval(() => {
      if ("serviceWorker" in navigator) {
        navigator.serviceWorker.getRegistration().then((registration) => {
          if (registration) {
            registration.update();
          }
        });
      }
    }, 60000); // Check every minute
  }

  /**
   * Create install banner UI
   */
  createInstallBanner() {
    if (this.isInstalled) return;

    const banner = document.createElement("div");
    banner.id = "pwa-install-banner";
    banner.className =
      "fixed bottom-4 left-4 right-4 bg-white border border-gray-200 rounded-lg shadow-lg p-4 hidden z-50";
    banner.innerHTML = `
      <div class="flex items-start gap-3">
        <div class="flex-shrink-0">
          <img src="/static/icons/icon-72x72.png" alt="Pynomaly" class="w-12 h-12 rounded-lg">
        </div>
        <div class="flex-grow min-w-0">
          <h3 class="font-semibold text-gray-900">Install Pynomaly</h3>
          <p class="text-sm text-gray-600 mt-1">
            Get faster access and work offline by installing the app.
          </p>
          <div class="flex gap-2 mt-3">
            <button id="pwa-install-button" class="btn-base btn-primary btn-sm">
              Install App
            </button>
            <button id="pwa-install-dismiss" class="btn-base btn-secondary btn-sm">
              Maybe Later
            </button>
          </div>
        </div>
        <button id="pwa-install-close" class="flex-shrink-0 text-gray-400 hover:text-gray-600">
          <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
          </svg>
        </button>
      </div>
    `;

    document.body.appendChild(banner);

    // Setup event listeners
    document
      .getElementById("pwa-install-button")
      .addEventListener("click", () => {
        this.installApp();
      });

    document
      .getElementById("pwa-install-dismiss")
      .addEventListener("click", () => {
        this.dismissInstallBanner();
      });

    document
      .getElementById("pwa-install-close")
      .addEventListener("click", () => {
        this.hideInstallBanner();
      });
  }

  /**
   * Show install banner
   */
  showInstallBanner() {
    if (this.isInstalled) return;

    const banner = document.getElementById("pwa-install-banner");
    if (banner && this.deferredPrompt) {
      // Check if user hasn't dismissed recently
      const dismissedTime = localStorage.getItem("pwa-install-dismissed");
      const oneDayAgo = Date.now() - 24 * 60 * 60 * 1000;

      if (!dismissedTime || parseInt(dismissedTime) < oneDayAgo) {
        banner.classList.remove("hidden");
      }
    }
  }

  /**
   * Hide install banner
   */
  hideInstallBanner() {
    const banner = document.getElementById("pwa-install-banner");
    if (banner) {
      banner.classList.add("hidden");
    }
  }

  /**
   * Dismiss install banner with timestamp
   */
  dismissInstallBanner() {
    this.hideInstallBanner();
    localStorage.setItem("pwa-install-dismissed", Date.now().toString());
  }

  /**
   * Install the app
   */
  async installApp() {
    if (!this.deferredPrompt) {
      console.log("[PWA] No install prompt available");
      return;
    }

    try {
      const result = await this.deferredPrompt.prompt();
      console.log("[PWA] Install prompt result:", result.outcome);

      if (result.outcome === "accepted") {
        console.log("[PWA] User accepted the install prompt");
      } else {
        console.log("[PWA] User dismissed the install prompt");
      }

      this.deferredPrompt = null;
      this.hideInstallBanner();
    } catch (error) {
      console.error("[PWA] Install prompt failed:", error);
    }
  }

  /**
   * Update offline indicator
   */
  updateOfflineIndicator() {
    let indicator = document.getElementById("offline-indicator");

    if (!this.isOnline) {
      if (!indicator) {
        indicator = document.createElement("div");
        indicator.id = "offline-indicator";
        indicator.className =
          "fixed top-4 left-1/2 transform -translate-x-1/2 bg-orange-500 text-white px-4 py-2 rounded-lg shadow-lg z-50";
        indicator.innerHTML = `
          <div class="flex items-center gap-2">
            <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
            </svg>
            <span>You're offline - Some features may be limited</span>
          </div>
        `;
        document.body.appendChild(indicator);
      }
    } else {
      if (indicator) {
        indicator.remove();
      }
    }
  }

  /**
   * Show offline message
   */
  showOfflineMessage() {
    this.showToast(
      "You're now offline. The app will continue to work with cached data.",
      "warning",
    );
  }

  /**
   * Show update available notification
   */
  showUpdateAvailable() {
    const updateBanner = document.createElement("div");
    updateBanner.className =
      "fixed top-4 left-1/2 transform -translate-x-1/2 bg-blue-500 text-white px-6 py-3 rounded-lg shadow-lg z-50";
    updateBanner.innerHTML = `
      <div class="flex items-center gap-3">
        <span>A new version is available!</span>
        <button id="pwa-update-button" class="bg-white text-blue-500 px-3 py-1 rounded text-sm font-medium hover:bg-blue-50">
          Update Now
        </button>
        <button id="pwa-update-dismiss" class="text-blue-100 hover:text-white ml-2">
          <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
          </svg>
        </button>
      </div>
    `;

    document.body.appendChild(updateBanner);

    document
      .getElementById("pwa-update-button")
      .addEventListener("click", () => {
        this.updateApp();
        updateBanner.remove();
      });

    document
      .getElementById("pwa-update-dismiss")
      .addEventListener("click", () => {
        updateBanner.remove();
      });

    // Auto-dismiss after 10 seconds
    setTimeout(() => {
      if (updateBanner.parentNode) {
        updateBanner.remove();
      }
    }, 10000);
  }

  /**
   * Update the app
   */
  async updateApp() {
    if ("serviceWorker" in navigator) {
      const registration = await navigator.serviceWorker.getRegistration();
      if (registration?.waiting) {
        registration.waiting.postMessage({ type: "SKIP_WAITING" });
        window.location.reload();
      }
    }
  }

  /**
   * Show install success message
   */
  showInstallSuccessMessage() {
    this.showToast(
      "Pynomaly has been installed! You can now access it from your home screen.",
      "success",
    );
  }

  /**
   * Setup push notifications
   */
  async setupNotifications() {
    if ("Notification" in window && "serviceWorker" in navigator) {
      // Check if notifications are supported and get permission
      if (Notification.permission === "default") {
        // Don't request permission immediately, wait for user interaction
        this.createNotificationPrompt();
      }
    }
  }

  /**
   * Create notification permission prompt
   */
  createNotificationPrompt() {
    // Only show after user has been active for a while
    setTimeout(() => {
      if (Notification.permission === "default") {
        this.showNotificationPermissionBanner();
      }
    }, 30000); // Show after 30 seconds
  }

  /**
   * Show notification permission banner
   */
  showNotificationPermissionBanner() {
    const banner = document.createElement("div");
    banner.className =
      "fixed bottom-4 right-4 bg-white border border-gray-200 rounded-lg shadow-lg p-4 max-w-sm z-50";
    banner.innerHTML = `
      <div class="flex items-start gap-3">
        <div class="flex-shrink-0">
          <svg class="w-6 h-6 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z"></path>
          </svg>
        </div>
        <div class="flex-grow">
          <h3 class="font-semibold text-gray-900">Stay Updated</h3>
          <p class="text-sm text-gray-600 mt-1">
            Get notified about anomaly detection results and system updates.
          </p>
          <div class="flex gap-2 mt-3">
            <button id="enable-notifications" class="btn-base btn-primary btn-sm">
              Enable Notifications
            </button>
            <button id="dismiss-notifications" class="btn-base btn-secondary btn-sm">
              Not Now
            </button>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(banner);

    document
      .getElementById("enable-notifications")
      .addEventListener("click", async () => {
        await this.requestNotificationPermission();
        banner.remove();
      });

    document
      .getElementById("dismiss-notifications")
      .addEventListener("click", () => {
        banner.remove();
        localStorage.setItem(
          "notification-permission-dismissed",
          Date.now().toString(),
        );
      });
  }

  /**
   * Request notification permission
   */
  async requestNotificationPermission() {
    try {
      const permission = await Notification.requestPermission();

      if (permission === "granted") {
        console.log("[PWA] Notification permission granted");
        this.showToast(
          "Notifications enabled! You'll receive updates about your analysis.",
          "success",
        );

        // Subscribe to push notifications if supported
        await this.subscribeToPushNotifications();
      } else {
        console.log("[PWA] Notification permission denied");
        this.showToast(
          "Notification permission denied. You can enable it later in settings.",
          "info",
        );
      }
    } catch (error) {
      console.error("[PWA] Error requesting notification permission:", error);
    }
  }

  /**
   * Subscribe to push notifications
   */
  async subscribeToPushNotifications() {
    if ("serviceWorker" in navigator && "PushManager" in window) {
      try {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration) {
          const subscription = await registration.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: this.urlBase64ToUint8Array(
              "BPvL1UbMHR8E8VGMbVxlYVqXL4nH-TzkqLONfXJdKKsZfbIQ1B8EQVe7OGO_LIjFwOmFOt9r_cgjEYdLWF8f_A",
            ),
          });

          // Send subscription to server
          await this.sendSubscriptionToServer(subscription);
          console.log("[PWA] Push notification subscription successful");
        }
      } catch (error) {
        console.error("[PWA] Push notification subscription failed:", error);
      }
    }
  }

  /**
   * Send subscription to server
   */
  async sendSubscriptionToServer(subscription) {
    try {
      await fetch("/api/notifications/subscribe", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(subscription),
      });
    } catch (error) {
      console.error("[PWA] Failed to send subscription to server:", error);
    }
  }

  /**
   * Convert VAPID key to Uint8Array
   */
  urlBase64ToUint8Array(base64String) {
    const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
    const base64 = (base64String + padding)
      .replace(/-/g, "+")
      .replace(/_/g, "/");

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  }

  /**
   * Handle messages from service worker
   */
  handleServiceWorkerMessage(data) {
    switch (data.type) {
      case "DETECTION_COMPLETE":
        this.showToast("Anomaly detection completed!", "success", {
          action: {
            label: "View Results",
            url: "/results/" + data.data.requestId,
          },
        });
        break;
      case "UPLOAD_COMPLETE":
        this.showToast("File upload completed!", "success");
        break;
      case "SYNC_ERROR":
        this.showToast(
          "Failed to sync data. Will retry when connection improves.",
          "warning",
        );
        break;
    }
  }

  /**
   * Sync queued requests when coming back online
   */
  async syncQueuedRequests() {
    if (
      "serviceWorker" in navigator &&
      "sync" in window.ServiceWorkerRegistration.prototype
    ) {
      const registration = await navigator.serviceWorker.getRegistration();
      if (registration) {
        await registration.sync.register("sync-queued-requests");
      }
    }
  }

  /**
   * Show toast notification
   */
  showToast(message, type = "info", options = {}) {
    const toast = document.createElement("div");
    const typeClasses = {
      success: "bg-green-500",
      error: "bg-red-500",
      warning: "bg-orange-500",
      info: "bg-blue-500",
    };

    toast.className = `fixed top-4 right-4 ${typeClasses[type]} text-white px-6 py-4 rounded-lg shadow-lg z-50 max-w-sm`;
    toast.innerHTML = `
      <div class="flex items-start gap-3">
        <div class="flex-grow">
          <p class="text-sm">${message}</p>
          ${
            options.action
              ? `
            <button class="mt-2 bg-white bg-opacity-20 hover:bg-opacity-30 px-3 py-1 rounded text-sm"
                    onclick="window.location.href='${options.action.url}'">
              ${options.action.label}
            </button>
          `
              : ""
          }
        </div>
        <button onclick="this.parentElement.parentElement.remove()" class="text-white hover:text-gray-200">
          <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
          </svg>
        </button>
      </div>
    `;

    document.body.appendChild(toast);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (toast.parentNode) {
        toast.remove();
      }
    }, 5000);
  }

  /**
   * Get cache status for debugging
   */
  async getCacheStatus() {
    if ("serviceWorker" in navigator) {
      const registration = await navigator.serviceWorker.getRegistration();
      if (registration) {
        return new Promise((resolve) => {
          const messageChannel = new MessageChannel();
          messageChannel.port1.onmessage = (event) => {
            resolve(event.data);
          };

          registration.active.postMessage({ type: "GET_CACHE_STATUS" }, [
            messageChannel.port2,
          ]);
        });
      }
    }
    return null;
  }
}

// Initialize PWA manager when DOM is loaded
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    window.pwaManager = new PWAManager();
  });
} else {
  window.pwaManager = new PWAManager();
}

// Export for module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = PWAManager;
}
