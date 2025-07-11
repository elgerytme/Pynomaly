/**
 * Enhanced PWA Manager - Complete Progressive Web App Implementation
 * Handles installation, offline capabilities, background sync, and push notifications
 */

class EnhancedPWAManager {
  constructor(options = {}) {
    this.options = {
      enableInstallPrompt: true,
      enablePushNotifications: true,
      enableBackgroundSync: true,
      enableOfflineAnalytics: true,
      cacheVersion: 'v1.2.0',
      offlinePagePath: '/offline',
      manifestPath: '/manifest.json',
      maxOfflineStorage: 50 * 1024 * 1024, // 50MB
      syncRetryInterval: 30000, // 30 seconds
      ...options
    };

    this.serviceWorker = null;
    this.deferredPrompt = null;
    this.isInstalled = false;
    this.isOnline = navigator.onLine;
    this.offlineQueue = [];
    this.syncInProgress = false;
    this.notificationPermission = 'default';
    this.installListeners = new Set();
    this.offlineListeners = new Set();
    this.syncListeners = new Set();

    this.init();
  }

  async init() {
    await this.registerServiceWorker();
    this.setupInstallPrompt();
    this.setupNetworkHandling();
    this.setupBackgroundSync();
    this.setupPushNotifications();
    this.setupOfflineAnalytics();
    this.checkInstallationStatus();
    this.setupAppBadge();
    this.setupPeriodicSync();
  }

  async registerServiceWorker() {
    if (!('serviceWorker' in navigator)) {
      console.warn('[PWA] Service Workers not supported');
      return false;
    }

    try {
      console.log('[PWA] Registering service worker...');
      
      const registration = await navigator.serviceWorker.register('/static/sw.js', {
        scope: '/',
        updateViaCache: 'none'
      });

      this.serviceWorker = registration;
      
      // Handle updates
      registration.addEventListener('updatefound', () => {
        this.handleServiceWorkerUpdate(registration);
      });

      // Handle controlling changes
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        console.log('[PWA] Service worker controller changed');
        window.location.reload();
      });

      // Setup message handling
      navigator.serviceWorker.addEventListener('message', (event) => {
        this.handleServiceWorkerMessage(event);
      });

      console.log('[PWA] Service worker registered successfully');
      return true;
    } catch (error) {
      console.error('[PWA] Service worker registration failed:', error);
      return false;
    }
  }

  handleServiceWorkerUpdate(registration) {
    const newWorker = registration.installing;
    
    newWorker.addEventListener('statechange', () => {
      if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
        console.log('[PWA] New service worker available');
        this.showUpdateNotification();
      }
    });
  }

  showUpdateNotification() {
    const notification = this.createNotificationElement({
      title: 'App Update Available',
      message: 'A new version of Pynomaly is available. Refresh to update.',
      actions: [
        {
          text: 'Update Now',
          primary: true,
          action: () => this.updateServiceWorker()
        },
        {
          text: 'Later',
          action: () => this.dismissNotification()
        }
      ]
    });

    document.body.appendChild(notification);
  }

  updateServiceWorker() {
    if (this.serviceWorker && this.serviceWorker.waiting) {
      this.serviceWorker.waiting.postMessage({ type: 'SKIP_WAITING' });
    }
  }

  setupInstallPrompt() {
    if (!this.options.enableInstallPrompt) return;

    // Listen for beforeinstallprompt event
    window.addEventListener('beforeinstallprompt', (e) => {
      console.log('[PWA] Install prompt available');
      e.preventDefault();
      this.deferredPrompt = e;
      this.showInstallButton();
    });

    // Listen for app installed event
    window.addEventListener('appinstalled', () => {
      console.log('[PWA] App installed');
      this.isInstalled = true;
      this.hideInstallButton();
      this.notifyInstallListeners('installed');
    });
  }

  showInstallButton() {
    const installButton = document.getElementById('pwa-install-button') || 
                         this.createInstallButton();
    
    installButton.style.display = 'block';
    installButton.addEventListener('click', () => this.promptInstall());
  }

  createInstallButton() {
    const button = document.createElement('button');
    button.id = 'pwa-install-button';
    button.className = 'pwa-install-btn';
    button.innerHTML = `
      <span class="install-icon">📱</span>
      <span class="install-text">Install App</span>
    `;
    button.style.display = 'none';
    
    // Add to appropriate location
    const header = document.querySelector('header') || document.querySelector('nav');
    if (header) {
      header.appendChild(button);
    } else {
      document.body.appendChild(button);
    }
    
    return button;
  }

  hideInstallButton() {
    const installButton = document.getElementById('pwa-install-button');
    if (installButton) {
      installButton.style.display = 'none';
    }
  }

  async promptInstall() {
    if (!this.deferredPrompt) {
      console.log('[PWA] Install prompt not available');
      return false;
    }

    try {
      this.deferredPrompt.prompt();
      const result = await this.deferredPrompt.userChoice;
      
      console.log('[PWA] Install prompt result:', result.outcome);
      
      if (result.outcome === 'accepted') {
        this.notifyInstallListeners('accepted');
      } else {
        this.notifyInstallListeners('dismissed');
      }
      
      this.deferredPrompt = null;
      return result.outcome === 'accepted';
    } catch (error) {
      console.error('[PWA] Install prompt failed:', error);
      return false;
    }
  }

  setupNetworkHandling() {
    // Online/offline detection
    window.addEventListener('online', () => {
      console.log('[PWA] Network online');
      this.isOnline = true;
      this.handleNetworkChange(true);
    });

    window.addEventListener('offline', () => {
      console.log('[PWA] Network offline');
      this.isOnline = false;
      this.handleNetworkChange(false);
    });

    // Setup network status indicator
    this.createNetworkStatusIndicator();
  }

  handleNetworkChange(isOnline) {
    this.updateNetworkStatusIndicator(isOnline);
    
    if (isOnline) {
      this.processOfflineQueue();
      this.syncOfflineData();
    } else {
      this.enableOfflineMode();
    }

    // Notify listeners
    this.notifyOfflineListeners(isOnline ? 'online' : 'offline');
  }

  createNetworkStatusIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'network-status-indicator';
    indicator.className = 'network-status-indicator';
    indicator.innerHTML = `
      <div class="status-dot"></div>
      <span class="status-text">Online</span>
    `;
    
    document.body.appendChild(indicator);
    this.updateNetworkStatusIndicator(this.isOnline);
  }

  updateNetworkStatusIndicator(isOnline) {
    const indicator = document.getElementById('network-status-indicator');
    if (!indicator) return;

    const dot = indicator.querySelector('.status-dot');
    const text = indicator.querySelector('.status-text');
    
    if (isOnline) {
      dot.className = 'status-dot online';
      text.textContent = 'Online';
      indicator.classList.remove('offline');
    } else {
      dot.className = 'status-dot offline';
      text.textContent = 'Offline';
      indicator.classList.add('offline');
    }
  }

  enableOfflineMode() {
    // Enable offline functionality
    document.body.classList.add('offline-mode');
    
    // Show offline banner
    this.showOfflineBanner();
    
    // Enable offline data loading
    this.loadOfflineData();
  }

  showOfflineBanner() {
    const banner = document.createElement('div');
    banner.id = 'offline-banner';
    banner.className = 'offline-banner';
    banner.innerHTML = `
      <div class="banner-content">
        <span class="banner-icon">📡</span>
        <span class="banner-text">You're offline. Some features may be limited.</span>
        <button class="banner-close" onclick="this.parentElement.parentElement.remove()">×</button>
      </div>
    `;
    
    document.body.insertBefore(banner, document.body.firstChild);
  }

  async loadOfflineData() {
    if (!this.serviceWorker) return;

    try {
      // Request offline data from service worker
      const channel = new MessageChannel();
      
      return new Promise((resolve) => {
        channel.port1.onmessage = (event) => {
          const offlineData = event.data;
          this.populateOfflineUI(offlineData);
          resolve(offlineData);
        };

        navigator.serviceWorker.controller.postMessage({
          type: 'GET_OFFLINE_DASHBOARD_DATA'
        }, [channel.port2]);
      });
    } catch (error) {
      console.error('[PWA] Failed to load offline data:', error);
    }
  }

  populateOfflineUI(data) {
    // Populate dashboard with offline data
    window.dispatchEvent(new CustomEvent('offline-data-loaded', {
      detail: data
    }));
  }

  setupBackgroundSync() {
    if (!this.options.enableBackgroundSync || !this.serviceWorker) return;

    // Setup sync event listeners
    this.setupSyncEventListeners();
    
    // Start periodic sync check
    this.startPeriodicSyncCheck();
  }

  setupSyncEventListeners() {
    // Listen for sync requests from the app
    window.addEventListener('sync-request', (event) => {
      this.queueSyncRequest(event.detail);
    });

    // Listen for form submissions in offline mode
    document.addEventListener('submit', (event) => {
      if (!this.isOnline) {
        this.handleOfflineFormSubmission(event);
      }
    });
  }

  handleOfflineFormSubmission(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const syncData = {
      type: 'form_submission',
      url: form.action,
      method: form.method,
      data: Object.fromEntries(formData),
      timestamp: Date.now()
    };

    this.queueSyncRequest(syncData);
    this.showOfflineSubmissionNotification();
  }

  queueSyncRequest(syncData) {
    this.offlineQueue.push(syncData);
    
    // Store in service worker for persistence
    if (navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'QUEUE_REQUEST',
        payload: {
          request: syncData,
          tag: this.determineSyncTag(syncData.type)
        }
      });
    }
  }

  determineSyncTag(type) {
    const syncTags = {
      'form_submission': 'upload-queue',
      'detection_request': 'detection-queue',
      'analysis_request': 'analysis-queue',
      'metric_data': 'metrics-sync'
    };
    
    return syncTags[type] || 'general-sync';
  }

  startPeriodicSyncCheck() {
    setInterval(() => {
      if (this.isOnline && !this.syncInProgress) {
        this.processOfflineQueue();
      }
    }, this.options.syncRetryInterval);
  }

  async processOfflineQueue() {
    if (this.offlineQueue.length === 0 || this.syncInProgress) return;

    this.syncInProgress = true;
    this.notifySyncListeners('sync_started');

    try {
      for (const syncData of this.offlineQueue) {
        await this.processSyncItem(syncData);
      }
      
      this.offlineQueue = [];
      this.notifySyncListeners('sync_completed');
    } catch (error) {
      console.error('[PWA] Sync failed:', error);
      this.notifySyncListeners('sync_failed');
    } finally {
      this.syncInProgress = false;
    }
  }

  async processSyncItem(syncData) {
    try {
      const response = await fetch(syncData.url, {
        method: syncData.method || 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(syncData.data)
      });

      if (response.ok) {
        console.log('[PWA] Sync item processed:', syncData.type);
      } else {
        throw new Error(`Sync failed: ${response.status}`);
      }
    } catch (error) {
      console.error('[PWA] Failed to process sync item:', error);
      throw error;
    }
  }

  async syncOfflineData() {
    if (!navigator.serviceWorker.controller) return;

    navigator.serviceWorker.controller.postMessage({
      type: 'SYNC_ALL_QUEUES'
    });
  }

  setupPushNotifications() {
    if (!this.options.enablePushNotifications || !('Notification' in window)) return;

    this.checkNotificationPermission();
    this.setupNotificationSubscription();
  }

  async checkNotificationPermission() {
    this.notificationPermission = Notification.permission;
    
    if (this.notificationPermission === 'default') {
      this.showNotificationPrompt();
    }
  }

  showNotificationPrompt() {
    const prompt = this.createNotificationElement({
      title: 'Enable Notifications',
      message: 'Get real-time alerts for anomaly detections and system updates.',
      actions: [
        {
          text: 'Enable',
          primary: true,
          action: () => this.requestNotificationPermission()
        },
        {
          text: 'Not Now',
          action: () => this.dismissNotification()
        }
      ]
    });

    document.body.appendChild(prompt);
  }

  async requestNotificationPermission() {
    try {
      const permission = await Notification.requestPermission();
      this.notificationPermission = permission;
      
      if (permission === 'granted') {
        await this.subscribeToNotifications();
      }
      
      this.dismissNotification();
    } catch (error) {
      console.error('[PWA] Notification permission request failed:', error);
    }
  }

  async subscribeToNotifications() {
    if (!this.serviceWorker) return;

    try {
      const subscription = await this.serviceWorker.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: await this.getVAPIDKey()
      });

      // Send subscription to server
      await this.sendSubscriptionToServer(subscription);
      
      console.log('[PWA] Push notification subscription successful');
    } catch (error) {
      console.error('[PWA] Push notification subscription failed:', error);
    }
  }

  async getVAPIDKey() {
    // Get VAPID public key from server
    try {
      const response = await fetch('/api/vapid-key');
      const data = await response.json();
      return this.urlBase64ToUint8Array(data.publicKey);
    } catch (error) {
      console.error('[PWA] Failed to get VAPID key:', error);
      return null;
    }
  }

  urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
      .replace(/-/g, '+')
      .replace(/_/g, '/');

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  }

  async sendSubscriptionToServer(subscription) {
    try {
      const response = await fetch('/api/push-subscription', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(subscription)
      });

      if (!response.ok) {
        throw new Error('Failed to send subscription to server');
      }
    } catch (error) {
      console.error('[PWA] Failed to send subscription to server:', error);
    }
  }

  setupOfflineAnalytics() {
    if (!this.options.enableOfflineAnalytics) return;

    // Track offline usage
    this.trackOfflineEvents();
    
    // Setup offline analytics sync
    this.setupOfflineAnalyticsSync();
  }

  trackOfflineEvents() {
    // Track page views in offline mode
    window.addEventListener('offline-page-view', (event) => {
      this.recordOfflineEvent('page_view', event.detail);
    });

    // Track user interactions in offline mode
    document.addEventListener('click', (event) => {
      if (!this.isOnline) {
        this.recordOfflineEvent('offline_interaction', {
          target: event.target.tagName,
          timestamp: Date.now()
        });
      }
    });
  }

  recordOfflineEvent(type, data) {
    const event = {
      type,
      data,
      timestamp: Date.now(),
      url: window.location.href
    };

    // Store offline event
    if (navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'UPDATE_LOCAL_DATA',
        payload: {
          entityType: 'analytics',
          entityId: `${type}_${Date.now()}`,
          data: event
        }
      });
    }
  }

  setupOfflineAnalyticsSync() {
    // Sync analytics data when online
    window.addEventListener('online', () => {
      this.syncOfflineAnalytics();
    });
  }

  async syncOfflineAnalytics() {
    if (!navigator.serviceWorker.controller) return;

    navigator.serviceWorker.controller.postMessage({
      type: 'SYNC_ALL_QUEUES'
    });
  }

  checkInstallationStatus() {
    // Check if app is installed
    if (window.matchMedia('(display-mode: standalone)').matches) {
      this.isInstalled = true;
      console.log('[PWA] App is installed');
    }

    // Check for iOS standalone mode
    if (window.navigator.standalone === true) {
      this.isInstalled = true;
      console.log('[PWA] App is installed (iOS)');
    }
  }

  setupAppBadge() {
    if (!('setAppBadge' in navigator)) return;

    // Listen for badge updates
    window.addEventListener('app-badge-update', (event) => {
      this.updateAppBadge(event.detail.count);
    });
  }

  async updateAppBadge(count) {
    try {
      if (count > 0) {
        await navigator.setAppBadge(count);
      } else {
        await navigator.clearAppBadge();
      }
    } catch (error) {
      console.error('[PWA] Failed to update app badge:', error);
    }
  }

  setupPeriodicSync() {
    if (!('serviceWorker' in navigator) || !('sync' in window.ServiceWorkerRegistration.prototype)) {
      return;
    }

    // Register periodic background sync
    navigator.serviceWorker.ready.then(registration => {
      if ('periodicSync' in registration) {
        registration.periodicSync.register('background-data-sync', {
          minInterval: 24 * 60 * 60 * 1000 // 24 hours
        }).catch(error => {
          console.error('[PWA] Periodic sync registration failed:', error);
        });
      }
    });
  }

  // Service Worker Message Handling
  handleServiceWorkerMessage(event) {
    const { type, data } = event.data;

    switch (type) {
      case 'OFFLINE_DATA_LOADED':
        this.populateOfflineUI(data.data);
        break;
      case 'SYNC_QUEUE_UPDATE':
        this.notifySyncListeners('queue_updated', data.queue);
        break;
      case 'DETECTION_COMPLETE':
        this.handleOfflineDetectionComplete(data);
        break;
      case 'CACHE_STATUS':
        this.handleCacheStatus(data);
        break;
    }
  }

  handleOfflineDetectionComplete(data) {
    // Show notification for completed offline detection
    this.showNotification({
      title: 'Detection Complete',
      body: 'Your offline anomaly detection has finished processing.',
      actions: [
        {
          action: 'view',
          title: 'View Results'
        }
      ]
    });
  }

  handleCacheStatus(data) {
    console.log('[PWA] Cache status:', data);
    
    // Update storage usage indicator
    this.updateStorageUsage(data);
  }

  updateStorageUsage(cacheStatus) {
    // Update UI with storage information
    const usage = {
      used: cacheStatus.totalSize || 0,
      available: this.options.maxOfflineStorage,
      percentage: (cacheStatus.totalSize / this.options.maxOfflineStorage) * 100
    };

    window.dispatchEvent(new CustomEvent('storage-usage-update', {
      detail: usage
    }));
  }

  // Utility methods
  createNotificationElement(options) {
    const notification = document.createElement('div');
    notification.className = 'pwa-notification';
    notification.innerHTML = `
      <div class="notification-content">
        <h3 class="notification-title">${options.title}</h3>
        <p class="notification-message">${options.message}</p>
        <div class="notification-actions">
          ${options.actions.map(action => `
            <button class="notification-btn ${action.primary ? 'primary' : 'secondary'}" 
                    data-action="${action.action || action.text}">
              ${action.text}
            </button>
          `).join('')}
        </div>
      </div>
    `;

    // Add event listeners
    notification.querySelectorAll('.notification-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const actionText = e.target.dataset.action;
        const action = options.actions.find(a => (a.action || a.text) === actionText);
        if (action && action.action) {
          action.action();
        }
      });
    });

    return notification;
  }

  dismissNotification() {
    const notifications = document.querySelectorAll('.pwa-notification');
    notifications.forEach(notification => {
      notification.remove();
    });
  }

  showOfflineSubmissionNotification() {
    const notification = this.createNotificationElement({
      title: 'Form Saved',
      message: 'Your submission has been saved and will be sent when you\'re back online.',
      actions: [
        {
          text: 'OK',
          primary: true,
          action: () => this.dismissNotification()
        }
      ]
    });

    document.body.appendChild(notification);
  }

  // Event listener management
  addInstallListener(callback) {
    this.installListeners.add(callback);
    return () => this.installListeners.delete(callback);
  }

  addOfflineListener(callback) {
    this.offlineListeners.add(callback);
    return () => this.offlineListeners.delete(callback);
  }

  addSyncListener(callback) {
    this.syncListeners.add(callback);
    return () => this.syncListeners.delete(callback);
  }

  notifyInstallListeners(event, data) {
    this.installListeners.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error('[PWA] Install listener failed:', error);
      }
    });
  }

  notifyOfflineListeners(event, data) {
    this.offlineListeners.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error('[PWA] Offline listener failed:', error);
      }
    });
  }

  notifySyncListeners(event, data) {
    this.syncListeners.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error('[PWA] Sync listener failed:', error);
      }
    });
  }

  // Public API
  async getInstallationStatus() {
    return {
      isInstalled: this.isInstalled,
      canInstall: !!this.deferredPrompt,
      isOnline: this.isOnline
    };
  }

  async getCacheStatus() {
    if (!navigator.serviceWorker.controller) return null;

    return new Promise((resolve) => {
      const channel = new MessageChannel();
      
      channel.port1.onmessage = (event) => {
        resolve(event.data);
      };

      navigator.serviceWorker.controller.postMessage({
        type: 'GET_CACHE_STATUS'
      }, [channel.port2]);
    });
  }

  async clearCache() {
    if (!navigator.serviceWorker.controller) return false;

    navigator.serviceWorker.controller.postMessage({
      type: 'CLEAR_CACHE',
      payload: { cacheName: 'all' }
    });

    return true;
  }

  // Static methods
  static getInstance(options) {
    if (!EnhancedPWAManager.instance) {
      EnhancedPWAManager.instance = new EnhancedPWAManager(options);
    }
    return EnhancedPWAManager.instance;
  }

  static destroyInstance() {
    if (EnhancedPWAManager.instance) {
      EnhancedPWAManager.instance = null;
    }
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = EnhancedPWAManager;
}

// Global access
if (typeof window !== 'undefined') {
  window.EnhancedPWAManager = EnhancedPWAManager;
}

export default EnhancedPWAManager;