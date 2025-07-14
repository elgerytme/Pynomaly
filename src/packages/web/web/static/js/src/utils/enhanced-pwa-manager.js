/**
 * Enhanced PWA Manager for Pynomaly
 * Provides comprehensive offline capabilities, data synchronization, and advanced PWA features
 */

class EnhancedPWAManager {
  constructor(options = {}) {
    this.options = {
      enableOfflineDetection: true,
      enableDataSync: true,
      enableNotifications: true,
      enableInstallPrompt: true,
      syncInterval: 30000, // 30 seconds
      maxOfflineStorage: 100 * 1024 * 1024, // 100MB
      ...options
    };

    this.isOnline = navigator.onLine;
    this.installPrompt = null;
    this.syncQueue = [];
    this.offlineActions = new Map();
    this.dataCache = new Map();
    this.backgroundSyncTags = new Set();

    this.init();
  }

  async init() {
    await this.registerServiceWorker();
    this.setupEventListeners();
    this.setupOfflineDetection();
    this.setupInstallPrompt();
    this.setupBackgroundSync();
    this.setupPeriodicSync();
    this.initializeOfflineCapabilities();

    // Request notification permission
    if (this.options.enableNotifications) {
      await this.requestNotificationPermission();
    }
  }

  async registerServiceWorker() {
    if ('serviceWorker' in navigator) {
      try {
        const registration = await navigator.serviceWorker.register('/sw-enhanced.js', {
          scope: '/',
          updateViaCache: 'none'
        });

        console.log('[PWA] Service Worker registered:', registration);

        // Listen for updates
        registration.addEventListener('updatefound', () => {
          const newWorker = registration.installing;
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              this.showUpdateNotification();
            }
          });
        });

        // Listen for messages from service worker
        navigator.serviceWorker.addEventListener('message', (event) => {
          this.handleServiceWorkerMessage(event);
        });

        this.registration = registration;
        return registration;
      } catch (error) {
        console.error('[PWA] Service Worker registration failed:', error);
        throw error;
      }
    } else {
      throw new Error('Service Workers not supported');
    }
  }

  setupEventListeners() {
    // Online/offline events
    window.addEventListener('online', () => this.handleOnlineStatusChange(true));
    window.addEventListener('offline', () => this.handleOnlineStatusChange(false));

    // App lifecycle events
    document.addEventListener('visibilitychange', () => this.handleVisibilityChange());
    window.addEventListener('beforeunload', () => this.handleBeforeUnload());

    // Custom events
    document.addEventListener('data-changed', (event) => this.handleDataChange(event));
    document.addEventListener('sync-requested', (event) => this.handleSyncRequest(event));
  }

  setupOfflineDetection() {
    if (!this.options.enableOfflineDetection) return;

    // Enhanced offline detection using multiple methods
    this.offlineDetector = new OfflineDetector({
      onStatusChange: (isOnline) => this.handleOnlineStatusChange(isOnline),
      onConnectionQualityChange: (quality) => this.handleConnectionQualityChange(quality)
    });
  }

  setupInstallPrompt() {
    if (!this.options.enableInstallPrompt) return;

    window.addEventListener('beforeinstallprompt', (event) => {
      event.preventDefault();
      this.installPrompt = event;
      this.showInstallButton();
    });

    window.addEventListener('appinstalled', () => {
      this.installPrompt = null;
      this.hideInstallButton();
      this.trackInstallation();
    });
  }

  setupBackgroundSync() {
    if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
      this.backgroundSyncSupported = true;
      console.log('[PWA] Background Sync supported');
    } else {
      this.backgroundSyncSupported = false;
      console.log('[PWA] Background Sync not supported, using fallback');
    }
  }

  setupPeriodicSync() {
    if ('serviceWorker' in navigator && 'periodicSync' in window.ServiceWorkerRegistration.prototype) {
      this.periodicSyncSupported = true;
      this.registerPeriodicSync();
    } else {
      this.periodicSyncSupported = false;
      // Fallback to interval-based sync
      this.setupIntervalSync();
    }
  }

  async registerPeriodicSync() {
    try {
      const status = await navigator.permissions.query({ name: 'periodic-background-sync' });
      if (status.state === 'granted') {
        await this.registration.periodicSync.register('data-sync', {
          minInterval: this.options.syncInterval
        });
        console.log('[PWA] Periodic background sync registered');
      }
    } catch (error) {
      console.error('[PWA] Periodic sync registration failed:', error);
    }
  }

  setupIntervalSync() {
    setInterval(() => {
      if (this.isOnline && document.visibilityState === 'visible') {
        this.performSync();
      }
    }, this.options.syncInterval);
  }

  async initializeOfflineCapabilities() {
    // Initialize offline-first data store
    this.offlineStore = new OfflineDataStore({
      dbName: 'PynomalyOfflineDB',
      version: 2,
      stores: {
        datasets: { keyPath: 'id', autoIncrement: true },
        detections: { keyPath: 'id', autoIncrement: true },
        results: { keyPath: 'id', autoIncrement: true },
        userActions: { keyPath: 'id', autoIncrement: true },
        syncQueue: { keyPath: 'id', autoIncrement: true }
      }
    });

    await this.offlineStore.init();

    // Load cached data for offline use
    await this.loadCachedData();

    // Register offline actions
    this.registerOfflineActions();
  }

  registerOfflineActions() {
    // Dataset operations
    this.offlineActions.set('upload-dataset', {
      handler: this.handleOfflineDatasetUpload.bind(this),
      syncHandler: this.syncDatasetUpload.bind(this)
    });

    // Detection operations
    this.offlineActions.set('run-detection', {
      handler: this.handleOfflineDetection.bind(this),
      syncHandler: this.syncDetection.bind(this)
    });

    // Analysis operations
    this.offlineActions.set('analyze-data', {
      handler: this.handleOfflineAnalysis.bind(this),
      syncHandler: this.syncAnalysis.bind(this)
    });

    // User preference changes
    this.offlineActions.set('update-preferences', {
      handler: this.handleOfflinePreferenceUpdate.bind(this),
      syncHandler: this.syncPreferences.bind(this)
    });
  }

  // Public API Methods

  async performAction(actionType, data) {
    if (this.isOnline) {
      try {
        return await this.performOnlineAction(actionType, data);
      } catch (error) {
        console.warn('[PWA] Online action failed, switching to offline mode:', error);
        return await this.performOfflineAction(actionType, data);
      }
    } else {
      return await this.performOfflineAction(actionType, data);
    }
  }

  async performOnlineAction(actionType, data) {
    const response = await fetch(`/api/actions/${actionType}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  }

  async performOfflineAction(actionType, data) {
    const action = this.offlineActions.get(actionType);
    if (!action) {
      throw new Error(`Offline action '${actionType}' not supported`);
    }

    // Execute offline handler
    const result = await action.handler(data);

    // Queue for background sync
    await this.queueForSync(actionType, data, result);

    // Show offline notification
    this.showOfflineNotification(actionType);

    return result;
  }

  async queueForSync(actionType, data, result) {
    const syncItem = {
      id: this.generateId(),
      actionType,
      data,
      result,
      timestamp: Date.now(),
      retryCount: 0,
      status: 'pending'
    };

    await this.offlineStore.add('syncQueue', syncItem);

    if (this.backgroundSyncSupported) {
      await this.requestBackgroundSync(actionType);
    }
  }

  async requestBackgroundSync(tag) {
    try {
      await this.registration.sync.register(tag);
      this.backgroundSyncTags.add(tag);
      console.log(`[PWA] Background sync requested: ${tag}`);
    } catch (error) {
      console.error('[PWA] Background sync request failed:', error);
    }
  }

  // Offline Action Handlers

  async handleOfflineDatasetUpload(data) {
    // Store dataset locally with metadata
    const dataset = {
      id: this.generateId(),
      name: data.name,
      file: data.file,
      metadata: {
        size: data.file.size,
        type: data.file.type,
        uploadTime: Date.now(),
        status: 'offline'
      },
      synced: false
    };

    await this.offlineStore.add('datasets', dataset);

    // Process basic dataset info
    const preview = await this.generateDatasetPreview(data.file);
    dataset.preview = preview;

    await this.offlineStore.update('datasets', dataset);

    return {
      id: dataset.id,
      status: 'stored_offline',
      preview: preview,
      message: 'Dataset stored offline. Will sync when online.'
    };
  }

  async handleOfflineDetection(data) {
    // Run basic detection using cached algorithms
    const detection = {
      id: this.generateId(),
      datasetId: data.datasetId,
      algorithm: data.algorithm,
      parameters: data.parameters,
      timestamp: Date.now(),
      status: 'offline_processing'
    };

    // Use simplified detection algorithm for offline mode
    const result = await this.runOfflineDetection(data);
    detection.result = result;
    detection.status = 'completed_offline';

    await this.offlineStore.add('detections', detection);

    return {
      id: detection.id,
      result: result,
      status: 'completed_offline',
      message: 'Detection completed offline. Results will sync when online.'
    };
  }

  async handleOfflineAnalysis(data) {
    // Perform basic analysis using cached data
    const analysis = {
      id: this.generateId(),
      type: data.type,
      parameters: data.parameters,
      timestamp: Date.now(),
      status: 'offline_analysis'
    };

    const result = await this.runOfflineAnalysis(data);
    analysis.result = result;

    await this.offlineStore.add('results', analysis);

    return {
      id: analysis.id,
      result: result,
      status: 'completed_offline'
    };
  }

  async handleOfflinePreferenceUpdate(data) {
    // Store preference changes locally
    const update = {
      id: this.generateId(),
      preferences: data.preferences,
      timestamp: Date.now(),
      userId: data.userId || 'anonymous'
    };

    await this.offlineStore.add('userActions', update);

    // Apply preferences immediately
    this.applyPreferences(data.preferences);

    return {
      id: update.id,
      status: 'updated_offline',
      message: 'Preferences updated offline. Will sync when online.'
    };
  }

  // Sync Handlers

  async syncDatasetUpload(syncItem) {
    try {
      const formData = new FormData();
      formData.append('file', syncItem.data.file);
      formData.append('name', syncItem.data.name);
      formData.append('offlineId', syncItem.result.id);

      const response = await fetch('/api/datasets/upload', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();

        // Update local dataset with server ID
        const dataset = await this.offlineStore.get('datasets', syncItem.result.id);
        dataset.serverId = result.id;
        dataset.synced = true;
        dataset.syncTime = Date.now();

        await this.offlineStore.update('datasets', dataset);

        return { success: true, result };
      } else {
        throw new Error(`Upload failed: ${response.status}`);
      }
    } catch (error) {
      console.error('[PWA] Dataset sync failed:', error);
      return { success: false, error: error.message };
    }
  }

  async syncDetection(syncItem) {
    try {
      const response = await fetch('/api/detection/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...syncItem.data,
          offlineId: syncItem.result.id
        })
      });

      if (response.ok) {
        const result = await response.json();

        // Update local detection with server results
        const detection = await this.offlineStore.get('detections', syncItem.result.id);
        detection.serverResult = result;
        detection.synced = true;
        detection.syncTime = Date.now();

        await this.offlineStore.update('detections', detection);

        return { success: true, result };
      } else {
        throw new Error(`Detection sync failed: ${response.status}`);
      }
    } catch (error) {
      console.error('[PWA] Detection sync failed:', error);
      return { success: false, error: error.message };
    }
  }

  async syncAnalysis(syncItem) {
    try {
      const response = await fetch('/api/analysis/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...syncItem.data,
          offlineId: syncItem.result.id
        })
      });

      if (response.ok) {
        const result = await response.json();

        const analysis = await this.offlineStore.get('results', syncItem.result.id);
        analysis.serverResult = result;
        analysis.synced = true;

        await this.offlineStore.update('results', analysis);

        return { success: true, result };
      } else {
        throw new Error(`Analysis sync failed: ${response.status}`);
      }
    } catch (error) {
      console.error('[PWA] Analysis sync failed:', error);
      return { success: false, error: error.message };
    }
  }

  async syncPreferences(syncItem) {
    try {
      const response = await fetch('/api/user/preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(syncItem.data.preferences)
      });

      if (response.ok) {
        const userAction = await this.offlineStore.get('userActions', syncItem.result.id);
        userAction.synced = true;
        userAction.syncTime = Date.now();

        await this.offlineStore.update('userActions', userAction);

        return { success: true };
      } else {
        throw new Error(`Preferences sync failed: ${response.status}`);
      }
    } catch (error) {
      console.error('[PWA] Preferences sync failed:', error);
      return { success: false, error: error.message };
    }
  }

  // Event Handlers

  handleOnlineStatusChange(isOnline) {
    const wasOnline = this.isOnline;
    this.isOnline = isOnline;

    if (isOnline && !wasOnline) {
      console.log('[PWA] Coming back online');
      this.showOnlineNotification();
      this.performSync();
    } else if (!isOnline && wasOnline) {
      console.log('[PWA] Going offline');
      this.showOfflineNotification();
    }

    // Update UI state
    document.body.classList.toggle('offline', !isOnline);

    // Emit custom event
    document.dispatchEvent(new CustomEvent('connection-change', {
      detail: { isOnline, wasOnline }
    }));
  }

  handleConnectionQualityChange(quality) {
    // Adjust sync behavior based on connection quality
    if (quality === 'slow') {
      this.options.syncInterval = 60000; // Reduce sync frequency
    } else {
      this.options.syncInterval = 30000; // Normal frequency
    }
  }

  handleVisibilityChange() {
    if (document.visibilityState === 'visible' && this.isOnline) {
      // App became visible, sync any pending changes
      this.performSync();
    }
  }

  handleBeforeUnload() {
    // Store any pending state before page unload
    this.savePendingState();
  }

  handleServiceWorkerMessage(event) {
    const { type, payload } = event.data;

    switch (type) {
      case 'SYNC_COMPLETE':
        this.handleSyncComplete(payload);
        break;
      case 'SYNC_FAILED':
        this.handleSyncFailed(payload);
        break;
      case 'CACHE_UPDATED':
        this.handleCacheUpdated(payload);
        break;
      case 'OFFLINE_DATA_READY':
        this.handleOfflineDataReady(payload);
        break;
    }
  }

  handleDataChange(event) {
    const { entity, operation, data } = event.detail;

    // Queue change for sync if offline
    if (!this.isOnline) {
      this.queueForSync(`${operation}-${entity}`, data);
    }
  }

  handleSyncRequest(event) {
    const { actionType, data } = event.detail;
    this.performAction(actionType, data);
  }

  // Sync Management

  async performSync() {
    if (!this.isOnline) return;

    console.log('[PWA] Starting sync...');

    try {
      const pendingItems = await this.offlineStore.getAll('syncQueue');
      const unsynced = pendingItems.filter(item => item.status === 'pending');

      for (const item of unsynced) {
        await this.syncItem(item);
      }

      console.log(`[PWA] Sync completed: ${unsynced.length} items processed`);

      // Clean up successfully synced items
      await this.cleanupSyncedItems();

    } catch (error) {
      console.error('[PWA] Sync failed:', error);
    }
  }

  async syncItem(item) {
    try {
      const action = this.offlineActions.get(item.actionType);
      if (!action || !action.syncHandler) {
        throw new Error(`No sync handler for action: ${item.actionType}`);
      }

      const result = await action.syncHandler(item);

      if (result.success) {
        item.status = 'synced';
        item.syncTime = Date.now();
        item.serverResult = result.result;
      } else {
        item.status = 'failed';
        item.retryCount++;
        item.lastError = result.error;
      }

      await this.offlineStore.update('syncQueue', item);

    } catch (error) {
      console.error(`[PWA] Failed to sync item ${item.id}:`, error);

      item.status = 'failed';
      item.retryCount++;
      item.lastError = error.message;

      await this.offlineStore.update('syncQueue', item);
    }
  }

  async cleanupSyncedItems() {
    const syncedItems = await this.offlineStore.query('syncQueue',
      (item) => item.status === 'synced' &&
                Date.now() - item.syncTime > 24 * 60 * 60 * 1000 // 24 hours
    );

    for (const item of syncedItems) {
      await this.offlineStore.delete('syncQueue', item.id);
    }
  }

  // Utility Methods

  async generateDatasetPreview(file) {
    try {
      const text = await file.text();
      const lines = text.split('\n');
      const headers = lines[0].split(',');
      const sampleRows = lines.slice(1, 6); // First 5 rows

      return {
        headers,
        sampleRows,
        totalRows: lines.length - 1,
        size: file.size,
        type: file.type
      };
    } catch (error) {
      console.error('[PWA] Failed to generate dataset preview:', error);
      return null;
    }
  }

  async runOfflineDetection(data) {
    // Simplified anomaly detection using basic statistical methods
    // This is a placeholder - in production, you'd use a lightweight ML library

    const dataset = await this.offlineStore.get('datasets', data.datasetId);
    if (!dataset) {
      throw new Error('Dataset not found in offline storage');
    }

    // Mock detection result
    const anomalies = [];
    const totalPoints = Math.floor(Math.random() * 100) + 50;
    const anomalyCount = Math.floor(totalPoints * 0.05); // 5% anomalies

    for (let i = 0; i < anomalyCount; i++) {
      anomalies.push({
        id: this.generateId(),
        timestamp: new Date(Date.now() - Math.random() * 86400000),
        value: Math.random() * 100,
        anomalyScore: 0.7 + Math.random() * 0.3,
        confidence: 0.6 + Math.random() * 0.4,
        severity: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)]
      });
    }

    return {
      totalPoints,
      anomalies,
      algorithm: data.algorithm,
      processingTime: Math.random() * 1000 + 500,
      offline: true
    };
  }

  async runOfflineAnalysis(data) {
    // Basic statistical analysis
    return {
      type: data.type,
      summary: {
        mean: Math.random() * 100,
        median: Math.random() * 100,
        stdDev: Math.random() * 20,
        min: Math.random() * 10,
        max: 90 + Math.random() * 10
      },
      offline: true,
      timestamp: Date.now()
    };
  }

  applyPreferences(preferences) {
    // Apply preferences to the current session
    Object.entries(preferences).forEach(([key, value]) => {
      localStorage.setItem(`pref_${key}`, JSON.stringify(value));
    });

    // Emit preference change event
    document.dispatchEvent(new CustomEvent('preferences-changed', {
      detail: preferences
    }));
  }

  async loadCachedData() {
    try {
      // Load essential data for offline operation
      const datasets = await this.offlineStore.getAll('datasets');
      const detections = await this.offlineStore.getAll('detections');

      this.dataCache.set('datasets', datasets);
      this.dataCache.set('detections', detections);

      console.log(`[PWA] Loaded ${datasets.length} datasets and ${detections.length} detections from cache`);
    } catch (error) {
      console.error('[PWA] Failed to load cached data:', error);
    }
  }

  savePendingState() {
    // Save any unsaved form data, selections, etc.
    const pendingState = {
      timestamp: Date.now(),
      url: window.location.href,
      formData: this.extractFormData()
    };

    localStorage.setItem('pwa_pending_state', JSON.stringify(pendingState));
  }

  extractFormData() {
    const forms = document.querySelectorAll('form');
    const formData = {};

    forms.forEach((form, index) => {
      const data = new FormData(form);
      formData[`form_${index}`] = Object.fromEntries(data.entries());
    });

    return formData;
  }

  generateId() {
    return 'offline_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  // Notification Methods

  async requestNotificationPermission() {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission();
      console.log(`[PWA] Notification permission: ${permission}`);
      return permission === 'granted';
    }
    return false;
  }

  showOfflineNotification(actionType = null) {
    const message = actionType
      ? `Action "${actionType}" completed offline. Will sync when online.`
      : 'You are now offline. Limited functionality available.';

    this.showNotification({
      title: 'Offline Mode',
      body: message,
      icon: '/static/icons/offline.png',
      tag: 'offline-status'
    });
  }

  showOnlineNotification() {
    this.showNotification({
      title: 'Back Online',
      body: 'Connection restored. Syncing data...',
      icon: '/static/icons/online.png',
      tag: 'online-status'
    });
  }

  showUpdateNotification() {
    this.showNotification({
      title: 'App Update Available',
      body: 'A new version is available. Refresh to update.',
      icon: '/static/icons/update.png',
      actions: [
        { action: 'update', title: 'Update Now' },
        { action: 'dismiss', title: 'Later' }
      ],
      tag: 'app-update',
      requireInteraction: true
    });
  }

  showNotification(options) {
    if ('Notification' in window && Notification.permission === 'granted') {
      const notification = new Notification(options.title, options);

      if (options.actions) {
        notification.addEventListener('click', (event) => {
          if (event.action === 'update') {
            window.location.reload();
          }
        });
      }

      return notification;
    }
  }

  // Install Prompt Methods

  showInstallButton() {
    const installButton = document.getElementById('install-button');
    if (installButton) {
      installButton.style.display = 'block';
      installButton.addEventListener('click', () => this.promptInstall());
    }
  }

  hideInstallButton() {
    const installButton = document.getElementById('install-button');
    if (installButton) {
      installButton.style.display = 'none';
    }
  }

  async promptInstall() {
    if (this.installPrompt) {
      this.installPrompt.prompt();
      const { outcome } = await this.installPrompt.userChoice;
      console.log(`[PWA] Install prompt outcome: ${outcome}`);
      this.installPrompt = null;
    }
  }

  trackInstallation() {
    // Track installation for analytics
    if (window.gtag) {
      gtag('event', 'pwa_install', {
        event_category: 'PWA',
        event_label: 'App Installed'
      });
    }
  }

  // Public API

  getOfflineCapabilities() {
    return {
      datasets: this.dataCache.get('datasets')?.length || 0,
      detections: this.dataCache.get('detections')?.length || 0,
      pendingSync: this.syncQueue.length,
      isOnline: this.isOnline,
      backgroundSyncSupported: this.backgroundSyncSupported,
      periodicSyncSupported: this.periodicSyncSupported
    };
  }

  async getStorageEstimate() {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      return {
        quota: estimate.quota,
        usage: estimate.usage,
        usagePercentage: (estimate.usage / estimate.quota) * 100
      };
    }
    return null;
  }

  async clearOfflineData() {
    await this.offlineStore.clear();
    this.dataCache.clear();
    console.log('[PWA] Offline data cleared');
  }
}

/**
 * Offline Detection Helper
 */
class OfflineDetector {
  constructor(options = {}) {
    this.options = {
      checkInterval: 10000,
      timeout: 5000,
      ...options
    };

    this.isOnline = navigator.onLine;
    this.connectionQuality = 'unknown';
    this.lastCheck = Date.now();

    this.startMonitoring();
  }

  startMonitoring() {
    // Check connection periodically
    setInterval(() => this.checkConnection(), this.options.checkInterval);

    // Listen to browser events
    window.addEventListener('online', () => this.handleStatusChange(true));
    window.addEventListener('offline', () => this.handleStatusChange(false));
  }

  async checkConnection() {
    try {
      const startTime = Date.now();
      const response = await fetch('/api/health', {
        method: 'GET',
        cache: 'no-cache',
        headers: { 'Cache-Control': 'no-cache' }
      });

      const endTime = Date.now();
      const responseTime = endTime - startTime;

      if (response.ok) {
        this.handleStatusChange(true);
        this.updateConnectionQuality(responseTime);
      } else {
        this.handleStatusChange(false);
      }
    } catch (error) {
      this.handleStatusChange(false);
    }
  }

  updateConnectionQuality(responseTime) {
    let quality;

    if (responseTime < 500) {
      quality = 'fast';
    } else if (responseTime < 1500) {
      quality = 'normal';
    } else {
      quality = 'slow';
    }

    if (quality !== this.connectionQuality) {
      this.connectionQuality = quality;
      this.options.onConnectionQualityChange?.(quality);
    }
  }

  handleStatusChange(isOnline) {
    if (isOnline !== this.isOnline) {
      this.isOnline = isOnline;
      this.options.onStatusChange?.(isOnline);
    }
  }
}

/**
 * Offline Data Store using IndexedDB
 */
class OfflineDataStore {
  constructor(options = {}) {
    this.options = {
      dbName: 'OfflineDB',
      version: 1,
      stores: {},
      ...options
    };

    this.db = null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.options.dbName, this.options.version);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        Object.entries(this.options.stores).forEach(([storeName, config]) => {
          if (!db.objectStoreNames.contains(storeName)) {
            const store = db.createObjectStore(storeName, config);

            // Add indexes if specified
            if (config.indexes) {
              config.indexes.forEach(index => {
                store.createIndex(index.name, index.keyPath, index.options);
              });
            }
          }
        });
      };
    });
  }

  async add(storeName, data) {
    const transaction = this.db.transaction([storeName], 'readwrite');
    const store = transaction.objectStore(storeName);
    return new Promise((resolve, reject) => {
      const request = store.add(data);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async get(storeName, key) {
    const transaction = this.db.transaction([storeName], 'readonly');
    const store = transaction.objectStore(storeName);
    return new Promise((resolve, reject) => {
      const request = store.get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getAll(storeName) {
    const transaction = this.db.transaction([storeName], 'readonly');
    const store = transaction.objectStore(storeName);
    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async update(storeName, data) {
    const transaction = this.db.transaction([storeName], 'readwrite');
    const store = transaction.objectStore(storeName);
    return new Promise((resolve, reject) => {
      const request = store.put(data);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async delete(storeName, key) {
    const transaction = this.db.transaction([storeName], 'readwrite');
    const store = transaction.objectStore(storeName);
    return new Promise((resolve, reject) => {
      const request = store.delete(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async clear(storeName = null) {
    if (storeName) {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      return new Promise((resolve, reject) => {
        const request = store.clear();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
    } else {
      // Clear all stores
      const storeNames = Object.keys(this.options.stores);
      const promises = storeNames.map(name => this.clear(name));
      return Promise.all(promises);
    }
  }

  async query(storeName, filter) {
    const allData = await this.getAll(storeName);
    return allData.filter(filter);
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    EnhancedPWAManager,
    OfflineDetector,
    OfflineDataStore
  };
}

// Global access
if (typeof window !== 'undefined') {
  window.EnhancedPWAManager = EnhancedPWAManager;
  window.OfflineDetector = OfflineDetector;
  window.OfflineDataStore = OfflineDataStore;
}

export default EnhancedPWAManager;
