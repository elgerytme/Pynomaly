/**
 * Background Sync Manager
 *
 * Enhanced background sync management for PWA with queue handling,
 * retry logic, and conflict resolution for offline operations.
 */

export class BackgroundSyncManager {
  constructor(options = {}) {
    this.options = {
      maxRetries: 3,
      retryDelay: 1000, // Base delay in ms
      exponentialBackoff: true,
      maxRetryDelay: 30000, // 30 seconds max
      enableLogging: true,
      queueStoreName: 'sync_queue',
      dbName: 'PynomalyOfflineDB',
      dbVersion: 2,
      enablePeriodicSync: false,
      periodicSyncInterval: 24 * 60 * 60 * 1000, // 24 hours
      enableConflictResolution: true,
      ...options
    };

    this.db = null;
    this.syncQueues = new Map();
    this.isOnline = navigator.onLine;
    this.isProcessing = false;
    this.processingPromise = null;

    this.init();
  }

  async init() {
    await this.openDatabase();
    this.setupEventListeners();
    this.registerSyncEvents();

    // Initial queue processing if online
    if (this.isOnline) {
      this.processAllQueues();
    }

    this.log('Background Sync Manager initialized');
  }

  /**
   * Database Operations
   */
  async openDatabase() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.options.dbName, this.options.dbVersion);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(request.result);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        if (!db.objectStoreNames.contains(this.options.queueStoreName)) {
          const store = db.createObjectStore(this.options.queueStoreName, {
            keyPath: 'id',
            autoIncrement: true
          });

          store.createIndex('tag', 'tag');
          store.createIndex('status', 'status');
          store.createIndex('priority', 'priority');
          store.createIndex('timestamp', 'timestamp');
          store.createIndex('retryCount', 'retryCount');
        }
      };
    });
  }

  /**
   * Event Listeners Setup
   */
  setupEventListeners() {
    // Network status monitoring
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.log('Network restored - processing sync queues');
      this.processAllQueues();
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
      this.log('Network lost - queuing operations for sync');
    });

    // Service Worker message handling
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.addEventListener('message', (event) => {
        this.handleServiceWorkerMessage(event.data);
      });
    }

    // Page visibility change
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && this.isOnline) {
        this.processAllQueues();
      }
    });
  }

  /**
   * Service Worker Sync Registration
   */
  async registerSyncEvents() {
    if (!('serviceWorker' in navigator) || !('sync' in window.ServiceWorkerRegistration.prototype)) {
      this.log('Background sync not supported');
      return;
    }

    try {
      const registration = await navigator.serviceWorker.ready;

      // Register sync events for different types
      await registration.sync.register('detection-sync');
      await registration.sync.register('upload-sync');
      await registration.sync.register('analysis-sync');
      await registration.sync.register('preferences-sync');

      // Register periodic sync if supported and enabled
      if (this.options.enablePeriodicSync && 'periodicSync' in registration) {
        await registration.periodicSync.register('periodic-data-sync', {
          minInterval: this.options.periodicSyncInterval
        });
        this.log('Periodic sync registered');
      }

      this.log('Background sync events registered');
    } catch (error) {
      this.log('Failed to register sync events:', error);
    }
  }

  /**
   * Queue Management
   */
  async addToQueue(tag, data, options = {}) {
    const queueItem = {
      tag,
      data,
      status: 'pending',
      priority: options.priority || 'normal',
      timestamp: Date.now(),
      retryCount: 0,
      maxRetries: options.maxRetries || this.options.maxRetries,
      metadata: {
        userAgent: navigator.userAgent,
        url: window.location.href,
        ...options.metadata
      }
    };

    try {
      const transaction = this.db.transaction([this.options.queueStoreName], 'readwrite');
      const store = transaction.objectStore(this.options.queueStoreName);

      const id = await this.addToStore(store, queueItem);
      queueItem.id = id;

      this.log(`Added to queue [${tag}]:`, queueItem.id);

      // Try immediate sync if online
      if (this.isOnline) {
        this.processQueue(tag);
      }

      return id;
    } catch (error) {
      this.log('Failed to add to queue:', error);
      throw error;
    }
  }

  async getQueue(tag = null) {
    try {
      const transaction = this.db.transaction([this.options.queueStoreName], 'readonly');
      const store = transaction.objectStore(this.options.queueStoreName);

      if (tag) {
        const index = store.index('tag');
        return await this.getAllFromIndex(index, tag);
      } else {
        return await this.getAllFromStore(store);
      }
    } catch (error) {
      this.log('Failed to get queue:', error);
      return [];
    }
  }

  async updateQueueItem(id, updates) {
    try {
      const transaction = this.db.transaction([this.options.queueStoreName], 'readwrite');
      const store = transaction.objectStore(this.options.queueStoreName);

      const existing = await this.getFromStore(store, id);
      if (!existing) {
        throw new Error(`Queue item ${id} not found`);
      }

      const updated = { ...existing, ...updates, lastModified: Date.now() };
      await this.updateInStore(store, updated);

      this.log(`Updated queue item ${id}:`, updates);
      return updated;
    } catch (error) {
      this.log('Failed to update queue item:', error);
      throw error;
    }
  }

  async removeFromQueue(id) {
    try {
      const transaction = this.db.transaction([this.options.queueStoreName], 'readwrite');
      const store = transaction.objectStore(this.options.queueStoreName);

      await this.deleteFromStore(store, id);
      this.log(`Removed from queue: ${id}`);
    } catch (error) {
      this.log('Failed to remove from queue:', error);
      throw error;
    }
  }

  async clearQueue(tag = null) {
    try {
      if (tag) {
        const items = await this.getQueue(tag);
        for (const item of items) {
          await this.removeFromQueue(item.id);
        }
      } else {
        const transaction = this.db.transaction([this.options.queueStoreName], 'readwrite');
        const store = transaction.objectStore(this.options.queueStoreName);
        await store.clear();
      }

      this.log(`Cleared queue${tag ? ` [${tag}]` : ''}`);
    } catch (error) {
      this.log('Failed to clear queue:', error);
      throw error;
    }
  }

  /**
   * Queue Processing
   */
  async processAllQueues() {
    if (this.isProcessing) {
      return this.processingPromise;
    }

    this.isProcessing = true;
    this.processingPromise = this._processAllQueuesInternal();

    try {
      await this.processingPromise;
    } finally {
      this.isProcessing = false;
      this.processingPromise = null;
    }
  }

  async _processAllQueuesInternal() {
    if (!this.isOnline) {
      this.log('Offline - skipping queue processing');
      return;
    }

    try {
      const allItems = await this.getQueue();
      const pendingItems = allItems.filter(item =>
        item.status === 'pending' && item.retryCount < item.maxRetries
      );

      if (pendingItems.length === 0) {
        this.log('No pending items to process');
        return;
      }

      // Group by tag for organized processing
      const itemsByTag = pendingItems.reduce((groups, item) => {
        if (!groups[item.tag]) {
          groups[item.tag] = [];
        }
        groups[item.tag].push(item);
        return groups;
      }, {});

      this.log(`Processing ${pendingItems.length} items across ${Object.keys(itemsByTag).length} queues`);

      // Process each queue
      for (const [tag, items] of Object.entries(itemsByTag)) {
        await this.processQueueItems(tag, items);
      }

      this.log('Queue processing complete');
    } catch (error) {
      this.log('Error processing queues:', error);
    }
  }

  async processQueue(tag) {
    try {
      const items = await this.getQueue(tag);
      const pendingItems = items.filter(item =>
        item.status === 'pending' && item.retryCount < item.maxRetries
      );

      if (pendingItems.length > 0) {
        await this.processQueueItems(tag, pendingItems);
      }
    } catch (error) {
      this.log(`Error processing queue [${tag}]:`, error);
    }
  }

  async processQueueItems(tag, items) {
    this.log(`Processing ${items.length} items for tag [${tag}]`);

    // Sort by priority and timestamp
    items.sort((a, b) => {
      const priorityOrder = { high: 3, normal: 2, low: 1 };
      const aPriority = priorityOrder[a.priority] || 2;
      const bPriority = priorityOrder[b.priority] || 2;

      if (aPriority !== bPriority) {
        return bPriority - aPriority; // Higher priority first
      }
      return a.timestamp - b.timestamp; // Older first
    });

    for (const item of items) {
      try {
        await this.updateQueueItem(item.id, { status: 'processing' });

        const success = await this.syncItem(item);

        if (success) {
          await this.updateQueueItem(item.id, {
            status: 'completed',
            completedAt: Date.now()
          });

          // Optionally remove completed items after delay
          setTimeout(() => this.removeFromQueue(item.id), 60000); // 1 minute
        } else {
          await this.handleSyncFailure(item);
        }
      } catch (error) {
        this.log(`Error processing item ${item.id}:`, error);
        await this.handleSyncFailure(item);
      }
    }
  }

  async syncItem(item) {
    this.log(`Syncing item ${item.id} [${item.tag}]`);

    try {
      const result = await this.performSync(item);

      if (result.success) {
        this.log(`Successfully synced item ${item.id}`);
        this.notifyUI('sync_success', { item, result });
        return true;
      } else {
        this.log(`Failed to sync item ${item.id}:`, result.error);
        return false;
      }
    } catch (error) {
      this.log(`Exception syncing item ${item.id}:`, error);
      return false;
    }
  }

  async performSync(item) {
    const { tag, data } = item;

    switch (tag) {
      case 'detection-sync':
        return await this.syncDetection(data);

      case 'upload-sync':
        return await this.syncUpload(data);

      case 'analysis-sync':
        return await this.syncAnalysis(data);

      case 'preferences-sync':
        return await this.syncPreferences(data);

      default:
        return await this.syncGeneric(item);
    }
  }

  async syncDetection(data) {
    try {
      const response = await fetch('/api/detection/background-sync', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`
        },
        body: JSON.stringify(data)
      });

      if (response.ok) {
        const result = await response.json();
        return { success: true, result };
      } else {
        const error = await response.text();
        return { success: false, error };
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async syncUpload(data) {
    try {
      const formData = new FormData();

      if (data.file) {
        formData.append('file', data.file);
      }

      if (data.metadata) {
        formData.append('metadata', JSON.stringify(data.metadata));
      }

      const response = await fetch('/api/uploads/background-sync', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.getAuthToken()}`
        },
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        return { success: true, result };
      } else {
        const error = await response.text();
        return { success: false, error };
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async syncAnalysis(data) {
    try {
      const response = await fetch('/api/analysis/background-sync', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`
        },
        body: JSON.stringify(data)
      });

      if (response.ok) {
        const result = await response.json();
        return { success: true, result };
      } else {
        const error = await response.text();
        return { success: false, error };
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async syncPreferences(data) {
    try {
      const response = await fetch('/api/user/preferences/sync', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`
        },
        body: JSON.stringify(data)
      });

      if (response.ok) {
        const result = await response.json();
        return { success: true, result };
      } else {
        const error = await response.text();
        return { success: false, error };
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async syncGeneric(item) {
    this.log(`No specific sync handler for tag [${item.tag}] - using generic`);

    try {
      const response = await fetch('/api/sync/generic', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`
        },
        body: JSON.stringify(item)
      });

      if (response.ok) {
        const result = await response.json();
        return { success: true, result };
      } else {
        const error = await response.text();
        return { success: false, error };
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  /**
   * Failure Handling and Retry Logic
   */
  async handleSyncFailure(item) {
    const newRetryCount = item.retryCount + 1;

    if (newRetryCount >= item.maxRetries) {
      // Max retries reached - mark as failed
      await this.updateQueueItem(item.id, {
        status: 'failed',
        retryCount: newRetryCount,
        failedAt: Date.now()
      });

      this.log(`Item ${item.id} failed after ${newRetryCount} attempts`);
      this.notifyUI('sync_failed', { item, reason: 'max_retries_exceeded' });
    } else {
      // Schedule retry with exponential backoff
      const delay = this.calculateRetryDelay(newRetryCount);

      await this.updateQueueItem(item.id, {
        status: 'pending',
        retryCount: newRetryCount,
        nextRetryAt: Date.now() + delay
      });

      this.log(`Scheduled retry for item ${item.id} in ${delay}ms (attempt ${newRetryCount})`);

      // Schedule retry
      setTimeout(() => {
        if (this.isOnline) {
          this.processQueue(item.tag);
        }
      }, delay);
    }
  }

  calculateRetryDelay(retryCount) {
    if (!this.options.exponentialBackoff) {
      return this.options.retryDelay;
    }

    const delay = this.options.retryDelay * Math.pow(2, retryCount - 1);
    return Math.min(delay, this.options.maxRetryDelay);
  }

  /**
   * Conflict Resolution
   */
  async resolveConflicts(localData, serverData, conflictType) {
    if (!this.options.enableConflictResolution) {
      return localData; // Default to local changes
    }

    switch (conflictType) {
      case 'timestamp':
        // Use most recent timestamp
        return localData.timestamp > serverData.timestamp ? localData : serverData;

      case 'user_preference':
        // User preferences: local wins
        return localData;

      case 'data_upload':
        // Data uploads: merge or prompt user
        return await this.mergeDataUploads(localData, serverData);

      default:
        // Default strategy: prompt user or use local
        return await this.promptUserForResolution(localData, serverData);
    }
  }

  async mergeDataUploads(local, server) {
    // Simple merge strategy - combine unique entries
    const merged = { ...server };

    if (local.data && server.data) {
      merged.data = [...new Set([...server.data, ...local.data])];
    }

    merged.mergedAt = Date.now();
    merged.mergeStrategy = 'auto_combine';

    return merged;
  }

  async promptUserForResolution(local, server) {
    // In a real implementation, this would show a UI prompt
    // For now, return local data
    this.log('Conflict detected - defaulting to local data');
    return local;
  }

  /**
   * Service Worker Communication
   */
  handleServiceWorkerMessage(data) {
    const { type, payload } = data;

    switch (type) {
      case 'SYNC_COMPLETE':
        this.handleSyncComplete(payload);
        break;

      case 'SYNC_FAILED':
        this.handleSyncFailed(payload);
        break;

      case 'QUEUE_UPDATED':
        this.notifyUI('queue_updated', payload);
        break;
    }
  }

  handleSyncComplete(payload) {
    this.log('Service Worker sync complete:', payload);
    this.notifyUI('sw_sync_complete', payload);

    // Refresh queue status
    this.getQueueStatus().then(status => {
      this.notifyUI('queue_status', status);
    });
  }

  handleSyncFailed(payload) {
    this.log('Service Worker sync failed:', payload);
    this.notifyUI('sw_sync_failed', payload);
  }

  /**
   * Status and Monitoring
   */
  async getQueueStatus() {
    try {
      const allItems = await this.getQueue();

      const status = {
        total: allItems.length,
        pending: allItems.filter(item => item.status === 'pending').length,
        processing: allItems.filter(item => item.status === 'processing').length,
        completed: allItems.filter(item => item.status === 'completed').length,
        failed: allItems.filter(item => item.status === 'failed').length,
        byTag: {}
      };

      // Group by tag
      allItems.forEach(item => {
        if (!status.byTag[item.tag]) {
          status.byTag[item.tag] = { total: 0, pending: 0, completed: 0, failed: 0 };
        }
        status.byTag[item.tag].total++;
        status.byTag[item.tag][item.status]++;
      });

      return status;
    } catch (error) {
      this.log('Failed to get queue status:', error);
      return { total: 0, pending: 0, processing: 0, completed: 0, failed: 0, byTag: {} };
    }
  }

  /**
   * Utility Functions
   */
  getAuthToken() {
    // Get auth token from localStorage or session
    return localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
  }

  notifyUI(eventType, data) {
    // Dispatch custom events for UI to listen to
    const event = new CustomEvent('background-sync', {
      detail: { type: eventType, data }
    });
    window.dispatchEvent(event);
  }

  log(...args) {
    if (this.options.enableLogging) {
      console.log('[BackgroundSync]', ...args);
    }
  }

  // IndexedDB helper methods
  async addToStore(store, data) {
    return new Promise((resolve, reject) => {
      const request = store.add(data);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getFromStore(store, key) {
    return new Promise((resolve, reject) => {
      const request = store.get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getAllFromStore(store) {
    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getAllFromIndex(index, query) {
    return new Promise((resolve, reject) => {
      const request = index.getAll(query);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async updateInStore(store, data) {
    return new Promise((resolve, reject) => {
      const request = store.put(data);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async deleteFromStore(store, key) {
    return new Promise((resolve, reject) => {
      const request = store.delete(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
}

// Global instance
let globalSyncManager = null;

export function getBackgroundSyncManager(options = {}) {
  if (!globalSyncManager) {
    globalSyncManager = new BackgroundSyncManager(options);
  }
  return globalSyncManager;
}

export default BackgroundSyncManager;
