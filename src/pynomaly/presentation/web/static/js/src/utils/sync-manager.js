/**
 * Sync Manager - Handles data synchronization between offline and online modes
 * Manages background sync, conflict resolution, and data consistency
 */
export class SyncManager {
  constructor() {
    this.syncQueue = [];
    this.conflictQueue = [];
    this.isOnline = navigator.onLine;
    this.isSyncing = false;
    this.syncStrategy = 'smart'; // 'immediate', 'smart', 'manual'
    this.retryAttempts = 3;
    this.retryDelay = 1000; // ms
    
    this.init();
  }

  async init() {
    this.setupEventListeners();
    await this.loadSyncQueue();
    this.startPeriodicSync();
  }

  /**
   * Setup event listeners for online/offline detection
   */
  setupEventListeners() {
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.onConnectionRestore();
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
      this.onConnectionLost();
    });

    // Listen for service worker messages
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.addEventListener('message', (event) => {
        this.handleServiceWorkerMessage(event.data);
      });
    }
  }

  /**
   * Handle service worker messages
   */
  handleServiceWorkerMessage(data) {
    const { type, payload } = data;
    
    switch (type) {
      case 'SYNC_COMPLETE':
        this.onSyncComplete(payload);
        break;
      case 'SYNC_FAILED':
        this.onSyncFailed(payload);
        break;
      case 'CONFLICT_DETECTED':
        this.onConflictDetected(payload);
        break;
      case 'SYNC_QUEUE_UPDATE':
        this.onSyncQueueUpdate(payload);
        break;
    }
  }

  /**
   * Load sync queue from IndexedDB
   */
  async loadSyncQueue() {
    try {
      if ('serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          registration.active.postMessage({ type: 'GET_SYNC_QUEUE' });
        }
      }
    } catch (error) {
      console.error('[SyncManager] Failed to load sync queue:', error);
    }
  }

  /**
   * Queue data for synchronization
   */
  async queueForSync(operation, data, priority = 'normal') {
    const syncItem = {
      id: this.generateSyncId(),
      operation, // 'create', 'update', 'delete'
      entityType: data.entityType, // 'dataset', 'result', 'model', etc.
      entityId: data.entityId,
      data: data,
      priority, // 'high', 'normal', 'low'
      timestamp: Date.now(),
      retryCount: 0,
      status: 'pending', // 'pending', 'syncing', 'completed', 'failed'
      conflicts: []
    };

    this.syncQueue.push(syncItem);
    await this.persistSyncQueue();

    // Trigger immediate sync for high priority items when online
    if (this.isOnline && priority === 'high' && this.syncStrategy !== 'manual') {
      this.processSyncQueue();
    }

    this.notifyUI('sync_queue_updated', { pendingCount: this.getPendingCount() });
    return syncItem.id;
  }

  /**
   * Process sync queue
   */
  async processSyncQueue() {
    if (this.isSyncing || !this.isOnline) return;

    this.isSyncing = true;
    this.notifyUI('sync_started');

    try {
      // Sort by priority and timestamp
      const pendingItems = this.syncQueue
        .filter(item => item.status === 'pending')
        .sort((a, b) => {
          const priorityOrder = { high: 3, normal: 2, low: 1 };
          const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
          return priorityDiff !== 0 ? priorityDiff : a.timestamp - b.timestamp;
        });

      const results = {
        completed: 0,
        failed: 0,
        conflicts: 0
      };

      for (const item of pendingItems) {
        try {
          item.status = 'syncing';
          await this.persistSyncQueue();

          const result = await this.syncItem(item);
          
          if (result.success) {
            item.status = 'completed';
            item.completedAt = Date.now();
            results.completed++;
          } else if (result.conflict) {
            item.status = 'conflict';
            item.conflicts.push(result.conflict);
            this.conflictQueue.push(item);
            results.conflicts++;
          } else {
            throw new Error(result.error || 'Sync failed');
          }
        } catch (error) {
          console.error('[SyncManager] Failed to sync item:', item.id, error);
          item.retryCount++;
          
          if (item.retryCount >= this.retryAttempts) {
            item.status = 'failed';
            item.error = error.message;
            results.failed++;
          } else {
            item.status = 'pending';
            // Exponential backoff
            await this.delay(this.retryDelay * Math.pow(2, item.retryCount - 1));
          }
        }

        await this.persistSyncQueue();
      }

      // Clean up completed items older than 24 hours
      this.cleanupCompletedItems();

      this.notifyUI('sync_completed', results);
    } catch (error) {
      console.error('[SyncManager] Sync processing failed:', error);
      this.notifyUI('sync_failed', { error: error.message });
    } finally {
      this.isSyncing = false;
    }
  }

  /**
   * Sync individual item
   */
  async syncItem(item) {
    const { operation, entityType, entityId, data } = item;
    
    try {
      let endpoint;
      let method;
      let payload;

      switch (operation) {
        case 'create':
          endpoint = `/api/${entityType}s`;
          method = 'POST';
          payload = data.payload;
          break;
        case 'update':
          endpoint = `/api/${entityType}s/${entityId}`;
          method = 'PUT';
          payload = data.payload;
          break;
        case 'delete':
          endpoint = `/api/${entityType}s/${entityId}`;
          method = 'DELETE';
          break;
        default:
          throw new Error(`Unknown operation: ${operation}`);
      }

      // Check for conflicts before syncing
      const conflictCheck = await this.checkForConflicts(item);
      if (conflictCheck.hasConflict) {
        return { success: false, conflict: conflictCheck.conflict };
      }

      const response = await fetch(endpoint, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': this.getAuthHeader()
        },
        body: payload ? JSON.stringify(payload) : undefined
      });

      if (!response.ok) {
        if (response.status === 409) {
          // Conflict detected
          const conflictData = await response.json();
          return { success: false, conflict: conflictData };
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Update local data with server response
      await this.updateLocalData(item, result);

      return { success: true, data: result };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  /**
   * Check for conflicts before syncing
   */
  async checkForConflicts(item) {
    if (item.operation === 'create') {
      return { hasConflict: false };
    }

    try {
      const response = await fetch(`/api/${item.entityType}s/${item.entityId}`, {
        method: 'HEAD',
        headers: {
          'Authorization': this.getAuthHeader()
        }
      });

      if (response.status === 404) {
        // Entity no longer exists on server
        if (item.operation === 'update') {
          return {
            hasConflict: true,
            conflict: {
              type: 'entity_deleted',
              message: 'Entity was deleted on server',
              serverVersion: null,
              localVersion: item.data.version
            }
          };
        }
      }

      const serverVersion = response.headers.get('etag') || response.headers.get('last-modified');
      const localVersion = item.data.version;

      if (serverVersion && localVersion && serverVersion !== localVersion) {
        return {
          hasConflict: true,
          conflict: {
            type: 'version_mismatch',
            message: 'Entity was modified on server',
            serverVersion,
            localVersion
          }
        };
      }

      return { hasConflict: false };
    } catch (error) {
      console.warn('[SyncManager] Conflict check failed:', error);
      return { hasConflict: false }; // Proceed with sync
    }
  }

  /**
   * Resolve conflict using specified strategy
   */
  async resolveConflict(conflictId, strategy, resolution = null) {
    const conflict = this.conflictQueue.find(c => c.id === conflictId);
    if (!conflict) {
      throw new Error('Conflict not found');
    }

    try {
      let resolvedData;

      switch (strategy) {
        case 'server_wins':
          resolvedData = await this.fetchServerVersion(conflict);
          await this.updateLocalData(conflict, resolvedData);
          break;
        case 'client_wins':
          // Force sync local version
          conflict.retryCount = 0;
          conflict.status = 'pending';
          await this.forceSyncItem(conflict);
          break;
        case 'merge':
          if (!resolution) {
            throw new Error('Merge resolution data required');
          }
          resolvedData = await this.mergeVersions(conflict, resolution);
          await this.updateLocalData(conflict, resolvedData);
          await this.syncMergedData(conflict, resolvedData);
          break;
        case 'manual':
          // User will resolve manually
          conflict.status = 'manual_resolution';
          break;
        default:
          throw new Error(`Unknown resolution strategy: ${strategy}`);
      }

      // Remove from conflict queue
      this.conflictQueue = this.conflictQueue.filter(c => c.id !== conflictId);
      
      // Update sync queue
      const syncItem = this.syncQueue.find(s => s.id === conflictId);
      if (syncItem && strategy !== 'manual') {
        syncItem.status = 'completed';
        syncItem.resolvedAt = Date.now();
        syncItem.resolutionStrategy = strategy;
      }

      await this.persistSyncQueue();
      this.notifyUI('conflict_resolved', { conflictId, strategy });

      return { success: true };
    } catch (error) {
      console.error('[SyncManager] Failed to resolve conflict:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Start periodic background sync
   */
  startPeriodicSync() {
    if (this.syncStrategy === 'manual') return;

    const interval = this.syncStrategy === 'immediate' ? 30000 : 300000; // 30s or 5min
    
    setInterval(() => {
      if (this.isOnline && this.getPendingCount() > 0) {
        this.processSyncQueue();
      }
    }, interval);
  }

  /**
   * Force manual sync
   */
  async forceSyncAll() {
    if (!this.isOnline) {
      throw new Error('Cannot sync while offline');
    }

    await this.processSyncQueue();
  }

  /**
   * Get sync status
   */
  getSyncStatus() {
    const pending = this.syncQueue.filter(item => item.status === 'pending').length;
    const syncing = this.syncQueue.filter(item => item.status === 'syncing').length;
    const failed = this.syncQueue.filter(item => item.status === 'failed').length;
    const conflicts = this.conflictQueue.length;

    return {
      isOnline: this.isOnline,
      isSyncing: this.isSyncing,
      pending,
      syncing,
      failed,
      conflicts,
      strategy: this.syncStrategy,
      lastSyncAt: this.getLastSyncTime()
    };
  }

  /**
   * Set sync strategy
   */
  setSyncStrategy(strategy) {
    if (!['immediate', 'smart', 'manual'].includes(strategy)) {
      throw new Error('Invalid sync strategy');
    }
    
    this.syncStrategy = strategy;
    this.notifyUI('sync_strategy_changed', { strategy });
  }

  /**
   * Clear completed sync items
   */
  async clearCompleted() {
    this.syncQueue = this.syncQueue.filter(item => item.status !== 'completed');
    await this.persistSyncQueue();
    this.notifyUI('completed_cleared');
  }

  /**
   * Event handlers
   */
  onConnectionRestore() {
    console.log('[SyncManager] Connection restored');
    this.notifyUI('connection_restored');
    
    if (this.syncStrategy !== 'manual' && this.getPendingCount() > 0) {
      setTimeout(() => this.processSyncQueue(), 1000);
    }
  }

  onConnectionLost() {
    console.log('[SyncManager] Connection lost');
    this.notifyUI('connection_lost');
  }

  onSyncComplete(payload) {
    this.notifyUI('sync_item_completed', payload);
  }

  onSyncFailed(payload) {
    this.notifyUI('sync_item_failed', payload);
  }

  onConflictDetected(payload) {
    this.conflictQueue.push(payload);
    this.notifyUI('conflict_detected', payload);
  }

  onSyncQueueUpdate(payload) {
    this.syncQueue = payload.queue || [];
    this.notifyUI('sync_queue_updated', { pendingCount: this.getPendingCount() });
  }

  // Helper methods...

  generateSyncId() {
    return `sync_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  getPendingCount() {
    return this.syncQueue.filter(item => 
      ['pending', 'syncing'].includes(item.status)
    ).length;
  }

  getLastSyncTime() {
    const completed = this.syncQueue.filter(item => item.status === 'completed');
    if (!completed.length) return null;
    
    return Math.max(...completed.map(item => item.completedAt));
  }

  async persistSyncQueue() {
    try {
      if ('serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          registration.active.postMessage({
            type: 'UPDATE_SYNC_QUEUE',
            payload: { queue: this.syncQueue }
          });
        }
      }
    } catch (error) {
      console.error('[SyncManager] Failed to persist sync queue:', error);
    }
  }

  cleanupCompletedItems() {
    const oneDayAgo = Date.now() - 24 * 60 * 60 * 1000;
    this.syncQueue = this.syncQueue.filter(item => 
      item.status !== 'completed' || item.completedAt > oneDayAgo
    );
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  getAuthHeader() {
    // Get auth token from storage or context
    const token = localStorage.getItem('auth_token');
    return token ? `Bearer ${token}` : '';
  }

  async fetchServerVersion(item) {
    const response = await fetch(`/api/${item.entityType}s/${item.entityId}`, {
      headers: { 'Authorization': this.getAuthHeader() }
    });
    return await response.json();
  }

  async updateLocalData(item, serverData) {
    // Update local IndexedDB with server data
    if ('serviceWorker' in navigator) {
      const registration = await navigator.serviceWorker.getRegistration();
      if (registration?.active) {
        registration.active.postMessage({
          type: 'UPDATE_LOCAL_DATA',
          payload: {
            entityType: item.entityType,
            entityId: item.entityId,
            data: serverData
          }
        });
      }
    }
  }

  async forceSyncItem(item) {
    // Force sync without conflict checking
    item.forceSync = true;
    return await this.syncItem(item);
  }

  async mergeVersions(conflict, resolution) {
    // Implement merge logic based on entity type
    return resolution.mergedData;
  }

  async syncMergedData(conflict, mergedData) {
    // Sync merged data to server
    const endpoint = `/api/${conflict.entityType}s/${conflict.entityId}`;
    const response = await fetch(endpoint, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': this.getAuthHeader()
      },
      body: JSON.stringify(mergedData)
    });

    if (!response.ok) {
      throw new Error(`Failed to sync merged data: ${response.statusText}`);
    }

    return await response.json();
  }

  notifyUI(eventType, data = {}) {
    // Dispatch custom events for UI updates
    window.dispatchEvent(new CustomEvent('sync-manager', {
      detail: { type: eventType, data }
    }));
  }

  /**
   * Public API methods
   */
  
  // Queue operations for different entity types
  async queueDatasetSync(operation, dataset, priority = 'normal') {
    return await this.queueForSync(operation, {
      entityType: 'dataset',
      entityId: dataset.id,
      payload: dataset,
      version: dataset.version
    }, priority);
  }

  async queueResultSync(operation, result, priority = 'normal') {
    return await this.queueForSync(operation, {
      entityType: 'result',
      entityId: result.id,
      payload: result,
      version: result.version
    }, priority);
  }

  async queueModelSync(operation, model, priority = 'high') {
    return await this.queueForSync(operation, {
      entityType: 'model',
      entityId: model.id,
      payload: model,
      version: model.version
    }, priority);
  }

  // Get conflicts for UI
  getConflicts() {
    return this.conflictQueue.map(conflict => ({
      id: conflict.id,
      entityType: conflict.entityType,
      entityId: conflict.entityId,
      operation: conflict.operation,
      conflicts: conflict.conflicts,
      timestamp: conflict.timestamp
    }));
  }

  // Get pending sync items for UI
  getPendingItems() {
    return this.syncQueue
      .filter(item => ['pending', 'syncing', 'failed'].includes(item.status))
      .map(item => ({
        id: item.id,
        operation: item.operation,
        entityType: item.entityType,
        priority: item.priority,
        status: item.status,
        retryCount: item.retryCount,
        timestamp: item.timestamp
      }));
  }
}

// Initialize and expose globally
if (typeof window !== 'undefined') {
  window.SyncManager = new SyncManager();
}
