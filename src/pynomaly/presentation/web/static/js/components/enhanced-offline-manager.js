/**
 * Enhanced Offline Manager for PWA
 * 
 * Comprehensive offline functionality including data caching, background sync,
 * offline detection capabilities, and intelligent sync management
 */

export class EnhancedOfflineManager {
    constructor(options = {}) {
        this.options = {
            enableOfflineDetection: true,
            enableDataCaching: true,
            enableBackgroundSync: true,
            enableOfflineAnalytics: true,
            cacheStrategy: 'cache-first',
            maxOfflineData: 10000,
            syncInterval: 30000, // 30 seconds
            offlineDetectionThreshold: 5000, // 5 seconds
            dbName: 'PynomalyOfflineDB',
            dbVersion: 1,
            enablePushNotifications: true,
            ...options
        };

        this.isOnline = navigator.onLine;
        this.db = null;
        this.serviceWorker = null;
        this.syncQueue = [];
        this.offlineData = new Map();
        this.offlineDetectors = new Map();
        this.pendingOperations = new Map();
        this.listeners = new Map();

        this.init();
    }

    async init() {
        console.log('Initializing Enhanced Offline Manager...');
        
        await this.initDatabase();
        await this.initServiceWorker();
        this.setupNetworkDetection();
        this.setupDataSyncManager();
        this.setupOfflineDetectionCapabilities();
        
        if (this.options.enableOfflineAnalytics) {
            await this.initOfflineAnalytics();
        }

        console.log('Enhanced Offline Manager initialized successfully');
        this.emit('initialized');
    }

    async initDatabase() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.options.dbName, this.options.dbVersion);

            request.onerror = () => reject(request.error);
            
            request.onsuccess = () => {
                this.db = request.result;
                console.log('IndexedDB initialized');
                resolve();
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // Create object stores for different data types
                this.createObjectStores(db);
            };
        });
    }

    createObjectStores(db) {
        // Datasets store
        if (!db.objectStoreNames.contains('datasets')) {
            const datasetsStore = db.createObjectStore('datasets', { keyPath: 'id' });
            datasetsStore.createIndex('timestamp', 'timestamp');
            datasetsStore.createIndex('name', 'name');
        }

        // Detection results store
        if (!db.objectStoreNames.contains('detectionResults')) {
            const resultsStore = db.createObjectStore('detectionResults', { keyPath: 'id' });
            resultsStore.createIndex('timestamp', 'timestamp');
            resultsStore.createIndex('datasetId', 'datasetId');
            resultsStore.createIndex('score', 'score');
        }

        // Models store
        if (!db.objectStoreNames.contains('models')) {
            const modelsStore = db.createObjectStore('models', { keyPath: 'id' });
            modelsStore.createIndex('name', 'name');
            modelsStore.createIndex('type', 'type');
        }

        // Sync queue store
        if (!db.objectStoreNames.contains('syncQueue')) {
            const syncStore = db.createObjectStore('syncQueue', { keyPath: 'id', autoIncrement: true });
            syncStore.createIndex('operation', 'operation');
            syncStore.createIndex('timestamp', 'timestamp');
            syncStore.createIndex('priority', 'priority');
        }

        // Offline analytics store
        if (!db.objectStoreNames.contains('offlineAnalytics')) {
            const analyticsStore = db.createObjectStore('offlineAnalytics', { keyPath: 'id', autoIncrement: true });
            analyticsStore.createIndex('timestamp', 'timestamp');
            analyticsStore.createIndex('type', 'type');
        }

        // User actions store
        if (!db.objectStoreNames.contains('userActions')) {
            const actionsStore = db.createObjectStore('userActions', { keyPath: 'id', autoIncrement: true });
            actionsStore.createIndex('timestamp', 'timestamp');
            actionsStore.createIndex('action', 'action');
        }
    }

    async initServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register('/sw-enhanced.js');
                this.serviceWorker = registration;
                
                navigator.serviceWorker.addEventListener('message', (event) => {
                    this.handleServiceWorkerMessage(event);
                });

                console.log('Enhanced service worker registered');
                this.emit('service-worker-ready');
            } catch (error) {
                console.error('Service worker registration failed:', error);
            }
        }
    }

    setupNetworkDetection() {
        window.addEventListener('online', () => {
            this.isOnline = true;
            console.log('Network connection restored');
            this.emit('online');
            this.syncPendingData();
        });

        window.addEventListener('offline', () => {
            this.isOnline = false;
            console.log('Network connection lost');
            this.emit('offline');
        });

        // Advanced network detection
        this.setupAdvancedNetworkDetection();
    }

    setupAdvancedNetworkDetection() {
        // Monitor actual connectivity, not just navigator.onLine
        let lastOnlineCheck = Date.now();
        
        const checkConnectivity = async () => {
            try {
                const response = await fetch('/api/ping', {
                    method: 'HEAD',
                    cache: 'no-cache',
                    timeout: this.options.offlineDetectionThreshold
                });
                
                if (response.ok) {
                    if (!this.isOnline) {
                        this.isOnline = true;
                        this.emit('connectivity-restored');
                    }
                    lastOnlineCheck = Date.now();
                } else {
                    throw new Error('Server not responding');
                }
            } catch (error) {
                const timeSinceLastCheck = Date.now() - lastOnlineCheck;
                if (timeSinceLastCheck > this.options.offlineDetectionThreshold && this.isOnline) {
                    this.isOnline = false;
                    this.emit('connectivity-lost');
                }
            }
        };

        // Check connectivity every 30 seconds
        setInterval(checkConnectivity, 30000);
    }

    setupDataSyncManager() {
        // Periodic sync for pending operations
        setInterval(() => {
            if (this.isOnline && this.syncQueue.length > 0) {
                this.processSyncQueue();
            }
        }, this.options.syncInterval);
    }

    async setupOfflineDetectionCapabilities() {
        // Load lightweight models for offline detection
        await this.loadOfflineModels();
        
        // Setup offline detection workers
        this.setupOfflineWorkers();
    }

    async loadOfflineModels() {
        try {
            // Load basic anomaly detection models for offline use
            const models = await this.getStoredData('models');
            
            for (const model of models) {
                if (model.offlineCapable) {
                    await this.loadOfflineModel(model);
                }
            }
            
            console.log(`Loaded ${models.length} offline detection models`);
        } catch (error) {
            console.error('Failed to load offline models:', error);
        }
    }

    async loadOfflineModel(modelConfig) {
        // Simplified offline detection using statistical methods
        const detector = new OfflineAnomalyDetector(modelConfig);
        await detector.initialize();
        
        this.offlineDetectors.set(modelConfig.id, detector);
    }

    setupOfflineWorkers() {
        if ('Worker' in window) {
            this.detectionWorker = new Worker('/static/js/workers/offline-detection-worker.js');
            
            this.detectionWorker.onmessage = (event) => {
                this.handleWorkerMessage(event);
            };
        }
    }

    // Data management methods
    async storeDataset(dataset) {
        try {
            await this.storeData('datasets', {
                ...dataset,
                timestamp: Date.now(),
                offline: true
            });
            
            this.emit('dataset-stored', dataset);
        } catch (error) {
            console.error('Failed to store dataset:', error);
            throw error;
        }
    }

    async getStoredDatasets() {
        return await this.getStoredData('datasets');
    }

    async storeDetectionResult(result) {
        try {
            await this.storeData('detectionResults', {
                ...result,
                timestamp: Date.now(),
                offline: !this.isOnline
            });
            
            this.emit('result-stored', result);
        } catch (error) {
            console.error('Failed to store detection result:', error);
            throw error;
        }
    }

    async getStoredResults(filters = {}) {
        const allResults = await this.getStoredData('detectionResults');
        
        // Apply filters
        return allResults.filter(result => {
            if (filters.datasetId && result.datasetId !== filters.datasetId) return false;
            if (filters.minScore && result.score < filters.minScore) return false;
            if (filters.startDate && new Date(result.timestamp) < new Date(filters.startDate)) return false;
            if (filters.endDate && new Date(result.timestamp) > new Date(filters.endDate)) return false;
            return true;
        });
    }

    // Offline detection capabilities
    async detectAnomaliesOffline(data, modelId = 'default') {
        if (!this.isOnline) {
            const detector = this.offlineDetectors.get(modelId);
            
            if (detector) {
                try {
                    const results = await detector.detect(data);
                    
                    // Store results locally
                    await this.storeDetectionResult({
                        id: `offline-${Date.now()}`,
                        results,
                        modelId,
                        datasetId: data.id || 'unknown',
                        timestamp: Date.now(),
                        offline: true
                    });
                    
                    this.emit('offline-detection-complete', results);
                    return results;
                } catch (error) {
                    console.error('Offline detection failed:', error);
                    throw error;
                }
            } else {
                throw new Error('No offline detector available for this model');
            }
        } else {
            // Online detection - queue for normal processing
            return await this.detectAnomaliesOnline(data, modelId);
        }
    }

    async detectAnomaliesOnline(data, modelId) {
        // Regular online detection
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data, modelId })
        });
        
        return await response.json();
    }

    // Sync management
    async addToSyncQueue(operation) {
        const syncItem = {
            operation: operation.type,
            data: operation.data,
            timestamp: Date.now(),
            priority: operation.priority || 'normal',
            retryCount: 0,
            maxRetries: 3
        };
        
        await this.storeData('syncQueue', syncItem);
        this.syncQueue.push(syncItem);
        
        // Try immediate sync if online
        if (this.isOnline) {
            this.processSyncQueue();
        }
    }

    async processSyncQueue() {
        if (!this.isOnline || this.syncQueue.length === 0) return;
        
        console.log(`Processing ${this.syncQueue.length} sync items`);
        
        const processed = [];
        
        for (const item of this.syncQueue) {
            try {
                await this.processSyncItem(item);
                processed.push(item);
                
                // Remove from database
                await this.removeFromSyncQueue(item.id);
            } catch (error) {
                console.error('Sync item failed:', error);
                
                item.retryCount++;
                if (item.retryCount >= item.maxRetries) {
                    console.error('Max retries reached for sync item:', item);
                    processed.push(item);
                    await this.removeFromSyncQueue(item.id);
                }
            }
        }
        
        // Remove processed items from memory queue
        this.syncQueue = this.syncQueue.filter(item => !processed.includes(item));
        
        if (processed.length > 0) {
            this.emit('sync-complete', { processed: processed.length });
        }
    }

    async processSyncItem(item) {
        switch (item.operation) {
            case 'upload-dataset':
                await this.syncDataset(item.data);
                break;
            case 'sync-results':
                await this.syncDetectionResults(item.data);
                break;
            case 'user-action':
                await this.syncUserAction(item.data);
                break;
            case 'analytics':
                await this.syncAnalytics(item.data);
                break;
            default:
                console.warn('Unknown sync operation:', item.operation);
        }
    }

    async syncDataset(dataset) {
        const response = await fetch('/api/datasets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dataset)
        });
        
        if (!response.ok) {
            throw new Error(`Sync failed: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async syncDetectionResults(results) {
        const response = await fetch('/api/detection-results', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(results)
        });
        
        if (!response.ok) {
            throw new Error(`Results sync failed: ${response.statusText}`);
        }
        
        return await response.json();
    }

    // Analytics for offline usage
    async initOfflineAnalytics() {
        this.analyticsBuffer = [];
        
        // Track offline usage patterns
        this.trackUsagePattern('app-start', { timestamp: Date.now() });
        
        // Periodic analytics sync
        setInterval(() => {
            if (this.isOnline && this.analyticsBuffer.length > 0) {
                this.syncAnalyticsBuffer();
            }
        }, 60000); // Every minute
    }

    trackUsagePattern(event, data = {}) {
        const analyticsData = {
            event,
            data,
            timestamp: Date.now(),
            offline: !this.isOnline,
            userAgent: navigator.userAgent,
            sessionId: this.getSessionId()
        };
        
        this.analyticsBuffer.push(analyticsData);
        this.storeData('offlineAnalytics', analyticsData);
    }

    async syncAnalyticsBuffer() {
        if (this.analyticsBuffer.length === 0) return;
        
        try {
            await fetch('/api/analytics/offline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.analyticsBuffer)
            });
            
            this.analyticsBuffer = [];
            console.log('Analytics synced successfully');
        } catch (error) {
            console.error('Analytics sync failed:', error);
        }
    }

    // Utility methods
    async storeData(storeName, data) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([storeName], 'readwrite');
            const store = transaction.objectStore(storeName);
            const request = store.add(data);
            
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getStoredData(storeName) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([storeName], 'readonly');
            const store = transaction.objectStore(storeName);
            const request = store.getAll();
            
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async removeFromSyncQueue(id) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['syncQueue'], 'readwrite');
            const store = transaction.objectStore('syncQueue');
            const request = store.delete(id);
            
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    getSessionId() {
        if (!this.sessionId) {
            this.sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        }
        return this.sessionId;
    }

    // Service Worker communication
    handleServiceWorkerMessage(event) {
        const { type, payload } = event.data;
        
        switch (type) {
            case 'SYNC_COMPLETE':
                this.emit('background-sync-complete', payload);
                break;
            case 'CACHE_UPDATE':
                this.emit('cache-updated', payload);
                break;
            case 'OFFLINE_READY':
                this.emit('offline-ready');
                break;
        }
    }

    // Data export for offline use
    async exportOfflineData() {
        const data = {
            datasets: await this.getStoredData('datasets'),
            results: await this.getStoredData('detectionResults'),
            models: await this.getStoredData('models'),
            analytics: await this.getStoredData('offlineAnalytics'),
            timestamp: Date.now()
        };
        
        const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pynomaly-offline-data-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    // Status and monitoring
    getOfflineStatus() {
        return {
            isOnline: this.isOnline,
            syncQueueLength: this.syncQueue.length,
            offlineDetectors: this.offlineDetectors.size,
            storedDatasets: this.offlineData.get('datasets')?.length || 0,
            analyticsBuffer: this.analyticsBuffer?.length || 0,
            lastSync: this.lastSyncTime,
            capabilities: {
                offlineDetection: this.offlineDetectors.size > 0,
                dataStorage: !!this.db,
                backgroundSync: !!this.serviceWorker
            }
        };
    }

    // Event emitter
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
        
        return () => this.listeners.get(event).delete(callback);
    }

    emit(event, data) {
        const callbacks = this.listeners.get(event);
        if (callbacks) {
            callbacks.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error('Event callback error:', error);
                }
            });
        }
    }

    // Cleanup
    destroy() {
        // Close database
        if (this.db) {
            this.db.close();
        }
        
        // Terminate workers
        if (this.detectionWorker) {
            this.detectionWorker.terminate();
        }
        
        // Clear data
        this.offlineData.clear();
        this.offlineDetectors.clear();
        this.pendingOperations.clear();
        this.listeners.clear();
    }
}

// Offline Anomaly Detector for basic detection capabilities
class OfflineAnomalyDetector {
    constructor(config) {
        this.config = config;
        this.model = null;
        this.statistics = null;
    }

    async initialize() {
        // Load basic statistical model for offline detection
        this.model = {
            type: 'statistical',
            threshold: this.config.threshold || 2.5, // Z-score threshold
            windowSize: this.config.windowSize || 100
        };
    }

    async detect(data) {
        if (!Array.isArray(data)) {
            throw new Error('Data must be an array');
        }

        const results = [];
        
        for (let i = 0; i < data.length; i++) {
            const point = data[i];
            const score = this.calculateAnomalyScore(point, data, i);
            
            results.push({
                index: i,
                data: point,
                score: score,
                isAnomaly: score > this.model.threshold,
                confidence: Math.min(score / this.model.threshold, 1.0)
            });
        }
        
        return results;
    }

    calculateAnomalyScore(point, data, index) {
        // Simple Z-score based anomaly detection
        const windowStart = Math.max(0, index - this.model.windowSize);
        const window = data.slice(windowStart, index);
        
        if (window.length < 10) return 0; // Not enough data
        
        const values = window.map(p => typeof p === 'number' ? p : p.value || 0);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        
        if (stdDev === 0) return 0;
        
        const currentValue = typeof point === 'number' ? point : point.value || 0;
        return Math.abs((currentValue - mean) / stdDev);
    }
}

export default EnhancedOfflineManager;