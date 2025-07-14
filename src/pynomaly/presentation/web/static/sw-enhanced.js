/**
 * Enhanced Service Worker for Pynomaly PWA
 * 
 * Provides advanced offline capabilities including:
 * - Intelligent caching strategies
 * - Background sync for data operations
 * - Push notifications for anomaly alerts
 * - Offline detection capabilities
 * - Performance optimization
 */

const CACHE_VERSION = 'v2.0.0';
const CACHE_NAMES = {
    static: `pynomaly-static-${CACHE_VERSION}`,
    dynamic: `pynomaly-dynamic-${CACHE_VERSION}`,
    api: `pynomaly-api-${CACHE_VERSION}`,
    models: `pynomaly-models-${CACHE_VERSION}`,
    datasets: `pynomaly-datasets-${CACHE_VERSION}`
};

// URLs to cache immediately
const STATIC_CACHE_URLS = [
    '/',
    '/static/css/design-system.css',
    '/static/css/accessibility.css',
    '/static/js/main.js',
    '/static/js/components/d3-chart-library.js',
    '/static/js/components/real-time-dashboard.js',
    '/static/js/accessibility-enhancements.js',
    '/static/js/voice-commands.js',
    '/static/manifest.json',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png',
    '/offline.html'
];

// API endpoints that should be cached
const API_CACHE_PATTERNS = [
    /^\/api\/models/,
    /^\/api\/datasets\/\w+$/,
    /^\/api\/detection-results/,
    /^\/api\/statistics/
];

// Background sync tags
const SYNC_TAGS = {
    UPLOAD_DATASET: 'upload-dataset',
    SYNC_RESULTS: 'sync-detection-results',
    SEND_ANALYTICS: 'send-analytics',
    UPDATE_MODEL: 'update-model'
};

// Install event - cache static resources
self.addEventListener('install', (event) => {
    console.log('Enhanced Service Worker installing...');
    
    event.waitUntil(
        Promise.all([
            caches.open(CACHE_NAMES.static).then((cache) => {
                return cache.addAll(STATIC_CACHE_URLS);
            }),
            // Initialize IndexedDB for offline data
            initializeOfflineDB(),
            // Skip waiting to activate immediately
            self.skipWaiting()
        ])
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('Enhanced Service Worker activating...');
    
    event.waitUntil(
        Promise.all([
            // Clean up old caches
            cleanupOldCaches(),
            // Take control of all pages
            self.clients.claim(),
            // Initialize background sync
            initializeBackgroundSync()
        ])
    );
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Handle different types of requests
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(handleApiRequest(request));
    } else if (url.pathname.startsWith('/static/')) {
        event.respondWith(handleStaticRequest(request));
    } else if (url.pathname.includes('detect') || url.pathname.includes('analyze')) {
        event.respondWith(handleDetectionRequest(request));
    } else {
        event.respondWith(handleNavigationRequest(request));
    }
});

// Background sync event
self.addEventListener('sync', (event) => {
    console.log('Background sync triggered:', event.tag);
    
    switch (event.tag) {
        case SYNC_TAGS.UPLOAD_DATASET:
            event.waitUntil(syncUploadedDatasets());
            break;
        case SYNC_TAGS.SYNC_RESULTS:
            event.waitUntil(syncDetectionResults());
            break;
        case SYNC_TAGS.SEND_ANALYTICS:
            event.waitUntil(syncAnalytics());
            break;
        case SYNC_TAGS.UPDATE_MODEL:
            event.waitUntil(syncModelUpdates());
            break;
        default:
            console.log('Unknown sync tag:', event.tag);
    }
});

// Push notification event
self.addEventListener('push', (event) => {
    console.log('Push notification received');
    
    let notificationData = {};
    
    if (event.data) {
        try {
            notificationData = event.data.json();
        } catch (error) {
            notificationData = { title: 'Pynomaly', body: event.data.text() };
        }
    }
    
    const options = {
        title: notificationData.title || 'Anomaly Detected',
        body: notificationData.body || 'A new anomaly has been detected in your data.',
        icon: '/static/icons/notification.png',
        badge: '/static/icons/badge.png',
        data: notificationData.data || {},
        actions: [
            {
                action: 'view',
                title: 'View Details'
            },
            {
                action: 'dismiss',
                title: 'Dismiss'
            }
        ],
        requireInteraction: notificationData.severity === 'critical',
        silent: false,
        vibrate: notificationData.severity === 'critical' ? [200, 100, 200] : [100]
    };
    
    event.waitUntil(
        self.registration.showNotification(options.title, options)
    );
});

// Notification click event
self.addEventListener('notificationclick', (event) => {
    console.log('Notification clicked:', event.action);
    
    event.notification.close();
    
    if (event.action === 'view') {
        event.waitUntil(
            clients.openWindow('/dashboard?notification=' + event.notification.data.id)
        );
    }
    // Dismiss action doesn't need handling as notification is already closed
});

// Message event - communication with main thread
self.addEventListener('message', (event) => {
    const { type, payload } = event.data;
    
    switch (type) {
        case 'SKIP_WAITING':
            self.skipWaiting();
            break;
        case 'CACHE_DATASET':
            cacheDataset(payload);
            break;
        case 'CACHE_MODEL':
            cacheModel(payload);
            break;
        case 'CLEAR_CACHE':
            clearCache(payload.cacheName);
            break;
        case 'GET_CACHE_SIZE':
            getCacheSize().then(size => {
                event.ports[0].postMessage({ type: 'CACHE_SIZE', size });
            });
            break;
        case 'ENABLE_OFFLINE_MODE':
            enableOfflineMode();
            break;
        case 'DISABLE_OFFLINE_MODE':
            disableOfflineMode();
            break;
        default:
            console.log('Unknown message type:', type);
    }
});

// Caching strategy implementations
async function handleApiRequest(request) {
    const url = new URL(request.url);
    
    // Check if this API endpoint should be cached
    const shouldCache = API_CACHE_PATTERNS.some(pattern => pattern.test(url.pathname));
    
    if (shouldCache && request.method === 'GET') {
        return cacheFirstStrategy(request, CACHE_NAMES.api);
    } else if (request.method === 'POST' || request.method === 'PUT') {
        return networkFirstWithOfflineFallback(request);
    } else {
        return networkOnlyStrategy(request);
    }
}

async function handleStaticRequest(request) {
    return cacheFirstStrategy(request, CACHE_NAMES.static);
}

async function handleDetectionRequest(request) {
    if (request.method === 'POST') {
        return handleOfflineDetection(request);
    } else {
        return networkFirstStrategy(request, CACHE_NAMES.dynamic);
    }
}

async function handleNavigationRequest(request) {
    return networkFirstWithOfflineFallback(request, '/offline.html');
}

// Caching strategies
async function cacheFirstStrategy(request, cacheName) {
    try {
        const cache = await caches.open(cacheName);
        const cachedResponse = await cache.match(request);
        
        if (cachedResponse) {
            // Update cache in background
            fetch(request).then(response => {
                if (response.ok) {
                    cache.put(request, response.clone());
                }
            }).catch(() => {}); // Ignore network errors
            
            return cachedResponse;
        }
        
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.error('Cache first strategy failed:', error);
        return new Response('Service unavailable', { status: 503 });
    }
}

async function networkFirstStrategy(request, cacheName) {
    try {
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok && cacheName) {
            const cache = await caches.open(cacheName);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        if (cacheName) {
            const cache = await caches.open(cacheName);
            const cachedResponse = await cache.match(request);
            if (cachedResponse) {
                return cachedResponse;
            }
        }
        
        throw error;
    }
}

async function networkFirstWithOfflineFallback(request, fallbackUrl = null) {
    try {
        return await fetch(request);
    } catch (error) {
        // Try to find cached version
        const cache = await caches.open(CACHE_NAMES.dynamic);
        const cachedResponse = await cache.match(request);
        
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Return offline fallback if available
        if (fallbackUrl) {
            return caches.match(fallbackUrl);
        }
        
        // Store request for background sync if it's a mutation
        if (request.method === 'POST' || request.method === 'PUT') {
            await storeRequestForSync(request);
        }
        
        return new Response('Offline', { 
            status: 503, 
            statusText: 'Service Unavailable' 
        });
    }
}

async function networkOnlyStrategy(request) {
    return fetch(request);
}

// Offline detection capabilities
async function handleOfflineDetection(request) {
    try {
        // Try network first
        return await fetch(request);
    } catch (error) {
        // Perform offline detection
        return performOfflineDetection(request);
    }
}

async function performOfflineDetection(request) {
    try {
        const requestData = await request.json();
        const db = await openOfflineDB();
        
        // Get offline models
        const models = await getOfflineModels(db);
        
        if (models.length === 0) {
            return new Response(JSON.stringify({
                error: 'No offline models available',
                offline: true
            }), {
                status: 503,
                headers: { 'Content-Type': 'application/json' }
            });
        }
        
        // Perform basic statistical anomaly detection
        const results = await basicAnomalyDetection(requestData.data, models[0]);
        
        // Store results for later sync
        await storeOfflineResults(db, results);
        
        return new Response(JSON.stringify({
            results,
            offline: true,
            model: models[0].id,
            timestamp: Date.now()
        }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    } catch (error) {
        console.error('Offline detection failed:', error);
        return new Response(JSON.stringify({
            error: 'Offline detection failed',
            offline: true
        }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

async function basicAnomalyDetection(data, model) {
    // Simple Z-score based anomaly detection for offline use
    const threshold = model.threshold || 2.5;
    const results = [];
    
    for (let i = 0; i < data.length; i++) {
        const point = data[i];
        const windowStart = Math.max(0, i - 50); // 50-point window
        const window = data.slice(windowStart, i);
        
        if (window.length < 10) {
            results.push({
                index: i,
                value: point,
                score: 0,
                isAnomaly: false,
                confidence: 0
            });
            continue;
        }
        
        const values = window.map(p => typeof p === 'number' ? p : p.value || 0);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        
        const currentValue = typeof point === 'number' ? point : point.value || 0;
        const zScore = stdDev === 0 ? 0 : Math.abs((currentValue - mean) / stdDev);
        
        results.push({
            index: i,
            value: currentValue,
            score: zScore / threshold,
            isAnomaly: zScore > threshold,
            confidence: Math.min(zScore / threshold, 1.0)
        });
    }
    
    return results;
}

// Background sync implementations
async function syncUploadedDatasets() {
    const db = await openOfflineDB();
    const transaction = db.transaction(['syncQueue'], 'readonly');
    const store = transaction.objectStore('syncQueue');
    const index = store.index('operation');
    const request = index.getAll('upload-dataset');
    
    return new Promise((resolve, reject) => {
        request.onsuccess = async () => {
            const datasets = request.result;
            
            for (const item of datasets) {
                try {
                    await fetch('/api/datasets', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(item.data)
                    });
                    
                    // Remove from sync queue
                    await removeFromSyncQueue(db, item.id);
                } catch (error) {
                    console.error('Failed to sync dataset:', error);
                }
            }
            
            resolve();
        };
        
        request.onerror = () => reject(request.error);
    });
}

async function syncDetectionResults() {
    const db = await openOfflineDB();
    const transaction = db.transaction(['detectionResults'], 'readonly');
    const store = transaction.objectStore('detectionResults');
    const request = store.getAll();
    
    return new Promise((resolve, reject) => {
        request.onsuccess = async () => {
            const results = request.result.filter(r => r.offline && !r.synced);
            
            for (const result of results) {
                try {
                    await fetch('/api/detection-results', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(result)
                    });
                    
                    // Mark as synced
                    result.synced = true;
                    await updateOfflineResult(db, result);
                } catch (error) {
                    console.error('Failed to sync detection result:', error);
                }
            }
            
            resolve();
        };
        
        request.onerror = () => reject(request.error);
    });
}

async function syncAnalytics() {
    const db = await openOfflineDB();
    const transaction = db.transaction(['offlineAnalytics'], 'readonly');
    const store = transaction.objectStore('offlineAnalytics');
    const request = store.getAll();
    
    return new Promise((resolve, reject) => {
        request.onsuccess = async () => {
            const analytics = request.result.filter(a => !a.synced);
            
            if (analytics.length > 0) {
                try {
                    await fetch('/api/analytics/offline', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(analytics)
                    });
                    
                    // Mark all as synced
                    for (const item of analytics) {
                        item.synced = true;
                        await updateAnalyticsItem(db, item);
                    }
                } catch (error) {
                    console.error('Failed to sync analytics:', error);
                }
            }
            
            resolve();
        };
        
        request.onerror = () => reject(request.error);
    });
}

// IndexedDB operations
async function initializeOfflineDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('PynomalyOfflineDB', 1);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            
            // Create object stores
            if (!db.objectStoreNames.contains('datasets')) {
                const datasetsStore = db.createObjectStore('datasets', { keyPath: 'id' });
                datasetsStore.createIndex('timestamp', 'timestamp');
            }
            
            if (!db.objectStoreNames.contains('detectionResults')) {
                const resultsStore = db.createObjectStore('detectionResults', { keyPath: 'id' });
                resultsStore.createIndex('timestamp', 'timestamp');
                resultsStore.createIndex('offline', 'offline');
            }
            
            if (!db.objectStoreNames.contains('models')) {
                const modelsStore = db.createObjectStore('models', { keyPath: 'id' });
                modelsStore.createIndex('type', 'type');
            }
            
            if (!db.objectStoreNames.contains('syncQueue')) {
                const syncStore = db.createObjectStore('syncQueue', { keyPath: 'id', autoIncrement: true });
                syncStore.createIndex('operation', 'operation');
                syncStore.createIndex('timestamp', 'timestamp');
            }
            
            if (!db.objectStoreNames.contains('offlineAnalytics')) {
                const analyticsStore = db.createObjectStore('offlineAnalytics', { keyPath: 'id', autoIncrement: true });
                analyticsStore.createIndex('timestamp', 'timestamp');
                analyticsStore.createIndex('synced', 'synced');
            }
        };
    });
}

async function openOfflineDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('PynomalyOfflineDB', 1);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function storeRequestForSync(request) {
    const db = await openOfflineDB();
    const clonedRequest = request.clone();
    const body = await clonedRequest.text();
    
    const syncItem = {
        url: request.url,
        method: request.method,
        headers: Object.fromEntries(request.headers.entries()),
        body: body,
        timestamp: Date.now()
    };
    
    const transaction = db.transaction(['syncQueue'], 'readwrite');
    const store = transaction.objectStore('syncQueue');
    await store.add(syncItem);
}

// Utility functions
async function cleanupOldCaches() {
    const cacheNames = await caches.keys();
    const currentCaches = Object.values(CACHE_NAMES);
    
    return Promise.all(
        cacheNames
            .filter(cacheName => !currentCaches.includes(cacheName))
            .map(cacheName => caches.delete(cacheName))
    );
}

async function initializeBackgroundSync() {
    // Register for background sync if supported
    if ('sync' in self.registration) {
        console.log('Background sync is supported');
    }
}

async function getCacheSize() {
    const cacheNames = await caches.keys();
    let totalSize = 0;
    
    for (const cacheName of cacheNames) {
        const cache = await caches.open(cacheName);
        const keys = await cache.keys();
        
        for (const request of keys) {
            const response = await cache.match(request);
            if (response) {
                const blob = await response.blob();
                totalSize += blob.size;
            }
        }
    }
    
    return totalSize;
}

// Send messages to clients
function broadcastMessage(message) {
    clients.matchAll().then(clients => {
        clients.forEach(client => client.postMessage(message));
    });
}

// Error handling
self.addEventListener('error', (event) => {
    console.error('Service Worker error:', event.error);
});

self.addEventListener('unhandledrejection', (event) => {
    console.error('Service Worker unhandled rejection:', event.reason);
});

console.log('Enhanced Service Worker loaded successfully');