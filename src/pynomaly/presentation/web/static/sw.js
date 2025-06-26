/**
 * Pynomaly Service Worker - Progressive Web App with Advanced Features
 * Provides offline caching, background sync, push notifications, and installability
 */

const CACHE_NAME = 'pynomaly-v1.0.0';
const STATIC_CACHE = 'pynomaly-static-v1.0.0';
const DYNAMIC_CACHE = 'pynomaly-dynamic-v1.0.0';
const API_CACHE = 'pynomaly-api-v1.0.0';

// Static assets to cache during install
const STATIC_ASSETS = [
  '/',
  '/static/css/design-system.css',
  '/static/css/tailwind.css',
  '/static/js/app.js',
  '/static/js/htmx.min.js',
  '/static/js/d3.min.js',
  '/static/js/echarts.min.js',
  '/static/icons/icon-192x192.png',
  '/static/icons/icon-512x512.png',
  '/static/fonts/inter.woff2',
  '/static/fonts/jetbrains-mono.woff2',
  '/manifest.json',
  '/offline',
  '/dashboard',
  '/datasets',
  '/models'
];

// API endpoints to cache with different strategies
const API_ENDPOINTS = {
  GET: [
    '/api/health',
    '/api/models',
    '/api/datasets',
    '/api/dashboard/stats',
    '/api/user/profile'
  ],
  CACHE_FIRST: [
    '/api/algorithms',
    '/api/presets',
    '/api/documentation'
  ],
  NETWORK_FIRST: [
    '/api/detection',
    '/api/analysis',
    '/api/train',
    '/api/predict'
  ]
};

// Background sync tags
const SYNC_TAGS = {
  DETECTION_QUEUE: 'detection-queue',
  UPLOAD_QUEUE: 'upload-queue',
  ANALYSIS_QUEUE: 'analysis-queue',
  METRICS_SYNC: 'metrics-sync'
};

// IndexedDB configuration for offline storage
const DB_NAME = 'PynomaolyOfflineDB';
const DB_VERSION = 1;
const STORES = {
  DETECTIONS: 'detections',
  DATASETS: 'datasets',
  RESULTS: 'results',
  USER_PREFERENCES: 'userPreferences',
  SYNC_QUEUE: 'syncQueue'
};

/**
 * Service Worker Installation
 */
self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('[SW] Static assets cached successfully');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('[SW] Failed to cache static assets:', error);
      })
  );
});

/**
 * Service Worker Activation
 */
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker...');
  
  event.waitUntil(
    Promise.all([
      // Clean up old caches
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== STATIC_CACHE && 
                cacheName !== DYNAMIC_CACHE && 
                cacheName !== API_CACHE) {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      }),
      
      // Initialize IndexedDB
      initializeIndexedDB(),
      
      // Claim clients
      self.clients.claim()
    ])
  );
});

/**
 * Fetch Event Handler - Advanced Network and Cache Strategies
 */
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests for caching (except for background sync)
  if (request.method !== 'GET') {
    if (request.method === 'POST' && isAPIRequest(url.pathname)) {
      event.respondWith(handlePostRequest(request));
    }
    return;
  }
  
  // Static assets - Cache First strategy
  if (isStaticAsset(url.pathname)) {
    event.respondWith(cacheFirstStrategy(request, STATIC_CACHE));
    return;
  }
  
  // API requests - Different strategies based on endpoint
  if (isAPIRequest(url.pathname)) {
    if (isCacheFirstAPI(url.pathname)) {
      event.respondWith(cacheFirstStrategy(request, API_CACHE));
    } else if (isNetworkFirstAPI(url.pathname)) {
      event.respondWith(networkFirstStrategy(request, API_CACHE));
    } else {
      event.respondWith(staleWhileRevalidateStrategy(request, API_CACHE));
    }
    return;
  }
  
  // HTML pages - Network First with offline fallback
  if (request.headers.get('accept')?.includes('text/html')) {
    event.respondWith(networkFirstWithOfflineFallback(request));
    return;
  }
  
  // Default: Network with cache fallback
  event.respondWith(networkWithCacheFallback(request));
});

/**
 * Background Sync Event Handler
 */
self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync event:', event.tag);
  
  switch (event.tag) {
    case SYNC_TAGS.DETECTION_QUEUE:
      event.waitUntil(processDetectionQueue());
      break;
    case SYNC_TAGS.UPLOAD_QUEUE:
      event.waitUntil(processUploadQueue());
      break;
    case SYNC_TAGS.ANALYSIS_QUEUE:
      event.waitUntil(processAnalysisQueue());
      break;
    case SYNC_TAGS.METRICS_SYNC:
      event.waitUntil(syncMetrics());
      break;
  }
});

/**
 * Push Notification Handler
 */
self.addEventListener('push', (event) => {
  console.log('[SW] Push notification received');
  
  const data = event.data ? event.data.json() : {};
  const title = data.title || 'Pynomaly Notification';
  const options = {
    body: data.body || 'You have a new notification',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/badge-72x72.png',
    data: data.data,
    actions: [
      {
        action: 'view',
        title: 'View Details',
        icon: '/static/icons/action-view.png'
      },
      {
        action: 'dismiss',
        title: 'Dismiss',
        icon: '/static/icons/action-dismiss.png'
      }
    ],
    requireInteraction: data.priority === 'high',
    silent: data.priority === 'low',
    timestamp: Date.now(),
    vibrate: [200, 100, 200]
  };
  
  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

/**
 * Notification Click Handler
 */
self.addEventListener('notificationclick', (event) => {
  console.log('[SW] Notification clicked:', event.action);
  
  event.notification.close();
  
  if (event.action === 'view') {
    const urlToOpen = event.notification.data?.url || '/dashboard';
    event.waitUntil(
      clients.matchAll({ type: 'window' })
        .then((clientList) => {
          // Check if app is already open
          for (const client of clientList) {
            if (client.url === urlToOpen && 'focus' in client) {
              return client.focus();
            }
          }
          // Open new window
          if (clients.openWindow) {
            return clients.openWindow(urlToOpen);
          }
        })
    );
  }
});

/**
 * Message Handler for Communication with Main Thread
 */
self.addEventListener('message', (event) => {
  const { type, data } = event.data;
  
  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
    case 'CACHE_URLS':
      event.waitUntil(cacheUrls(data.urls));
      break;
    case 'CLEAR_CACHE':
      event.waitUntil(clearCache(data.cacheName));
      break;
    case 'QUEUE_REQUEST':
      event.waitUntil(queueRequest(data.request, data.tag));
      break;
    case 'GET_CACHE_STATUS':
      event.waitUntil(getCacheStatus().then(status => {
        event.ports[0].postMessage(status);
      }));
      break;
  }
});

// =====================================================
// Caching Strategies
// =====================================================

/**
 * Cache First Strategy - Check cache first, fallback to network
 */
async function cacheFirstStrategy(request, cacheName) {
  try {
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    const networkResponse = await fetch(request);
    if (networkResponse.status === 200) {
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.error('[SW] Cache first strategy failed:', error);
    return new Response('Offline - Content not available', { status: 503 });
  }
}

/**
 * Network First Strategy - Try network first, fallback to cache
 */
async function networkFirstStrategy(request, cacheName) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.status === 200) {
      const cache = await caches.open(cacheName);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('[SW] Network failed, trying cache:', error.message);
    
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    return new Response('Offline - Service unavailable', { status: 503 });
  }
}

/**
 * Stale While Revalidate - Return cache immediately, update in background
 */
async function staleWhileRevalidateStrategy(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(request);
  
  const fetchPromise = fetch(request).then((networkResponse) => {
    if (networkResponse.status === 200) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  }).catch((error) => {
    console.log('[SW] Background fetch failed:', error.message);
  });
  
  // Return cached response immediately if available
  if (cachedResponse) {
    return cachedResponse;
  }
  
  // If no cache, wait for network
  return fetchPromise;
}

/**
 * Network with Cache Fallback
 */
async function networkWithCacheFallback(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.status === 200) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    const cache = await caches.open(DYNAMIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    return new Response('Offline - Resource not available', { status: 503 });
  }
}

/**
 * Network First with Offline Page Fallback for HTML
 */
async function networkFirstWithOfflineFallback(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.status === 200) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    const cache = await caches.open(DYNAMIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return offline page for HTML requests
    const offlineResponse = await cache.match('/offline');
    if (offlineResponse) {
      return offlineResponse;
    }
    
    return createOfflinePage();
  }
}

// =====================================================
// Background Sync Handlers
// =====================================================

/**
 * Process queued detection requests
 */
async function processDetectionQueue() {
  const db = await openIndexedDB();
  const transaction = db.transaction([STORES.SYNC_QUEUE], 'readonly');
  const store = transaction.objectStore(STORES.SYNC_QUEUE);
  const requests = await getAllFromStore(store);
  
  const detectionRequests = requests.filter(req => req.tag === SYNC_TAGS.DETECTION_QUEUE);
  
  for (const queuedRequest of detectionRequests) {
    try {
      const response = await fetch(queuedRequest.url, {
        method: queuedRequest.method,
        headers: queuedRequest.headers,
        body: queuedRequest.body
      });
      
      if (response.ok) {
        await removeFromSyncQueue(queuedRequest.id);
        await saveDetectionResult(await response.json());
        
        notifyClients('DETECTION_COMPLETE', {
          requestId: queuedRequest.id,
          result: await response.json()
        });
      }
    } catch (error) {
      console.error('[SW] Failed to process detection request:', error);
    }
  }
}

/**
 * Process queued upload requests
 */
async function processUploadQueue() {
  // Implementation similar to processDetectionQueue
  console.log('[SW] Processing upload queue...');
}

/**
 * Process queued analysis requests
 */
async function processAnalysisQueue() {
  console.log('[SW] Processing analysis queue...');
}

/**
 * Sync metrics and usage data
 */
async function syncMetrics() {
  try {
    const db = await openIndexedDB();
    const transaction = db.transaction([STORES.USER_PREFERENCES], 'readonly');
    const store = transaction.objectStore(STORES.USER_PREFERENCES);
    const metrics = await getFromStore(store, 'usage_metrics');
    
    if (metrics) {
      const response = await fetch('/api/metrics/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metrics.data)
      });
      
      if (response.ok) {
        await updateUserPreference('usage_metrics', { data: {}, lastSync: Date.now() });
      }
    }
  } catch (error) {
    console.error('[SW] Failed to sync metrics:', error);
  }
}

// =====================================================
// IndexedDB Operations
// =====================================================

/**
 * Initialize IndexedDB
 */
async function initializeIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      
      Object.values(STORES).forEach(storeName => {
        if (!db.objectStoreNames.contains(storeName)) {
          const store = db.createObjectStore(storeName, { keyPath: 'id', autoIncrement: true });
          
          if (storeName === STORES.DETECTIONS) {
            store.createIndex('timestamp', 'timestamp');
            store.createIndex('status', 'status');
          } else if (storeName === STORES.SYNC_QUEUE) {
            store.createIndex('tag', 'tag');
            store.createIndex('timestamp', 'timestamp');
          }
        }
      });
    };
  });
}

/**
 * Open IndexedDB connection
 */
async function openIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

// =====================================================
// Helper Functions
// =====================================================

function isStaticAsset(pathname) {
  return pathname.startsWith('/static/') || 
         pathname.endsWith('.css') || 
         pathname.endsWith('.js') || 
         pathname.endsWith('.png') || 
         pathname.endsWith('.jpg') || 
         pathname.endsWith('.svg') ||
         pathname.endsWith('.woff2') ||
         pathname === '/manifest.json';
}

function isAPIRequest(pathname) {
  return pathname.startsWith('/api/');
}

function isCacheFirstAPI(pathname) {
  return API_ENDPOINTS.CACHE_FIRST.some(endpoint => pathname.startsWith(endpoint));
}

function isNetworkFirstAPI(pathname) {
  return API_ENDPOINTS.NETWORK_FIRST.some(endpoint => pathname.startsWith(endpoint));
}

async function handlePostRequest(request) {
  try {
    const response = await fetch(request);
    return response;
  } catch (error) {
    const requestData = {
      id: Date.now(),
      url: request.url,
      method: request.method,
      headers: Object.fromEntries(request.headers.entries()),
      body: await request.text(),
      timestamp: Date.now(),
      tag: determineSyncTag(request.url)
    };
    
    await queueRequest(requestData, requestData.tag);
    
    return new Response(JSON.stringify({
      message: 'Request queued for background sync',
      requestId: requestData.id
    }), {
      status: 202,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

function determineSyncTag(url) {
  if (url.includes('/detection')) return SYNC_TAGS.DETECTION_QUEUE;
  if (url.includes('/upload')) return SYNC_TAGS.UPLOAD_QUEUE;
  if (url.includes('/analysis')) return SYNC_TAGS.ANALYSIS_QUEUE;
  return SYNC_TAGS.DETECTION_QUEUE;
}

function createOfflinePage() {
  return new Response(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Offline - Pynomaly</title>
      <link rel="stylesheet" href="/static/css/design-system.css">
    </head>
    <body class="bg-gray-50">
      <div class="min-h-screen flex items-center justify-center">
        <div class="card max-w-md text-center">
          <div class="card-body">
            <div class="text-6xl mb-6">📡</div>
            <h1 class="heading-2">You're Offline</h1>
            <p class="body-base mb-6">Please check your internet connection and try again.</p>
            <button onclick="location.reload()" class="btn-base btn-primary">
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    </body>
    </html>
  `, {
    status: 503,
    headers: { 'Content-Type': 'text/html' }
  });
}

function notifyClients(type, data) {
  self.clients.matchAll().then(clients => {
    clients.forEach(client => {
      client.postMessage({ type, data });
    });
  });
}

// Placeholder functions for IndexedDB operations
async function queueRequest(requestData, tag) { /* Implementation */ }
async function removeFromSyncQueue(id) { /* Implementation */ }
async function saveDetectionResult(result) { /* Implementation */ }
async function getAllFromStore(store) { /* Implementation */ }
async function getFromStore(store, key) { /* Implementation */ }
async function updateUserPreference(key, value) { /* Implementation */ }

console.log('[SW] Service worker script loaded successfully');