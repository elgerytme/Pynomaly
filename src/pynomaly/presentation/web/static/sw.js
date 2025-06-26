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

// Fetch event - handle requests with caching strategy
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip non-http(s) requests
  if (!url.protocol.startsWith('http')) {
    return;
  }
  
  // Handle different types of requests
  if (isStaticAsset(request)) {
    event.respondWith(handleStaticAsset(request));
  } else if (isAPIRequest(request)) {
    event.respondWith(handleAPIRequest(request));
  } else if (isHTMLRequest(request)) {
    event.respondWith(handleHTMLRequest(request));
  }
});

// Check if request is for static assets
function isStaticAsset(request) {
  const url = new URL(request.url);
  return url.pathname.startsWith('/static/') ||
         url.pathname.endsWith('.css') ||
         url.pathname.endsWith('.js') ||
         url.pathname.endsWith('.png') ||
         url.pathname.endsWith('.jpg') ||
         url.pathname.endsWith('.svg');
}

// Check if request is for API endpoints
function isAPIRequest(request) {
  const url = new URL(request.url);
  return url.pathname.startsWith('/api/');
}

// Check if request is for HTML pages
function isHTMLRequest(request) {
  return request.headers.get('accept')?.includes('text/html');
}

// Handle static assets with cache-first strategy
async function handleStaticAsset(request) {
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    return new Response('Asset not available offline', {
      status: 503,
      statusText: 'Service Unavailable'
    });
  }
}

// Handle API requests with network-first strategy
async function handleAPIRequest(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    return new Response(JSON.stringify({
      error: 'Offline',
      message: 'This feature requires an internet connection'
    }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

// Handle HTML requests with network-first, fallback to cache
async function handleHTMLRequest(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    return new Response(`
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Offline - Pynomaly</title>
        <style>
          body { font-family: system-ui, sans-serif; text-align: center; padding: 2rem; }
          .offline-message { max-width: 400px; margin: 2rem auto; }
          .icon { font-size: 4rem; margin-bottom: 1rem; }
        </style>
      </head>
      <body>
        <div class="offline-message">
          <div class="icon">📡</div>
          <h1>You're Offline</h1>
          <p>Please check your internet connection and try again.</p>
          <button onclick="location.reload()">Retry</button>
        </div>
      </body>
      </html>
    `, {
      status: 503,
      headers: { 'Content-Type': 'text/html' }
    });
  }
}

console.log('Service Worker: Script loaded successfully');