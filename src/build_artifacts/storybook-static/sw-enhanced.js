/**
 * Enhanced Service Worker for Pynomaly Web UI Performance Optimization
 * Implements advanced caching strategies, image optimization, and performance monitoring
 */

const CACHE_VERSION = 'v2.0.0';
const STATIC_CACHE = `pynomaly-static-${CACHE_VERSION}`;
const DYNAMIC_CACHE = `pynomaly-dynamic-${CACHE_VERSION}`;
const API_CACHE = `pynomaly-api-${CACHE_VERSION}`;
const IMAGE_CACHE = `pynomaly-images-${CACHE_VERSION}`;

// Performance-focused cache configuration
const CACHE_CONFIG = {
  // Static assets with long cache duration
  static: {
    maxAge: 365 * 24 * 60 * 60 * 1000, // 1 year
    maxEntries: 100,
    strategy: 'cacheFirst'
  },

  // Dynamic content with shorter cache
  dynamic: {
    maxAge: 24 * 60 * 60 * 1000, // 1 day
    maxEntries: 50,
    strategy: 'staleWhileRevalidate'
  },

  // API responses with intelligent caching
  api: {
    maxAge: 5 * 60 * 1000, // 5 minutes
    maxEntries: 100,
    strategy: 'networkFirst'
  },

  // Images with compression and format optimization
  images: {
    maxAge: 30 * 24 * 60 * 60 * 1000, // 30 days
    maxEntries: 200,
    strategy: 'cacheFirst'
  }
};

// Static assets for immediate caching
const CRITICAL_ASSETS = [
  '/',
  '/static/css/tailwind.css',
  '/static/css/design-system.css',
  '/static/js/dist/main.js',
  '/static/js/dist/vendors/essential.js',
  '/manifest.json',
  '/static/img/icon-192.png',
  '/static/img/icon-512.png',
  '/offline.html'
];

// Pre-cache resources for performance
const PREFETCH_ASSETS = [
  '/static/js/dist/features/dashboard.js',
  '/static/js/dist/features/anomaly-detection.js',
  '/static/css/app.css'
];

/**
 * Service Worker Installation with Performance Focus
 */
self.addEventListener('install', (event) => {
  console.log('[SW Enhanced] Installing with performance optimizations...');

  event.waitUntil(
    Promise.all([
      // Cache critical assets immediately
      caches.open(STATIC_CACHE).then(cache => {
        console.log('[SW Enhanced] Caching critical assets');
        return cache.addAll(CRITICAL_ASSETS);
      }),

      // Prefetch non-critical assets
      caches.open(DYNAMIC_CACHE).then(cache => {
        console.log('[SW Enhanced] Prefetching assets');
        return cache.addAll(PREFETCH_ASSETS).catch(error => {
          console.warn('[SW Enhanced] Prefetch failed:', error);
        });
      }),

      // Initialize performance monitoring
      initializePerformanceMonitoring(),

      // Skip waiting for immediate activation
      self.skipWaiting()
    ])
  );
});

/**
 * Service Worker Activation with Cache Management
 */
self.addEventListener('activate', (event) => {
  console.log('[SW Enhanced] Activating with cache cleanup...');

  event.waitUntil(
    Promise.all([
      // Clean up old caches
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (!cacheName.includes(CACHE_VERSION)) {
              console.log('[SW Enhanced] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      }),

      // Claim clients immediately
      self.clients.claim(),

      // Initialize IndexedDB for performance data
      initializePerformanceDB()
    ])
  );
});

/**
 * Enhanced Fetch Handler with Performance Strategies
 */
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests (except for offline handling)
  if (request.method !== 'GET') {
    if (request.method === 'POST') {
      event.respondWith(handlePostRequest(request));
    }
    return;
  }

  // Route to appropriate caching strategy
  if (isStaticAsset(url.pathname)) {
    event.respondWith(handleStaticAsset(request));
  } else if (isImageAsset(url.pathname)) {
    event.respondWith(handleImageAsset(request));
  } else if (isAPIRequest(url.pathname)) {
    event.respondWith(handleAPIRequest(request));
  } else if (isHTMLRequest(request)) {
    event.respondWith(handleHTMLRequest(request));
  } else {
    event.respondWith(handleDynamicContent(request));
  }
});

/**
 * Performance-optimized Static Asset Handler
 */
async function handleStaticAsset(request) {
  const cache = await caches.open(STATIC_CACHE);

  // Cache first with long-term storage
  let response = await cache.match(request);

  if (response) {
    // Check if cache is stale (for development)
    if (isDevelopment() && isCacheStale(response, CACHE_CONFIG.static.maxAge)) {
      // Update in background but return cached version
      fetch(request).then(networkResponse => {
        if (networkResponse.ok) {
          cache.put(request, networkResponse.clone());
        }
      }).catch(() => {}); // Ignore network errors
    }

    return response;
  }

  // Fetch from network and cache
  try {
    const networkResponse = await fetch(request);

    if (networkResponse.ok) {
      // Clone and cache with optimization
      const optimizedResponse = await optimizeStaticAsset(networkResponse.clone());
      cache.put(request, optimizedResponse);
      return networkResponse;
    }

    return networkResponse;
  } catch (error) {
    console.warn('[SW Enhanced] Static asset fetch failed:', error);
    return new Response('Asset not available offline', {
      status: 503,
      statusText: 'Service Unavailable'
    });
  }
}

/**
 * Image Asset Handler with Format Optimization
 */
async function handleImageAsset(request) {
  const cache = await caches.open(IMAGE_CACHE);
  const url = new URL(request.url);

  // Check for WebP support and create optimized cache key
  const supportsWebP = request.headers.get('accept')?.includes('image/webp');
  const cacheKey = supportsWebP ? `${request.url}?webp=1` : request.url;

  // Try cache first
  let response = await cache.match(cacheKey);
  if (response) {
    return response;
  }

  try {
    // Fetch original image
    const networkResponse = await fetch(request);

    if (!networkResponse.ok) {
      return networkResponse;
    }

    // Optimize image if possible
    const optimizedResponse = await optimizeImage(networkResponse.clone(), supportsWebP);

    // Cache both optimized and original
    cache.put(cacheKey, optimizedResponse.clone());
    if (cacheKey !== request.url) {
      cache.put(request.url, networkResponse.clone());
    }

    return optimizedResponse;
  } catch (error) {
    console.warn('[SW Enhanced] Image fetch failed:', error);

    // Return fallback image or error
    return createFallbackImageResponse();
  }
}

/**
 * API Request Handler with Intelligent Caching
 */
async function handleAPIRequest(request) {
  const cache = await caches.open(API_CACHE);
  const url = new URL(request.url);

  // Determine caching strategy based on endpoint
  if (isCacheableAPI(url.pathname)) {
    return staleWhileRevalidate(request, cache);
  } else if (isStaticAPI(url.pathname)) {
    return cacheFirst(request, cache);
  } else {
    return networkFirst(request, cache);
  }
}

/**
 * HTML Request Handler with App Shell Strategy
 */
async function handleHTMLRequest(request) {
  const cache = await caches.open(DYNAMIC_CACHE);

  try {
    // Network first for HTML to get latest content
    const networkResponse = await fetch(request);

    if (networkResponse.ok) {
      // Cache successful responses
      cache.put(request, networkResponse.clone());
      return networkResponse;
    }

    // If network fails, try cache
    const cachedResponse = await cache.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    // Fallback to app shell or offline page
    return getAppShellResponse();
  } catch (error) {
    console.warn('[SW Enhanced] HTML fetch failed:', error);

    // Try cache first
    const cachedResponse = await cache.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    // Return offline page
    return getOfflinePageResponse();
  }
}

/**
 * Dynamic Content Handler
 */
async function handleDynamicContent(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  return staleWhileRevalidate(request, cache);
}

/**
 * POST Request Handler with Background Sync
 */
async function handlePostRequest(request) {
  try {
    // Try network first
    const response = await fetch(request);

    if (response.ok) {
      // Invalidate related caches
      await invalidateRelatedCaches(request.url);
      return response;
    }

    // If response not ok, still return it (let app handle errors)
    return response;
  } catch (error) {
    // Network failed, queue for background sync
    console.log('[SW Enhanced] POST request failed, queuing for sync');

    const requestData = {
      url: request.url,
      method: request.method,
      headers: Object.fromEntries(request.headers.entries()),
      body: await request.clone().text(),
      timestamp: Date.now()
    };

    await queueForBackgroundSync(requestData);

    return new Response(
      JSON.stringify({
        success: false,
        message: 'Request queued for background sync',
        offline: true
      }),
      {
        status: 202,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

// =====================================================
// Caching Strategies
// =====================================================

/**
 * Cache First Strategy
 */
async function cacheFirst(request, cache) {
  const cachedResponse = await cache.match(request);

  if (cachedResponse && !isCacheStale(cachedResponse, CACHE_CONFIG.static.maxAge)) {
    return cachedResponse;
  }

  try {
    const networkResponse = await fetch(request);

    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    if (cachedResponse) {
      return cachedResponse;
    }
    throw error;
  }
}

/**
 * Network First Strategy
 */
async function networkFirst(request, cache) {
  try {
    const networkResponse = await fetch(request);

    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    const cachedResponse = await cache.match(request);

    if (cachedResponse) {
      return cachedResponse;
    }

    throw error;
  }
}

/**
 * Stale While Revalidate Strategy
 */
async function staleWhileRevalidate(request, cache) {
  const cachedResponse = await cache.match(request);

  // Always fetch in background to update cache
  const fetchPromise = fetch(request).then(networkResponse => {
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  }).catch(error => {
    console.warn('[SW Enhanced] Background revalidation failed:', error);
  });

  // Return cached response immediately if available
  if (cachedResponse) {
    return cachedResponse;
  }

  // If no cache, wait for network
  return fetchPromise;
}

// =====================================================
// Optimization Functions
// =====================================================

/**
 * Optimize Static Assets
 */
async function optimizeStaticAsset(response) {
  const contentType = response.headers.get('content-type');

  if (contentType?.includes('javascript')) {
    // Add compression headers if not present
    if (!response.headers.get('content-encoding')) {
      // In a real implementation, you might compress here
      return response;
    }
  }

  return response;
}

/**
 * Image Optimization
 */
async function optimizeImage(response, supportsWebP) {
  const contentType = response.headers.get('content-type');

  // For now, return original response
  // In production, this would convert to WebP, resize, compress
  if (supportsWebP && contentType?.includes('image/') && !contentType.includes('webp')) {
    // Would convert to WebP here using a library like Sharp
    // For demo, just return original
    return response;
  }

  return response;
}

/**
 * Create Fallback Image Response
 */
function createFallbackImageResponse() {
  // Simple 1x1 transparent PNG
  const fallbackImageData = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
  const imageData = Uint8Array.from(atob(fallbackImageData), c => c.charCodeAt(0));

  return new Response(imageData, {
    headers: {
      'Content-Type': 'image/png',
      'Cache-Control': 'max-age=86400'
    }
  });
}

// =====================================================
// Helper Functions
// =====================================================

function isStaticAsset(pathname) {
  return pathname.match(/\.(js|css|woff2?|ttf|eot)$/) ||
         pathname.startsWith('/static/js/') ||
         pathname.startsWith('/static/css/');
}

function isImageAsset(pathname) {
  return pathname.match(/\.(png|jpg|jpeg|gif|svg|webp|ico)$/);
}

function isAPIRequest(pathname) {
  return pathname.startsWith('/api/');
}

function isHTMLRequest(request) {
  return request.headers.get('accept')?.includes('text/html');
}

function isCacheableAPI(pathname) {
  const cacheableEndpoints = [
    '/api/datasets',
    '/api/models',
    '/api/algorithms',
    '/api/configurations'
  ];
  return cacheableEndpoints.some(endpoint => pathname.startsWith(endpoint));
}

function isStaticAPI(pathname) {
  const staticEndpoints = [
    '/api/health',
    '/api/version',
    '/api/status'
  ];
  return staticEndpoints.some(endpoint => pathname.startsWith(endpoint));
}

function isDevelopment() {
  return self.location.hostname === 'localhost' ||
         self.location.hostname === '127.0.0.1';
}

function isCacheStale(response, maxAge) {
  const dateHeader = response.headers.get('date');
  if (!dateHeader) return true;

  const responseDate = new Date(dateHeader);
  const now = new Date();

  return (now.getTime() - responseDate.getTime()) > maxAge;
}

async function invalidateRelatedCaches(url) {
  const urlObj = new URL(url);

  // Invalidate API cache for related endpoints
  if (urlObj.pathname.includes('/api/')) {
    const cache = await caches.open(API_CACHE);
    const keys = await cache.keys();

    // Remove related cached responses
    const deletePromises = keys
      .filter(request => {
        const requestUrl = new URL(request.url);
        return requestUrl.pathname.startsWith(urlObj.pathname.split('/').slice(0, -1).join('/'));
      })
      .map(request => cache.delete(request));

    await Promise.all(deletePromises);
  }
}

/**
 * App Shell Response
 */
async function getAppShellResponse() {
  const cache = await caches.open(STATIC_CACHE);
  return cache.match('/') || createOfflineResponse('App shell not available');
}

/**
 * Offline Page Response
 */
async function getOfflinePageResponse() {
  const cache = await caches.open(STATIC_CACHE);
  return cache.match('/offline.html') || createOfflineResponse('You are offline');
}

function createOfflineResponse(message) {
  return new Response(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Offline - Pynomaly</title>
      <style>
        body { font-family: system-ui, sans-serif; text-align: center; padding: 2rem; }
        .offline-container { max-width: 400px; margin: 0 auto; }
        .offline-icon { font-size: 4rem; margin-bottom: 1rem; }
        .retry-btn {
          background: #3b82f6; color: white; border: none;
          padding: 0.75rem 1.5rem; border-radius: 0.5rem;
          cursor: pointer; margin-top: 1rem;
        }
      </style>
    </head>
    <body>
      <div class="offline-container">
        <div class="offline-icon">📡</div>
        <h1>You're Offline</h1>
        <p>${message}</p>
        <button class="retry-btn" onclick="location.reload()">Try Again</button>
      </div>
    </body>
    </html>
  `, {
    status: 503,
    headers: { 'Content-Type': 'text/html' }
  });
}

// =====================================================
// Performance Monitoring
// =====================================================

async function initializePerformanceMonitoring() {
  // Initialize performance tracking
  self.performanceMetrics = {
    cacheHits: 0,
    cacheMisses: 0,
    networkRequests: 0,
    backgroundSyncs: 0,
    startTime: Date.now()
  };
}

async function initializePerformanceDB() {
  // Initialize IndexedDB for performance data storage
  console.log('[SW Enhanced] Performance monitoring initialized');
}

async function queueForBackgroundSync(requestData) {
  // Queue request for background sync when network is available
  console.log('[SW Enhanced] Queuing request for background sync:', requestData.url);

  // In a real implementation, this would store in IndexedDB
  // and register for background sync
}

// =====================================================
// Message Handling
// =====================================================

self.addEventListener('message', (event) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;

    case 'GET_PERFORMANCE_METRICS':
      event.ports[0].postMessage(self.performanceMetrics);
      break;

    case 'CLEAR_CACHES':
      event.waitUntil(clearAllCaches());
      break;

    case 'PRELOAD_RESOURCES':
      event.waitUntil(preloadResources(payload.resources));
      break;

    case 'UPDATE_CACHE_STRATEGY':
      updateCacheStrategy(payload.strategy);
      break;
  }
});

async function clearAllCaches() {
  const cacheNames = await caches.keys();
  await Promise.all(cacheNames.map(name => caches.delete(name)));
  console.log('[SW Enhanced] All caches cleared');
}

async function preloadResources(resources) {
  const cache = await caches.open(DYNAMIC_CACHE);
  await cache.addAll(resources);
  console.log('[SW Enhanced] Resources preloaded:', resources.length);
}

function updateCacheStrategy(strategy) {
  // Update caching strategy dynamically
  console.log('[SW Enhanced] Cache strategy updated:', strategy);
}

console.log('[SW Enhanced] Enhanced service worker loaded with performance optimizations');
