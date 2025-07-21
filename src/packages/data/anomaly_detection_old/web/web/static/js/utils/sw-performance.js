/**
 * Service Worker Performance Optimizations
 * Enhances the existing service worker with advanced performance features
 */

// Cache strategy configurations for different resource types
const CACHE_STRATEGIES = {
  CRITICAL: {
    name: 'pynomaly-critical-v1.3.0',
    strategy: 'cache-first',
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    maxEntries: 50,
  },
  STATIC: {
    name: 'pynomaly-static-v1.3.0',
    strategy: 'cache-first',
    maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    maxEntries: 200,
  },
  API: {
    name: 'pynomaly-api-v1.3.0',
    strategy: 'network-first',
    maxAge: 5 * 60 * 1000, // 5 minutes
    maxEntries: 100,
    networkTimeout: 3000,
  },
  DYNAMIC: {
    name: 'pynomaly-dynamic-v1.3.0',
    strategy: 'stale-while-revalidate',
    maxAge: 60 * 60 * 1000, // 1 hour
    maxEntries: 500,
  },
};

// Performance-optimized cache implementation
class PerformanceCache {
  constructor(config) {
    this.config = config;
    this.cache = null;
    this.requestQueue = new Map();
    this.performanceMetrics = {
      hits: 0,
      misses: 0,
      networkRequests: 0,
      cacheWrites: 0,
      averageResponseTime: 0,
    };
  }

  async init() {
    this.cache = await caches.open(this.config.name);
    await this.cleanup();
  }

  async get(request) {
    const startTime = performance.now();
    const cacheKey = this.getCacheKey(request);

    try {
      // Check if request is already in progress
      if (this.requestQueue.has(cacheKey)) {
        return await this.requestQueue.get(cacheKey);
      }

      let response;

      switch (this.config.strategy) {
        case 'cache-first':
          response = await this.cacheFirst(request);
          break;
        case 'network-first':
          response = await this.networkFirst(request);
          break;
        case 'stale-while-revalidate':
          response = await this.staleWhileRevalidate(request);
          break;
        default:
          response = await this.networkFirst(request);
      }

      const duration = performance.now() - startTime;
      this.updateMetrics(duration, response ? 'hit' : 'miss');

      return response;

    } catch (error) {
      console.error('Cache get failed:', error);
      return null;
    }
  }

  async cacheFirst(request) {
    const cachedResponse = await this.cache.match(request);

    if (cachedResponse && !this.isExpired(cachedResponse)) {
      this.performanceMetrics.hits++;
      return cachedResponse;
    }

    this.performanceMetrics.misses++;
    return await this.fetchAndCache(request);
  }

  async networkFirst(request) {
    const cacheKey = this.getCacheKey(request);

    // Create a promise for this request
    const networkPromise = this.fetchWithTimeout(request, this.config.networkTimeout)
      .then(response => {
        if (response && response.ok) {
          this.cache.put(request, response.clone());
          this.performanceMetrics.cacheWrites++;
        }
        return response;
      })
      .catch(async () => {
        // Network failed, try cache
        const cachedResponse = await this.cache.match(request);
        if (cachedResponse) {
          this.performanceMetrics.hits++;
          return cachedResponse;
        }
        throw new Error('Network and cache both failed');
      });

    this.requestQueue.set(cacheKey, networkPromise);

    try {
      const response = await networkPromise;
      this.performanceMetrics.networkRequests++;
      return response;
    } finally {
      this.requestQueue.delete(cacheKey);
    }
  }

  async staleWhileRevalidate(request) {
    const cachedResponse = await this.cache.match(request);

    // Always try to update in background
    const networkPromise = this.fetchAndCache(request).catch(() => {
      // Ignore network errors for SWR
    });

    if (cachedResponse && !this.isExpired(cachedResponse)) {
      this.performanceMetrics.hits++;
      // Don't await the network request
      networkPromise;
      return cachedResponse;
    }

    // No cache or expired, wait for network
    this.performanceMetrics.misses++;
    return await networkPromise;
  }

  async fetchAndCache(request) {
    try {
      const response = await fetch(request);

      if (response && response.ok) {
        // Clone before caching
        const responseToCache = response.clone();

        // Add timestamp for expiration checking
        const headers = new Headers(responseToCache.headers);
        headers.set('sw-cached-at', Date.now().toString());

        const cachedResponse = new Response(responseToCache.body, {
          status: responseToCache.status,
          statusText: responseToCache.statusText,
          headers: headers,
        });

        await this.cache.put(request, cachedResponse);
        this.performanceMetrics.cacheWrites++;
      }

      this.performanceMetrics.networkRequests++;
      return response;

    } catch (error) {
      console.error('Fetch failed:', error);
      throw error;
    }
  }

  async fetchWithTimeout(request, timeout) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(request, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  isExpired(response) {
    const cachedAt = response.headers.get('sw-cached-at');
    if (!cachedAt) return false;

    const age = Date.now() - parseInt(cachedAt);
    return age > this.config.maxAge;
  }

  getCacheKey(request) {
    return `${request.method}:${request.url}`;
  }

  async cleanup() {
    const keys = await this.cache.keys();

    if (keys.length <= this.config.maxEntries) return;

    // Sort by last accessed time (if available) or by URL
    const sortedKeys = keys.sort((a, b) => {
      // Simple LRU approximation using URL as tie-breaker
      return a.url.localeCompare(b.url);
    });

    // Remove oldest entries
    const entriesToRemove = sortedKeys.slice(0, keys.length - this.config.maxEntries);

    await Promise.all(
      entriesToRemove.map(key => this.cache.delete(key))
    );
  }

  updateMetrics(duration, type) {
    this.performanceMetrics.averageResponseTime =
      (this.performanceMetrics.averageResponseTime + duration) / 2;
  }

  getMetrics() {
    const total = this.performanceMetrics.hits + this.performanceMetrics.misses;
    const hitRate = total > 0 ? (this.performanceMetrics.hits / total) * 100 : 0;

    return {
      ...this.performanceMetrics,
      hitRate: hitRate.toFixed(2),
    };
  }
}

// Resource classifier for intelligent caching
class ResourceClassifier {
  static classify(request) {
    const url = new URL(request.url);
    const pathname = url.pathname;
    const method = request.method;

    // Critical resources that need immediate caching
    if (this.isCriticalResource(pathname)) {
      return 'CRITICAL';
    }

    // Static assets
    if (this.isStaticAsset(pathname)) {
      return 'STATIC';
    }

    // API requests
    if (this.isAPIRequest(pathname)) {
      return method === 'GET' ? 'API' : null; // Only cache GET requests
    }

    // Dynamic content
    if (this.isDynamicContent(pathname)) {
      return 'DYNAMIC';
    }

    return null; // Don't cache
  }

  static isCriticalResource(pathname) {
    const criticalPatterns = [
      /^\/static\/css\/design-system\.css$/,
      /^\/static\/js\/main.*\.js$/,
      /^\/static\/js\/state\/dashboard-state\.js$/,
      /^\/static\/js\/utils\/performance-optimizer\.js$/,
    ];

    return criticalPatterns.some(pattern => pattern.test(pathname));
  }

  static isStaticAsset(pathname) {
    const staticPatterns = [
      /^\/static\//,
      /\.(css|js|png|jpg|jpeg|gif|svg|woff|woff2|ttf|ico)$/,
      /^\/manifest\.json$/,
    ];

    return staticPatterns.some(pattern => pattern.test(pathname));
  }

  static isAPIRequest(pathname) {
    return pathname.startsWith('/api/');
  }

  static isDynamicContent(pathname) {
    const dynamicPatterns = [
      /^\/dashboard/,
      /^\/charts/,
      /^\/reports/,
    ];

    return dynamicPatterns.some(pattern => pattern.test(pathname));
  }
}

// Background sync with intelligent queueing
class SmartBackgroundSync {
  constructor() {
    this.syncQueue = new Map();
    this.syncInProgress = new Set();
    this.retryDelays = [1000, 5000, 15000, 60000]; // Progressive delays
  }

  async queueRequest(request, tag, priority = 'normal') {
    const id = this.generateRequestId(request);
    const queueItem = {
      id,
      request: await this.serializeRequest(request),
      tag,
      priority,
      timestamp: Date.now(),
      retryCount: 0,
    };

    this.syncQueue.set(id, queueItem);

    // Try to sync immediately if online
    if (navigator.onLine) {
      this.processQueue();
    }

    return id;
  }

  async processQueue() {
    if (this.syncInProgress.size > 3) { // Limit concurrent syncs
      return;
    }

    const sortedItems = Array.from(this.syncQueue.values())
      .filter(item => !this.syncInProgress.has(item.id))
      .sort((a, b) => {
        // Sort by priority then by timestamp
        const priorityOrder = { high: 3, normal: 2, low: 1 };
        const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];

        if (priorityDiff !== 0) return priorityDiff;
        return a.timestamp - b.timestamp;
      });

    const itemsToProcess = sortedItems.slice(0, 3 - this.syncInProgress.size);

    await Promise.all(
      itemsToProcess.map(item => this.processSyncItem(item))
    );
  }

  async processSyncItem(item) {
    this.syncInProgress.add(item.id);

    try {
      const request = await this.deserializeRequest(item.request);
      const response = await fetch(request);

      if (response.ok) {
        // Success - remove from queue
        this.syncQueue.delete(item.id);
        this.postMessage({
          type: 'sync-success',
          id: item.id,
          tag: item.tag,
        });
      } else {
        throw new Error(`HTTP ${response.status}`);
      }

    } catch (error) {
      // Retry with exponential backoff
      item.retryCount++;

      if (item.retryCount < this.retryDelays.length) {
        const delay = this.retryDelays[item.retryCount - 1];
        setTimeout(() => {
          if (this.syncQueue.has(item.id)) {
            this.processSyncItem(item);
          }
        }, delay);
      } else {
        // Max retries reached
        this.syncQueue.delete(item.id);
        this.postMessage({
          type: 'sync-failed',
          id: item.id,
          tag: item.tag,
          error: error.message,
        });
      }
    } finally {
      this.syncInProgress.delete(item.id);
    }
  }

  generateRequestId(request) {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  async serializeRequest(request) {
    return {
      url: request.url,
      method: request.method,
      headers: Object.fromEntries(request.headers.entries()),
      body: request.body ? await request.text() : null,
    };
  }

  async deserializeRequest(serialized) {
    return new Request(serialized.url, {
      method: serialized.method,
      headers: serialized.headers,
      body: serialized.body,
    });
  }

  postMessage(message) {
    // Send message to all clients
    self.clients.matchAll().then(clients => {
      clients.forEach(client => {
        client.postMessage(message);
      });
    });
  }
}

// Performance monitoring for service worker
class ServiceWorkerPerformanceMonitor {
  constructor() {
    this.metrics = {
      cacheHits: 0,
      cacheMisses: 0,
      networkRequests: 0,
      backgroundSyncs: 0,
      averageResponseTime: 0,
      errorRate: 0,
    };
    this.responseTimeHistory = [];
    this.errorCount = 0;
    this.totalRequests = 0;
  }

  recordCacheHit() {
    this.metrics.cacheHits++;
  }

  recordCacheMiss() {
    this.metrics.cacheMisses++;
  }

  recordNetworkRequest(duration) {
    this.metrics.networkRequests++;
    this.totalRequests++;
    this.recordResponseTime(duration);
  }

  recordError() {
    this.errorCount++;
    this.totalRequests++;
    this.updateErrorRate();
  }

  recordBackgroundSync() {
    this.metrics.backgroundSyncs++;
  }

  recordResponseTime(duration) {
    this.responseTimeHistory.push(duration);

    // Keep only last 100 measurements
    if (this.responseTimeHistory.length > 100) {
      this.responseTimeHistory = this.responseTimeHistory.slice(-100);
    }

    this.metrics.averageResponseTime =
      this.responseTimeHistory.reduce((sum, time) => sum + time, 0) /
      this.responseTimeHistory.length;
  }

  updateErrorRate() {
    this.metrics.errorRate = (this.errorCount / this.totalRequests) * 100;
  }

  getMetrics() {
    const total = this.metrics.cacheHits + this.metrics.cacheMisses;
    const cacheHitRate = total > 0 ? (this.metrics.cacheHits / total) * 100 : 0;

    return {
      ...this.metrics,
      cacheHitRate: cacheHitRate.toFixed(2),
      totalRequests: this.totalRequests,
    };
  }

  reset() {
    this.metrics = {
      cacheHits: 0,
      cacheMisses: 0,
      networkRequests: 0,
      backgroundSyncs: 0,
      averageResponseTime: 0,
      errorRate: 0,
    };
    this.responseTimeHistory = [];
    this.errorCount = 0;
    this.totalRequests = 0;
  }
}

// Export classes for use in service worker
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    PerformanceCache,
    ResourceClassifier,
    SmartBackgroundSync,
    ServiceWorkerPerformanceMonitor,
    CACHE_STRATEGIES,
  };
}

// For use in service worker context
if (typeof self !== 'undefined' && self.constructor.name === 'ServiceWorkerGlobalScope') {
  self.PerformanceCache = PerformanceCache;
  self.ResourceClassifier = ResourceClassifier;
  self.SmartBackgroundSync = SmartBackgroundSync;
  self.ServiceWorkerPerformanceMonitor = ServiceWorkerPerformanceMonitor;
  self.CACHE_STRATEGIES = CACHE_STRATEGIES;
}
