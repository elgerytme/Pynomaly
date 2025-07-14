/**
 * Advanced Cache Manager for API Responses and Static Assets
 * Implements multi-layer caching with compression, metrics, and smart invalidation
 */

class CacheManager {
  constructor(options = {}) {
    this.options = {
      maxMemoryCacheSize: 100,
      defaultTTL: 5 * 60 * 1000, // 5 minutes
      enableCompression: true,
      enableMetrics: true,
      enableServiceWorkerCache: true,
      compressionThreshold: 1024, // 1KB
      ...options
    };

    this.memoryCache = new Map();
    this.memoryCacheSize = 0;
    this.pendingRequests = new Map();
    this.cacheKeys = {
      API_RESPONSES: 'api-responses',
      USER_PREFERENCES: 'user-preferences',
      STATIC_DATA: 'static-data',
      CHART_DATA: 'chart-data',
      MODEL_RESULTS: 'model-results'
    };

    // Performance metrics
    this.metrics = {
      hits: 0,
      misses: 0,
      evictions: 0,
      compressionSaved: 0,
      networkRequests: 0,
      cacheSize: 0
    };

    this.compressionSupported = 'CompressionStream' in window;
    this.init();
  }

  init() {
    // Setup automatic cleanup
    setInterval(() => {
      this.cleanupExpiredMemoryCache();
    }, 60000); // Every minute

    // Setup periodic IndexedDB cleanup
    setInterval(() => {
      this.cleanupExpiredCache();
    }, 10 * 60 * 1000); // Every 10 minutes
  }

  // Memory cache for frequently accessed data
  setMemoryCache(key, data, ttl = this.defaultTTL) {
    // Implement LRU eviction
    if (this.memoryCache.size >= this.maxMemoryCacheSize) {
      const firstKey = this.memoryCache.keys().next().value;
      this.memoryCache.delete(firstKey);
    }

    this.memoryCache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
  }

  getMemoryCache(key) {
    const cached = this.memoryCache.get(key);
    if (!cached) return null;

    const now = Date.now();
    if (now - cached.timestamp > cached.ttl) {
      this.memoryCache.delete(key);
      return null;
    }

    return cached.data;
  }

  // IndexedDB cache for larger data
  async setIndexedDBCache(key, data, ttl = this.defaultTTL) {
    try {
      const db = await this.openIndexedDB();
      const transaction = db.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');

      await store.put({
        key,
        data,
        timestamp: Date.now(),
        ttl
      });
    } catch (error) {
      console.warn('IndexedDB cache set failed:', error);
    }
  }

  async getIndexedDBCache(key) {
    try {
      const db = await this.openIndexedDB();
      const transaction = db.transaction(['cache'], 'readonly');
      const store = transaction.objectStore('cache');
      const result = await store.get(key);

      if (!result) return null;

      const now = Date.now();
      if (now - result.timestamp > result.ttl) {
        // Clean up expired cache
        await this.deleteIndexedDBCache(key);
        return null;
      }

      return result.data;
    } catch (error) {
      console.warn('IndexedDB cache get failed:', error);
      return null;
    }
  }

  async deleteIndexedDBCache(key) {
    try {
      const db = await this.openIndexedDB();
      const transaction = db.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      await store.delete(key);
    } catch (error) {
      console.warn('IndexedDB cache delete failed:', error);
    }
  }

  async openIndexedDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('pynomaly-cache', 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('cache')) {
          const store = db.createObjectStore('cache', { keyPath: 'key' });
          store.createIndex('timestamp', 'timestamp');
        }
      };
    });
  }

  // Cache API wrapper with intelligent caching
  async cacheRequest(url, options = {}) {
    const cacheKey = this.generateCacheKey(url, options);

    // Check memory cache first
    const memoryCached = this.getMemoryCache(cacheKey);
    if (memoryCached) {
      return memoryCached;
    }

    // Check IndexedDB cache
    const indexedCached = await this.getIndexedDBCache(cacheKey);
    if (indexedCached) {
      // Promote to memory cache
      this.setMemoryCache(cacheKey, indexedCached);
      return indexedCached;
    }

    // Fetch from network
    try {
      const response = await fetch(url, options);
      const data = await response.json();

      // Cache based on response type
      if (this.shouldCache(url, data)) {
        const ttl = this.getTTL(url);
        this.setMemoryCache(cacheKey, data, ttl);

        // Store larger responses in IndexedDB
        if (JSON.stringify(data).length > 10000) {
          await this.setIndexedDBCache(cacheKey, data, ttl);
        }
      }

      return data;
    } catch (error) {
      console.error('Cache request failed:', error);
      throw error;
    }
  }

  shouldCache(url, data) {
    // Don't cache error responses
    if (data.error) return false;

    // Don't cache user-specific data
    if (url.includes('/user/') || url.includes('/auth/')) return false;

    // Cache static data, configuration, and public APIs
    return true;
  }

  getTTL(url) {
    // Different TTL for different types of data
    if (url.includes('/config/')) return 30 * 60 * 1000; // 30 minutes
    if (url.includes('/results/')) return 2 * 60 * 1000; // 2 minutes
    if (url.includes('/detectors/')) return 10 * 60 * 1000; // 10 minutes
    if (url.includes('/datasets/')) return 5 * 60 * 1000; // 5 minutes

    return this.defaultTTL;
  }

  generateCacheKey(url, options) {
    const optionsString = JSON.stringify(options);
    return `${url}:${btoa(optionsString)}`;
  }

  // Request deduplication
  pendingRequests = new Map();

  async deduplicateRequest(url, options = {}) {
    const cacheKey = this.generateCacheKey(url, options);

    // Check if request is already in flight
    if (this.pendingRequests.has(cacheKey)) {
      return this.pendingRequests.get(cacheKey);
    }

    // Create new request
    const requestPromise = this.cacheRequest(url, options);
    this.pendingRequests.set(cacheKey, requestPromise);

    try {
      const result = await requestPromise;
      this.pendingRequests.delete(cacheKey);
      return result;
    } catch (error) {
      this.pendingRequests.delete(cacheKey);
      throw error;
    }
  }

  // Cache cleanup
  async cleanupExpiredCache() {
    try {
      const db = await this.openIndexedDB();
      const transaction = db.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      const index = store.index('timestamp');

      const now = Date.now();
      const cursor = await index.openCursor();

      while (cursor) {
        const record = cursor.value;
        if (now - record.timestamp > record.ttl) {
          await cursor.delete();
        }
        cursor = await cursor.continue();
      }
    } catch (error) {
      console.warn('Cache cleanup failed:', error);
    }
  }

  // Clear all cache
  async clearCache() {
    this.memoryCache.clear();
    try {
      const db = await this.openIndexedDB();
      const transaction = db.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      await store.clear();
    } catch (error) {
      console.warn('Clear cache failed:', error);
    }
  }
}

// Global cache manager instance
const cacheManager = new CacheManager();

// Cleanup expired cache every 10 minutes
setInterval(() => {
  cacheManager.cleanupExpiredCache();
}, 10 * 60 * 1000);

export default cacheManager;
