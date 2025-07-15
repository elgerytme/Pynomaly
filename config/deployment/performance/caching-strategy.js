/**
 * Advanced Caching Strategy for Pynomaly Web UI
 * Implements multi-layer caching with intelligent invalidation
 */

/**
 * Cache Configuration
 */
export const CACHE_CONFIG = {
  // Browser caches
  browser: {
    memory: {
      maxSize: 50 * 1024 * 1024, // 50MB
      ttl: 10 * 60 * 1000,       // 10 minutes
    },
    localStorage: {
      maxSize: 10 * 1024 * 1024, // 10MB
      ttl: 24 * 60 * 60 * 1000,  // 24 hours
    },
    sessionStorage: {
      maxSize: 5 * 1024 * 1024,  // 5MB
      ttl: 0, // Session only
    },
    indexedDB: {
      maxSize: 100 * 1024 * 1024, // 100MB
      ttl: 7 * 24 * 60 * 60 * 1000, // 7 days
    }
  },

  // Service Worker caches
  serviceWorker: {
    static: {
      name: 'pynomaly-static',
      maxEntries: 100,
      maxAge: 365 * 24 * 60 * 60 * 1000, // 1 year
    },
    dynamic: {
      name: 'pynomaly-dynamic',
      maxEntries: 50,
      maxAge: 24 * 60 * 60 * 1000, // 1 day
    },
    api: {
      name: 'pynomaly-api',
      maxEntries: 200,
      maxAge: 5 * 60 * 1000, // 5 minutes
    },
    images: {
      name: 'pynomaly-images',
      maxEntries: 300,
      maxAge: 30 * 24 * 60 * 60 * 1000, // 30 days
    }
  },

  // API response caching
  api: {
    // Static data - long cache
    static: {
      endpoints: ['/api/algorithms', '/api/presets'],
      ttl: 60 * 60 * 1000, // 1 hour
      strategy: 'cache-first'
    },

    // Dynamic data - short cache
    dynamic: {
      endpoints: ['/api/datasets', '/api/models'],
      ttl: 5 * 60 * 1000, // 5 minutes
      strategy: 'stale-while-revalidate'
    },

    // Real-time data - no cache
    realtime: {
      endpoints: ['/api/detection', '/api/real-time'],
      ttl: 0,
      strategy: 'network-only'
    }
  }
};

/**
 * Multi-layer Cache Manager
 */
export const createCacheManager = () => `
/**
 * Advanced Cache Manager with Multiple Storage Layers
 */
class CacheManager {
  constructor() {
    this.memoryCache = new Map();
    this.memoryCacheSize = 0;
    this.maxMemorySize = ${CACHE_CONFIG.browser.memory.maxSize};

    this.storageQuota = null;
    this.init();
  }

  async init() {
    // Check storage quota
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      this.storageQuota = await navigator.storage.estimate();
      console.log('💾 Storage quota:', this.storageQuota);
    }

    // Clean up expired entries on init
    this.cleanup();

    // Set up periodic cleanup
    setInterval(() => this.cleanup(), 5 * 60 * 1000); // Every 5 minutes
  }

  /**
   * Get value from cache with fallback strategy
   */
  async get(key, options = {}) {
    const {
      fallback = null,
      ttl = ${CACHE_CONFIG.browser.memory.ttl},
      forceRefresh = false
    } = options;

    if (forceRefresh) {
      await this.delete(key);
      return fallback;
    }

    // Try memory cache first (fastest)
    const memoryResult = this.getFromMemory(key, ttl);
    if (memoryResult) {
      return memoryResult;
    }

    // Try localStorage (fast)
    const localStorageResult = await this.getFromLocalStorage(key, ttl);
    if (localStorageResult) {
      // Store in memory for faster access
      this.setInMemory(key, localStorageResult, ttl);
      return localStorageResult;
    }

    // Try IndexedDB (slower but larger capacity)
    const indexedDBResult = await this.getFromIndexedDB(key, ttl);
    if (indexedDBResult) {
      // Store in faster caches
      this.setInMemory(key, indexedDBResult, ttl);
      this.setInLocalStorage(key, indexedDBResult, ttl);
      return indexedDBResult;
    }

    return fallback;
  }

  /**
   * Set value in cache with intelligent storage selection
   */
  async set(key, value, options = {}) {
    const {
      ttl = ${CACHE_CONFIG.browser.memory.ttl},
      priority = 'normal',
      compress = false
    } = options;

    const dataSize = this.calculateSize(value);
    const cacheEntry = {
      value: compress ? this.compress(value) : value,
      timestamp: Date.now(),
      ttl,
      size: dataSize,
      compressed: compress,
      priority
    };

    // Always store in memory for fast access
    this.setInMemory(key, cacheEntry, ttl);

    // Store in localStorage for medium-term persistence
    if (dataSize < 1024 * 100) { // < 100KB
      await this.setInLocalStorage(key, cacheEntry, ttl);
    }

    // Store in IndexedDB for large data or long-term storage
    if (dataSize > 1024 * 10 || ttl > 60 * 60 * 1000) { // > 10KB or > 1 hour TTL
      await this.setInIndexedDB(key, cacheEntry, ttl);
    }
  }

  /**
   * Memory cache operations
   */
  getFromMemory(key, ttl) {
    const entry = this.memoryCache.get(key);
    if (!entry) return null;

    if (Date.now() - entry.timestamp > ttl) {
      this.memoryCache.delete(key);
      this.memoryCacheSize -= entry.size;
      return null;
    }

    return entry.compressed ? this.decompress(entry.value) : entry.value;
  }

  setInMemory(key, entry, ttl) {
    // Check if we need to evict entries
    if (this.memoryCacheSize + entry.size > this.maxMemorySize) {
      this.evictMemoryEntries(entry.size);
    }

    this.memoryCache.set(key, entry);
    this.memoryCacheSize += entry.size;
  }

  evictMemoryEntries(requiredSize) {
    // LRU eviction with priority consideration
    const entries = Array.from(this.memoryCache.entries())
      .sort((a, b) => {
        // Priority: low priority items first, then by timestamp (oldest first)
        if (a[1].priority === 'low' && b[1].priority !== 'low') return -1;
        if (a[1].priority !== 'low' && b[1].priority === 'low') return 1;
        return a[1].timestamp - b[1].timestamp;
      });

    let freedSize = 0;
    for (const [key, entry] of entries) {
      this.memoryCache.delete(key);
      this.memoryCacheSize -= entry.size;
      freedSize += entry.size;

      if (freedSize >= requiredSize) break;
    }
  }

  /**
   * localStorage operations
   */
  async getFromLocalStorage(key, ttl) {
    try {
      const item = localStorage.getItem(\`cache_\${key}\`);
      if (!item) return null;

      const entry = JSON.parse(item);
      if (Date.now() - entry.timestamp > ttl) {
        localStorage.removeItem(\`cache_\${key}\`);
        return null;
      }

      return entry.compressed ? this.decompress(entry.value) : entry.value;
    } catch (error) {
      console.warn('localStorage cache error:', error);
      return null;
    }
  }

  async setInLocalStorage(key, entry, ttl) {
    try {
      // Check storage quota
      const available = this.getAvailableLocalStorage();
      const required = JSON.stringify(entry).length;

      if (required > available) {
        this.cleanupLocalStorage();
      }

      localStorage.setItem(\`cache_\${key}\`, JSON.stringify(entry));
    } catch (error) {
      if (error.name === 'QuotaExceededError') {
        this.cleanupLocalStorage();
        // Retry once after cleanup
        try {
          localStorage.setItem(\`cache_\${key}\`, JSON.stringify(entry));
        } catch (retryError) {
          console.warn('localStorage quota exceeded after cleanup');
        }
      }
    }
  }

  getAvailableLocalStorage() {
    // Estimate available localStorage space
    const test = 'test';
    let totalSize = 0;

    for (let key in localStorage) {
      if (localStorage.hasOwnProperty(key)) {
        totalSize += localStorage[key].length + key.length;
      }
    }

    return (5 * 1024 * 1024) - totalSize; // Assume 5MB limit
  }

  cleanupLocalStorage() {
    const keys = Object.keys(localStorage).filter(key => key.startsWith('cache_'));
    const entries = keys.map(key => {
      try {
        const entry = JSON.parse(localStorage.getItem(key));
        return { key, entry };
      } catch {
        return null;
      }
    }).filter(Boolean);

    // Sort by timestamp (oldest first)
    entries.sort((a, b) => a.entry.timestamp - b.entry.timestamp);

    // Remove oldest 25%
    const toRemove = Math.ceil(entries.length * 0.25);
    for (let i = 0; i < toRemove; i++) {
      localStorage.removeItem(entries[i].key);
    }
  }

  /**
   * IndexedDB operations
   */
  async getFromIndexedDB(key, ttl) {
    try {
      const db = await this.openIndexedDB();
      const transaction = db.transaction(['cache'], 'readonly');
      const store = transaction.objectStore('cache');
      const request = store.get(key);

      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          const result = request.result;
          if (!result) {
            resolve(null);
            return;
          }

          if (Date.now() - result.timestamp > ttl) {
            // Expired, delete it
            this.deleteFromIndexedDB(key);
            resolve(null);
            return;
          }

          resolve(result.compressed ? this.decompress(result.value) : result.value);
        };

        request.onerror = () => resolve(null);
      });
    } catch (error) {
      console.warn('IndexedDB cache error:', error);
      return null;
    }
  }

  async setInIndexedDB(key, entry, ttl) {
    try {
      const db = await this.openIndexedDB();
      const transaction = db.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');

      store.put({
        id: key,
        ...entry
      });
    } catch (error) {
      console.warn('IndexedDB set error:', error);
    }
  }

  async deleteFromIndexedDB(key) {
    try {
      const db = await this.openIndexedDB();
      const transaction = db.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      store.delete(key);
    } catch (error) {
      console.warn('IndexedDB delete error:', error);
    }
  }

  openIndexedDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('PynomalyCache', 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        const store = db.createObjectStore('cache', { keyPath: 'id' });
        store.createIndex('timestamp', 'timestamp');
      };
    });
  }

  /**
   * Utility methods
   */
  calculateSize(value) {
    return JSON.stringify(value).length;
  }

  compress(value) {
    // Simple compression - in production, use a proper compression library
    return JSON.stringify(value);
  }

  decompress(value) {
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }

  async delete(key) {
    this.memoryCache.delete(key);
    localStorage.removeItem(\`cache_\${key}\`);
    await this.deleteFromIndexedDB(key);
  }

  async clear() {
    this.memoryCache.clear();
    this.memoryCacheSize = 0;

    // Clear localStorage cache entries
    const keys = Object.keys(localStorage).filter(key => key.startsWith('cache_'));
    keys.forEach(key => localStorage.removeItem(key));

    // Clear IndexedDB
    try {
      const db = await this.openIndexedDB();
      const transaction = db.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      store.clear();
    } catch (error) {
      console.warn('IndexedDB clear error:', error);
    }
  }

  cleanup() {
    const now = Date.now();

    // Cleanup memory cache
    for (const [key, entry] of this.memoryCache.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        this.memoryCache.delete(key);
        this.memoryCacheSize -= entry.size;
      }
    }

    // Cleanup localStorage
    const localKeys = Object.keys(localStorage).filter(key => key.startsWith('cache_'));
    localKeys.forEach(key => {
      try {
        const entry = JSON.parse(localStorage.getItem(key));
        if (now - entry.timestamp > entry.ttl) {
          localStorage.removeItem(key);
        }
      } catch {
        localStorage.removeItem(key); // Remove corrupted entries
      }
    });
  }

  getStats() {
    return {
      memory: {
        entries: this.memoryCache.size,
        size: this.memoryCacheSize,
        utilization: (this.memoryCacheSize / this.maxMemorySize) * 100
      },
      localStorage: {
        entries: Object.keys(localStorage).filter(key => key.startsWith('cache_')).length,
        available: this.getAvailableLocalStorage()
      },
      quota: this.storageQuota
    };
  }
}

// Global cache manager instance
window.cacheManager = new CacheManager();
`;

/**
 * API Response Caching
 */
export const createAPICache = () => `
/**
 * Intelligent API Response Caching
 */
class APICache {
  constructor() {
    this.cache = window.cacheManager;
    this.pendingRequests = new Map();
    this.config = ${JSON.stringify(CACHE_CONFIG.api)};
  }

  async get(url, options = {}) {
    const {
      method = 'GET',
      headers = {},
      body = null,
      bypassCache = false,
      cacheStrategy = null
    } = options;

    // Only cache GET requests
    if (method !== 'GET' || bypassCache) {
      return this.fetchFromNetwork(url, options);
    }

    const cacheKey = this.getCacheKey(url, options);
    const strategy = cacheStrategy || this.getStrategy(url);

    switch (strategy) {
      case 'cache-first':
        return this.cacheFirst(cacheKey, url, options);
      case 'network-first':
        return this.networkFirst(cacheKey, url, options);
      case 'stale-while-revalidate':
        return this.staleWhileRevalidate(cacheKey, url, options);
      case 'network-only':
      default:
        return this.fetchFromNetwork(url, options);
    }
  }

  async cacheFirst(cacheKey, url, options) {
    const cached = await this.cache.get(cacheKey);
    if (cached) {
      return cached;
    }

    const response = await this.fetchFromNetwork(url, options);
    if (response && !response.error) {
      const ttl = this.getTTL(url);
      await this.cache.set(cacheKey, response, { ttl });
    }

    return response;
  }

  async networkFirst(cacheKey, url, options) {
    try {
      const response = await this.fetchFromNetwork(url, options);
      if (response && !response.error) {
        const ttl = this.getTTL(url);
        await this.cache.set(cacheKey, response, { ttl });
      }
      return response;
    } catch (error) {
      const cached = await this.cache.get(cacheKey);
      if (cached) {
        console.warn('Network failed, serving cached response:', url);
        return cached;
      }
      throw error;
    }
  }

  async staleWhileRevalidate(cacheKey, url, options) {
    const cached = await this.cache.get(cacheKey);

    // Start network request in background
    const networkPromise = this.fetchFromNetwork(url, options)
      .then(response => {
        if (response && !response.error) {
          const ttl = this.getTTL(url);
          this.cache.set(cacheKey, response, { ttl });
        }
        return response;
      })
      .catch(error => {
        console.warn('Background revalidation failed:', error);
      });

    // Return cached response immediately if available
    if (cached) {
      return cached;
    }

    // If no cache, wait for network
    return networkPromise;
  }

  async fetchFromNetwork(url, options) {
    // Prevent duplicate requests
    if (this.pendingRequests.has(url)) {
      return this.pendingRequests.get(url);
    }

    const requestPromise = fetch(url, options)
      .then(response => {
        if (!response.ok) {
          throw new Error(\`HTTP \${response.status}: \${response.statusText}\`);
        }
        return response.json();
      })
      .finally(() => {
        this.pendingRequests.delete(url);
      });

    this.pendingRequests.set(url, requestPromise);
    return requestPromise;
  }

  getCacheKey(url, options) {
    const { headers = {}, body = null } = options;
    const relevantHeaders = Object.keys(headers)
      .filter(key => key.toLowerCase() !== 'authorization')
      .sort()
      .map(key => \`\${key}:\${headers[key]}\`)
      .join(',');

    return \`api:\${url}:\${relevantHeaders}:\${body || ''}\`;
  }

  getStrategy(url) {
    for (const [type, config] of Object.entries(this.config)) {
      if (config.endpoints.some(endpoint => url.includes(endpoint))) {
        return config.strategy;
      }
    }
    return 'network-first';
  }

  getTTL(url) {
    for (const [type, config] of Object.entries(this.config)) {
      if (config.endpoints.some(endpoint => url.includes(endpoint))) {
        return config.ttl;
      }
    }
    return 5 * 60 * 1000; // Default 5 minutes
  }

  async invalidate(pattern) {
    // Invalidate cache entries matching pattern
    const keys = await this.getCacheKeys();
    const toDelete = keys.filter(key => key.includes(pattern));

    await Promise.all(toDelete.map(key => this.cache.delete(key)));
    console.log(\`Invalidated \${toDelete.length} cache entries matching: \${pattern}\`);
  }

  async getCacheKeys() {
    // Get all cache keys (this would need to be implemented in CacheManager)
    return [];
  }
}

// Global API cache instance
window.apiCache = new APICache();

// Patch fetch to use cache
const originalFetch = window.fetch;
window.fetch = function(url, options = {}) {
  if (url.includes('/api/')) {
    return window.apiCache.get(url, options);
  }
  return originalFetch(url, options);
};
`;

/**
 * Request Deduplication
 */
export const createRequestDeduplication = () => `
/**
 * Request Deduplication to Prevent Duplicate Network Calls
 */
class RequestDeduplicator {
  constructor() {
    this.pendingRequests = new Map();
    this.requestMetrics = {
      total: 0,
      deduplicated: 0,
      saved: 0
    };
  }

  async request(key, requestFn) {
    this.requestMetrics.total++;

    // Check if request is already pending
    if (this.pendingRequests.has(key)) {
      this.requestMetrics.deduplicated++;
      console.log('🔄 Deduplicated request:', key);
      return this.pendingRequests.get(key);
    }

    // Execute request and store promise
    const promise = requestFn()
      .finally(() => {
        this.pendingRequests.delete(key);
      });

    this.pendingRequests.set(key, promise);
    return promise;
  }

  getMetrics() {
    return {
      ...this.requestMetrics,
      deduplicationRate: this.requestMetrics.total > 0 ?
        (this.requestMetrics.deduplicated / this.requestMetrics.total) * 100 : 0
    };
  }

  clear() {
    this.pendingRequests.clear();
  }
}

// Global request deduplicator
window.requestDeduplicator = new RequestDeduplicator();
`;

/**
 * Cache Warming Strategy
 */
export const createCacheWarming = () => `
/**
 * Intelligent Cache Warming
 */
class CacheWarmer {
  constructor() {
    this.warmingQueue = [];
    this.isWarming = false;
    this.priority = {
      critical: [
        '/api/health',
        '/api/user/profile',
        '/api/algorithms'
      ],
      important: [
        '/api/datasets',
        '/api/models',
        '/api/presets'
      ],
      background: [
        '/api/metrics/summary',
        '/api/documentation'
      ]
    };
  }

  async warmCache() {
    if (this.isWarming) return;

    this.isWarming = true;
    console.log('🔥 Starting cache warming...');

    try {
      // Warm critical resources first
      await this.warmPriority('critical');

      // Wait a bit before warming important resources
      await this.delay(1000);
      await this.warmPriority('important');

      // Background warming with longer delay
      await this.delay(5000);
      await this.warmPriority('background');

      console.log('✅ Cache warming completed');
    } catch (error) {
      console.error('❌ Cache warming failed:', error);
    } finally {
      this.isWarming = false;
    }
  }

  async warmPriority(priorityLevel) {
    const urls = this.priority[priorityLevel];
    const promises = urls.map(url => this.warmURL(url));

    await Promise.allSettled(promises);
  }

  async warmURL(url) {
    try {
      await window.apiCache.get(url, {
        cacheStrategy: 'cache-first'
      });
      console.log('🔥 Warmed:', url);
    } catch (error) {
      console.warn('⚠️ Failed to warm:', url, error);
    }
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  addToQueue(url, priority = 'background') {
    if (!this.priority[priority].includes(url)) {
      this.priority[priority].push(url);
    }
  }

  // Warm cache based on user behavior
  warmBasedOnRoute(route) {
    const routeMap = {
      '/dashboard': ['/api/metrics/summary', '/api/anomalies/recent'],
      '/datasets': ['/api/datasets', '/api/datasets/recent'],
      '/models': ['/api/models', '/api/algorithms'],
      '/settings': ['/api/user/profile', '/api/configurations']
    };

    if (routeMap[route]) {
      routeMap[route].forEach(url => {
        this.warmURL(url);
      });
    }
  }
}

// Global cache warmer
window.cacheWarmer = new CacheWarmer();

// Start warming on page load
document.addEventListener('DOMContentLoaded', () => {
  // Delay cache warming to not interfere with critical page load
  setTimeout(() => {
    window.cacheWarmer.warmCache();
  }, 2000);
});
`;

export { CACHE_CONFIG };
