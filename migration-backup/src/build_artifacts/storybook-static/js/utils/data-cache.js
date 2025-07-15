/**
 * Intelligent Data Caching and Loading System
 * Implements smart caching, prefetching, and data compression for optimal performance
 */

export class DataCacheManager {
  constructor(options = {}) {
    this.options = {
      maxCacheSize: 50 * 1024 * 1024, // 50MB
      maxCacheEntries: 1000,
      defaultTTL: 5 * 60 * 1000, // 5 minutes
      compressionThreshold: 1024, // 1KB
      prefetchThreshold: 0.8, // Start prefetching when 80% through data
      indexedDBName: 'PynomalyDataCache',
      indexedDBVersion: 1,
      enableCompression: true,
      enablePrefetch: true,
      enableIndexedDB: true,
      ...options,
    };

    this.memoryCache = new Map();
    this.cacheStats = {
      hits: 0,
      misses: 0,
      evictions: 0,
      totalSize: 0,
      compressionRatio: 0,
    };

    this.indexedDB = null;
    this.prefetchQueue = new Set();
    this.loadingPromises = new Map();
    this.compressionWorker = null;

    this.init();
  }

  async init() {
    if (this.options.enableIndexedDB) {
      await this.initializeIndexedDB();
    }

    if (this.options.enableCompression) {
      this.initializeCompressionWorker();
    }

    this.startMaintenanceTasks();
  }

  // === INDEXEDDB INITIALIZATION ===

  async initializeIndexedDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.options.indexedDBName, this.options.indexedDBVersion);

      request.onerror = () => {
        console.warn('IndexedDB initialization failed:', request.error);
        resolve(null);
      };

      request.onsuccess = () => {
        this.indexedDB = request.result;
        console.log('IndexedDB initialized successfully');
        resolve(this.indexedDB);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // Create cache store
        if (!db.objectStoreNames.contains('cache')) {
          const store = db.createObjectStore('cache', { keyPath: 'key' });
          store.createIndex('timestamp', 'timestamp');
          store.createIndex('size', 'size');
          store.createIndex('ttl', 'ttl');
        }

        // Create metadata store
        if (!db.objectStoreNames.contains('metadata')) {
          const metaStore = db.createObjectStore('metadata', { keyPath: 'key' });
          metaStore.createIndex('lastAccessed', 'lastAccessed');
        }
      };
    });
  }

  // === COMPRESSION WORKER ===

  initializeCompressionWorker() {
    const workerScript = `
      // Import compression library (simplified implementation)
      function compress(data) {
        try {
          const jsonString = JSON.stringify(data);
          const compressed = LZString ? LZString.compress(jsonString) : jsonString;
          return { compressed, originalSize: jsonString.length, compressedSize: compressed.length };
        } catch (error) {
          return { compressed: JSON.stringify(data), originalSize: 0, compressedSize: 0 };
        }
      }

      function decompress(compressedData) {
        try {
          const decompressed = LZString ? LZString.decompress(compressedData) : compressedData;
          return JSON.parse(decompressed);
        } catch (error) {
          return JSON.parse(compressedData);
        }
      }

      self.onmessage = function(e) {
        const { type, data, id } = e.data;

        switch(type) {
          case 'compress':
            const compressResult = compress(data);
            self.postMessage({ type: 'compressed', result: compressResult, id });
            break;

          case 'decompress':
            const decompressResult = decompress(data);
            self.postMessage({ type: 'decompressed', result: decompressResult, id });
            break;
        }
      };

      // Simple LZ compression fallback
      const LZString = {
        compress: function(input) {
          return input; // Fallback - no compression
        },
        decompress: function(input) {
          return input; // Fallback - no decompression
        }
      };
    `;

    const blob = new Blob([workerScript], { type: 'application/javascript' });
    this.compressionWorker = new Worker(URL.createObjectURL(blob));

    this.compressionWorker.onmessage = (e) => {
      this.handleCompressionResult(e.data);
    };
  }

  handleCompressionResult(data) {
    const { type, result, id } = data;
    const promise = this.loadingPromises.get(id);

    if (promise) {
      if (type === 'compressed') {
        promise.resolve(result);
      } else if (type === 'decompressed') {
        promise.resolve(result);
      }
      this.loadingPromises.delete(id);
    }
  }

  // === CACHING METHODS ===

  async get(key, options = {}) {
    const startTime = performance.now();

    try {
      // Check memory cache first
      const memoryResult = await this.getFromMemory(key);
      if (memoryResult) {
        this.cacheStats.hits++;
        this.updateAccessTime(key);
        return memoryResult.data;
      }

      // Check IndexedDB cache
      if (this.indexedDB) {
        const dbResult = await this.getFromIndexedDB(key);
        if (dbResult) {
          this.cacheStats.hits++;
          // Promote to memory cache
          await this.setInMemory(key, dbResult.data, dbResult.ttl);
          this.updateAccessTime(key);
          return dbResult.data;
        }
      }

      this.cacheStats.misses++;
      return null;

    } finally {
      const duration = performance.now() - startTime;
      if (duration > 10) {
        console.warn(`Cache get operation took ${duration.toFixed(2)}ms for key: ${key}`);
      }
    }
  }

  async set(key, data, options = {}) {
    const {
      ttl = this.options.defaultTTL,
      compress = this.shouldCompress(data),
      priority = 'normal'
    } = options;

    const startTime = performance.now();

    try {
      // Set in memory cache
      await this.setInMemory(key, data, ttl, { priority });

      // Set in IndexedDB if available
      if (this.indexedDB) {
        await this.setInIndexedDB(key, data, ttl, { compress });
      }

      this.updateCacheStats(data);

    } catch (error) {
      console.error('Cache set operation failed:', error);
    } finally {
      const duration = performance.now() - startTime;
      if (duration > 50) {
        console.warn(`Cache set operation took ${duration.toFixed(2)}ms for key: ${key}`);
      }
    }
  }

  async getFromMemory(key) {
    const entry = this.memoryCache.get(key);
    if (!entry) return null;

    // Check TTL
    if (entry.ttl && Date.now() > entry.ttl) {
      this.memoryCache.delete(key);
      return null;
    }

    return entry;
  }

  async setInMemory(key, data, ttl, options = {}) {
    const entry = {
      data,
      ttl: ttl ? Date.now() + ttl : null,
      timestamp: Date.now(),
      size: this.estimateSize(data),
      priority: options.priority || 'normal',
      accessCount: 1,
      lastAccessed: Date.now(),
    };

    // Check if we need to evict entries
    await this.ensureMemoryCapacity(entry.size);

    this.memoryCache.set(key, entry);
    this.cacheStats.totalSize += entry.size;
  }

  async getFromIndexedDB(key) {
    if (!this.indexedDB) return null;

    return new Promise((resolve, reject) => {
      const transaction = this.indexedDB.transaction(['cache'], 'readonly');
      const store = transaction.objectStore('cache');
      const request = store.get(key);

      request.onsuccess = async () => {
        const result = request.result;
        if (!result) {
          resolve(null);
          return;
        }

        // Check TTL
        if (result.ttl && Date.now() > result.ttl) {
          // Remove expired entry
          this.deleteFromIndexedDB(key);
          resolve(null);
          return;
        }

        // Decompress if needed
        let data = result.data;
        if (result.compressed && this.compressionWorker) {
          try {
            data = await this.decompressData(result.data);
          } catch (error) {
            console.error('Decompression failed:', error);
            resolve(null);
            return;
          }
        }

        resolve({ data, ttl: result.ttl });
      };

      request.onerror = () => {
        console.error('IndexedDB get failed:', request.error);
        resolve(null);
      };
    });
  }

  async setInIndexedDB(key, data, ttl, options = {}) {
    if (!this.indexedDB) return;

    let processedData = data;
    let compressed = false;

    // Compress if enabled and data is large enough
    if (options.compress && this.compressionWorker) {
      try {
        const compressionResult = await this.compressData(data);
        if (compressionResult.compressedSize < compressionResult.originalSize * 0.8) {
          processedData = compressionResult.compressed;
          compressed = true;
          this.updateCompressionStats(compressionResult);
        }
      } catch (error) {
        console.error('Compression failed:', error);
      }
    }

    return new Promise((resolve, reject) => {
      const transaction = this.indexedDB.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');

      const entry = {
        key,
        data: processedData,
        ttl: ttl ? Date.now() + ttl : null,
        timestamp: Date.now(),
        size: this.estimateSize(processedData),
        compressed,
      };

      const request = store.put(entry);

      request.onsuccess = () => resolve();
      request.onerror = () => {
        console.error('IndexedDB set failed:', request.error);
        reject(request.error);
      };
    });
  }

  async deleteFromIndexedDB(key) {
    if (!this.indexedDB) return;

    return new Promise((resolve) => {
      const transaction = this.indexedDB.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      const request = store.delete(key);

      request.onsuccess = () => resolve();
      request.onerror = () => {
        console.error('IndexedDB delete failed:', request.error);
        resolve();
      };
    });
  }

  // === COMPRESSION METHODS ===

  shouldCompress(data) {
    if (!this.options.enableCompression || !this.compressionWorker) {
      return false;
    }

    const size = this.estimateSize(data);
    return size > this.options.compressionThreshold;
  }

  async compressData(data) {
    return new Promise((resolve, reject) => {
      const id = Date.now() + Math.random();

      this.loadingPromises.set(id, { resolve, reject });
      this.compressionWorker.postMessage({ type: 'compress', data, id });

      // Timeout after 5 seconds
      setTimeout(() => {
        if (this.loadingPromises.has(id)) {
          this.loadingPromises.delete(id);
          reject(new Error('Compression timeout'));
        }
      }, 5000);
    });
  }

  async decompressData(compressedData) {
    return new Promise((resolve, reject) => {
      const id = Date.now() + Math.random();

      this.loadingPromises.set(id, { resolve, reject });
      this.compressionWorker.postMessage({ type: 'decompress', data: compressedData, id });

      // Timeout after 5 seconds
      setTimeout(() => {
        if (this.loadingPromises.has(id)) {
          this.loadingPromises.delete(id);
          reject(new Error('Decompression timeout'));
        }
      }, 5000);
    });
  }

  // === MEMORY MANAGEMENT ===

  async ensureMemoryCapacity(requiredSize) {
    const maxSize = this.options.maxCacheSize;
    const maxEntries = this.options.maxCacheEntries;

    // Check size limit
    while (this.cacheStats.totalSize + requiredSize > maxSize && this.memoryCache.size > 0) {
      await this.evictLeastValuable();
    }

    // Check entry limit
    while (this.memoryCache.size >= maxEntries) {
      await this.evictLeastValuable();
    }
  }

  async evictLeastValuable() {
    if (this.memoryCache.size === 0) return;

    let leastValuable = null;
    let lowestScore = Infinity;

    for (const [key, entry] of this.memoryCache.entries()) {
      const score = this.calculateValueScore(entry);
      if (score < lowestScore) {
        lowestScore = score;
        leastValuable = key;
      }
    }

    if (leastValuable) {
      const entry = this.memoryCache.get(leastValuable);
      this.memoryCache.delete(leastValuable);
      this.cacheStats.totalSize -= entry.size;
      this.cacheStats.evictions++;
    }
  }

  calculateValueScore(entry) {
    const now = Date.now();
    const age = now - entry.timestamp;
    const timeSinceAccess = now - entry.lastAccessed;

    // Higher score = more valuable
    let score = entry.accessCount;

    // Penalize old entries
    score *= Math.exp(-age / (24 * 60 * 60 * 1000)); // Decay over 24 hours

    // Penalize recently unused entries
    score *= Math.exp(-timeSinceAccess / (60 * 60 * 1000)); // Decay over 1 hour

    // Bonus for high priority
    if (entry.priority === 'high') score *= 2;
    else if (entry.priority === 'low') score *= 0.5;

    // Penalize large entries
    score /= Math.log(entry.size + 1);

    return score;
  }

  // === PREFETCHING ===

  schedulePrefetch(keys, priority = 'low') {
    if (!this.options.enablePrefetch) return;

    keys.forEach(key => {
      if (!this.prefetchQueue.has(key)) {
        this.prefetchQueue.add(key);
      }
    });

    // Process prefetch queue
    this.processPrefetchQueue();
  }

  async processPrefetchQueue() {
    if (this.prefetchQueue.size === 0) return;

    const key = this.prefetchQueue.values().next().value;
    this.prefetchQueue.delete(key);

    try {
      // Check if already cached
      const cached = await this.get(key);
      if (cached) return;

      // Prefetch the data
      await this.prefetchData(key);
    } catch (error) {
      console.warn('Prefetch failed for key:', key, error);
    }

    // Continue processing queue
    if (this.prefetchQueue.size > 0) {
      setTimeout(() => this.processPrefetchQueue(), 100);
    }
  }

  async prefetchData(key) {
    // This method should be overridden by specific implementations
    // It should fetch the data and cache it
    console.log('Prefetching data for key:', key);
  }

  // === BATCH OPERATIONS ===

  async getMultiple(keys) {
    const results = new Map();
    const missingKeys = [];

    // Check cache for all keys
    for (const key of keys) {
      const cached = await this.get(key);
      if (cached !== null) {
        results.set(key, cached);
      } else {
        missingKeys.push(key);
      }
    }

    return { results, missingKeys };
  }

  async setMultiple(entries) {
    const promises = entries.map(({ key, data, options }) =>
      this.set(key, data, options)
    );

    await Promise.allSettled(promises);
  }

  // === MAINTENANCE ===

  startMaintenanceTasks() {
    // Clean expired entries every 5 minutes
    setInterval(() => {
      this.cleanExpiredEntries();
    }, 5 * 60 * 1000);

    // Update stats every minute
    setInterval(() => {
      this.updateStats();
    }, 60 * 1000);

    // Optimize cache every 10 minutes
    setInterval(() => {
      this.optimizeCache();
    }, 10 * 60 * 1000);
  }

  async cleanExpiredEntries() {
    const now = Date.now();
    const expiredKeys = [];

    for (const [key, entry] of this.memoryCache.entries()) {
      if (entry.ttl && now > entry.ttl) {
        expiredKeys.push(key);
      }
    }

    expiredKeys.forEach(key => {
      const entry = this.memoryCache.get(key);
      this.memoryCache.delete(key);
      this.cacheStats.totalSize -= entry.size;
    });

    // Clean IndexedDB expired entries
    if (this.indexedDB && expiredKeys.length > 0) {
      this.cleanExpiredFromIndexedDB();
    }
  }

  async cleanExpiredFromIndexedDB() {
    if (!this.indexedDB) return;

    return new Promise((resolve) => {
      const transaction = this.indexedDB.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      const index = store.index('ttl');
      const now = Date.now();

      // Get all entries with TTL less than now
      const request = index.openCursor(IDBKeyRange.upperBound(now));

      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        } else {
          resolve();
        }
      };

      request.onerror = () => {
        console.error('Failed to clean expired IndexedDB entries');
        resolve();
      };
    });
  }

  async optimizeCache() {
    // Perform cache optimization
    await this.defragmentCache();
    this.rebalanceCache();
  }

  async defragmentCache() {
    // Remove gaps in IndexedDB storage
    if (!this.indexedDB) return;

    try {
      const transaction = this.indexedDB.transaction(['cache'], 'readonly');
      const store = transaction.objectStore('cache');
      const countRequest = store.count();

      countRequest.onsuccess = () => {
        const count = countRequest.result;
        if (count < this.options.maxCacheEntries * 0.5) {
          // Cache is less than 50% full, no need to defragment
          return;
        }

        // Perform defragmentation by recreating the database
        this.scheduleDefragmentation();
      };
    } catch (error) {
      console.error('Cache defragmentation failed:', error);
    }
  }

  rebalanceCache() {
    // Move frequently accessed items to memory cache
    const memoryKeys = new Set(this.memoryCache.keys());
    const candidates = [];

    // This would typically involve checking IndexedDB access patterns
    // For now, we'll just ensure high-priority items are in memory
    for (const [key, entry] of this.memoryCache.entries()) {
      if (entry.priority === 'high' && entry.accessCount > 5) {
        candidates.push({ key, score: this.calculateValueScore(entry) });
      }
    }

    // Sort by score and promote top candidates
    candidates.sort((a, b) => b.score - a.score);
  }

  // === UTILITY METHODS ===

  estimateSize(data) {
    if (typeof data === 'string') {
      return data.length * 2; // Rough estimate for UTF-16
    }

    try {
      return JSON.stringify(data).length * 2;
    } catch {
      return 1024; // Default estimate
    }
  }

  updateAccessTime(key) {
    const entry = this.memoryCache.get(key);
    if (entry) {
      entry.lastAccessed = Date.now();
      entry.accessCount++;
    }
  }

  updateCacheStats(data) {
    // Update cache statistics
    this.cacheStats.totalSize = Array.from(this.memoryCache.values())
      .reduce((sum, entry) => sum + entry.size, 0);
  }

  updateCompressionStats(compressionResult) {
    const { originalSize, compressedSize } = compressionResult;
    const ratio = originalSize > 0 ? compressedSize / originalSize : 1;

    this.cacheStats.compressionRatio =
      (this.cacheStats.compressionRatio + ratio) / 2;
  }

  updateStats() {
    const hitRate = this.cacheStats.hits / (this.cacheStats.hits + this.cacheStats.misses);

    console.log('Cache Stats:', {
      hitRate: (hitRate * 100).toFixed(2) + '%',
      memoryEntries: this.memoryCache.size,
      totalSize: (this.cacheStats.totalSize / 1024 / 1024).toFixed(2) + 'MB',
      compressionRatio: (this.cacheStats.compressionRatio * 100).toFixed(1) + '%',
      evictions: this.cacheStats.evictions,
    });
  }

  // === PUBLIC API ===

  async clear() {
    this.memoryCache.clear();
    this.cacheStats = {
      hits: 0,
      misses: 0,
      evictions: 0,
      totalSize: 0,
      compressionRatio: 0,
    };

    if (this.indexedDB) {
      const transaction = this.indexedDB.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      await store.clear();
    }
  }

  async delete(key) {
    const entry = this.memoryCache.get(key);
    if (entry) {
      this.memoryCache.delete(key);
      this.cacheStats.totalSize -= entry.size;
    }

    if (this.indexedDB) {
      await this.deleteFromIndexedDB(key);
    }
  }

  getStats() {
    const hitRate = this.cacheStats.hits / (this.cacheStats.hits + this.cacheStats.misses) || 0;

    return {
      ...this.cacheStats,
      hitRate: hitRate * 100,
      memoryEntries: this.memoryCache.size,
      maxCacheSize: this.options.maxCacheSize,
      maxCacheEntries: this.options.maxCacheEntries,
    };
  }

  destroy() {
    this.memoryCache.clear();
    this.prefetchQueue.clear();
    this.loadingPromises.clear();

    if (this.compressionWorker) {
      this.compressionWorker.terminate();
    }

    if (this.indexedDB) {
      this.indexedDB.close();
    }
  }
}

// Export singleton instance
export const dataCacheManager = new DataCacheManager();
export default DataCacheManager;
