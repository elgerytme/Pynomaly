/**
 * Advanced Image Optimization Pipeline
 * Provides intelligent image loading, format conversion, and progressive enhancement
 */

export class ImageOptimizer {
  constructor() {
    this.supportedFormats = new Map();
    this.imageCache = new Map();
    this.loadingQueue = new Map();
    this.intersectionObserver = null;
    this.performanceObserver = null;

    this.config = {
      lazyLoadOffset: '50px 0px',
      qualitySettings: {
        high: { quality: 90, scale: 1.0 },
        medium: { quality: 75, scale: 0.8 },
        low: { quality: 60, scale: 0.6 }
      },
      fallbackFormats: ['webp', 'jpeg', 'png'],
      maxRetries: 3,
      timeoutMs: 10000
    };

    this.init();
  }

  async init() {
    await this.detectFormatSupport();
    this.setupLazyLoading();
    this.setupPerformanceMonitoring();
    this.preloadCriticalImages();
  }

  /**
   * Detect browser support for modern image formats
   */
  async detectFormatSupport() {
    const formats = {
      webp: 'data:image/webp;base64,UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==',
      avif: 'data:image/avif;base64,AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADybWV0YQAAAAAAAAAoaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAGxpYmF2aWYAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABGgAAAB0AAAAoaWluZgAAAAAAAQAAABppbmZlAgAAAAABAABhdjAxQ29sb3IAAAAAamlwcnAAAABLaXBjbwAAABRpc3BlAAAAAAAAAAIAAAACAAAAEHBpeGkAAAAAAwgICAAAAAxhdjFDgQ0MAAAAABNjb2xybmNseAACAAIAAYAAAAAXaXBtYQAAAAAAAAABAAEEAQKDBAAAACVtZGF0EgAKCBgABogQEAwgMg8f8D///8WfhwB8+ErK42A=',
      heic: 'data:image/heic;base64,AAAAGGZ0eXBoZWljAAAAAG1pZjFoZWljbWlhZg=='
    };

    for (const [format, dataUrl] of Object.entries(formats)) {
      try {
        const supported = await this.testImageFormat(dataUrl);
        this.supportedFormats.set(format, supported);
        console.log(`[ImageOptimizer] ${format.toUpperCase()} support: ${supported}`);
      } catch (error) {
        this.supportedFormats.set(format, false);
      }
    }
  }

  testImageFormat(dataUrl) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve(img.width === 1 && img.height === 1);
      img.onerror = () => resolve(false);
      img.src = dataUrl;
    });
  }

  /**
   * Setup intersection observer for lazy loading
   */
  setupLazyLoading() {
    if (!('IntersectionObserver' in window)) {
      // Fallback for browsers without intersection observer
      this.fallbackLazyLoading();
      return;
    }

    this.intersectionObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            this.loadImage(entry.target);
            this.intersectionObserver.unobserve(entry.target);
          }
        });
      },
      {
        rootMargin: this.config.lazyLoadOffset,
        threshold: 0.1
      }
    );

    // Observe all images with data-src
    this.observeImages();
  }

  observeImages() {
    const images = document.querySelectorAll('img[data-src], img[data-srcset]');
    images.forEach(img => {
      this.intersectionObserver.observe(img);

      // Add loading placeholder
      if (!img.src) {
        img.src = this.generatePlaceholder(
          img.dataset.width || 300,
          img.dataset.height || 200
        );
      }
    });
  }

  /**
   * Load and optimize image with intelligent format selection
   */
  async loadImage(img) {
    const originalSrc = img.dataset.src || img.src;
    const loadingId = `loading-${Date.now()}-${Math.random()}`;

    try {
      // Add loading state
      img.classList.add('loading');
      img.setAttribute('aria-busy', 'true');

      // Determine optimal image source
      const optimizedSrc = await this.getOptimizedImageSrc(originalSrc, img);

      // Check cache first
      if (this.imageCache.has(optimizedSrc)) {
        const cachedData = this.imageCache.get(optimizedSrc);
        this.applyImageData(img, cachedData);
        return;
      }

      // Check if already loading
      if (this.loadingQueue.has(optimizedSrc)) {
        await this.loadingQueue.get(optimizedSrc);
        return;
      }

      // Create loading promise
      const loadPromise = this.fetchAndProcessImage(optimizedSrc, img);
      this.loadingQueue.set(optimizedSrc, loadPromise);

      const imageData = await loadPromise;
      this.imageCache.set(optimizedSrc, imageData);
      this.applyImageData(img, imageData);

    } catch (error) {
      console.error('[ImageOptimizer] Failed to load image:', error);
      this.handleImageError(img, originalSrc);
    } finally {
      img.classList.remove('loading');
      img.removeAttribute('aria-busy');
      this.loadingQueue.delete(originalSrc);
    }
  }

  /**
   * Get optimized image source based on device capabilities and format support
   */
  async getOptimizedImageSrc(originalSrc, img) {
    const url = new URL(originalSrc, window.location.origin);

    // Determine optimal format
    const bestFormat = this.getBestSupportedFormat();
    if (bestFormat !== 'original') {
      url.searchParams.set('format', bestFormat);
    }

    // Determine optimal quality based on connection and device
    const quality = this.getOptimalQuality();
    url.searchParams.set('quality', quality.quality);

    // Handle responsive images
    const devicePixelRatio = window.devicePixelRatio || 1;
    const containerWidth = img.offsetWidth || parseInt(img.dataset.width) || 300;
    const optimalWidth = Math.ceil(containerWidth * devicePixelRatio * quality.scale);

    url.searchParams.set('width', optimalWidth);

    // Progressive loading for large images
    if (optimalWidth > 800) {
      url.searchParams.set('progressive', 'true');
    }

    return url.toString();
  }

  getBestSupportedFormat() {
    for (const format of this.config.fallbackFormats) {
      if (this.supportedFormats.get(format)) {
        return format;
      }
    }
    return 'original';
  }

  getOptimalQuality() {
    // Adapt quality based on connection speed
    const connection = navigator.connection;

    if (connection) {
      if (connection.effectiveType === '4g' && !connection.saveData) {
        return this.config.qualitySettings.high;
      } else if (connection.effectiveType === '3g') {
        return this.config.qualitySettings.medium;
      } else {
        return this.config.qualitySettings.low;
      }
    }

    // Default to medium quality
    return this.config.qualitySettings.medium;
  }

  /**
   * Fetch and process image with retries and error handling
   */
  async fetchAndProcessImage(src, img) {
    let retries = 0;

    while (retries < this.config.maxRetries) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.timeoutMs);

        const response = await fetch(src, {
          signal: controller.signal,
          headers: {
            'Accept': 'image/webp,image/avif,image/*,*/*;q=0.8'
          }
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);

        return {
          src: objectUrl,
          blob: blob,
          size: blob.size,
          type: blob.type,
          timestamp: Date.now()
        };

      } catch (error) {
        retries++;
        if (retries >= this.config.maxRetries) {
          throw error;
        }

        // Exponential backoff
        await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, retries)));
      }
    }
  }

  applyImageData(img, imageData) {
    // Create new image to avoid flicker
    const newImg = new Image();

    newImg.onload = () => {
      img.src = imageData.src;
      img.classList.add('loaded');

      // Handle srcset if provided
      if (img.dataset.srcset) {
        img.srcset = img.dataset.srcset;
      }

      // Trigger load event
      img.dispatchEvent(new Event('optimized-load'));
    };

    newImg.src = imageData.src;
  }

  handleImageError(img, originalSrc) {
    // Try fallback sources
    const fallbackSrc = img.dataset.fallback || this.generateErrorPlaceholder();

    if (fallbackSrc && fallbackSrc !== originalSrc) {
      img.src = fallbackSrc;
    } else {
      img.classList.add('error');
      img.alt = 'Failed to load image';
    }

    img.dispatchEvent(new Event('optimized-error'));
  }

  /**
   * Generate placeholder images
   */
  generatePlaceholder(width = 300, height = 200) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');

    // Create gradient background
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#f3f4f6');
    gradient.addColorStop(1, '#e5e7eb');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Add loading text
    ctx.fillStyle = '#9ca3af';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Loading...', width / 2, height / 2);

    return canvas.toDataURL('image/png');
  }

  generateErrorPlaceholder() {
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 200;

    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#fef2f2';
    ctx.fillRect(0, 0, 300, 200);

    ctx.fillStyle = '#dc2626';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Image not available', 150, 100);

    return canvas.toDataURL('image/png');
  }

  /**
   * Setup performance monitoring for images
   */
  setupPerformanceMonitoring() {
    if ('PerformanceObserver' in window) {
      this.performanceObserver = new PerformanceObserver((list) => {
        list.getEntries().forEach(entry => {
          if (entry.name.includes('image') || entry.initiatorType === 'img') {
            this.recordImageMetric(entry);
          }
        });
      });

      this.performanceObserver.observe({
        entryTypes: ['resource', 'navigation']
      });
    }
  }

  recordImageMetric(entry) {
    const metrics = {
      url: entry.name,
      loadTime: entry.duration,
      size: entry.transferSize,
      timestamp: Date.now()
    };

    console.log('[ImageOptimizer] Image loaded:', metrics);

    // Warn about slow loading images
    if (entry.duration > 2000) {
      console.warn('[ImageOptimizer] Slow image load detected:', entry.name, `${entry.duration}ms`);
    }
  }

  /**
   * Preload critical images for better perceived performance
   */
  async preloadCriticalImages() {
    const criticalImages = document.querySelectorAll('img[data-critical="true"]');

    const preloadPromises = Array.from(criticalImages).map(async (img) => {
      try {
        await this.loadImage(img);
      } catch (error) {
        console.warn('[ImageOptimizer] Failed to preload critical image:', error);
      }
    });

    await Promise.allSettled(preloadPromises);
    console.log('[ImageOptimizer] Critical images preloaded');
  }

  /**
   * Fallback lazy loading for browsers without intersection observer
   */
  fallbackLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');

    const checkImages = () => {
      images.forEach(img => {
        if (this.isInViewport(img)) {
          this.loadImage(img);
        }
      });
    };

    // Check on scroll and resize
    window.addEventListener('scroll', this.throttle(checkImages, 100));
    window.addEventListener('resize', this.throttle(checkImages, 100));

    // Initial check
    checkImages();
  }

  isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
      rect.bottom >= 0 &&
      rect.right >= 0 &&
      rect.top <= (window.innerHeight || document.documentElement.clientHeight) &&
      rect.left <= (window.innerWidth || document.documentElement.clientWidth)
    );
  }

  throttle(func, delay) {
    let timeoutId;
    let lastExecTime = 0;

    return (...args) => {
      const currentTime = Date.now();

      if (currentTime - lastExecTime > delay) {
        func.apply(this, args);
        lastExecTime = currentTime;
      } else {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          func.apply(this, args);
          lastExecTime = Date.now();
        }, delay);
      }
    };
  }

  /**
   * Get optimization statistics
   */
  getStats() {
    return {
      supportedFormats: Object.fromEntries(this.supportedFormats),
      cachedImages: this.imageCache.size,
      totalCacheSize: Array.from(this.imageCache.values())
        .reduce((total, data) => total + (data.size || 0), 0),
      loadingQueue: this.loadingQueue.size
    };
  }

  /**
   * Clear image cache to free memory
   */
  clearCache() {
    // Revoke object URLs to prevent memory leaks
    this.imageCache.forEach(data => {
      if (data.src.startsWith('blob:')) {
        URL.revokeObjectURL(data.src);
      }
    });

    this.imageCache.clear();
    console.log('[ImageOptimizer] Cache cleared');
  }

  /**
   * Cleanup resources
   */
  destroy() {
    if (this.intersectionObserver) {
      this.intersectionObserver.disconnect();
    }

    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }

    this.clearCache();
  }
}

// Export singleton instance
export const imageOptimizer = new ImageOptimizer();

// Auto-initialize if in browser environment
if (typeof window !== 'undefined') {
  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      imageOptimizer.observeImages();
    });
  } else {
    imageOptimizer.observeImages();
  }

  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    imageOptimizer.destroy();
  });
}
