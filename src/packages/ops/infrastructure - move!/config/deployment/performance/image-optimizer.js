/**
 * Image Optimization System for Pynomaly Web UI
 * Implements responsive images, WebP conversion, and lazy loading
 */

import fs from 'fs';
import path from 'path';

/**
 * Image Optimization Configuration
 */
export const IMAGE_CONFIG = {
  // Supported formats in order of preference
  formats: ['webp', 'avif', 'jpg', 'png'],

  // Quality settings
  quality: {
    webp: 80,
    avif: 70,
    jpg: 85,
    png: 90
  },

  // Responsive breakpoints
  breakpoints: {
    sm: 640,
    md: 768,
    lg: 1024,
    xl: 1280,
    '2xl': 1536
  },

  // Size limits
  maxSize: 150 * 1024, // 150KB max per image
  maxDimensions: {
    width: 2048,
    height: 2048
  },

  // Lazy loading configuration
  lazyLoading: {
    threshold: '10px',
    fadeInDuration: 300
  }
};

/**
 * Generate Responsive Image HTML
 */
export function generateResponsiveImage(src, alt, options = {}) {
  const {
    sizes = '(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw',
    loading = 'lazy',
    className = '',
    priority = false
  } = options;

  const baseName = path.parse(src).name;
  const baseDir = path.dirname(src);

  // Generate source sets for different formats
  const webpSrcSet = generateSrcSet(baseName, baseDir, 'webp');
  const jpgSrcSet = generateSrcSet(baseName, baseDir, 'jpg');

  const loadingAttr = priority ? 'eager' : loading;
  const fetchPriority = priority ? 'high' : 'auto';

  return `
    <picture class="responsive-image ${className}">
      <source
        srcset="${webpSrcSet}"
        sizes="${sizes}"
        type="image/webp"
      />
      <source
        srcset="${jpgSrcSet}"
        sizes="${sizes}"
        type="image/jpeg"
      />
      <img
        src="${src}"
        alt="${alt}"
        sizes="${sizes}"
        loading="${loadingAttr}"
        fetchpriority="${fetchPriority}"
        decoding="async"
        class="responsive-image__img"
      />
    </picture>
  `;
}

/**
 * Generate srcset for responsive images
 */
function generateSrcSet(baseName, baseDir, format) {
  const sizes = [320, 640, 768, 1024, 1280, 1536];

  return sizes
    .map(size => `${baseDir}/${baseName}-${size}w.${format} ${size}w`)
    .join(', ');
}

/**
 * Lazy Loading Implementation
 */
export const createLazyLoadingScript = () => `
/**
 * Intersection Observer based Lazy Loading
 */
class LazyImageLoader {
  constructor() {
    this.imageObserver = null;
    this.config = {
      rootMargin: '${IMAGE_CONFIG.lazyLoading.threshold}',
      threshold: 0.01
    };

    this.init();
  }

  init() {
    // Check for Intersection Observer support
    if ('IntersectionObserver' in window) {
      this.imageObserver = new IntersectionObserver(
        this.onIntersection.bind(this),
        this.config
      );

      this.observeImages();
    } else {
      // Fallback for older browsers
      this.loadAllImages();
    }
  }

  observeImages() {
    const lazyImages = document.querySelectorAll('img[loading="lazy"], picture img[loading="lazy"]');

    lazyImages.forEach(img => {
      this.imageObserver.observe(img);
    });
  }

  onIntersection(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        this.loadImage(img);
        this.imageObserver.unobserve(img);
      }
    });
  }

  loadImage(img) {
    // Handle picture element
    const picture = img.closest('picture');
    if (picture) {
      const sources = picture.querySelectorAll('source');
      sources.forEach(source => {
        if (source.dataset.srcset) {
          source.srcset = source.dataset.srcset;
          delete source.dataset.srcset;
        }
      });
    }

    // Load main image
    if (img.dataset.src) {
      img.src = img.dataset.src;
      delete img.dataset.src;
    }

    if (img.dataset.srcset) {
      img.srcset = img.dataset.srcset;
      delete img.dataset.srcset;
    }

    // Add fade-in effect
    img.style.opacity = '0';
    img.style.transition = \`opacity \${${IMAGE_CONFIG.lazyLoading.fadeInDuration}}ms ease\`;

    img.onload = () => {
      img.style.opacity = '1';
      img.classList.add('loaded');
    };

    img.onerror = () => {
      img.classList.add('error');
      this.handleImageError(img);
    };
  }

  handleImageError(img) {
    // Replace with fallback image
    const fallbackSrc = img.dataset.fallback || '/static/img/placeholder.svg';
    if (img.src !== fallbackSrc) {
      img.src = fallbackSrc;
    }
  }

  loadAllImages() {
    const lazyImages = document.querySelectorAll('img[loading="lazy"]');
    lazyImages.forEach(img => this.loadImage(img));
  }

  // Add new images to observer
  observe(img) {
    if (this.imageObserver) {
      this.imageObserver.observe(img);
    } else {
      this.loadImage(img);
    }
  }

  // Manually trigger loading for high priority images
  loadHighPriority() {
    const priorityImages = document.querySelectorAll('img[fetchpriority="high"]');
    priorityImages.forEach(img => this.loadImage(img));
  }
}

// Initialize lazy loading
document.addEventListener('DOMContentLoaded', () => {
  window.lazyLoader = new LazyImageLoader();
});

// Preload critical images
document.addEventListener('DOMContentLoaded', () => {
  const criticalImages = document.querySelectorAll('img[fetchpriority="high"]');
  criticalImages.forEach(img => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'image';
    link.href = img.src || img.dataset.src;
    document.head.appendChild(link);
  });
});
`;

/**
 * Image Format Detection and Optimization
 */
export const createImageFormatDetection = () => `
/**
 * Image Format Support Detection
 */
class ImageFormatDetector {
  constructor() {
    this.supportedFormats = new Set();
    this.detect();
  }

  async detect() {
    // Test WebP support
    if (await this.testFormat('webp')) {
      this.supportedFormats.add('webp');
    }

    // Test AVIF support
    if (await this.testFormat('avif')) {
      this.supportedFormats.add('avif');
    }

    // Update HTML based on support
    this.updateImageSources();
  }

  testFormat(format) {
    return new Promise(resolve => {
      const img = new Image();

      img.onload = () => resolve(true);
      img.onerror = () => resolve(false);

      // Test images for format detection
      const testImages = {
        webp: 'data:image/webp;base64,UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEADsD+JaQAA3AAAAAA',
        avif: 'data:image/avif;base64,AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADybWV0YQ=='
      };

      img.src = testImages[format];
    });
  }

  updateImageSources() {
    const pictures = document.querySelectorAll('picture');

    pictures.forEach(picture => {
      const sources = picture.querySelectorAll('source');

      sources.forEach(source => {
        const type = source.getAttribute('type');

        if (type === 'image/avif' && !this.supportedFormats.has('avif')) {
          source.remove();
        } else if (type === 'image/webp' && !this.supportedFormats.has('webp')) {
          source.remove();
        }
      });
    });
  }

  supports(format) {
    return this.supportedFormats.has(format);
  }

  getBestFormat() {
    if (this.supportedFormats.has('avif')) return 'avif';
    if (this.supportedFormats.has('webp')) return 'webp';
    return 'jpg';
  }
}

// Initialize format detector
window.imageFormatDetector = new ImageFormatDetector();
`;

/**
 * Placeholder Image Generator
 */
export function generatePlaceholder(width, height, text = '') {
  const svg = `
    <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="#f3f4f6"/>
      <text
        x="50%"
        y="50%"
        dominant-baseline="middle"
        text-anchor="middle"
        fill="#6b7280"
        font-family="system-ui, sans-serif"
        font-size="14"
      >${text || `${width}×${height}`}</text>
    </svg>
  `;

  return `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
}

/**
 * Image Optimization Build Plugin
 */
export const createImageOptimizationPlugin = () => ({
  name: 'image-optimization',
  setup(build) {
    // Process images during build
    build.onLoad({ filter: /\.(png|jpg|jpeg|gif|svg)$/ }, async (args) => {
      const imagePath = args.path;
      const imageDir = path.dirname(imagePath);
      const imageExt = path.extname(imagePath);
      const imageName = path.basename(imagePath, imageExt);

      // In production, this would:
      // 1. Generate multiple sizes
      // 2. Convert to WebP/AVIF
      // 3. Optimize file sizes
      // 4. Generate responsive markup

      console.log(`[Image Optimizer] Processing: ${imagePath}`);

      // For now, just pass through
      const contents = await fs.promises.readFile(imagePath);

      return {
        contents,
        loader: 'file'
      };
    });
  }
});

/**
 * Critical Images Preloader
 */
export const createCriticalImagePreloader = () => `
/**
 * Preload Critical Images for Performance
 */
function preloadCriticalImages() {
  const criticalImages = [
    '/static/img/logo.svg',
    '/static/img/icon-192.png',
    '/static/img/favicon.png'
  ];

  criticalImages.forEach(src => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'image';
    link.href = src;
    link.crossOrigin = 'anonymous';
    document.head.appendChild(link);
  });
}

// Preload immediately
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', preloadCriticalImages);
} else {
  preloadCriticalImages();
}
`;

/**
 * Image Loading Performance Monitor
 */
export const createImagePerformanceMonitor = () => `
/**
 * Monitor Image Loading Performance
 */
class ImagePerformanceMonitor {
  constructor() {
    this.metrics = {
      totalImages: 0,
      loadedImages: 0,
      failedImages: 0,
      averageLoadTime: 0,
      totalLoadTime: 0
    };

    this.init();
  }

  init() {
    // Monitor existing images
    this.monitorExistingImages();

    // Monitor dynamically added images
    this.observeNewImages();
  }

  monitorExistingImages() {
    const images = document.querySelectorAll('img');
    images.forEach(img => this.monitorImage(img));
  }

  observeNewImages() {
    const observer = new MutationObserver(mutations => {
      mutations.forEach(mutation => {
        mutation.addedNodes.forEach(node => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            const images = node.querySelectorAll ?
              node.querySelectorAll('img') :
              (node.tagName === 'IMG' ? [node] : []);

            images.forEach(img => this.monitorImage(img));
          }
        });
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  monitorImage(img) {
    this.metrics.totalImages++;
    const startTime = performance.now();

    const onLoad = () => {
      const loadTime = performance.now() - startTime;
      this.metrics.loadedImages++;
      this.metrics.totalLoadTime += loadTime;
      this.metrics.averageLoadTime = this.metrics.totalLoadTime / this.metrics.loadedImages;

      cleanup();
    };

    const onError = () => {
      this.metrics.failedImages++;
      cleanup();
    };

    const cleanup = () => {
      img.removeEventListener('load', onLoad);
      img.removeEventListener('error', onError);
    };

    if (img.complete) {
      if (img.naturalWidth > 0) {
        onLoad();
      } else {
        onError();
      }
    } else {
      img.addEventListener('load', onLoad);
      img.addEventListener('error', onError);
    }
  }

  getMetrics() {
    return {
      ...this.metrics,
      successRate: this.metrics.totalImages > 0 ?
        (this.metrics.loadedImages / this.metrics.totalImages) * 100 : 0
    };
  }

  report() {
    const metrics = this.getMetrics();
    console.log('📸 Image Performance Metrics:', metrics);

    // Send to analytics if available
    if (window.gtag) {
      window.gtag('event', 'image_performance', {
        custom_map: {
          metric1: 'average_load_time',
          metric2: 'success_rate'
        },
        metric1: Math.round(metrics.averageLoadTime),
        metric2: Math.round(metrics.successRate)
      });
    }
  }
}

// Initialize monitoring
window.imagePerformanceMonitor = new ImagePerformanceMonitor();

// Report metrics after page load
window.addEventListener('load', () => {
  setTimeout(() => {
    window.imagePerformanceMonitor.report();
  }, 2000);
});
`;

/**
 * CSS for Responsive Images
 */
export const getResponsiveImageCSS = () => `
/* Responsive Image Styles */
.responsive-image {
  display: block;
  width: 100%;
  height: auto;
}

.responsive-image__img {
  width: 100%;
  height: auto;
  max-width: 100%;
  vertical-align: middle;
  transition: opacity 300ms ease;
}

.responsive-image__img[loading="lazy"] {
  opacity: 0;
}

.responsive-image__img.loaded {
  opacity: 1;
}

.responsive-image__img.error {
  opacity: 0.5;
  filter: grayscale(100%);
}

/* Aspect ratio containers */
.aspect-ratio-16-9 {
  aspect-ratio: 16 / 9;
  overflow: hidden;
}

.aspect-ratio-4-3 {
  aspect-ratio: 4 / 3;
  overflow: hidden;
}

.aspect-ratio-1-1 {
  aspect-ratio: 1 / 1;
  overflow: hidden;
}

.aspect-ratio-16-9 img,
.aspect-ratio-4-3 img,
.aspect-ratio-1-1 img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

/* Loading placeholder */
.image-placeholder {
  background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%);
  background-size: 200% 100%;
  animation: loading 2s infinite;
}

@keyframes loading {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Print optimizations */
@media print {
  .responsive-image__img {
    max-width: 100% !important;
    height: auto !important;
  }
}
`;

export { IMAGE_CONFIG };
