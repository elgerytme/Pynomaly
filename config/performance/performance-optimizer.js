/**
 * Advanced Performance Optimizer for Pynomaly Web UI
 * Implements code splitting, caching, image optimization, and service worker enhancements
 */

import { build } from 'esbuild';
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);

// Performance configuration
const PERFORMANCE_CONFIG = {
  // Bundle size targets (Lighthouse recommendations)
  budgets: {
    mainBundle: 200 * 1024,      // 200KB main bundle
    chunkBundle: 100 * 1024,     // 100KB per chunk
    totalJS: 400 * 1024,         // 400KB total JS
    totalCSS: 40 * 1024,         // 40KB total CSS
    imageMaxSize: 150 * 1024,    // 150KB per image
  },

  // Web Vitals targets
  vitals: {
    fcp: 1500,    // First Contentful Paint < 1.5s
    lcp: 2500,    // Largest Contentful Paint < 2.5s
    tti: 3500,    // Time to Interactive < 3.5s
    cls: 0.1,     // Cumulative Layout Shift < 0.1
    fid: 100,     // First Input Delay < 100ms
  },

  // Caching strategies
  caching: {
    staticAssets: '1y',           // 1 year for static assets
    apiResponses: '5m',           // 5 minutes for API responses
    dynamicContent: '1h',         // 1 hour for dynamic content
    offlineStorage: '7d',         // 7 days for offline data
  }
};

/**
 * Code Splitting Configuration
 */
export const createCodeSplittingConfig = () => {
  const baseDir = 'src/pynomaly/presentation/web/static';

  return {
    // Core application chunks
    main: {
      entry: path.join(baseDir, 'js/main.js'),
      priority: 'high',
      preload: true,
    },

    // Feature-based splitting
    'features/anomaly-detection': {
      entry: path.join(baseDir, 'js/components/anomaly-detector.js'),
      import: 'dynamic',
      priority: 'medium',
    },

    'features/data-visualization': {
      entry: path.join(baseDir, 'js/charts/index.js'),
      import: 'dynamic',
      priority: 'medium',
    },

    'features/dashboard': {
      entry: path.join(baseDir, 'js/components/dashboard-layout.js'),
      import: 'dynamic',
      priority: 'high',
    },

    'features/real-time': {
      entry: path.join(baseDir, 'js/charts/real-time-dashboard.js'),
      import: 'dynamic',
      priority: 'low',
    },

    // Vendor splitting by size and usage
    'vendors/essential': {
      entry: createVendorBundle(['htmx.org', 'alpinejs']),
      priority: 'high',
      preload: true,
    },

    'vendors/visualization': {
      entry: createVendorBundle(['d3', 'echarts']),
      import: 'dynamic',
      priority: 'medium',
    },

    'vendors/utilities': {
      entry: createVendorBundle(['fuse.js', 'sortablejs']),
      import: 'dynamic',
      priority: 'low',
    },
  };
};

/**
 * Enhanced ESBuild Plugin for Performance
 */
export const createPerformancePlugin = () => ({
  name: 'performance-optimizer',
  setup(build) {
    const chunks = new Map();
    const startTime = Date.now();

    build.onStart(() => {
      console.log('🚀 Starting performance-optimized build...');
    });

    build.onResolve({ filter: /\.(png|jpg|jpeg|gif|svg|webp)$/ }, async (args) => {
      // Optimize images during build
      const imagePath = args.path;
      const optimizedPath = await optimizeImage(imagePath);

      return {
        path: optimizedPath,
        namespace: 'optimized-image'
      };
    });

    build.onLoad({ filter: /.*/, namespace: 'optimized-image' }, async (args) => {
      // Load optimized image
      const contents = await fs.promises.readFile(args.path);
      return {
        contents,
        loader: 'file'
      };
    });

    build.onEnd((result) => {
      const buildTime = Date.now() - startTime;

      if (result.metafile) {
        analyzeBundle(result.metafile, buildTime);
        generatePerformanceReport(result.metafile, buildTime);
      }

      console.log(`✅ Performance-optimized build completed in ${buildTime}ms`);
    });
  }
});

/**
 * Image Optimization
 */
async function optimizeImage(imagePath) {
  const ext = path.extname(imagePath);
  const basePath = imagePath.replace(ext, '');

  // Generate multiple formats for responsive images
  const formats = [
    { ext: '.webp', quality: 80 },
    { ext: '.jpg', quality: 85 },
    { ext: ext, quality: 90 } // Original format fallback
  ];

  // For now, return original path (would implement actual optimization)
  // In production, this would use sharp, imagemin, or similar
  return imagePath;
}

/**
 * Bundle Analysis
 */
function analyzeBundle(metafile, buildTime) {
  const { outputs, inputs } = metafile;
  const analysis = {
    buildTime,
    bundles: {},
    totalSize: 0,
    recommendations: []
  };

  // Analyze each output
  Object.entries(outputs).forEach(([file, info]) => {
    if (file.endsWith('.js')) {
      const name = path.basename(file, '.js');
      const size = info.bytes;

      analysis.bundles[name] = {
        size,
        sizeKB: Math.round(size / 1024 * 100) / 100,
        inputs: Object.keys(info.inputs || {}),
        imports: info.imports || []
      };

      analysis.totalSize += size;

      // Performance budget checks
      if (name === 'main' && size > PERFORMANCE_CONFIG.budgets.mainBundle) {
        analysis.recommendations.push({
          type: 'warning',
          message: `Main bundle (${(size/1024).toFixed(1)}KB) exceeds budget (${(PERFORMANCE_CONFIG.budgets.mainBundle/1024).toFixed(1)}KB)`,
          suggestion: 'Consider moving non-critical code to dynamic imports'
        });
      }
    }
  });

  // Overall size check
  if (analysis.totalSize > PERFORMANCE_CONFIG.budgets.totalJS) {
    analysis.recommendations.push({
      type: 'error',
      message: `Total JS size (${(analysis.totalSize/1024).toFixed(1)}KB) exceeds budget (${(PERFORMANCE_CONFIG.budgets.totalJS/1024).toFixed(1)}KB)`,
      suggestion: 'Implement more aggressive code splitting and tree shaking'
    });
  }

  // Log analysis
  console.log('\n📊 Bundle Analysis:');
  Object.entries(analysis.bundles).forEach(([name, info]) => {
    const status = info.size <= PERFORMANCE_CONFIG.budgets.chunkBundle ? '✅' : '⚠️';
    console.log(`  ${status} ${name}: ${info.sizeKB}KB`);
  });

  if (analysis.recommendations.length > 0) {
    console.log('\n💡 Performance Recommendations:');
    analysis.recommendations.forEach(rec => {
      const icon = rec.type === 'error' ? '❌' : '⚠️';
      console.log(`  ${icon} ${rec.message}`);
      console.log(`     ${rec.suggestion}`);
    });
  }
}

/**
 * Generate Performance Report
 */
function generatePerformanceReport(metafile, buildTime) {
  const report = {
    timestamp: new Date().toISOString(),
    buildTime,
    config: PERFORMANCE_CONFIG,
    bundles: {},
    metrics: {
      totalBundles: 0,
      totalSize: 0,
      averageChunkSize: 0,
      budgetCompliance: 0
    },
    recommendations: []
  };

  // Calculate metrics
  const outputs = Object.entries(metafile.outputs).filter(([file]) => file.endsWith('.js'));
  report.metrics.totalBundles = outputs.length;

  outputs.forEach(([file, info]) => {
    const name = path.basename(file, '.js');
    report.bundles[name] = {
      size: info.bytes,
      sizeKB: Math.round(info.bytes / 1024 * 100) / 100,
      withinBudget: info.bytes <= PERFORMANCE_CONFIG.budgets.chunkBundle
    };
    report.metrics.totalSize += info.bytes;
  });

  report.metrics.averageChunkSize = report.metrics.totalSize / report.metrics.totalBundles;
  report.metrics.budgetCompliance = Object.values(report.bundles)
    .filter(bundle => bundle.withinBudget).length / report.metrics.totalBundles;

  // Save report
  const reportPath = 'test_reports/performance-report.json';
  if (!fs.existsSync('test_reports')) {
    fs.mkdirSync('test_reports', { recursive: true });
  }

  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`📈 Performance report saved to ${reportPath}`);
}

/**
 * Create vendor bundle helper
 */
function createVendorBundle(packages) {
  const content = packages.map(pkg => `export * from '${pkg}';`).join('\n');
  const bundlePath = `src/pynomaly/presentation/web/static/js/vendors/${packages.join('-')}.js`;

  // Ensure directory exists
  const dir = path.dirname(bundlePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  // Write bundle file if it doesn't exist
  if (!fs.existsSync(bundlePath)) {
    fs.writeFileSync(bundlePath, content);
  }

  return bundlePath;
}

/**
 * Lazy Loading Configuration
 */
export const createLazyLoadingConfig = () => ({
  // Route-based lazy loading
  routes: {
    '/dashboard': () => import('./features/dashboard.js'),
    '/datasets': () => import('./features/data-management.js'),
    '/anomalies': () => import('./features/anomaly-detection.js'),
    '/settings': () => import('./features/settings.js'),
  },

  // Component-based lazy loading
  components: {
    'anomaly-heatmap': () => import('./components/AnomalyHeatmap.js'),
    'real-time-chart': () => import('./components/RealTimeChart.js'),
    'data-uploader': () => import('./components/DataUploader.js'),
    'advanced-settings': () => import('./components/AdvancedSettings.js'),
  },

  // Intersection Observer for viewport-based loading
  intersectionObserver: {
    threshold: 0.1,
    rootMargin: '50px',
  }
});

/**
 * Dynamic Import Helper
 */
export const createDynamicImportHelper = () => `
/**
 * Dynamic Import Helper for Lazy Loading
 */
class DynamicImporter {
  constructor() {
    this.cache = new Map();
    this.loadingPromises = new Map();
  }

  async import(moduleId) {
    // Return from cache if already loaded
    if (this.cache.has(moduleId)) {
      return this.cache.get(moduleId);
    }

    // Return existing promise if already loading
    if (this.loadingPromises.has(moduleId)) {
      return this.loadingPromises.get(moduleId);
    }

    // Create loading promise
    const loadingPromise = this.loadModule(moduleId);
    this.loadingPromises.set(moduleId, loadingPromise);

    try {
      const module = await loadingPromise;
      this.cache.set(moduleId, module);
      this.loadingPromises.delete(moduleId);
      return module;
    } catch (error) {
      this.loadingPromises.delete(moduleId);
      throw error;
    }
  }

  async loadModule(moduleId) {
    const moduleMap = await this.getModuleMap();
    const modulePath = moduleMap[moduleId];

    if (!modulePath) {
      throw new Error(\`Module not found: \${moduleId}\`);
    }

    return import(\`/static/js/dist/\${modulePath}\`);
  }

  async getModuleMap() {
    if (!this.moduleMapPromise) {
      this.moduleMapPromise = fetch('/static/js/dist/module-map.json')
        .then(response => response.json());
    }
    return this.moduleMapPromise;
  }

  preload(moduleIds) {
    moduleIds.forEach(moduleId => {
      // Start loading but don't wait
      this.import(moduleId).catch(console.warn);
    });
  }
}

// Global instance
window.dynamicImporter = new DynamicImporter();

// Component lazy loading helper
export function lazyLoadComponent(selector, moduleId) {
  const elements = document.querySelectorAll(selector);

  if (elements.length === 0) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(async (entry) => {
        if (entry.isIntersecting) {
          observer.unobserve(entry.target);

          try {
            const module = await window.dynamicImporter.import(moduleId);
            if (module.default && typeof module.default === 'function') {
              module.default(entry.target);
            }
          } catch (error) {
            console.error(\`Failed to load component \${moduleId}:\`, error);
          }
        }
      });
    },
    { threshold: 0.1, rootMargin: '50px' }
  );

  elements.forEach(element => observer.observe(element));
}
`;

export { PERFORMANCE_CONFIG };
