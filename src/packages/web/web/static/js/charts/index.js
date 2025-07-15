/**
 * Charts Module Entry Point - Lazy Loading Implementation
 * Provides dynamic imports for heavy visualization components
 */

/**
 * Dynamically import and initialize chart components based on requirements
 */
export class ChartLoader {
  constructor() {
    this.loadedCharts = new Map();
    this.loadingPromises = new Map();
  }

  /**
   * Load D3-based timeline chart with lazy loading
   */
  async loadTimelineChart(container, config = {}) {
    const chartType = 'timeline';

    if (this.loadedCharts.has(chartType)) {
      return this.loadedCharts.get(chartType);
    }

    if (this.loadingPromises.has(chartType)) {
      return this.loadingPromises.get(chartType);
    }

    const loadPromise = this._loadTimelineChartModule().then(module => {
      const chart = module.createAnomalyTimeline(container, config);
      this.loadedCharts.set(chartType, chart);
      return chart;
    });

    this.loadingPromises.set(chartType, loadPromise);
    return loadPromise;
  }

  /**
   * Load D3-based heatmap chart with lazy loading
   */
  async loadHeatmapChart(container, config = {}) {
    const chartType = 'heatmap';

    if (this.loadedCharts.has(chartType)) {
      return this.loadedCharts.get(chartType);
    }

    if (this.loadingPromises.has(chartType)) {
      return this.loadingPromises.get(chartType);
    }

    const loadPromise = this._loadHeatmapChartModule().then(module => {
      const chart = module.createAnomalyHeatmap(container, config);
      this.loadedCharts.set(chartType, chart);
      return chart;
    });

    this.loadingPromises.set(chartType, loadPromise);
    return loadPromise;
  }

  /**
   * Load ECharts-based real-time dashboard with lazy loading
   */
  async loadRealtimeDashboard(container, config = {}) {
    const chartType = 'realtime';

    if (this.loadedCharts.has(chartType)) {
      return this.loadedCharts.get(chartType);
    }

    if (this.loadingPromises.has(chartType)) {
      return this.loadingPromises.get(chartType);
    }

    const loadPromise = this._loadRealtimeDashboardModule().then(module => {
      const chart = module.createRealTimeDashboard(container, config);
      this.loadedCharts.set(chartType, chart);
      return chart;
    });

    this.loadingPromises.set(chartType, loadPromise);
    return loadPromise;
  }

  /**
   * Load Plotly-based 3D visualizations with lazy loading
   */
  async load3DVisualization(container, config = {}) {
    const chartType = '3d-plot';

    if (this.loadedCharts.has(chartType)) {
      return this.loadedCharts.get(chartType);
    }

    if (this.loadingPromises.has(chartType)) {
      return this.loadingPromises.get(chartType);
    }

    const loadPromise = this._load3DVisualizationModule().then(module => {
      const chart = module.create3DScatterPlot(container, config);
      this.loadedCharts.set(chartType, chart);
      return chart;
    });

    this.loadingPromises.set(chartType, loadPromise);
    return loadPromise;
  }

  /**
   * Load advanced statistical charts with lazy loading
   */
  async loadAdvancedCharts(container, chartType, config = {}) {
    const cacheKey = `advanced-${chartType}`;

    if (this.loadedCharts.has(cacheKey)) {
      return this.loadedCharts.get(cacheKey);
    }

    if (this.loadingPromises.has(cacheKey)) {
      return this.loadingPromises.get(cacheKey);
    }

    const loadPromise = this._loadAdvancedChartsModule().then(module => {
      let chart;
      switch (chartType) {
        case 'violin':
          chart = module.createViolinPlot(container, config);
          break;
        case 'correlation':
          chart = module.createCorrelationMatrix(container, config);
          break;
        case 'distribution':
          chart = module.createDistributionPlot(container, config);
          break;
        case 'time-series-decomposition':
          chart = module.createTimeSeriesDecomposition(container, config);
          break;
        default:
          throw new Error(`Unknown advanced chart type: ${chartType}`);
      }
      this.loadedCharts.set(cacheKey, chart);
      return chart;
    });

    this.loadingPromises.set(cacheKey, loadPromise);
    return loadPromise;
  }

  /**
   * Preload critical charts for better performance
   */
  async preloadCriticalCharts() {
    const criticalCharts = [
      this._loadTimelineChartModule(),
      this._loadHeatmapChartModule()
    ];

    try {
      await Promise.all(criticalCharts);
      console.log('✅ Critical charts preloaded');
    } catch (error) {
      console.warn('⚠️ Failed to preload some critical charts:', error);
    }
  }

  /**
   * Load chart based on data-chart-type attribute
   */
  async loadChartFromElement(element) {
    const chartType = element.dataset.chartType;
    const config = this._parseConfigFromElement(element);

    switch (chartType) {
      case 'timeline':
        return this.loadTimelineChart(element, config);
      case 'heatmap':
        return this.loadHeatmapChart(element, config);
      case 'realtime':
        return this.loadRealtimeDashboard(element, config);
      case '3d-scatter':
        return this.load3DVisualization(element, config);
      default:
        if (chartType.startsWith('advanced-')) {
          const advancedType = chartType.replace('advanced-', '');
          return this.loadAdvancedCharts(element, advancedType, config);
        }
        throw new Error(`Unknown chart type: ${chartType}`);
    }
  }

  /**
   * Cleanup loaded charts and free memory
   */
  cleanup() {
    this.loadedCharts.forEach((chart, type) => {
      if (chart && typeof chart.destroy === 'function') {
        chart.destroy();
      }
    });
    this.loadedCharts.clear();
    this.loadingPromises.clear();
  }

  // Private methods for dynamic imports

  async _loadTimelineChartModule() {
    const { createAnomalyTimeline } = await import(
      /* webpackChunkName: "chart-timeline" */
      './anomaly-timeline.js'
    );
    return { createAnomalyTimeline };
  }

  async _loadHeatmapChartModule() {
    const { createAnomalyHeatmap } = await import(
      /* webpackChunkName: "chart-heatmap" */
      './anomaly-heatmap.js'
    );
    return { createAnomalyHeatmap };
  }

  async _loadRealtimeDashboardModule() {
    const { createRealTimeDashboard } = await import(
      /* webpackChunkName: "chart-realtime" */
      './real-time-dashboard.js'
    );
    return { createRealTimeDashboard };
  }

  async _load3DVisualizationModule() {
    // Lazy load Plotly.js for 3D visualizations
    const [plotlyModule, chartModule] = await Promise.all([
      import(/* webpackChunkName: "plotly" */ 'plotly.js-dist-min'),
      import(/* webpackChunkName: "chart-3d" */ './3d-visualization.js')
    ]);

    return {
      create3DScatterPlot: chartModule.create3DScatterPlot,
      Plotly: plotlyModule.default
    };
  }

  async _loadAdvancedChartsModule() {
    const {
      createViolinPlot,
      createCorrelationMatrix,
      createDistributionPlot,
      createTimeSeriesDecomposition
    } = await import(
      /* webpackChunkName: "chart-advanced" */
      './advanced-charts.js'
    );

    return {
      createViolinPlot,
      createCorrelationMatrix,
      createDistributionPlot,
      createTimeSeriesDecomposition
    };
  }

  _parseConfigFromElement(element) {
    const config = {};

    // Parse data attributes
    Object.keys(element.dataset).forEach(key => {
      if (key.startsWith('chart')) {
        const configKey = key.replace('chart', '').toLowerCase();
        let value = element.dataset[key];

        // Try to parse as JSON, fallback to string
        try {
          value = JSON.parse(value);
        } catch (e) {
          // Keep as string
        }

        config[configKey] = value;
      }
    });

    // Default configuration
    return {
      width: element.clientWidth || 800,
      height: element.clientHeight || 400,
      responsive: true,
      ...config
    };
  }
}

/**
 * Global chart loader instance
 */
export const chartLoader = new ChartLoader();

/**
 * Initialize charts on page load with intersection observer for performance
 */
export function initializeChartsWithLazyLoading() {
  const chartElements = document.querySelectorAll('[data-chart-type]');

  if (!chartElements.length) return;

  // Use intersection observer for lazy loading
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const element = entry.target;

        // Add loading indicator
        element.innerHTML = `
          <div class="chart-loading">
            <div class="loading-spinner"></div>
            <p>Loading chart...</p>
          </div>
        `;

        // Load chart
        chartLoader.loadChartFromElement(element)
          .then(() => {
            console.log(`✅ Chart loaded: ${element.dataset.chartType}`);
          })
          .catch(error => {
            console.error(`❌ Failed to load chart: ${element.dataset.chartType}`, error);
            element.innerHTML = `
              <div class="chart-error">
                <p>Failed to load chart. <button onclick="location.reload()">Retry</button></p>
              </div>
            `;
          });

        // Stop observing this element
        observer.unobserve(element);
      }
    });
  }, {
    rootMargin: '50px 0px', // Start loading 50px before element is visible
    threshold: 0.1
  });

  // Observe all chart elements
  chartElements.forEach(element => observer.observe(element));

  // Preload critical charts
  chartLoader.preloadCriticalCharts();
}

/**
 * Cleanup function for page unload
 */
export function cleanupCharts() {
  chartLoader.cleanup();
}

// Auto-initialize when module is loaded
if (typeof window !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeChartsWithLazyLoading);
  } else {
    initializeChartsWithLazyLoading();
  }

  // Cleanup on page unload
  window.addEventListener('beforeunload', cleanupCharts);
}
