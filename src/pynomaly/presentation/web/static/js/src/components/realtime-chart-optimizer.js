/**
 * Real-Time Chart Optimization
 * 60 FPS chart rendering with efficient data buffering and smooth animations
 * Performance-optimized rendering engine for live data visualization
 */

/**
 * Frame Rate Controller
 * Manages frame rate optimization and smooth animations
 */
class FrameRateController {
  constructor(targetFPS = 60) {
    this.targetFPS = targetFPS;
    this.frameInterval = 1000 / targetFPS;
    this.lastFrameTime = 0;
    this.animationFrame = null;
    this.updateCallbacks = new Set();
    this.isRunning = false;
    
    // Performance monitoring
    this.frameCount = 0;
    this.actualFPS = 0;
    this.frameTimeHistory = [];
    this.maxHistorySize = 100;
  }

  start() {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.lastFrameTime = performance.now();
    this.frameLoop();
  }

  stop() {
    this.isRunning = false;
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  frameLoop() {
    if (!this.isRunning) return;

    const currentTime = performance.now();
    const deltaTime = currentTime - this.lastFrameTime;

    if (deltaTime >= this.frameInterval) {
      // Update performance metrics
      this.frameCount++;
      this.frameTimeHistory.push(deltaTime);
      if (this.frameTimeHistory.length > this.maxHistorySize) {
        this.frameTimeHistory.shift();
      }

      // Calculate actual FPS
      this.actualFPS = 1000 / (this.frameTimeHistory.reduce((a, b) => a + b, 0) / this.frameTimeHistory.length);

      // Execute update callbacks
      this.updateCallbacks.forEach(callback => {
        try {
          callback(deltaTime, currentTime);
        } catch (error) {
          console.error('Frame update callback error:', error);
        }
      });

      this.lastFrameTime = currentTime - (deltaTime % this.frameInterval);
    }

    this.animationFrame = requestAnimationFrame(() => this.frameLoop());
  }

  addUpdateCallback(callback) {
    this.updateCallbacks.add(callback);
    return () => this.updateCallbacks.delete(callback);
  }

  getPerformanceMetrics() {
    return {
      targetFPS: this.targetFPS,
      actualFPS: Math.round(this.actualFPS),
      frameCount: this.frameCount,
      averageFrameTime: this.frameTimeHistory.reduce((a, b) => a + b, 0) / this.frameTimeHistory.length,
      isRunning: this.isRunning
    };
  }
}

/**
 * Data Buffer Manager
 * Efficient circular buffer for real-time data with automatic memory management
 */
class DataBufferManager {
  constructor(maxSize = 10000, compressionThreshold = 0.8) {
    this.maxSize = maxSize;
    this.compressionThreshold = compressionThreshold;
    this.buffer = [];
    this.writeIndex = 0;
    this.readIndex = 0;
    this.size = 0;
    this.isCircular = false;
    
    // Compression settings
    this.compressionRatio = 0.5; // Compress to 50% when threshold reached
    this.lastCompressionTime = 0;
    
    // Performance tracking
    this.totalWrites = 0;
    this.totalReads = 0;
    this.compressionCount = 0;
  }

  add(item) {
    const timestamp = performance.now();
    const wrappedItem = {
      data: item,
      timestamp,
      id: this.totalWrites++
    };

    if (this.size < this.maxSize) {
      this.buffer[this.writeIndex] = wrappedItem;
      this.size++;
    } else {
      // Buffer is full, overwrite oldest item
      this.buffer[this.writeIndex] = wrappedItem;
      this.readIndex = (this.readIndex + 1) % this.maxSize;
      this.isCircular = true;
    }

    this.writeIndex = (this.writeIndex + 1) % this.maxSize;

    // Check if compression is needed
    if (this.size / this.maxSize > this.compressionThreshold) {
      this.compress();
    }

    return wrappedItem.id;
  }

  getLast(count = 1) {
    if (count <= 0 || this.size === 0) return [];

    const result = [];
    const actualCount = Math.min(count, this.size);
    
    for (let i = 0; i < actualCount; i++) {
      const index = this.isCircular 
        ? ((this.writeIndex - 1 - i + this.maxSize) % this.maxSize)
        : (this.writeIndex - 1 - i);
      
      if (index >= 0 && this.buffer[index]) {
        result.unshift(this.buffer[index]);
      }
    }

    this.totalReads += result.length;
    return result;
  }

  getRange(startTime, endTime) {
    const result = [];
    const items = this.getLast(this.size);
    
    for (const item of items) {
      if (item.timestamp >= startTime && item.timestamp <= endTime) {
        result.push(item);
      }
    }

    return result;
  }

  compress() {
    if (performance.now() - this.lastCompressionTime < 1000) {
      return; // Avoid frequent compression
    }

    const targetSize = Math.floor(this.maxSize * this.compressionRatio);
    const items = this.getLast(this.size);
    
    // Simple compression: keep every nth item
    const compressionFactor = Math.ceil(items.length / targetSize);
    const compressed = items.filter((_, index) => index % compressionFactor === 0);
    
    // Rebuild buffer
    this.buffer = new Array(this.maxSize);
    this.writeIndex = 0;
    this.readIndex = 0;
    this.size = 0;
    this.isCircular = false;
    
    // Add compressed items back
    compressed.forEach(item => this.add(item.data));
    
    this.compressionCount++;
    this.lastCompressionTime = performance.now();
  }

  clear() {
    this.buffer = [];
    this.writeIndex = 0;
    this.readIndex = 0;
    this.size = 0;
    this.isCircular = false;
  }

  getStats() {
    return {
      size: this.size,
      maxSize: this.maxSize,
      utilization: (this.size / this.maxSize * 100).toFixed(1) + '%',
      totalWrites: this.totalWrites,
      totalReads: this.totalReads,
      compressionCount: this.compressionCount,
      isCircular: this.isCircular
    };
  }
}

/**
 * Animation Manager
 * Smooth animations and transitions for real-time charts
 */
class AnimationManager {
  constructor() {
    this.animations = new Map();
    this.easingFunctions = {
      linear: t => t,
      easeInOut: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
      easeOut: t => 1 - Math.pow(1 - t, 3),
      easeIn: t => t * t * t,
      elastic: t => t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10 * (t - 1)) * Math.sin((t - 1.1) * 5 * Math.PI)
    };
  }

  animate(id, fromValue, toValue, duration, easing = 'easeInOut', onUpdate, onComplete) {
    const animation = {
      id,
      fromValue,
      toValue,
      duration,
      easing: this.easingFunctions[easing] || this.easingFunctions.easeInOut,
      onUpdate,
      onComplete,
      startTime: performance.now(),
      completed: false
    };

    this.animations.set(id, animation);
    return id;
  }

  update(currentTime) {
    for (const [id, animation] of this.animations) {
      const elapsed = currentTime - animation.startTime;
      const progress = Math.min(elapsed / animation.duration, 1);
      
      if (progress >= 1) {
        // Animation complete
        const finalValue = animation.toValue;
        animation.onUpdate?.(finalValue, 1);
        animation.onComplete?.(finalValue);
        this.animations.delete(id);
      } else {
        // Animation in progress
        const easedProgress = animation.easing(progress);
        const currentValue = animation.fromValue + (animation.toValue - animation.fromValue) * easedProgress;
        animation.onUpdate?.(currentValue, progress);
      }
    }
  }

  cancel(id) {
    return this.animations.delete(id);
  }

  cancelAll() {
    this.animations.clear();
  }

  isAnimating(id) {
    return this.animations.has(id);
  }
}

/**
 * Real-Time Chart Optimizer
 * Main optimization engine for real-time chart rendering
 */
class RealTimeChartOptimizer {
  constructor(options = {}) {
    this.options = {
      targetFPS: 60,
      maxDataPoints: 1000,
      bufferSize: 10000,
      enableAnimations: true,
      animationDuration: 300,
      adaptiveQuality: true,
      performanceThreshold: 30, // FPS threshold for quality adjustment
      ...options
    };

    // Core components
    this.frameController = new FrameRateController(this.options.targetFPS);
    this.dataBuffer = new DataBufferManager(this.options.bufferSize);
    this.animationManager = new AnimationManager();
    
    // Chart registry
    this.charts = new Map();
    this.chartUpdateQueue = new Set();
    
    // Performance monitoring
    this.performanceMonitor = {
      frameDrops: 0,
      averageFPS: 0,
      memoryUsage: 0,
      renderTime: 0,
      lastQualityAdjustment: 0
    };

    // Quality settings
    this.qualityLevel = 1.0; // 0.0 to 1.0
    this.qualitySettings = {
      high: { pointSize: 4, lineWidth: 2, antiAliasing: true, shadows: true },
      medium: { pointSize: 3, lineWidth: 1.5, antiAliasing: true, shadows: false },
      low: { pointSize: 2, lineWidth: 1, antiAliasing: false, shadows: false }
    };

    this.init();
  }

  init() {
    // Set up frame controller callback
    this.frameController.addUpdateCallback((deltaTime, currentTime) => {
      this.update(deltaTime, currentTime);
    });

    // Monitor performance
    setInterval(() => {
      this.updatePerformanceMetrics();
    }, 1000);

    // Start the optimization engine
    this.start();
  }

  start() {
    this.frameController.start();
  }

  stop() {
    this.frameController.stop();
  }

  /**
   * Chart Registration and Management
   */
  registerChart(chartId, chartInstance, options = {}) {
    const chartConfig = {
      instance: chartInstance,
      lastUpdate: 0,
      updateInterval: options.updateInterval || 16, // ~60 FPS
      isDirty: false,
      dataBuffer: new DataBufferManager(options.bufferSize || this.options.bufferSize),
      renderMode: options.renderMode || 'canvas', // 'canvas', 'svg', 'webgl'
      priority: options.priority || 1, // Higher priority = more frequent updates
      ...options
    };

    this.charts.set(chartId, chartConfig);
    return chartId;
  }

  unregisterChart(chartId) {
    const chart = this.charts.get(chartId);
    if (chart) {
      chart.dataBuffer.clear();
      this.charts.delete(chartId);
      this.chartUpdateQueue.delete(chartId);
    }
  }

  /**
   * Data Management
   */
  addDataPoint(chartId, dataPoint) {
    const chart = this.charts.get(chartId);
    if (!chart) return;

    const itemId = chart.dataBuffer.add(dataPoint);
    chart.isDirty = true;
    this.chartUpdateQueue.add(chartId);
    
    return itemId;
  }

  addBatchData(chartId, dataPoints) {
    const chart = this.charts.get(chartId);
    if (!chart) return;

    const itemIds = dataPoints.map(point => chart.dataBuffer.add(point));
    chart.isDirty = true;
    this.chartUpdateQueue.add(chartId);
    
    return itemIds;
  }

  getChartData(chartId, count = null) {
    const chart = this.charts.get(chartId);
    if (!chart) return [];

    return chart.dataBuffer.getLast(count || this.options.maxDataPoints);
  }

  /**
   * Update Loop
   */
  update(deltaTime, currentTime) {
    // Update animations
    this.animationManager.update(currentTime);

    // Update charts that need rendering
    this.updateCharts(currentTime);

    // Adaptive quality adjustment
    if (this.options.adaptiveQuality) {
      this.adjustQuality();
    }
  }

  updateCharts(currentTime) {
    const renderStartTime = performance.now();
    let chartsUpdated = 0;

    // Sort charts by priority
    const sortedCharts = Array.from(this.chartUpdateQueue)
      .map(id => ({ id, chart: this.charts.get(id) }))
      .filter(({ chart }) => chart)
      .sort((a, b) => b.chart.priority - a.chart.priority);

    for (const { id, chart } of sortedCharts) {
      if (currentTime - chart.lastUpdate >= chart.updateInterval) {
        try {
          this.renderChart(id, chart, currentTime);
          chart.lastUpdate = currentTime;
          chart.isDirty = false;
          chartsUpdated++;
        } catch (error) {
          console.error(`Error updating chart ${id}:`, error);
        }
      }
    }

    // Clear update queue for updated charts
    for (const { id, chart } of sortedCharts) {
      if (!chart.isDirty) {
        this.chartUpdateQueue.delete(id);
      }
    }

    this.performanceMonitor.renderTime = performance.now() - renderStartTime;
  }

  renderChart(chartId, chartConfig, currentTime) {
    const { instance, dataBuffer, renderMode } = chartConfig;
    
    // Get current data
    const data = dataBuffer.getLast(this.options.maxDataPoints);
    const processedData = this.processDataForRendering(data);

    // Apply quality settings
    const qualitySettings = this.getCurrentQualitySettings();
    
    // Render based on mode
    switch (renderMode) {
      case 'canvas':
        this.renderCanvasChart(instance, processedData, qualitySettings);
        break;
      case 'svg':
        this.renderSVGChart(instance, processedData, qualitySettings);
        break;
      case 'webgl':
        this.renderWebGLChart(instance, processedData, qualitySettings);
        break;
      default:
        this.renderDefaultChart(instance, processedData, qualitySettings);
    }

    // Emit update event
    instance.dispatchEvent?.(new CustomEvent('chart-updated', {
      detail: { chartId, dataPoints: processedData.length, timestamp: currentTime }
    }));
  }

  processDataForRendering(data) {
    // Convert wrapped data back to raw format
    return data.map(item => ({
      ...item.data,
      timestamp: item.timestamp,
      id: item.id
    }));
  }

  renderCanvasChart(chartInstance, data, quality) {
    if (chartInstance.updateData) {
      chartInstance.updateData(data, {
        pointSize: quality.pointSize,
        lineWidth: quality.lineWidth,
        antiAliasing: quality.antiAliasing,
        enableAnimations: this.options.enableAnimations && quality.antiAliasing
      });
    }
  }

  renderSVGChart(chartInstance, data, quality) {
    if (chartInstance.updateData) {
      chartInstance.updateData(data, {
        strokeWidth: quality.lineWidth,
        pointRadius: quality.pointSize,
        enableTransitions: this.options.enableAnimations
      });
    }
  }

  renderWebGLChart(chartInstance, data, quality) {
    if (chartInstance.updateData) {
      chartInstance.updateData(data, {
        pointSize: quality.pointSize,
        lineWidth: quality.lineWidth,
        enableShaders: quality.antiAliasing
      });
    }
  }

  renderDefaultChart(chartInstance, data, quality) {
    if (chartInstance.setData) {
      chartInstance.setData(data);
    } else if (chartInstance.updateChart) {
      chartInstance.updateChart();
    }
  }

  /**
   * Quality Management
   */
  adjustQuality() {
    const now = performance.now();
    if (now - this.performanceMonitor.lastQualityAdjustment < 2000) {
      return; // Don't adjust too frequently
    }

    const metrics = this.frameController.getPerformanceMetrics();
    const currentFPS = metrics.actualFPS;

    if (currentFPS < this.options.performanceThreshold && this.qualityLevel > 0.3) {
      // Decrease quality
      this.qualityLevel = Math.max(0.3, this.qualityLevel - 0.1);
      console.log(`Quality decreased to ${this.qualityLevel.toFixed(1)} (FPS: ${currentFPS})`);
    } else if (currentFPS > this.options.performanceThreshold + 10 && this.qualityLevel < 1.0) {
      // Increase quality
      this.qualityLevel = Math.min(1.0, this.qualityLevel + 0.05);
      console.log(`Quality increased to ${this.qualityLevel.toFixed(1)} (FPS: ${currentFPS})`);
    }

    this.performanceMonitor.lastQualityAdjustment = now;
  }

  getCurrentQualitySettings() {
    if (this.qualityLevel >= 0.8) {
      return this.qualitySettings.high;
    } else if (this.qualityLevel >= 0.5) {
      return this.qualitySettings.medium;
    } else {
      return this.qualitySettings.low;
    }
  }

  setQualityLevel(level) {
    this.qualityLevel = Math.max(0.1, Math.min(1.0, level));
  }

  /**
   * Animation Support
   */
  animateChartTransition(chartId, property, fromValue, toValue, duration = null) {
    const animationId = `${chartId}_${property}`;
    const animationDuration = duration || this.options.animationDuration;

    return this.animationManager.animate(
      animationId,
      fromValue,
      toValue,
      animationDuration,
      'easeInOut',
      (value, progress) => {
        const chart = this.charts.get(chartId);
        if (chart && chart.instance.setProperty) {
          chart.instance.setProperty(property, value);
        }
      },
      (finalValue) => {
        const chart = this.charts.get(chartId);
        if (chart) {
          chart.isDirty = true;
          this.chartUpdateQueue.add(chartId);
        }
      }
    );
  }

  /**
   * Performance Monitoring
   */
  updatePerformanceMetrics() {
    const frameMetrics = this.frameController.getPerformanceMetrics();
    this.performanceMonitor.averageFPS = frameMetrics.actualFPS;

    // Detect frame drops
    if (frameMetrics.actualFPS < this.options.targetFPS * 0.8) {
      this.performanceMonitor.frameDrops++;
    }

    // Memory usage (if available)
    if (performance.memory) {
      this.performanceMonitor.memoryUsage = performance.memory.usedJSHeapSize / 1024 / 1024; // MB
    }
  }

  getPerformanceReport() {
    const frameMetrics = this.frameController.getPerformanceMetrics();
    
    return {
      fps: {
        target: frameMetrics.targetFPS,
        actual: frameMetrics.actualFPS,
        frameCount: frameMetrics.frameCount
      },
      quality: {
        level: this.qualityLevel,
        settings: this.getCurrentQualitySettings()
      },
      charts: {
        registered: this.charts.size,
        updateQueue: this.chartUpdateQueue.size
      },
      performance: {
        ...this.performanceMonitor,
        renderTime: this.performanceMonitor.renderTime.toFixed(2) + 'ms'
      },
      buffers: Array.from(this.charts.entries()).map(([id, chart]) => ({
        chartId: id,
        ...chart.dataBuffer.getStats()
      }))
    };
  }

  /**
   * Cleanup
   */
  destroy() {
    this.stop();
    this.animationManager.cancelAll();
    
    // Clear all chart buffers
    for (const [id, chart] of this.charts) {
      chart.dataBuffer.clear();
    }
    
    this.charts.clear();
    this.chartUpdateQueue.clear();
  }
}

/**
 * Global optimizer instance
 */
const globalChartOptimizer = new RealTimeChartOptimizer();

// Export classes and global instance
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    RealTimeChartOptimizer,
    FrameRateController,
    DataBufferManager,
    AnimationManager,
    globalChartOptimizer
  };
} else {
  // Browser environment
  window.RealTimeChartOptimizer = RealTimeChartOptimizer;
  window.FrameRateController = FrameRateController;
  window.DataBufferManager = DataBufferManager;
  window.AnimationManager = AnimationManager;
  window.globalChartOptimizer = globalChartOptimizer;
}
