/**
 * High-Performance Chart Rendering System
 * Optimizes chart performance through canvas pooling, level-of-detail, and smart updates
 */

export class ChartPerformanceManager {
  constructor(options = {}) {
    this.options = {
      canvasPoolSize: 10,
      maxDataPoints: 10000,
      levelOfDetailThreshold: 1000,
      updateThrottleMs: 16, // 60fps
      enableWebGL: true,
      enableOffscreenCanvas: true,
      memoryThreshold: 100 * 1024 * 1024, // 100MB
      ...options,
    };

    this.canvasPool = [];
    this.activeCharts = new Map();
    this.renderQueue = [];
    this.animationFrame = null;
    this.performanceStats = {
      renderCount: 0,
      averageRenderTime: 0,
      memoryUsage: 0,
      lastUpdate: Date.now(),
    };

    this.init();
  }

  init() {
    this.initializeCanvasPool();
    this.setupRenderLoop();
    this.initializeWebGL();
    this.setupPerformanceMonitoring();
  }

  // === CANVAS POOLING ===

  initializeCanvasPool() {
    for (let i = 0; i < this.options.canvasPoolSize; i++) {
      const canvas = this.createOptimizedCanvas();
      this.canvasPool.push(canvas);
    }
  }

  createOptimizedCanvas(width = 800, height = 400) {
    const canvas = document.createElement('canvas');
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d', {
      alpha: false,
      desynchronized: true,
      powerPreference: 'high-performance',
    });

    // Apply optimizations
    ctx.imageSmoothingEnabled = false;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    canvas._ctx = ctx;
    canvas._inUse = false;
    canvas._chartId = null;

    return canvas;
  }

  acquireCanvas(chartId, width, height) {
    let canvas = this.canvasPool.find(c => !c._inUse);

    if (!canvas) {
      canvas = this.createOptimizedCanvas(width, height);
      this.canvasPool.push(canvas);
    } else {
      this.resizeCanvas(canvas, width, height);
    }

    canvas._inUse = true;
    canvas._chartId = chartId;
    return canvas;
  }

  releaseCanvas(canvas) {
    if (canvas && canvas._inUse) {
      canvas._inUse = false;
      canvas._chartId = null;

      // Clear canvas
      const ctx = canvas._ctx;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  resizeCanvas(canvas, width, height) {
    const dpr = window.devicePixelRatio;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas._ctx.scale(dpr, dpr);
  }

  // === LEVEL OF DETAIL (LOD) ===

  applyLevelOfDetail(data, chartType, viewportWidth) {
    if (data.length <= this.options.levelOfDetailThreshold) {
      return data;
    }

    const targetPoints = Math.min(
      Math.floor(viewportWidth / 2), // One point per 2 pixels
      this.options.maxDataPoints
    );

    switch (chartType) {
      case 'line':
      case 'area':
        return this.simplifyLineData(data, targetPoints);
      case 'scatter':
        return this.clusterScatterData(data, targetPoints);
      case 'heatmap':
        return this.downsampleHeatmapData(data, targetPoints);
      default:
        return this.uniformSample(data, targetPoints);
    }
  }

  simplifyLineData(data, targetPoints) {
    // Douglas-Peucker algorithm for line simplification
    if (data.length <= targetPoints) return data;

    const epsilon = this.calculateEpsilon(data, targetPoints);
    return this.douglasPeucker(data, epsilon);
  }

  douglasPeucker(points, epsilon) {
    if (points.length <= 2) return points;

    let maxDistance = 0;
    let maxIndex = 0;

    for (let i = 1; i < points.length - 1; i++) {
      const distance = this.perpendicularDistance(
        points[i],
        points[0],
        points[points.length - 1]
      );
      if (distance > maxDistance) {
        maxDistance = distance;
        maxIndex = i;
      }
    }

    if (maxDistance > epsilon) {
      const left = this.douglasPeucker(points.slice(0, maxIndex + 1), epsilon);
      const right = this.douglasPeucker(points.slice(maxIndex), epsilon);
      return left.slice(0, -1).concat(right);
    } else {
      return [points[0], points[points.length - 1]];
    }
  }

  perpendicularDistance(point, lineStart, lineEnd) {
    const dx = lineEnd.x - lineStart.x;
    const dy = lineEnd.y - lineStart.y;
    const normalLength = Math.sqrt(dx * dx + dy * dy);

    if (normalLength === 0) return 0;

    return Math.abs(
      (dy * point.x - dx * point.y + lineEnd.x * lineStart.y - lineEnd.y * lineStart.x) / normalLength
    );
  }

  calculateEpsilon(data, targetPoints) {
    const dataRange = this.getDataRange(data);
    const tolerance = (dataRange.maxY - dataRange.minY) / 1000;
    return tolerance * (data.length / targetPoints);
  }

  clusterScatterData(data, targetPoints) {
    // K-means clustering for scatter plot data reduction
    const clusters = this.kMeansClustering(data, targetPoints);
    return clusters.map(cluster => ({
      x: cluster.centroid.x,
      y: cluster.centroid.y,
      size: cluster.points.length,
      originalPoints: cluster.points,
    }));
  }

  kMeansClustering(data, k) {
    // Simplified k-means implementation
    let centroids = this.initializeCentroids(data, k);
    let clusters = [];
    let iterations = 0;
    const maxIterations = 10;

    while (iterations < maxIterations) {
      clusters = this.assignPointsToClusters(data, centroids);
      const newCentroids = this.calculateNewCentroids(clusters);

      if (this.centroidsConverged(centroids, newCentroids)) break;

      centroids = newCentroids;
      iterations++;
    }

    return clusters;
  }

  initializeCentroids(data, k) {
    const centroids = [];
    for (let i = 0; i < k; i++) {
      centroids.push(data[Math.floor(Math.random() * data.length)]);
    }
    return centroids;
  }

  assignPointsToClusters(data, centroids) {
    const clusters = centroids.map(centroid => ({ centroid, points: [] }));

    data.forEach(point => {
      let minDistance = Infinity;
      let closestClusterIndex = 0;

      centroids.forEach((centroid, index) => {
        const distance = this.euclideanDistance(point, centroid);
        if (distance < minDistance) {
          minDistance = distance;
          closestClusterIndex = index;
        }
      });

      clusters[closestClusterIndex].points.push(point);
    });

    return clusters;
  }

  calculateNewCentroids(clusters) {
    return clusters.map(cluster => {
      if (cluster.points.length === 0) return cluster.centroid;

      const sumX = cluster.points.reduce((sum, point) => sum + point.x, 0);
      const sumY = cluster.points.reduce((sum, point) => sum + point.y, 0);

      return {
        x: sumX / cluster.points.length,
        y: sumY / cluster.points.length,
      };
    });
  }

  centroidsConverged(oldCentroids, newCentroids, threshold = 0.1) {
    return oldCentroids.every((oldCentroid, index) => {
      const newCentroid = newCentroids[index];
      return this.euclideanDistance(oldCentroid, newCentroid) < threshold;
    });
  }

  euclideanDistance(point1, point2) {
    const dx = point1.x - point2.x;
    const dy = point1.y - point2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  downsampleHeatmapData(data, targetPoints) {
    // Grid-based downsampling for heatmap data
    const gridSize = Math.ceil(Math.sqrt(data.length / targetPoints));
    const grid = new Map();

    data.forEach(point => {
      const gridX = Math.floor(point.x / gridSize) * gridSize;
      const gridY = Math.floor(point.y / gridSize) * gridSize;
      const key = `${gridX},${gridY}`;

      if (!grid.has(key)) {
        grid.set(key, { x: gridX, y: gridY, values: [], count: 0 });
      }

      const cell = grid.get(key);
      cell.values.push(point.value);
      cell.count++;
    });

    return Array.from(grid.values()).map(cell => ({
      x: cell.x,
      y: cell.y,
      value: cell.values.reduce((sum, v) => sum + v, 0) / cell.values.length,
      count: cell.count,
    }));
  }

  uniformSample(data, targetPoints) {
    if (data.length <= targetPoints) return data;

    const step = data.length / targetPoints;
    const sampled = [];

    for (let i = 0; i < targetPoints; i++) {
      const index = Math.floor(i * step);
      sampled.push(data[index]);
    }

    return sampled;
  }

  // === SMART RENDERING ===

  setupRenderLoop() {
    const render = () => {
      this.processRenderQueue();
      this.animationFrame = requestAnimationFrame(render);
    };
    this.animationFrame = requestAnimationFrame(render);
  }

  scheduleRender(chartId, renderFunction, priority = 'normal') {
    // Remove existing render request for this chart
    this.renderQueue = this.renderQueue.filter(item => item.chartId !== chartId);

    // Add new render request
    this.renderQueue.push({
      chartId,
      renderFunction,
      priority,
      timestamp: Date.now(),
    });

    // Sort by priority
    this.renderQueue.sort((a, b) => {
      const priorityOrder = { high: 3, normal: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  processRenderQueue() {
    const startTime = performance.now();
    const frameTimeLimit = 16; // 16ms for 60fps

    while (this.renderQueue.length > 0 && (performance.now() - startTime) < frameTimeLimit) {
      const renderTask = this.renderQueue.shift();

      try {
        renderTask.renderFunction();
        this.updatePerformanceStats(performance.now() - startTime);
      } catch (error) {
        console.error(`Render error for chart ${renderTask.chartId}:`, error);
      }
    }
  }

  // === WEBGL OPTIMIZATION ===

  initializeWebGL() {
    if (!this.options.enableWebGL) return;

    this.webglCanvas = document.createElement('canvas');
    this.gl = this.webglCanvas.getContext('webgl2') || this.webglCanvas.getContext('webgl');

    if (this.gl) {
      this.initializeWebGLPrograms();
    }
  }

  initializeWebGLPrograms() {
    // Vertex shader for scatter plots
    const scatterVertexShader = this.createShader(this.gl.VERTEX_SHADER, `
      attribute vec2 a_position;
      attribute float a_size;
      attribute vec3 a_color;

      uniform vec2 u_resolution;
      uniform vec2 u_translation;
      uniform vec2 u_scale;

      varying vec3 v_color;

      void main() {
        vec2 position = (a_position + u_translation) * u_scale;
        vec2 clipSpace = ((position / u_resolution) * 2.0) - 1.0;

        gl_Position = vec4(clipSpace * vec2(1, -1), 0, 1);
        gl_PointSize = a_size;
        v_color = a_color;
      }
    `);

    // Fragment shader for scatter plots
    const scatterFragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, `
      precision mediump float;
      varying vec3 v_color;

      void main() {
        vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
        if (dot(circCoord, circCoord) > 1.0) {
          discard;
        }
        gl_FragColor = vec4(v_color, 1.0);
      }
    `);

    this.scatterProgram = this.createProgram(scatterVertexShader, scatterFragmentShader);
  }

  createShader(type, source) {
    const shader = this.gl.createShader(type);
    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      console.error('Shader compilation error:', this.gl.getShaderInfoLog(shader));
      this.gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  createProgram(vertexShader, fragmentShader) {
    const program = this.gl.createProgram();
    this.gl.attachShader(program, vertexShader);
    this.gl.attachShader(program, fragmentShader);
    this.gl.linkProgram(program);

    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      console.error('Program linking error:', this.gl.getProgramInfoLog(program));
      this.gl.deleteProgram(program);
      return null;
    }

    return program;
  }

  renderScatterPlotWebGL(data, canvas, options) {
    if (!this.gl || !this.scatterProgram) return false;

    this.webglCanvas.width = canvas.width;
    this.webglCanvas.height = canvas.height;
    this.gl.viewport(0, 0, canvas.width, canvas.height);

    this.gl.useProgram(this.scatterProgram);

    // Prepare data
    const positions = [];
    const colors = [];
    const sizes = [];

    data.forEach(point => {
      positions.push(point.x, point.y);
      colors.push(point.color?.r || 1, point.color?.g || 0, point.color?.b || 0);
      sizes.push(point.size || 5);
    });

    // Set up buffers
    const positionBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, positionBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.STATIC_DRAW);

    const positionLocation = this.gl.getAttribLocation(this.scatterProgram, 'a_position');
    this.gl.enableVertexAttribArray(positionLocation);
    this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0);

    // Set uniforms
    const resolutionLocation = this.gl.getUniformLocation(this.scatterProgram, 'u_resolution');
    this.gl.uniform2f(resolutionLocation, canvas.width, canvas.height);

    // Clear and draw
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    this.gl.drawArrays(this.gl.POINTS, 0, data.length);

    // Copy to target canvas
    const ctx = canvas.getContext('2d');
    ctx.drawImage(this.webglCanvas, 0, 0);

    return true;
  }

  // === PERFORMANCE MONITORING ===

  setupPerformanceMonitoring() {
    setInterval(() => {
      this.collectMemoryStats();
      this.optimizeIfNeeded();
    }, 5000);
  }

  updatePerformanceStats(renderTime) {
    this.performanceStats.renderCount++;
    this.performanceStats.averageRenderTime =
      (this.performanceStats.averageRenderTime + renderTime) / 2;
    this.performanceStats.lastUpdate = Date.now();
  }

  collectMemoryStats() {
    if ('memory' in performance) {
      this.performanceStats.memoryUsage = performance.memory.usedJSHeapSize;
    }
  }

  optimizeIfNeeded() {
    const { memoryUsage, averageRenderTime } = this.performanceStats;

    if (memoryUsage > this.options.memoryThreshold) {
      this.freeUnusedCanvases();
    }

    if (averageRenderTime > 16) { // Over 60fps budget
      this.reduceLevelOfDetail();
    }
  }

  freeUnusedCanvases() {
    const unusedCanvases = this.canvasPool.filter(canvas => !canvas._inUse);
    if (unusedCanvases.length > this.options.canvasPoolSize / 2) {
      this.canvasPool = this.canvasPool.filter(canvas => canvas._inUse);
    }
  }

  reduceLevelOfDetail() {
    this.options.levelOfDetailThreshold *= 0.8;
    this.options.maxDataPoints *= 0.9;
  }

  // === UTILITY METHODS ===

  getDataRange(data) {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    data.forEach(point => {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    });

    return { minX, maxX, minY, maxY };
  }

  // === PUBLIC API ===

  optimizeChart(chartConfig) {
    const { data, type, container } = chartConfig;

    // Apply level of detail
    const optimizedData = this.applyLevelOfDetail(
      data,
      type,
      container.clientWidth
    );

    // Get optimized canvas
    const canvas = this.acquireCanvas(
      chartConfig.id,
      container.clientWidth,
      container.clientHeight
    );

    return {
      ...chartConfig,
      data: optimizedData,
      canvas,
      renderFunction: () => this.scheduleRender(chartConfig.id, chartConfig.render),
    };
  }

  registerChart(chartId, chart) {
    this.activeCharts.set(chartId, chart);
  }

  unregisterChart(chartId) {
    const chart = this.activeCharts.get(chartId);
    if (chart && chart.canvas) {
      this.releaseCanvas(chart.canvas);
    }
    this.activeCharts.delete(chartId);
  }

  getPerformanceStats() {
    return {
      ...this.performanceStats,
      activeCharts: this.activeCharts.size,
      canvasPoolSize: this.canvasPool.length,
      renderQueueSize: this.renderQueue.length,
    };
  }

  destroy() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }

    this.canvasPool.forEach(canvas => {
      if (canvas.parentNode) {
        canvas.parentNode.removeChild(canvas);
      }
    });

    this.activeCharts.clear();
    this.renderQueue = [];

    if (this.gl) {
      this.gl.getExtension('WEBGL_lose_context')?.loseContext();
    }
  }
}

// Export singleton instance
export const chartPerformanceManager = new ChartPerformanceManager();
export default ChartPerformanceManager;
