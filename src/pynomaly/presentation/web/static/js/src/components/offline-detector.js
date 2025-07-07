/**
 * Offline Anomaly Detection Component
 * Provides anomaly detection capabilities using cached data and local algorithms
 */
export class OfflineDetector {
  constructor() {
    this.algorithms = new Map();
    this.cachedDatasets = new Map();
    this.cachedModels = new Map();
    this.isInitialized = false;
    
    this.initializeAlgorithms();
  }

  /**
   * Initialize local anomaly detection algorithms
   */
  initializeAlgorithms() {
    // Simple statistical algorithms that can run in the browser
    this.algorithms.set('zscore', {
      name: 'Z-Score Detection',
      description: 'Statistical outlier detection using Z-scores',
      parameters: { threshold: 3.0 },
      detect: this.zScoreDetection.bind(this)
    });

    this.algorithms.set('iqr', {
      name: 'Interquartile Range',
      description: 'Outlier detection using IQR method',
      parameters: { factor: 1.5 },
      detect: this.iqrDetection.bind(this)
    });

    this.algorithms.set('isolation', {
      name: 'Simple Isolation Detection',
      description: 'Basic isolation-based anomaly detection',
      parameters: { contamination: 0.1 },
      detect: this.isolationDetection.bind(this)
    });

    this.algorithms.set('mad', {
      name: 'Median Absolute Deviation',
      description: 'Robust outlier detection using MAD',
      parameters: { threshold: 3.5 },
      detect: this.madDetection.bind(this)
    });

    this.isInitialized = true;
  }

  /**
   * Load cached datasets from IndexedDB
   */
  async loadCachedDatasets() {
    try {
      if ('serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          // Request cached datasets from service worker
          registration.active.postMessage({ type: 'GET_OFFLINE_DATASETS' });
          
          // Listen for response
          return new Promise((resolve) => {
            navigator.serviceWorker.addEventListener('message', function handler(event) {
              if (event.data.type === 'OFFLINE_DATASETS') {
                navigator.serviceWorker.removeEventListener('message', handler);
                event.data.datasets.forEach(dataset => {
                  this.cachedDatasets.set(dataset.id, dataset);
                });
                resolve(event.data.datasets);
              }
            }.bind(this));
          });
        }
      }
    } catch (error) {
      console.error('[OfflineDetector] Failed to load cached datasets:', error);
      return [];
    }
  }

  /**
   * Get available algorithms
   */
  getAlgorithms() {
    return Array.from(this.algorithms.entries()).map(([id, algo]) => ({
      id,
      name: algo.name,
      description: algo.description,
      parameters: algo.parameters
    }));
  }

  /**
   * Get cached datasets
   */
  getCachedDatasets() {
    return Array.from(this.cachedDatasets.values());
  }

  /**
   * Run anomaly detection on cached data
   */
  async detectAnomalies(datasetId, algorithmId, parameters = {}) {
    if (!this.isInitialized) {
      throw new Error('Offline detector not initialized');
    }

    const dataset = this.cachedDatasets.get(datasetId);
    if (!dataset) {
      throw new Error(`Dataset ${datasetId} not found in cache`);
    }

    const algorithm = this.algorithms.get(algorithmId);
    if (!algorithm) {
      throw new Error(`Algorithm ${algorithmId} not available`);
    }

    const startTime = performance.now();
    
    try {
      // Prepare data
      const data = this.prepareData(dataset.data);
      
      // Run detection
      const config = { ...algorithm.parameters, ...parameters };
      const result = algorithm.detect(data, config);
      
      const endTime = performance.now();
      const processingTime = endTime - startTime;

      // Create detection result
      const detectionResult = {
        id: `offline_${Date.now()}`,
        datasetId: datasetId,
        algorithmId: algorithmId,
        timestamp: new Date().toISOString(),
        processingTimeMs: processingTime,
        anomalies: result.anomalies,
        scores: result.scores,
        statistics: result.statistics,
        parameters: config,
        isOffline: true
      };

      // Save to offline storage
      await this.saveResult(detectionResult);

      return detectionResult;
    } catch (error) {
      console.error('[OfflineDetector] Detection failed:', error);
      throw error;
    }
  }

  /**
   * Prepare data for analysis
   */
  prepareData(rawData) {
    // Convert to numeric matrix if needed
    if (Array.isArray(rawData)) {
      return rawData.map(row => {
        if (typeof row === 'object') {
          return Object.values(row).map(val => {
            const num = parseFloat(val);
            return isNaN(num) ? 0 : num;
          });
        }
        return Array.isArray(row) ? row : [row];
      });
    }
    return rawData;
  }

  /**
   * Z-Score based anomaly detection
   */
  zScoreDetection(data, config) {
    const { threshold = 3.0 } = config;
    const anomalies = [];
    const scores = [];
    
    // Calculate statistics for each feature
    const features = data[0].length;
    const featureStats = [];
    
    for (let f = 0; f < features; f++) {
      const values = data.map(row => row[f]);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance);
      
      featureStats.push({ mean, std });
    }
    
    // Calculate Z-scores and detect anomalies
    data.forEach((row, index) => {
      let maxZScore = 0;
      
      row.forEach((value, featureIndex) => {
        const { mean, std } = featureStats[featureIndex];
        const zScore = std > 0 ? Math.abs((value - mean) / std) : 0;
        maxZScore = Math.max(maxZScore, zScore);
      });
      
      scores.push(maxZScore);
      
      if (maxZScore > threshold) {
        anomalies.push({
          index,
          score: maxZScore,
          values: row
        });
      }
    });

    return {
      anomalies,
      scores,
      statistics: {
        totalSamples: data.length,
        totalAnomalies: anomalies.length,
        anomalyRate: anomalies.length / data.length,
        averageScore: scores.reduce((a, b) => a + b, 0) / scores.length,
        maxScore: Math.max(...scores),
        threshold
      }
    };
  }

  /**
   * IQR based anomaly detection
   */
  iqrDetection(data, config) {
    const { factor = 1.5 } = config;
    const anomalies = [];
    const scores = [];
    
    // Calculate IQR for each feature
    const features = data[0].length;
    const featureBounds = [];
    
    for (let f = 0; f < features; f++) {
      const values = data.map(row => row[f]).sort((a, b) => a - b);
      const q1Index = Math.floor(values.length * 0.25);
      const q3Index = Math.floor(values.length * 0.75);
      const q1 = values[q1Index];
      const q3 = values[q3Index];
      const iqr = q3 - q1;
      
      featureBounds.push({
        lower: q1 - factor * iqr,
        upper: q3 + factor * iqr,
        iqr
      });
    }
    
    // Detect anomalies
    data.forEach((row, index) => {
      let anomalyScore = 0;
      let isAnomaly = false;
      
      row.forEach((value, featureIndex) => {
        const bounds = featureBounds[featureIndex];
        if (value < bounds.lower || value > bounds.upper) {
          isAnomaly = true;
          const deviation = Math.min(
            Math.abs(value - bounds.lower),
            Math.abs(value - bounds.upper)
          );
          anomalyScore = Math.max(anomalyScore, deviation / Math.max(bounds.iqr, 1));
        }
      });
      
      scores.push(anomalyScore);
      
      if (isAnomaly) {
        anomalies.push({
          index,
          score: anomalyScore,
          values: row
        });
      }
    });

    return {
      anomalies,
      scores,
      statistics: {
        totalSamples: data.length,
        totalAnomalies: anomalies.length,
        anomalyRate: anomalies.length / data.length,
        averageScore: scores.reduce((a, b) => a + b, 0) / scores.length,
        maxScore: Math.max(...scores),
        factor
      }
    };
  }

  /**
   * Simple isolation-based detection
   */
  isolationDetection(data, config) {
    const { contamination = 0.1 } = config;
    
    // Simple implementation: random feature selection and isolation
    const scores = data.map(() => 0);
    const numTrees = 100;
    const maxDepth = Math.ceil(Math.log2(data.length));
    
    for (let tree = 0; tree < numTrees; tree++) {
      const pathLengths = this.isolationTree(data, maxDepth);
      pathLengths.forEach((length, index) => {
        scores[index] += length;
      });
    }
    
    // Normalize scores
    const avgPathLength = scores.reduce((a, b) => a + b, 0) / scores.length;
    const normalizedScores = scores.map(score => 
      Math.pow(2, -(score / numTrees) / avgPathLength)
    );
    
    // Determine threshold based on contamination rate
    const sortedScores = [...normalizedScores].sort((a, b) => b - a);
    const thresholdIndex = Math.floor(data.length * contamination);
    const threshold = sortedScores[thresholdIndex] || 0.5;
    
    const anomalies = [];
    normalizedScores.forEach((score, index) => {
      if (score > threshold) {
        anomalies.push({
          index,
          score,
          values: data[index]
        });
      }
    });

    return {
      anomalies,
      scores: normalizedScores,
      statistics: {
        totalSamples: data.length,
        totalAnomalies: anomalies.length,
        anomalyRate: anomalies.length / data.length,
        averageScore: normalizedScores.reduce((a, b) => a + b, 0) / normalizedScores.length,
        maxScore: Math.max(...normalizedScores),
        threshold,
        contamination
      }
    };
  }

  /**
   * Simple isolation tree implementation
   */
  isolationTree(data, maxDepth, currentDepth = 0) {
    if (currentDepth >= maxDepth || data.length <= 1) {
      return data.map(() => currentDepth);
    }

    // Random feature selection
    const featureIndex = Math.floor(Math.random() * data[0].length);
    const values = data.map(row => row[featureIndex]);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    
    if (minVal === maxVal) {
      return data.map(() => currentDepth);
    }
    
    // Random split point
    const splitPoint = minVal + Math.random() * (maxVal - minVal);
    
    // Split data
    const leftData = [];
    const rightData = [];
    const leftIndices = [];
    const rightIndices = [];
    
    data.forEach((row, index) => {
      if (row[featureIndex] < splitPoint) {
        leftData.push(row);
        leftIndices.push(index);
      } else {
        rightData.push(row);
        rightIndices.push(index);
      }
    });
    
    // Recursive calls
    const leftPaths = leftData.length > 0 ? 
      this.isolationTree(leftData, maxDepth, currentDepth + 1) : [];
    const rightPaths = rightData.length > 0 ? 
      this.isolationTree(rightData, maxDepth, currentDepth + 1) : [];
    
    // Combine results
    const result = new Array(data.length);
    leftIndices.forEach((originalIndex, i) => {
      result[i] = leftPaths[i];
    });
    rightIndices.forEach((originalIndex, i) => {
      result[i + leftData.length] = rightPaths[i];
    });
    
    return result;
  }

  /**
   * MAD (Median Absolute Deviation) based detection
   */
  madDetection(data, config) {
    const { threshold = 3.5 } = config;
    const anomalies = [];
    const scores = [];
    
    // Calculate MAD for each feature
    const features = data[0].length;
    const featureStats = [];
    
    for (let f = 0; f < features; f++) {
      const values = data.map(row => row[f]).sort((a, b) => a - b);
      const median = values[Math.floor(values.length / 2)];
      const deviations = values.map(val => Math.abs(val - median)).sort((a, b) => a - b);
      const mad = deviations[Math.floor(deviations.length / 2)];
      
      featureStats.push({ median, mad });
    }
    
    // Calculate modified Z-scores and detect anomalies
    data.forEach((row, index) => {
      let maxScore = 0;
      
      row.forEach((value, featureIndex) => {
        const { median, mad } = featureStats[featureIndex];
        const modifiedZScore = mad > 0 ? (0.6745 * (value - median)) / mad : 0;
        maxScore = Math.max(maxScore, Math.abs(modifiedZScore));
      });
      
      scores.push(maxScore);
      
      if (maxScore > threshold) {
        anomalies.push({
          index,
          score: maxScore,
          values: row
        });
      }
    });

    return {
      anomalies,
      scores,
      statistics: {
        totalSamples: data.length,
        totalAnomalies: anomalies.length,
        anomalyRate: anomalies.length / data.length,
        averageScore: scores.reduce((a, b) => a + b, 0) / scores.length,
        maxScore: Math.max(...scores),
        threshold
      }
    };
  }

  /**
   * Save detection result to offline storage
   */
  async saveResult(result) {
    try {
      if ('serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          registration.active.postMessage({
            type: 'SAVE_DETECTION_RESULT',
            payload: result
          });
        }
      }
    } catch (error) {
      console.error('[OfflineDetector] Failed to save result:', error);
    }
  }

  /**
   * Get detection history from offline storage
   */
  async getDetectionHistory() {
    try {
      if ('serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.getRegistration();
        if (registration?.active) {
          registration.active.postMessage({ type: 'GET_OFFLINE_RESULTS' });
          
          return new Promise((resolve) => {
            navigator.serviceWorker.addEventListener('message', function handler(event) {
              if (event.data.type === 'OFFLINE_RESULTS') {
                navigator.serviceWorker.removeEventListener('message', handler);
                resolve(event.data.results);
              }
            });
          });
        }
      }
    } catch (error) {
      console.error('[OfflineDetector] Failed to get detection history:', error);
      return [];
    }
  }
}

// Initialize and expose globally
if (typeof window !== 'undefined') {
  window.OfflineDetector = new OfflineDetector();
}
