/**
 * Advanced Analytics Engine
 * Real-time statistical analysis with trend detection and smart alerting system
 * Provides comprehensive anomaly analysis, pattern recognition, and predictive insights
 */

/**
 * Statistical Analysis Library
 * Core statistical functions for real-time analysis
 */
class StatisticalAnalysis {
  static mean(values) {
    return values.length === 0 ? 0 : values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  static median(values) {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 
      ? (sorted[mid - 1] + sorted[mid]) / 2 
      : sorted[mid];
  }

  static standardDeviation(values) {
    if (values.length < 2) return 0;
    const mean = this.mean(values);
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (values.length - 1);
    return Math.sqrt(variance);
  }

  static percentile(values, p) {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) return sorted[lower];
    return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
  }

  static zScore(value, mean, stdDev) {
    return stdDev === 0 ? 0 : (value - mean) / stdDev;
  }

  static correlationCoefficient(x, y) {
    if (x.length !== y.length || x.length < 2) return 0;
    
    const meanX = this.mean(x);
    const meanY = this.mean(y);
    
    let numerator = 0;
    let denomX = 0;
    let denomY = 0;
    
    for (let i = 0; i < x.length; i++) {
      const xDiff = x[i] - meanX;
      const yDiff = y[i] - meanY;
      numerator += xDiff * yDiff;
      denomX += xDiff * xDiff;
      denomY += yDiff * yDiff;
    }
    
    const denominator = Math.sqrt(denomX * denomY);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  static linearRegression(x, y) {
    if (x.length !== y.length || x.length < 2) {
      return { slope: 0, intercept: 0, r2: 0 };
    }

    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
    const sumXX = x.reduce((acc, xi) => acc + xi * xi, 0);
    const sumYY = y.reduce((acc, yi) => acc + yi * yi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // Calculate R-squared
    const yMean = sumY / n;
    const ssReg = x.reduce((acc, xi, i) => {
      const predicted = slope * xi + intercept;
      return acc + Math.pow(predicted - yMean, 2);
    }, 0);
    const ssTot = y.reduce((acc, yi) => acc + Math.pow(yi - yMean, 2), 0);
    const r2 = ssTot === 0 ? 1 : ssReg / ssTot;

    return { slope, intercept, r2 };
  }

  static exponentialSmoothing(values, alpha = 0.3) {
    if (values.length === 0) return [];
    
    const smoothed = [values[0]];
    for (let i = 1; i < values.length; i++) {
      smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1];
    }
    return smoothed;
  }

  static movingAverage(values, window) {
    if (values.length < window) return values.slice();
    
    const result = [];
    for (let i = 0; i <= values.length - window; i++) {
      const windowValues = values.slice(i, i + window);
      result.push(this.mean(windowValues));
    }
    return result;
  }
}

/**
 * Trend Detection Engine
 * Identifies patterns and trends in time series data
 */
class TrendDetector {
  constructor(options = {}) {
    this.options = {
      minDataPoints: 10,
      trendThreshold: 0.05, // Minimum slope for trend detection
      significanceLevel: 0.05, // Statistical significance threshold
      seasonalityWindow: 24, // Hours for seasonal pattern detection
      changePointSensitivity: 2.0, // Standard deviations for change point detection
      ...options
    };

    this.trendHistory = [];
    this.seasonalPatterns = new Map();
    this.changePoints = [];
  }

  analyzeTrend(data, timestamps = null) {
    if (data.length < this.options.minDataPoints) {
      return {
        type: 'insufficient_data',
        confidence: 0,
        message: 'Not enough data points for trend analysis'
      };
    }

    // Create time indices if timestamps not provided
    const timeIndices = timestamps ? 
      timestamps.map(t => new Date(t).getTime()) : 
      data.map((_, i) => i);

    // Perform linear regression
    const regression = StatisticalAnalysis.linearRegression(timeIndices, data);
    
    // Determine trend type and confidence
    const trendAnalysis = this.classifyTrend(regression, data);
    
    // Detect seasonal patterns
    const seasonality = this.detectSeasonality(data, timestamps);
    
    // Detect change points
    const changePoints = this.detectChangePoints(data);
    
    // Calculate overall trend strength
    const trendStrength = this.calculateTrendStrength(data, regression);

    const result = {
      ...trendAnalysis,
      regression,
      seasonality,
      changePoints,
      trendStrength,
      dataPoints: data.length,
      timespan: timestamps ? {
        start: new Date(Math.min(...timestamps)),
        end: new Date(Math.max(...timestamps)),
        duration: Math.max(...timestamps) - Math.min(...timestamps)
      } : null
    };

    // Store in history
    this.trendHistory.push({
      timestamp: Date.now(),
      analysis: result
    });

    return result;
  }

  classifyTrend(regression, data) {
    const { slope, r2 } = regression;
    const absSlope = Math.abs(slope);
    const dataRange = Math.max(...data) - Math.min(...data);
    const normalizedSlope = dataRange > 0 ? absSlope / dataRange : 0;

    let type, confidence, direction, strength;

    // Determine direction
    if (Math.abs(slope) < this.options.trendThreshold) {
      direction = 'stable';
      type = 'stable';
    } else if (slope > 0) {
      direction = 'increasing';
      type = 'upward';
    } else {
      direction = 'decreasing';
      type = 'downward';
    }

    // Determine strength
    if (normalizedSlope < 0.01) {
      strength = 'weak';
    } else if (normalizedSlope < 0.05) {
      strength = 'moderate';
    } else {
      strength = 'strong';
    }

    // Calculate confidence based on R-squared and data consistency
    confidence = Math.min(r2 * 100, 95); // Cap at 95%

    return {
      type,
      direction,
      strength,
      confidence: Math.round(confidence),
      slope,
      normalizedSlope,
      message: this.generateTrendMessage(type, strength, confidence)
    };
  }

  detectSeasonality(data, timestamps) {
    if (!timestamps || data.length < this.options.seasonalityWindow * 2) {
      return { detected: false, message: 'Insufficient data for seasonality detection' };
    }

    // Group data by hour of day
    const hourlyPatterns = {};
    timestamps.forEach((timestamp, index) => {
      const hour = new Date(timestamp).getHours();
      if (!hourlyPatterns[hour]) hourlyPatterns[hour] = [];
      hourlyPatterns[hour].push(data[index]);
    });

    // Calculate average for each hour
    const hourlyAverages = {};
    Object.keys(hourlyPatterns).forEach(hour => {
      hourlyAverages[hour] = StatisticalAnalysis.mean(hourlyPatterns[hour]);
    });

    // Check for significant variation across hours
    const averageValues = Object.values(hourlyAverages);
    const overallMean = StatisticalAnalysis.mean(averageValues);
    const variance = StatisticalAnalysis.standardDeviation(averageValues);
    const coefficientOfVariation = variance / Math.abs(overallMean);

    const seasonalityDetected = coefficientOfVariation > 0.1; // 10% variation threshold

    return {
      detected: seasonalityDetected,
      patterns: hourlyAverages,
      variation: coefficientOfVariation,
      peak: seasonalityDetected ? this.findPeakHour(hourlyAverages) : null,
      message: seasonalityDetected ? 
        `Seasonal pattern detected with ${(coefficientOfVariation * 100).toFixed(1)}% variation` :
        'No significant seasonal pattern detected'
    };
  }

  findPeakHour(hourlyAverages) {
    let maxHour = null;
    let maxValue = -Infinity;
    
    Object.entries(hourlyAverages).forEach(([hour, value]) => {
      if (value > maxValue) {
        maxValue = value;
        maxHour = parseInt(hour);
      }
    });

    return { hour: maxHour, value: maxValue };
  }

  detectChangePoints(data) {
    if (data.length < 10) return [];

    const changePoints = [];
    const windowSize = Math.max(3, Math.floor(data.length / 10));
    
    for (let i = windowSize; i < data.length - windowSize; i++) {
      const leftWindow = data.slice(i - windowSize, i);
      const rightWindow = data.slice(i, i + windowSize);
      
      const leftMean = StatisticalAnalysis.mean(leftWindow);
      const rightMean = StatisticalAnalysis.mean(rightWindow);
      const leftStd = StatisticalAnalysis.standardDeviation(leftWindow);
      const rightStd = StatisticalAnalysis.standardDeviation(rightWindow);
      
      // Calculate change magnitude
      const meanDifference = Math.abs(rightMean - leftMean);
      const pooledStd = Math.sqrt((leftStd * leftStd + rightStd * rightStd) / 2);
      
      if (pooledStd > 0) {
        const changeMagnitude = meanDifference / pooledStd;
        
        if (changeMagnitude > this.options.changePointSensitivity) {
          changePoints.push({
            index: i,
            magnitude: changeMagnitude,
            before: leftMean,
            after: rightMean,
            change: rightMean - leftMean,
            changePercent: leftMean !== 0 ? ((rightMean - leftMean) / leftMean) * 100 : 0
          });
        }
      }
    }

    return changePoints;
  }

  calculateTrendStrength(data, regression) {
    const { r2 } = regression;
    const variance = StatisticalAnalysis.standardDeviation(data);
    const mean = StatisticalAnalysis.mean(data);
    const coefficientOfVariation = Math.abs(mean) > 0 ? variance / Math.abs(mean) : 0;

    return {
      r_squared: r2,
      coefficient_of_variation: coefficientOfVariation,
      strength_score: r2 / (1 + coefficientOfVariation), // Combined metric
      interpretation: this.interpretTrendStrength(r2, coefficientOfVariation)
    };
  }

  interpretTrendStrength(r2, cv) {
    if (r2 > 0.8 && cv < 0.2) return 'Very Strong';
    if (r2 > 0.6 && cv < 0.4) return 'Strong';
    if (r2 > 0.4 && cv < 0.6) return 'Moderate';
    if (r2 > 0.2) return 'Weak';
    return 'Very Weak';
  }

  generateTrendMessage(type, strength, confidence) {
    const strengthAdj = strength === 'weak' ? 'slight' : strength;
    return `${strengthAdj} ${type} trend detected with ${confidence}% confidence`;
  }

  getTrendHistory(limit = 10) {
    return this.trendHistory.slice(-limit);
  }

  clearHistory() {
    this.trendHistory = [];
    this.seasonalPatterns.clear();
    this.changePoints = [];
  }
}

/**
 * Alert System
 * Smart alerting based on statistical analysis and trend detection
 */
class SmartAlertSystem {
  constructor(options = {}) {
    this.options = {
      alertThresholds: {
        critical: { zScore: 3.0, trendChange: 50 },
        high: { zScore: 2.5, trendChange: 30 },
        medium: { zScore: 2.0, trendChange: 20 },
        low: { zScore: 1.5, trendChange: 10 }
      },
      cooldownPeriod: 5 * 60 * 1000, // 5 minutes
      maxAlertsPerHour: 10,
      enableAdaptiveThresholds: true,
      ...options
    };

    this.alerts = [];
    this.alertHistory = [];
    this.suppressedAlerts = new Set();
    this.adaptiveThresholds = new Map();
  }

  analyzeForAlerts(data, metadata = {}) {
    const timestamp = Date.now();
    const alerts = [];

    // Statistical outlier detection
    const outlierAlerts = this.detectOutlierAlerts(data, metadata);
    alerts.push(...outlierAlerts);

    // Trend-based alerts
    if (data.length >= 10) {
      const trendAlerts = this.detectTrendAlerts(data, metadata);
      alerts.push(...trendAlerts);
    }

    // Volume-based alerts
    const volumeAlerts = this.detectVolumeAlerts(data, metadata);
    alerts.push(...volumeAlerts);

    // Rate of change alerts
    const changeAlerts = this.detectChangeRateAlerts(data, metadata);
    alerts.push(...changeAlerts);

    // Filter and process alerts
    const processedAlerts = this.processAlerts(alerts, timestamp);

    // Store in history
    this.alertHistory.push({
      timestamp,
      alerts: processedAlerts,
      dataPoints: data.length,
      metadata
    });

    return processedAlerts;
  }

  detectOutlierAlerts(data, metadata) {
    if (data.length < 3) return [];

    const alerts = [];
    const mean = StatisticalAnalysis.mean(data);
    const stdDev = StatisticalAnalysis.standardDeviation(data);

    // Check most recent values
    const recentValues = data.slice(-5);
    
    recentValues.forEach((value, index) => {
      const zScore = StatisticalAnalysis.zScore(value, mean, stdDev);
      const severity = this.classifyZScoreSeverity(Math.abs(zScore));

      if (severity) {
        alerts.push({
          type: 'statistical_outlier',
          severity,
          value,
          zScore,
          mean,
          stdDev,
          message: `Statistical outlier detected: value ${value.toFixed(3)} is ${Math.abs(zScore).toFixed(2)} standard deviations from mean`,
          metadata: { index: data.length - recentValues.length + index, ...metadata }
        });
      }
    });

    return alerts;
  }

  detectTrendAlerts(data, metadata) {
    const trendDetector = new TrendDetector();
    const trendAnalysis = trendDetector.analyzeTrend(data);
    const alerts = [];

    // Alert on significant trend changes
    if (trendAnalysis.confidence > 70) {
      if (trendAnalysis.strength === 'strong') {
        alerts.push({
          type: 'trend_change',
          severity: trendAnalysis.type === 'stable' ? 'low' : 'medium',
          trend: trendAnalysis,
          message: `Strong ${trendAnalysis.direction} trend detected with ${trendAnalysis.confidence}% confidence`,
          metadata
        });
      }

      // Alert on change points
      if (trendAnalysis.changePoints && trendAnalysis.changePoints.length > 0) {
        const significantChanges = trendAnalysis.changePoints.filter(cp => Math.abs(cp.changePercent) > 20);
        
        significantChanges.forEach(changePoint => {
          alerts.push({
            type: 'change_point',
            severity: Math.abs(changePoint.changePercent) > 50 ? 'high' : 'medium',
            changePoint,
            message: `Significant change detected: ${changePoint.changePercent.toFixed(1)}% change at data point ${changePoint.index}`,
            metadata
          });
        });
      }
    }

    return alerts;
  }

  detectVolumeAlerts(data, metadata) {
    const alerts = [];
    const recentWindow = Math.min(20, Math.floor(data.length * 0.2));
    
    if (data.length < recentWindow * 2) return alerts;

    const recentData = data.slice(-recentWindow);
    const historicalData = data.slice(0, -recentWindow);
    
    const recentMean = StatisticalAnalysis.mean(recentData);
    const historicalMean = StatisticalAnalysis.mean(historicalData);
    
    const changePercent = historicalMean !== 0 ? 
      ((recentMean - historicalMean) / historicalMean) * 100 : 0;

    if (Math.abs(changePercent) > 25) {
      alerts.push({
        type: 'volume_change',
        severity: Math.abs(changePercent) > 50 ? 'high' : 'medium',
        changePercent,
        recentMean,
        historicalMean,
        message: `Significant volume change: ${changePercent > 0 ? 'increase' : 'decrease'} of ${Math.abs(changePercent).toFixed(1)}%`,
        metadata
      });
    }

    return alerts;
  }

  detectChangeRateAlerts(data, metadata) {
    if (data.length < 5) return [];

    const alerts = [];
    const recentChanges = [];
    
    // Calculate rate of change for recent points
    for (let i = data.length - 4; i < data.length; i++) {
      if (i > 0) {
        const changeRate = Math.abs((data[i] - data[i - 1]) / data[i - 1]) * 100;
        recentChanges.push(changeRate);
      }
    }

    const avgChangeRate = StatisticalAnalysis.mean(recentChanges);
    const maxChangeRate = Math.max(...recentChanges);

    if (maxChangeRate > 20) {
      alerts.push({
        type: 'rapid_change',
        severity: maxChangeRate > 50 ? 'critical' : maxChangeRate > 30 ? 'high' : 'medium',
        maxChangeRate,
        avgChangeRate,
        message: `Rapid change detected: maximum ${maxChangeRate.toFixed(1)}% change between consecutive points`,
        metadata
      });
    }

    return alerts;
  }

  classifyZScoreSeverity(absZScore) {
    if (absZScore >= this.options.alertThresholds.critical.zScore) return 'critical';
    if (absZScore >= this.options.alertThresholds.high.zScore) return 'high';
    if (absZScore >= this.options.alertThresholds.medium.zScore) return 'medium';
    if (absZScore >= this.options.alertThresholds.low.zScore) return 'low';
    return null;
  }

  processAlerts(alerts, timestamp) {
    // Filter out suppressed alerts
    const activeAlerts = alerts.filter(alert => !this.isAlertSuppressed(alert));

    // Apply rate limiting
    const rateLimitedAlerts = this.applyRateLimit(activeAlerts, timestamp);

    // Add timestamp and IDs
    const processedAlerts = rateLimitedAlerts.map(alert => ({
      ...alert,
      id: this.generateAlertId(),
      timestamp,
      acknowledged: false
    }));

    // Store active alerts
    this.alerts.push(...processedAlerts);

    // Clean up old alerts
    this.cleanupOldAlerts();

    return processedAlerts;
  }

  isAlertSuppressed(alert) {
    const alertKey = `${alert.type}_${alert.severity}`;
    const suppressEntry = this.suppressedAlerts.get(alertKey);
    
    if (!suppressEntry) return false;
    
    // Check if cooldown period has passed
    if (Date.now() - suppressEntry.timestamp > this.options.cooldownPeriod) {
      this.suppressedAlerts.delete(alertKey);
      return false;
    }
    
    return true;
  }

  applyRateLimit(alerts, timestamp) {
    const hourAgo = timestamp - 60 * 60 * 1000;
    const recentAlerts = this.alertHistory.filter(entry => entry.timestamp > hourAgo);
    const recentAlertCount = recentAlerts.reduce((count, entry) => count + entry.alerts.length, 0);

    if (recentAlertCount >= this.options.maxAlertsPerHour) {
      // Only allow critical alerts when rate limited
      return alerts.filter(alert => alert.severity === 'critical');
    }

    return alerts;
  }

  acknowledgeAlert(alertId) {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      alert.acknowledgedAt = Date.now();
      
      // Add to suppressed alerts to prevent duplicates
      const alertKey = `${alert.type}_${alert.severity}`;
      this.suppressedAlerts.set(alertKey, {
        timestamp: Date.now(),
        alertId
      });
      
      return true;
    }
    return false;
  }

  getActiveAlerts(severityFilter = null) {
    let alerts = this.alerts.filter(alert => !alert.acknowledged);
    
    if (severityFilter) {
      alerts = alerts.filter(alert => alert.severity === severityFilter);
    }
    
    return alerts.sort((a, b) => {
      const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    });
  }

  getAlertHistory(limit = 100) {
    return this.alertHistory.slice(-limit);
  }

  generateAlertId() {
    return `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  cleanupOldAlerts() {
    const cutoffTime = Date.now() - 24 * 60 * 60 * 1000; // 24 hours
    this.alerts = this.alerts.filter(alert => alert.timestamp > cutoffTime);
    this.alertHistory = this.alertHistory.filter(entry => entry.timestamp > cutoffTime);
  }

  getAlertStatistics() {
    const activeAlerts = this.getActiveAlerts();
    const severityCounts = { critical: 0, high: 0, medium: 0, low: 0 };
    
    activeAlerts.forEach(alert => {
      severityCounts[alert.severity]++;
    });

    return {
      total: activeAlerts.length,
      by_severity: severityCounts,
      acknowledged: this.alerts.filter(a => a.acknowledged).length,
      suppressed: this.suppressedAlerts.size,
      recent_history: this.alertHistory.slice(-10)
    };
  }
}

/**
 * Advanced Analytics Engine
 * Main orchestrator for real-time analytics and alerting
 */
class AdvancedAnalyticsEngine {
  constructor(options = {}) {
    this.options = {
      analysisInterval: 30000, // 30 seconds
      trendAnalysisWindow: 100, // Number of data points
      enableRealTimeAnalysis: true,
      enableAlerts: true,
      enablePrediction: true,
      ...options
    };

    // Core components
    this.trendDetector = new TrendDetector();
    this.alertSystem = new SmartAlertSystem();
    
    // Data storage
    this.dataStreams = new Map();
    this.analysisResults = new Map();
    
    // Timers
    this.analysisTimer = null;
    
    // Event system
    this.eventListeners = new Map();
    
    this.init();
  }

  init() {
    if (this.options.enableRealTimeAnalysis) {
      this.startRealTimeAnalysis();
    }
  }

  /**
   * Data Stream Management
   */
  registerDataStream(streamId, options = {}) {
    this.dataStreams.set(streamId, {
      data: [],
      timestamps: [],
      maxSize: options.maxSize || 1000,
      metadata: options.metadata || {},
      lastAnalysis: 0,
      analysisInterval: options.analysisInterval || this.options.analysisInterval
    });
  }

  addDataPoint(streamId, value, timestamp = null) {
    const stream = this.dataStreams.get(streamId);
    if (!stream) {
      console.warn(`Data stream ${streamId} not registered`);
      return;
    }

    const ts = timestamp || Date.now();
    stream.data.push(value);
    stream.timestamps.push(ts);

    // Maintain max size
    if (stream.data.length > stream.maxSize) {
      stream.data.shift();
      stream.timestamps.shift();
    }

    // Trigger analysis if needed
    if (ts - stream.lastAnalysis > stream.analysisInterval) {
      this.analyzeStream(streamId);
    }
  }

  addBatchData(streamId, values, timestamps = null) {
    values.forEach((value, index) => {
      const timestamp = timestamps ? timestamps[index] : null;
      this.addDataPoint(streamId, value, timestamp);
    });
  }

  /**
   * Analysis Execution
   */
  analyzeStream(streamId) {
    const stream = this.dataStreams.get(streamId);
    if (!stream || stream.data.length === 0) return null;

    const analysisStartTime = performance.now();

    // Trend analysis
    const trendAnalysis = this.trendDetector.analyzeTrend(stream.data, stream.timestamps);
    
    // Statistical summary
    const statistics = this.calculateStatistics(stream.data);
    
    // Alert analysis
    const alerts = this.options.enableAlerts ? 
      this.alertSystem.analyzeForAlerts(stream.data, { streamId, ...stream.metadata }) : [];

    // Prediction (if enabled)
    const prediction = this.options.enablePrediction ? 
      this.generatePrediction(stream.data, stream.timestamps) : null;

    const analysisResult = {
      streamId,
      timestamp: Date.now(),
      dataPoints: stream.data.length,
      timespan: stream.timestamps.length > 1 ? {
        start: Math.min(...stream.timestamps),
        end: Math.max(...stream.timestamps),
        duration: Math.max(...stream.timestamps) - Math.min(...stream.timestamps)
      } : null,
      trend: trendAnalysis,
      statistics,
      alerts,
      prediction,
      analysisTime: performance.now() - analysisStartTime
    };

    // Store results
    this.analysisResults.set(streamId, analysisResult);
    stream.lastAnalysis = Date.now();

    // Emit events
    this.emit('analysis_complete', { streamId, results: analysisResult });
    
    if (alerts.length > 0) {
      this.emit('alerts_generated', { streamId, alerts });
    }

    return analysisResult;
  }

  analyzeAllStreams() {
    const results = {};
    
    for (const streamId of this.dataStreams.keys()) {
      results[streamId] = this.analyzeStream(streamId);
    }

    return results;
  }

  calculateStatistics(data) {
    if (data.length === 0) return null;

    return {
      count: data.length,
      mean: StatisticalAnalysis.mean(data),
      median: StatisticalAnalysis.median(data),
      std_dev: StatisticalAnalysis.standardDeviation(data),
      min: Math.min(...data),
      max: Math.max(...data),
      range: Math.max(...data) - Math.min(...data),
      percentiles: {
        p25: StatisticalAnalysis.percentile(data, 25),
        p50: StatisticalAnalysis.percentile(data, 50),
        p75: StatisticalAnalysis.percentile(data, 75),
        p90: StatisticalAnalysis.percentile(data, 90),
        p95: StatisticalAnalysis.percentile(data, 95),
        p99: StatisticalAnalysis.percentile(data, 99)
      },
      recent_stats: data.length >= 10 ? {
        recent_mean: StatisticalAnalysis.mean(data.slice(-10)),
        recent_std: StatisticalAnalysis.standardDeviation(data.slice(-10))
      } : null
    };
  }

  generatePrediction(data, timestamps = null) {
    if (data.length < 10) return null;

    try {
      // Simple linear prediction for next few points
      const timeIndices = timestamps ? 
        timestamps.map(t => new Date(t).getTime()) : 
        data.map((_, i) => i);

      const regression = StatisticalAnalysis.linearRegression(timeIndices, data);
      
      if (regression.r2 < 0.1) {
        return {
          method: 'linear_regression',
          confidence: 'low',
          message: 'Low predictive confidence due to high variance'
        };
      }

      // Predict next 5 points
      const lastTime = Math.max(...timeIndices);
      const timeStep = timeIndices.length > 1 ? 
        (timeIndices[timeIndices.length - 1] - timeIndices[timeIndices.length - 2]) : 1;

      const predictions = [];
      for (let i = 1; i <= 5; i++) {
        const futureTime = lastTime + (timeStep * i);
        const predictedValue = regression.slope * futureTime + regression.intercept;
        predictions.push({
          time: futureTime,
          value: predictedValue,
          confidence: regression.r2
        });
      }

      return {
        method: 'linear_regression',
        predictions,
        confidence: regression.r2 > 0.7 ? 'high' : regression.r2 > 0.4 ? 'medium' : 'low',
        regression_stats: regression
      };
    } catch (error) {
      return {
        method: 'linear_regression',
        error: error.message,
        confidence: 'none'
      };
    }
  }

  /**
   * Real-time Analysis
   */
  startRealTimeAnalysis() {
    if (this.analysisTimer) return;

    this.analysisTimer = setInterval(() => {
      const results = this.analyzeAllStreams();
      this.emit('realtime_analysis', { results, timestamp: Date.now() });
    }, this.options.analysisInterval);
  }

  stopRealTimeAnalysis() {
    if (this.analysisTimer) {
      clearInterval(this.analysisTimer);
      this.analysisTimer = null;
    }
  }

  /**
   * Results and Alerts
   */
  getStreamAnalysis(streamId) {
    return this.analysisResults.get(streamId);
  }

  getAllAnalysisResults() {
    const results = {};
    for (const [streamId, result] of this.analysisResults) {
      results[streamId] = result;
    }
    return results;
  }

  getActiveAlerts(severityFilter = null) {
    return this.alertSystem.getActiveAlerts(severityFilter);
  }

  acknowledgeAlert(alertId) {
    return this.alertSystem.acknowledgeAlert(alertId);
  }

  getAlertStatistics() {
    return this.alertSystem.getAlertStatistics();
  }

  /**
   * Event System
   */
  on(event, listener) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event).add(listener);
    return () => this.off(event, listener);
  }

  off(event, listener) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).delete(listener);
    }
  }

  emit(event, data) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error('Analytics event listener error:', error);
        }
      });
    }
  }

  /**
   * Utilities and Reports
   */
  generateAnalyticsReport() {
    const report = {
      timestamp: Date.now(),
      summary: {
        total_streams: this.dataStreams.size,
        active_alerts: this.getActiveAlerts().length,
        analysis_results: this.analysisResults.size
      },
      streams: {},
      alerts: this.getAlertStatistics(),
      system_health: this.getSystemHealth()
    };

    // Add stream summaries
    for (const [streamId, stream] of this.dataStreams) {
      const analysis = this.analysisResults.get(streamId);
      report.streams[streamId] = {
        data_points: stream.data.length,
        last_analysis: stream.lastAnalysis,
        trend: analysis ? analysis.trend.type : 'unknown',
        alerts: analysis ? analysis.alerts.length : 0
      };
    }

    return report;
  }

  getSystemHealth() {
    const activeStreams = Array.from(this.dataStreams.values()).filter(s => s.data.length > 0);
    const recentAnalyses = Array.from(this.analysisResults.values())
      .filter(r => Date.now() - r.timestamp < 5 * 60 * 1000); // Last 5 minutes

    return {
      status: activeStreams.length > 0 ? 'active' : 'idle',
      active_streams: activeStreams.length,
      recent_analyses: recentAnalyses.length,
      avg_analysis_time: recentAnalyses.length > 0 ? 
        recentAnalyses.reduce((sum, r) => sum + r.analysisTime, 0) / recentAnalyses.length : 0,
      memory_usage: this.getMemoryUsage()
    };
  }

  getMemoryUsage() {
    const totalDataPoints = Array.from(this.dataStreams.values())
      .reduce((sum, stream) => sum + stream.data.length, 0);
    
    return {
      total_data_points: totalDataPoints,
      estimated_memory_mb: (totalDataPoints * 16) / 1024 / 1024, // Rough estimate
      streams: this.dataStreams.size,
      results: this.analysisResults.size
    };
  }

  /**
   * Cleanup
   */
  destroy() {
    this.stopRealTimeAnalysis();
    this.dataStreams.clear();
    this.analysisResults.clear();
    this.eventListeners.clear();
  }
}

// Export classes
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    AdvancedAnalyticsEngine,
    TrendDetector,
    SmartAlertSystem,
    StatisticalAnalysis
  };
} else {
  // Browser environment
  window.AdvancedAnalyticsEngine = AdvancedAnalyticsEngine;
  window.TrendDetector = TrendDetector;
  window.SmartAlertSystem = SmartAlertSystem;
  window.StatisticalAnalysis = StatisticalAnalysis;
}