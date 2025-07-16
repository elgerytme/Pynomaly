/**
 * Artillery Processor for Pynomaly Load Testing
 * Custom functions for data generation and request processing
 */

'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Test data generators
const detectorTypes = [
  'isolation_forest',
  'one_class_svm',
  'local_outlier_factor',
  'elliptic_envelope',
  'dbscan',
  'kmeans'
];

const datasetNames = [
  'financial_transactions',
  'network_traffic',
  'sensor_readings',
  'user_behavior',
  'system_logs',
  'performance_metrics'
];

const contaminationLevels = [0.01, 0.05, 0.1, 0.15, 0.2];

/**
 * Generate random test data for requests
 */
function generateTestData(context, events, done) {
  // Generate random dataset information
  context.vars.dataset_name = datasetNames[Math.floor(Math.random() * datasetNames.length)];
  context.vars.detector_type = detectorTypes[Math.floor(Math.random() * detectorTypes.length)];
  context.vars.contamination = contaminationLevels[Math.floor(Math.random() * contaminationLevels.length)];
  
  // Generate random user ID and session
  context.vars.user_id = `user_${Math.floor(Math.random() * 1000)}`;
  context.vars.session_id = crypto.randomUUID();
  
  // Generate random dataset size
  context.vars.dataset_size = Math.floor(Math.random() * 10000) + 100;
  
  // Generate timestamp
  context.vars.timestamp = Date.now();
  
  return done();
}

/**
 * Generate CSRF token
 */
function generateCSRFToken(context, events, done) {
  context.vars.csrf_token = crypto.randomBytes(32).toString('hex');
  return done();
}

/**
 * Generate realistic detection payload
 */
function generateDetectionPayload(context, events, done) {
  const payload = {
    dataset: {
      name: context.vars.dataset_name,
      size: context.vars.dataset_size,
      features: Math.floor(Math.random() * 20) + 5
    },
    detector: {
      type: context.vars.detector_type,
      parameters: {
        contamination: context.vars.contamination,
        random_state: 42
      }
    },
    options: {
      return_predictions: true,
      return_scores: true,
      return_confidence: Math.random() > 0.5
    },
    metadata: {
      user_id: context.vars.user_id,
      session_id: context.vars.session_id,
      timestamp: context.vars.timestamp
    }
  };
  
  context.vars.detection_payload = JSON.stringify(payload);
  return done();
}

/**
 * Generate realistic configuration update
 */
function generateConfigUpdate(context, events, done) {
  const config = {
    ui: {
      theme: Math.random() > 0.5 ? 'dark' : 'light',
      auto_refresh: Math.random() > 0.3,
      refresh_interval: Math.floor(Math.random() * 30) + 5
    },
    detection: {
      auto_analyze: Math.random() > 0.4,
      default_detector: context.vars.detector_type,
      default_contamination: context.vars.contamination
    },
    notifications: {
      enabled: Math.random() > 0.2,
      email: Math.random() > 0.5,
      browser: Math.random() > 0.7
    }
  };
  
  context.vars.config_update = JSON.stringify(config);
  return done();
}

/**
 * Validate response performance
 */
function validatePerformance(context, events, done) {
  const maxResponseTime = 5000; // 5 seconds
  const startTime = context.vars.request_start_time;
  const endTime = Date.now();
  const responseTime = endTime - startTime;
  
  if (responseTime > maxResponseTime) {
    console.warn(`Slow response detected: ${responseTime}ms (max: ${maxResponseTime}ms)`);
    
    // Log slow request details
    events.emit('counter', 'slow_requests', 1);
    events.emit('histogram', 'slow_request_time', responseTime);
  }
  
  return done();
}

/**
 * Track request start time
 */
function trackRequestStart(context, events, done) {
  context.vars.request_start_time = Date.now();
  return done();
}

/**
 * Log critical errors
 */
function logCriticalError(context, events, done) {
  const response = context.vars.$;
  
  if (response && response.statusCode >= 500) {
    const errorDetails = {
      url: context.vars.target + context.vars.url,
      statusCode: response.statusCode,
      timestamp: new Date().toISOString(),
      sessionId: context.vars.session_id
    };
    
    console.error('Critical error detected:', JSON.stringify(errorDetails));
    
    // Emit custom metrics
    events.emit('counter', 'critical_errors', 1);
    events.emit('counter', `error_${response.statusCode}`, 1);
  }
  
  return done();
}

/**
 * Generate load test summary
 */
function generateSummary(context, events, done) {
  const summary = {
    test_id: crypto.randomUUID(),
    start_time: new Date().toISOString(),
    target: context.vars.target,
    user_agent: 'Artillery Load Test',
    test_type: 'web_ui_load_test',
    expected_users: context.vars.expectedUsers || 'unknown',
    duration: context.vars.expectedDuration || 'unknown'
  };
  
  // Write summary to file
  const summaryPath = path.join(__dirname, '../../../test_reports/load-test-summary.json');
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
  
  context.vars.test_summary = summary;
  return done();
}

/**
 * Custom metrics collection
 */
function collectCustomMetrics(context, events, done) {
  // Memory usage simulation
  const memoryUsage = Math.floor(Math.random() * 1000) + 100; // MB
  events.emit('histogram', 'memory_usage_mb', memoryUsage);
  
  // CPU usage simulation
  const cpuUsage = Math.floor(Math.random() * 100); // Percentage
  events.emit('histogram', 'cpu_usage_percent', cpuUsage);
  
  // Active connections simulation
  const activeConnections = Math.floor(Math.random() * 500) + 10;
  events.emit('histogram', 'active_connections', activeConnections);
  
  // Database query time simulation
  const dbQueryTime = Math.floor(Math.random() * 100) + 5; // ms
  events.emit('histogram', 'db_query_time_ms', dbQueryTime);
  
  return done();
}

/**
 * Simulate realistic user behavior patterns
 */
function simulateUserBehavior(context, events, done) {
  const behaviors = [
    'quick_browser',    // Fast navigation, minimal interaction
    'thorough_analyst', // Detailed exploration of features
    'casual_user',      // Moderate interaction
    'power_user',       // Heavy feature usage
    'mobile_user'       // Touch-based interaction patterns
  ];
  
  const behavior = behaviors[Math.floor(Math.random() * behaviors.length)];
  context.vars.user_behavior = behavior;
  
  // Set behavior-specific delays
  switch (behavior) {
    case 'quick_browser':
      context.vars.think_time = Math.floor(Math.random() * 2) + 1; // 1-2 seconds
      break;
    case 'thorough_analyst':
      context.vars.think_time = Math.floor(Math.random() * 8) + 5; // 5-12 seconds
      break;
    case 'casual_user':
      context.vars.think_time = Math.floor(Math.random() * 5) + 2; // 2-6 seconds
      break;
    case 'power_user':
      context.vars.think_time = Math.floor(Math.random() * 3) + 1; // 1-3 seconds
      break;
    case 'mobile_user':
      context.vars.think_time = Math.floor(Math.random() * 4) + 2; // 2-5 seconds
      break;
  }
  
  return done();
}

/**
 * Test data cleanup
 */
function cleanup(context, events, done) {
  // Clean up any test data or connections
  delete context.vars.detection_payload;
  delete context.vars.config_update;
  delete context.vars.request_start_time;
  
  return done();
}

module.exports = {
  generateTestData,
  generateCSRFToken,
  generateDetectionPayload,
  generateConfigUpdate,
  validatePerformance,
  trackRequestStart,
  logCriticalError,
  generateSummary,
  collectCustomMetrics,
  simulateUserBehavior,
  cleanup
};