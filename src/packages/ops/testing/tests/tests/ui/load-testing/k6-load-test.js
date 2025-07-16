import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

/**
 * K6 Load Testing Configuration for Pynomaly Web UI
 * Tests various user scenarios and performance characteristics
 */

// Custom metrics
const errorRate = new Rate('errors');
const successRate = new Rate('success');
const pageLoadTime = new Trend('page_load_time');
const apiResponseTime = new Trend('api_response_time');
const errorCount = new Counter('error_count');
const userActions = new Counter('user_actions');

// Test configuration
export let options = {
  stages: [
    // Ramp up
    { duration: '2m', target: 10 },   // Warm up
    { duration: '5m', target: 50 },   // Normal load
    { duration: '3m', target: 100 },  // Peak load
    { duration: '2m', target: 150 },  // Stress test
    { duration: '3m', target: 100 },  // Scale down
    { duration: '2m', target: 0 },    // Cool down
  ],
  
  thresholds: {
    // Performance thresholds
    http_req_duration: ['p(95)<2000', 'p(99)<5000'],
    http_req_failed: ['rate<0.01'],
    
    // Custom thresholds
    'page_load_time': ['p(95)<3000'],
    'api_response_time': ['p(95)<1000'],
    'error_rate': ['rate<0.05'],
    'success_rate': ['rate>0.95'],
  },
  
  // Browser configuration for UI testing
  browser: {
    type: 'chromium',
  },
  
  // Test scenarios
  scenarios: {
    // Basic page load test
    page_load: {
      executor: 'constant-vus',
      vus: 20,
      duration: '10m',
      exec: 'testPageLoad',
    },
    
    // API endpoint stress test
    api_stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 30 },
        { duration: '5m', target: 30 },
        { duration: '2m', target: 0 },
      ],
      exec: 'testAPIEndpoints',
    },
    
    // Real user journey simulation
    user_journey: {
      executor: 'shared-iterations',
      vus: 10,
      iterations: 100,
      exec: 'testUserJourney',
    },
    
    // WebSocket stress test
    websocket_stress: {
      executor: 'constant-vus',
      vus: 15,
      duration: '5m',
      exec: 'testWebSocketConnections',
    },
  },
};

// Base URL configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test data
const testData = {
  datasets: [
    { name: 'test_dataset_1', size: 1000 },
    { name: 'test_dataset_2', size: 5000 },
    { name: 'test_dataset_3', size: 10000 },
  ],
  detectors: [
    { name: 'isolation_forest', params: { contamination: 0.1 } },
    { name: 'one_class_svm', params: { nu: 0.1 } },
    { name: 'local_outlier_factor', params: { n_neighbors: 20 } },
  ],
};

/**
 * Test basic page loading performance
 */
export function testPageLoad() {
  const pages = [
    '/',
    '/detection',
    '/detectors',
    '/datasets',
    '/visualizations',
    '/dashboard',
  ];
  
  const page = pages[Math.floor(Math.random() * pages.length)];
  const startTime = Date.now();
  
  const response = http.get(`${BASE_URL}${page}`, {
    headers: {
      'User-Agent': 'k6-load-test',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    },
  });
  
  const loadTime = Date.now() - startTime;
  pageLoadTime.add(loadTime);
  
  const success = check(response, {
    'page loads successfully': (r) => r.status === 200,
    'page contains expected content': (r) => r.body.includes('Pynomaly'),
    'load time acceptable': () => loadTime < 5000,
  });
  
  if (success) {
    successRate.add(1);
    userActions.add(1);
  } else {
    errorRate.add(1);
    errorCount.add(1);
  }
  
  sleep(1);
}

/**
 * Test API endpoints under load
 */
export function testAPIEndpoints() {
  const endpoints = [
    { path: '/api/ui/config', method: 'GET' },
    { path: '/api/ui/health', method: 'GET' },
    { path: '/api/session/status', method: 'GET' },
    { path: '/api/detection/status', method: 'GET' },
    { path: '/api/detectors', method: 'GET' },
    { path: '/api/datasets', method: 'GET' },
  ];
  
  const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
  const startTime = Date.now();
  
  const response = http.request(endpoint.method, `${BASE_URL}${endpoint.path}`, null, {
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'k6-api-test',
    },
  });
  
  const responseTime = Date.now() - startTime;
  apiResponseTime.add(responseTime);
  
  const success = check(response, {
    'API responds successfully': (r) => r.status >= 200 && r.status < 300,
    'API response time acceptable': () => responseTime < 2000,
    'API returns JSON': (r) => r.headers['Content-Type'] && r.headers['Content-Type'].includes('application/json'),
  });
  
  if (success) {
    successRate.add(1);
  } else {
    errorRate.add(1);
    errorCount.add(1);
  }
  
  sleep(0.5);
}

/**
 * Test complete user journey workflow
 */
export function testUserJourney() {
  const journeyStart = Date.now();
  
  // Step 1: Load dashboard
  let response = http.get(`${BASE_URL}/dashboard`);
  check(response, {
    'dashboard loads': (r) => r.status === 200,
  });
  
  sleep(2);
  
  // Step 2: Navigate to detection page
  response = http.get(`${BASE_URL}/detection`);
  check(response, {
    'detection page loads': (r) => r.status === 200,
  });
  
  sleep(1);
  
  // Step 3: Load detectors list
  response = http.get(`${BASE_URL}/api/detectors`);
  check(response, {
    'detectors API responds': (r) => r.status === 200,
  });
  
  sleep(1);
  
  // Step 4: Load dataset information
  response = http.get(`${BASE_URL}/api/datasets`);
  check(response, {
    'datasets API responds': (r) => r.status === 200,
  });
  
  sleep(1);
  
  // Step 5: Check system health
  response = http.get(`${BASE_URL}/api/ui/health`);
  check(response, {
    'health check responds': (r) => r.status === 200,
  });
  
  const journeyTime = Date.now() - journeyStart;
  pageLoadTime.add(journeyTime);
  
  userActions.add(5); // 5 actions in journey
  
  sleep(3);
}

/**
 * Test WebSocket connections under load
 */
export function testWebSocketConnections() {
  // Note: k6 doesn't support WebSockets directly in the OSS version
  // This would test the WebSocket endpoint availability
  
  const response = http.get(`${BASE_URL}/api/websocket/status`);
  check(response, {
    'WebSocket endpoint available': (r) => r.status === 200,
  });
  
  // Test real-time API endpoints that support WebSocket upgrades
  const realtimeEndpoints = [
    '/api/realtime/detection',
    '/api/realtime/metrics',
    '/api/realtime/alerts',
  ];
  
  const endpoint = realtimeEndpoints[Math.floor(Math.random() * realtimeEndpoints.length)];
  const wsResponse = http.get(`${BASE_URL}${endpoint}`);
  
  check(wsResponse, {
    'real-time endpoint responds': (r) => r.status === 200,
  });
  
  sleep(1);
}

/**
 * Setup function - runs once before all tests
 */
export function setup() {
  console.log('Starting Pynomaly Web UI Load Test');
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`Test duration: ${options.stages.reduce((sum, stage) => sum + parseInt(stage.duration), 0)} minutes`);
  
  // Verify server is running
  const response = http.get(`${BASE_URL}/api/ui/health`);
  if (response.status !== 200) {
    throw new Error(`Server not available at ${BASE_URL}`);
  }
  
  return {
    startTime: Date.now(),
    baseUrl: BASE_URL,
  };
}

/**
 * Teardown function - runs once after all tests
 */
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Load test completed in ${duration} seconds`);
  
  // Generate summary report
  const report = {
    duration: duration,
    baseUrl: data.baseUrl,
    totalRequests: userActions.count,
    errors: errorCount.count,
    timestamp: new Date().toISOString(),
  };
  
  console.log('Test Summary:', JSON.stringify(report, null, 2));
}