import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTimeTrend = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 10 },  // Ramp up to 10 users
    { duration: '1m', target: 10 },   // Stay at 10 users
    { duration: '30s', target: 50 },  // Ramp up to 50 users
    { duration: '2m', target: 50 },   // Stay at 50 users
    { duration: '30s', target: 100 }, // Ramp up to 100 users
    { duration: '2m', target: 100 },  // Stay at 100 users
    { duration: '30s', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests must complete below 2s
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    errors: ['rate<0.1'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test data
const testDatasets = [
  {
    name: 'load_test_dataset_1',
    description: 'Dataset for load testing',
    data: generateSampleData(100)
  },
  {
    name: 'load_test_dataset_2',
    description: 'Another dataset for load testing',
    data: generateSampleData(200)
  }
];

function generateSampleData(rows) {
  const data = [];
  for (let i = 0; i < rows; i++) {
    data.push({
      feature1: Math.random() * 100,
      feature2: Math.random() * 50,
      feature3: Math.random() * 25,
      timestamp: new Date().toISOString()
    });
  }
  return data;
}

export function setup() {
  // Health check before starting tests
  const healthResponse = http.get(`${BASE_URL}/health`);
  check(healthResponse, {
    'health check passes': (r) => r.status === 200,
  });

  return { baseUrl: BASE_URL };
}

export default function(data) {
  group('API Health Checks', function() {
    const healthResponse = http.get(`${data.baseUrl}/health`);
    check(healthResponse, {
      'health endpoint responds': (r) => r.status === 200,
    }) || errorRate.add(1);

    const apiHealthResponse = http.get(`${data.baseUrl}/api/health`);
    check(apiHealthResponse, {
      'API health endpoint responds': (r) => r.status === 200,
    }) || errorRate.add(1);

    responseTimeTrend.add(healthResponse.timings.duration);
  });

  group('Dashboard Load', function() {
    const dashboardResponse = http.get(`${data.baseUrl}/`);
    check(dashboardResponse, {
      'dashboard loads successfully': (r) => r.status === 200,
      'dashboard contains expected content': (r) => r.body.includes('anomaly_detection'),
    }) || errorRate.add(1);

    responseTimeTrend.add(dashboardResponse.timings.duration);
  });

  group('API Endpoints', function() {
    // Test detector endpoints
    const detectorsResponse = http.get(`${data.baseUrl}/api/detectors`);
    check(detectorsResponse, {
      'detectors endpoint responds': (r) => r.status === 200,
    }) || errorRate.add(1);

    // Test datasets endpoints
    const datasetsResponse = http.get(`${data.baseUrl}/api/datasets`);
    check(datasetsResponse, {
      'datasets endpoint responds': (r) => r.status === 200,
    }) || errorRate.add(1);

    responseTimeTrend.add(detectorsResponse.timings.duration);
    responseTimeTrend.add(datasetsResponse.timings.duration);
  });

  group('Dataset Upload Performance', function() {
    const dataset = testDatasets[Math.floor(Math.random() * testDatasets.length)];

    const uploadResponse = http.post(`${data.baseUrl}/api/datasets/upload`, {
      name: `${dataset.name}_${__VU}_${__ITER}`,
      description: dataset.description,
      data: JSON.stringify(dataset.data)
    }, {
      headers: { 'Content-Type': 'application/json' },
    });

    check(uploadResponse, {
      'dataset upload succeeds or already exists': (r) => r.status === 200 || r.status === 201 || r.status === 409,
    }) || errorRate.add(1);

    responseTimeTrend.add(uploadResponse.timings.duration);
  });

  group('Detection Performance', function() {
    // Simulate running detection on existing data
    const detectionResponse = http.post(`${data.baseUrl}/api/detection/run`, {
      detector_id: 'default_isolation_forest',
      dataset_id: 'load_test_dataset',
      parameters: {
        contamination: 0.1,
        n_estimators: 100
      }
    }, {
      headers: { 'Content-Type': 'application/json' },
    });

    check(detectionResponse, {
      'detection request is accepted': (r) => r.status === 200 || r.status === 202,
    }) || errorRate.add(1);

    responseTimeTrend.add(detectionResponse.timings.duration);
  });

  group('Monitoring Endpoints', function() {
    const metricsResponse = http.get(`${data.baseUrl}/api/monitoring/metrics`);
    check(metricsResponse, {
      'metrics endpoint responds': (r) => r.status === 200,
    }) || errorRate.add(1);

    const systemInfoResponse = http.get(`${data.baseUrl}/api/monitoring/system`);
    check(systemInfoResponse, {
      'system info endpoint responds': (r) => r.status === 200,
    }) || errorRate.add(1);

    responseTimeTrend.add(metricsResponse.timings.duration);
    responseTimeTrend.add(systemInfoResponse.timings.duration);
  });

  // Random sleep between 1-3 seconds to simulate real user behavior
  sleep(Math.random() * 2 + 1);
}

export function teardown(data) {
  // Cleanup test data if needed
  console.log('Load test completed');

  // Final health check
  const finalHealthResponse = http.get(`${data.baseUrl}/health`);
  check(finalHealthResponse, {
    'final health check passes': (r) => r.status === 200,
  });
}
