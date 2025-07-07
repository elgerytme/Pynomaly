import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
export const errorRate = new Rate('errors');
export const responseTimeTrend = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 }, // Ramp up
    { duration: '5m', target: 50 }, // Stay at 50 users
    { duration: '2m', target: 100 }, // Ramp to 100 users
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 0 }, // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000'], // 95% of requests under 1s
    http_req_failed: ['rate<0.1'], // Error rate under 10%
    errors: ['rate<0.1'],
  },
};

const BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8000';

// Test data
const testData = {
  normal: [1.0, 2.0, 1.5, 2.2, 1.8, 2.1, 1.9, 2.0, 1.7, 2.3],
  anomalous: [1.0, 2.0, 1.5, 10.0, 1.8, 2.1, 1.9, 2.0, 1.7, 2.3],
};

export default function () {
  group('Health Checks', () => {
    const healthRes = http.get(`${BASE_URL}/api/v1/health`);
    check(healthRes, {
      'health check status is 200': (r) => r.status === 200,
      'health check response time < 500ms': (r) => r.timings.duration < 500,
    });
    
    const readyRes = http.get(`${BASE_URL}/api/v1/health/ready`);
    check(readyRes, {
      'ready check status is 200': (r) => r.status === 200,
    });
    
    errorRate.add(healthRes.status !== 200);
    responseTimeTrend.add(healthRes.timings.duration);
  });

  group('API Documentation', () => {
    const docsRes = http.get(`${BASE_URL}/docs`);
    check(docsRes, {
      'docs accessible': (r) => r.status === 200,
    });
    
    errorRate.add(docsRes.status !== 200);
  });

  group('Anomaly Detection API', () => {
    // Test normal data detection
    const normalPayload = JSON.stringify({
      data: testData.normal,
      algorithm: 'isolation_forest',
      contamination: 0.1
    });
    
    const normalRes = http.post(`${BASE_URL}/api/v1/detect`, normalPayload, {
      headers: { 'Content-Type': 'application/json' },
    });
    
    check(normalRes, {
      'normal detection status is 200': (r) => r.status === 200,
      'normal detection response time < 2s': (r) => r.timings.duration < 2000,
      'normal detection returns results': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.anomalies && Array.isArray(body.anomalies);
        } catch (e) {
          return false;
        }
      },
    });
    
    errorRate.add(normalRes.status !== 200);
    responseTimeTrend.add(normalRes.timings.duration);
    
    // Test anomalous data detection
    const anomalousPayload = JSON.stringify({
      data: testData.anomalous,
      algorithm: 'isolation_forest',
      contamination: 0.1
    });
    
    const anomalousRes = http.post(`${BASE_URL}/api/v1/detect`, anomalousPayload, {
      headers: { 'Content-Type': 'application/json' },
    });
    
    check(anomalousRes, {
      'anomalous detection status is 200': (r) => r.status === 200,
      'anomalous detection finds anomalies': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.anomalies && body.anomalies.some(a => a.is_anomaly === true);
        } catch (e) {
          return false;
        }
      },
    });
    
    errorRate.add(anomalousRes.status !== 200);
  });

  group('Batch Processing', () => {
    const batchPayload = JSON.stringify({
      datasets: [
        { id: 'dataset1', data: testData.normal },
        { id: 'dataset2', data: testData.anomalous },
      ],
      algorithm: 'lof',
      contamination: 0.1
    });
    
    const batchRes = http.post(`${BASE_URL}/api/v1/detect/batch`, batchPayload, {
      headers: { 'Content-Type': 'application/json' },
    });
    
    check(batchRes, {
      'batch processing status is 200': (r) => r.status === 200,
      'batch processing response time < 5s': (r) => r.timings.duration < 5000,
    });
    
    errorRate.add(batchRes.status !== 200);
    responseTimeTrend.add(batchRes.timings.duration);
  });

  group('Model Management', () => {
    // List available algorithms
    const algorithmsRes = http.get(`${BASE_URL}/api/v1/algorithms`);
    check(algorithmsRes, {
      'algorithms list status is 200': (r) => r.status === 200,
      'algorithms list returns data': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body.algorithms);
        } catch (e) {
          return false;
        }
      },
    });
    
    errorRate.add(algorithmsRes.status !== 200);
  });

  group('Metrics Endpoint', () => {
    const metricsRes = http.get(`${BASE_URL}/metrics`);
    check(metricsRes, {
      'metrics endpoint accessible': (r) => r.status === 200,
      'metrics contain prometheus format': (r) => r.body.includes('# HELP'),
    });
    
    errorRate.add(metricsRes.status !== 200);
  });

  // Add some think time to simulate real user behavior
  sleep(Math.random() * 2 + 1);
}

export function handleSummary(data) {
  return {
    'performance-summary.json': JSON.stringify(data, null, 2),
    'performance-summary.html': htmlReport(data),
  };
}

function htmlReport(data) {
  return `<!DOCTYPE html>
<html>
<head>
    <title>Pynomaly Performance Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .metric { margin: 20px 0; }
        .metric h3 { color: #333; }
        .pass { color: green; }
        .fail { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Pynomaly Performance Test Results</h1>
    <h2>Summary</h2>
    <div class="metric">
        <h3>Test Duration</h3>
        <p>${Math.round(data.state.testRunDurationMs / 1000)}s</p>
    </div>
    <div class="metric">
        <h3>Total Requests</h3>
        <p>${data.metrics.http_reqs.count}</p>
    </div>
    <div class="metric">
        <h3>Error Rate</h3>
        <p class="${data.metrics.http_req_failed.rate < 0.1 ? 'pass' : 'fail'}">
            ${(data.metrics.http_req_failed.rate * 100).toFixed(2)}%
        </p>
    </div>
    <div class="metric">
        <h3>Average Response Time</h3>
        <p>${data.metrics.http_req_duration.avg.toFixed(2)}ms</p>
    </div>
    <div class="metric">
        <h3>95th Percentile Response Time</h3>
        <p class="${data.metrics.http_req_duration['p(95)'] < 1000 ? 'pass' : 'fail'}">
            ${data.metrics.http_req_duration['p(95)'].toFixed(2)}ms
        </p>
    </div>
    
    <h2>Detailed Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
        <tr>
            <td>Error Rate</td>
            <td>${(data.metrics.http_req_failed.rate * 100).toFixed(2)}%</td>
            <td>&lt; 10%</td>
            <td class="${data.metrics.http_req_failed.rate < 0.1 ? 'pass' : 'fail'}">
                ${data.metrics.http_req_failed.rate < 0.1 ? 'PASS' : 'FAIL'}
            </td>
        </tr>
        <tr>
            <td>95th Percentile Response Time</td>
            <td>${data.metrics.http_req_duration['p(95)'].toFixed(2)}ms</td>
            <td>&lt; 1000ms</td>
            <td class="${data.metrics.http_req_duration['p(95)'] < 1000 ? 'pass' : 'fail'}">
                ${data.metrics.http_req_duration['p(95)'] < 1000 ? 'PASS' : 'FAIL'}
            </td>
        </tr>
    </table>
</body>
</html>`;
}