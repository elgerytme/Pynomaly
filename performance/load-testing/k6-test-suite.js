/*
Comprehensive Load Testing Suite
K6-based performance testing for Domain-Driven Monorepo Platform
*/

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

// Custom metrics
export const errorRate = new Rate('errors');
export const responseTimeP95 = new Trend('response_time_p95');
export const requestCounter = new Counter('request_count');
export const authFailures = new Counter('auth_failures');

// Test configuration
export const options = {
  stages: [
    // Warm-up
    { duration: '2m', target: 10 },
    // Ramp-up
    { duration: '5m', target: 50 },
    // Stay at 50 users
    { duration: '10m', target: 50 },
    // Ramp-up to peak
    { duration: '5m', target: 100 },
    // Stay at peak
    { duration: '20m', target: 100 },
    // Spike test
    { duration: '2m', target: 200 },
    { duration: '3m', target: 200 },
    // Ramp-down
    { duration: '5m', target: 50 },
    { duration: '5m', target: 0 },
  ],
  thresholds: {
    // Error rate should be less than 1%
    'errors': ['rate<0.01'],
    // 95% of requests should be below 500ms
    'http_req_duration': ['p(95)<500'],
    // 99% of requests should be below 1000ms
    'http_req_duration': ['p(99)<1000'],
    // Average response time should be below 200ms
    'http_req_duration': ['avg<200'],
    // Failed requests should be less than 0.5%
    'http_req_failed': ['rate<0.005'],
    // Authentication failures should be 0
    'auth_failures': ['count==0'],
  },
  ext: {
    loadimpact: {
      distribution: {
        'amazon:us:ashburn': { loadZone: 'amazon:us:ashburn', percent: 50 },
        'amazon:ie:dublin': { loadZone: 'amazon:ie:dublin', percent: 25 },
        'amazon:sg:singapore': { loadZone: 'amazon:sg:singapore', percent: 25 },
      },
    },
  },
};

// Test data
const baseUrl = __ENV.BASE_URL || 'https://api.platform.com';
const authUrl = __ENV.AUTH_URL || 'https://auth.platform.com';

// User credentials for testing
const testUsers = [
  { username: 'testuser1@platform.com', password: 'TestPass123!', role: 'developer' },
  { username: 'testuser2@platform.com', password: 'TestPass123!', role: 'viewer' },
  { username: 'testuser3@platform.com', password: 'TestPass123!', role: 'admin' },
];

// Package test data
const testPackages = [
  { name: 'user-management', domain: 'identity', type: 'domain' },
  { name: 'order-processing', domain: 'commerce', type: 'domain' },
  { name: 'payment-gateway', domain: 'finance', type: 'domain' },
  { name: 'notification-service', domain: 'communication', type: 'infrastructure' },
  { name: 'audit-logging', domain: 'security', type: 'shared' },
];

// Authentication helper
function authenticate(user) {
  const loginPayload = {
    username: user.username,
    password: user.password,
  };

  const loginResponse = http.post(`${authUrl}/auth/login`, JSON.stringify(loginPayload), {
    headers: {
      'Content-Type': 'application/json',
    },
  });

  const loginSuccess = check(loginResponse, {
    'authentication successful': (resp) => resp.status === 200,
    'received access token': (resp) => resp.json('access_token') !== undefined,
  });

  if (!loginSuccess) {
    authFailures.add(1);
    return null;
  }

  return loginResponse.json('access_token');
}

// Helper function to get random item from array
function getRandomItem(array) {
  return array[Math.floor(Math.random() * array.length)];
}

// Main test function
export default function () {
  const user = getRandomItem(testUsers);
  const accessToken = authenticate(user);

  if (!accessToken) {
    errorRate.add(1);
    return;
  }

  const headers = {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'application/json',
  };

  // Test different scenarios based on user load distribution
  const scenario = Math.random();
  
  if (scenario < 0.4) {
    // 40% - Package discovery and browsing
    packageDiscoveryTest(headers);
  } else if (scenario < 0.7) {
    // 30% - Package development workflow
    packageDevelopmentTest(headers);
  } else if (scenario < 0.9) {
    // 20% - Analytics and monitoring
    analyticsTest(headers);
  } else {
    // 10% - Admin operations
    adminOperationsTest(headers, user.role);
  }

  sleep(Math.random() * 3 + 1); // Random sleep between 1-4 seconds
}

function packageDiscoveryTest(headers) {
  group('Package Discovery', () => {
    // Browse package catalog
    let response = http.get(`${baseUrl}/packages`, { headers });
    check(response, {
      'package list loaded': (resp) => resp.status === 200,
      'package list response time': (resp) => resp.timings.duration < 300,
    }) || errorRate.add(1);

    responseTimeP95.add(response.timings.duration);
    requestCounter.add(1);

    // Search packages
    const searchTerms = ['user', 'payment', 'notification', 'auth', 'order'];
    const searchTerm = getRandomItem(searchTerms);
    
    response = http.get(`${baseUrl}/packages/search?q=${searchTerm}`, { headers });
    check(response, {
      'search results returned': (resp) => resp.status === 200,
      'search response time': (resp) => resp.timings.duration < 500,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Get package details
    const packages = response.json('packages') || [];
    if (packages.length > 0) {
      const randomPackage = getRandomItem(packages);
      response = http.get(`${baseUrl}/packages/${randomPackage.name}`, { headers });
      check(response, {
        'package details loaded': (resp) => resp.status === 200,
        'package details response time': (resp) => resp.timings.duration < 200,
      }) || errorRate.add(1);

      requestCounter.add(1);

      // Get package metrics
      response = http.get(`${baseUrl}/packages/${randomPackage.name}/metrics`, { headers });
      check(response, {
        'package metrics loaded': (resp) => resp.status === 200,
      }) || errorRate.add(1);

      requestCounter.add(1);
    }
  });
}

function packageDevelopmentTest(headers) {
  group('Package Development', () => {
    const testPackage = getRandomItem(testPackages);
    
    // Create new package
    const packagePayload = {
      name: `test-package-${Date.now()}`,
      description: 'Load test package',
      domain: testPackage.domain,
      type: testPackage.type,
      version: '1.0.0',
    };

    let response = http.post(`${baseUrl}/packages`, JSON.stringify(packagePayload), { headers });
    const packageCreated = check(response, {
      'package created successfully': (resp) => resp.status === 201,
      'package creation response time': (resp) => resp.timings.duration < 1000,
    });

    if (!packageCreated) {
      errorRate.add(1);
      return;
    }

    requestCounter.add(1);
    const packageId = response.json('id');

    // Validate package
    response = http.post(`${baseUrl}/packages/${packageId}/validate`, null, { headers });
    check(response, {
      'package validation completed': (resp) => resp.status === 200,
      'validation response time': (resp) => resp.timings.duration < 2000,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Run tests
    response = http.post(`${baseUrl}/packages/${packageId}/test`, null, { headers });
    check(response, {
      'package tests executed': (resp) => resp.status === 200,
      'test execution response time': (resp) => resp.timings.duration < 5000,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Get test results
    response = http.get(`${baseUrl}/packages/${packageId}/test-results`, { headers });
    check(response, {
      'test results retrieved': (resp) => resp.status === 200,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Security scan
    response = http.post(`${baseUrl}/packages/${packageId}/security-scan`, null, { headers });
    check(response, {
      'security scan initiated': (resp) => resp.status === 200 || resp.status === 202,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Update package
    const updatePayload = {
      description: 'Updated load test package',
      version: '1.0.1',
    };

    response = http.put(`${baseUrl}/packages/${packageId}`, JSON.stringify(updatePayload), { headers });
    check(response, {
      'package updated successfully': (resp) => resp.status === 200,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Delete package (cleanup)
    response = http.del(`${baseUrl}/packages/${packageId}`, null, { headers });
    check(response, {
      'package deleted successfully': (resp) => resp.status === 204,
    }) || errorRate.add(1);

    requestCounter.add(1);
  });
}

function analyticsTest(headers) {
  group('Analytics & Monitoring', () => {
    // Get platform metrics
    let response = http.get(`${baseUrl}/analytics/platform-metrics`, { headers });
    check(response, {
      'platform metrics loaded': (resp) => resp.status === 200,
      'metrics response time': (resp) => resp.timings.duration < 1000,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Get usage statistics
    response = http.get(`${baseUrl}/analytics/usage-stats?period=7d`, { headers });
    check(response, {
      'usage stats loaded': (resp) => resp.status === 200,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Get performance metrics
    response = http.get(`${baseUrl}/analytics/performance?timerange=1h`, { headers });
    check(response, {
      'performance metrics loaded': (resp) => resp.status === 200,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Get security events
    response = http.get(`${baseUrl}/analytics/security-events?limit=100`, { headers });
    check(response, {
      'security events loaded': (resp) => resp.status === 200,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Export analytics data
    response = http.post(`${baseUrl}/analytics/export`, JSON.stringify({
      format: 'csv',
      metrics: ['usage', 'performance'],
      period: '24h'
    }), { headers });
    check(response, {
      'analytics export initiated': (resp) => resp.status === 200 || resp.status === 202,
    }) || errorRate.add(1);

    requestCounter.add(1);
  });
}

function adminOperationsTest(headers, userRole) {
  if (userRole !== 'admin') {
    return;
  }

  group('Admin Operations', () => {
    // Get system health
    let response = http.get(`${baseUrl}/admin/health`, { headers });
    check(response, {
      'system health retrieved': (resp) => resp.status === 200,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Get user management
    response = http.get(`${baseUrl}/admin/users?limit=50`, { headers });
    check(response, {
      'user list loaded': (resp) => resp.status === 200,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Get audit logs
    response = http.get(`${baseUrl}/admin/audit-logs?limit=100`, { headers });
    check(response, {
      'audit logs retrieved': (resp) => resp.status === 200,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // System configuration
    response = http.get(`${baseUrl}/admin/config`, { headers });
    check(response, {
      'system config loaded': (resp) => resp.status === 200,
    }) || errorRate.add(1);

    requestCounter.add(1);

    // Run maintenance task
    response = http.post(`${baseUrl}/admin/maintenance/cleanup`, null, { headers });
    check(response, {
      'maintenance task initiated': (resp) => resp.status === 200 || resp.status === 202,
    }) || errorRate.add(1);

    requestCounter.add(1);
  });
}

// Generate comprehensive test report
export function handleSummary(data) {
  return {
    'performance-report.html': htmlReport(data),
    'performance-summary.txt': textSummary(data, { indent: ' ', enableColors: true }),
    'performance-results.json': JSON.stringify(data, null, 2),
  };
}

// Test teardown
export function teardown(data) {
  console.log('Load test completed');
  console.log(`Total requests: ${data.metrics.request_count.values.count}`);
  console.log(`Error rate: ${(data.metrics.errors.values.rate * 100).toFixed(2)}%`);
  console.log(`Average response time: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms`);
  console.log(`95th percentile: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms`);
  console.log(`99th percentile: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms`);
}