/**
 * Test utility functions and helpers
 */

import { PynomalyClient, PynomalyConfig, AuthToken, User } from '../../src/index';

// Mock data generators
export const createMockConfig = (overrides: Partial<PynomalyConfig> = {}): PynomalyConfig => ({
  apiKey: 'test-api-key',
  baseUrl: 'https://api.test.com',
  timeout: 5000,
  debug: false,
  ...overrides
});

export const createMockAuthToken = (overrides: Partial<AuthToken> = {}): AuthToken => ({
  token: 'mock-token-12345',
  refreshToken: 'mock-refresh-token-12345',
  expiresAt: new Date(Date.now() + 3600000), // 1 hour from now
  tokenType: 'Bearer',
  ...overrides
});

export const createMockUser = (overrides: Partial<User> = {}): User => ({
  id: 'user-123',
  email: 'test@example.com',
  name: 'Test User',
  roles: ['user'],
  permissions: ['read', 'write'],
  createdAt: new Date('2023-01-01'),
  lastLogin: new Date(),
  ...overrides
});

export const createMockAnomalyResult = (overrides: any = {}) => ({
  id: 'result-123',
  anomalies: [
    {
      index: 4,
      score: 0.85,
      isAnomaly: true,
      confidence: 0.92,
      explanation: 'Outlier detected',
      data: { value: 100 }
    }
  ],
  algorithm: 'isolation_forest',
  parameters: { contamination: 0.1 },
  metrics: {
    accuracy: 0.95,
    precision: 0.88,
    recall: 0.82,
    f1Score: 0.85,
    anomalyRate: 0.2,
    totalPoints: 5,
    anomalyCount: 1
  },
  createdAt: new Date(),
  processingTime: 150,
  ...overrides
});

export const createMockDataQualityResult = (overrides: any = {}) => ({
  id: 'quality-123',
  overallScore: 0.85,
  dimensionScores: {
    completeness: 0.9,
    accuracy: 0.8,
    consistency: 0.85,
    validity: 0.9,
    uniqueness: 0.75
  },
  ruleResults: [
    {
      ruleId: 'rule-1',
      ruleName: 'Non-null check',
      passed: true,
      score: 0.95,
      violationCount: 1,
      details: 'Most fields are complete'
    }
  ],
  issues: [
    {
      id: 'issue-1',
      type: 'completeness',
      severity: 'medium' as const,
      column: 'name',
      row: 5,
      message: 'Missing value',
      suggestedFix: 'Fill missing value'
    }
  ],
  recommendations: ['Review missing values in name column'],
  createdAt: new Date(),
  ...overrides
});

export const createMockJobStatus = (overrides: any = {}) => ({
  id: 'job-123',
  status: 'completed' as const,
  progress: 100,
  message: 'Job completed successfully',
  result: createMockAnomalyResult(),
  createdAt: new Date(),
  updatedAt: new Date(),
  ...overrides
});

// Mock API responses
export const createMockApiResponse = <T>(data: T, overrides: any = {}) => ({
  success: true,
  data,
  timestamp: new Date().toISOString(),
  ...overrides
});

export const createMockErrorResponse = (message: string = 'Test error') => ({
  success: false,
  error: message,
  timestamp: new Date().toISOString()
});

// HTTP mocking utilities
export const mockFetchSuccess = <T>(data: T) => {
  const mockResponse = {
    ok: true,
    status: 200,
    json: jest.fn().mockResolvedValue(createMockApiResponse(data)),
    text: jest.fn().mockResolvedValue(JSON.stringify(createMockApiResponse(data)))
  };
  (global.fetch as jest.Mock).mockResolvedValue(mockResponse);
  return mockResponse;
};

export const mockFetchError = (status: number = 500, message: string = 'Server error') => {
  const mockResponse = {
    ok: false,
    status,
    json: jest.fn().mockResolvedValue(createMockErrorResponse(message)),
    text: jest.fn().mockResolvedValue(JSON.stringify(createMockErrorResponse(message)))
  };
  (global.fetch as jest.Mock).mockResolvedValue(mockResponse);
  return mockResponse;
};

export const mockAxiosSuccess = <T>(data: T) => {
  const mockAxios = require('axios');
  mockAxios.mockResolvedValue({
    status: 200,
    data: createMockApiResponse(data),
    headers: {}
  });
  return mockAxios;
};

export const mockAxiosError = (status: number = 500, message: string = 'Server error') => {
  const mockAxios = require('axios');
  const error = new Error(message);
  (error as any).response = {
    status,
    data: createMockErrorResponse(message),
    headers: {}
  };
  mockAxios.mockRejectedValue(error);
  return mockAxios;
};

// WebSocket mocking utilities
export const createMockWebSocket = () => {
  const mockWS = {
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    send: jest.fn(),
    close: jest.fn(),
    readyState: 1,
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3,
    url: 'wss://test.com',
    protocol: '',
    bufferedAmount: 0,
    extensions: '',
    binaryType: 'blob' as BinaryType,
    onopen: null,
    onclose: null,
    onerror: null,
    onmessage: null,
    dispatchEvent: jest.fn()
  };

  // Simulate connection events
  mockWS.connect = () => {
    setTimeout(() => {
      if (mockWS.onopen) mockWS.onopen({} as Event);
    }, 0);
  };

  mockWS.disconnect = () => {
    setTimeout(() => {
      if (mockWS.onclose) mockWS.onclose({} as CloseEvent);
    }, 0);
  };

  mockWS.sendMessage = (data: any) => {
    setTimeout(() => {
      if (mockWS.onmessage) {
        mockWS.onmessage({ data: JSON.stringify(data) } as MessageEvent);
      }
    }, 0);
  };

  return mockWS;
};

// Timer utilities for testing async operations
export const flushPromises = () => new Promise(resolve => setImmediate(resolve));

export const waitFor = (condition: () => boolean, timeout: number = 1000): Promise<void> => {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    const checkCondition = () => {
      if (condition()) {
        resolve();
      } else if (Date.now() - startTime >= timeout) {
        reject(new Error('Timeout waiting for condition'));
      } else {
        setTimeout(checkCondition, 10);
      }
    };
    checkCondition();
  });
};

// Environment setup utilities
export const mockBrowserEnvironment = () => {
  Object.defineProperty(window, 'location', {
    value: {
      protocol: 'https:',
      hostname: 'test.com',
      port: '443'
    },
    writable: true
  });

  Object.defineProperty(window, 'navigator', {
    value: {
      userAgent: 'Mozilla/5.0 (Test Browser)'
    },
    writable: true
  });
};

export const mockNodeEnvironment = () => {
  Object.defineProperty(global, 'process', {
    value: {
      version: 'v16.0.0',
      env: { NODE_ENV: 'test' },
      versions: { node: '16.0.0' }
    },
    writable: true
  });
};

// Test data sets
export const sampleDatasets = {
  simple: [[1, 2], [2, 3], [3, 4], [100, 200]],
  numeric: [
    [1.1, 2.2, 3.3],
    [2.1, 3.2, 4.3],
    [3.1, 4.2, 5.3],
    [4.1, 5.2, 6.3],
    [100.1, 200.2, 300.3]
  ],
  categorical: [
    { name: 'John', age: 30, city: 'NYC' },
    { name: 'Jane', age: 25, city: 'LA' },
    { name: 'Bob', age: 35, city: 'Chicago' },
    { name: 'Alice', age: 28, city: 'Boston' },
    { name: 'Unknown', age: 999, city: 'Mars' } // Anomaly
  ],
  timeSeries: Array.from({ length: 100 }, (_, i) => [
    new Date(2023, 0, i + 1).getTime(),
    Math.sin(i * 0.1) + (i === 50 ? 10 : 0) // Anomaly at index 50
  ]),
  empty: [],
  single: [[1, 2, 3]]
};

// Assertion helpers
export const expectToBeValidUUID = (uuid: string) => {
  expect(uuid).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
};

export const expectToBeValidDate = (date: any) => {
  expect(date).toBeInstanceOf(Date);
  expect(date.getTime()).not.toBeNaN();
};

export const expectToBeValidApiResponse = (response: any) => {
  expect(response).toHaveProperty('success');
  expect(response).toHaveProperty('timestamp');
  if (response.success) {
    expect(response).toHaveProperty('data');
  } else {
    expect(response).toHaveProperty('error');
  }
};

// Performance testing utilities
export const measureExecutionTime = async (fn: () => Promise<any>): Promise<number> => {
  const start = performance.now();
  await fn();
  return performance.now() - start;
};

export const expectPerformanceWithin = async (
  fn: () => Promise<any>,
  maxTime: number
): Promise<void> => {
  const executionTime = await measureExecutionTime(fn);
  expect(executionTime).toBeLessThan(maxTime);
};