/**
 * Integration tests for end-to-end SDK functionality
 */

import { PynomalyClient } from '../../src/core/client';
import { 
  createMockConfig,
  createMockAnomalyResult,
  createMockDataQualityResult,
  mockAxiosSuccess,
  mockAxiosError,
  flushPromises,
  sampleDatasets
} from '../utils/test-helpers';

// Mock axios for integration tests
jest.mock('axios');

describe('End-to-End Integration Tests', () => {
  let client: PynomalyClient;
  let config: any;

  beforeEach(() => {
    config = createMockConfig({
      baseUrl: 'https://integration-test.pynomaly.com',
      timeout: 10000,
      debug: true
    });
    client = new PynomalyClient(config);
    jest.clearAllMocks();
  });

  afterEach(() => {
    if (client) {
      client.disconnect();
    }
  });

  describe('complete authentication flow', () => {
    it('should authenticate, perform operations, and logout', async () => {
      // Mock authentication response
      const mockAuthResponse = {
        token: {
          token: 'integration-test-token',
          refreshToken: 'integration-test-refresh',
          expiresAt: new Date(Date.now() + 3600000),
          tokenType: 'Bearer'
        },
        user: {
          id: 'user-integration-test',
          email: 'integration@test.com',
          name: 'Integration Test User',
          roles: ['user'],
          permissions: ['read', 'write']
        }
      };

      const mockAxios = require('axios');
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: mockAuthResponse }
      });

      // Step 1: Authenticate
      const authResult = await client.authenticate('integration@test.com', 'test-password');
      
      expect(authResult.token.token).toBe('integration-test-token');
      expect(authResult.user.email).toBe('integration@test.com');
      expect(client.isAuthenticated()).toBe(true);

      // Step 2: Perform anomaly detection
      const mockAnomalyResult = createMockAnomalyResult({
        id: 'integration-anomaly-123',
        algorithm: 'isolation_forest',
        metrics: {
          accuracy: 0.96,
          anomalyCount: 2
        }
      });

      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: mockAnomalyResult }
      });

      const detectionResult = await client.detectAnomalies({
        data: sampleDatasets.numeric,
        algorithm: 'isolation_forest',
        parameters: { contamination: 0.1 }
      });

      expect(detectionResult.id).toBe('integration-anomaly-123');
      expect(detectionResult.metrics.anomalyCount).toBe(2);

      // Step 3: Analyze data quality
      const mockQualityResult = createMockDataQualityResult({
        id: 'integration-quality-123',
        overallScore: 0.88
      });

      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: mockQualityResult }
      });

      const qualityResult = await client.analyzeDataQuality({
        data: sampleDatasets.categorical,
        rules: ['completeness', 'validity', 'consistency']
      });

      expect(qualityResult.id).toBe('integration-quality-123');
      expect(qualityResult.overallScore).toBe(0.88);

      // Step 4: Logout
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true }
      });

      await client.logout();
      expect(client.isAuthenticated()).toBe(false);
    });

    it('should handle authentication failure and retry', async () => {
      const mockAxios = require('axios');
      
      // First attempt fails
      const authError = new Error('Invalid credentials');
      (authError as any).response = {
        status: 401,
        data: { success: false, error: 'Invalid credentials' }
      };
      mockAxios.mockRejectedValueOnce(authError);

      // First login attempt should fail
      await expect(client.authenticate('wrong@email.com', 'wrongpassword'))
        .rejects.toThrow('Invalid credentials');

      expect(client.isAuthenticated()).toBe(false);

      // Second attempt succeeds
      const mockAuthResponse = {
        token: {
          token: 'correct-token',
          refreshToken: 'correct-refresh',
          expiresAt: new Date(Date.now() + 3600000),
          tokenType: 'Bearer'
        },
        user: {
          id: 'user-correct',
          email: 'correct@test.com',
          name: 'Correct User'
        }
      };

      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: mockAuthResponse }
      });

      const authResult = await client.authenticate('correct@test.com', 'correctpassword');
      expect(authResult.token.token).toBe('correct-token');
      expect(client.isAuthenticated()).toBe(true);
    });
  });

  describe('async operation workflow', () => {
    beforeEach(async () => {
      // Mock authentication
      const mockAxios = require('axios');
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: {
          success: true,
          data: {
            token: {
              token: 'async-test-token',
              expiresAt: new Date(Date.now() + 3600000)
            }
          }
        }
      });

      await client.authenticateWithApiKey('async-test-key');
    });

    it('should handle complete async detection workflow', async () => {
      const mockAxios = require('axios');

      // Step 1: Start async detection
      const jobId = 'async-job-integration-123';
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: { jobId } }
      });

      const startResult = await client.detectAnomaliesAsync({
        data: sampleDatasets.timeSeries,
        algorithm: 'isolation_forest'
      });

      expect(startResult.jobId).toBe(jobId);

      // Step 2: Poll job status - in progress
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: {
          success: true,
          data: {
            id: jobId,
            status: 'running',
            progress: 45,
            message: 'Processing data...'
          }
        }
      });

      const statusInProgress = await client.getJobStatus(jobId);
      expect(statusInProgress.status).toBe('running');
      expect(statusInProgress.progress).toBe(45);

      // Step 3: Poll job status - completed
      const finalResult = createMockAnomalyResult({
        id: 'async-result-123',
        algorithm: 'isolation_forest',
        metrics: { anomalyCount: 3 }
      });

      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: {
          success: true,
          data: {
            id: jobId,
            status: 'completed',
            progress: 100,
            result: finalResult,
            message: 'Detection completed successfully'
          }
        }
      });

      const statusCompleted = await client.getJobStatus(jobId);
      expect(statusCompleted.status).toBe('completed');
      expect(statusCompleted.progress).toBe(100);
      expect(statusCompleted.result.metrics.anomalyCount).toBe(3);
    });

    it('should handle job cancellation', async () => {
      const mockAxios = require('axios');

      // Start async job
      const jobId = 'cancellation-test-job';
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: { jobId } }
      });

      await client.detectAnomaliesAsync({
        data: sampleDatasets.simple,
        algorithm: 'local_outlier_factor'
      });

      // Cancel job
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: { cancelled: true, jobId } }
      });

      const cancelResult = await client.cancelJob(jobId);
      expect(cancelResult.cancelled).toBe(true);
      expect(cancelResult.jobId).toBe(jobId);

      // Check status after cancellation
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: {
          success: true,
          data: {
            id: jobId,
            status: 'cancelled',
            progress: 30,
            message: 'Job was cancelled'
          }
        }
      });

      const finalStatus = await client.getJobStatus(jobId);
      expect(finalStatus.status).toBe('cancelled');
    });
  });

  describe('error handling and recovery', () => {
    beforeEach(async () => {
      // Mock authentication
      const mockAxios = require('axios');
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: {
          success: true,
          data: {
            token: {
              token: 'error-test-token',
              expiresAt: new Date(Date.now() + 3600000)
            }
          }
        }
      });

      await client.authenticateWithApiKey('error-test-key');
    });

    it('should handle server errors and retry', async () => {
      const mockAxios = require('axios');

      // First request fails with server error
      const serverError = new Error('Internal Server Error');
      (serverError as any).response = {
        status: 500,
        data: { success: false, error: 'Internal Server Error' }
      };
      mockAxios.mockRejectedValueOnce(serverError);

      // Second request succeeds
      const mockResult = createMockAnomalyResult();
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: mockResult }
      });

      // Should retry and succeed
      const result = await client.detectAnomalies({
        data: sampleDatasets.simple,
        algorithm: 'isolation_forest'
      });

      expect(result).toEqual(mockResult);
      expect(mockAxios).toHaveBeenCalledTimes(3); // auth + failed request + retry
    });

    it('should handle token expiration and refresh', async () => {
      const mockAxios = require('axios');

      // First request fails with 401 (token expired)
      const tokenExpiredError = new Error('Token expired');
      (tokenExpiredError as any).response = {
        status: 401,
        data: { success: false, error: 'Token expired' }
      };
      mockAxios.mockRejectedValueOnce(tokenExpiredError);

      // Token refresh succeeds
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: {
          success: true,
          data: {
            token: {
              token: 'refreshed-token',
              expiresAt: new Date(Date.now() + 3600000)
            }
          }
        }
      });

      // Retry with new token succeeds
      const mockResult = createMockAnomalyResult();
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: mockResult }
      });

      const result = await client.detectAnomalies({
        data: sampleDatasets.simple,
        algorithm: 'isolation_forest'
      });

      expect(result).toEqual(mockResult);
      // Should have made: auth + failed request + refresh + retry
      expect(mockAxios).toHaveBeenCalledTimes(4);
    });

    it('should handle network timeout and retry', async () => {
      const mockAxios = require('axios');

      // First request times out
      const timeoutError = new Error('Request timeout');
      (timeoutError as any).code = 'ECONNABORTED';
      mockAxios.mockRejectedValueOnce(timeoutError);

      // Second request succeeds
      const mockResult = createMockDataQualityResult();
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: mockResult }
      });

      const result = await client.analyzeDataQuality({
        data: sampleDatasets.categorical,
        rules: ['completeness']
      });

      expect(result).toEqual(mockResult);
    });
  });

  describe('data validation and preprocessing', () => {
    beforeEach(async () => {
      // Mock authentication
      const mockAxios = require('axios');
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: {
          success: true,
          data: {
            token: {
              token: 'validation-test-token',
              expiresAt: new Date(Date.now() + 3600000)
            }
          }
        }
      });

      await client.authenticateWithApiKey('validation-test-key');
    });

    it('should validate and process different data types', async () => {
      const mockAxios = require('axios');

      // Test numeric data
      const numericResult = createMockAnomalyResult({
        algorithm: 'isolation_forest'
      });
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: numericResult }
      });

      await client.detectAnomalies({
        data: sampleDatasets.numeric,
        algorithm: 'isolation_forest'
      });

      // Test categorical data
      const categoricalResult = createMockDataQualityResult();
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: categoricalResult }
      });

      await client.analyzeDataQuality({
        data: sampleDatasets.categorical,
        rules: ['completeness', 'validity']
      });

      // Test time series data
      const timeSeriesResult = createMockAnomalyResult({
        algorithm: 'seasonal_decompose'
      });
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: timeSeriesResult }
      });

      await client.detectAnomalies({
        data: sampleDatasets.timeSeries,
        algorithm: 'seasonal_decompose'
      });

      expect(mockAxios).toHaveBeenCalledTimes(4); // auth + 3 operations
    });

    it('should handle edge cases in data', async () => {
      const mockAxios = require('axios');

      // Test empty data
      await expect(client.detectAnomalies({
        data: sampleDatasets.empty,
        algorithm: 'isolation_forest'
      })).rejects.toThrow('Data cannot be empty');

      // Test single data point
      const singlePointResult = createMockAnomalyResult({
        metrics: { anomalyCount: 0 }
      });
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: { success: true, data: singlePointResult }
      });

      const result = await client.detectAnomalies({
        data: sampleDatasets.single,
        algorithm: 'isolation_forest'
      });

      expect(result.metrics.anomalyCount).toBe(0);
    });
  });

  describe('concurrent operations', () => {
    beforeEach(async () => {
      // Mock authentication
      const mockAxios = require('axios');
      mockAxios.mockResolvedValueOnce({
        status: 200,
        data: {
          success: true,
          data: {
            token: {
              token: 'concurrent-test-token',
              expiresAt: new Date(Date.now() + 3600000)
            }
          }
        }
      });

      await client.authenticateWithApiKey('concurrent-test-key');
    });

    it('should handle multiple simultaneous operations', async () => {
      const mockAxios = require('axios');

      // Mock responses for concurrent operations
      const anomalyResult1 = createMockAnomalyResult({ id: 'concurrent-1' });
      const anomalyResult2 = createMockAnomalyResult({ id: 'concurrent-2' });
      const qualityResult = createMockDataQualityResult({ id: 'concurrent-3' });

      mockAxios
        .mockResolvedValueOnce({
          status: 200,
          data: { success: true, data: anomalyResult1 }
        })
        .mockResolvedValueOnce({
          status: 200,
          data: { success: true, data: anomalyResult2 }
        })
        .mockResolvedValueOnce({
          status: 200,
          data: { success: true, data: qualityResult }
        });

      // Start all operations concurrently
      const [result1, result2, result3] = await Promise.all([
        client.detectAnomalies({
          data: sampleDatasets.numeric,
          algorithm: 'isolation_forest'
        }),
        client.detectAnomalies({
          data: sampleDatasets.simple,
          algorithm: 'local_outlier_factor'
        }),
        client.analyzeDataQuality({
          data: sampleDatasets.categorical,
          rules: ['completeness']
        })
      ]);

      expect(result1.id).toBe('concurrent-1');
      expect(result2.id).toBe('concurrent-2');
      expect(result3.id).toBe('concurrent-3');
    });
  });

  describe('configuration and customization', () => {
    it('should respect custom configuration', async () => {
      const customConfig = createMockConfig({
        baseUrl: 'https://custom.pynomaly.com',
        timeout: 5000,
        retryAttempts: 2,
        debug: true
      });

      const customClient = new PynomalyClient(customConfig);
      
      expect(customClient.config.baseUrl).toBe('https://custom.pynomaly.com');
      expect(customClient.config.timeout).toBe(5000);
      expect(customClient.config.retryAttempts).toBe(2);
      expect(customClient.config.debug).toBe(true);
    });

    it('should allow runtime configuration updates', () => {
      const updatedConfig = createMockConfig({
        timeout: 15000,
        debug: false
      });

      client.updateConfig(updatedConfig);

      expect(client.config.timeout).toBe(15000);
      expect(client.config.debug).toBe(false);
    });
  });
});