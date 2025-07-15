/**
 * Unit tests for PynomalyClient
 */

import { PynomalyClient } from '../../src/core/client';
import { AuthManager } from '../../src/auth/auth-manager';
import { 
  createMockConfig, 
  createMockAuthToken, 
  createMockUser,
  createMockAnomalyResult,
  createMockDataQualityResult,
  mockAxiosSuccess,
  mockAxiosError,
  flushPromises 
} from '../utils/test-helpers';

// Mock axios
jest.mock('axios');

// Mock AuthManager
jest.mock('../../src/auth/auth-manager');

describe('PynomalyClient', () => {
  let client: PynomalyClient;
  let mockConfig: any;
  let mockAuthManager: jest.Mocked<AuthManager>;

  beforeEach(() => {
    mockConfig = createMockConfig();
    mockAuthManager = {
      login: jest.fn(),
      loginWithApiKey: jest.fn(),
      logout: jest.fn(),
      refreshToken: jest.fn(),
      getAuthState: jest.fn(),
      isAuthenticated: jest.fn(),
      getCurrentToken: jest.fn(),
      on: jest.fn(),
      off: jest.fn(),
      emit: jest.fn()
    } as any;

    (AuthManager as jest.Mock).mockImplementation(() => mockAuthManager);
    client = new PynomalyClient(mockConfig);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('should create client with valid config', () => {
      expect(client).toBeInstanceOf(PynomalyClient);
      expect(client.config).toEqual(mockConfig);
    });

    it('should throw error with invalid config', () => {
      expect(() => new PynomalyClient({} as any)).toThrow('API key is required');
    });

    it('should set default config values', () => {
      const minimalConfig = { apiKey: 'test-key' };
      const clientWithDefaults = new PynomalyClient(minimalConfig);
      
      expect(clientWithDefaults.config.baseUrl).toBe('https://api.pynomaly.com');
      expect(clientWithDefaults.config.timeout).toBe(30000);
      expect(clientWithDefaults.config.debug).toBe(false);
    });
  });

  describe('authentication', () => {
    it('should authenticate with credentials', async () => {
      const mockToken = createMockAuthToken();
      const mockUser = createMockUser();
      mockAuthManager.login.mockResolvedValue({ token: mockToken, user: mockUser });

      const result = await client.authenticate('user@test.com', 'password');

      expect(mockAuthManager.login).toHaveBeenCalledWith({
        email: 'user@test.com',
        password: 'password'
      });
      expect(result).toEqual({ token: mockToken, user: mockUser });
    });

    it('should authenticate with API key', async () => {
      const mockToken = createMockAuthToken();
      mockAuthManager.loginWithApiKey.mockResolvedValue({ token: mockToken });

      const result = await client.authenticateWithApiKey('test-api-key');

      expect(mockAuthManager.loginWithApiKey).toHaveBeenCalledWith('test-api-key');
      expect(result).toEqual({ token: mockToken });
    });

    it('should handle authentication failure', async () => {
      const error = new Error('Invalid credentials');
      mockAuthManager.login.mockRejectedValue(error);

      await expect(client.authenticate('user@test.com', 'wrong-password'))
        .rejects.toThrow('Invalid credentials');
    });

    it('should logout successfully', async () => {
      mockAuthManager.logout.mockResolvedValue(undefined);

      await client.logout();

      expect(mockAuthManager.logout).toHaveBeenCalled();
    });
  });

  describe('anomaly detection', () => {
    beforeEach(() => {
      mockAuthManager.isAuthenticated.mockReturnValue(true);
      mockAuthManager.getCurrentToken.mockReturnValue(createMockAuthToken());
    });

    it('should detect anomalies successfully', async () => {
      const mockResult = createMockAnomalyResult();
      mockAxiosSuccess(mockResult);

      const result = await client.detectAnomalies({
        data: [[1, 2], [3, 4], [100, 200]],
        algorithm: 'isolation_forest'
      });

      expect(result).toEqual(mockResult);
    });

    it('should handle detection with custom parameters', async () => {
      const mockResult = createMockAnomalyResult();
      mockAxiosSuccess(mockResult);

      const params = {
        data: [[1, 2], [3, 4]],
        algorithm: 'local_outlier_factor' as const,
        parameters: { n_neighbors: 20, contamination: 0.2 }
      };

      const result = await client.detectAnomalies(params);

      expect(result).toEqual(mockResult);
    });

    it('should throw error when not authenticated', async () => {
      mockAuthManager.isAuthenticated.mockReturnValue(false);

      await expect(client.detectAnomalies({
        data: [[1, 2]],
        algorithm: 'isolation_forest'
      })).rejects.toThrow('Authentication required');
    });

    it('should handle API errors', async () => {
      mockAxiosError(500, 'Internal server error');

      await expect(client.detectAnomalies({
        data: [[1, 2]],
        algorithm: 'isolation_forest'
      })).rejects.toThrow('Internal server error');
    });

    it('should validate input data', async () => {
      await expect(client.detectAnomalies({
        data: [],
        algorithm: 'isolation_forest'
      })).rejects.toThrow('Data cannot be empty');

      await expect(client.detectAnomalies({
        data: [[1, 2]],
        algorithm: 'invalid_algorithm' as any
      })).rejects.toThrow('Invalid algorithm');
    });
  });

  describe('data quality analysis', () => {
    beforeEach(() => {
      mockAuthManager.isAuthenticated.mockReturnValue(true);
      mockAuthManager.getCurrentToken.mockReturnValue(createMockAuthToken());
    });

    it('should analyze data quality successfully', async () => {
      const mockResult = createMockDataQualityResult();
      mockAxiosSuccess(mockResult);

      const result = await client.analyzeDataQuality({
        data: [{ name: 'John', age: 30 }, { name: 'Jane', age: 25 }],
        rules: ['completeness', 'validity']
      });

      expect(result).toEqual(mockResult);
    });

    it('should handle custom quality rules', async () => {
      const mockResult = createMockDataQualityResult();
      mockAxiosSuccess(mockResult);

      const result = await client.analyzeDataQuality({
        data: [{ email: 'test@example.com' }],
        rules: ['email_format'],
        customRules: [{
          name: 'email_format',
          description: 'Validate email format',
          condition: 'email matches regex'
        }]
      });

      expect(result).toEqual(mockResult);
    });
  });

  describe('data profiling', () => {
    beforeEach(() => {
      mockAuthManager.isAuthenticated.mockReturnValue(true);
      mockAuthManager.getCurrentToken.mockReturnValue(createMockAuthToken());
    });

    it('should profile data successfully', async () => {
      const mockProfile = {
        columns: [
          {
            name: 'age',
            type: 'numeric',
            statistics: { mean: 30, std: 5, min: 25, max: 35 },
            nullCount: 0,
            uniqueCount: 3
          }
        ],
        rowCount: 3,
        columnCount: 2,
        memoryUsage: 1024
      };
      mockAxiosSuccess(mockProfile);

      const result = await client.profileData({
        data: [{ name: 'John', age: 30 }]
      });

      expect(result).toEqual(mockProfile);
    });
  });

  describe('async operations', () => {
    beforeEach(() => {
      mockAuthManager.isAuthenticated.mockReturnValue(true);
      mockAuthManager.getCurrentToken.mockReturnValue(createMockAuthToken());
    });

    it('should start async anomaly detection', async () => {
      const mockJobId = { jobId: 'job-123' };
      mockAxiosSuccess(mockJobId);

      const result = await client.detectAnomaliesAsync({
        data: [[1, 2], [3, 4]],
        algorithm: 'isolation_forest'
      });

      expect(result).toEqual(mockJobId);
    });

    it('should get job status', async () => {
      const mockStatus = {
        id: 'job-123',
        status: 'completed' as const,
        progress: 100,
        result: createMockAnomalyResult()
      };
      mockAxiosSuccess(mockStatus);

      const result = await client.getJobStatus('job-123');

      expect(result).toEqual(mockStatus);
    });

    it('should cancel job', async () => {
      mockAxiosSuccess({ cancelled: true });

      const result = await client.cancelJob('job-123');

      expect(result).toEqual({ cancelled: true });
    });
  });

  describe('request handling', () => {
    beforeEach(() => {
      mockAuthManager.isAuthenticated.mockReturnValue(true);
      mockAuthManager.getCurrentToken.mockReturnValue(createMockAuthToken());
    });

    it('should handle timeout errors', async () => {
      const timeoutError = new Error('timeout');
      (timeoutError as any).code = 'ECONNABORTED';
      mockAxiosError(0, 'timeout');

      await expect(client.detectAnomalies({
        data: [[1, 2]],
        algorithm: 'isolation_forest'
      })).rejects.toThrow('timeout');
    });

    it('should retry failed requests', async () => {
      const mockResult = createMockAnomalyResult();
      
      // First call fails, second succeeds
      const mockAxios = require('axios');
      mockAxios
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          status: 200,
          data: { success: true, data: mockResult }
        });

      const result = await client.detectAnomalies({
        data: [[1, 2]],
        algorithm: 'isolation_forest'
      });

      expect(result).toEqual(mockResult);
      expect(mockAxios).toHaveBeenCalledTimes(2);
    });

    it('should handle 401 unauthorized and refresh token', async () => {
      const newToken = createMockAuthToken({ token: 'new-token' });
      const mockResult = createMockAnomalyResult();
      
      // First call returns 401, token refresh succeeds, retry succeeds
      const mockAxios = require('axios');
      const unauthorizedError = new Error('Unauthorized');
      (unauthorizedError as any).response = { status: 401 };
      
      mockAxios
        .mockRejectedValueOnce(unauthorizedError)
        .mockResolvedValueOnce({
          status: 200,
          data: { success: true, data: mockResult }
        });

      mockAuthManager.refreshToken.mockResolvedValue(newToken);
      mockAuthManager.getCurrentToken.mockReturnValue(newToken);

      const result = await client.detectAnomalies({
        data: [[1, 2]],
        algorithm: 'isolation_forest'
      });

      expect(mockAuthManager.refreshToken).toHaveBeenCalled();
      expect(result).toEqual(mockResult);
    });
  });

  describe('event handling', () => {
    it('should emit events on authentication', async () => {
      const authListener = jest.fn();
      client.on('authenticated', authListener);

      const mockToken = createMockAuthToken();
      const mockUser = createMockUser();
      mockAuthManager.login.mockResolvedValue({ token: mockToken, user: mockUser });

      await client.authenticate('user@test.com', 'password');

      expect(authListener).toHaveBeenCalledWith({ token: mockToken, user: mockUser });
    });

    it('should emit events on errors', async () => {
      const errorListener = jest.fn();
      client.on('error', errorListener);

      const error = new Error('Test error');
      mockAuthManager.login.mockRejectedValue(error);

      try {
        await client.authenticate('user@test.com', 'password');
      } catch (e) {
        // Expected
      }

      expect(errorListener).toHaveBeenCalledWith(error);
    });

    it('should remove event listeners', () => {
      const listener = jest.fn();
      client.on('authenticated', listener);
      client.off('authenticated', listener);

      // Verify listener was removed (this is more of an integration test)
      expect(client.listenerCount('authenticated')).toBe(0);
    });
  });

  describe('configuration', () => {
    it('should update configuration', () => {
      const newConfig = createMockConfig({ timeout: 60000 });
      client.updateConfig(newConfig);

      expect(client.config.timeout).toBe(60000);
    });

    it('should validate updated configuration', () => {
      expect(() => {
        client.updateConfig({ apiKey: '' } as any);
      }).toThrow('API key is required');
    });
  });
});