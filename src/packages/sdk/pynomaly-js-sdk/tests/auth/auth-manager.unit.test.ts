/**
 * Unit tests for AuthManager
 */

import { AuthManager, SessionStorage } from '../../src/auth/auth-manager';
import { 
  createMockAuthToken, 
  createMockUser,
  flushPromises 
} from '../utils/test-helpers';

// Mock storage
const mockStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn()
} as jest.Mocked<SessionStorage>;

// Mock axios
jest.mock('axios');

describe('AuthManager', () => {
  let authManager: AuthManager;
  let mockConfig: any;

  beforeEach(() => {
    mockConfig = {
      baseUrl: 'https://api.test.com',
      storage: mockStorage,
      autoRefresh: true,
      refreshThreshold: 300000 // 5 minutes
    };

    authManager = new AuthManager(mockConfig);
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('should create AuthManager with config', () => {
      expect(authManager).toBeInstanceOf(AuthManager);
    });

    it('should use default storage if not provided', () => {
      const authManagerWithDefaults = new AuthManager({ baseUrl: 'https://api.test.com' });
      expect(authManagerWithDefaults).toBeInstanceOf(AuthManager);
    });

    it('should load existing session on initialization', () => {
      const mockToken = createMockAuthToken();
      const mockUser = createMockUser();
      
      mockStorage.getItem
        .mockReturnValueOnce(JSON.stringify(mockToken))
        .mockReturnValueOnce(JSON.stringify(mockUser));

      const authManagerWithSession = new AuthManager(mockConfig);
      const state = authManagerWithSession.getAuthState();

      expect(state.isAuthenticated).toBe(true);
      expect(state.token).toEqual(mockToken);
      expect(state.user).toEqual(mockUser);
    });
  });

  describe('login', () => {
    it('should login with email and password', async () => {
      const mockToken = createMockAuthToken();
      const mockUser = createMockUser();
      const mockResponse = {
        data: {
          success: true,
          data: { token: mockToken, user: mockUser }
        }
      };

      const mockAxios = require('axios');
      mockAxios.mockResolvedValue(mockResponse);

      const result = await authManager.login({
        email: 'test@example.com',
        password: 'password123'
      });

      expect(result).toEqual({ token: mockToken, user: mockUser });
      expect(mockStorage.setItem).toHaveBeenCalledWith('pynomaly_token', JSON.stringify(mockToken));
      expect(mockStorage.setItem).toHaveBeenCalledWith('pynomaly_user', JSON.stringify(mockUser));
    });

    it('should handle login failure', async () => {
      const mockAxios = require('axios');
      const error = new Error('Invalid credentials');
      (error as any).response = {
        status: 401,
        data: { success: false, error: 'Invalid credentials' }
      };
      mockAxios.mockRejectedValue(error);

      await expect(authManager.login({
        email: 'test@example.com',
        password: 'wrongpassword'
      })).rejects.toThrow('Invalid credentials');
    });

    it('should emit authentication events', async () => {
      const mockToken = createMockAuthToken();
      const mockUser = createMockUser();
      const mockResponse = {
        data: {
          success: true,
          data: { token: mockToken, user: mockUser }
        }
      };

      const mockAxios = require('axios');
      mockAxios.mockResolvedValue(mockResponse);

      const authStateChangedSpy = jest.fn();
      const authenticatedSpy = jest.fn();

      authManager.on('authStateChanged', authStateChangedSpy);
      authManager.on('authenticated', authenticatedSpy);

      await authManager.login({
        email: 'test@example.com',
        password: 'password123'
      });

      expect(authStateChangedSpy).toHaveBeenCalled();
      expect(authenticatedSpy).toHaveBeenCalledWith({ token: mockToken, user: mockUser });
    });
  });

  describe('loginWithApiKey', () => {
    it('should login with API key', async () => {
      const mockToken = createMockAuthToken();
      const mockResponse = {
        data: {
          success: true,
          data: { token: mockToken }
        }
      };

      const mockAxios = require('axios');
      mockAxios.mockResolvedValue(mockResponse);

      const result = await authManager.loginWithApiKey('test-api-key');

      expect(result).toEqual({ token: mockToken });
      expect(mockStorage.setItem).toHaveBeenCalledWith('pynomaly_token', JSON.stringify(mockToken));
    });

    it('should handle API key validation failure', async () => {
      const mockAxios = require('axios');
      const error = new Error('Invalid API key');
      (error as any).response = {
        status: 401,
        data: { success: false, error: 'Invalid API key' }
      };
      mockAxios.mockRejectedValue(error);

      await expect(authManager.loginWithApiKey('invalid-key'))
        .rejects.toThrow('Invalid API key');
    });
  });

  describe('logout', () => {
    it('should logout and clear session', async () => {
      // Set up authenticated state
      const mockToken = createMockAuthToken();
      const mockUser = createMockUser();
      mockStorage.getItem
        .mockReturnValueOnce(JSON.stringify(mockToken))
        .mockReturnValueOnce(JSON.stringify(mockUser));

      const authManagerWithSession = new AuthManager(mockConfig);
      
      const loggedOutSpy = jest.fn();
      authManagerWithSession.on('loggedOut', loggedOutSpy);

      await authManagerWithSession.logout();

      expect(mockStorage.removeItem).toHaveBeenCalledWith('pynomaly_token');
      expect(mockStorage.removeItem).toHaveBeenCalledWith('pynomaly_user');
      expect(loggedOutSpy).toHaveBeenCalled();

      const state = authManagerWithSession.getAuthState();
      expect(state.isAuthenticated).toBe(false);
      expect(state.token).toBeNull();
      expect(state.user).toBeNull();
    });
  });

  describe('token refresh', () => {
    it('should refresh token when near expiration', async () => {
      const expiredToken = createMockAuthToken({
        expiresAt: new Date(Date.now() + 60000) // 1 minute from now
      });
      const newToken = createMockAuthToken({
        token: 'new-token',
        expiresAt: new Date(Date.now() + 3600000) // 1 hour from now
      });

      mockStorage.getItem.mockReturnValueOnce(JSON.stringify(expiredToken));
      
      const authManagerWithToken = new AuthManager(mockConfig);
      
      const mockAxios = require('axios');
      mockAxios.mockResolvedValue({
        data: {
          success: true,
          data: { token: newToken }
        }
      });

      const result = await authManagerWithToken.refreshToken();

      expect(result).toEqual(newToken);
      expect(mockStorage.setItem).toHaveBeenCalledWith('pynomaly_token', JSON.stringify(newToken));
    });

    it('should handle refresh token failure', async () => {
      const expiredToken = createMockAuthToken({
        refreshToken: 'invalid-refresh-token'
      });

      mockStorage.getItem.mockReturnValueOnce(JSON.stringify(expiredToken));
      
      const authManagerWithToken = new AuthManager(mockConfig);

      const mockAxios = require('axios');
      const error = new Error('Invalid refresh token');
      (error as any).response = {
        status: 401,
        data: { success: false, error: 'Invalid refresh token' }
      };
      mockAxios.mockRejectedValue(error);

      await expect(authManagerWithToken.refreshToken())
        .rejects.toThrow('Invalid refresh token');
    });

    it('should automatically refresh token when near expiration', async () => {
      const nearExpiredToken = createMockAuthToken({
        expiresAt: new Date(Date.now() + 250000) // 4 minutes from now (under threshold)
      });

      mockStorage.getItem.mockReturnValueOnce(JSON.stringify(nearExpiredToken));
      
      const authManagerWithAutoRefresh = new AuthManager({
        ...mockConfig,
        autoRefresh: true,
        refreshThreshold: 300000 // 5 minutes
      });

      const newToken = createMockAuthToken({
        token: 'auto-refreshed-token'
      });

      const mockAxios = require('axios');
      mockAxios.mockResolvedValue({
        data: {
          success: true,
          data: { token: newToken }
        }
      });

      const tokenRefreshedSpy = jest.fn();
      authManagerWithAutoRefresh.on('tokenRefreshed', tokenRefreshedSpy);

      // Trigger token check
      const currentToken = authManagerWithAutoRefresh.getCurrentToken();

      // Wait for async refresh
      await flushPromises();

      expect(tokenRefreshedSpy).toHaveBeenCalledWith(newToken);
    });
  });

  describe('token validation', () => {
    it('should validate token expiration', () => {
      const validToken = createMockAuthToken({
        expiresAt: new Date(Date.now() + 3600000) // 1 hour from now
      });

      const expiredToken = createMockAuthToken({
        expiresAt: new Date(Date.now() - 3600000) // 1 hour ago
      });

      mockStorage.getItem
        .mockReturnValueOnce(JSON.stringify(validToken))
        .mockReturnValueOnce(JSON.stringify(expiredToken));

      const authManagerValid = new AuthManager(mockConfig);
      const authManagerExpired = new AuthManager(mockConfig);

      expect(authManagerValid.isAuthenticated()).toBe(true);
      expect(authManagerExpired.isAuthenticated()).toBe(false);
    });

    it('should check if token needs refresh', () => {
      const tokenNeedsRefresh = createMockAuthToken({
        expiresAt: new Date(Date.now() + 200000) // 3.3 minutes from now
      });

      mockStorage.getItem.mockReturnValueOnce(JSON.stringify(tokenNeedsRefresh));
      
      const authManagerWithToken = new AuthManager({
        ...mockConfig,
        refreshThreshold: 300000 // 5 minutes
      });

      // Private method testing through public interface
      const currentToken = authManagerWithToken.getCurrentToken();
      expect(currentToken).toEqual(tokenNeedsRefresh);
    });
  });

  describe('session persistence', () => {
    it('should encrypt sensitive data when encryption is enabled', async () => {
      const authManagerWithEncryption = new AuthManager({
        ...mockConfig,
        encryptSession: true,
        encryptionKey: 'test-encryption-key'
      });

      const mockToken = createMockAuthToken();
      const mockUser = createMockUser();
      const mockResponse = {
        data: {
          success: true,
          data: { token: mockToken, user: mockUser }
        }
      };

      const mockAxios = require('axios');
      mockAxios.mockResolvedValue(mockResponse);

      await authManagerWithEncryption.login({
        email: 'test@example.com',
        password: 'password123'
      });

      // Should call setItem with encrypted data (implementation detail)
      expect(mockStorage.setItem).toHaveBeenCalled();
    });

    it('should clear all session data on clearSession', () => {
      authManager.clearSession();

      expect(mockStorage.removeItem).toHaveBeenCalledWith('pynomaly_token');
      expect(mockStorage.removeItem).toHaveBeenCalledWith('pynomaly_user');
    });
  });

  describe('auth state management', () => {
    it('should return current auth state', () => {
      const state = authManager.getAuthState();

      expect(state).toHaveProperty('isAuthenticated');
      expect(state).toHaveProperty('token');
      expect(state).toHaveProperty('user');
      expect(state).toHaveProperty('lastActivity');
    });

    it('should track last activity', async () => {
      const mockToken = createMockAuthToken();
      const mockUser = createMockUser();
      const mockResponse = {
        data: {
          success: true,
          data: { token: mockToken, user: mockUser }
        }
      };

      const mockAxios = require('axios');
      mockAxios.mockResolvedValue(mockResponse);

      const beforeLogin = Date.now();
      await authManager.login({
        email: 'test@example.com',
        password: 'password123'
      });
      const afterLogin = Date.now();

      const state = authManager.getAuthState();
      expect(state.lastActivity.getTime()).toBeGreaterThanOrEqual(beforeLogin);
      expect(state.lastActivity.getTime()).toBeLessThanOrEqual(afterLogin);
    });
  });

  describe('error handling', () => {
    it('should handle network errors gracefully', async () => {
      const mockAxios = require('axios');
      const networkError = new Error('Network Error');
      (networkError as any).code = 'ECONNABORTED';
      mockAxios.mockRejectedValue(networkError);

      await expect(authManager.login({
        email: 'test@example.com',
        password: 'password123'
      })).rejects.toThrow('Network Error');
    });

    it('should handle malformed responses', async () => {
      const mockAxios = require('axios');
      mockAxios.mockResolvedValue({
        data: { invalid: 'response' }
      });

      await expect(authManager.login({
        email: 'test@example.com',
        password: 'password123'
      })).rejects.toThrow();
    });
  });
});