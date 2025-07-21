/**
 * Unit tests for React hooks
 */

import React from 'react';
import { renderHook, act } from '@testing-library/react';
import { usePynomalyClient, usePynomalyAuth } from '../../../src/frameworks/react/hooks';
import { PynomalyProvider } from '../../../src/frameworks/react/components/PynomalyProvider';
import { 
  createMockConfig, 
  createMockAuthToken, 
  createMockUser,
  flushPromises 
} from '../../utils/test-helpers';

// Mock the PynomalyClient
jest.mock('../../../src/core/client');
jest.mock('../../../src/auth/auth-manager');

const MockPynomalyClient = require('../../../src/core/client').PynomalyClient as jest.Mock;

describe('React Hooks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('usePynomalyClient', () => {
    let mockClient: any;

    beforeEach(() => {
      mockClient = {
        connect: jest.fn().mockResolvedValue(undefined),
        disconnect: jest.fn(),
        on: jest.fn(),
        off: jest.fn(),
        removeAllListeners: jest.fn(),
        isConnected: jest.fn().mockReturnValue(false),
        authenticate: jest.fn(),
        logout: jest.fn()
      };
      MockPynomalyClient.mockImplementation(() => mockClient);
    });

    it('should initialize client with config', () => {
      const config = createMockConfig();
      const { result } = renderHook(() => usePynomalyClient(config));

      expect(MockPynomalyClient).toHaveBeenCalledWith(config);
      expect(result.current.client).toBe(mockClient);
    });

    it('should handle auto-connect', async () => {
      const config = createMockConfig();
      const { result } = renderHook(() => 
        usePynomalyClient({ ...config, autoConnect: true })
      );

      expect(result.current.isLoading).toBe(true);
      
      await act(async () => {
        await flushPromises();
      });

      expect(mockClient.connect).toHaveBeenCalled();
    });

    it('should track connection state', async () => {
      const config = createMockConfig();
      const { result } = renderHook(() => usePynomalyClient(config));

      expect(result.current.isReady).toBe(false);
      expect(result.current.isLoading).toBe(false);

      // Simulate connecting
      act(() => {
        result.current.reconnect();
      });

      expect(result.current.isLoading).toBe(true);

      // Simulate connection success
      await act(async () => {
        mockClient.isConnected.mockReturnValue(true);
        await flushPromises();
      });

      expect(result.current.isReady).toBe(true);
      expect(result.current.isLoading).toBe(false);
    });

    it('should handle connection errors', async () => {
      const config = createMockConfig();
      const error = new Error('Connection failed');
      mockClient.connect.mockRejectedValue(error);

      const { result } = renderHook(() => usePynomalyClient(config));

      await act(async () => {
        try {
          await result.current.reconnect();
        } catch (e) {
          // Expected
        }
      });

      expect(result.current.error).toBe(error);
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isReady).toBe(false);
    });

    it('should disconnect on unmount', () => {
      const config = createMockConfig();
      const { unmount } = renderHook(() => usePynomalyClient(config));

      unmount();

      expect(mockClient.disconnect).toHaveBeenCalled();
      expect(mockClient.removeAllListeners).toHaveBeenCalled();
    });

    it('should handle manual reconnect', async () => {
      const config = createMockConfig();
      const { result } = renderHook(() => usePynomalyClient(config));

      await act(async () => {
        await result.current.reconnect();
      });

      expect(mockClient.connect).toHaveBeenCalled();
    });

    it('should handle manual disconnect', () => {
      const config = createMockConfig();
      const { result } = renderHook(() => usePynomalyClient(config));

      act(() => {
        result.current.disconnect();
      });

      expect(mockClient.disconnect).toHaveBeenCalled();
    });
  });

  describe('usePynomalyAuth', () => {
    let mockClient: any;
    let mockAuthManager: any;

    beforeEach(() => {
      mockAuthManager = {
        login: jest.fn(),
        loginWithApiKey: jest.fn(),
        logout: jest.fn(),
        refreshToken: jest.fn(),
        getAuthState: jest.fn().mockReturnValue({
          isAuthenticated: false,
          token: null,
          user: null,
          lastActivity: new Date()
        }),
        on: jest.fn(),
        off: jest.fn()
      };

      mockClient = {
        authManager: mockAuthManager,
        authenticate: jest.fn(),
        authenticateWithApiKey: jest.fn(),
        logout: jest.fn()
      };
    });

    it('should initialize with auth state', () => {
      const { result } = renderHook(() => usePynomalyAuth({ client: mockClient }));

      expect(result.current.authState.isAuthenticated).toBe(false);
      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBeNull();
    });

    it('should handle login', async () => {
      const mockToken = createMockAuthToken();
      const mockUser = createMockUser();
      mockClient.authenticate.mockResolvedValue({ token: mockToken, user: mockUser });

      const { result } = renderHook(() => usePynomalyAuth({ client: mockClient }));

      await act(async () => {
        await result.current.login({
          email: 'test@example.com',
          password: 'password123'
        });
      });

      expect(mockClient.authenticate).toHaveBeenCalledWith('test@example.com', 'password123');
      expect(result.current.isLoading).toBe(false);
    });

    it('should handle login with API key', async () => {
      const mockToken = createMockAuthToken();
      mockClient.authenticateWithApiKey.mockResolvedValue({ token: mockToken });

      const { result } = renderHook(() => usePynomalyAuth({ client: mockClient }));

      await act(async () => {
        await result.current.loginWithApiKey('test-api-key');
      });

      expect(mockClient.authenticateWithApiKey).toHaveBeenCalledWith('test-api-key');
    });

    it('should handle login errors', async () => {
      const error = new Error('Invalid credentials');
      mockClient.authenticate.mockRejectedValue(error);

      const { result } = renderHook(() => usePynomalyAuth({ client: mockClient }));

      await act(async () => {
        try {
          await result.current.login({
            email: 'test@example.com',
            password: 'wrongpassword'
          });
        } catch (e) {
          // Expected
        }
      });

      expect(result.current.error).toBe(error);
      expect(result.current.isLoading).toBe(false);
    });

    it('should handle logout', async () => {
      mockClient.logout.mockResolvedValue(undefined);

      const { result } = renderHook(() => usePynomalyAuth({ client: mockClient }));

      await act(async () => {
        await result.current.logout();
      });

      expect(mockClient.logout).toHaveBeenCalled();
    });

    it('should track loading state during operations', async () => {
      mockClient.authenticate.mockImplementation(() => 
        new Promise(resolve => setTimeout(resolve, 100))
      );

      const { result } = renderHook(() => usePynomalyAuth({ client: mockClient }));

      act(() => {
        result.current.login({
          email: 'test@example.com',
          password: 'password123'
        });
      });

      expect(result.current.isLoading).toBe(true);

      await act(async () => {
        await flushPromises();
      });

      expect(result.current.isLoading).toBe(false);
    });

    it('should update auth state on auth manager events', () => {
      const { result } = renderHook(() => usePynomalyAuth({ client: mockClient }));

      // Simulate auth state change
      const newAuthState = {
        isAuthenticated: true,
        token: createMockAuthToken(),
        user: createMockUser(),
        lastActivity: new Date()
      };

      mockAuthManager.getAuthState.mockReturnValue(newAuthState);

      // Simulate auth manager event
      act(() => {
        const authStateChangeCallback = mockAuthManager.on.mock.calls
          .find(call => call[0] === 'authStateChanged')?.[1];
        if (authStateChangeCallback) {
          authStateChangeCallback(newAuthState);
        }
      });

      expect(result.current.authState.isAuthenticated).toBe(true);
      expect(result.current.authState.user).toEqual(newAuthState.user);
    });

    it('should handle refresh token', async () => {
      const newToken = createMockAuthToken({ token: 'new-token' });
      mockClient.authManager.refreshToken.mockResolvedValue(newToken);

      const { result } = renderHook(() => usePynomalyAuth({ client: mockClient }));

      await act(async () => {
        await result.current.refreshToken();
      });

      expect(mockClient.authManager.refreshToken).toHaveBeenCalled();
    });

    it('should cleanup event listeners on unmount', () => {
      const { unmount } = renderHook(() => usePynomalyAuth({ client: mockClient }));

      unmount();

      expect(mockAuthManager.off).toHaveBeenCalled();
    });
  });

  describe('PynomalyProvider integration', () => {
    let mockClient: any;

    beforeEach(() => {
      mockClient = {
        connect: jest.fn().mockResolvedValue(undefined),
        disconnect: jest.fn(),
        on: jest.fn(),
        off: jest.fn(),
        removeAllListeners: jest.fn(),
        isConnected: jest.fn().mockReturnValue(true),
        authManager: {
          getAuthState: jest.fn().mockReturnValue({
            isAuthenticated: false,
            token: null,
            user: null,
            lastActivity: new Date()
          }),
          on: jest.fn(),
          off: jest.fn()
        }
      };
      MockPynomalyClient.mockImplementation(() => mockClient);
    });

    it('should provide client through context', () => {
      const config = createMockConfig();
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <PynomalyProvider config={config} autoConnect={false}>
          {children}
        </PynomalyProvider>
      );

      const { result } = renderHook(() => usePynomalyClient(), { wrapper });

      expect(result.current.client).toBe(mockClient);
      expect(result.current.isReady).toBe(true);
    });

    it('should handle provider errors', () => {
      const config = createMockConfig();
      const error = new Error('Provider error');
      
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <PynomalyProvider 
          config={config} 
          autoConnect={false}
          onError={() => {}}
        >
          {children}
        </PynomalyProvider>
      );

      // Simulate error in provider
      mockClient.on.mockImplementation((event: string, callback: Function) => {
        if (event === 'error') {
          setTimeout(() => callback(error), 0);
        }
      });

      const { result } = renderHook(() => usePynomalyClient(), { wrapper });

      expect(result.current.client).toBe(mockClient);
    });

    it('should handle provider ready state', async () => {
      const config = createMockConfig();
      const onReady = jest.fn();
      
      const wrapper = ({ children }: { children: React.ReactNode }) => (
        <PynomalyProvider 
          config={config} 
          autoConnect={true}
          onReady={onReady}
        >
          {children}
        </PynomalyProvider>
      );

      renderHook(() => usePynomalyClient(), { wrapper });

      await act(async () => {
        await flushPromises();
      });

      expect(mockClient.connect).toHaveBeenCalled();
    });
  });
});