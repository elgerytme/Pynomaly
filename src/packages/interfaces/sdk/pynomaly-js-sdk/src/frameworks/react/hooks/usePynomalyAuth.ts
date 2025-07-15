/**
 * React hook for Pynomaly authentication
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { AuthManager, AuthState, AuthCredentials, SessionConfig } from '../../../index';
import { PynomalyClient } from '../../../core/client';

export interface UsePynomalyAuthOptions extends Partial<SessionConfig> {
  client?: PynomalyClient;
  onLogin?: (user: any) => void;
  onLogout?: () => void;
  onError?: (error: string, type: string) => void;
  onTokenRefresh?: (token: any) => void;
}

export interface UsePynomalyAuthReturn {
  authManager: AuthManager | null;
  authState: AuthState;
  isAuthenticated: boolean;
  isLoading: boolean;
  user: any;
  token: any;
  login: (credentials: AuthCredentials) => Promise<void>;
  loginWithApiKey: (apiKey: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
  error: string | null;
}

export function usePynomalyAuth(
  options: UsePynomalyAuthOptions = {}
): UsePynomalyAuthReturn {
  const [authManager, setAuthManager] = useState<AuthManager | null>(null);
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    user: null,
    token: null,
    expiresAt: null
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const optionsRef = useRef(options);
  const mountedRef = useRef(true);

  // Update options ref when options change
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  // Initialize auth manager
  useEffect(() => {
    const manager = new AuthManager({
      enablePersistence: true,
      autoRefresh: true,
      ...options
    });

    // Set up event listeners
    manager.on('auth:login', ({ user, token }) => {
      if (mountedRef.current) {
        setAuthState(manager.getAuthState());
        setError(null);
        optionsRef.current.onLogin?.(user);
      }
    });

    manager.on('auth:logout', () => {
      if (mountedRef.current) {
        setAuthState(manager.getAuthState());
        setError(null);
        optionsRef.current.onLogout?.();
      }
    });

    manager.on('auth:refresh', (token) => {
      if (mountedRef.current) {
        setAuthState(manager.getAuthState());
        optionsRef.current.onTokenRefresh?.(token);
      }
    });

    manager.on('auth:error', ({ error, type }) => {
      if (mountedRef.current) {
        setError(error);
        optionsRef.current.onError?.(error, type);
      }
    });

    manager.on('auth:expired', () => {
      if (mountedRef.current) {
        setAuthState(manager.getAuthState());
        setError('Session expired');
      }
    });

    setAuthManager(manager);
    setAuthState(manager.getAuthState());

    return () => {
      mountedRef.current = false;
      manager.destroy();
    };
  }, []);

  const login = useCallback(async (credentials: AuthCredentials) => {
    if (!authManager || !optionsRef.current.client) {
      throw new Error('Auth manager or client not available');
    }

    setIsLoading(true);
    setError(null);

    try {
      await authManager.login(credentials, optionsRef.current.client);
    } catch (err) {
      const error = err as Error;
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [authManager]);

  const loginWithApiKey = useCallback(async (apiKey: string) => {
    if (!authManager || !optionsRef.current.client) {
      throw new Error('Auth manager or client not available');
    }

    setIsLoading(true);
    setError(null);

    try {
      await authManager.loginWithApiKey(apiKey, optionsRef.current.client);
    } catch (err) {
      const error = err as Error;
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [authManager]);

  const logout = useCallback(async () => {
    if (!authManager || !optionsRef.current.client) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      await authManager.logout(optionsRef.current.client);
    } catch (err) {
      const error = err as Error;
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  }, [authManager]);

  const refreshToken = useCallback(async () => {
    if (!authManager || !optionsRef.current.client) {
      throw new Error('Auth manager or client not available');
    }

    setIsLoading(true);
    setError(null);

    try {
      await authManager.refreshToken(optionsRef.current.client);
    } catch (err) {
      const error = err as Error;
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [authManager]);

  return {
    authManager,
    authState,
    isAuthenticated: authState.isAuthenticated,
    isLoading,
    user: authState.user,
    token: authState.token,
    login,
    loginWithApiKey,
    logout,
    refreshToken,
    error
  };
}