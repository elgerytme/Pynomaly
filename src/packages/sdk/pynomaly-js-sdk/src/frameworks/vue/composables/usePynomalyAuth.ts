/**
 * Vue 3 composable for Pynomaly authentication
 */

import { ref, reactive, onUnmounted } from 'vue';
import { AuthManager, AuthState, AuthCredentials, SessionConfig } from '../../../index';
import { PynomalyClient } from '../../../core/client';

export interface UsePynomalyAuthOptions extends Partial<SessionConfig> {
  client?: PynomalyClient;
}

export function usePynomalyAuth(options: UsePynomalyAuthOptions = {}) {
  const authManager = ref<AuthManager | null>(null);
  const authState = reactive<AuthState>({
    isAuthenticated: false,
    user: null,
    token: null,
    expiresAt: null
  });
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  // Initialize auth manager
  const manager = new AuthManager({
    enablePersistence: true,
    autoRefresh: true,
    ...options
  });

  // Set up event listeners
  manager.on('auth:login', ({ user, token }) => {
    Object.assign(authState, manager.getAuthState());
    error.value = null;
  });

  manager.on('auth:logout', () => {
    Object.assign(authState, manager.getAuthState());
    error.value = null;
  });

  manager.on('auth:refresh', (token) => {
    Object.assign(authState, manager.getAuthState());
  });

  manager.on('auth:error', ({ error: authError, type }) => {
    error.value = authError;
  });

  manager.on('auth:expired', () => {
    Object.assign(authState, manager.getAuthState());
    error.value = 'Session expired';
  });

  authManager.value = manager;
  Object.assign(authState, manager.getAuthState());

  const login = async (credentials: AuthCredentials) => {
    if (!authManager.value || !options.client) {
      throw new Error('Auth manager or client not available');
    }

    isLoading.value = true;
    error.value = null;

    try {
      await authManager.value.login(credentials, options.client);
    } catch (err) {
      const loginError = err as Error;
      error.value = loginError.message;
      throw loginError;
    } finally {
      isLoading.value = false;
    }
  };

  const loginWithApiKey = async (apiKey: string) => {
    if (!authManager.value || !options.client) {
      throw new Error('Auth manager or client not available');
    }

    isLoading.value = true;
    error.value = null;

    try {
      await authManager.value.loginWithApiKey(apiKey, options.client);
    } catch (err) {
      const loginError = err as Error;
      error.value = loginError.message;
      throw loginError;
    } finally {
      isLoading.value = false;
    }
  };

  const logout = async () => {
    if (!authManager.value || !options.client) {
      return;
    }

    isLoading.value = true;
    error.value = null;

    try {
      await authManager.value.logout(options.client);
    } catch (err) {
      const logoutError = err as Error;
      error.value = logoutError.message;
    } finally {
      isLoading.value = false;
    }
  };

  const refreshToken = async () => {
    if (!authManager.value || !options.client) {
      throw new Error('Auth manager or client not available');
    }

    isLoading.value = true;
    error.value = null;

    try {
      await authManager.value.refreshToken(options.client);
    } catch (err) {
      const refreshError = err as Error;
      error.value = refreshError.message;
      throw refreshError;
    } finally {
      isLoading.value = false;
    }
  };

  // Cleanup on unmount
  onUnmounted(() => {
    if (authManager.value) {
      authManager.value.destroy();
    }
  });

  return {
    authManager,
    authState,
    isLoading,
    error,
    login,
    loginWithApiKey,
    logout,
    refreshToken
  };
}