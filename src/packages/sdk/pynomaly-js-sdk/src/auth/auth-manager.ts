/**
 * Authentication and session management utilities
 */

import { EventEmitter } from 'eventemitter3';
import {
  AuthToken,
  User,
  PynomalyConfig,
  EventMap,
  EventCallback
} from '../types';
import { StorageFactory, UniversalStorage } from '../utils/compatibility';
import { Environment } from '../utils/environment';

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: AuthToken | null;
  expiresAt: Date | null;
}

export interface SessionConfig {
  storageKey?: string;
  enablePersistence?: boolean;
  autoRefresh?: boolean;
  refreshThreshold?: number; // minutes before expiry
}

export interface AuthCredentials {
  email: string;
  password: string;
}

export interface AuthEventMap extends EventMap {
  'auth:login': { user: User; token: AuthToken };
  'auth:logout': void;
  'auth:refresh': AuthToken;
  'auth:expired': void;
  'auth:error': { error: string; type: 'login' | 'refresh' | 'logout' };
}

export class AuthManager extends EventEmitter<AuthEventMap> {
  private config: SessionConfig;
  private state: AuthState;
  private refreshTimer: NodeJS.Timeout | null = null;
  private storage: UniversalStorage;

  constructor(config: SessionConfig = {}) {
    super();
    
    this.config = {
      storageKey: 'pynomaly_auth',
      enablePersistence: true,
      autoRefresh: true,
      refreshThreshold: 5,
      ...config
    };

    this.state = {
      isAuthenticated: false,
      user: null,
      token: null,
      expiresAt: null
    };

    // Initialize storage
    this.initializeStorage();
    
    // Load persisted session
    this.loadSession();
  }

  private initializeStorage(): void {
    try {
      if (Environment.isBrowser()) {
        this.storage = StorageFactory.getStorage('localStorage');
      } else if (Environment.isNode()) {
        this.storage = StorageFactory.getStorage('filesystem');
      } else {
        this.storage = StorageFactory.getStorage('memory');
      }
    } catch (error) {
      console.warn('Storage initialization failed, using memory storage:', error);
      this.storage = StorageFactory.getStorage('memory');
    }
  }

  private loadSession(): void {
    if (!this.config.enablePersistence) {
      return;
    }

    try {
      const stored = this.storage.getItem(this.config.storageKey!);
      if (stored) {
        const sessionData = JSON.parse(stored);
        
        // Validate session data
        if (sessionData.token && sessionData.user && sessionData.expiresAt) {
          const expiresAt = new Date(sessionData.expiresAt);
          
          if (expiresAt > new Date()) {
            this.updateState({
              isAuthenticated: true,
              user: sessionData.user,
              token: {
                ...sessionData.token,
                expiresAt
              },
              expiresAt
            });
            
            this.scheduleRefresh();
          } else {
            this.clearSession();
          }
        }
      }
    } catch (error) {
      console.warn('Failed to load session:', error);
      this.clearSession();
    }
  }

  private saveSession(): void {
    if (!this.config.enablePersistence || !this.state.isAuthenticated) {
      return;
    }

    try {
      const sessionData = {
        user: this.state.user,
        token: this.state.token,
        expiresAt: this.state.expiresAt?.toISOString()
      };

      this.storage.setItem(this.config.storageKey!, JSON.stringify(sessionData));
    } catch (error) {
      console.warn('Failed to save session:', error);
    }
  }

  private clearSession(): void {
    try {
      this.storage.removeItem(this.config.storageKey!);
    } catch (error) {
      console.warn('Failed to clear session:', error);
    }

    this.updateState({
      isAuthenticated: false,
      user: null,
      token: null,
      expiresAt: null
    });
  }

  private updateState(newState: Partial<AuthState>): void {
    this.state = { ...this.state, ...newState };
  }

  private scheduleRefresh(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }

    if (!this.config.autoRefresh || !this.state.token || !this.state.expiresAt) {
      return;
    }

    const now = new Date();
    const expiresAt = this.state.expiresAt;
    const refreshTime = new Date(expiresAt.getTime() - (this.config.refreshThreshold! * 60 * 1000));

    if (refreshTime > now) {
      const timeToRefresh = refreshTime.getTime() - now.getTime();
      
      this.refreshTimer = setTimeout(() => {
        this.refreshToken();
      }, timeToRefresh);
    }
  }

  // Public methods
  async login(credentials: AuthCredentials, client: any): Promise<User> {
    try {
      const user = await client.authenticate(credentials);
      const token = client.getAuthToken();
      
      if (!token) {
        throw new Error('No authentication token received');
      }

      this.updateState({
        isAuthenticated: true,
        user,
        token,
        expiresAt: token.expiresAt
      });

      this.saveSession();
      this.scheduleRefresh();

      this.emit('auth:login', { user, token });
      return user;
    } catch (error) {
      this.emit('auth:error', { error: error.message, type: 'login' });
      throw error;
    }
  }

  async loginWithApiKey(apiKey: string, client: any): Promise<User> {
    try {
      const user = await client.authenticateWithApiKey(apiKey);
      
      const token: AuthToken = {
        token: apiKey,
        tokenType: 'API-Key',
        expiresAt: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000) // 1 year
      };

      this.updateState({
        isAuthenticated: true,
        user,
        token,
        expiresAt: token.expiresAt
      });

      this.saveSession();

      this.emit('auth:login', { user, token });
      return user;
    } catch (error) {
      this.emit('auth:error', { error: error.message, type: 'login' });
      throw error;
    }
  }

  async logout(client: any): Promise<void> {
    try {
      if (this.state.isAuthenticated) {
        await client.logout();
      }
    } catch (error) {
      this.emit('auth:error', { error: error.message, type: 'logout' });
    } finally {
      this.clearSession();
      
      if (this.refreshTimer) {
        clearTimeout(this.refreshTimer);
        this.refreshTimer = null;
      }

      this.emit('auth:logout');
    }
  }

  async refreshToken(client?: any): Promise<AuthToken> {
    if (!this.state.token || !client) {
      throw new Error('No token or client available for refresh');
    }

    try {
      await client.refreshToken();
      const newToken = client.getAuthToken();
      
      if (!newToken) {
        throw new Error('Failed to refresh token');
      }

      this.updateState({
        token: newToken,
        expiresAt: newToken.expiresAt
      });

      this.saveSession();
      this.scheduleRefresh();

      this.emit('auth:refresh', newToken);
      return newToken;
    } catch (error) {
      this.emit('auth:error', { error: error.message, type: 'refresh' });
      this.emit('auth:expired');
      this.clearSession();
      throw error;
    }
  }

  // Getters
  getAuthState(): AuthState {
    return { ...this.state };
  }

  getUser(): User | null {
    return this.state.user;
  }

  getToken(): AuthToken | null {
    return this.state.token;
  }

  isAuthenticated(): boolean {
    return this.state.isAuthenticated && this.state.token !== null;
  }

  isTokenExpired(): boolean {
    if (!this.state.expiresAt) {
      return false;
    }
    return new Date() >= this.state.expiresAt;
  }

  getTimeUntilExpiry(): number {
    if (!this.state.expiresAt) {
      return 0;
    }
    return Math.max(0, this.state.expiresAt.getTime() - Date.now());
  }

  // Session management
  extendSession(additionalTime: number = 60 * 60 * 1000): void {
    if (this.state.token && this.state.expiresAt) {
      const newExpiresAt = new Date(this.state.expiresAt.getTime() + additionalTime);
      
      this.updateState({
        token: {
          ...this.state.token,
          expiresAt: newExpiresAt
        },
        expiresAt: newExpiresAt
      });

      this.saveSession();
      this.scheduleRefresh();
    }
  }

  clearStoredSession(): void {
    this.clearSession();
  }

  // Event methods with type safety
  on<K extends keyof AuthEventMap>(event: K, callback: EventCallback<AuthEventMap[K]>): this {
    return super.on(event, callback);
  }

  off<K extends keyof AuthEventMap>(event: K, callback: EventCallback<AuthEventMap[K]>): this {
    return super.off(event, callback);
  }

  once<K extends keyof AuthEventMap>(event: K, callback: EventCallback<AuthEventMap[K]>): this {
    return super.once(event, callback);
  }

  emit<K extends keyof AuthEventMap>(event: K, ...args: Parameters<EventCallback<AuthEventMap[K]>>): boolean {
    return super.emit(event, ...args);
  }

  // Cleanup
  destroy(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }
    
    this.removeAllListeners();
    this.clearSession();
  }
}

// Factory function
export function createAuthManager(config?: SessionConfig): AuthManager {
  return new AuthManager(config);
}

// Session storage utilities
export class SessionStorage {
  private storageKey: string;
  private storage: Storage | null = null;

  constructor(storageKey: string = 'pynomaly_session') {
    this.storageKey = storageKey;
    this.initializeStorage();
  }

  private initializeStorage(): void {
    try {
      if (typeof window !== 'undefined' && window.sessionStorage) {
        this.storage = window.sessionStorage;
      } else if (typeof global !== 'undefined' && global.sessionStorage) {
        this.storage = global.sessionStorage;
      }
    } catch (error) {
      console.warn('Session storage not available:', error);
    }
  }

  set(key: string, value: any): void {
    if (!this.storage) return;

    try {
      const fullKey = `${this.storageKey}:${key}`;
      this.storage.setItem(fullKey, JSON.stringify(value));
    } catch (error) {
      console.warn('Failed to set session data:', error);
    }
  }

  get<T = any>(key: string): T | null {
    if (!this.storage) return null;

    try {
      const fullKey = `${this.storageKey}:${key}`;
      const item = this.storage.getItem(fullKey);
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.warn('Failed to get session data:', error);
      return null;
    }
  }

  remove(key: string): void {
    if (!this.storage) return;

    try {
      const fullKey = `${this.storageKey}:${key}`;
      this.storage.removeItem(fullKey);
    } catch (error) {
      console.warn('Failed to remove session data:', error);
    }
  }

  clear(): void {
    if (!this.storage) return;

    try {
      const keys = Object.keys(this.storage);
      const prefix = `${this.storageKey}:`;
      
      keys.forEach(key => {
        if (key.startsWith(prefix)) {
          this.storage!.removeItem(key);
        }
      });
    } catch (error) {
      console.warn('Failed to clear session data:', error);
    }
  }

  has(key: string): boolean {
    if (!this.storage) return false;

    try {
      const fullKey = `${this.storageKey}:${key}`;
      return this.storage.getItem(fullKey) !== null;
    } catch (error) {
      return false;
    }
  }
}