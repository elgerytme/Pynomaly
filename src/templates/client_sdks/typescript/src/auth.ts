/**
 * Authentication manager for the anomaly_detection TypeScript SDK
 * Handles JWT tokens, API keys, and session management
 */

import { AuthToken, LoginCredentials, UserProfile } from './types';

export interface AuthManagerConfig {
  /** API key for authentication */
  apiKey?: string;
  /** JWT token storage key */
  tokenStorageKey?: string;
  /** Enable automatic token refresh */
  autoRefresh?: boolean;
  /** Token refresh threshold in seconds before expiry */
  refreshThreshold?: number;
}

/**
 * Manages authentication state and tokens for the anomaly_detection client
 */
export class AuthManager {
  private apiKey?: string;
  private jwtToken?: string;
  private refreshToken?: string;
  private tokenExpiresAt?: number;
  private readonly tokenStorageKey: string;
  private readonly autoRefresh: boolean;
  private readonly refreshThreshold: number;
  private refreshTimer?: NodeJS.Timeout;

  constructor(config: AuthManagerConfig = {}) {
    this.apiKey = config.apiKey;
    this.tokenStorageKey = config.tokenStorageKey || 'anomaly_detection_auth_token';
    this.autoRefresh = config.autoRefresh ?? true;
    this.refreshThreshold = config.refreshThreshold || 300; // 5 minutes

    // Load stored token if available
    this.loadStoredToken();
  }

  /**
   * Set API key for authentication
   */
  setApiKey(apiKey: string): void {
    this.apiKey = apiKey;
  }

  /**
   * Set JWT token for authentication
   */
  setJwtToken(token: string, refreshToken?: string, expiresAt?: number): void {
    this.jwtToken = token;
    this.refreshToken = refreshToken;
    this.tokenExpiresAt = expiresAt;

    // Store token for persistence
    this.storeToken();

    // Set up auto-refresh if enabled
    if (this.autoRefresh && expiresAt) {
      this.scheduleTokenRefresh();
    }
  }

  /**
   * Set authentication token from AuthToken response
   */
  setAuthToken(authToken: AuthToken): void {
    this.setJwtToken(
      authToken.accessToken,
      authToken.refreshToken,
      authToken.expiresAt
    );
  }

  /**
   * Clear authentication token
   */
  clearToken(): void {
    this.jwtToken = undefined;
    this.refreshToken = undefined;
    this.tokenExpiresAt = undefined;

    // Clear stored token
    this.clearStoredToken();

    // Cancel refresh timer
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = undefined;
    }
  }

  /**
   * Get authentication headers for requests
   */
  getAuthHeaders(): Record<string, string> {
    const headers: Record<string, string> = {};

    if (this.jwtToken) {
      headers['Authorization'] = `Bearer ${this.jwtToken}`;
    } else if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    return headers;
  }

  /**
   * Check if currently authenticated
   */
  isAuthenticated(): boolean {
    return !!(this.jwtToken || this.apiKey);
  }

  /**
   * Check if token is expired or about to expire
   */
  isTokenExpired(): boolean {
    if (!this.tokenExpiresAt) {
      return false;
    }

    const now = Date.now() / 1000;
    return now >= this.tokenExpiresAt;
  }

  /**
   * Check if token needs refresh
   */
  needsRefresh(): boolean {
    if (!this.tokenExpiresAt || !this.refreshToken) {
      return false;
    }

    const now = Date.now() / 1000;
    return now >= (this.tokenExpiresAt - this.refreshThreshold);
  }

  /**
   * Get current JWT token
   */
  getJwtToken(): string | undefined {
    return this.jwtToken;
  }

  /**
   * Get refresh token
   */
  getRefreshToken(): string | undefined {
    return this.refreshToken;
  }

  /**
   * Get token expiration timestamp
   */
  getTokenExpiresAt(): number | undefined {
    return this.tokenExpiresAt;
  }

  /**
   * Store token in localStorage (browser) or equivalent
   */
  private storeToken(): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      const tokenData = {
        accessToken: this.jwtToken,
        refreshToken: this.refreshToken,
        expiresAt: this.tokenExpiresAt,
      };
      localStorage.setItem(this.tokenStorageKey, JSON.stringify(tokenData));
    }
  }

  /**
   * Load stored token from localStorage
   */
  private loadStoredToken(): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      const storedData = localStorage.getItem(this.tokenStorageKey);
      if (storedData) {
        try {
          const tokenData = JSON.parse(storedData);
          if (tokenData.accessToken) {
            this.setJwtToken(
              tokenData.accessToken,
              tokenData.refreshToken,
              tokenData.expiresAt
            );
          }
        } catch (error) {
          console.warn('Failed to parse stored token:', error);
          this.clearStoredToken();
        }
      }
    }
  }

  /**
   * Clear stored token from localStorage
   */
  private clearStoredToken(): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.removeItem(this.tokenStorageKey);
    }
  }

  /**
   * Schedule automatic token refresh
   */
  private scheduleTokenRefresh(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }

    if (!this.tokenExpiresAt) {
      return;
    }

    const now = Date.now() / 1000;
    const refreshTime = this.tokenExpiresAt - this.refreshThreshold;
    const delay = Math.max(0, (refreshTime - now) * 1000);

    this.refreshTimer = setTimeout(() => {
      this.onTokenRefreshNeeded();
    }, delay);
  }

  /**
   * Called when token refresh is needed
   * Override this method to implement custom refresh logic
   */
  protected onTokenRefreshNeeded(): void {
    console.warn('Token refresh needed but no refresh handler configured');
  }

  /**
   * Set token refresh callback
   */
  setTokenRefreshCallback(callback: () => Promise<void>): void {
    this.onTokenRefreshNeeded = callback;
  }
}

/**
 * Session manager for handling user sessions
 */
export class SessionManager {
  private userProfile?: UserProfile;
  private sessionId?: string;
  private sessionStartTime: number;

  constructor() {
    this.sessionStartTime = Date.now();
  }

  /**
   * Start a new session
   */
  startSession(userProfile: UserProfile, sessionId?: string): void {
    this.userProfile = userProfile;
    this.sessionId = sessionId || this.generateSessionId();
    this.sessionStartTime = Date.now();
  }

  /**
   * End current session
   */
  endSession(): void {
    this.userProfile = undefined;
    this.sessionId = undefined;
  }

  /**
   * Get current user profile
   */
  getUserProfile(): UserProfile | undefined {
    return this.userProfile;
  }

  /**
   * Get current session ID
   */
  getSessionId(): string | undefined {
    return this.sessionId;
  }

  /**
   * Get session duration in seconds
   */
  getSessionDuration(): number {
    return Math.floor((Date.now() - this.sessionStartTime) / 1000);
  }

  /**
   * Check if user is logged in
   */
  isLoggedIn(): boolean {
    return !!this.userProfile;
  }

  /**
   * Generate a unique session ID
   */
  private generateSessionId(): string {
    return `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

/**
 * Multi-factor authentication helper
 */
export class MFAManager {
  /**
   * Validate MFA code format
   */
  static validateMfaCode(code: string): boolean {
    // Basic validation for 6-digit TOTP codes
    return /^\d{6}$/.test(code);
  }

  /**
   * Generate backup codes
   */
  static generateBackupCodes(count: number = 10): string[] {
    const codes: string[] = [];
    for (let i = 0; i < count; i++) {
      codes.push(this.generateBackupCode());
    }
    return codes;
  }

  /**
   * Generate a single backup code
   */
  private static generateBackupCode(): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let code = '';
    for (let i = 0; i < 8; i++) {
      code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
  }
}

/**
 * OAuth helper for third-party authentication
 */
export class OAuthManager {
  private clientId: string;
  private redirectUri: string;
  private scopes: string[];

  constructor(clientId: string, redirectUri: string, scopes: string[] = []) {
    this.clientId = clientId;
    this.redirectUri = redirectUri;
    this.scopes = scopes;
  }

  /**
   * Generate OAuth authorization URL
   */
  getAuthorizationUrl(provider: string, state?: string): string {
    const params = new URLSearchParams({
      client_id: this.clientId,
      redirect_uri: this.redirectUri,
      scope: this.scopes.join(' '),
      response_type: 'code',
      ...(state && { state }),
    });

    return `https://auth.anomaly_detection.com/oauth/${provider}/authorize?${params}`;
  }

  /**
   * Parse authorization callback URL
   */
  parseCallback(url: string): { code?: string; state?: string; error?: string } {
    const urlObj = new URL(url);
    const params = urlObj.searchParams;

    return {
      code: params.get('code') || undefined,
      state: params.get('state') || undefined,
      error: params.get('error') || undefined,
    };
  }
}