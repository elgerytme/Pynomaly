/**
 * Pynomaly JavaScript SDK
 * Main entry point for the SDK
 */

// Setup polyfills for cross-platform compatibility
// import { setupPolyfills } from './utils/polyfills';
// setupPolyfills();

// Core exports
import { PynomalyClient } from './core/client';
export { PynomalyClient };
export { AsyncAPI } from './core/async-api';
export { PynomalyWebSocket, WebSocketManager, createPynomalyWebSocket } from './core/websocket';

// Authentication exports
export { AuthManager, createAuthManager, SessionStorage } from './auth/auth-manager';
export { SecurityUtils } from './auth/security-utils';
export { TokenValidator } from './auth/token-validator';

// Utility exports
export { Environment } from './utils/environment';
export { StorageFactory, TimerUtils, CryptoUtils, HTTPUtils, EventUtils, CompatibilityChecker } from './utils/compatibility';
// export { setupPolyfills, checkPolyfillsNeeded } from './utils/polyfills';

// Framework integration exports
export * as Frameworks from './frameworks';

// Type exports
export type * from './types';

// Auth type exports
export type {
  AuthState,
  SessionConfig,
  AuthCredentials,
  AuthEventMap
} from './auth/auth-manager';

export type {
  SecurityPolicy,
  SecurityOptions,
  PasswordStrengthResult,
  SecurityAuditResult
} from './auth/security-utils';

export type {
  TokenValidationResult,
  TokenValidationOptions
} from './auth/token-validator';

export type {
  WebSocketConfig
} from './core/websocket';

// Version
export const VERSION = '1.0.0';

// Default export: main client class
export default PynomalyClient;