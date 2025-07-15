/**
 * Authentication module exports
 */

export { AuthManager, createAuthManager, SessionStorage } from './auth-manager';
export { SecurityUtils } from './security-utils';
export { TokenValidator } from './token-validator';

export type {
  AuthState,
  SessionConfig,
  AuthCredentials,
  AuthEventMap
} from './auth-manager';

export type {
  SecurityPolicy,
  SecurityOptions,
  PasswordStrengthResult,
  SecurityAuditResult
} from './security-utils';

export type {
  TokenValidationResult,
  TokenValidationOptions
} from './token-validator';