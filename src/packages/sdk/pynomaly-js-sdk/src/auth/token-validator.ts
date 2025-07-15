/**
 * Token validation utilities for authentication
 */

import { AuthToken } from '../types';

export interface TokenValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  expiresIn: number; // milliseconds
  shouldRefresh: boolean;
}

export interface TokenValidationOptions {
  allowExpired: boolean;
  refreshThreshold: number; // minutes before expiry
  validateFormat: boolean;
  validateClaims: boolean;
  requiredClaims: string[];
}

export class TokenValidator {
  private static readonly JWT_REGEX = /^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$/;
  private static readonly API_KEY_REGEX = /^[A-Za-z0-9_-]{20,}$/;

  private static readonly DEFAULT_OPTIONS: TokenValidationOptions = {
    allowExpired: false,
    refreshThreshold: 5,
    validateFormat: true,
    validateClaims: false,
    requiredClaims: []
  };

  // Main validation method
  static validate(token: AuthToken, options?: Partial<TokenValidationOptions>): TokenValidationResult {
    const opts = { ...TokenValidator.DEFAULT_OPTIONS, ...options };
    const result: TokenValidationResult = {
      isValid: true,
      errors: [],
      warnings: [],
      expiresIn: 0,
      shouldRefresh: false
    };

    // Basic token presence check
    if (!token || !token.token) {
      result.isValid = false;
      result.errors.push('Token is missing or empty');
      return result;
    }

    // Format validation
    if (opts.validateFormat) {
      const formatResult = TokenValidator.validateFormat(token);
      if (!formatResult.isValid) {
        result.isValid = false;
        result.errors.push(...formatResult.errors);
        result.warnings.push(...formatResult.warnings);
      }
    }

    // Expiration validation
    const expirationResult = TokenValidator.validateExpiration(token, opts);
    result.expiresIn = expirationResult.expiresIn;
    result.shouldRefresh = expirationResult.shouldRefresh;

    if (!expirationResult.isValid) {
      if (!opts.allowExpired) {
        result.isValid = false;
        result.errors.push(...expirationResult.errors);
      } else {
        result.warnings.push(...expirationResult.errors);
      }
    }
    result.warnings.push(...expirationResult.warnings);

    // Claims validation (for JWT tokens)
    if (opts.validateClaims && token.tokenType === 'Bearer') {
      const claimsResult = TokenValidator.validateClaims(token, opts.requiredClaims);
      if (!claimsResult.isValid) {
        result.isValid = false;
        result.errors.push(...claimsResult.errors);
      }
      result.warnings.push(...claimsResult.warnings);
    }

    return result;
  }

  // Format validation
  private static validateFormat(token: AuthToken): TokenValidationResult {
    const result: TokenValidationResult = {
      isValid: true,
      errors: [],
      warnings: [],
      expiresIn: 0,
      shouldRefresh: false
    };

    switch (token.tokenType) {
      case 'Bearer':
        if (!TokenValidator.JWT_REGEX.test(token.token)) {
          result.isValid = false;
          result.errors.push('Invalid JWT format - must have three parts separated by dots');
        } else {
          // Additional JWT format checks
          const parts = token.token.split('.');
          try {
            // Validate header
            const header = JSON.parse(TokenValidator.base64UrlDecode(parts[0]));
            if (!header.alg || !header.typ) {
              result.warnings.push('JWT header missing required fields (alg, typ)');
            }

            // Validate payload structure
            const payload = JSON.parse(TokenValidator.base64UrlDecode(parts[1]));
            if (!payload.iat && !payload.exp) {
              result.warnings.push('JWT payload missing timestamp fields (iat, exp)');
            }
          } catch (error) {
            result.isValid = false;
            result.errors.push('Invalid JWT structure - cannot decode header or payload');
          }
        }
        break;

      case 'API-Key':
        if (!TokenValidator.API_KEY_REGEX.test(token.token)) {
          result.isValid = false;
          result.errors.push('Invalid API key format - must be at least 20 characters of alphanumeric, underscore, or dash');
        }
        break;

      default:
        result.isValid = false;
        result.errors.push(`Unknown token type: ${token.tokenType}`);
    }

    return result;
  }

  // Expiration validation
  private static validateExpiration(token: AuthToken, options: TokenValidationOptions): TokenValidationResult {
    const result: TokenValidationResult = {
      isValid: true,
      errors: [],
      warnings: [],
      expiresIn: 0,
      shouldRefresh: false
    };

    const now = new Date();
    
    if (token.expiresAt) {
      const expiresAt = new Date(token.expiresAt);
      result.expiresIn = expiresAt.getTime() - now.getTime();

      if (result.expiresIn <= 0) {
        result.isValid = false;
        result.errors.push('Token has expired');
      } else {
        // Check if token should be refreshed
        const refreshThresholdMs = options.refreshThreshold * 60 * 1000;
        if (result.expiresIn <= refreshThresholdMs) {
          result.shouldRefresh = true;
          result.warnings.push('Token expires soon and should be refreshed');
        }

        // Warn if token expires very soon
        if (result.expiresIn <= 60000) { // 1 minute
          result.warnings.push('Token expires in less than 1 minute');
        }
      }
    } else {
      result.warnings.push('Token does not have expiration time');
    }

    return result;
  }

  // Claims validation for JWT tokens
  private static validateClaims(token: AuthToken, requiredClaims: string[]): TokenValidationResult {
    const result: TokenValidationResult = {
      isValid: true,
      errors: [],
      warnings: [],
      expiresIn: 0,
      shouldRefresh: false
    };

    try {
      const parts = token.token.split('.');
      const payload = JSON.parse(TokenValidator.base64UrlDecode(parts[1]));

      // Check required claims
      for (const claim of requiredClaims) {
        if (!(claim in payload)) {
          result.isValid = false;
          result.errors.push(`Missing required claim: ${claim}`);
        }
      }

      // Standard JWT claims validation
      const now = Math.floor(Date.now() / 1000);

      // Check 'nbf' (not before) claim
      if (payload.nbf && payload.nbf > now) {
        result.isValid = false;
        result.errors.push('Token is not yet valid (nbf claim)');
      }

      // Check 'iat' (issued at) claim
      if (payload.iat && payload.iat > now) {
        result.warnings.push('Token issued in the future (iat claim)');
      }

      // Check 'exp' (expiration) claim
      if (payload.exp && payload.exp <= now) {
        result.isValid = false;
        result.errors.push('Token has expired (exp claim)');
      }

      // Check 'aud' (audience) claim if present
      if (payload.aud && typeof payload.aud === 'string') {
        // This would typically be validated against expected audience
        // For now, just warn if it's present
        result.warnings.push('Token has audience claim - ensure it matches expected audience');
      }

      // Check 'iss' (issuer) claim if present
      if (payload.iss && typeof payload.iss === 'string') {
        result.warnings.push('Token has issuer claim - ensure it matches expected issuer');
      }

    } catch (error) {
      result.isValid = false;
      result.errors.push('Failed to decode JWT payload for claims validation');
    }

    return result;
  }

  // Utility methods
  private static base64UrlDecode(str: string): string {
    // Replace URL-safe characters and add padding
    str = str.replace(/-/g, '+').replace(/_/g, '/');
    while (str.length % 4) {
      str += '=';
    }
    
    try {
      return atob(str);
    } catch (error) {
      throw new Error('Invalid base64url encoding');
    }
  }

  // Quick validation methods
  static isExpired(token: AuthToken): boolean {
    if (!token.expiresAt) return false;
    return new Date() >= new Date(token.expiresAt);
  }

  static getTimeUntilExpiry(token: AuthToken): number {
    if (!token.expiresAt) return Infinity;
    return Math.max(0, new Date(token.expiresAt).getTime() - Date.now());
  }

  static shouldRefresh(token: AuthToken, thresholdMinutes: number = 5): boolean {
    const timeUntilExpiry = TokenValidator.getTimeUntilExpiry(token);
    return timeUntilExpiry <= (thresholdMinutes * 60 * 1000);
  }

  static extractClaims(token: AuthToken): Record<string, any> | null {
    if (token.tokenType !== 'Bearer') return null;

    try {
      const parts = token.token.split('.');
      return JSON.parse(TokenValidator.base64UrlDecode(parts[1]));
    } catch (error) {
      return null;
    }
  }

  static getTokenInfo(token: AuthToken): {
    type: string;
    expiresAt: Date | null;
    isExpired: boolean;
    timeUntilExpiry: number;
    claims: Record<string, any> | null;
  } {
    return {
      type: token.tokenType,
      expiresAt: token.expiresAt ? new Date(token.expiresAt) : null,
      isExpired: TokenValidator.isExpired(token),
      timeUntilExpiry: TokenValidator.getTimeUntilExpiry(token),
      claims: TokenValidator.extractClaims(token)
    };
  }

  // Token strength analysis
  static analyzeTokenStrength(token: AuthToken): {
    strength: 'weak' | 'fair' | 'good' | 'strong';
    score: number;
    recommendations: string[];
  } {
    let score = 0;
    const recommendations: string[] = [];

    // Length check
    if (token.token.length >= 64) {
      score += 25;
    } else if (token.token.length >= 32) {
      score += 15;
    } else {
      recommendations.push('Token should be longer for better security');
    }

    // Character variety
    const hasLower = /[a-z]/.test(token.token);
    const hasUpper = /[A-Z]/.test(token.token);
    const hasNumbers = /[0-9]/.test(token.token);
    const hasSpecial = /[^a-zA-Z0-9]/.test(token.token);

    const varietyScore = [hasLower, hasUpper, hasNumbers, hasSpecial].filter(Boolean).length;
    score += varietyScore * 10;

    if (varietyScore < 3) {
      recommendations.push('Token should include more character variety');
    }

    // Expiration time
    if (token.expiresAt) {
      const timeUntilExpiry = TokenValidator.getTimeUntilExpiry(token);
      const hoursUntilExpiry = timeUntilExpiry / (1000 * 60 * 60);

      if (hoursUntilExpiry > 24) {
        score += 10;
      } else if (hoursUntilExpiry > 1) {
        score += 5;
      } else {
        recommendations.push('Token expires very soon');
      }
    } else {
      recommendations.push('Token should have an expiration time');
    }

    // Token type
    if (token.tokenType === 'Bearer') {
      score += 20; // JWT tokens are more secure
    } else if (token.tokenType === 'API-Key') {
      score += 10;
    }

    // Determine strength
    let strength: 'weak' | 'fair' | 'good' | 'strong';
    if (score >= 80) {
      strength = 'strong';
    } else if (score >= 60) {
      strength = 'good';
    } else if (score >= 40) {
      strength = 'fair';
    } else {
      strength = 'weak';
    }

    return {
      strength,
      score,
      recommendations
    };
  }
}