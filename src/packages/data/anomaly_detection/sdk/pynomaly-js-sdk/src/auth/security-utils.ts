/**
 * Security utilities for authentication and session management
 */

import { AuthToken } from '../types';
import { CryptoUtils } from '../utils/compatibility';

export interface SecurityPolicy {
  passwordMinLength: number;
  passwordRequireUppercase: boolean;
  passwordRequireLowercase: boolean;
  passwordRequireNumbers: boolean;
  passwordRequireSpecialChars: boolean;
  sessionTimeout: number; // minutes
  maxLoginAttempts: number;
  lockoutDuration: number; // minutes
  enforceHttps: boolean;
  enableCsrfProtection: boolean;
}

export interface SecurityOptions {
  enableAutoLock: boolean;
  enableActivityTracking: boolean;
  enableSecurityAudit: boolean;
  allowRememberMe: boolean;
  secureCookies: boolean;
}

export interface PasswordStrengthResult {
  score: number; // 0-100
  strength: 'weak' | 'fair' | 'good' | 'strong';
  feedback: string[];
  meets: {
    length: boolean;
    uppercase: boolean;
    lowercase: boolean;
    numbers: boolean;
    specialChars: boolean;
  };
}

export interface SecurityAuditResult {
  timestamp: Date;
  event: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  details: Record<string, any>;
  userAgent?: string;
  ipAddress?: string;
  sessionId?: string;
}

export class SecurityUtils {
  private static readonly DEFAULT_POLICY: SecurityPolicy = {
    passwordMinLength: 8,
    passwordRequireUppercase: true,
    passwordRequireLowercase: true,
    passwordRequireNumbers: true,
    passwordRequireSpecialChars: true,
    sessionTimeout: 60, // 1 hour
    maxLoginAttempts: 5,
    lockoutDuration: 15, // 15 minutes
    enforceHttps: true,
    enableCsrfProtection: true
  };

  private static readonly COMMON_PASSWORDS = [
    'password', '123456', 'password123', 'admin', 'letmein',
    'welcome', 'monkey', '1234567890', 'qwerty', 'abc123'
  ];

  private static policy: SecurityPolicy = SecurityUtils.DEFAULT_POLICY;
  private static auditLog: SecurityAuditResult[] = [];

  // Policy management
  static setSecurityPolicy(policy: Partial<SecurityPolicy>): void {
    SecurityUtils.policy = { ...SecurityUtils.DEFAULT_POLICY, ...policy };
  }

  static getSecurityPolicy(): SecurityPolicy {
    return { ...SecurityUtils.policy };
  }

  // Password validation
  static validatePassword(password: string): PasswordStrengthResult {
    const feedback: string[] = [];
    const meets = {
      length: password.length >= SecurityUtils.policy.passwordMinLength,
      uppercase: /[A-Z]/.test(password),
      lowercase: /[a-z]/.test(password),
      numbers: /[0-9]/.test(password),
      specialChars: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)
    };

    let score = 0;
    
    // Length check
    if (meets.length) {
      score += 20;
    } else {
      feedback.push(`Password must be at least ${SecurityUtils.policy.passwordMinLength} characters long`);
    }

    // Character type checks
    if (SecurityUtils.policy.passwordRequireUppercase) {
      if (meets.uppercase) {
        score += 15;
      } else {
        feedback.push('Password must contain at least one uppercase letter');
      }
    }

    if (SecurityUtils.policy.passwordRequireLowercase) {
      if (meets.lowercase) {
        score += 15;
      } else {
        feedback.push('Password must contain at least one lowercase letter');
      }
    }

    if (SecurityUtils.policy.passwordRequireNumbers) {
      if (meets.numbers) {
        score += 15;
      } else {
        feedback.push('Password must contain at least one number');
      }
    }

    if (SecurityUtils.policy.passwordRequireSpecialChars) {
      if (meets.specialChars) {
        score += 15;
      } else {
        feedback.push('Password must contain at least one special character');
      }
    }

    // Additional security checks
    if (SecurityUtils.COMMON_PASSWORDS.includes(password.toLowerCase())) {
      score -= 30;
      feedback.push('Password is too common');
    }

    if (password.length > 12) {
      score += 10;
    }

    if (/(.)\1{2,}/.test(password)) {
      score -= 10;
      feedback.push('Password contains repeated characters');
    }

    if (/^\d+$/.test(password)) {
      score -= 20;
      feedback.push('Password cannot be only numbers');
    }

    // Ensure score is within bounds
    score = Math.max(0, Math.min(100, score));

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
      score,
      strength,
      feedback,
      meets
    };
  }

  // Token validation
  static validateToken(token: AuthToken): boolean {
    if (!token || !token.token) {
      return false;
    }

    // Check expiration
    if (token.expiresAt && new Date() >= token.expiresAt) {
      return false;
    }

    // Basic token format validation
    if (token.tokenType === 'Bearer') {
      // JWT tokens typically have 3 parts separated by dots
      const parts = token.token.split('.');
      if (parts.length !== 3) {
        return false;
      }
    }

    return true;
  }

  // Session security
  static generateSecureSessionId(): string {
    return CryptoUtils.generateSecureRandomString(32);
  }

  // Security headers
  static getSecurityHeaders(): Record<string, string> {
    return {
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'X-XSS-Protection': '1; mode=block',
      'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
      'Referrer-Policy': 'strict-origin-when-cross-origin',
      'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
    };
  }

  // Input sanitization
  static sanitizeInput(input: string): string {
    return input
      .replace(/[<>]/g, '')
      .replace(/javascript:/gi, '')
      .replace(/data:/gi, '')
      .trim();
  }

  static sanitizeUrl(url: string): string {
    try {
      const parsed = new URL(url);
      
      // Only allow http and https protocols
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        throw new Error('Invalid protocol');
      }

      return parsed.toString();
    } catch (error) {
      throw new Error('Invalid URL');
    }
  }

  // Rate limiting helpers
  static createRateLimiter(maxRequests: number, windowMs: number) {
    const requests = new Map<string, number[]>();

    return {
      isAllowed(identifier: string): boolean {
        const now = Date.now();
        const userRequests = requests.get(identifier) || [];
        
        // Remove old requests outside the window
        const validRequests = userRequests.filter(time => now - time < windowMs);
        
        if (validRequests.length >= maxRequests) {
          return false;
        }

        validRequests.push(now);
        requests.set(identifier, validRequests);
        
        return true;
      },

      getRemainingRequests(identifier: string): number {
        const now = Date.now();
        const userRequests = requests.get(identifier) || [];
        const validRequests = userRequests.filter(time => now - time < windowMs);
        
        return Math.max(0, maxRequests - validRequests.length);
      },

      getResetTime(identifier: string): number {
        const userRequests = requests.get(identifier) || [];
        if (userRequests.length === 0) {
          return 0;
        }
        
        const oldestRequest = Math.min(...userRequests);
        return oldestRequest + windowMs;
      }
    };
  }

  // Security audit logging
  static logSecurityEvent(event: string, severity: 'low' | 'medium' | 'high' | 'critical', details: Record<string, any>): void {
    const auditEntry: SecurityAuditResult = {
      timestamp: new Date(),
      event,
      severity,
      details,
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : undefined,
      sessionId: SecurityUtils.generateSecureSessionId()
    };

    SecurityUtils.auditLog.push(auditEntry);
    
    // Keep only last 1000 entries
    if (SecurityUtils.auditLog.length > 1000) {
      SecurityUtils.auditLog.shift();
    }

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`[SECURITY] ${severity.toUpperCase()}: ${event}`, details);
    }
  }

  static getSecurityAuditLog(): SecurityAuditResult[] {
    return [...SecurityUtils.auditLog];
  }

  static clearSecurityAuditLog(): void {
    SecurityUtils.auditLog.length = 0;
  }

  // CSRF protection
  static generateCsrfToken(): string {
    return SecurityUtils.generateSecureSessionId();
  }

  static validateCsrfToken(token: string, expectedToken: string): boolean {
    return token === expectedToken;
  }

  // Secure storage helpers
  static encryptData(data: string, key: string): string {
    // Simple XOR encryption for demonstration
    // In production, use proper encryption libraries
    let encrypted = '';
    for (let i = 0; i < data.length; i++) {
      encrypted += String.fromCharCode(data.charCodeAt(i) ^ key.charCodeAt(i % key.length));
    }
    return btoa(encrypted);
  }

  static decryptData(encryptedData: string, key: string): string {
    try {
      const encrypted = atob(encryptedData);
      let decrypted = '';
      for (let i = 0; i < encrypted.length; i++) {
        decrypted += String.fromCharCode(encrypted.charCodeAt(i) ^ key.charCodeAt(i % key.length));
      }
      return decrypted;
    } catch (error) {
      throw new Error('Failed to decrypt data');
    }
  }

  // Security checks
  static isSecureContext(): boolean {
    if (typeof window !== 'undefined') {
      return window.isSecureContext || window.location.protocol === 'https:';
    }
    return true; // Assume secure in non-browser environments
  }

  static detectSuspiciousActivity(userAgent: string, ipAddress?: string): boolean {
    // Basic bot detection
    const botPatterns = [
      /bot/i, /crawler/i, /spider/i, /scraper/i,
      /curl/i, /wget/i, /postman/i
    ];

    for (const pattern of botPatterns) {
      if (pattern.test(userAgent)) {
        return true;
      }
    }

    // Check for suspicious IP patterns (this is a simplified example)
    if (ipAddress) {
      // Block common suspicious IP ranges
      const suspiciousRanges = [
        '127.0.0.1', 'localhost', '0.0.0.0'
      ];

      if (suspiciousRanges.includes(ipAddress)) {
        return true;
      }
    }

    return false;
  }

  // Password generation
  static generateSecurePassword(length: number = 16): string {
    const uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const lowercase = 'abcdefghijklmnopqrstuvwxyz';
    const numbers = '0123456789';
    const specialChars = '!@#$%^&*()_+-=[]{}|;:,.<>?';
    
    const allChars = uppercase + lowercase + numbers + specialChars;
    let password = '';

    // Ensure at least one character from each category
    password += uppercase[Math.floor(Math.random() * uppercase.length)];
    password += lowercase[Math.floor(Math.random() * lowercase.length)];
    password += numbers[Math.floor(Math.random() * numbers.length)];
    password += specialChars[Math.floor(Math.random() * specialChars.length)];

    // Fill the rest randomly
    for (let i = 4; i < length; i++) {
      password += allChars[Math.floor(Math.random() * allChars.length)];
    }

    // Shuffle the password
    return password.split('').sort(() => Math.random() - 0.5).join('');
  }

  // Clean up sensitive data
  static secureClear(obj: any): void {
    if (typeof obj === 'object' && obj !== null) {
      for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
          if (typeof obj[key] === 'string') {
            obj[key] = '';
          } else if (typeof obj[key] === 'object') {
            SecurityUtils.secureClear(obj[key]);
          }
        }
      }
    }
  }
}