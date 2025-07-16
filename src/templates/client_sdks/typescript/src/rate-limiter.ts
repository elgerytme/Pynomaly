/**
 * Rate limiting implementation for the Pynomaly TypeScript SDK
 * Provides client-side rate limiting with token bucket algorithm
 */

import { RateLimitError } from './errors';

export interface RateLimiterConfig {
  /** Maximum number of requests allowed */
  maxRequests: number;
  /** Time window in milliseconds */
  windowMs: number;
  /** Enable adaptive rate limiting */
  adaptive?: boolean;
  /** Burst allowance multiplier */
  burstMultiplier?: number;
}

/**
 * Token bucket rate limiter implementation
 */
export class RateLimiter {
  private readonly maxTokens: number;
  private readonly refillRate: number; // tokens per second
  private readonly burstMultiplier: number;
  private tokens: number;
  private lastRefill: number;
  private readonly adaptive: boolean;
  private readonly requestHistory: number[] = [];
  private adaptiveRate: number;

  constructor(config: RateLimiterConfig) {
    this.maxTokens = config.maxRequests;
    this.refillRate = config.maxRequests / (config.windowMs / 1000);
    this.burstMultiplier = config.burstMultiplier || 1.5;
    this.adaptive = config.adaptive || false;
    this.tokens = this.maxTokens;
    this.lastRefill = Date.now();
    this.adaptiveRate = this.refillRate;
  }

  /**
   * Attempt to consume a token for a request
   */
  async consumeToken(): Promise<void> {
    this.refillTokens();
    
    if (this.adaptive) {
      this.updateAdaptiveRate();
    }

    if (this.tokens < 1) {
      const waitTime = this.calculateWaitTime();
      throw new RateLimitError(
        `Rate limit exceeded. Wait ${Math.ceil(waitTime / 1000)} seconds.`,
        Math.ceil(waitTime / 1000),
        this.maxTokens,
        Math.floor(this.tokens),
        Date.now() + waitTime
      );
    }

    this.tokens -= 1;
    this.recordRequest();
  }

  /**
   * Check if a request can be made without consuming a token
   */
  canMakeRequest(): boolean {
    this.refillTokens();
    return this.tokens >= 1;
  }

  /**
   * Wait until a token is available
   */
  async waitIfNeeded(): Promise<void> {
    while (!this.canMakeRequest()) {
      const waitTime = this.calculateWaitTime();
      await this.sleep(Math.min(waitTime, 1000)); // Max wait of 1 second at a time
    }
    await this.consumeToken();
  }

  /**
   * Get current rate limiting status
   */
  getStatus(): {
    tokensRemaining: number;
    maxTokens: number;
    refillRate: number;
    nextRefillTime: number;
    adaptiveRate?: number;
  } {
    this.refillTokens();
    return {
      tokensRemaining: Math.floor(this.tokens),
      maxTokens: this.maxTokens,
      refillRate: this.refillRate,
      nextRefillTime: this.lastRefill + (1000 / this.refillRate),
      ...(this.adaptive && { adaptiveRate: this.adaptiveRate }),
    };
  }

  /**
   * Reset rate limiter state
   */
  reset(): void {
    this.tokens = this.maxTokens;
    this.lastRefill = Date.now();
    this.requestHistory.length = 0;
    this.adaptiveRate = this.refillRate;
  }

  /**
   * Refill tokens based on elapsed time
   */
  private refillTokens(): void {
    const now = Date.now();
    const timeSinceLastRefill = now - this.lastRefill;
    const tokensToAdd = (timeSinceLastRefill / 1000) * this.getCurrentRefillRate();
    
    this.tokens = Math.min(this.maxTokens * this.burstMultiplier, this.tokens + tokensToAdd);
    this.lastRefill = now;
  }

  /**
   * Calculate wait time until next token is available
   */
  private calculateWaitTime(): number {
    const tokensNeeded = 1 - this.tokens;
    return (tokensNeeded / this.getCurrentRefillRate()) * 1000;
  }

  /**
   * Get current refill rate (adaptive or fixed)
   */
  private getCurrentRefillRate(): number {
    return this.adaptive ? this.adaptiveRate : this.refillRate;
  }

  /**
   * Record request for adaptive rate limiting
   */
  private recordRequest(): void {
    if (this.adaptive) {
      const now = Date.now();
      this.requestHistory.push(now);
      
      // Keep only recent history (last 5 minutes)
      const cutoff = now - 5 * 60 * 1000;
      while (this.requestHistory.length > 0 && this.requestHistory[0] < cutoff) {
        this.requestHistory.shift();
      }
    }
  }

  /**
   * Update adaptive rate based on request patterns
   */
  private updateAdaptiveRate(): void {
    if (this.requestHistory.length < 10) {
      return; // Need sufficient history
    }

    const now = Date.now();
    const recentRequests = this.requestHistory.filter(time => now - time < 60000); // Last minute
    const currentRate = recentRequests.length / 60; // requests per second

    // Adjust adaptive rate based on current usage
    if (currentRate > this.refillRate * 0.8) {
      // High usage, reduce rate slightly
      this.adaptiveRate = Math.max(this.refillRate * 0.5, this.adaptiveRate * 0.9);
    } else if (currentRate < this.refillRate * 0.2) {
      // Low usage, increase rate slightly
      this.adaptiveRate = Math.min(this.refillRate * 1.5, this.adaptiveRate * 1.1);
    }
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Simple fixed window rate limiter
 */
export class FixedWindowRateLimiter {
  private readonly maxRequests: number;
  private readonly windowMs: number;
  private requests: number = 0;
  private windowStart: number;

  constructor(maxRequests: number, windowMs: number) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.windowStart = Date.now();
  }

  /**
   * Check if request is allowed
   */
  isAllowed(): boolean {
    this.resetWindowIfNeeded();
    return this.requests < this.maxRequests;
  }

  /**
   * Consume a request slot
   */
  consumeRequest(): boolean {
    if (this.isAllowed()) {
      this.requests++;
      return true;
    }
    return false;
  }

  /**
   * Get remaining requests in current window
   */
  getRemainingRequests(): number {
    this.resetWindowIfNeeded();
    return Math.max(0, this.maxRequests - this.requests);
  }

  /**
   * Get time until window resets
   */
  getTimeUntilReset(): number {
    return Math.max(0, this.windowStart + this.windowMs - Date.now());
  }

  /**
   * Reset window if expired
   */
  private resetWindowIfNeeded(): void {
    const now = Date.now();
    if (now >= this.windowStart + this.windowMs) {
      this.requests = 0;
      this.windowStart = now;
    }
  }
}

/**
 * Sliding window rate limiter with more precise tracking
 */
export class SlidingWindowRateLimiter {
  private readonly maxRequests: number;
  private readonly windowMs: number;
  private readonly requestTimes: number[] = [];

  constructor(maxRequests: number, windowMs: number) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
  }

  /**
   * Check if request is allowed
   */
  isAllowed(): boolean {
    this.cleanupOldRequests();
    return this.requestTimes.length < this.maxRequests;
  }

  /**
   * Record a new request
   */
  recordRequest(): boolean {
    if (this.isAllowed()) {
      this.requestTimes.push(Date.now());
      return true;
    }
    return false;
  }

  /**
   * Get current request count in window
   */
  getCurrentCount(): number {
    this.cleanupOldRequests();
    return this.requestTimes.length;
  }

  /**
   * Get remaining requests
   */
  getRemainingRequests(): number {
    return Math.max(0, this.maxRequests - this.getCurrentCount());
  }

  /**
   * Remove requests outside the current window
   */
  private cleanupOldRequests(): void {
    const now = Date.now();
    const cutoff = now - this.windowMs;
    
    while (this.requestTimes.length > 0 && this.requestTimes[0] < cutoff) {
      this.requestTimes.shift();
    }
  }
}

/**
 * Distributed rate limiter for coordinating across multiple instances
 */
export class DistributedRateLimiter {
  private readonly storage: RateLimiterStorage;
  private readonly key: string;
  private readonly maxRequests: number;
  private readonly windowMs: number;

  constructor(
    storage: RateLimiterStorage,
    key: string,
    maxRequests: number,
    windowMs: number
  ) {
    this.storage = storage;
    this.key = key;
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
  }

  /**
   * Check if request is allowed (distributed)
   */
  async isAllowed(): Promise<boolean> {
    const current = await this.storage.getCurrentCount(this.key, this.windowMs);
    return current < this.maxRequests;
  }

  /**
   * Record a request (distributed)
   */
  async recordRequest(): Promise<boolean> {
    const current = await this.storage.incrementAndGet(this.key, this.windowMs);
    return current <= this.maxRequests;
  }
}

/**
 * Interface for distributed rate limiter storage
 */
export interface RateLimiterStorage {
  getCurrentCount(key: string, windowMs: number): Promise<number>;
  incrementAndGet(key: string, windowMs: number): Promise<number>;
}

/**
 * In-memory storage implementation (for testing)
 */
export class InMemoryRateLimiterStorage implements RateLimiterStorage {
  private readonly data = new Map<string, { count: number; timestamp: number }>();

  async getCurrentCount(key: string, windowMs: number): Promise<number> {
    const entry = this.data.get(key);
    if (!entry || Date.now() - entry.timestamp > windowMs) {
      return 0;
    }
    return entry.count;
  }

  async incrementAndGet(key: string, windowMs: number): Promise<number> {
    const now = Date.now();
    const entry = this.data.get(key);
    
    if (!entry || now - entry.timestamp > windowMs) {
      this.data.set(key, { count: 1, timestamp: now });
      return 1;
    }
    
    entry.count++;
    return entry.count;
  }
}