package anomaly_detection

import (
	"context"
	"sync"
	"time"
)

// RateLimiter implements a token bucket rate limiter
type RateLimiter struct {
	tokens     int
	maxTokens  int
	refillRate time.Duration
	lastRefill time.Time
	mutex      sync.Mutex
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(requestsPerMinute int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		tokens:     requestsPerMinute,
		maxTokens:  requestsPerMinute,
		refillRate: window / time.Duration(requestsPerMinute),
		lastRefill: time.Now(),
	}
}

// Wait waits for permission to make a request, respecting the rate limit
func (rl *RateLimiter) Wait(ctx context.Context) error {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	// Refill tokens based on elapsed time
	now := time.Now()
	elapsed := now.Sub(rl.lastRefill)
	tokensToAdd := int(elapsed / rl.refillRate)
	
	if tokensToAdd > 0 {
		rl.tokens += tokensToAdd
		if rl.tokens > rl.maxTokens {
			rl.tokens = rl.maxTokens
		}
		rl.lastRefill = now
	}

	// If we have tokens, consume one and proceed
	if rl.tokens > 0 {
		rl.tokens--
		return nil
	}

	// Calculate wait time for next token
	waitTime := rl.refillRate - (elapsed % rl.refillRate)
	
	// Release mutex while waiting
	rl.mutex.Unlock()
	
	select {
	case <-time.After(waitTime):
		// Re-acquire mutex and try again
		rl.mutex.Lock()
		rl.tokens = rl.maxTokens - 1 // Consume the token we just got
		return nil
	case <-ctx.Done():
		// Re-acquire mutex before returning
		rl.mutex.Lock()
		return ctx.Err()
	}
}

// TryWait attempts to acquire a token without blocking
func (rl *RateLimiter) TryWait() bool {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	// Refill tokens based on elapsed time
	now := time.Now()
	elapsed := now.Sub(rl.lastRefill)
	tokensToAdd := int(elapsed / rl.refillRate)
	
	if tokensToAdd > 0 {
		rl.tokens += tokensToAdd
		if rl.tokens > rl.maxTokens {
			rl.tokens = rl.maxTokens
		}
		rl.lastRefill = now
	}

	// If we have tokens, consume one and return true
	if rl.tokens > 0 {
		rl.tokens--
		return true
	}

	return false
}

// Reset resets the rate limiter to its initial state
func (rl *RateLimiter) Reset() {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	rl.tokens = rl.maxTokens
	rl.lastRefill = time.Now()
}

// GetStatus returns the current status of the rate limiter
func (rl *RateLimiter) GetStatus() (availableTokens int, nextRefillTime time.Time) {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	// Calculate when the next token will be available
	now := time.Now()
	elapsed := now.Sub(rl.lastRefill)
	timeToNextToken := rl.refillRate - (elapsed % rl.refillRate)
	
	return rl.tokens, now.Add(timeToNextToken)
}