"""Rate limiting and throttling middleware for API security."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


class RateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, max_tokens: int, refill_rate: float, window_seconds: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_tokens: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
            window_seconds: Time window for rate limiting
        """
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.window_seconds = window_seconds
        
        # Storage for token buckets {client_id: (tokens, last_refill)}
        self.buckets: Dict[str, Tuple[float, float]] = {}
        
        # Request counts for different time windows
        self.request_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str, tokens_requested: int = 1) -> Tuple[bool, Dict[str, any]]:
        """Check if request is allowed under rate limits.
        
        Args:
            client_id: Client identifier
            tokens_requested: Number of tokens requested
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        async with self._lock:
            now = time.time()
            
            # Get or create bucket
            if client_id not in self.buckets:
                self.buckets[client_id] = (self.max_tokens, now)
            
            tokens, last_refill = self.buckets[client_id]
            
            # Refill tokens based on elapsed time
            elapsed = now - last_refill
            tokens = min(self.max_tokens, tokens + elapsed * self.refill_rate)
            
            # Check if enough tokens available
            allowed = tokens >= tokens_requested
            
            if allowed:
                tokens -= tokens_requested
            
            # Update bucket
            self.buckets[client_id] = (tokens, now)
            
            # Update request counts
            current_window = int(now // self.window_seconds)
            self.request_counts[client_id][current_window] += 1
            
            # Clean old windows
            cutoff_window = current_window - 10  # Keep last 10 windows
            for window in list(self.request_counts[client_id].keys()):
                if window < cutoff_window:
                    del self.request_counts[client_id][window]
            
            # Calculate rate limit info
            requests_in_window = sum(
                count for window, count in self.request_counts[client_id].items()
                if window >= current_window - 1
            )
            
            rate_limit_info = {
                'allowed': allowed,
                'tokens_remaining': int(tokens),
                'max_tokens': self.max_tokens,
                'refill_rate': self.refill_rate,
                'requests_in_window': requests_in_window,
                'window_seconds': self.window_seconds,
                'retry_after': max(0, int((tokens_requested - tokens) / self.refill_rate)) if not allowed else 0
            }
            
            return allowed, rate_limit_info


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        """Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            rate_limiter: Rate limiter instance
        """
        self.app = app
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limits
        allowed, rate_info = await self.rate_limiter.is_allowed(client_id)
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Try again in {rate_info["retry_after"]} seconds.',
                    'rate_limit': rate_info
                },
                headers={
                    'X-RateLimit-Limit': str(rate_info['max_tokens']),
                    'X-RateLimit-Remaining': str(rate_info['tokens_remaining']),
                    'X-RateLimit-Reset': str(int(time.time() + rate_info['retry_after'])),
                    'Retry-After': str(rate_info['retry_after'])
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers['X-RateLimit-Limit'] = str(rate_info['max_tokens'])
        response.headers['X-RateLimit-Remaining'] = str(rate_info['tokens_remaining'])
        response.headers['X-RateLimit-Window'] = str(rate_info['window_seconds'])
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Client identifier
        """
        # Priority order for client identification:
        # 1. User ID from authentication
        # 2. API key
        # 3. IP address
        
        # Check for authenticated user
        if hasattr(request.state, 'user') and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Check for API key
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"api_key:{api_key[:8]}"  # Use first 8 chars for privacy
        
        # Fall back to IP address
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # Take first IP in case of proxy chain
            client_ip = forwarded_for.split(',')[0].strip()
        else:
            client_ip = request.client.host if request.client else 'unknown'
        
        return f"ip:{client_ip}"


class SecurityMiddleware:
    """General security middleware for various protections."""
    
    def __init__(self, app):
        """Initialize security middleware.
        
        Args:
            app: FastAPI application
        """
        self.app = app
        
        # Security headers to add
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        }
        
        # Blocked user agents (basic bot protection)
        self.blocked_user_agents = {
            'curl', 'wget', 'python-requests', 'bot', 'spider', 'crawler'
        }
    
    async def __call__(self, request: Request, call_next):
        """Process request with security checks.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Basic bot protection
        user_agent = request.headers.get('User-Agent', '').lower()
        if any(blocked in user_agent for blocked in self.blocked_user_agents):
            if not self._is_api_request(request):
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={'error': 'Access denied'}
                )
        
        # Content length check
        content_length = request.headers.get('Content-Length')
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={'error': 'Request too large'}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response
    
    def _is_api_request(self, request: Request) -> bool:
        """Check if request is an API request.
        
        Args:
            request: HTTP request
            
        Returns:
            True if API request
        """
        return (
            request.url.path.startswith('/api/') or
            'application/json' in request.headers.get('Accept', '') or
            request.headers.get('X-API-Key') is not None
        )


class DDoSProtection:
    """Simple DDoS protection middleware."""
    
    def __init__(self, app, max_requests_per_minute: int = 1000):
        """Initialize DDoS protection.
        
        Args:
            app: FastAPI application
            max_requests_per_minute: Maximum requests per minute per IP
        """
        self.app = app
        self.max_requests_per_minute = max_requests_per_minute
        
        # Request tracking {ip: {minute: count}}
        self.request_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Blocked IPs {ip: until_time}
        self.blocked_ips: Dict[str, float] = {}
        
        self._lock = asyncio.Lock()
    
    async def __call__(self, request: Request, call_next):
        """Process request with DDoS protection.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        client_ip = self._get_client_ip(request)
        
        async with self._lock:
            now = time.time()
            current_minute = int(now // 60)
            
            # Check if IP is blocked
            if client_ip in self.blocked_ips:
                if now < self.blocked_ips[client_ip]:
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={'error': 'IP temporarily blocked due to suspicious activity'}
                    )
                else:
                    # Unblock IP
                    del self.blocked_ips[client_ip]
            
            # Update request count
            self.request_counts[client_ip][current_minute] += 1
            
            # Check current minute requests
            current_requests = self.request_counts[client_ip][current_minute]
            
            if current_requests > self.max_requests_per_minute:
                # Block IP for 15 minutes
                self.blocked_ips[client_ip] = now + 900
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        'error': 'Too many requests from this IP',
                        'blocked_until': int(self.blocked_ips[client_ip])
                    }
                )
            
            # Clean old data
            cutoff_minute = current_minute - 5  # Keep last 5 minutes
            for ip in list(self.request_counts.keys()):
                for minute in list(self.request_counts[ip].keys()):
                    if minute < cutoff_minute:
                        del self.request_counts[ip][minute]
                
                # Remove empty IP entries
                if not self.request_counts[ip]:
                    del self.request_counts[ip]
        
        # Process request
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address.
        
        Args:
            request: HTTP request
            
        Returns:
            Client IP address
        """
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else 'unknown'


# Rate limiter configurations for different tiers
RATE_LIMITERS = {
    'free': RateLimiter(max_tokens=100, refill_rate=1.0, window_seconds=60),
    'premium': RateLimiter(max_tokens=1000, refill_rate=10.0, window_seconds=60),
    'enterprise': RateLimiter(max_tokens=10000, refill_rate=100.0, window_seconds=60),
    'api': RateLimiter(max_tokens=5000, refill_rate=50.0, window_seconds=60),
}


def get_rate_limiter_for_user(user_tier: str = 'free') -> RateLimiter:
    """Get rate limiter for user tier.
    
    Args:
        user_tier: User tier (free, premium, enterprise, api)
        
    Returns:
        Rate limiter instance
    """
    return RATE_LIMITERS.get(user_tier, RATE_LIMITERS['free'])