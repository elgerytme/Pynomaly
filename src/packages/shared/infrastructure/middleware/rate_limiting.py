"""Rate limiting middleware for API endpoints."""

import time
import asyncio
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit
        
        # Storage for rate limiting data
        self.minute_buckets: Dict[str, deque] = defaultdict(deque)
        self.hour_buckets: Dict[str, deque] = defaultdict(deque)
        self.burst_buckets: Dict[str, int] = defaultdict(int)
        self.last_reset: Dict[str, datetime] = defaultdict(lambda: datetime.now())
        
        logger.info(
            "Rate limiter initialized",
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            burst_limit=burst_limit
        )
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try to get real IP from headers (for proxy setups)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def _clean_old_requests(self, bucket: deque, window_seconds: int) -> None:
        """Remove old requests from bucket."""
        cutoff_time = time.time() - window_seconds
        while bucket and bucket[0] < cutoff_time:
            bucket.popleft()
    
    def _reset_burst_if_needed(self, client_id: str) -> None:
        """Reset burst counter if enough time has passed."""
        now = datetime.now()
        if now - self.last_reset[client_id] >= timedelta(minutes=1):
            self.burst_buckets[client_id] = 0
            self.last_reset[client_id] = now
    
    async def is_allowed(self, request: Request) -> Tuple[bool, Dict[str, str]]:
        """
        Check if request should be allowed.
        
        Returns:
            Tuple of (is_allowed, headers_dict)
        """
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        # Clean old requests
        self._clean_old_requests(self.minute_buckets[client_id], 60)
        self._clean_old_requests(self.hour_buckets[client_id], 3600)
        self._reset_burst_if_needed(client_id)
        
        # Get current counts
        minute_count = len(self.minute_buckets[client_id])
        hour_count = len(self.hour_buckets[client_id])
        burst_count = self.burst_buckets[client_id]
        
        # Prepare response headers
        headers = {
            "X-RateLimit-Limit-Minute": str(self.requests_per_minute),
            "X-RateLimit-Limit-Hour": str(self.requests_per_hour),
            "X-RateLimit-Limit-Burst": str(self.burst_limit),
            "X-RateLimit-Remaining-Minute": str(max(0, self.requests_per_minute - minute_count)),
            "X-RateLimit-Remaining-Hour": str(max(0, self.requests_per_hour - hour_count)),
            "X-RateLimit-Remaining-Burst": str(max(0, self.burst_limit - burst_count)),
            "X-RateLimit-Reset-Minute": str(int(current_time + 60)),
            "X-RateLimit-Reset-Hour": str(int(current_time + 3600))
        }
        
        # Check limits
        if minute_count >= self.requests_per_minute:
            logger.warning(
                "Rate limit exceeded - minute limit",
                client_id=client_id,
                minute_count=minute_count,
                limit=self.requests_per_minute
            )
            headers["Retry-After"] = "60"
            return False, headers
        
        if hour_count >= self.requests_per_hour:
            logger.warning(
                "Rate limit exceeded - hour limit",
                client_id=client_id,
                hour_count=hour_count,
                limit=self.requests_per_hour
            )
            headers["Retry-After"] = "3600"
            return False, headers
        
        if burst_count >= self.burst_limit:
            logger.warning(
                "Rate limit exceeded - burst limit",
                client_id=client_id,
                burst_count=burst_count,
                limit=self.burst_limit
            )
            headers["Retry-After"] = "60"
            return False, headers
        
        # Allow request and record it
        self.minute_buckets[client_id].append(current_time)
        self.hour_buckets[client_id].append(current_time)
        self.burst_buckets[client_id] += 1
        
        logger.debug(
            "Request allowed",
            client_id=client_id,
            minute_count=minute_count + 1,
            hour_count=hour_count + 1,
            burst_count=burst_count + 1
        )
        
        return True, headers


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10,
        exempt_paths: Optional[list] = None
    ):
        self.rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            burst_limit=burst_limit
        )
        self.exempt_paths = exempt_paths or ["/docs", "/redoc", "/openapi.json", "/health"]
        
        logger.info(
            "Rate limiting middleware initialized",
            exempt_paths=self.exempt_paths
        )
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Skip rate limiting for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Check rate limit
        is_allowed, headers = await self.rate_limiter.is_allowed(request)
        
        if not is_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "type": "rate_limit_error"
                },
                headers=headers
            )
        
        # Process request and add rate limit headers to response
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        for key, value in headers.items():
            response.headers[key] = value
        
        return response


# Factory function for easy setup
def create_rate_limit_middleware(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    burst_limit: int = 10,
    exempt_paths: Optional[list] = None
) -> RateLimitMiddleware:
    """Create rate limiting middleware with specified limits."""
    return RateLimitMiddleware(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        burst_limit=burst_limit,
        exempt_paths=exempt_paths
    )