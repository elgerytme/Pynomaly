"""
Comprehensive rate limiting middleware for API security and performance protection.
Provides multiple rate limiting strategies with Redis backend support.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any

import redis
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from monorepo.infrastructure.config import Settings
from monorepo.infrastructure.security.audit_logger import (
    AuditLevel,
    SecurityEventType,
    get_audit_logger,
)

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    SLIDING_LOG = "sliding_log"


class RateLimitScope(str, Enum):
    """Rate limit scopes."""

    GLOBAL = "global"
    IP = "ip"
    USER = "user"
    ENDPOINT = "endpoint"
    API_KEY = "api_key"


@dataclass
class RateLimit:
    """Rate limit configuration."""

    requests: int
    window: int  # seconds
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    scope: RateLimitScope = RateLimitScope.IP
    burst: int | None = None
    message: str | None = None


@dataclass
class RateLimitResult:
    """Rate limit check result."""

    allowed: bool
    limit: int
    remaining: int
    reset_time: int
    retry_after: int | None = None


class RateLimiter:
    """Rate limiter implementation with multiple strategies."""

    def __init__(self, redis_client: redis.Redis, key_prefix: str = "rate_limit"):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.audit_logger = get_audit_logger()

    def _get_key(
        self, scope: RateLimitScope, identifier: str, endpoint: str = None
    ) -> str:
        """Generate Redis key for rate limit tracking."""
        if scope == RateLimitScope.GLOBAL:
            return f"{self.key_prefix}:global"
        elif scope == RateLimitScope.IP:
            return f"{self.key_prefix}:ip:{identifier}"
        elif scope == RateLimitScope.USER:
            return f"{self.key_prefix}:user:{identifier}"
        elif scope == RateLimitScope.ENDPOINT:
            return f"{self.key_prefix}:endpoint:{endpoint}:{identifier}"
        elif scope == RateLimitScope.API_KEY:
            return f"{self.key_prefix}:api_key:{identifier}"
        else:
            return f"{self.key_prefix}:unknown:{identifier}"

    async def check_rate_limit(
        self, rate_limit: RateLimit, identifier: str, endpoint: str = None
    ) -> RateLimitResult:
        """Check if request is within rate limit."""
        key = self._get_key(rate_limit.scope, identifier, endpoint)

        if rate_limit.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window_check(key, rate_limit)
        elif rate_limit.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_check(key, rate_limit)
        elif rate_limit.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket_check(key, rate_limit)
        elif rate_limit.strategy == RateLimitStrategy.SLIDING_LOG:
            return await self._sliding_log_check(key, rate_limit)
        else:
            raise ValueError(f"Unknown rate limit strategy: {rate_limit.strategy}")

    async def _fixed_window_check(
        self, key: str, rate_limit: RateLimit
    ) -> RateLimitResult:
        """Fixed window rate limiting."""
        now = int(time.time())
        window_start = now - (now % rate_limit.window)
        window_key = f"{key}:fw:{window_start}"

        pipe = self.redis.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, rate_limit.window)
        results = pipe.execute()

        current_count = results[0]
        allowed = current_count <= rate_limit.requests
        remaining = max(0, rate_limit.requests - current_count)
        reset_time = window_start + rate_limit.window

        return RateLimitResult(
            allowed=allowed,
            limit=rate_limit.requests,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=reset_time - now if not allowed else None,
        )

    async def _sliding_window_check(
        self, key: str, rate_limit: RateLimit
    ) -> RateLimitResult:
        """Sliding window rate limiting."""
        now = int(time.time())
        window_start = now - rate_limit.window

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, rate_limit.window)
        results = pipe.execute()

        current_count = results[1]
        allowed = current_count < rate_limit.requests
        remaining = max(0, rate_limit.requests - current_count - 1)
        reset_time = now + rate_limit.window

        return RateLimitResult(
            allowed=allowed,
            limit=rate_limit.requests,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=rate_limit.window if not allowed else None,
        )

    async def _token_bucket_check(
        self, key: str, rate_limit: RateLimit
    ) -> RateLimitResult:
        """Token bucket rate limiting."""
        now = int(time.time())
        bucket_key = f"{key}:tb"

        # Get current bucket state
        bucket_data = self.redis.hmget(bucket_key, ["tokens", "last_refill"])
        current_tokens = (
            float(bucket_data[0]) if bucket_data[0] else rate_limit.requests
        )
        last_refill = float(bucket_data[1]) if bucket_data[1] else now

        # Calculate tokens to add
        refill_rate = rate_limit.requests / rate_limit.window
        time_passed = now - last_refill
        tokens_to_add = time_passed * refill_rate

        # Update bucket
        new_tokens = min(rate_limit.requests, current_tokens + tokens_to_add)

        if new_tokens >= 1:
            # Allow request
            new_tokens -= 1
            allowed = True
            remaining = int(new_tokens)
            retry_after = None
        else:
            # Deny request
            allowed = False
            remaining = 0
            retry_after = int((1 - new_tokens) / refill_rate)

        # Update Redis
        pipe = self.redis.pipeline()
        pipe.hset(bucket_key, mapping={"tokens": new_tokens, "last_refill": now})
        pipe.expire(bucket_key, rate_limit.window * 2)
        pipe.execute()

        return RateLimitResult(
            allowed=allowed,
            limit=rate_limit.requests,
            remaining=remaining,
            reset_time=now + rate_limit.window,
            retry_after=retry_after,
        )

    async def _sliding_log_check(
        self, key: str, rate_limit: RateLimit
    ) -> RateLimitResult:
        """Sliding log rate limiting (most accurate but memory intensive)."""
        now = int(time.time() * 1000)  # Use milliseconds for precision
        window_start = now - (rate_limit.window * 1000)

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, rate_limit.window)
        results = pipe.execute()

        current_count = results[1]
        allowed = current_count < rate_limit.requests
        remaining = max(0, rate_limit.requests - current_count - 1)

        # Calculate when the oldest request will expire
        oldest_requests = self.redis.zrange(key, 0, 0, withscores=True)
        if oldest_requests and not allowed:
            oldest_time = oldest_requests[0][1]
            retry_after = int((oldest_time + (rate_limit.window * 1000) - now) / 1000)
        else:
            retry_after = None

        return RateLimitResult(
            allowed=allowed,
            limit=rate_limit.requests,
            remaining=remaining,
            reset_time=int((now + (rate_limit.window * 1000)) / 1000),
            retry_after=retry_after,
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.redis_client = redis.from_url(settings.redis_url)
        self.rate_limiter = RateLimiter(self.redis_client)
        self.audit_logger = get_audit_logger()

        # Default rate limits
        self.default_limits = {
            "global": RateLimit(
                requests=10000,
                window=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.GLOBAL,
            ),
            "ip": RateLimit(
                requests=100,
                window=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.IP,
            ),
            "user": RateLimit(
                requests=200,
                window=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.USER,
            ),
            "auth": RateLimit(
                requests=10,
                window=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.IP,
                message="Too many authentication attempts",
            ),
            "admin": RateLimit(
                requests=50,
                window=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.USER,
                message="Too many admin requests",
            ),
            "api_key": RateLimit(
                requests=1000,
                window=60,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.API_KEY,
            ),
        }

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/v1/health/"]:
            return await call_next(request)

        # Get client identifier
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        api_key = self._get_api_key(request)
        endpoint = request.url.path

        # Determine applicable rate limits
        rate_limits = self._get_applicable_limits(request, endpoint)

        # Check each rate limit
        for limit_name, rate_limit in rate_limits.items():
            identifier = self._get_identifier(
                rate_limit.scope, client_ip, user_id, api_key
            )

            try:
                result = await self.rate_limiter.check_rate_limit(
                    rate_limit, identifier, endpoint
                )

                if not result.allowed:
                    # Log rate limit violation
                    self.audit_logger.log_security_event(
                        SecurityEventType.SECURITY_RATE_LIMIT_EXCEEDED,
                        f"Rate limit exceeded: {limit_name}",
                        level=AuditLevel.WARNING,
                        details={
                            "limit_name": limit_name,
                            "limit": result.limit,
                            "identifier": identifier,
                            "endpoint": endpoint,
                            "scope": rate_limit.scope.value,
                            "strategy": rate_limit.strategy.value,
                        },
                        risk_score=60,
                    )

                    # Return rate limit error
                    return self._create_rate_limit_response(
                        result,
                        rate_limit.message or f"Rate limit exceeded: {limit_name}",
                    )

                # Add rate limit headers to response
                response = await call_next(request)
                self._add_rate_limit_headers(response, result)
                return response

            except Exception as e:
                logger.error(f"Rate limit check failed: {str(e)}")
                # Continue without rate limiting on error
                return await call_next(request)

        # No rate limits triggered
        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct connection
        return request.client.host if request.client else "unknown"

    def _get_user_id(self, request: Request) -> str | None:
        """Get user ID from request."""
        # This would typically come from JWT token or session
        return getattr(request.state, "user_id", None)

    def _get_api_key(self, request: Request) -> str | None:
        """Get API key from request."""
        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        # Check for Bearer token that might be an API key
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token.startswith("pyn_"):  # Our API key format
                return token

        return None

    def _get_applicable_limits(
        self, request: Request, endpoint: str
    ) -> dict[str, RateLimit]:
        """Get applicable rate limits for request."""
        limits = {}

        # Global rate limit
        limits["global"] = self.default_limits["global"]

        # IP-based rate limit
        limits["ip"] = self.default_limits["ip"]

        # User-based rate limit (if authenticated)
        if hasattr(request.state, "user_id") and request.state.user_id:
            limits["user"] = self.default_limits["user"]

        # API key rate limit
        api_key = self._get_api_key(request)
        if api_key:
            limits["api_key"] = self.default_limits["api_key"]

        # Endpoint-specific rate limits
        if endpoint.startswith("/api/v1/auth/"):
            limits["auth"] = self.default_limits["auth"]
        elif endpoint.startswith("/api/v1/admin/"):
            limits["admin"] = self.default_limits["admin"]

        return limits

    def _get_identifier(
        self,
        scope: RateLimitScope,
        client_ip: str,
        user_id: str | None,
        api_key: str | None,
    ) -> str:
        """Get identifier for rate limit scope."""
        if scope == RateLimitScope.GLOBAL:
            return "global"
        elif scope == RateLimitScope.IP:
            return client_ip
        elif scope == RateLimitScope.USER:
            return user_id or client_ip
        elif scope == RateLimitScope.API_KEY:
            return api_key or client_ip
        else:
            return client_ip

    def _create_rate_limit_response(
        self, result: RateLimitResult, message: str
    ) -> JSONResponse:
        """Create rate limit error response."""
        headers = {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(result.reset_time),
        }

        if result.retry_after:
            headers["Retry-After"] = str(result.retry_after)

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": message,
                "limit": result.limit,
                "remaining": result.remaining,
                "reset_time": result.reset_time,
                "retry_after": result.retry_after,
            },
            headers=headers,
        )

    def _add_rate_limit_headers(self, response: Response, result: RateLimitResult):
        """Add rate limit headers to response."""
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(result.reset_time)


# Rate limiting decorators
def rate_limit(
    requests: int,
    window: int,
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
    scope: RateLimitScope = RateLimitScope.IP,
    message: str = None,
):
    """Decorator for applying rate limits to specific endpoints."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would integrate with the middleware
            # For now, it's a placeholder for future implementation
            return await func(*args, **kwargs)

        # Store rate limit config on function
        wrapper._rate_limit = RateLimit(
            requests=requests,
            window=window,
            strategy=strategy,
            scope=scope,
            message=message,
        )

        return wrapper

    return decorator


# Convenience decorators
def ip_rate_limit(requests: int, window: int = 60, message: str = None):
    """IP-based rate limiting decorator."""
    return rate_limit(
        requests=requests, window=window, scope=RateLimitScope.IP, message=message
    )


def user_rate_limit(requests: int, window: int = 60, message: str = None):
    """User-based rate limiting decorator."""
    return rate_limit(
        requests=requests, window=window, scope=RateLimitScope.USER, message=message
    )


def strict_rate_limit(requests: int, window: int = 60, message: str = None):
    """Strict rate limiting for sensitive endpoints."""
    return rate_limit(
        requests=requests,
        window=window,
        strategy=RateLimitStrategy.SLIDING_LOG,
        scope=RateLimitScope.IP,
        message=message,
    )


# Rate limit configuration utilities
class RateLimitConfig:
    """Rate limit configuration manager."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.redis_client = redis.from_url(settings.redis_url)

    def get_rate_limits(self) -> dict[str, RateLimit]:
        """Get rate limit configuration."""
        return {
            "global": RateLimit(
                requests=int(self.settings.get("GLOBAL_RATE_LIMIT", 10000)),
                window=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.GLOBAL,
            ),
            "ip": RateLimit(
                requests=int(self.settings.get("IP_RATE_LIMIT", 100)),
                window=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.IP,
            ),
            "user": RateLimit(
                requests=int(self.settings.get("USER_RATE_LIMIT", 200)),
                window=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.USER,
            ),
            "auth": RateLimit(
                requests=int(self.settings.get("AUTH_RATE_LIMIT", 10)),
                window=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.IP,
                message="Too many authentication attempts",
            ),
        }

    def update_rate_limit(self, name: str, rate_limit: RateLimit):
        """Update rate limit configuration."""
        # This would typically update a configuration store
        pass

    def get_rate_limit_stats(self) -> dict[str, Any]:
        """Get rate limiting statistics."""
        # This would return usage statistics
        return {"total_requests": 0, "blocked_requests": 0, "rate_limit_violations": 0}
