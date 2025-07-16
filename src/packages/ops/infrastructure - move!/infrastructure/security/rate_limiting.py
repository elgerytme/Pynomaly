"""Advanced rate limiting system with multiple algorithms and strategies."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from monorepo.shared.error_handling import ErrorContext, create_external_service_error

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""

    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_LOG = "sliding_window_log"
    EXPONENTIAL_BACKOFF = "exponential_backoff"


class RateLimitScope(Enum):
    """Rate limit scope levels."""

    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    API_KEY = "api_key"
    CUSTOM = "custom"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    requests_per_second: float = 10.0
    requests_per_minute: float = 600.0
    requests_per_hour: float = 36000.0
    burst_capacity: int = 100
    window_size: int = 60  # seconds
    cleanup_interval: int = 300  # seconds
    enabled: bool = True

    # Exponential backoff specific
    backoff_multiplier: float = 2.0
    max_backoff: int = 3600  # seconds

    # Custom limits for different operations
    custom_limits: dict[str, float] = field(default_factory=dict)

    def get_limit_for_operation(self, operation: str) -> float:
        """Get rate limit for specific operation."""
        return self.custom_limits.get(operation, self.requests_per_second)


@dataclass
class RateLimitStatus:
    """Rate limit status information."""

    allowed: bool
    remaining: int
    reset_time: float
    retry_after: int | None = None
    current_requests: int = 0
    window_start: float = 0.0
    algorithm: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitViolation:
    """Rate limit violation information."""

    identifier: str
    scope: RateLimitScope
    violation_time: float
    exceeded_limit: float
    current_rate: float
    algorithm: RateLimitAlgorithm
    context: dict[str, Any] = field(default_factory=dict)


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second refill rate
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False otherwise
        """
        async with self.lock:
            current_time = time.time()

            # Refill tokens based on elapsed time
            elapsed = current_time - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = current_time

            # Check if we can consume tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def get_status(self) -> dict[str, Any]:
        """Get current bucket status."""
        async with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_refill
            current_tokens = min(
                self.capacity, self.tokens + elapsed * self.refill_rate
            )

            return {
                "tokens": current_tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "utilization": 1.0 - (current_tokens / self.capacity),
            }


class LeakyBucket:
    """Leaky bucket rate limiter implementation."""

    def __init__(self, capacity: int, leak_rate: float):
        """Initialize leaky bucket.

        Args:
            capacity: Maximum bucket capacity
            leak_rate: Requests per second leak rate
        """
        self.capacity = capacity
        self.leak_rate = leak_rate
        self.volume = 0.0
        self.last_leak = time.time()
        self.lock = asyncio.Lock()

    async def add_request(self, volume: float = 1.0) -> bool:
        """Add request to bucket.

        Args:
            volume: Request volume to add

        Returns:
            True if request was accepted, False otherwise
        """
        async with self.lock:
            current_time = time.time()

            # Leak requests based on elapsed time
            elapsed = current_time - self.last_leak
            self.volume = max(0, self.volume - elapsed * self.leak_rate)
            self.last_leak = current_time

            # Check if we can add the request
            if self.volume + volume <= self.capacity:
                self.volume += volume
                return True

            return False

    async def get_status(self) -> dict[str, Any]:
        """Get current bucket status."""
        async with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_leak
            current_volume = max(0, self.volume - elapsed * self.leak_rate)

            return {
                "volume": current_volume,
                "capacity": self.capacity,
                "leak_rate": self.leak_rate,
                "utilization": current_volume / self.capacity,
            }


class SlidingWindowCounter:
    """Sliding window rate limiter implementation."""

    def __init__(self, window_size: int, max_requests: int):
        """Initialize sliding window counter.

        Args:
            window_size: Window size in seconds
            max_requests: Maximum requests in window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = asyncio.Lock()

    async def add_request(self, timestamp: float | None = None) -> bool:
        """Add request to window.

        Args:
            timestamp: Request timestamp (current time if None)

        Returns:
            True if request was accepted, False otherwise
        """
        if timestamp is None:
            timestamp = time.time()

        async with self.lock:
            # Remove old requests outside window
            cutoff_time = timestamp - self.window_size
            while self.requests and self.requests[0] < cutoff_time:
                self.requests.popleft()

            # Check if we can add the request
            if len(self.requests) < self.max_requests:
                self.requests.append(timestamp)
                return True

            return False

    async def get_status(self) -> dict[str, Any]:
        """Get current window status."""
        async with self.lock:
            current_time = time.time()
            cutoff_time = current_time - self.window_size

            # Count requests in current window
            current_requests = sum(
                1 for req_time in self.requests if req_time >= cutoff_time
            )

            return {
                "current_requests": current_requests,
                "max_requests": self.max_requests,
                "window_size": self.window_size,
                "utilization": current_requests / self.max_requests,
            }


class ExponentialBackoffLimiter:
    """Exponential backoff rate limiter."""

    def __init__(
        self,
        base_delay: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 3600.0,
    ):
        """Initialize exponential backoff limiter.

        Args:
            base_delay: Base delay in seconds
            multiplier: Backoff multiplier
            max_delay: Maximum delay in seconds
        """
        self.base_delay = base_delay
        self.multiplier = multiplier
        self.max_delay = max_delay
        self.failure_count = 0
        self.last_failure = 0.0
        self.lock = asyncio.Lock()

    async def check_request(self) -> tuple[bool, float]:
        """Check if request is allowed.

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        async with self.lock:
            current_time = time.time()

            if self.failure_count == 0:
                return True, 0.0

            # Calculate required delay
            delay = min(
                self.base_delay * (self.multiplier ** (self.failure_count - 1)),
                self.max_delay,
            )

            if current_time - self.last_failure >= delay:
                # Reset on successful request
                self.failure_count = 0
                return True, 0.0

            # Calculate remaining delay
            remaining_delay = delay - (current_time - self.last_failure)
            return False, remaining_delay

    async def record_failure(self) -> None:
        """Record a failure for backoff calculation."""
        async with self.lock:
            self.failure_count += 1
            self.last_failure = time.time()

    async def record_success(self) -> None:
        """Record a success to reset backoff."""
        async with self.lock:
            self.failure_count = 0

    async def get_status(self) -> dict[str, Any]:
        """Get current backoff status."""
        async with self.lock:
            current_time = time.time()

            if self.failure_count == 0:
                return {
                    "failure_count": 0,
                    "delay": 0.0,
                    "remaining_delay": 0.0,
                    "blocked": False,
                }

            delay = min(
                self.base_delay * (self.multiplier ** (self.failure_count - 1)),
                self.max_delay,
            )
            remaining_delay = max(0, delay - (current_time - self.last_failure))

            return {
                "failure_count": self.failure_count,
                "delay": delay,
                "remaining_delay": remaining_delay,
                "blocked": remaining_delay > 0,
            }


class RateLimiter:
    """Advanced rate limiter with multiple algorithms."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.limiters: dict[str, Any] = {}
        self.violations: list[RateLimitViolation] = []
        self.stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "blocked_requests": 0,
            "violations": 0,
        }
        self.lock = asyncio.Lock()

        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def check_limit(
        self,
        identifier: str,
        scope: RateLimitScope = RateLimitScope.GLOBAL,
        operation: str = "default",
        tokens: int = 1,
    ) -> RateLimitStatus:
        """Check if request is within rate limit.

        Args:
            identifier: Unique identifier for rate limiting
            scope: Rate limit scope
            operation: Operation name for custom limits
            tokens: Number of tokens to consume

        Returns:
            Rate limit status
        """
        if not self.config.enabled:
            return RateLimitStatus(
                allowed=True,
                remaining=999999,
                reset_time=time.time() + 3600,
                algorithm=self.config.algorithm.value,
            )

        async with self.lock:
            self.stats["total_requests"] += 1

            # Generate limiter key
            limiter_key = f"{scope.value}:{identifier}:{operation}"

            # Get or create limiter
            limiter = await self._get_or_create_limiter(limiter_key, operation)

            # Check rate limit based on algorithm
            status = await self._check_algorithm_limit(limiter, tokens, operation)

            # Update statistics
            if status.allowed:
                self.stats["allowed_requests"] += 1
            else:
                self.stats["blocked_requests"] += 1
                self.stats["violations"] += 1

                # Record violation
                violation = RateLimitViolation(
                    identifier=identifier,
                    scope=scope,
                    violation_time=time.time(),
                    exceeded_limit=self.config.get_limit_for_operation(operation),
                    current_rate=status.current_requests,
                    algorithm=self.config.algorithm,
                    context={"operation": operation, "tokens": tokens},
                )
                self.violations.append(violation)

                # Keep only recent violations
                if len(self.violations) > 1000:
                    self.violations = self.violations[-500:]

            return status

    async def _get_or_create_limiter(self, key: str, operation: str) -> Any:
        """Get or create rate limiter for key."""
        if key not in self.limiters:
            if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                self.limiters[key] = TokenBucket(
                    capacity=self.config.burst_capacity,
                    refill_rate=self.config.get_limit_for_operation(operation),
                )
            elif self.config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                self.limiters[key] = LeakyBucket(
                    capacity=self.config.burst_capacity,
                    leak_rate=self.config.get_limit_for_operation(operation),
                )
            elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                self.limiters[key] = SlidingWindowCounter(
                    window_size=self.config.window_size,
                    max_requests=int(
                        self.config.get_limit_for_operation(operation)
                        * self.config.window_size
                    ),
                )
            elif self.config.algorithm == RateLimitAlgorithm.EXPONENTIAL_BACKOFF:
                self.limiters[key] = ExponentialBackoffLimiter(
                    base_delay=1.0 / self.config.get_limit_for_operation(operation),
                    multiplier=self.config.backoff_multiplier,
                    max_delay=self.config.max_backoff,
                )
            else:
                # Default to token bucket
                self.limiters[key] = TokenBucket(
                    capacity=self.config.burst_capacity,
                    refill_rate=self.config.get_limit_for_operation(operation),
                )

        return self.limiters[key]

    async def _check_algorithm_limit(
        self, limiter: Any, tokens: int, operation: str
    ) -> RateLimitStatus:
        """Check rate limit based on algorithm."""
        current_time = time.time()

        if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            allowed = await limiter.consume(tokens)
            limiter_status = await limiter.get_status()

            return RateLimitStatus(
                allowed=allowed,
                remaining=int(limiter_status["tokens"]),
                reset_time=current_time
                + (limiter.capacity - limiter_status["tokens"]) / limiter.refill_rate,
                retry_after=None if allowed else int(tokens / limiter.refill_rate) + 1,
                algorithm=self.config.algorithm.value,
                metadata=limiter_status,
            )

        elif self.config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            allowed = await limiter.add_request(tokens)
            limiter_status = await limiter.get_status()

            return RateLimitStatus(
                allowed=allowed,
                remaining=int(limiter.capacity - limiter_status["volume"]),
                reset_time=current_time + limiter_status["volume"] / limiter.leak_rate,
                retry_after=None if allowed else int(tokens / limiter.leak_rate) + 1,
                algorithm=self.config.algorithm.value,
                metadata=limiter_status,
            )

        elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            allowed = await limiter.add_request(current_time)
            limiter_status = await limiter.get_status()

            return RateLimitStatus(
                allowed=allowed,
                remaining=limiter.max_requests - limiter_status["current_requests"],
                reset_time=current_time + self.config.window_size,
                retry_after=None if allowed else self.config.window_size,
                current_requests=limiter_status["current_requests"],
                algorithm=self.config.algorithm.value,
                metadata=limiter_status,
            )

        elif self.config.algorithm == RateLimitAlgorithm.EXPONENTIAL_BACKOFF:
            allowed, retry_after = await limiter.check_request()
            limiter_status = await limiter.get_status()

            return RateLimitStatus(
                allowed=allowed,
                remaining=1 if allowed else 0,
                reset_time=current_time + retry_after,
                retry_after=int(retry_after) if retry_after > 0 else None,
                algorithm=self.config.algorithm.value,
                metadata=limiter_status,
            )

        # Default fallback
        return RateLimitStatus(
            allowed=True,
            remaining=999999,
            reset_time=current_time + 3600,
            algorithm="fallback",
        )

    async def record_success(
        self,
        identifier: str,
        scope: RateLimitScope = RateLimitScope.GLOBAL,
        operation: str = "default",
    ) -> None:
        """Record successful request (for exponential backoff)."""
        if self.config.algorithm == RateLimitAlgorithm.EXPONENTIAL_BACKOFF:
            limiter_key = f"{scope.value}:{identifier}:{operation}"
            limiter = await self._get_or_create_limiter(limiter_key, operation)
            await limiter.record_success()

    async def record_failure(
        self,
        identifier: str,
        scope: RateLimitScope = RateLimitScope.GLOBAL,
        operation: str = "default",
    ) -> None:
        """Record failed request (for exponential backoff)."""
        if self.config.algorithm == RateLimitAlgorithm.EXPONENTIAL_BACKOFF:
            limiter_key = f"{scope.value}:{identifier}:{operation}"
            limiter = await self._get_or_create_limiter(limiter_key, operation)
            await limiter.record_failure()

    async def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        async with self.lock:
            return {
                "config": {
                    "algorithm": self.config.algorithm.value,
                    "enabled": self.config.enabled,
                    "requests_per_second": self.config.requests_per_second,
                    "burst_capacity": self.config.burst_capacity,
                    "window_size": self.config.window_size,
                },
                "statistics": self.stats.copy(),
                "active_limiters": len(self.limiters),
                "recent_violations": len(
                    [
                        v
                        for v in self.violations
                        if time.time() - v.violation_time < 3600
                    ]
                ),
                "violation_rate": self.stats["violations"]
                / max(1, self.stats["total_requests"]),
            }

    async def get_violations(self, hours: int = 1) -> list[RateLimitViolation]:
        """Get rate limit violations from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [v for v in self.violations if v.violation_time >= cutoff_time]

    async def reset_limiter(
        self,
        identifier: str,
        scope: RateLimitScope = RateLimitScope.GLOBAL,
        operation: str = "default",
    ) -> bool:
        """Reset rate limiter for specific identifier."""
        async with self.lock:
            limiter_key = f"{scope.value}:{identifier}:{operation}"
            if limiter_key in self.limiters:
                del self.limiters[limiter_key]
                return True
            return False

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old limiters and violations."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")

    async def _cleanup_old_data(self) -> None:
        """Clean up old limiters and violations."""
        async with self.lock:
            # Clean up old violations (keep last 24 hours)
            cutoff_time = time.time() - 86400
            self.violations = [
                v for v in self.violations if v.violation_time >= cutoff_time
            ]

            # Could add logic to clean up inactive limiters
            # For now, we keep all limiters as they might be needed again

    async def close(self) -> None:
        """Close rate limiter and cleanup resources."""
        try:
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass

            async with self.lock:
                self.limiters.clear()
                self.violations.clear()

            logger.info("Rate limiter closed")
        except Exception as e:
            logger.error(f"Error closing rate limiter: {e}")


class RateLimitManager:
    """Manages multiple rate limiters for different scopes and operations."""

    def __init__(self):
        """Initialize rate limit manager."""
        self.limiters: dict[str, RateLimiter] = {}
        self.default_config = RateLimitConfig()
        self.lock = asyncio.Lock()

    async def get_or_create_limiter(
        self, name: str, config: RateLimitConfig | None = None
    ) -> RateLimiter:
        """Get or create rate limiter."""
        async with self.lock:
            if name not in self.limiters:
                self.limiters[name] = RateLimiter(config or self.default_config)
            return self.limiters[name]

    async def check_limit(
        self,
        limiter_name: str,
        identifier: str,
        scope: RateLimitScope = RateLimitScope.GLOBAL,
        operation: str = "default",
        tokens: int = 1,
        config: RateLimitConfig | None = None,
    ) -> RateLimitStatus:
        """Check rate limit across all limiters."""
        limiter = await self.get_or_create_limiter(limiter_name, config)
        return await limiter.check_limit(identifier, scope, operation, tokens)

    async def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all limiters."""
        async with self.lock:
            stats = {}
            for name, limiter in self.limiters.items():
                stats[name] = await limiter.get_stats()
            return stats

    async def close_all(self) -> None:
        """Close all rate limiters."""
        async with self.lock:
            for limiter in self.limiters.values():
                await limiter.close()
            self.limiters.clear()


# Global rate limit manager
_rate_limit_manager: RateLimitManager | None = None


def get_rate_limit_manager() -> RateLimitManager:
    """Get global rate limit manager."""
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
    return _rate_limit_manager


async def close_rate_limit_manager() -> None:
    """Close global rate limit manager."""
    global _rate_limit_manager
    if _rate_limit_manager:
        await _rate_limit_manager.close_all()
        _rate_limit_manager = None


# Convenience functions
async def check_rate_limit(
    identifier: str,
    limiter_name: str = "default",
    scope: RateLimitScope = RateLimitScope.GLOBAL,
    operation: str = "default",
    tokens: int = 1,
    config: RateLimitConfig | None = None,
) -> RateLimitStatus:
    """Check rate limit for identifier."""
    manager = get_rate_limit_manager()
    return await manager.check_limit(
        limiter_name, identifier, scope, operation, tokens, config
    )


async def get_rate_limit_stats() -> dict[str, Any]:
    """Get rate limit statistics."""
    manager = get_rate_limit_manager()
    return await manager.get_all_stats()


@asynccontextmanager
async def rate_limit_context(
    identifier: str,
    limiter_name: str = "default",
    scope: RateLimitScope = RateLimitScope.GLOBAL,
    operation: str = "default",
    tokens: int = 1,
    config: RateLimitConfig | None = None,
):
    """Context manager for rate limited operations."""
    status = await check_rate_limit(
        identifier, limiter_name, scope, operation, tokens, config
    )

    if not status.allowed:
        raise create_external_service_error(
            service_name="rate_limiter",
            message=f"Rate limit exceeded for {identifier}",
            context=ErrorContext(
                operation=operation,
                additional_context={
                    "identifier": identifier,
                    "scope": scope.value,
                    "retry_after": status.retry_after,
                    "remaining": status.remaining,
                    "algorithm": status.algorithm,
                },
            ),
        )

    try:
        yield status
    except Exception:
        # Record failure for exponential backoff
        manager = get_rate_limit_manager()
        limiter = await manager.get_or_create_limiter(limiter_name, config)
        await limiter.record_failure(identifier, scope, operation)
        raise
    else:
        # Record success for exponential backoff
        manager = get_rate_limit_manager()
        limiter = await manager.get_or_create_limiter(limiter_name, config)
        await limiter.record_success(identifier, scope, operation)
