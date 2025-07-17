"""
Rate limiting and DDoS protection for Software API.

This module provides:
- Rate limiting per user/IP
- DDoS protection
- API throttling
- Circuit breaker pattern
- Request queuing
"""

import logging
import time
from collections import defaultdict, deque
from enum import Enum
from threading import Lock
from typing import Any

import redis

logger = logging.getLogger(__name__)


class RateLimitType(str, Enum):
    """Rate limit types."""

    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_ENDPOINT = "per_endpoint"
    GLOBAL = "global"


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimiter:
    """Advanced rate limiting with multiple strategies."""

    def __init__(self, redis_client: redis.Redis | None = None):
        self.redis_client = redis_client
        self.local_cache = {}
        self.cache_lock = Lock()

        # Default rate limits
        self.default_limits = {
            RateLimitType.PER_USER: {"requests": 1000, "window": 3600},  # 1000 req/hour
            RateLimitType.PER_IP: {"requests": 100, "window": 3600},  # 100 req/hour
            RateLimitType.PER_ENDPOINT: {
                "requests": 10000,
                "window": 3600,
            },  # 10k req/hour
            RateLimitType.GLOBAL: {"requests": 100000, "window": 3600},  # 100k req/hour
        }

        # Custom limits for specific resources
        self.custom_limits = {}

    def set_rate_limit(
        self, key: str, requests: int, window: int, limit_type: RateLimitType
    ) -> None:
        """Set custom rate limit for specific key."""
        self.custom_limits[f"{limit_type}:{key}"] = {
            "requests": requests,
            "window": window,
        }

    def is_allowed(
        self, key: str, limit_type: RateLimitType, endpoint: str = ""
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request is allowed under rate limit."""
        limit_key = f"{limit_type}:{key}"
        if endpoint:
            limit_key = f"{limit_key}:{endpoint}"

        # Get rate limit configuration
        limit_config = self.custom_limits.get(
            limit_key, self.default_limits.get(limit_type)
        )
        if not limit_config:
            return True, {}

        requests_allowed = limit_config["requests"]
        window_seconds = limit_config["window"]

        # Use Redis if available, otherwise use local cache
        if self.redis_client:
            return self._check_redis_rate_limit(
                limit_key, requests_allowed, window_seconds
            )
        else:
            return self._check_local_rate_limit(
                limit_key, requests_allowed, window_seconds
            )

    def _check_redis_rate_limit(
        self, key: str, max_requests: int, window: int
    ) -> tuple[bool, dict[str, Any]]:
        """Check rate limit using Redis (sliding window)."""
        now = time.time()
        pipeline = self.redis_client.pipeline()

        # Remove old entries
        pipeline.zremrangebyscore(key, 0, now - window)

        # Count current requests
        pipeline.zcard(key)

        # Add current request
        pipeline.zadd(key, {str(now): now})

        # Set expiry
        pipeline.expire(key, window)

        results = pipeline.execute()
        current_requests = results[1]

        allowed = current_requests < max_requests

        if not allowed:
            # Remove the request we just added if not allowed
            self.redis_client.zrem(key, str(now))

        return allowed, {
            "requests_made": current_requests + (1 if allowed else 0),
            "requests_allowed": max_requests,
            "window_seconds": window,
            "reset_time": int(now + window),
        }

    def _check_local_rate_limit(
        self, key: str, max_requests: int, window: int
    ) -> tuple[bool, dict[str, Any]]:
        """Check rate limit using local cache (sliding window)."""
        now = time.time()

        with self.cache_lock:
            if key not in self.local_cache:
                self.local_cache[key] = deque()

            request_times = self.local_cache[key]

            # Remove old requests
            while request_times and request_times[0] <= now - window:
                request_times.popleft()

            allowed = len(request_times) < max_requests

            if allowed:
                request_times.append(now)

            return allowed, {
                "requests_made": len(request_times),
                "requests_allowed": max_requests,
                "window_seconds": window,
                "reset_time": int(now + window),
            }

    def get_rate_limit_status(
        self, key: str, limit_type: RateLimitType
    ) -> dict[str, Any]:
        """Get current rate limit status."""
        limit_key = f"{limit_type}:{key}"
        limit_config = self.custom_limits.get(
            limit_key, self.default_limits.get(limit_type)
        )

        if not limit_config:
            return {"status": "no_limit"}

        _, status = self.is_allowed(key, limit_type)
        return status


class TokenBucket:
    """Token bucket rate limiting implementation."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self.lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket."""
        with self.lock:
            now = time.time()

            # Add tokens based on time passed
            time_passed = now - self.last_refill
            tokens_to_add = time_passed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False

    def get_status(self) -> dict[str, Any]:
        """Get bucket status."""
        with self.lock:
            return {
                "tokens": self.tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
            }


class DDoSProtection:
    """DDoS protection and anomaly processing."""

    def __init__(self, redis_client: redis.Redis | None = None):
        self.redis_client = redis_client
        self.request_patterns = defaultdict(list)
        self.blocked_ips = set()
        self.suspicious_ips = defaultdict(int)

        # DDoS processing thresholds
        self.burst_threshold = 100  # requests per minute
        self.sustained_threshold = 1000  # requests per hour
        self.pattern_threshold = 0.8  # similarity threshold for pattern processing

        # Block durations
        self.temporary_block_duration = 300  # 5 minutes
        self.permanent_block_duration = 86400  # 24 hours

    def analyze_request(
        self, ip: str, user_agent: str, endpoint: str
    ) -> dict[str, Any]:
        """Analyze request for DDoS patterns."""
        now = time.time()

        analysis = {
            "allowed": True,
            "risk_score": 0.0,
            "reasons": [],
            "action": "allow",
        }

        # Check if IP is blocked
        if self._is_ip_blocked(ip):
            analysis.update(
                {
                    "allowed": False,
                    "action": "block",
                    "reasons": ["IP temporarily blocked"],
                }
            )
            return analysis

        # Record request
        self._record_request(ip, user_agent, endpoint, now)

        # Check for burst attacks
        burst_score = self._check_burst_attack(ip, now)
        if burst_score > 0.7:
            analysis["risk_score"] += burst_score
            analysis["reasons"].append("Burst attack detected")

        # Check for sustained attacks
        sustained_score = self._check_sustained_attack(ip, now)
        if sustained_score > 0.7:
            analysis["risk_score"] += sustained_score
            analysis["reasons"].append("Sustained attack detected")

        # Check for pattern anomalies
        pattern_score = self._check_pattern_anomalies(ip, user_agent, endpoint)
        if pattern_score > 0.5:
            analysis["risk_score"] += pattern_score
            analysis["reasons"].append("Anomalous request pattern")

        # Determine action based on risk score
        if analysis["risk_score"] > 0.8:
            self._block_ip(ip, self.temporary_block_duration)
            analysis.update({"allowed": False, "action": "block"})
        elif analysis["risk_score"] > 0.5:
            analysis["action"] = "throttle"

        return analysis

    def _record_request(
        self, ip: str, user_agent: str, endpoint: str, timestamp: float
    ) -> None:
        """Record request for pattern analysis."""
        request_data = {
            "timestamp": timestamp,
            "user_agent": user_agent,
            "endpoint": endpoint,
        }

        if self.redis_client:
            # Store in Redis with expiry
            key = f"requests:{ip}"
            self.redis_client.lpush(key, str(request_data))
            self.redis_client.expire(key, 3600)  # Keep for 1 hour
        else:
            # Store in local cache
            self.request_patterns[ip].append(request_data)

            # Keep only last hour of requests
            cutoff = timestamp - 3600
            self.request_patterns[ip] = [
                req for req in self.request_patterns[ip] if req["timestamp"] > cutoff
            ]

    def _check_burst_attack(self, ip: str, now: float) -> float:
        """Check for burst attack (high frequency in short time)."""
        if self.redis_client:
            key = f"requests:{ip}"
            recent_count = 0
            requests = self.redis_client.lrange(key, 0, -1)

            for req_str in requests:
                try:
                    req_data = eval(
                        req_str.decode()
                    )  # Note: In production, use json.loads
                    if now - req_data["timestamp"] <= 60:  # Last minute
                        recent_count += 1
                except:
                    continue
        else:
            recent_count = sum(
                1 for req in self.request_patterns[ip] if now - req["timestamp"] <= 60
            )

        if recent_count > self.burst_threshold:
            return min(1.0, recent_count / self.burst_threshold)

        return 0.0

    def _check_sustained_attack(self, ip: str, now: float) -> float:
        """Check for sustained attack (high volume over time)."""
        if self.redis_client:
            key = f"requests:{ip}"
            hour_count = self.redis_client.llen(key)
        else:
            hour_count = len(self.request_patterns[ip])

        if hour_count > self.sustained_threshold:
            return min(1.0, hour_count / self.sustained_threshold)

        return 0.0

    def _check_pattern_anomalies(
        self, ip: str, user_agent: str, endpoint: str
    ) -> float:
        """Check for anomalous request patterns."""
        # Simple pattern analysis - can be enhanced with ML
        risk_factors = 0

        # Check for suspicious user agents
        suspicious_agents = ["bot", "crawler", "scanner", "exploit"]
        if any(agent in user_agent.lower() for agent in suspicious_agents):
            risk_factors += 0.3

        # Check for suspicious endpoints
        suspicious_endpoints = ["/admin", "/.env", "/config", "/wp-admin"]
        if any(sus_ep in endpoint.lower() for sus_ep in suspicious_endpoints):
            risk_factors += 0.4

        # Check request frequency vs normal pattern
        if self.redis_client:
            key = f"requests:{ip}"
            requests = self.redis_client.lrange(key, 0, 9)  # Last 10 requests
        else:
            requests = self.request_patterns[ip][-10:]  # Last 10 requests

        if len(requests) >= 5:
            # Check if all requests are identical (possible bot)
            unique_requests = set()
            for req in requests:
                if isinstance(req, dict):
                    unique_requests.add(req.get("endpoint", ""))

            if len(unique_requests) == 1:
                risk_factors += 0.2

        return min(1.0, risk_factors)

    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked."""
        if self.redis_client:
            return bool(self.redis_client.get(f"blocked:{ip}"))
        else:
            return ip in self.blocked_ips

    def _block_ip(self, ip: str, duration: int) -> None:
        """Block IP for specified duration."""
        if self.redis_client:
            self.redis_client.setex(f"blocked:{ip}", duration, "1")
        else:
            self.blocked_ips.add(ip)
            # Note: Local blocking doesn't have automatic expiry

        logger.warning(f"IP {ip} blocked for {duration} seconds")

    def unblock_ip(self, ip: str) -> None:
        """Unblock IP address."""
        if self.redis_client:
            self.redis_client.delete(f"blocked:{ip}")
        else:
            self.blocked_ips.discard(ip)

        logger.info(f"IP {ip} unblocked")

    def get_blocked_ips(self) -> list:
        """Get list of currently blocked IPs."""
        if self.redis_client:
            # Scan for blocked IP keys
            blocked = []
            for key in self.redis_client.scan_iter("blocked:*"):
                ip = key.decode().replace("blocked:", "")
                blocked.append(ip)
            return blocked
        else:
            return list(self.blocked_ips)


class CircuitBreaker:
    """Circuit breaker pattern for service protection."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self.lock = Lock()

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def get_state(self) -> dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
        }


class RequestQueue:
    """Request queuing for load management."""

    def __init__(self, max_queue_size: int = 1000, max_wait_time: int = 30):
        self.max_queue_size = max_queue_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.processing = 0
        self.max_concurrent = 100
        self.lock = Lock()

    def enqueue_request(self, request_id: str, priority: int = 1) -> bool:
        """Enqueue request for processing."""
        with self.lock:
            if len(self.queue) >= self.max_queue_size:
                return False

            request_data = {
                "id": request_id,
                "priority": priority,
                "enqueue_time": time.time(),
            }

            # Insert based on priority
            inserted = False
            for i, existing_req in enumerate(self.queue):
                if priority > existing_req["priority"]:
                    self.queue.insert(i, request_data)
                    inserted = True
                    break

            if not inserted:
                self.queue.append(request_data)

            return True

    def dequeue_request(self) -> dict[str, Any] | None:
        """Dequeue request for processing."""
        with self.lock:
            if not self.queue or self.processing >= self.max_concurrent:
                return None

            request = self.queue.popleft()

            # Check if request has expired
            if time.time() - request["enqueue_time"] > self.max_wait_time:
                return None

            self.processing += 1
            return request

    def complete_request(self, request_id: str) -> None:
        """Mark request as completed."""
        with self.lock:
            self.processing = max(0, self.processing - 1)

    def get_queue_status(self) -> dict[str, Any]:
        """Get queue status."""
        with self.lock:
            return {
                "queue_size": len(self.queue),
                "max_queue_size": self.max_queue_size,
                "processing": self.processing,
                "max_concurrent": self.max_concurrent,
            }
