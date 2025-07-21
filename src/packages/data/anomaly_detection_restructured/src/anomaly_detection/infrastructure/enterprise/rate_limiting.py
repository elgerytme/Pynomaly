"""
API Rate Limiting and Throttling for Pynomaly Detection
=======================================================

Comprehensive rate limiting system providing:
- Flexible rate limiting strategies
- Tenant-specific quotas and throttling
- Distributed rate limiting support
- Performance monitoring and analytics
- Burst handling and sliding windows
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import hashlib

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    """Rate limiting strategy enumeration."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_LOG = "sliding_log"

class ThrottleAction(Enum):
    """Throttling action enumeration."""
    REJECT = "reject"
    DELAY = "delay"
    DEGRADE = "degrade"
    QUEUE = "queue"

class RateLimitScope(Enum):
    """Rate limit scope enumeration."""
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    IP = "ip"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"

@dataclass
class RateLimitRule:
    """Rate limiting rule definition."""
    rule_id: str
    name: str
    scope: RateLimitScope
    strategy: RateLimitStrategy
    limit: int  # requests per window
    window_seconds: int  # time window in seconds
    burst_limit: Optional[int] = None  # burst allowance
    throttle_action: ThrottleAction = ThrottleAction.REJECT
    priority: int = 1  # higher priority rules evaluated first
    is_active: bool = True
    created_date: datetime = field(default_factory=datetime.now)
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RateLimitRequest:
    """Rate limit evaluation request."""
    identifier: str  # user_id, tenant_id, ip, etc.
    scope: RateLimitScope
    endpoint: str
    method: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RateLimitResult:
    """Rate limit evaluation result."""
    allowed: bool
    rule_id: Optional[str] = None
    limit: Optional[int] = None
    remaining: Optional[int] = None
    reset_time: Optional[datetime] = None
    retry_after: Optional[int] = None
    throttle_action: Optional[ThrottleAction] = None
    reason: str = ""

@dataclass
class ThrottleConfig:
    """Throttling configuration."""
    enable_delay: bool = True
    max_delay_seconds: float = 10.0
    delay_multiplier: float = 1.5
    enable_degradation: bool = True
    degradation_levels: List[str] = field(default_factory=lambda: ["reduced", "minimal"])
    queue_size: int = 1000
    queue_timeout: float = 30.0

class RateLimiter:
    """Comprehensive rate limiting system."""
    
    def __init__(self, redis_client: Optional[Any] = None, 
                 enable_distributed: bool = False):
        """Initialize rate limiter.
        
        Args:
            redis_client: Optional Redis client for distributed limiting
            enable_distributed: Enable distributed rate limiting
        """
        self.redis_client = redis_client
        self.enable_distributed = enable_distributed and redis_client is not None
        
        # Rate limiting rules
        self.rules: Dict[str, RateLimitRule] = {}
        
        # Local storage for non-distributed mode
        self.request_counts: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.token_buckets: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.sliding_logs: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Threading
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'throttled_requests': 0,
            'rules_evaluated': 0,
            'avg_evaluation_time': 0.0
        }
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info(f"Rate Limiter initialized (distributed: {self.enable_distributed})")
    
    def add_rule(self, rule: RateLimitRule) -> bool:
        """Add rate limiting rule.
        
        Args:
            rule: Rate limiting rule
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                self.rules[rule.rule_id] = rule
            
            logger.info(f"Rate limiting rule added: {rule.rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add rate limiting rule: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove rate limiting rule.
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                if rule_id in self.rules:
                    del self.rules[rule_id]
                    logger.info(f"Rate limiting rule removed: {rule_id}")
                    return True
                else:
                    logger.warning(f"Rate limiting rule not found: {rule_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to remove rate limiting rule: {e}")
            return False
    
    def check_rate_limit(self, request: RateLimitRequest) -> RateLimitResult:
        """Check rate limit for request.
        
        Args:
            request: Rate limit request
            
        Returns:
            Rate limit result
        """
        start_time = time.time()
        
        try:
            with self.lock:
                self.stats['total_requests'] += 1
                
                # Get applicable rules
                applicable_rules = self._get_applicable_rules(request)
                
                # Evaluate rules by priority
                for rule in sorted(applicable_rules, key=lambda r: r.priority, reverse=True):
                    self.stats['rules_evaluated'] += 1
                    
                    # Check if request exceeds limit
                    result = self._evaluate_rule(rule, request)
                    
                    if not result.allowed:
                        self.stats['blocked_requests'] += 1
                        if result.throttle_action != ThrottleAction.REJECT:
                            self.stats['throttled_requests'] += 1
                        
                        # Update evaluation time
                        evaluation_time = time.time() - start_time
                        self.stats['avg_evaluation_time'] = (
                            self.stats['avg_evaluation_time'] * 0.9 + evaluation_time * 0.1
                        )
                        
                        return result
                
                # All rules passed
                evaluation_time = time.time() - start_time
                self.stats['avg_evaluation_time'] = (
                    self.stats['avg_evaluation_time'] * 0.9 + evaluation_time * 0.1
                )
                
                return RateLimitResult(allowed=True, reason="Rate limit check passed")
                
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return RateLimitResult(allowed=True, reason=f"Evaluation error: {e}")
    
    def increment_usage(self, request: RateLimitRequest) -> bool:
        """Increment usage counters for successful requests.
        
        Args:
            request: Rate limit request
            
        Returns:
            Success status
        """
        try:
            # Get applicable rules and increment their counters
            applicable_rules = self._get_applicable_rules(request)
            
            for rule in applicable_rules:
                self._increment_rule_counter(rule, request)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to increment usage: {e}")
            return False
    
    def get_usage_info(self, identifier: str, scope: RateLimitScope) -> Dict[str, Any]:
        """Get usage information for identifier.
        
        Args:
            identifier: Identifier (user_id, tenant_id, etc.)
            scope: Rate limit scope
            
        Returns:
            Usage information
        """
        try:
            usage_info = {
                'identifier': identifier,
                'scope': scope.value,
                'rules': []
            }
            
            # Get usage for applicable rules
            for rule in self.rules.values():
                if rule.scope == scope and rule.is_active:
                    key = self._get_key(rule, identifier)
                    
                    if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                        bucket_info = self._get_token_bucket_info(key, rule)
                        usage_info['rules'].append({
                            'rule_id': rule.rule_id,
                            'strategy': rule.strategy.value,
                            'tokens_available': bucket_info.get('tokens', rule.limit),
                            'limit': rule.limit,
                            'window_seconds': rule.window_seconds
                        })
                    else:
                        count_info = self._get_count_info(key, rule)
                        usage_info['rules'].append({
                            'rule_id': rule.rule_id,
                            'strategy': rule.strategy.value,
                            'current_count': count_info.get('count', 0),
                            'limit': rule.limit,
                            'window_seconds': rule.window_seconds,
                            'reset_time': count_info.get('reset_time')
                        })
            
            return usage_info
            
        except Exception as e:
            logger.error(f"Failed to get usage info: {e}")
            return {}
    
    def reset_usage(self, identifier: str, scope: RateLimitScope, 
                   rule_id: Optional[str] = None) -> bool:
        """Reset usage counters.
        
        Args:
            identifier: Identifier to reset
            scope: Rate limit scope
            rule_id: Optional specific rule to reset
            
        Returns:
            Success status
        """
        try:
            rules_to_reset = []
            
            if rule_id:
                rule = self.rules.get(rule_id)
                if rule and rule.scope == scope:
                    rules_to_reset.append(rule)
            else:
                rules_to_reset = [
                    rule for rule in self.rules.values()
                    if rule.scope == scope and rule.is_active
                ]
            
            for rule in rules_to_reset:
                key = self._get_key(rule, identifier)
                self._reset_rule_counter(key, rule)
            
            logger.info(f"Usage reset for {identifier} ({scope.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset usage: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            stats = self.stats.copy()
            
            # Add additional metrics
            if stats['total_requests'] > 0:
                stats['block_rate'] = stats['blocked_requests'] / stats['total_requests']
                stats['throttle_rate'] = stats['throttled_requests'] / stats['total_requests']
            else:
                stats['block_rate'] = 0.0
                stats['throttle_rate'] = 0.0
            
            stats['active_rules'] = len([r for r in self.rules.values() if r.is_active])
            stats['distributed_mode'] = self.enable_distributed
            
            return stats
    
    def _initialize_default_rules(self):
        """Initialize default rate limiting rules."""
        default_rules = [
            # Global rate limits
            RateLimitRule(
                rule_id="global_api_limit",
                name="Global API Rate Limit",
                scope=RateLimitScope.GLOBAL,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                limit=10000,
                window_seconds=3600,  # 1 hour
                priority=1
            ),
            
            # Tenant-specific limits
            RateLimitRule(
                rule_id="tenant_api_limit",
                name="Tenant API Rate Limit",
                scope=RateLimitScope.TENANT,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                limit=1000,
                window_seconds=3600,
                burst_limit=100,
                priority=2
            ),
            
            # User-specific limits
            RateLimitRule(
                rule_id="user_api_limit", 
                name="User API Rate Limit",
                scope=RateLimitScope.USER,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                limit=100,
                window_seconds=900,  # 15 minutes
                priority=3
            ),
            
            # IP-based limits for security
            RateLimitRule(
                rule_id="ip_security_limit",
                name="IP Security Rate Limit",
                scope=RateLimitScope.IP,
                strategy=RateLimitStrategy.FIXED_WINDOW,
                limit=500,
                window_seconds=3600,
                throttle_action=ThrottleAction.DELAY,
                priority=4
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def _get_applicable_rules(self, request: RateLimitRequest) -> List[RateLimitRule]:
        """Get rules applicable to request."""
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.is_active:
                continue
            
            # Check scope match
            if rule.scope == request.scope:
                # Check additional conditions
                if self._check_rule_conditions(rule, request):
                    applicable_rules.append(rule)
            
            # Also check global rules
            elif rule.scope == RateLimitScope.GLOBAL:
                if self._check_rule_conditions(rule, request):
                    applicable_rules.append(rule)
        
        return applicable_rules
    
    def _check_rule_conditions(self, rule: RateLimitRule, request: RateLimitRequest) -> bool:
        """Check if rule conditions are met."""
        if not rule.conditions:
            return True
        
        # Check endpoint patterns
        if 'endpoints' in rule.conditions:
            allowed_endpoints = rule.conditions['endpoints']
            if request.endpoint not in allowed_endpoints:
                return False
        
        # Check HTTP methods
        if 'methods' in rule.conditions:
            allowed_methods = rule.conditions['methods']
            if request.method not in allowed_methods:
                return False
        
        # Check time-based conditions
        if 'time_restrictions' in rule.conditions:
            current_hour = datetime.now().hour
            allowed_hours = rule.conditions['time_restrictions'].get('allowed_hours', [])
            if allowed_hours and current_hour not in allowed_hours:
                return False
        
        return True
    
    def _evaluate_rule(self, rule: RateLimitRule, request: RateLimitRequest) -> RateLimitResult:
        """Evaluate specific rule against request."""
        key = self._get_key(rule, request.identifier)
        
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._evaluate_fixed_window(rule, key, request)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._evaluate_sliding_window(rule, key, request)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._evaluate_token_bucket(rule, key, request)
        elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._evaluate_leaky_bucket(rule, key, request)
        elif rule.strategy == RateLimitStrategy.SLIDING_LOG:
            return self._evaluate_sliding_log(rule, key, request)
        else:
            return RateLimitResult(allowed=True, reason="Unknown strategy")
    
    def _evaluate_fixed_window(self, rule: RateLimitRule, key: str, 
                              request: RateLimitRequest) -> RateLimitResult:
        """Evaluate fixed window rate limit."""
        current_time = time.time()
        window_start = int(current_time // rule.window_seconds) * rule.window_seconds
        
        if self.enable_distributed:
            count = self._get_redis_count(key, window_start, rule.window_seconds)
        else:
            count_data = self.request_counts[key].get('fixed_window', {})
            if count_data.get('window_start') != window_start:
                count_data = {'window_start': window_start, 'count': 0}
                self.request_counts[key]['fixed_window'] = count_data
            count = count_data['count']
        
        if count >= rule.limit:
            reset_time = datetime.fromtimestamp(window_start + rule.window_seconds)
            return RateLimitResult(
                allowed=False,
                rule_id=rule.rule_id,
                limit=rule.limit,
                remaining=0,
                reset_time=reset_time,
                retry_after=int(window_start + rule.window_seconds - current_time),
                throttle_action=rule.throttle_action,
                reason=f"Fixed window limit exceeded ({count}/{rule.limit})"
            )
        
        return RateLimitResult(
            allowed=True,
            rule_id=rule.rule_id,
            limit=rule.limit,
            remaining=rule.limit - count - 1,
            reset_time=datetime.fromtimestamp(window_start + rule.window_seconds)
        )
    
    def _evaluate_sliding_window(self, rule: RateLimitRule, key: str,
                                request: RateLimitRequest) -> RateLimitResult:
        """Evaluate sliding window rate limit."""
        current_time = time.time()
        window_start = current_time - rule.window_seconds
        
        if self.enable_distributed:
            count = self._get_redis_sliding_count(key, window_start, current_time)
        else:
            # Clean old entries
            sliding_data = self.request_counts[key].setdefault('sliding_window', [])
            sliding_data[:] = [t for t in sliding_data if t > window_start]
            count = len(sliding_data)
        
        if count >= rule.limit:
            return RateLimitResult(
                allowed=False,
                rule_id=rule.rule_id,
                limit=rule.limit,
                remaining=0,
                retry_after=1,  # Retry in 1 second for sliding window
                throttle_action=rule.throttle_action,
                reason=f"Sliding window limit exceeded ({count}/{rule.limit})"
            )
        
        return RateLimitResult(
            allowed=True,
            rule_id=rule.rule_id,
            limit=rule.limit,
            remaining=rule.limit - count - 1
        )
    
    def _evaluate_token_bucket(self, rule: RateLimitRule, key: str,
                              request: RateLimitRequest) -> RateLimitResult:
        """Evaluate token bucket rate limit."""
        current_time = time.time()
        
        if self.enable_distributed:
            bucket_info = self._get_redis_token_bucket(key, rule, current_time)
        else:
            bucket_data = self.token_buckets[key].setdefault('token_bucket', {
                'tokens': rule.limit,
                'last_refill': current_time
            })
            
            # Refill tokens
            time_passed = current_time - bucket_data['last_refill']
            tokens_to_add = int(time_passed * rule.limit / rule.window_seconds)
            bucket_data['tokens'] = min(rule.limit, bucket_data['tokens'] + tokens_to_add)
            bucket_data['last_refill'] = current_time
            
            bucket_info = bucket_data
        
        if bucket_info['tokens'] < 1:
            return RateLimitResult(
                allowed=False,
                rule_id=rule.rule_id,
                limit=rule.limit,
                remaining=0,
                retry_after=int(rule.window_seconds / rule.limit),
                throttle_action=rule.throttle_action,
                reason="Token bucket empty"
            )
        
        return RateLimitResult(
            allowed=True,
            rule_id=rule.rule_id,
            limit=rule.limit,
            remaining=int(bucket_info['tokens']) - 1
        )
    
    def _evaluate_leaky_bucket(self, rule: RateLimitRule, key: str,
                              request: RateLimitRequest) -> RateLimitResult:
        """Evaluate leaky bucket rate limit."""
        # Simplified leaky bucket implementation
        return self._evaluate_token_bucket(rule, key, request)
    
    def _evaluate_sliding_log(self, rule: RateLimitRule, key: str,
                             request: RateLimitRequest) -> RateLimitResult:
        """Evaluate sliding log rate limit."""
        current_time = time.time()
        window_start = current_time - rule.window_seconds
        
        # Get or create sliding log
        log = self.sliding_logs[key]
        
        # Remove old entries
        while log and log[0] <= window_start:
            log.popleft()
        
        if len(log) >= rule.limit:
            # Find when oldest entry will expire
            oldest_time = log[0] if log else current_time
            retry_after = int(oldest_time + rule.window_seconds - current_time + 1)
            
            return RateLimitResult(
                allowed=False,
                rule_id=rule.rule_id,
                limit=rule.limit,
                remaining=0,
                retry_after=retry_after,
                throttle_action=rule.throttle_action,
                reason=f"Sliding log limit exceeded ({len(log)}/{rule.limit})"
            )
        
        return RateLimitResult(
            allowed=True,
            rule_id=rule.rule_id,
            limit=rule.limit,
            remaining=rule.limit - len(log) - 1
        )
    
    def _increment_rule_counter(self, rule: RateLimitRule, request: RateLimitRequest):
        """Increment counter for rule."""
        key = self._get_key(rule, request.identifier)
        current_time = time.time()
        
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            if self.enable_distributed:
                self._increment_redis_count(key, current_time, rule.window_seconds)
            else:
                window_start = int(current_time // rule.window_seconds) * rule.window_seconds
                count_data = self.request_counts[key].setdefault('fixed_window', {
                    'window_start': window_start, 'count': 0
                })
                if count_data['window_start'] == window_start:
                    count_data['count'] += 1
        
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            if self.enable_distributed:
                self._add_redis_sliding_entry(key, current_time)
            else:
                sliding_data = self.request_counts[key].setdefault('sliding_window', [])
                sliding_data.append(current_time)
        
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            if self.enable_distributed:
                self._consume_redis_token(key, rule)
            else:
                bucket_data = self.token_buckets[key].get('token_bucket')
                if bucket_data and bucket_data['tokens'] > 0:
                    bucket_data['tokens'] -= 1
        
        elif rule.strategy == RateLimitStrategy.SLIDING_LOG:
            log = self.sliding_logs[key]
            log.append(current_time)
    
    def _get_key(self, rule: RateLimitRule, identifier: str) -> str:
        """Generate cache key for rule and identifier."""
        return f"ratelimit:{rule.rule_id}:{identifier}"
    
    def _get_count_info(self, key: str, rule: RateLimitRule) -> Dict[str, Any]:
        """Get count information for key."""
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            count_data = self.request_counts[key].get('fixed_window', {})
            return {
                'count': count_data.get('count', 0),
                'reset_time': count_data.get('window_start', 0) + rule.window_seconds
            }
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            sliding_data = self.request_counts[key].get('sliding_window', [])
            current_time = time.time()
            window_start = current_time - rule.window_seconds
            valid_entries = [t for t in sliding_data if t > window_start]
            return {'count': len(valid_entries)}
        
        return {'count': 0}
    
    def _get_token_bucket_info(self, key: str, rule: RateLimitRule) -> Dict[str, Any]:
        """Get token bucket information."""
        bucket_data = self.token_buckets[key].get('token_bucket', {})
        current_time = time.time()
        
        # Refill calculation
        tokens = bucket_data.get('tokens', rule.limit)
        last_refill = bucket_data.get('last_refill', current_time)
        time_passed = current_time - last_refill
        tokens_to_add = time_passed * rule.limit / rule.window_seconds
        tokens = min(rule.limit, tokens + tokens_to_add)
        
        return {'tokens': tokens}
    
    def _reset_rule_counter(self, key: str, rule: RateLimitRule):
        """Reset counter for rule."""
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            if key in self.request_counts:
                self.request_counts[key].pop('fixed_window', None)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            if key in self.request_counts:
                self.request_counts[key].pop('sliding_window', None)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            if key in self.token_buckets:
                bucket_data = self.token_buckets[key].get('token_bucket')
                if bucket_data:
                    bucket_data['tokens'] = rule.limit
        elif rule.strategy == RateLimitStrategy.SLIDING_LOG:
            if key in self.sliding_logs:
                self.sliding_logs[key].clear()
    
    # Redis operations (simplified implementations)
    def _get_redis_count(self, key: str, window_start: float, window_seconds: int) -> int:
        """Get count from Redis for fixed window."""
        if not self.redis_client:
            return 0
        try:
            return int(self.redis_client.get(f"{key}:{int(window_start)}") or 0)
        except Exception:
            return 0
    
    def _increment_redis_count(self, key: str, current_time: float, window_seconds: int):
        """Increment Redis counter for fixed window."""
        if not self.redis_client:
            return
        try:
            window_start = int(current_time // window_seconds) * window_seconds
            redis_key = f"{key}:{window_start}"
            pipe = self.redis_client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, window_seconds)
            pipe.execute()
        except Exception as e:
            logger.error(f"Redis increment failed: {e}")
    
    def _get_redis_sliding_count(self, key: str, window_start: float, current_time: float) -> int:
        """Get sliding window count from Redis."""
        if not self.redis_client:
            return 0
        try:
            return self.redis_client.zcount(key, window_start, current_time)
        except Exception:
            return 0
    
    def _add_redis_sliding_entry(self, key: str, timestamp: float):
        """Add entry to Redis sliding window."""
        if not self.redis_client:
            return
        try:
            pipe = self.redis_client.pipeline()
            pipe.zadd(key, {str(timestamp): timestamp})
            pipe.zremrangebyscore(key, 0, timestamp - 3600)  # Clean old entries
            pipe.expire(key, 3600)
            pipe.execute()
        except Exception as e:
            logger.error(f"Redis sliding entry failed: {e}")
    
    def _get_redis_token_bucket(self, key: str, rule: RateLimitRule, current_time: float) -> Dict[str, Any]:
        """Get token bucket from Redis."""
        if not self.redis_client:
            return {'tokens': rule.limit, 'last_refill': current_time}
        
        try:
            bucket_data = self.redis_client.hgetall(key)
            if not bucket_data:
                bucket_data = {'tokens': str(rule.limit), 'last_refill': str(current_time)}
                self.redis_client.hmset(key, bucket_data)
                self.redis_client.expire(key, rule.window_seconds * 2)
            
            tokens = float(bucket_data.get('tokens', rule.limit))
            last_refill = float(bucket_data.get('last_refill', current_time))
            
            # Refill tokens
            time_passed = current_time - last_refill
            tokens_to_add = time_passed * rule.limit / rule.window_seconds
            tokens = min(rule.limit, tokens + tokens_to_add)
            
            return {'tokens': tokens, 'last_refill': current_time}
            
        except Exception as e:
            logger.error(f"Redis token bucket failed: {e}")
            return {'tokens': rule.limit, 'last_refill': current_time}
    
    def _consume_redis_token(self, key: str, rule: RateLimitRule):
        """Consume token from Redis bucket."""
        if not self.redis_client:
            return
        try:
            current_time = time.time()
            bucket_info = self._get_redis_token_bucket(key, rule, current_time)
            
            if bucket_info['tokens'] > 0:
                new_tokens = bucket_info['tokens'] - 1
                self.redis_client.hmset(key, {
                    'tokens': str(new_tokens),
                    'last_refill': str(current_time)
                })
        except Exception as e:
            logger.error(f"Redis token consumption failed: {e}")


class ThrottlingService:
    """Advanced throttling service for handling rate limit violations."""
    
    def __init__(self, rate_limiter: RateLimiter, config: ThrottleConfig = None):
        """Initialize throttling service.
        
        Args:
            rate_limiter: Rate limiter instance
            config: Throttling configuration
        """
        self.rate_limiter = rate_limiter
        self.config = config or ThrottleConfig()
        
        # Request queues for throttling
        self.request_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.queue_size))
        self.processing_delays: Dict[str, float] = defaultdict(float)
        
        logger.info("Throttling Service initialized")
    
    def process_request(self, request: RateLimitRequest, 
                       handler: Callable) -> Any:
        """Process request with throttling.
        
        Args:
            request: Rate limit request
            handler: Request handler function
            
        Returns:
            Handler result or throttling response
        """
        # Check rate limit
        limit_result = self.rate_limiter.check_rate_limit(request)
        
        if limit_result.allowed:
            # Request allowed, increment usage and process
            self.rate_limiter.increment_usage(request)
            return handler()
        
        # Request rate limited, apply throttling action
        return self._apply_throttling(request, limit_result, handler)
    
    def _apply_throttling(self, request: RateLimitRequest, 
                         limit_result: RateLimitResult,
                         handler: Callable) -> Any:
        """Apply throttling action based on limit result.
        
        Args:
            request: Rate limit request
            limit_result: Rate limit result
            handler: Request handler function
            
        Returns:
            Throttling response
        """
        action = limit_result.throttle_action or ThrottleAction.REJECT
        
        if action == ThrottleAction.REJECT:
            return self._reject_request(limit_result)
        
        elif action == ThrottleAction.DELAY:
            return self._delay_request(request, limit_result, handler)
        
        elif action == ThrottleAction.DEGRADE:
            return self._degrade_request(request, limit_result, handler)
        
        elif action == ThrottleAction.QUEUE:
            return self._queue_request(request, limit_result, handler)
        
        else:
            return self._reject_request(limit_result)
    
    def _reject_request(self, limit_result: RateLimitResult) -> Dict[str, Any]:
        """Reject rate-limited request."""
        return {
            'error': 'Rate limit exceeded',
            'message': limit_result.reason,
            'retry_after': limit_result.retry_after,
            'limit': limit_result.limit,
            'remaining': limit_result.remaining
        }
    
    def _delay_request(self, request: RateLimitRequest, 
                      limit_result: RateLimitResult,
                      handler: Callable) -> Any:
        """Delay request execution."""
        if not self.config.enable_delay:
            return self._reject_request(limit_result)
        
        # Calculate delay
        identifier = request.identifier
        current_delay = self.processing_delays[identifier]
        
        if current_delay < self.config.max_delay_seconds:
            # Increase delay exponentially
            new_delay = min(
                current_delay * self.config.delay_multiplier,
                self.config.max_delay_seconds
            )
            self.processing_delays[identifier] = new_delay
            
            # Sleep for delay duration
            time.sleep(new_delay)
            
            # Try processing again
            return handler()
        else:
            # Max delay reached, reject
            return self._reject_request(limit_result)
    
    def _degrade_request(self, request: RateLimitRequest,
                        limit_result: RateLimitResult,
                        handler: Callable) -> Any:
        """Process request with degraded service."""
        if not self.config.enable_degradation:
            return self._reject_request(limit_result)
        
        # Add degradation markers to request
        degradation_level = self.config.degradation_levels[0] if self.config.degradation_levels else "reduced"
        request.metadata['degradation_level'] = degradation_level
        request.metadata['degraded'] = True
        
        # Process with degraded handler
        return handler()
    
    def _queue_request(self, request: RateLimitRequest,
                      limit_result: RateLimitResult,
                      handler: Callable) -> Any:
        """Queue request for later processing."""
        identifier = request.identifier
        queue = self.request_queues[identifier]
        
        if len(queue) >= self.config.queue_size:
            return {
                'error': 'Request queue full',
                'message': 'Too many queued requests, please try again later',
                'queue_size': len(queue)
            }
        
        # Add to queue with timestamp
        queued_item = {
            'request': request,
            'handler': handler,
            'queued_at': time.time()
        }
        queue.append(queued_item)
        
        return {
            'message': 'Request queued for processing',
            'queue_position': len(queue),
            'estimated_wait': len(queue) * 2  # Rough estimate
        }
    
    def process_queued_requests(self):
        """Process queued requests (should be called periodically)."""
        current_time = time.time()
        
        for identifier, queue in self.request_queues.items():
            if not queue:
                continue
            
            # Process one request from queue if rate limit allows
            queued_item = queue[0]
            
            # Check if request has timed out
            if current_time - queued_item['queued_at'] > self.config.queue_timeout:
                queue.popleft()
                continue
            
            # Check rate limit for queued request
            limit_result = self.rate_limiter.check_rate_limit(queued_item['request'])
            
            if limit_result.allowed:
                # Process the request
                queue.popleft()
                try:
                    self.rate_limiter.increment_usage(queued_item['request'])
                    queued_item['handler']()
                except Exception as e:
                    logger.error(f"Queued request processing failed: {e}")
    
    def get_queue_status(self, identifier: str) -> Dict[str, Any]:
        """Get queue status for identifier.
        
        Args:
            identifier: Identifier to check
            
        Returns:
            Queue status information
        """
        queue = self.request_queues[identifier]
        
        return {
            'identifier': identifier,
            'queue_size': len(queue),
            'max_queue_size': self.config.queue_size,
            'oldest_request_age': time.time() - queue[0]['queued_at'] if queue else 0,
            'current_delay': self.processing_delays[identifier]
        }