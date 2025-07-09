"""Performance utilities for cache system optimization."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable

from .optimized_key_generator import OptimizedCacheKeyGenerator

logger = logging.getLogger(__name__)


class CachePerformanceOptimizer:
    """Optimizes cache performance through various strategies."""

    def __init__(self):
        self.optimization_applied = False
        self.original_generate_key = None
        self.performance_metrics = deque(maxlen=1000)

    def enable_optimized_key_generation(self, cache_decorator_class: type) -> None:
        """Enable optimized key generation for cache decorators."""
        if self.optimization_applied:
            logger.warning("Optimization already applied")
            return

        # Store original method
        if hasattr(cache_decorator_class, 'generate_cache_key'):
            self.original_generate_key = cache_decorator_class.generate_cache_key
            
            # Replace with optimized version
            cache_decorator_class.generate_cache_key = self._optimized_generate_cache_key
            self.optimization_applied = True
            logger.info("Optimized cache key generation enabled")
        else:
            logger.error("Cache decorator class doesn't have generate_cache_key method")

    def _optimized_generate_cache_key(
        self, 
        cache_decorator_instance: Any,
        func: Callable,
        args: tuple,
        kwargs: dict[str, Any],
    ) -> str:
        """Optimized cache key generation method."""
        start_time = time.perf_counter()
        
        try:
            key = OptimizedCacheKeyGenerator.generate_key(
                func=func,
                args=args,
                kwargs=kwargs,
                prefix=cache_decorator_instance.config.key_prefix,
                ignore_args=cache_decorator_instance.config.ignore_args,
                ignore_kwargs=cache_decorator_instance.config.ignore_kwargs,
                serialize_complex_args=cache_decorator_instance.config.serialize_complex_args,
            )
            
            # Record performance metrics
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000
            self.performance_metrics.append({
                'time_ms': time_ms,
                'key_length': len(key),
                'function': func.__name__,
                'timestamp': time.time()
            })
            
            return key
            
        except Exception as e:
            logger.error(f"Optimized key generation failed: {e}")
            # Fallback to original method
            if self.original_generate_key:
                return self.original_generate_key(cache_decorator_instance, func, args, kwargs)
            raise

    def disable_optimization(self, cache_decorator_class: type) -> None:
        """Disable optimization and restore original behavior."""
        if not self.optimization_applied:
            logger.warning("No optimization to disable")
            return

        if self.original_generate_key and hasattr(cache_decorator_class, 'generate_cache_key'):
            cache_decorator_class.generate_cache_key = self.original_generate_key
            self.optimization_applied = False
            logger.info("Optimization disabled, original behavior restored")

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.performance_metrics:
            return {"status": "no_data"}

        metrics = list(self.performance_metrics)
        
        # Calculate statistics
        times = [m['time_ms'] for m in metrics]
        key_lengths = [m['key_length'] for m in metrics]
        functions = [m['function'] for m in metrics]
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        avg_key_length = sum(key_lengths) / len(key_lengths)
        max_key_length = max(key_lengths)
        
        # Function-specific stats
        function_stats = {}
        for func_name in set(functions):
            func_metrics = [m for m in metrics if m['function'] == func_name]
            func_times = [m['time_ms'] for m in func_metrics]
            function_stats[func_name] = {
                'count': len(func_metrics),
                'avg_time_ms': sum(func_times) / len(func_times),
                'max_time_ms': max(func_times),
            }
        
        # Get optimization-specific stats
        optimizer_stats = OptimizedCacheKeyGenerator.get_performance_stats()
        
        return {
            "optimization_enabled": self.optimization_applied,
            "total_measurements": len(metrics),
            "key_generation_performance": {
                "avg_time_ms": avg_time,
                "max_time_ms": max_time,
                "min_time_ms": min_time,
                "p95_time_ms": sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max_time,
            },
            "key_characteristics": {
                "avg_length": avg_key_length,
                "max_length": max_key_length,
            },
            "function_breakdown": function_stats,
            "optimizer_internal_stats": optimizer_stats,
            "recommendations": self._generate_recommendations(avg_time, max_time, optimizer_stats),
        }

    def _generate_recommendations(self, avg_time: float, max_time: float, optimizer_stats: dict) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if avg_time > 2.0:
            recommendations.append("Average key generation time is high - consider reducing argument complexity")
        
        if max_time > 10.0:
            recommendations.append("Maximum key generation time is very high - investigate worst-case scenarios")
        
        if optimizer_stats.get("status") != "no_data":
            recommendations.extend(optimizer_stats.get("performance_recommendations", []))
        
        return recommendations

    def warm_up_function_cache(self, functions: list[Callable]) -> None:
        """Pre-warm cache for frequently used functions."""
        for func in functions:
            try:
                OptimizedCacheKeyGenerator.optimize_for_function(func)
                logger.debug(f"Warmed up cache for function: {func.__name__}")
            except Exception as e:
                logger.warning(f"Failed to warm up cache for {func.__name__}: {e}")


class CacheHealthMonitor:
    """Monitors cache health and performance."""

    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self.last_check = 0
        self.health_history = deque(maxlen=100)
        self.alerts = deque(maxlen=50)

    def check_health(self) -> dict[str, Any]:
        """Check cache system health."""
        current_time = time.time()
        
        # Skip if too frequent
        if current_time - self.last_check < self.check_interval:
            return {"status": "skipped", "reason": "too_frequent"}
        
        self.last_check = current_time
        
        # Get performance stats
        optimizer_stats = OptimizedCacheKeyGenerator.get_performance_stats()
        
        # Analyze health
        health_status = self._analyze_health(optimizer_stats)
        
        # Record in history
        health_record = {
            "timestamp": current_time,
            "status": health_status,
            "stats": optimizer_stats,
        }
        self.health_history.append(health_record)
        
        # Generate alerts if needed
        self._check_for_alerts(health_status, optimizer_stats)
        
        return {
            "status": health_status,
            "timestamp": current_time,
            "statistics": optimizer_stats,
            "recent_alerts": list(self.alerts)[-5:],  # Last 5 alerts
            "health_trend": self._get_health_trend(),
        }

    def _analyze_health(self, stats: dict) -> str:
        """Analyze cache health based on statistics."""
        if stats.get("status") == "no_data":
            return "unknown"
        
        avg_time = stats.get("average_generation_time_ms", 0)
        p95_time = stats.get("p95_generation_time_ms", 0)
        
        if avg_time > 5.0 or p95_time > 20.0:
            return "unhealthy"
        elif avg_time > 2.0 or p95_time > 10.0:
            return "degraded"
        else:
            return "healthy"

    def _check_for_alerts(self, health_status: str, stats: dict) -> None:
        """Check for conditions that require alerts."""
        current_time = time.time()
        
        if health_status == "unhealthy":
            alert = {
                "timestamp": current_time,
                "level": "error",
                "message": "Cache key generation performance is unhealthy",
                "stats": stats,
            }
            self.alerts.append(alert)
        elif health_status == "degraded":
            alert = {
                "timestamp": current_time,
                "level": "warning",
                "message": "Cache key generation performance is degraded",
                "stats": stats,
            }
            self.alerts.append(alert)

    def _get_health_trend(self) -> str:
        """Get health trend over time."""
        if len(self.health_history) < 3:
            return "insufficient_data"
        
        recent_statuses = [h["status"] for h in list(self.health_history)[-3:]]
        
        if all(status == "healthy" for status in recent_statuses):
            return "stable_healthy"
        elif all(status == "unhealthy" for status in recent_statuses):
            return "stable_unhealthy"
        elif recent_statuses == ["unhealthy", "degraded", "healthy"]:
            return "improving"
        elif recent_statuses == ["healthy", "degraded", "unhealthy"]:
            return "degrading"
        else:
            return "fluctuating"


# Global instances
_performance_optimizer = CachePerformanceOptimizer()
_health_monitor = CacheHealthMonitor()


def get_performance_optimizer() -> CachePerformanceOptimizer:
    """Get global performance optimizer instance."""
    return _performance_optimizer


def get_health_monitor() -> CacheHealthMonitor:
    """Get global health monitor instance."""
    return _health_monitor


def enable_cache_optimizations() -> None:
    """Enable all cache performance optimizations."""
    try:
        # Import here to avoid circular imports
        from .decorators import CacheDecorator
        
        _performance_optimizer.enable_optimized_key_generation(CacheDecorator)
        logger.info("Cache performance optimizations enabled")
    except Exception as e:
        logger.error(f"Failed to enable cache optimizations: {e}")


def get_cache_performance_report() -> dict[str, Any]:
    """Get comprehensive cache performance report."""
    return {
        "optimizer_report": _performance_optimizer.get_performance_report(),
        "health_report": _health_monitor.check_health(),
        "key_generator_stats": OptimizedCacheKeyGenerator.get_performance_stats(),
    }