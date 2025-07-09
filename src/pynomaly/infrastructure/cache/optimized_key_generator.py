"""Optimized cache key generation for high-performance caching."""

from __future__ import annotations

import hashlib
import inspect
import logging
import statistics
import time
from collections import deque
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class OptimizedCacheKeyGenerator:
    """High-performance cache key generator with optimizations."""

    # LRU cache for function signatures to avoid repeated inspection
    _signature_cache: dict[Callable, inspect.Signature] = {}
    _func_name_cache: dict[Callable, str] = {}
    _max_cache_size = 1000

    # Performance monitoring
    _key_generation_times = deque(maxlen=1000)
    _key_size_distribution = {"small": 0, "medium": 0, "large": 0, "xlarge": 0}

    @classmethod
    def generate_key(
        cls,
        func: Callable,
        args: tuple,
        kwargs: dict[str, Any],
        prefix: str = "",
        ignore_args: list[str] | None = None,
        ignore_kwargs: list[str] | None = None,
        serialize_complex_args: bool = True,
    ) -> str:
        """Generate optimized cache key for function call.

        Args:
            func: Function being cached
            args: Positional arguments
            kwargs: Keyword arguments
            prefix: Key prefix
            ignore_args: Arguments to ignore in key generation
            ignore_kwargs: Keyword arguments to ignore
            serialize_complex_args: Whether to serialize complex arguments

        Returns:
            Cache key string
        """
        start_time = time.perf_counter()

        try:
            # Manage cache size to prevent memory leaks
            if len(cls._signature_cache) > cls._max_cache_size:
                cls._clear_oldest_cache_entries()

            # Use cached signature if available
            if func not in cls._signature_cache:
                cls._signature_cache[func] = inspect.signature(func)
            sig = cls._signature_cache[func]

            # Use cached function name if available
            if func not in cls._func_name_cache:
                cls._func_name_cache[func] = f"{func.__module__}.{func.__qualname__}"
            func_name = cls._func_name_cache[func]

            # Fast path for simple cases (no kwargs, no ignored args)
            if not kwargs and not ignore_args and not ignore_kwargs:
                key = cls._generate_simple_key(func_name, args, prefix, serialize_complex_args)
            else:
                # Complex path with full signature binding
                key = cls._generate_complex_key(
                    sig, func_name, args, kwargs, prefix,
                    ignore_args, ignore_kwargs, serialize_complex_args
                )

            # Record performance metrics
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000
            cls._record_performance(time_ms, len(key))

            return key

        except Exception as e:
            logger.error(f"Cache key generation failed for {func.__name__}: {e}")
            # Fallback to simple key generation
            return cls._generate_fallback_key(func, args, kwargs, prefix)

    @classmethod
    def _generate_simple_key(
        cls, func_name: str, args: tuple, prefix: str, serialize_complex: bool
    ) -> str:
        """Generate key for simple case without kwargs or ignored args."""
        key_parts = [prefix, func_name] if prefix else [func_name]

        # Process args directly with optimized serialization
        for i, value in enumerate(args):
            serialized_value = cls._serialize_value_fast(value, serialize_complex)
            key_parts.append(f"arg{i}={serialized_value}")

        key = ":".join(key_parts)
        return cls._hash_if_long(key, prefix)

    @classmethod
    def _generate_complex_key(
        cls,
        sig: inspect.Signature,
        func_name: str,
        args: tuple,
        kwargs: dict[str, Any],
        prefix: str,
        ignore_args: list[str] | None,
        ignore_kwargs: list[str] | None,
        serialize_complex: bool,
    ) -> str:
        """Generate key for complex case with full signature binding."""
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        key_parts = [prefix, func_name] if prefix else [func_name]
        ignore_set = set(ignore_args or []) | set(ignore_kwargs or [])

        # Add arguments
        for param_name, value in bound_args.arguments.items():
            if param_name not in ignore_set:
                serialized_value = cls._serialize_value_fast(value, serialize_complex)
                key_parts.append(f"{param_name}={serialized_value}")

        key = ":".join(key_parts)
        return cls._hash_if_long(key, prefix)

    @classmethod
    def _serialize_value_fast(cls, value: Any, serialize_complex: bool) -> str:
        """Fast value serialization optimized for common cases."""
        # Handle None
        if value is None:
            return "None"

        # Get type once for efficiency
        value_type = type(value)

        # Fast path for primitive types
        if value_type in (str, int, float, bool):
            return str(value)

        # Handle collections with size limits for performance
        if value_type in (list, tuple):
            if not serialize_complex:
                return f"{value_type.__name__}_{len(value)}"
            if len(value) == 0:
                return "[]"
            if len(value) > 20:  # Limit for performance
                # Sample-based approach for large collections
                sample_str = cls._serialize_value_fast(value[0], False)
                return f"{value_type.__name__}_{len(value)}_sample_{sample_str}"
            return f"[{','.join(cls._serialize_value_fast(v, serialize_complex) for v in value)}]"

        if value_type is dict:
            if not serialize_complex:
                return f"dict_{len(value)}"
            if len(value) == 0:
                return "{}"
            if len(value) > 20:  # Limit for performance
                # Sample-based approach for large dictionaries
                first_key = next(iter(value))
                sample_str = cls._serialize_value_fast(value[first_key], False)
                return f"dict_{len(value)}_sample_{first_key}:{sample_str}"

            # Sort items for consistent keys
            sorted_items = sorted(value.items())
            return f"{{{','.join(f'{k}:{cls._serialize_value_fast(v, serialize_complex)}' for k, v in sorted_items)}}}"

        # Handle objects with __dict__
        if hasattr(value, "__dict__"):
            if serialize_complex:
                return f"obj_{value_type.__name__}_{id(value)}"
            else:
                return f"obj_{value_type.__name__}"

        # Handle other types
        return f"type_{value_type.__name__}"

    @classmethod
    def _hash_if_long(cls, key: str, prefix: str) -> str:
        """Hash key if it's too long, using fastest available algorithm."""
        if len(key) <= 250:  # Slightly higher threshold for better performance
            return key

        key_bytes = key.encode('utf-8')

        # Try xxhash first (fastest available)
        try:
            import xxhash
            key_hash = xxhash.xxh64(key_bytes).hexdigest()
            return f"{prefix}:xxh64:{key_hash}"
        except ImportError:
            pass

        # Fall back to blake2b (faster than md5 for larger inputs)
        try:
            key_hash = hashlib.blake2b(key_bytes, digest_size=16).hexdigest()
            return f"{prefix}:blake2b:{key_hash}"
        except (AttributeError, TypeError):
            pass

        # Final fallback to md5
        key_hash = hashlib.md5(key_bytes).hexdigest()
        return f"{prefix}:md5:{key_hash}"

    @classmethod
    def _generate_fallback_key(
        cls, func: Callable, args: tuple, kwargs: dict[str, Any], prefix: str
    ) -> str:
        """Generate fallback key when normal generation fails."""
        try:
            func_name = f"{func.__module__}.{func.__name__}"
            args_str = str(hash(args))
            kwargs_str = str(hash(frozenset(kwargs.items()))) if kwargs else ""
            key = f"{prefix}:{func_name}:{args_str}:{kwargs_str}"
            return cls._hash_if_long(key, prefix)
        except Exception:
            # Ultimate fallback
            return f"{prefix}:fallback:{hash((func, args, tuple(kwargs.items())))}"

    @classmethod
    def _record_performance(cls, time_ms: float, key_size: int) -> None:
        """Record performance metrics."""
        cls._key_generation_times.append(time_ms)

        # Track key size distribution
        if key_size < 50:
            cls._key_size_distribution["small"] += 1
        elif key_size < 100:
            cls._key_size_distribution["medium"] += 1
        elif key_size < 250:
            cls._key_size_distribution["large"] += 1
        else:
            cls._key_size_distribution["xlarge"] += 1

    @classmethod
    def _clear_oldest_cache_entries(cls) -> None:
        """Clear oldest cache entries to prevent memory leaks."""
        # Remove 25% of entries
        remove_count = max(1, len(cls._signature_cache) // 4)

        # Remove from both caches
        for _ in range(remove_count):
            if cls._signature_cache:
                func = next(iter(cls._signature_cache))
                cls._signature_cache.pop(func, None)
                cls._func_name_cache.pop(func, None)

    @classmethod
    def get_performance_stats(cls) -> dict[str, Any]:
        """Get performance statistics."""
        if not cls._key_generation_times:
            return {"status": "no_data"}

        times = list(cls._key_generation_times)
        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else avg_time

        return {
            "average_generation_time_ms": avg_time,
            "p95_generation_time_ms": p95_time,
            "total_generated_keys": len(times),
            "key_size_distribution": cls._key_size_distribution.copy(),
            "cache_sizes": {
                "signature_cache": len(cls._signature_cache),
                "func_name_cache": len(cls._func_name_cache),
            },
            "performance_recommendations": cls._get_recommendations(avg_time, p95_time),
        }

    @classmethod
    def _get_recommendations(cls, avg_time: float, p95_time: float) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        if avg_time > 1.0:
            recommendations.append("Consider reducing complex argument serialization")
        if p95_time > 5.0:
            recommendations.append("P95 key generation time is high - consider caching more aggressively")

        # Check key size distribution
        total_keys = sum(cls._key_size_distribution.values())
        if total_keys > 0:
            large_ratio = (cls._key_size_distribution["large"] + cls._key_size_distribution["xlarge"]) / total_keys
            if large_ratio > 0.3:
                recommendations.append("High percentage of large keys - consider optimizing argument serialization")

        return recommendations

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all internal caches."""
        cls._signature_cache.clear()
        cls._func_name_cache.clear()
        cls._key_generation_times.clear()
        cls._key_size_distribution = {"small": 0, "medium": 0, "large": 0, "xlarge": 0}

    @classmethod
    def optimize_for_function(cls, func: Callable) -> None:
        """Pre-warm cache for a specific function."""
        if func not in cls._signature_cache:
            cls._signature_cache[func] = inspect.signature(func)
        if func not in cls._func_name_cache:
            cls._func_name_cache[func] = f"{func.__module__}.{func.__qualname__}"
