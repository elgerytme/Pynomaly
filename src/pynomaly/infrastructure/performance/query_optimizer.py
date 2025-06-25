"""Advanced query optimization and database performance tuning."""

from __future__ import annotations

import functools
import hashlib
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class QueryPlan:
    """Query execution plan with optimization information."""

    query_id: str
    operation: str
    estimated_cost: float
    estimated_rows: int
    actual_cost: float | None = None
    actual_rows: int | None = None
    execution_time_ms: float | None = None
    index_usage: list[str] = field(default_factory=list)
    table_scans: list[str] = field(default_factory=list)
    joins: list[dict[str, Any]] = field(default_factory=list)
    filters: list[dict[str, Any]] = field(default_factory=list)
    optimizations_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class QueryStats:
    """Query execution statistics."""

    query_hash: str
    operation: str
    execution_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    last_executed: datetime | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    optimization_level: int = 0

    def update(
        self, execution_time_ms: float, cached: bool = False, error: bool = False
    ):
        """Update statistics with new execution."""
        self.execution_count += 1
        self.last_executed = datetime.utcnow()

        if error:
            self.error_count += 1
            return

        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        self.total_time_ms += execution_time_ms
        self.min_time_ms = min(self.min_time_ms, execution_time_ms)
        self.max_time_ms = max(self.max_time_ms, execution_time_ms)
        self.avg_time_ms = self.total_time_ms / (
            self.execution_count - self.error_count
        )


class DataFrameOptimizer:
    """Optimizer for pandas DataFrame operations."""

    def __init__(self):
        """Initialize DataFrame optimizer."""
        self.optimization_rules = [
            self._optimize_column_selection,
            self._optimize_filtering,
            self._optimize_groupby,
            self._optimize_sorting,
            self._optimize_joins,
            self._optimize_dtypes,
        ]

    def optimize_dataframe_operation(
        self, df: Any, operation: str, **kwargs  # pandas.DataFrame
    ) -> tuple[Any, list[str]]:
        """Optimize DataFrame operation.

        Args:
            df: Input DataFrame
            operation: Operation type
            **kwargs: Operation parameters

        Returns:
            Tuple of (optimized_df, applied_optimizations)
        """
        if not PANDAS_AVAILABLE:
            return df, []

        optimizations_applied = []
        optimized_df = df

        # Apply optimization rules
        for rule in self.optimization_rules:
            try:
                new_df, applied = rule(optimized_df, operation, **kwargs)
                if applied:
                    optimized_df = new_df
                    optimizations_applied.extend(applied)
            except Exception as e:
                print(f"Optimization rule failed: {e}")

        return optimized_df, optimizations_applied

    def _optimize_column_selection(
        self, df: Any, operation: str, **kwargs
    ) -> tuple[Any, list[str]]:
        """Optimize column selection."""
        optimizations = []

        if operation in ["filter", "groupby", "sort"] and "columns" in kwargs:
            # Select only necessary columns early
            columns = kwargs["columns"]
            if isinstance(columns, list) and len(columns) < len(df.columns):
                df = df[columns]
                optimizations.append("early_column_selection")

        return df, optimizations

    def _optimize_filtering(
        self, df: Any, operation: str, **kwargs
    ) -> tuple[Any, list[str]]:
        """Optimize filtering operations."""
        optimizations = []

        if operation == "filter" and "conditions" in kwargs:
            conditions = kwargs["conditions"]

            # Optimize filter order (most selective first)
            if isinstance(conditions, list) and len(conditions) > 1:
                # Sort conditions by estimated selectivity
                sorted_conditions = self._sort_conditions_by_selectivity(df, conditions)
                if sorted_conditions != conditions:
                    optimizations.append("filter_reordering")

            # Use query() for complex conditions
            if len(conditions) > 2:
                optimizations.append("query_method")

        return df, optimizations

    def _optimize_groupby(
        self, df: Any, operation: str, **kwargs
    ) -> tuple[Any, list[str]]:
        """Optimize groupby operations."""
        optimizations = []

        if operation == "groupby":
            # Check if categorical conversion would help
            groupby_cols = kwargs.get("by", [])
            if isinstance(groupby_cols, str):
                groupby_cols = [groupby_cols]

            for col in groupby_cols:
                if col in df.columns:
                    # Convert to categorical if beneficial
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.1 and df[col].dtype == "object":
                        df[col] = df[col].astype("category")
                        optimizations.append(f"categorical_conversion_{col}")

        return df, optimizations

    def _optimize_sorting(
        self, df: Any, operation: str, **kwargs
    ) -> tuple[Any, list[str]]:
        """Optimize sorting operations."""
        optimizations = []

        if operation == "sort":
            sort_cols = kwargs.get("by", [])
            if isinstance(sort_cols, str):
                sort_cols = [sort_cols]

            # Check if data is already sorted
            for col in sort_cols:
                if col in df.columns:
                    if (
                        df[col].is_monotonic_increasing
                        or df[col].is_monotonic_decreasing
                    ):
                        optimizations.append(f"already_sorted_{col}")

        return df, optimizations

    def _optimize_joins(
        self, df: Any, operation: str, **kwargs
    ) -> tuple[Any, list[str]]:
        """Optimize join operations."""
        optimizations = []

        if operation == "join" and "right" in kwargs:
            right_df = kwargs["right"]
            join_keys = kwargs.get("on", [])

            # Suggest index optimization
            if isinstance(join_keys, str):
                join_keys = [join_keys]

            for key in join_keys:
                if key in df.columns and key in right_df.columns:
                    # Check if setting index would help
                    if not df.index.name == key:
                        optimizations.append(f"index_optimization_{key}")

        return df, optimizations

    def _optimize_dtypes(
        self, df: Any, operation: str, **kwargs
    ) -> tuple[Any, list[str]]:
        """Optimize data types."""
        optimizations = []

        # Downcast numeric types
        for col in df.select_dtypes(include=["int64", "float64"]).columns:
            original_dtype = df[col].dtype

            if df[col].dtype == "int64":
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif df[col].dtype == "float64":
                df[col] = pd.to_numeric(df[col], downcast="float")

            if df[col].dtype != original_dtype:
                optimizations.append(f"downcast_{col}")

        return df, optimizations

    def _sort_conditions_by_selectivity(
        self, df: Any, conditions: list[str]
    ) -> list[str]:
        """Sort filter conditions by estimated selectivity."""
        # Simple heuristic: shorter conditions are often more selective
        return sorted(conditions, key=len)


class QueryCache:
    """Advanced query result caching with TTL and invalidation."""

    def __init__(
        self, max_size: int = 1000, default_ttl: int = 3600, max_memory_mb: int = 500
    ):
        """Initialize query cache.

        Args:
            max_size: Maximum number of cached queries
            default_ttl: Default TTL in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

        self._cache: dict[str, dict[str, Any]] = {}
        self._access_times: dict[str, datetime] = {}
        self._size_estimates: dict[str, int] = {}
        self._lock = threading.RLock()

        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _calculate_size(self, result: Any) -> int:
        """Estimate memory size of result."""
        try:
            if PANDAS_AVAILABLE and isinstance(result, pd.DataFrame):
                return result.memory_usage(deep=True).sum()
            else:
                import sys

                return sys.getsizeof(result)
        except Exception:
            return 1024  # Default estimate

    def _evict_lru(self):
        """Evict least recently used entries."""
        while (
            len(self._cache) >= self.max_size
            or sum(self._size_estimates.values()) >= self.max_memory_bytes
        ):

            if not self._cache:
                break

            # Find LRU entry
            lru_key = min(
                self._access_times.keys(), key=lambda k: self._access_times[k]
            )

            # Remove from cache
            del self._cache[lru_key]
            del self._access_times[lru_key]
            del self._size_estimates[lru_key]

            self.evictions += 1

    def _cleanup_expired(self):
        """Remove expired entries."""
        now = datetime.utcnow()
        expired_keys = []

        for key, entry in self._cache.items():
            expires_at = entry.get("expires_at")
            if expires_at and now >= expires_at:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            del self._access_times[key]
            del self._size_estimates[key]
            self.evictions += 1

    def get(self, query_hash: str) -> Any | None:
        """Get cached query result."""
        with self._lock:
            self._cleanup_expired()

            if query_hash not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[query_hash]
            self._access_times[query_hash] = datetime.utcnow()
            self.hits += 1

            return entry["result"]

    def set(self, query_hash: str, result: Any, ttl: int | None = None) -> bool:
        """Cache query result."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        size_estimate = self._calculate_size(result)

        with self._lock:
            # Evict if necessary
            self._evict_lru()

            entry = {
                "result": result,
                "cached_at": datetime.utcnow(),
                "expires_at": expires_at,
            }

            self._cache[query_hash] = entry
            self._access_times[query_hash] = datetime.utcnow()
            self._size_estimates[query_hash] = size_estimate

            return True

    def invalidate(self, pattern: str | None = None):
        """Invalidate cached entries."""
        with self._lock:
            if pattern is None:
                # Clear all
                self._cache.clear()
                self._access_times.clear()
                self._size_estimates.clear()
            else:
                # Pattern-based invalidation
                import fnmatch

                keys_to_remove = [
                    key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)
                ]

                for key in keys_to_remove:
                    del self._cache[key]
                    del self._access_times[key]
                    del self._size_estimates[key]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "memory_usage_bytes": sum(self._size_estimates.values()),
                "memory_usage_mb": sum(self._size_estimates.values()) / 1024 / 1024,
            }


class QueryOptimizer:
    """Comprehensive query optimizer with caching and performance monitoring."""

    def __init__(
        self,
        enable_caching: bool = True,
        enable_optimization: bool = True,
        cache_ttl: int = 3600,
        max_cache_size: int = 1000,
    ):
        """Initialize query optimizer.

        Args:
            enable_caching: Enable query result caching
            enable_optimization: Enable query optimization
            cache_ttl: Default cache TTL in seconds
            max_cache_size: Maximum cache size
        """
        self.enable_caching = enable_caching
        self.enable_optimization = enable_optimization

        # Components
        self.cache = (
            QueryCache(max_size=max_cache_size, default_ttl=cache_ttl)
            if enable_caching
            else None
        )

        self.df_optimizer = DataFrameOptimizer() if enable_optimization else None

        # Query statistics
        self._query_stats: dict[str, QueryStats] = {}
        self._query_plans: dict[str, QueryPlan] = {}
        self._lock = threading.RLock()

        # Performance history
        self._performance_history: deque = deque(maxlen=10000)

    def _generate_query_hash(self, operation: str, **kwargs) -> str:
        """Generate hash for query caching."""
        # Create a stable hash from operation and parameters
        hash_input = f"{operation}:{sorted(kwargs.items())}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def execute_optimized(
        self, operation: str, data: Any, cache_key: str | None = None, **kwargs
    ) -> tuple[Any, dict[str, Any]]:
        """Execute operation with optimization and caching.

        Args:
            operation: Operation type
            data: Input data
            cache_key: Optional cache key override
            **kwargs: Operation parameters

        Returns:
            Tuple of (result, execution_info)
        """
        start_time = time.time()
        query_hash = cache_key or self._generate_query_hash(operation, **kwargs)

        # Check cache first
        cached_result = None
        if self.cache:
            cached_result = self.cache.get(query_hash)
            if cached_result is not None:
                execution_time_ms = (time.time() - start_time) * 1000

                # Update statistics
                self._update_stats(
                    query_hash, operation, execution_time_ms, cached=True
                )

                return cached_result, {
                    "cached": True,
                    "execution_time_ms": execution_time_ms,
                    "optimizations_applied": [],
                    "query_hash": query_hash,
                }

        # Execute with optimization
        optimizations_applied = []
        optimized_data = data

        if self.df_optimizer and PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            optimized_data, optimizations_applied = (
                self.df_optimizer.optimize_dataframe_operation(
                    data, operation, **kwargs
                )
            )

        # Execute operation
        try:
            result = self._execute_operation(operation, optimized_data, **kwargs)
            execution_time_ms = (time.time() - start_time) * 1000

            # Cache result
            if self.cache and result is not None:
                self.cache.set(query_hash, result)

            # Update statistics
            self._update_stats(query_hash, operation, execution_time_ms, cached=False)

            # Record performance
            self._record_performance(
                query_hash, operation, execution_time_ms, optimizations_applied
            )

            return result, {
                "cached": False,
                "execution_time_ms": execution_time_ms,
                "optimizations_applied": optimizations_applied,
                "query_hash": query_hash,
            }

        except Exception:
            execution_time_ms = (time.time() - start_time) * 1000
            self._update_stats(query_hash, operation, execution_time_ms, error=True)
            raise

    def _execute_operation(self, operation: str, data: Any, **kwargs) -> Any:
        """Execute the actual operation."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            raise ValueError("Operation execution requires pandas DataFrame")

        if operation == "filter":
            conditions = kwargs.get("conditions", [])
            if isinstance(conditions, str):
                return data.query(conditions)
            else:
                # Apply multiple conditions
                result = data
                for condition in conditions:
                    result = result.query(condition)
                return result

        elif operation == "groupby":
            by = kwargs.get("by", [])
            agg = kwargs.get("agg", {})
            return data.groupby(by).agg(agg)

        elif operation == "sort":
            by = kwargs.get("by", [])
            ascending = kwargs.get("ascending", True)
            return data.sort_values(by=by, ascending=ascending)

        elif operation == "join":
            right = kwargs.get("right")
            on = kwargs.get("on", None)
            how = kwargs.get("how", "inner")
            return data.merge(right, on=on, how=how)

        elif operation == "select":
            columns = kwargs.get("columns", [])
            return data[columns]

        elif operation == "aggregate":
            agg = kwargs.get("agg", {})
            return data.agg(agg)

        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _update_stats(
        self,
        query_hash: str,
        operation: str,
        execution_time_ms: float,
        cached: bool = False,
        error: bool = False,
    ):
        """Update query statistics."""
        with self._lock:
            if query_hash not in self._query_stats:
                self._query_stats[query_hash] = QueryStats(
                    query_hash=query_hash, operation=operation
                )

            self._query_stats[query_hash].update(execution_time_ms, cached, error)

    def _record_performance(
        self,
        query_hash: str,
        operation: str,
        execution_time_ms: float,
        optimizations_applied: list[str],
    ):
        """Record performance information."""
        record = {
            "timestamp": datetime.utcnow(),
            "query_hash": query_hash,
            "operation": operation,
            "execution_time_ms": execution_time_ms,
            "optimizations_applied": optimizations_applied,
        }

        self._performance_history.append(record)

    def get_query_stats(self, query_hash: str | None = None) -> dict[str, Any]:
        """Get query statistics."""
        with self._lock:
            if query_hash:
                stats = self._query_stats.get(query_hash)
                return stats.__dict__ if stats else {}
            else:
                return {
                    hash_key: stats.__dict__
                    for hash_key, stats in self._query_stats.items()
                }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if not self._performance_history:
            return {"message": "No performance data available"}

        # Calculate summary statistics
        execution_times = [
            record["execution_time_ms"] for record in self._performance_history
        ]
        operations = [record["operation"] for record in self._performance_history]
        optimizations = [
            opt
            for record in self._performance_history
            for opt in record["optimizations_applied"]
        ]

        import statistics
        from collections import Counter

        summary = {
            "total_queries": len(self._performance_history),
            "avg_execution_time_ms": statistics.mean(execution_times),
            "median_execution_time_ms": statistics.median(execution_times),
            "min_execution_time_ms": min(execution_times),
            "max_execution_time_ms": max(execution_times),
            "operation_counts": dict(Counter(operations)),
            "optimization_counts": dict(Counter(optimizations)),
            "cache_stats": self.cache.get_stats() if self.cache else {},
            "time_range": {
                "start": min(
                    record["timestamp"] for record in self._performance_history
                ).isoformat(),
                "end": max(
                    record["timestamp"] for record in self._performance_history
                ).isoformat(),
            },
        }

        return summary

    def invalidate_cache(self, pattern: str | None = None):
        """Invalidate cache entries."""
        if self.cache:
            self.cache.invalidate(pattern)

    def get_slow_queries(self, threshold_ms: float = 1000) -> list[dict[str, Any]]:
        """Get queries that exceed performance threshold."""
        slow_queries = []

        with self._lock:
            for query_hash, stats in self._query_stats.items():
                if stats.avg_time_ms > threshold_ms:
                    slow_queries.append(
                        {
                            "query_hash": query_hash,
                            "operation": stats.operation,
                            "avg_time_ms": stats.avg_time_ms,
                            "execution_count": stats.execution_count,
                            "total_time_ms": stats.total_time_ms,
                        }
                    )

        # Sort by average execution time
        slow_queries.sort(key=lambda x: x["avg_time_ms"], reverse=True)
        return slow_queries

    def optimize_decorator(self, operation: str, cache_ttl: int | None = None):
        """Decorator for automatic query optimization."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Extract data from first argument (assuming it's a DataFrame)
                if args and PANDAS_AVAILABLE and isinstance(args[0], pd.DataFrame):
                    data = args[0]
                    func_kwargs = kwargs.copy()

                    # Generate cache key
                    cache_key = self._generate_query_hash(
                        f"{func.__name__}_{operation}",
                        args=str(args[1:]) if len(args) > 1 else "",
                        **func_kwargs,
                    )

                    # Execute with optimization
                    result, info = self.execute_optimized(
                        operation, data, cache_key, **func_kwargs
                    )

                    return result
                else:
                    # Fallback to original function
                    return func(*args, **kwargs)

            return wrapper

        return decorator


# Global optimizer instance
_global_optimizer: QueryOptimizer | None = None


def get_optimizer() -> QueryOptimizer | None:
    """Get global query optimizer."""
    return _global_optimizer


def configure_optimizer(**kwargs) -> QueryOptimizer:
    """Configure global query optimizer."""
    global _global_optimizer
    _global_optimizer = QueryOptimizer(**kwargs)
    return _global_optimizer


def optimize_query(operation: str, cache_ttl: int | None = None):
    """Decorator for query optimization using global optimizer."""

    def decorator(func: Callable) -> Callable:
        if not _global_optimizer:
            return func  # Return original function if no optimizer configured

        return _global_optimizer.optimize_decorator(operation, cache_ttl)(func)

    return decorator
