"""Query optimization and caching implementation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

logger = structlog.get_logger(__name__)


class QueryType(Enum):
    """Query type enumeration."""

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"


@dataclass
class QueryMetrics:
    """Query execution metrics."""

    query_hash: str
    query_type: QueryType
    execution_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    last_executed: float = field(default_factory=time.time)
    cache_hits: int = 0
    cache_misses: int = 0

    def update_execution(self, execution_time: float) -> None:
        """Update execution metrics."""
        self.execution_count += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.execution_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.last_executed = time.time()


@dataclass
class CacheEntry:
    """Query cache entry."""

    result: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: float | None = None

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def access(self) -> None:
        """Mark cache entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class IndexRecommendation:
    """Index recommendation."""

    table_name: str
    columns: list[str]
    index_type: str = "btree"
    reason: str = ""
    estimated_benefit: float = 0.0
    query_count: int = 0
    avg_time_improvement: float = 0.0


class QueryCache:
    """Query result cache with TTL and LRU eviction."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float | None = 3600,  # 1 hour
        cleanup_interval: float = 300,  # 5 minutes
    ):
        """Initialize query cache.

        Args:
            max_size: Maximum number of cached entries
            default_ttl: Default TTL in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self._lock = asyncio.Lock()

        # Start cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._start_cleanup_task()

    def _generate_key(self, query: str, params: dict[str, Any] | None = None) -> str:
        """Generate cache key for query and parameters."""
        query_data = {"query": query.strip().lower(), "params": params or {}}

        # Create deterministic hash
        query_str = json.dumps(query_data, sort_keys=True)
        return hashlib.sha256(query_str.encode()).hexdigest()[:16]

    async def get(
        self, query: str, params: dict[str, Any] | None = None
    ) -> Any | None:
        """Get cached query result.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(query, params)

        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None

            # Update access
            entry.access()

            # Move to end of access order (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            return entry.result

    async def set(
        self,
        query: str,
        result: Any,
        params: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        """Cache query result.

        Args:
            query: SQL query
            result: Query result
            params: Query parameters
            ttl: Time to live in seconds
        """
        key = self._generate_key(query, params)
        ttl = ttl or self.default_ttl

        async with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()

            # Create cache entry
            entry = CacheEntry(
                result=result,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
            )

            self._cache[key] = entry

            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return

        lru_key = self._access_order.pop(0)
        if lru_key in self._cache:
            del self._cache[lru_key]

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cache cleanup error", error=str(e))

    async def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        async with self._lock:
            expired_keys = []

            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)

            if expired_keys:
                logger.debug(
                    "Cleaned up expired cache entries", count=len(expired_keys)
                )

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(entry.access_count for entry in self._cache.values())
        total_entries = len(self._cache)

        return {
            "total_entries": total_entries,
            "max_size": self.max_size,
            "total_hits": total_hits,
            "hit_rate": total_hits / max(1, total_entries),
            "memory_usage_mb": self._estimate_memory_usage() / 1024 / 1024,
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in bytes."""
        # Rough estimation
        return len(self._cache) * 1024  # Assume 1KB per entry on average

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()

    async def close(self) -> None:
        """Close cache and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        await self.clear()


class QueryAnalyzer:
    """Analyze queries for optimization opportunities."""

    def __init__(self):
        """Initialize query analyzer."""
        self.slow_queries: list[tuple[str, float, dict[str, Any]]] = []
        self.query_patterns: dict[str, int] = defaultdict(int)
        self.table_access_patterns: dict[str, set[str]] = defaultdict(set)

    def analyze_query(
        self, query: str, execution_time: float, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Analyze a query for optimization opportunities.

        Args:
            query: SQL query
            execution_time: Query execution time
            params: Query parameters

        Returns:
            Analysis results
        """
        analysis = {
            "is_slow": execution_time > 1.0,  # Consider queries > 1s as slow
            "execution_time": execution_time,
            "recommendations": [],
            "issues": [],
        }

        query_lower = query.lower().strip()

        # Track slow queries
        if analysis["is_slow"]:
            self.slow_queries.append((query, execution_time, params or {}))

            # Keep only last 100 slow queries
            if len(self.slow_queries) > 100:
                self.slow_queries = self.slow_queries[-100:]

        # Analyze query patterns
        self._analyze_query_pattern(query_lower, analysis)

        # Analyze table access
        self._analyze_table_access(query_lower, analysis)

        # Check for common issues
        self._check_query_issues(query_lower, analysis)

        return analysis

    def _analyze_query_pattern(self, query: str, analysis: dict[str, Any]) -> None:
        """Analyze query patterns."""
        # Extract query type
        if query.startswith("select"):
            query_type = "select"
        elif query.startswith("insert"):
            query_type = "insert"
        elif query.startswith("update"):
            query_type = "update"
        elif query.startswith("delete"):
            query_type = "delete"
        else:
            query_type = "other"

        self.query_patterns[query_type] += 1
        analysis["query_type"] = query_type

        # Check for N+1 query pattern
        if query_type == "select" and " in (" in query:
            analysis["recommendations"].append(
                "Consider using JOIN instead of IN clause for better performance"
            )

    def _analyze_table_access(self, query: str, analysis: dict[str, Any]) -> None:
        """Analyze table access patterns."""
        # Simple table extraction (would be more sophisticated in production)
        words = query.split()
        tables = []

        for i, word in enumerate(words):
            if word.lower() in ("from", "join", "update", "into") and i + 1 < len(
                words
            ):
                table_name = words[i + 1].replace(",", "").replace("(", "")
                tables.append(table_name)

        analysis["tables"] = tables

        # Track column access for each table
        for table in tables:
            if "where" in query:
                # Extract columns from WHERE clause (simplified)
                where_part = query.split("where", 1)[1]
                columns = self._extract_columns_from_where(where_part)
                self.table_access_patterns[table].update(columns)

    def _extract_columns_from_where(self, where_clause: str) -> list[str]:
        """Extract column names from WHERE clause."""
        # Simplified column extraction
        import re

        # Find patterns like column_name = value or column_name IN (...)
        column_pattern = r"(\w+)\s*(?:=|in|like|>|<|>=|<=)"
        matches = re.findall(column_pattern, where_clause.lower())

        return [match for match in matches if match not in ("and", "or", "not")]

    def _check_query_issues(self, query: str, analysis: dict[str, Any]) -> None:
        """Check for common query issues."""
        issues = []
        recommendations = []

        # Check for SELECT *
        if "select *" in query:
            issues.append("Using SELECT * can be inefficient")
            recommendations.append("Specify only needed columns in SELECT clause")

        # Check for missing LIMIT in potentially large results
        if "select" in query and "limit" not in query and "count(" not in query:
            recommendations.append("Consider adding LIMIT clause for large result sets")

        # Check for functions in WHERE clause
        if any(func in query for func in ["upper(", "lower(", "substring("]):
            issues.append("Functions in WHERE clause prevent index usage")
            recommendations.append(
                "Consider using functional indexes or preprocessing data"
            )

        # Check for OR conditions that might benefit from UNION
        if query.count(" or ") > 2:
            recommendations.append(
                "Consider using UNION instead of multiple OR conditions"
            )

        analysis["issues"].extend(issues)
        analysis["recommendations"].extend(recommendations)

    def get_index_recommendations(self) -> list[IndexRecommendation]:
        """Generate index recommendations based on query patterns."""
        recommendations = []

        for table, columns in self.table_access_patterns.items():
            if len(columns) > 0:
                # Recommend single-column indexes for frequently accessed columns
                for column in columns:
                    recommendations.append(
                        IndexRecommendation(
                            table_name=table,
                            columns=[column],
                            reason="Frequently used in WHERE clauses",
                            estimated_benefit=0.3,  # Rough estimate
                        )
                    )

                # Recommend composite indexes for multiple columns
                if len(columns) > 1:
                    column_list = list(columns)[:3]  # Limit to 3 columns
                    recommendations.append(
                        IndexRecommendation(
                            table_name=table,
                            columns=column_list,
                            reason="Multiple columns used together in queries",
                            estimated_benefit=0.5,
                        )
                    )

        return recommendations

    def get_slow_query_report(self) -> dict[str, Any]:
        """Get report of slow queries."""
        if not self.slow_queries:
            return {"total_slow_queries": 0, "queries": []}

        avg_time = sum(q[1] for q in self.slow_queries) / len(self.slow_queries)
        max_time = max(q[1] for q in self.slow_queries)

        return {
            "total_slow_queries": len(self.slow_queries),
            "avg_execution_time": avg_time,
            "max_execution_time": max_time,
            "queries": [
                {
                    "query": query[:200] + "..." if len(query) > 200 else query,
                    "execution_time": exec_time,
                    "params": params,
                }
                for query, exec_time, params in self.slow_queries[-10:]  # Last 10
            ],
        }


class IndexManager:
    """Manage database indexes for optimization."""

    def __init__(self, engine: Engine | AsyncEngine):
        """Initialize index manager.

        Args:
            engine: Database engine
        """
        self.engine = engine
        self.existing_indexes: dict[str, list[dict[str, Any]]] = {}
        self._refresh_existing_indexes()

    def _refresh_existing_indexes(self) -> None:
        """Refresh list of existing indexes."""
        try:
            inspector = inspect(self.engine)

            for table_name in inspector.get_table_names():
                indexes = inspector.get_indexes(table_name)
                self.existing_indexes[table_name] = indexes

        except Exception as e:
            logger.error("Failed to refresh existing indexes", error=str(e))

    def create_index_if_not_exists(
        self,
        table_name: str,
        columns: list[str],
        index_name: str | None = None,
        index_type: str = "btree",
    ) -> bool:
        """Create index if it doesn't exist.

        Args:
            table_name: Table name
            columns: Column names
            index_name: Index name (auto-generated if None)
            index_type: Index type

        Returns:
            True if index was created, False if already exists
        """
        if not index_name:
            index_name = f"idx_{table_name}_{'_'.join(columns)}"

        # Check if similar index exists
        if self._index_exists(table_name, columns):
            logger.info(
                "Index already exists or similar index found",
                table=table_name,
                columns=columns,
            )
            return False

        try:
            # Create index using raw SQL
            column_list = ", ".join(columns)
            sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table_name} ({column_list})
            """

            with self.engine.connect() as conn:
                conn.execute(text(sql))
                conn.commit()

            logger.info(
                "Created index",
                index_name=index_name,
                table=table_name,
                columns=columns,
            )

            # Refresh index cache
            self._refresh_existing_indexes()
            return True

        except Exception as e:
            logger.error(
                "Failed to create index",
                index_name=index_name,
                table=table_name,
                columns=columns,
                error=str(e),
            )
            return False

    def _index_exists(self, table_name: str, columns: list[str]) -> bool:
        """Check if an index exists for the given columns."""
        if table_name not in self.existing_indexes:
            return False

        for index in self.existing_indexes[table_name]:
            index_columns = index.get("column_names", [])

            # Check if all columns are covered by existing index
            if all(col in index_columns for col in columns):
                return True

        return False

    def get_index_usage_stats(self) -> dict[str, Any]:
        """Get index usage statistics."""
        try:
            # This would require database-specific queries
            # For PostgreSQL: pg_stat_user_indexes
            # For MySQL: INFORMATION_SCHEMA.STATISTICS
            # Simplified implementation

            stats = {
                "total_indexes": sum(
                    len(indexes) for indexes in self.existing_indexes.values()
                ),
                "tables_with_indexes": len(self.existing_indexes),
                "indexes_by_table": {
                    table: len(indexes)
                    for table, indexes in self.existing_indexes.items()
                },
            }

            return stats

        except Exception as e:
            logger.error("Failed to get index usage stats", error=str(e))
            return {}


class QueryPerformanceTracker:
    """Track query performance metrics."""

    def __init__(self, max_queries: int = 10000):
        """Initialize performance tracker.

        Args:
            max_queries: Maximum number of queries to track
        """
        self.max_queries = max_queries
        self.metrics: dict[str, QueryMetrics] = {}
        self._lock = asyncio.Lock()

    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for query normalization."""
        # Normalize query by removing parameters and extra whitespace
        normalized = " ".join(query.strip().lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    async def record_query(
        self, query: str, execution_time: float, params: dict[str, Any] | None = None
    ) -> QueryMetrics:
        """Record query execution.

        Args:
            query: SQL query
            execution_time: Execution time in seconds
            params: Query parameters

        Returns:
            Updated query metrics
        """
        query_hash = self._generate_query_hash(query)
        query_type = self._determine_query_type(query)

        async with self._lock:
            if query_hash not in self.metrics:
                # Check if we need to evict old entries
                if len(self.metrics) >= self.max_queries:
                    await self._evict_oldest_metrics()

                self.metrics[query_hash] = QueryMetrics(
                    query_hash=query_hash, query_type=query_type
                )

            metrics = self.metrics[query_hash]
            metrics.update_execution(execution_time)

            return metrics

    def _determine_query_type(self, query: str) -> QueryType:
        """Determine query type from SQL."""
        query_lower = query.strip().lower()

        if query_lower.startswith("select"):
            return QueryType.SELECT
        elif query_lower.startswith("insert"):
            return QueryType.INSERT
        elif query_lower.startswith("update"):
            return QueryType.UPDATE
        elif query_lower.startswith("delete"):
            return QueryType.DELETE
        elif query_lower.startswith("create"):
            return QueryType.CREATE
        elif query_lower.startswith("drop"):
            return QueryType.DROP
        else:
            return QueryType.SELECT  # Default

    async def _evict_oldest_metrics(self) -> None:
        """Evict oldest query metrics."""
        if not self.metrics:
            return

        # Remove queries that haven't been executed recently
        oldest_key = min(
            self.metrics.keys(), key=lambda k: self.metrics[k].last_executed
        )

        del self.metrics[oldest_key]

    def get_slow_queries(self, threshold: float = 1.0) -> list[QueryMetrics]:
        """Get queries that exceed the threshold.

        Args:
            threshold: Time threshold in seconds

        Returns:
            List of slow query metrics
        """
        return [
            metrics for metrics in self.metrics.values() if metrics.avg_time > threshold
        ]

    def get_most_frequent_queries(self, limit: int = 10) -> list[QueryMetrics]:
        """Get most frequently executed queries.

        Args:
            limit: Number of queries to return

        Returns:
            List of query metrics sorted by execution count
        """
        return sorted(
            self.metrics.values(), key=lambda m: m.execution_count, reverse=True
        )[:limit]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {"total_queries": 0}

        total_queries = sum(m.execution_count for m in self.metrics.values())
        total_time = sum(m.total_time for m in self.metrics.values())
        avg_time = total_time / total_queries if total_queries > 0 else 0

        slow_queries = self.get_slow_queries()

        query_types = defaultdict(int)
        for metrics in self.metrics.values():
            query_types[metrics.query_type.value] += metrics.execution_count

        return {
            "total_queries": total_queries,
            "unique_queries": len(self.metrics),
            "total_time": total_time,
            "avg_time": avg_time,
            "slow_queries": len(slow_queries),
            "query_types": dict(query_types),
            "slowest_query": max(
                self.metrics.values(), key=lambda m: m.avg_time
            ).avg_time
            if self.metrics
            else 0,
        }


class QueryOptimizer:
    """Main query optimization service."""

    def __init__(
        self,
        engine: Engine | AsyncEngine,
        cache_size: int = 1000,
        cache_ttl: float = 3600,
    ):
        """Initialize query optimizer.

        Args:
            engine: Database engine
            cache_size: Cache size
            cache_ttl: Cache TTL in seconds
        """
        self.engine = engine
        self.cache = QueryCache(max_size=cache_size, default_ttl=cache_ttl)
        self.analyzer = QueryAnalyzer()
        self.index_manager = IndexManager(engine)
        self.performance_tracker = QueryPerformanceTracker()

        # Weak reference to avoid circular references
        self._engines = weakref.WeakSet()
        self._engines.add(engine)

    async def execute_with_optimization(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        cache_ttl: float | None = None,
    ) -> Any:
        """Execute query with optimization.

        Args:
            query: SQL query
            params: Query parameters
            use_cache: Whether to use cache
            cache_ttl: Cache TTL override

        Returns:
            Query result
        """
        start_time = time.time()

        # Try cache first
        if use_cache:
            cached_result = await self.cache.get(query, params)
            if cached_result is not None:
                await self.performance_tracker.record_query(
                    query, 0.001, params
                )  # Minimal time for cache hit
                return cached_result

        # Execute query
        try:
            if hasattr(self.engine, "connect"):
                # Sync engine
                with self.engine.connect() as conn:
                    result = conn.execute(text(query), params or {})
                    result_data = (
                        result.fetchall() if result.returns_rows else result.rowcount
                    )
            else:
                # Async engine
                async with self.engine.connect() as conn:
                    result = await conn.execute(text(query), params or {})
                    result_data = (
                        await result.fetchall()
                        if result.returns_rows
                        else result.rowcount
                    )

            execution_time = time.time() - start_time

            # Cache result if appropriate
            if use_cache and self._should_cache_query(query, execution_time):
                await self.cache.set(query, result_data, params, cache_ttl)

            # Track performance
            await self.performance_tracker.record_query(query, execution_time, params)

            # Analyze query
            analysis = self.analyzer.analyze_query(query, execution_time, params)

            # Log slow queries
            if analysis["is_slow"]:
                logger.warning(
                    "Slow query detected",
                    query=query[:200],
                    execution_time=execution_time,
                    recommendations=analysis["recommendations"],
                )

            return result_data

        except Exception as e:
            execution_time = time.time() - start_time
            await self.performance_tracker.record_query(query, execution_time, params)
            logger.error("Query execution failed", query=query, error=str(e))
            raise

    def _should_cache_query(self, query: str, execution_time: float) -> bool:
        """Determine if query result should be cached."""
        query_lower = query.lower().strip()

        # Don't cache write operations
        if any(
            op in query_lower
            for op in ["insert", "update", "delete", "create", "drop", "alter"]
        ):
            return False

        # Cache expensive queries
        if execution_time > 0.5:
            return True

        # Cache frequently used patterns
        if any(
            pattern in query_lower
            for pattern in ["count(", "sum(", "avg(", "max(", "min("]
        ):
            return True

        return True  # Cache by default for SELECT queries

    async def optimize_database(self) -> dict[str, Any]:
        """Perform database optimization."""
        optimization_results = {
            "indexes_created": 0,
            "recommendations": [],
            "performance_summary": {},
            "cache_stats": {},
        }

        try:
            # Get index recommendations
            recommendations = self.analyzer.get_index_recommendations()

            # Create recommended indexes
            indexes_created = 0
            for rec in recommendations:
                if self.index_manager.create_index_if_not_exists(
                    rec.table_name, rec.columns
                ):
                    indexes_created += 1

            optimization_results["indexes_created"] = indexes_created
            optimization_results["recommendations"] = [
                {
                    "table": rec.table_name,
                    "columns": rec.columns,
                    "reason": rec.reason,
                    "estimated_benefit": rec.estimated_benefit,
                }
                for rec in recommendations
            ]

            # Get performance summary
            optimization_results["performance_summary"] = (
                self.performance_tracker.get_performance_summary()
            )

            # Get cache stats
            optimization_results["cache_stats"] = self.cache.get_stats()

            logger.info(
                "Database optimization completed",
                indexes_created=indexes_created,
                recommendations_count=len(recommendations),
            )

        except Exception as e:
            logger.error("Database optimization failed", error=str(e))
            optimization_results["error"] = str(e)

        return optimization_results

    async def get_optimization_report(self) -> dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            "performance_summary": self.performance_tracker.get_performance_summary(),
            "slow_queries": self.analyzer.get_slow_query_report(),
            "index_recommendations": [
                {
                    "table": rec.table_name,
                    "columns": rec.columns,
                    "reason": rec.reason,
                    "estimated_benefit": rec.estimated_benefit,
                }
                for rec in self.analyzer.get_index_recommendations()
            ],
            "cache_stats": self.cache.get_stats(),
            "index_usage": self.index_manager.get_index_usage_stats(),
            "most_frequent_queries": [
                {
                    "query_hash": m.query_hash,
                    "execution_count": m.execution_count,
                    "avg_time": m.avg_time,
                    "total_time": m.total_time,
                }
                for m in self.performance_tracker.get_most_frequent_queries()
            ],
        }

    async def clear_cache(self) -> None:
        """Clear query cache."""
        await self.cache.clear()
        logger.info("Query cache cleared")

    async def close(self) -> None:
        """Close optimizer and cleanup resources."""
        await self.cache.close()
        logger.info("Query optimizer closed")


# Global query optimizer
_query_optimizer: QueryOptimizer | None = None


def get_query_optimizer() -> QueryOptimizer | None:
    """Get global query optimizer instance."""
    global _query_optimizer
    return _query_optimizer


def set_query_optimizer(optimizer: QueryOptimizer) -> None:
    """Set global query optimizer instance."""
    global _query_optimizer
    _query_optimizer = optimizer
