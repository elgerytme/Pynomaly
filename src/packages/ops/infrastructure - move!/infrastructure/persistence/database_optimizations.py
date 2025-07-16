"""Database query optimizations and performance enhancements."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any
from uuid import UUID

from sqlalchemy import Index, and_, desc, func, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query, Session

from .database_repositories import (
    Base,
    DatasetModel,
    DetectionResultModel,
    DetectorModel,
    MetricModel,
    UserModel,
    UserRoleModel,
)

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """Database query optimizer with performance enhancements."""

    def __init__(self, session_factory):
        """Initialize optimizer with session factory."""
        self.session_factory = session_factory
        self._query_cache = {}
        self._cache_ttl = timedelta(minutes=5)

    def create_indexes(self, engine) -> None:
        """Create optimized indexes for common query patterns."""
        logger.info("Creating database indexes for performance optimization")

        indexes = [
            # Datasets table indexes
            Index("idx_datasets_name", DatasetModel.name),
            Index("idx_datasets_created_at", DatasetModel.created_at),
            Index("idx_datasets_updated_at", DatasetModel.updated_at),
            Index("idx_datasets_target_column", DatasetModel.target_column),
            # Detectors table indexes
            Index("idx_detectors_algorithm", DetectorModel.algorithm),
            Index("idx_detectors_is_fitted", DetectorModel.is_fitted),
            Index("idx_detectors_created_at", DetectorModel.created_at),
            Index(
                "idx_detectors_algorithm_fitted",
                DetectorModel.algorithm,
                DetectorModel.is_fitted,
            ),
            # Detection results table indexes
            Index("idx_results_detector_id", DetectionResultModel.detector_id),
            Index("idx_results_dataset_id", DetectionResultModel.dataset_id),
            Index("idx_results_created_at", DetectionResultModel.created_at),
            Index(
                "idx_results_detector_created",
                DetectionResultModel.detector_id,
                DetectionResultModel.created_at,
            ),
            Index(
                "idx_results_dataset_created",
                DetectionResultModel.dataset_id,
                DetectionResultModel.created_at,
            ),
            # Users table indexes
            Index("idx_users_email", UserModel.email),
            Index("idx_users_username", UserModel.username),
            Index("idx_users_status", UserModel.status),
            Index("idx_users_last_login", UserModel.last_login_at),
            # User roles table indexes
            Index("idx_user_roles_user_id", UserRoleModel.user_id),
            Index("idx_user_roles_tenant_id", UserRoleModel.tenant_id),
            Index("idx_user_roles_role_id", UserRoleModel.role_id),
            Index(
                "idx_user_roles_composite",
                UserRoleModel.user_id,
                UserRoleModel.tenant_id,
            ),
            # Metrics table indexes
            Index("idx_metrics_name", MetricModel.name),
            Index("idx_metrics_timestamp", MetricModel.timestamp),
            Index("idx_metrics_entity_type", MetricModel.entity_type),
            Index("idx_metrics_entity_id", MetricModel.entity_id),
            Index(
                "idx_metrics_name_timestamp", MetricModel.name, MetricModel.timestamp
            ),
            Index(
                "idx_metrics_entity_timestamp",
                MetricModel.entity_type,
                MetricModel.entity_id,
                MetricModel.timestamp,
            ),
        ]

        # Create indexes that don't already exist
        for index in indexes:
            try:
                index.create(engine, checkfirst=True)
                logger.debug(f"Created index: {index.name}")
            except Exception as e:
                logger.warning(f"Failed to create index {index.name}: {e}")

    @contextmanager
    def optimized_session(self):
        """Create session with optimized settings."""
        session = self.session_factory()
        try:
            # Configure session for better performance
            session.execute(text("PRAGMA temp_store = memory"))  # For SQLite
            session.execute(text("PRAGMA cache_size = 10000"))  # For SQLite
            session.execute(text("PRAGMA journal_mode = WAL"))  # For SQLite
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def enable_query_logging(self, level: str = "INFO") -> None:
        """Enable SQL query logging for performance analysis."""
        sql_logger = logging.getLogger("sqlalchemy.engine")
        sql_logger.setLevel(getattr(logging, level))

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        sql_logger.addHandler(handler)

    def analyze_query_performance(
        self, query: Query, session: Session
    ) -> dict[str, Any]:
        """Analyze query performance and suggest optimizations."""
        import time

        # Get query plan (for PostgreSQL/SQLite)
        try:
            plan_query = f"EXPLAIN QUERY PLAN {query.statement.compile(compile_kwargs={'literal_binds': True})}"
            plan_result = session.execute(text(plan_query)).fetchall()
            plan = [dict(row._mapping) for row in plan_result]
        except Exception:
            plan = []

        # Measure execution time
        start_time = time.perf_counter()
        result = query.all()
        execution_time = time.perf_counter() - start_time

        analysis = {
            "execution_time_ms": execution_time * 1000,
            "result_count": len(result),
            "query_plan": plan,
            "suggestions": self._generate_optimization_suggestions(
                query, execution_time, len(result)
            ),
        }

        return analysis

    def _generate_optimization_suggestions(
        self, query: Query, execution_time: float, result_count: int
    ) -> list[str]:
        """Generate optimization suggestions based on query analysis."""
        suggestions = []

        if execution_time > 1.0:  # Slow query
            suggestions.append("Consider adding appropriate indexes")
            suggestions.append("Review WHERE clause conditions")

        if result_count > 1000:
            suggestions.append("Consider implementing pagination")
            suggestions.append("Use LIMIT clause to reduce result set")

        # Analyze query structure
        query_str = str(query.statement.compile(compile_kwargs={"literal_binds": True}))

        if "JOIN" in query_str.upper() and execution_time > 0.5:
            suggestions.append(
                "Consider using eager loading or selectinload for relationships"
            )

        if "ORDER BY" in query_str.upper() and not any(
            "idx_" in str(col) for col in query.column_descriptions
        ):
            suggestions.append("Consider adding index on ORDER BY columns")

        return suggestions


class OptimizedRepositoryMixin:
    """Mixin class providing optimized query methods for repositories."""

    def __init__(self, session_factory, optimizer: DatabaseOptimizer = None):
        """Initialize with session factory and optional optimizer."""
        self.session_factory = session_factory
        self.optimizer = optimizer or DatabaseOptimizer(session_factory)

    def find_with_pagination(
        self, query: Query, page: int = 1, page_size: int = 20
    ) -> tuple[list[Any], int]:
        """Execute paginated query with total count."""
        # Get total count (optimized)
        count_query = query.statement.with_only_columns([func.count()]).order_by(None)
        total_count = query.session.execute(count_query).scalar()

        # Apply pagination
        offset = (page - 1) * page_size
        items = query.offset(offset).limit(page_size).all()

        return items, total_count

    def bulk_insert(
        self, session: Session, model_class, items: list[dict[str, Any]]
    ) -> None:
        """Optimized bulk insert operation."""
        if not items:
            return

        try:
            session.bulk_insert_mappings(model_class, items)
            session.commit()
            logger.debug(f"Bulk inserted {len(items)} {model_class.__name__} records")
        except IntegrityError as e:
            session.rollback()
            logger.error(f"Bulk insert failed: {e}")
            raise

    def bulk_update(
        self,
        session: Session,
        model_class,
        items: list[dict[str, Any]],
        update_fields: list[str],
    ) -> None:
        """Optimized bulk update operation."""
        if not items:
            return

        try:
            session.bulk_update_mappings(model_class, items)
            session.commit()
            logger.debug(f"Bulk updated {len(items)} {model_class.__name__} records")
        except IntegrityError as e:
            session.rollback()
            logger.error(f"Bulk update failed: {e}")
            raise

    def cached_query(self, cache_key: str, query_func, cache_ttl: timedelta = None):
        """Execute query with caching support."""
        cache_ttl = cache_ttl or self.optimizer._cache_ttl
        now = datetime.utcnow()

        # Check cache
        if cache_key in self.optimizer._query_cache:
            cached_data, cached_time = self.optimizer._query_cache[cache_key]
            if now - cached_time < cache_ttl:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data

        # Execute query and cache result
        result = query_func()
        self.optimizer._query_cache[cache_key] = (result, now)
        logger.debug(f"Cache miss for key: {cache_key}, cached new result")

        return result

    def clear_cache(self, pattern: str = None) -> None:
        """Clear query cache, optionally by pattern."""
        if pattern:
            keys_to_remove = [
                k for k in self.optimizer._query_cache.keys() if pattern in k
            ]
            for key in keys_to_remove:
                del self.optimizer._query_cache[key]
            logger.debug(
                f"Cleared {len(keys_to_remove)} cache entries matching pattern: {pattern}"
            )
        else:
            self.optimizer._query_cache.clear()
            logger.debug("Cleared all cache entries")


class OptimizedDetectorRepository(OptimizedRepositoryMixin):
    """Optimized detector repository with performance enhancements."""

    def find_by_algorithm_optimized(
        self,
        algorithm_name: str,
        include_fitted_only: bool = False,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Any], int]:
        """Find detectors by algorithm with optimization."""
        with self.optimizer.optimized_session() as session:
            query = session.query(DetectorModel).filter(
                DetectorModel.algorithm == algorithm_name
            )

            if include_fitted_only:
                query = query.filter(DetectorModel.is_fitted == True)

            # Order by most recently updated
            query = query.order_by(desc(DetectorModel.updated_at))

            return self.find_with_pagination(query, page, page_size)

    def find_recent_fitted(self, limit: int = 10) -> list[Any]:
        """Find recently fitted detectors with caching."""
        cache_key = f"recent_fitted_detectors_{limit}"

        def query_func():
            with self.optimizer.optimized_session() as session:
                return (
                    session.query(DetectorModel)
                    .filter(DetectorModel.is_fitted == True)
                    .order_by(desc(DetectorModel.updated_at))
                    .limit(limit)
                    .all()
                )

        return self.cached_query(cache_key, query_func, timedelta(minutes=2))

    def get_algorithm_stats(self) -> dict[str, Any]:
        """Get algorithm usage statistics with caching."""
        cache_key = "algorithm_statistics"

        def query_func():
            with self.optimizer.optimized_session() as session:
                # Optimized aggregation query
                stats = (
                    session.query(
                        DetectorModel.algorithm,
                        func.count(DetectorModel.id).label("total_count"),
                        func.sum(func.cast(DetectorModel.is_fitted, "integer")).label(
                            "fitted_count"
                        ),
                    )
                    .group_by(DetectorModel.algorithm)
                    .all()
                )

                return {
                    stat.algorithm: {
                        "total": stat.total_count,
                        "fitted": stat.fitted_count or 0,
                        "fitted_percentage": (stat.fitted_count or 0)
                        / stat.total_count
                        * 100,
                    }
                    for stat in stats
                }

        return self.cached_query(cache_key, query_func, timedelta(minutes=5))


class OptimizedDatasetRepository(OptimizedRepositoryMixin):
    """Optimized dataset repository with performance enhancements."""

    def find_by_metadata_optimized(
        self, filters: dict[str, Any], page: int = 1, page_size: int = 20
    ) -> tuple[list[Any], int]:
        """Find datasets by metadata with JSON optimization."""
        with self.optimizer.optimized_session() as session:
            query = session.query(DatasetModel)

            # Build efficient JSON queries based on database type
            for key, value in filters.items():
                # For PostgreSQL, use JSONB operators
                # For SQLite, use JSON_EXTRACT
                if session.bind.dialect.name == "postgresql":
                    query = query.filter(
                        DatasetModel.entity_metadata.op("->>")(key) == str(value)
                    )
                else:
                    query = query.filter(
                        func.json_extract(DatasetModel.entity_metadata, f"$.{key}")
                        == str(value)
                    )

            query = query.order_by(desc(DatasetModel.updated_at))

            return self.find_with_pagination(query, page, page_size)

    def find_by_features_optimized(self, required_features: list[str]) -> list[Any]:
        """Find datasets containing all required features."""
        cache_key = f"datasets_with_features_{hash(tuple(sorted(required_features)))}"

        def query_func():
            with self.optimizer.optimized_session() as session:
                query = session.query(DatasetModel)

                # Efficient feature matching
                for feature in required_features:
                    if session.bind.dialect.name == "postgresql":
                        query = query.filter(DatasetModel.features.op("@>")([feature]))
                    else:
                        # For SQLite, use a different approach
                        query = query.filter(
                            func.json_extract(DatasetModel.features, "$").like(
                                f"%{feature}%"
                            )
                        )

                return query.order_by(desc(DatasetModel.updated_at)).all()

        return self.cached_query(cache_key, query_func)

    def get_feature_usage_stats(self) -> dict[str, int]:
        """Get feature usage statistics across all datasets."""
        cache_key = "feature_usage_statistics"

        def query_func():
            with self.optimizer.optimized_session() as session:
                datasets = session.query(DatasetModel.features).all()

                feature_counts = {}
                for dataset in datasets:
                    if dataset.features:
                        for feature in dataset.features:
                            feature_counts[feature] = feature_counts.get(feature, 0) + 1

                return dict(
                    sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
                )

        return self.cached_query(cache_key, query_func, timedelta(minutes=10))


class OptimizedDetectionResultRepository(OptimizedRepositoryMixin):
    """Optimized detection result repository with performance enhancements."""

    def find_results_with_stats(
        self,
        detector_id: UUID = None,
        dataset_id: UUID = None,
        start_date: datetime = None,
        end_date: datetime = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Any], int, dict[str, Any]]:
        """Find detection results with aggregated statistics."""
        with self.optimizer.optimized_session() as session:
            query = session.query(DetectionResultModel)

            # Apply filters
            filters = []
            if detector_id:
                filters.append(DetectionResultModel.detector_id == detector_id)
            if dataset_id:
                filters.append(DetectionResultModel.dataset_id == dataset_id)
            if start_date:
                filters.append(DetectionResultModel.created_at >= start_date)
            if end_date:
                filters.append(DetectionResultModel.created_at <= end_date)

            if filters:
                query = query.filter(and_(*filters))

            # Get statistics (optimized single query)
            stats_query = session.query(
                func.count(DetectionResultModel.id).label("total_count"),
                func.min(DetectionResultModel.created_at).label("earliest"),
                func.max(DetectionResultModel.created_at).label("latest"),
                func.count(func.distinct(DetectionResultModel.detector_id)).label(
                    "unique_detectors"
                ),
                func.count(func.distinct(DetectionResultModel.dataset_id)).label(
                    "unique_datasets"
                ),
            )

            if filters:
                stats_query = stats_query.filter(and_(*filters))

            stats_result = stats_query.first()
            stats = {
                "total_count": stats_result.total_count or 0,
                "earliest_result": stats_result.earliest,
                "latest_result": stats_result.latest,
                "unique_detectors": stats_result.unique_detectors or 0,
                "unique_datasets": stats_result.unique_datasets or 0,
            }

            # Get paginated results
            query = query.order_by(desc(DetectionResultModel.created_at))
            results, total_count = self.find_with_pagination(query, page, page_size)

            return results, total_count, stats

    def get_detection_trends(self, days: int = 30) -> dict[str, Any]:
        """Get detection trends over time period with caching."""
        cache_key = f"detection_trends_{days}_days"

        def query_func():
            with self.optimizer.optimized_session() as session:
                start_date = datetime.utcnow() - timedelta(days=days)

                # Daily aggregation query
                daily_stats = (
                    session.query(
                        func.date(DetectionResultModel.created_at).label("date"),
                        func.count(DetectionResultModel.id).label("count"),
                        func.count(
                            func.distinct(DetectionResultModel.detector_id)
                        ).label("unique_detectors"),
                    )
                    .filter(DetectionResultModel.created_at >= start_date)
                    .group_by(func.date(DetectionResultModel.created_at))
                    .order_by("date")
                    .all()
                )

                return {
                    "daily_counts": [
                        {
                            "date": str(stat.date),
                            "count": stat.count,
                            "unique_detectors": stat.unique_detectors,
                        }
                        for stat in daily_stats
                    ],
                    "period_start": start_date.isoformat(),
                    "period_end": datetime.utcnow().isoformat(),
                }

        return self.cached_query(cache_key, query_func, timedelta(minutes=15))


# Query optimization decorators


def log_slow_queries(threshold_ms: float = 100.0):
    """Decorator to log slow database queries."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000

                if execution_time > threshold_ms:
                    logger.warning(
                        f"Slow query detected: {func.__name__} took {execution_time:.2f}ms "
                        f"(threshold: {threshold_ms}ms)"
                    )

                return result
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Query failed: {func.__name__} after {execution_time:.2f}ms - {e}"
                )
                raise

        return wrapper

    return decorator


def batch_queries(batch_size: int = 100):
    """Decorator to batch database operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract items from args/kwargs
            items = kwargs.get("items") or (args[1] if len(args) > 1 else [])

            if not items or len(items) <= batch_size:
                return func(*args, **kwargs)

            # Process in batches
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                batch_kwargs = kwargs.copy()
                batch_kwargs["items"] = batch

                batch_result = func(args[0], **batch_kwargs)
                if batch_result:
                    results.extend(
                        batch_result
                        if isinstance(batch_result, list)
                        else [batch_result]
                    )

            return results

        return wrapper

    return decorator


# Connection pool optimization
class DatabaseConnectionManager:
    """Manages database connections with optimization."""

    def __init__(self, database_url: str, **kwargs):
        """Initialize connection manager."""
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool

        # Optimized engine configuration
        engine_config = {
            "poolclass": QueuePool,
            "pool_size": kwargs.get("pool_size", 10),
            "max_overflow": kwargs.get("max_overflow", 20),
            "pool_timeout": kwargs.get("pool_timeout", 30),
            "pool_recycle": kwargs.get("pool_recycle", 3600),  # 1 hour
            "pool_pre_ping": True,  # Validate connections
            "echo": kwargs.get("echo", False),
            "echo_pool": kwargs.get("echo_pool", False),
        }

        # Add database-specific optimizations
        if "postgresql" in database_url:
            engine_config.update(
                {
                    "connect_args": {
                        "options": "-c default_transaction_isolation=read_committed"
                    }
                }
            )
        elif "sqlite" in database_url:
            engine_config.update(
                {"connect_args": {"check_same_thread": False, "timeout": 20}}
            )

        self.engine = create_engine(database_url, **engine_config)
        self.optimizer = DatabaseOptimizer(self.get_session_factory())

    def get_session_factory(self):
        """Get SQLAlchemy session factory."""
        from sqlalchemy.orm import sessionmaker

        return sessionmaker(bind=self.engine)

    def initialize_database(self):
        """Initialize database with optimizations."""
        # Create tables
        Base.metadata.create_all(self.engine)

        # Create performance indexes
        self.optimizer.create_indexes(self.engine)

        logger.info("Database initialized with performance optimizations")

    def get_connection_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
        }
