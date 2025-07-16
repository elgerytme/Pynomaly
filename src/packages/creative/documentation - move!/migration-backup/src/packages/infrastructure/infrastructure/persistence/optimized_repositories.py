"""Optimized database repository implementations with performance enhancements."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, Index, and_, func, or_, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import Select

from monorepo.domain.entities import Dataset, DetectionResult, Detector
from monorepo.shared.protocols import (
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)

from .database_repositories import DatasetModel, DetectionResultModel, DetectorModel

logger = logging.getLogger(__name__)


# Enhanced Models with Indexes


class OptimizedDatasetModel(DatasetModel):
    """Enhanced Dataset model with performance indexes."""

    __table_args__ = (
        Index("idx_datasets_name", "name"),
        Index("idx_datasets_created_at", "created_at"),
        Index("idx_datasets_updated_at", "updated_at"),
        Index("idx_datasets_target_column", "target_column"),
        # Composite indexes for common query patterns
        Index("idx_datasets_name_created", "name", "created_at"),
        # PostgreSQL-specific JSON indexes
        Index("idx_datasets_metadata_gin", "metadata", postgresql_using="gin"),
    )


class OptimizedDetectorModel(DetectorModel):
    """Enhanced Detector model with performance indexes."""

    __table_args__ = (
        Index("idx_detectors_algorithm", "algorithm"),
        Index("idx_detectors_is_fitted", "is_fitted"),
        Index("idx_detectors_created_at", "created_at"),
        Index("idx_detectors_updated_at", "updated_at"),
        # Composite indexes for common query patterns
        Index("idx_detectors_algorithm_fitted", "algorithm", "is_fitted"),
        Index("idx_detectors_fitted_created", "is_fitted", "created_at"),
        # PostgreSQL-specific JSON indexes
        Index("idx_detectors_metadata_gin", "metadata", postgresql_using="gin"),
        Index("idx_detectors_parameters_gin", "parameters", postgresql_using="gin"),
    )


class OptimizedDetectionResultModel(DetectionResultModel):
    """Enhanced DetectionResult model with performance indexes."""

    __table_args__ = (
        Index("idx_results_detector_id", "detector_id"),
        Index("idx_results_dataset_id", "dataset_id"),
        Index("idx_results_created_at", "created_at"),
        # Composite indexes for common query patterns
        Index("idx_results_detector_created", "detector_id", "created_at"),
        Index("idx_results_dataset_created", "dataset_id", "created_at"),
        Index("idx_results_detector_dataset", "detector_id", "dataset_id"),
        # PostgreSQL-specific JSON indexes
        Index("idx_results_metadata_gin", "metadata", postgresql_using="gin"),
        Index("idx_results_scores_gin", "scores", postgresql_using="gin"),
    )


class QueryOptimizer:
    """Query optimization utilities."""

    @staticmethod
    def add_pagination(
        query: Select, page: int = 1, page_size: int = 50
    ) -> tuple[Select, dict[str, Any]]:
        """Add pagination to query with metadata."""
        offset = (page - 1) * page_size
        paginated_query = query.offset(offset).limit(page_size)

        pagination_info = {
            "page": page,
            "page_size": page_size,
            "offset": offset,
            "has_next": None,  # Will be determined after execution
            "has_prev": page > 1,
            "total_pages": None,  # Requires count query
        }

        return paginated_query, pagination_info

    @staticmethod
    def build_metadata_filter(model_class, key: str, value: Any, operator: str = "eq"):
        """Build efficient metadata filter based on database dialect."""
        metadata_column = getattr(model_class, "metadata", None) or getattr(
            model_class, "entity_metadata", None
        )

        if not metadata_column:
            return None

        # PostgreSQL JSONB operators
        if operator == "eq":
            return metadata_column.op("->>")("key") == str(value)
        elif operator == "contains":
            return metadata_column.op("@>")({key: value})
        elif operator == "exists":
            return metadata_column.op("?")(key)
        elif operator == "in":
            return metadata_column.op("->>")("key").in_([str(v) for v in value])
        else:
            # Fallback for SQLite or other databases
            return text(f"JSON_EXTRACT(metadata, '$.{key}') = :value").params(
                value=str(value)
            )


class OptimizedDetectorRepository(DetectorRepositoryProtocol):
    """Performance-optimized detector repository."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        enable_caching: bool = True,
    ):
        """Initialize with async session factory and optional caching."""
        self.session_factory = session_factory
        self.enable_caching = enable_caching
        self._cache = {} if enable_caching else None
        self._cache_ttl = timedelta(minutes=5)
        self._cache_timestamps = {} if enable_caching else None

    async def save(self, detector: Detector) -> None:
        """Save detector with optimized upsert."""
        async with self.session_factory() as session:
            # Use merge for efficient upsert
            model = OptimizedDetectorModel(
                id=detector.id,
                algorithm=detector.algorithm_name,
                parameters=detector.parameters,
                is_fitted=detector.is_fitted,
                model_data=getattr(detector, "model_data", None),
                entity_metadata=detector.metadata,
                created_at=detector.created_at,
                updated_at=detector.updated_at,
            )

            await session.merge(model)
            await session.commit()

            # Invalidate cache
            if self._cache is not None:
                self._invalidate_cache(detector.id)

    def find_by_id(self, detector_id: UUID) -> Detector | None:
        """Find detector by ID with caching."""
        cache_key = f"detector:{detector_id}"

        # Check cache first
        if self._cache is not None and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        with self.session_factory() as session:
            model = (
                session.query(OptimizedDetectorModel).filter_by(id=detector_id).first()
            )
            if not model:
                return None

            entity = self._model_to_entity(model)

            # Cache result
            if self._cache is not None:
                self._cache[cache_key] = entity
                self._cache_timestamps[cache_key] = datetime.utcnow()

            return entity

    def find_by_algorithm(
        self,
        algorithm_name: str,
        fitted_only: bool = False,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Detector], dict[str, Any]]:
        """Find detectors by algorithm with pagination and filtering."""
        with self.session_factory() as session:
            query = session.query(OptimizedDetectorModel).filter_by(
                algorithm=algorithm_name
            )

            if fitted_only:
                query = query.filter_by(is_fitted=True)

            # Add ordering for consistent pagination
            query = query.order_by(OptimizedDetectorModel.created_at.desc())

            # Get total count for pagination
            total_count = query.count()

            # Apply pagination
            paginated_query, pagination_info = QueryOptimizer.add_pagination(
                query, page, page_size
            )
            models = paginated_query.all()

            # Calculate pagination metadata
            pagination_info["total_count"] = total_count
            pagination_info["total_pages"] = (total_count + page_size - 1) // page_size
            pagination_info["has_next"] = page < pagination_info["total_pages"]

            entities = [self._model_to_entity(model) for model in models]
            return entities, pagination_info

    def find_fitted_by_algorithm_batch(
        self, algorithm_names: list[str]
    ) -> dict[str, list[Detector]]:
        """Efficiently find fitted detectors for multiple algorithms."""
        with self.session_factory() as session:
            # Single query for all algorithms
            models = (
                session.query(OptimizedDetectorModel)
                .filter(
                    and_(
                        OptimizedDetectorModel.algorithm.in_(algorithm_names),
                        OptimizedDetectorModel.is_fitted == True,
                    )
                )
                .order_by(
                    OptimizedDetectorModel.algorithm,
                    OptimizedDetectorModel.created_at.desc(),
                )
                .all()
            )

            # Group by algorithm
            result = {algo: [] for algo in algorithm_names}
            for model in models:
                if model.algorithm in result:
                    result[model.algorithm].append(self._model_to_entity(model))

            return result

    def find_by_metadata_optimized(
        self, filters: dict[str, Any], page: int = 1, page_size: int = 50
    ) -> tuple[list[Detector], dict[str, Any]]:
        """Optimized metadata search using database-specific features."""
        with self.session_factory() as session:
            query = session.query(OptimizedDetectorModel)

            # Build metadata filters
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_condition = QueryOptimizer.build_metadata_filter(
                        OptimizedDetectorModel, key, value, "in"
                    )
                else:
                    filter_condition = QueryOptimizer.build_metadata_filter(
                        OptimizedDetectorModel, key, value, "eq"
                    )

                if filter_condition is not None:
                    query = query.filter(filter_condition)

            # Add ordering
            query = query.order_by(OptimizedDetectorModel.created_at.desc())

            # Get total count
            total_count = query.count()

            # Apply pagination
            paginated_query, pagination_info = QueryOptimizer.add_pagination(
                query, page, page_size
            )
            models = paginated_query.all()

            # Calculate pagination metadata
            pagination_info["total_count"] = total_count
            pagination_info["total_pages"] = (total_count + page_size - 1) // page_size
            pagination_info["has_next"] = page < pagination_info["total_pages"]

            entities = [self._model_to_entity(model) for model in models]
            return entities, pagination_info

    def get_algorithm_summary(self) -> dict[str, Any]:
        """Get algorithm usage summary with aggregated statistics."""
        with self.session_factory() as session:
            # Aggregate query for algorithm statistics
            results = (
                session.query(
                    OptimizedDetectorModel.algorithm,
                    func.count().label("total_count"),
                    func.sum(OptimizedDetectorModel.is_fitted.cast(Boolean)).label(
                        "fitted_count"
                    ),
                    func.max(OptimizedDetectorModel.created_at).label("latest_created"),
                    func.min(OptimizedDetectorModel.created_at).label(
                        "earliest_created"
                    ),
                )
                .group_by(OptimizedDetectorModel.algorithm)
                .order_by(func.count().desc())
                .all()
            )

            summary = {}
            for result in results:
                algorithm = result.algorithm
                summary[algorithm] = {
                    "total_detectors": result.total_count,
                    "fitted_detectors": result.fitted_count or 0,
                    "unfitted_detectors": result.total_count
                    - (result.fitted_count or 0),
                    "fitted_percentage": (result.fitted_count or 0)
                    / result.total_count
                    * 100,
                    "latest_created": result.latest_created.isoformat()
                    if result.latest_created
                    else None,
                    "earliest_created": result.earliest_created.isoformat()
                    if result.earliest_created
                    else None,
                }

            return summary

    def bulk_update_fitted_status(
        self, detector_ids: list[UUID], is_fitted: bool
    ) -> int:
        """Bulk update fitted status for multiple detectors."""
        with self.session_factory() as session:
            updated_count = (
                session.query(OptimizedDetectorModel)
                .filter(OptimizedDetectorModel.id.in_(detector_ids))
                .update(
                    {
                        OptimizedDetectorModel.is_fitted: is_fitted,
                        OptimizedDetectorModel.updated_at: datetime.utcnow(),
                    },
                    synchronize_session=False,
                )
            )
            session.commit()

            # Invalidate cache for updated detectors
            if self._cache is not None:
                for detector_id in detector_ids:
                    self._invalidate_cache(detector_id)

            return updated_count

    def delete_old_unfitted(self, days_old: int = 7) -> int:
        """Delete old unfitted detectors to clean up storage."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        with self.session_factory() as session:
            deleted_count = (
                session.query(OptimizedDetectorModel)
                .filter(
                    and_(
                        OptimizedDetectorModel.is_fitted == False,
                        OptimizedDetectorModel.created_at < cutoff_date,
                    )
                )
                .delete(synchronize_session=False)
            )
            session.commit()

            logger.info(f"Deleted {deleted_count} old unfitted detectors")
            return deleted_count

    def _model_to_entity(self, model: OptimizedDetectorModel) -> Detector:
        """Convert database model to domain entity."""
        return Detector(
            algorithm_name=model.algorithm,
            parameters=model.parameters or {},
            is_fitted=model.is_fitted,
            metadata=model.entity_metadata or {},
            id=model.id,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if not self._cache or cache_key not in self._cache:
            return False

        timestamp = self._cache_timestamps.get(cache_key)
        if not timestamp:
            return False

        return datetime.utcnow() - timestamp < self._cache_ttl

    def _invalidate_cache(self, detector_id: UUID) -> None:
        """Invalidate cache entries for a detector."""
        if not self._cache:
            return

        cache_key = f"detector:{detector_id}"
        self._cache.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)


class OptimizedDatasetRepository(DatasetRepositoryProtocol):
    """Performance-optimized dataset repository."""

    def __init__(self, session_factory: sessionmaker, enable_caching: bool = True):
        """Initialize with session factory and optional caching."""
        self.session_factory = session_factory
        self.enable_caching = enable_caching
        self._cache = {} if enable_caching else None
        self._cache_ttl = timedelta(minutes=10)  # Longer TTL for datasets
        self._cache_timestamps = {} if enable_caching else None

    def save(self, dataset: Dataset) -> None:
        """Save dataset with optimized upsert."""
        with self.session_factory() as session:
            model = OptimizedDatasetModel(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                file_path=dataset.file_path,
                target_column=dataset.target_column,
                features=dataset.features,
                entity_metadata=dataset.metadata,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at,
            )

            session.merge(model)
            session.commit()

            # Invalidate cache
            if self._cache is not None:
                self._invalidate_cache(dataset.id)

    def find_by_name_pattern(
        self, pattern: str, page: int = 1, page_size: int = 50
    ) -> tuple[list[Dataset], dict[str, Any]]:
        """Find datasets by name pattern with pagination."""
        with self.session_factory() as session:
            query = (
                session.query(OptimizedDatasetModel)
                .filter(OptimizedDatasetModel.name.ilike(f"%{pattern}%"))
                .order_by(OptimizedDatasetModel.created_at.desc())
            )

            # Get total count
            total_count = query.count()

            # Apply pagination
            paginated_query, pagination_info = QueryOptimizer.add_pagination(
                query, page, page_size
            )
            models = paginated_query.all()

            # Calculate pagination metadata
            pagination_info["total_count"] = total_count
            pagination_info["total_pages"] = (total_count + page_size - 1) // page_size
            pagination_info["has_next"] = page < pagination_info["total_pages"]

            entities = [self._model_to_entity(model) for model in models]
            return entities, pagination_info

    def find_by_features(
        self, features: list[str], match_all: bool = True
    ) -> list[Dataset]:
        """Find datasets that contain specific features."""
        with self.session_factory() as session:
            if match_all:
                # Dataset must contain all specified features
                query = session.query(OptimizedDatasetModel)
                for feature in features:
                    query = query.filter(
                        OptimizedDatasetModel.features.op("@>")([feature])
                    )
            else:
                # Dataset must contain at least one specified feature
                conditions = [
                    OptimizedDatasetModel.features.op("@>")([feature])
                    for feature in features
                ]
                query = session.query(OptimizedDatasetModel).filter(or_(*conditions))

            models = query.order_by(OptimizedDatasetModel.created_at.desc()).all()
            return [self._model_to_entity(model) for model in models]

    def get_datasets_summary(self) -> dict[str, Any]:
        """Get comprehensive datasets summary with statistics."""
        with self.session_factory() as session:
            # Basic counts
            total_datasets = session.query(OptimizedDatasetModel).count()

            # Feature statistics
            feature_usage = {}
            models = session.query(OptimizedDatasetModel.features).all()
            for model in models:
                if model.features:
                    for feature in model.features:
                        feature_usage[feature] = feature_usage.get(feature, 0) + 1

            # Most common features
            top_features = sorted(
                feature_usage.items(), key=lambda x: x[1], reverse=True
            )[:10]

            # Target column statistics
            target_columns = (
                session.query(
                    OptimizedDatasetModel.target_column, func.count().label("count")
                )
                .filter(OptimizedDatasetModel.target_column.isnot(None))
                .group_by(OptimizedDatasetModel.target_column)
                .order_by(func.count().desc())
                .limit(10)
                .all()
            )

            # Recent activity
            recent_datasets = (
                session.query(OptimizedDatasetModel)
                .order_by(OptimizedDatasetModel.created_at.desc())
                .limit(5)
                .all()
            )

            return {
                "total_datasets": total_datasets,
                "feature_statistics": {
                    "unique_features": len(feature_usage),
                    "total_feature_usage": sum(feature_usage.values()),
                    "average_features_per_dataset": sum(feature_usage.values())
                    / max(total_datasets, 1),
                    "top_features": [
                        {"feature": f, "usage_count": c} for f, c in top_features
                    ],
                },
                "target_column_distribution": [
                    {"target_column": tc.target_column, "count": tc.count}
                    for tc in target_columns
                ],
                "recent_datasets": [
                    {
                        "id": str(d.id),
                        "name": d.name,
                        "created_at": d.created_at.isoformat(),
                        "feature_count": len(d.features) if d.features else 0,
                    }
                    for d in recent_datasets
                ],
            }

    def _model_to_entity(self, model: OptimizedDatasetModel) -> Dataset:
        """Convert database model to domain entity."""
        return Dataset(
            name=model.name,
            data=None,  # Data loaded separately when needed
            description=model.description,
            file_path=model.file_path,
            target_column=model.target_column,
            features=model.features or [],
            metadata=model.entity_metadata or {},
            id=model.id,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _invalidate_cache(self, dataset_id: UUID) -> None:
        """Invalidate cache entries for a dataset."""
        if not self._cache:
            return

        cache_key = f"dataset:{dataset_id}"
        self._cache.pop(cache_key, None)
        self._cache_timestamps.pop(cache_key, None)


class OptimizedDetectionResultRepository(DetectionResultRepositoryProtocol):
    """Performance-optimized detection result repository."""

    def __init__(self, session_factory: sessionmaker):
        """Initialize with session factory."""
        self.session_factory = session_factory

    def find_by_detector_paginated(
        self,
        detector_id: UUID,
        page: int = 1,
        page_size: int = 50,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> tuple[list[DetectionResult], dict[str, Any]]:
        """Find detection results by detector with pagination and date filtering."""
        with self.session_factory() as session:
            query = session.query(OptimizedDetectionResultModel).filter_by(
                detector_id=detector_id
            )

            # Add date filtering
            if start_date:
                query = query.filter(
                    OptimizedDetectionResultModel.created_at >= start_date
                )
            if end_date:
                query = query.filter(
                    OptimizedDetectionResultModel.created_at <= end_date
                )

            # Order by creation date (newest first)
            query = query.order_by(OptimizedDetectionResultModel.created_at.desc())

            # Get total count
            total_count = query.count()

            # Apply pagination
            paginated_query, pagination_info = QueryOptimizer.add_pagination(
                query, page, page_size
            )
            models = paginated_query.all()

            # Calculate pagination metadata
            pagination_info["total_count"] = total_count
            pagination_info["total_pages"] = (total_count + page_size - 1) // page_size
            pagination_info["has_next"] = page < pagination_info["total_pages"]

            entities = [self._model_to_entity(model) for model in models]
            return entities, pagination_info

    def get_detection_statistics(
        self,
        detector_id: UUID | None = None,
        dataset_id: UUID | None = None,
        days_back: int = 30,
    ) -> dict[str, Any]:
        """Get comprehensive detection statistics."""
        start_date = datetime.utcnow() - timedelta(days=days_back)

        with self.session_factory() as session:
            query = session.query(OptimizedDetectionResultModel).filter(
                OptimizedDetectionResultModel.created_at >= start_date
            )

            if detector_id:
                query = query.filter_by(detector_id=detector_id)
            if dataset_id:
                query = query.filter_by(dataset_id=dataset_id)

            # Get basic counts
            total_results = query.count()

            # Results by day
            daily_results = (
                query.with_entities(
                    func.date(OptimizedDetectionResultModel.created_at).label("date"),
                    func.count().label("count"),
                )
                .group_by(func.date(OptimizedDetectionResultModel.created_at))
                .order_by(func.date(OptimizedDetectionResultModel.created_at))
                .all()
            )

            # Top detectors by usage
            detector_usage = (
                query.with_entities(
                    OptimizedDetectionResultModel.detector_id,
                    func.count().label("usage_count"),
                )
                .group_by(OptimizedDetectionResultModel.detector_id)
                .order_by(func.count().desc())
                .limit(10)
                .all()
            )

            # Top datasets by analysis
            dataset_usage = (
                query.with_entities(
                    OptimizedDetectionResultModel.dataset_id,
                    func.count().label("analysis_count"),
                )
                .group_by(OptimizedDetectionResultModel.dataset_id)
                .order_by(func.count().desc())
                .limit(10)
                .all()
            )

            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": datetime.utcnow().isoformat(),
                    "days": days_back,
                },
                "total_results": total_results,
                "daily_breakdown": [
                    {"date": r.date.isoformat(), "count": r.count}
                    for r in daily_results
                ],
                "top_detectors": [
                    {"detector_id": str(r.detector_id), "usage_count": r.usage_count}
                    for r in detector_usage
                ],
                "top_datasets": [
                    {
                        "dataset_id": str(r.dataset_id),
                        "analysis_count": r.analysis_count,
                    }
                    for r in dataset_usage
                ],
                "average_per_day": total_results / max(days_back, 1),
            }

    def cleanup_old_results(self, days_to_keep: int = 90) -> int:
        """Clean up old detection results to manage storage."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        with self.session_factory() as session:
            deleted_count = (
                session.query(OptimizedDetectionResultModel)
                .filter(OptimizedDetectionResultModel.created_at < cutoff_date)
                .delete(synchronize_session=False)
            )
            session.commit()

            logger.info(f"Cleaned up {deleted_count} old detection results")
            return deleted_count

    def _model_to_entity(self, model: OptimizedDetectionResultModel) -> DetectionResult:
        """Convert database model to domain entity."""
        # Deserialize scores
        scores = []
        if model.scores:
            from monorepo.domain.value_objects import AnomalyScore

            scores = [
                AnomalyScore(
                    value=score_data["value"], confidence=score_data.get("confidence")
                )
                for score_data in model.scores
            ]

        return DetectionResult(
            detector_id=model.detector_id,
            dataset_id=model.dataset_id,
            scores=scores,
            metadata=model.entity_metadata or {},
            id=model.id,
            timestamp=model.created_at,
        )


class DatabaseConnectionPool:
    """Optimized database connection pool management."""

    def __init__(self, database_url: str, pool_size: int = 10, max_overflow: int = 20):
        """Initialize connection pool with optimized settings."""
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool

        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Validate connections before use
            pool_recycle=3600,  # Recycle connections every hour
            connect_args={
                "connect_timeout": 30,
                "application_name": "pynomaly-optimized",
            }
            if "postgresql" in database_url
            else {},
        )

        self.session_factory = sessionmaker(bind=self.engine)

    def get_session_factory(self) -> sessionmaker:
        """Get the session factory."""
        return self.session_factory

    def get_engine(self):
        """Get the database engine."""
        return self.engine

    def close(self):
        """Close all connections in the pool."""
        self.engine.dispose()


class QueryPerformanceMonitor:
    """Monitor and log slow database queries."""

    def __init__(self, slow_query_threshold: float = 1.0):
        """Initialize with slow query threshold in seconds."""
        self.threshold = slow_query_threshold
        self.slow_queries = []

    def log_query(self, query: str, duration: float, params: dict | None = None):
        """Log query execution time."""
        if duration > self.threshold:
            slow_query = {
                "query": query,
                "duration": duration,
                "params": params,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.slow_queries.append(slow_query)
            logger.warning(f"Slow query detected: {duration:.2f}s - {query[:100]}...")

    def get_slow_queries(self) -> list[dict[str, Any]]:
        """Get list of slow queries."""
        return self.slow_queries.copy()

    def clear_slow_queries(self):
        """Clear the slow query log."""
        self.slow_queries.clear()
