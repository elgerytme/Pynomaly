"""Async database repository implementations using SQLAlchemy."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, Column, DateTime, String, Text, delete, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import VARCHAR, TypeDecorator

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.shared.protocols import (
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)

Base = declarative_base()


class JSONType(TypeDecorator):
    """JSON type that works with both PostgreSQL and SQLite."""

    impl = VARCHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value


class UUIDType(TypeDecorator):
    """UUID type that works with both PostgreSQL and SQLite."""

    impl = String
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PGUUID())
        else:
            return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return UUID(value)
        return value


class DatasetModel(Base):
    """SQLAlchemy model for Dataset entity."""

    __tablename__ = "datasets"

    id = Column(UUIDType, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    file_path = Column(String(500))
    target_column = Column(String(100))
    features = Column(JSONType)
    entity_metadata = Column("metadata", JSONType)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class DetectorModel(Base):
    """SQLAlchemy model for Detector entity."""

    __tablename__ = "detectors"

    id = Column(UUIDType, primary_key=True)
    algorithm = Column(String(100), nullable=False)
    parameters = Column(JSONType)
    is_fitted = Column(Boolean, default=False)
    model_data = Column(Text)  # Serialized model
    entity_metadata = Column("metadata", JSONType)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class DetectionResultModel(Base):
    """SQLAlchemy model for DetectionResult entity."""

    __tablename__ = "detection_results"

    id = Column(UUIDType, primary_key=True)
    detector_id = Column(UUIDType, nullable=False)
    dataset_id = Column(UUIDType, nullable=False)
    scores = Column(JSONType)
    labels = Column(JSONType)
    entity_metadata = Column("metadata", JSONType)
    created_at = Column(DateTime, nullable=False)


class AsyncDatabaseDetectorRepository(DetectorRepositoryProtocol):
    """Async database-backed detector repository."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """Initialize with async session factory."""
        self.session_factory = session_factory

    async def save(self, detector: Detector) -> None:
        """Save detector to database."""
        async with self.session_factory() as session:
            stmt = select(DetectorModel).filter_by(id=detector.id)
            result = await session.execute(stmt)
            existing = result.scalars().first()

            if existing:
                # Update existing
                existing.algorithm = detector.algorithm_name
                existing.parameters = detector.parameters
                existing.is_fitted = detector.is_fitted
                existing.model_data = (
                    detector.model_data if hasattr(detector, "model_data") else None
                )
                existing.entity_metadata = detector.metadata
                existing.updated_at = detector.updated_at
            else:
                # Insert new
                model = DetectorModel(
                    id=detector.id,
                    algorithm=detector.algorithm_name,
                    parameters=detector.parameters,
                    is_fitted=detector.is_fitted,
                    model_data=(
                        detector.model_data if hasattr(detector, "model_data") else None
                    ),
                    entity_metadata=detector.metadata,
                    created_at=detector.created_at,
                    updated_at=detector.updated_at,
                )
                session.add(model)

            await session.commit()

    async def find_by_id(self, detector_id: UUID) -> Detector | None:
        """Find detector by ID."""
        async with self.session_factory() as session:
            stmt = select(DetectorModel).filter_by(id=detector_id)
            result = await session.execute(stmt)
            model = result.scalars().first()

            if not model:
                return None

            return self._model_to_entity(model)

    async def find_by_name(self, name: str) -> Detector | None:
        """Find detector by name."""
        async with self.session_factory() as session:
            stmt = select(DetectorModel)
            result = await session.execute(stmt)
            models = result.scalars().all()

            # Search by metadata for now
            for model in models:
                if model.entity_metadata and model.entity_metadata.get("name") == name:
                    return self._model_to_entity(model)
            return None

    async def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
        """Find detectors by algorithm."""
        async with self.session_factory() as session:
            stmt = select(DetectorModel).filter_by(algorithm=algorithm_name)
            result = await session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_entity(model) for model in models]

    async def find_fitted(self) -> list[Detector]:
        """Find all fitted detectors."""
        async with self.session_factory() as session:
            stmt = select(DetectorModel).filter_by(is_fitted=True)
            result = await session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_entity(model) for model in models]

    async def find_all(self) -> list[Detector]:
        """Find all detectors."""
        async with self.session_factory() as session:
            stmt = select(DetectorModel)
            result = await session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_entity(model) for model in models]

    async def delete(self, detector_id: UUID) -> bool:
        """Delete detector by ID."""
        async with self.session_factory() as session:
            stmt = delete(DetectorModel).filter_by(id=detector_id)
            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    async def exists(self, detector_id: UUID) -> bool:
        """Check if detector exists."""
        async with self.session_factory() as session:
            stmt = select(DetectorModel).filter_by(id=detector_id)
            result = await session.execute(stmt)

            return result.scalars().first() is not None

    async def count(self) -> int:
        """Count total number of detectors."""
        async with self.session_factory() as session:
            stmt = select(DetectorModel)
            result = await session.execute(stmt)
            models = result.scalars().all()

            return len(models)

    async def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        """Save the trained model artifact."""
        async with self.session_factory() as session:
            stmt = select(DetectorModel).filter_by(id=detector_id)
            result = await session.execute(stmt)
            model = result.scalars().first()

            if model:
                model.model_data = (
                    artifact.decode("utf-8")
                    if isinstance(artifact, bytes)
                    else str(artifact)
                )
                await session.commit()

    async def load_model_artifact(self, detector_id: UUID) -> bytes | None:
        """Load the trained model artifact."""
        async with self.session_factory() as session:
            stmt = select(DetectorModel).filter_by(id=detector_id)
            result = await session.execute(stmt)
            model = result.scalars().first()

            if model and model.model_data:
                return (
                    model.model_data.encode("utf-8")
                    if isinstance(model.model_data, str)
                    else model.model_data
                )
            return None

    def _model_to_entity(self, model: DetectorModel) -> Detector:
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


class AsyncDatabaseDatasetRepository(DatasetRepositoryProtocol):
    """Async database-backed dataset repository."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """Initialize with async session factory."""
        self.session_factory = session_factory

    async def save(self, dataset: Dataset) -> None:
        """Save dataset to database."""
        async with self.session_factory() as session:
            stmt = select(DatasetModel).filter_by(id=dataset.id)
            result = await session.execute(stmt)
            existing = result.scalars().first()

            if existing:
                # Update existing
                existing.name = dataset.name
                existing.description = dataset.description
                existing.file_path = dataset.file_path
                existing.target_column = dataset.target_column
                existing.features = dataset.features
                existing.entity_metadata = dataset.metadata
                existing.updated_at = dataset.updated_at
            else:
                # Insert new
                model = DatasetModel(
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
                session.add(model)

            await session.commit()

    async def find_by_id(self, dataset_id: UUID) -> Dataset | None:
        """Find dataset by ID."""
        async with self.session_factory() as session:
            stmt = select(DatasetModel).filter_by(id=dataset_id)
            result = await session.execute(stmt)
            model = result.scalars().first()

            if not model:
                return None

            return self._model_to_entity(model)

    async def find_by_name(self, name: str) -> Dataset | None:
        """Find dataset by name."""
        async with self.session_factory() as session:
            stmt = select(DatasetModel).filter_by(name=name)
            result = await session.execute(stmt)
            model = result.scalars().first()

            if not model:
                return None

            return self._model_to_entity(model)

    async def find_by_metadata(self, key: str, value: Any) -> list[Dataset]:
        """Find datasets by metadata key-value pair."""
        async with self.session_factory() as session:
            stmt = select(DatasetModel)
            result = await session.execute(stmt)
            models = result.scalars().all()

            matching_models = []
            for model in models:
                if model.entity_metadata and model.entity_metadata.get(key) == value:
                    matching_models.append(model)

            return [self._model_to_entity(model) for model in matching_models]

    async def find_all(self) -> list[Dataset]:
        """Find all datasets."""
        async with self.session_factory() as session:
            stmt = select(DatasetModel)
            result = await session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_entity(model) for model in models]

    async def delete(self, dataset_id: UUID) -> bool:
        """Delete dataset by ID."""
        async with self.session_factory() as session:
            stmt = delete(DatasetModel).filter_by(id=dataset_id)
            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    async def exists(self, dataset_id: UUID) -> bool:
        """Check if dataset exists."""
        async with self.session_factory() as session:
            stmt = select(DatasetModel).filter_by(id=dataset_id)
            result = await session.execute(stmt)

            return result.scalars().first() is not None

    async def count(self) -> int:
        """Count total number of datasets."""
        async with self.session_factory() as session:
            stmt = select(DatasetModel)
            result = await session.execute(stmt)
            models = result.scalars().all()

            return len(models)

    async def save_data(self, dataset_id: UUID, format: str = "parquet") -> str:
        """Save dataset data to persistent storage."""
        async with self.session_factory() as session:
            stmt = select(DatasetModel).filter_by(id=dataset_id)
            result = await session.execute(stmt)
            model = result.scalars().first()

            if not model:
                raise ValueError(f"Dataset {dataset_id} not found")

            # For now, just update metadata to indicate data was saved
            if not model.entity_metadata:
                model.entity_metadata = {}
            model.entity_metadata["data_saved"] = True
            model.entity_metadata["data_format"] = format
            model.entity_metadata["data_saved_at"] = datetime.utcnow().isoformat()
            await session.commit()

            return f"database://{dataset_id}.{format}"

    async def load_data(self, dataset_id: UUID) -> Dataset | None:
        """Load dataset with its data from storage."""
        # For database implementation, we don't store the actual data
        # This would typically load from file system or object storage
        return await self.find_by_id(dataset_id)

    def _model_to_entity(self, model: DatasetModel) -> Dataset:
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


class AsyncDatabaseDetectionResultRepository(DetectionResultRepositoryProtocol):
    """Async database-backed detection result repository."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """Initialize with async session factory."""
        self.session_factory = session_factory

    async def save(self, result: DetectionResult) -> None:
        """Save detection result to database."""
        async with self.session_factory() as session:
            # Serialize scores for storage
            scores_data = []
            if hasattr(result, "scores") and result.scores:
                scores_data = [
                    {
                        "value": score.value,
                        "confidence": getattr(score, "confidence", None),
                    }
                    for score in result.scores
                ]

            stmt = select(DetectionResultModel).filter_by(id=result.id)
            db_result = await session.execute(stmt)
            existing = db_result.scalars().first()

            if existing:
                # Update existing
                existing.scores = scores_data
                existing.labels = getattr(result, "labels", [])
                existing.entity_metadata = result.metadata
            else:
                # Insert new
                model = DetectionResultModel(
                    id=result.id,
                    detector_id=result.detector_id,
                    dataset_id=result.dataset_id,
                    scores=scores_data,
                    labels=getattr(result, "labels", []),
                    entity_metadata=result.metadata,
                    created_at=result.timestamp,
                )
                session.add(model)

            await session.commit()

    async def find_by_id(self, result_id: UUID) -> DetectionResult | None:
        """Find detection result by ID."""
        async with self.session_factory() as session:
            stmt = select(DetectionResultModel).filter_by(id=result_id)
            result = await session.execute(stmt)
            model = result.scalars().first()

            if not model:
                return None

            return self._model_to_entity(model)

    async def find_by_detector(self, detector_id: UUID) -> list[DetectionResult]:
        """Find detection results by detector ID."""
        async with self.session_factory() as session:
            stmt = select(DetectionResultModel).filter_by(detector_id=detector_id)
            result = await session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_entity(model) for model in models]

    async def find_by_dataset(self, dataset_id: UUID) -> list[DetectionResult]:
        """Find detection results by dataset ID."""
        async with self.session_factory() as session:
            stmt = select(DetectionResultModel).filter_by(dataset_id=dataset_id)
            result = await session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_entity(model) for model in models]

    async def find_recent(self, limit: int = 10) -> list[DetectionResult]:
        """Find most recent detection results."""
        async with self.session_factory() as session:
            stmt = (
                select(DetectionResultModel)
                .order_by(DetectionResultModel.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_entity(model) for model in models]

    async def find_all(self) -> list[DetectionResult]:
        """Find all detection results."""
        async with self.session_factory() as session:
            stmt = select(DetectionResultModel)
            result = await session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_entity(model) for model in models]

    async def delete(self, result_id: UUID) -> bool:
        """Delete detection result by ID."""
        async with self.session_factory() as session:
            stmt = delete(DetectionResultModel).filter_by(id=result_id)
            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    async def exists(self, result_id: UUID) -> bool:
        """Check if detection result exists."""
        async with self.session_factory() as session:
            stmt = select(DetectionResultModel).filter_by(id=result_id)
            result = await session.execute(stmt)

            return result.scalars().first() is not None

    async def count(self) -> int:
        """Count total number of detection results."""
        async with self.session_factory() as session:
            stmt = select(DetectionResultModel)
            result = await session.execute(stmt)
            models = result.scalars().all()

            return len(models)

    async def get_summary_stats(self, result_id: UUID) -> dict[str, Any]:
        """Get summary statistics for a result."""
        async with self.session_factory() as session:
            stmt = select(DetectionResultModel).filter_by(id=result_id)
            result = await session.execute(stmt)
            model = result.scalars().first()

            if not model:
                return {}

            detection_result = self._model_to_entity(model)

            # Calculate statistics from the result
            stats = {
                "id": str(detection_result.id),
                "detector_id": str(detection_result.detector_id),
                "dataset_id": str(detection_result.dataset_id),
                "timestamp": detection_result.timestamp.isoformat(),
            }

            # Add computed statistics if available
            if hasattr(detection_result, "n_samples"):
                stats.update(
                    {
                        "n_samples": detection_result.n_samples,
                        "n_anomalies": getattr(detection_result, "n_anomalies", 0),
                        "anomaly_rate": getattr(detection_result, "anomaly_rate", 0.0),
                        "threshold": getattr(detection_result, "threshold", None),
                        "execution_time_ms": getattr(
                            detection_result, "execution_time_ms", None
                        ),
                        "score_statistics": getattr(
                            detection_result, "score_statistics", {}
                        ),
                        "has_confidence_intervals": getattr(
                            detection_result, "has_confidence_intervals", False
                        ),
                    }
                )

            return stats

    def _model_to_entity(self, model: DetectionResultModel) -> DetectionResult:
        """Convert database model to domain entity."""
        # Deserialize scores
        scores = []
        if model.scores:
            from pynomaly.domain.value_objects import AnomalyScore

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
