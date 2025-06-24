"""Database repository implementations using SQLAlchemy."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import Column, String, DateTime, Text, Float, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.types import TypeDecorator, VARCHAR

from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.exceptions import EntityNotFoundError
from pynomaly.shared.protocols import (
    DatasetRepositoryProtocol,
    DetectorRepositoryProtocol,
    DetectionResultRepositoryProtocol
)

Base = declarative_base()


class JSONType(TypeDecorator):
    """JSON type that works with both PostgreSQL and SQLite."""
    
    impl = VARCHAR
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
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
        if dialect.name == 'postgresql':
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
    entity_metadata = Column('metadata', JSONType)
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
    entity_metadata = Column('metadata', JSONType)
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
    entity_metadata = Column('metadata', JSONType)
    created_at = Column(DateTime, nullable=False)


class DatabaseDetectorRepository(DetectorRepositoryProtocol):
    """Database-backed detector repository."""
    
    def __init__(self, session_factory):
        """Initialize with session factory."""
        self.session_factory = session_factory
    
    def save(self, detector: Detector) -> None:
        """Save detector to database."""
        with self.session_factory() as session:
            existing = session.query(DetectorModel).filter_by(id=detector.id).first()
            if existing:
                # Update existing
                existing.algorithm = detector.algorithm_name
                existing.parameters = detector.parameters
                existing.is_fitted = detector.is_fitted
                existing.model_data = detector.model_data if hasattr(detector, 'model_data') else None
                existing.metadata = detector.metadata
                existing.updated_at = detector.updated_at
            else:
                # Insert new
                model = DetectorModel(
                    id=detector.id,
                    algorithm=detector.algorithm_name,
                    parameters=detector.parameters,
                    is_fitted=detector.is_fitted,
                    model_data=detector.model_data if hasattr(detector, 'model_data') else None,
                    metadata=detector.metadata,
                    created_at=detector.created_at,
                    updated_at=detector.updated_at
                )
                session.add(model)
            
            session.commit()
    
    def find_by_id(self, detector_id: UUID) -> Optional[Detector]:
        """Find detector by ID."""
        with self.session_factory() as session:
            model = session.query(DetectorModel).filter_by(id=detector_id).first()
            if not model:
                return None
            
            return self._model_to_entity(model)
    
    def find_by_name(self, name: str) -> Optional[Detector]:
        """Find detector by name."""
        with self.session_factory() as session:
            # Note: DetectorModel doesn't have a name field currently
            # We'll search by metadata for now
            models = session.query(DetectorModel).all()
            for model in models:
                if model.metadata and model.metadata.get('name') == name:
                    return self._model_to_entity(model)
            return None
    
    def find_by_algorithm(self, algorithm_name: str) -> List[Detector]:
        """Find detectors by algorithm."""
        with self.session_factory() as session:
            models = session.query(DetectorModel).filter_by(algorithm=algorithm_name).all()
            return [self._model_to_entity(model) for model in models]
    
    def find_fitted(self) -> List[Detector]:
        """Find all fitted detectors."""
        with self.session_factory() as session:
            models = session.query(DetectorModel).filter_by(is_fitted=True).all()
            return [self._model_to_entity(model) for model in models]
    
    def find_all(self) -> List[Detector]:
        """Find all detectors."""
        with self.session_factory() as session:
            models = session.query(DetectorModel).all()
            return [self._model_to_entity(model) for model in models]
    
    def delete(self, detector_id: UUID) -> bool:
        """Delete detector by ID."""
        with self.session_factory() as session:
            model = session.query(DetectorModel).filter_by(id=detector_id).first()
            if model:
                session.delete(model)
                session.commit()
                return True
            return False
    
    def exists(self, detector_id: UUID) -> bool:
        """Check if detector exists."""
        with self.session_factory() as session:
            return session.query(DetectorModel).filter_by(id=detector_id).first() is not None
    
    def count(self) -> int:
        """Count total number of detectors."""
        with self.session_factory() as session:
            return session.query(DetectorModel).count()
    
    def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        """Save the trained model artifact."""
        with self.session_factory() as session:
            model = session.query(DetectorModel).filter_by(id=detector_id).first()
            if model:
                model.model_data = artifact.decode('utf-8') if isinstance(artifact, bytes) else str(artifact)
                session.commit()
    
    def load_model_artifact(self, detector_id: UUID) -> Optional[bytes]:
        """Load the trained model artifact."""
        with self.session_factory() as session:
            model = session.query(DetectorModel).filter_by(id=detector_id).first()
            if model and model.model_data:
                return model.model_data.encode('utf-8') if isinstance(model.model_data, str) else model.model_data
            return None
    
    def _model_to_entity(self, model: DetectorModel) -> Detector:
        """Convert database model to domain entity."""
        return Detector(
            algorithm_name=model.algorithm,
            parameters=model.parameters or {},
            is_fitted=model.is_fitted,
            metadata=model.metadata or {},
            id=model.id,
            created_at=model.created_at,
            updated_at=model.updated_at
        )


class DatabaseDatasetRepository(DatasetRepositoryProtocol):
    """Database-backed dataset repository."""
    
    def __init__(self, session_factory):
        """Initialize with session factory."""
        self.session_factory = session_factory
    
    def save(self, dataset: Dataset) -> None:
        """Save dataset to database."""
        with self.session_factory() as session:
            existing = session.query(DatasetModel).filter_by(id=dataset.id).first()
            if existing:
                # Update existing
                existing.name = dataset.name
                existing.description = dataset.description
                existing.file_path = dataset.file_path
                existing.target_column = dataset.target_column
                existing.features = dataset.features
                existing.metadata = dataset.metadata
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
                    metadata=dataset.metadata,
                    created_at=dataset.created_at,
                    updated_at=dataset.updated_at
                )
                session.add(model)
            
            session.commit()
    
    def find_by_id(self, dataset_id: UUID) -> Optional[Dataset]:
        """Find dataset by ID."""
        with self.session_factory() as session:
            model = session.query(DatasetModel).filter_by(id=dataset_id).first()
            if not model:
                return None
            
            return self._model_to_entity(model)
    
    def find_by_name(self, name: str) -> Optional[Dataset]:
        """Find dataset by name."""
        with self.session_factory() as session:
            model = session.query(DatasetModel).filter_by(name=name).first()
            if not model:
                return None
            
            return self._model_to_entity(model)
    
    def find_by_metadata(self, key: str, value: Any) -> List[Dataset]:
        """Find datasets by metadata key-value pair."""
        with self.session_factory() as session:
            models = session.query(DatasetModel).all()
            matching_models = []
            for model in models:
                if model.metadata and model.metadata.get(key) == value:
                    matching_models.append(model)
            
            return [self._model_to_entity(model) for model in matching_models]
    
    def find_all(self) -> List[Dataset]:
        """Find all datasets."""
        with self.session_factory() as session:
            models = session.query(DatasetModel).all()
            return [self._model_to_entity(model) for model in models]
    
    def delete(self, dataset_id: UUID) -> bool:
        """Delete dataset by ID."""
        with self.session_factory() as session:
            model = session.query(DatasetModel).filter_by(id=dataset_id).first()
            if model:
                session.delete(model)
                session.commit()
                return True
            return False
    
    def exists(self, dataset_id: UUID) -> bool:
        """Check if dataset exists."""
        with self.session_factory() as session:
            return session.query(DatasetModel).filter_by(id=dataset_id).first() is not None
    
    def count(self) -> int:
        """Count total number of datasets."""
        with self.session_factory() as session:
            return session.query(DatasetModel).count()
    
    def save_data(self, dataset_id: UUID, format: str = "parquet") -> str:
        """Save dataset data to persistent storage."""
        with self.session_factory() as session:
            model = session.query(DatasetModel).filter_by(id=dataset_id).first()
            if not model:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # For now, just update metadata to indicate data was saved
            if not model.metadata:
                model.metadata = {}
            model.metadata["data_saved"] = True
            model.metadata["data_format"] = format
            model.metadata["data_saved_at"] = datetime.utcnow().isoformat()
            session.commit()
            
            return f"database://{dataset_id}.{format}"
    
    def load_data(self, dataset_id: UUID) -> Optional[Dataset]:
        """Load dataset with its data from storage."""
        # For database implementation, we don't store the actual data
        # This would typically load from file system or object storage
        return self.find_by_id(dataset_id)
    
    def _model_to_entity(self, model: DatasetModel) -> Dataset:
        """Convert database model to domain entity."""
        return Dataset(
            name=model.name,
            data=None,  # Data loaded separately when needed
            description=model.description,
            file_path=model.file_path,
            target_column=model.target_column,
            features=model.features or [],
            metadata=model.metadata or {},
            id=model.id,
            created_at=model.created_at,
            updated_at=model.updated_at
        )


class DatabaseDetectionResultRepository(DetectionResultRepositoryProtocol):
    """Database-backed detection result repository."""
    
    def __init__(self, session_factory):
        """Initialize with session factory."""
        self.session_factory = session_factory
    
    def save(self, result: DetectionResult) -> None:
        """Save detection result to database."""
        with self.session_factory() as session:
            # Serialize scores for storage
            scores_data = []
            if hasattr(result, 'scores') and result.scores:
                scores_data = [
                    {"value": score.value, "confidence": getattr(score, 'confidence', None)}
                    for score in result.scores
                ]
            
            existing = session.query(DetectionResultModel).filter_by(id=result.id).first()
            if existing:
                # Update existing
                existing.scores = scores_data
                existing.labels = getattr(result, 'labels', [])
                existing.metadata = result.metadata
            else:
                # Insert new
                model = DetectionResultModel(
                    id=result.id,
                    detector_id=result.detector_id,
                    dataset_id=result.dataset_id,
                    scores=scores_data,
                    labels=getattr(result, 'labels', []),
                    metadata=result.metadata,
                    created_at=result.timestamp
                )
                session.add(model)
            
            session.commit()
    
    def find_by_id(self, result_id: UUID) -> Optional[DetectionResult]:
        """Find detection result by ID."""
        with self.session_factory() as session:
            model = session.query(DetectionResultModel).filter_by(id=result_id).first()
            if not model:
                return None
            
            return self._model_to_entity(model)
    
    def find_by_detector(self, detector_id: UUID) -> List[DetectionResult]:
        """Find detection results by detector ID."""
        with self.session_factory() as session:
            models = session.query(DetectionResultModel).filter_by(detector_id=detector_id).all()
            return [self._model_to_entity(model) for model in models]
    
    def find_by_dataset(self, dataset_id: UUID) -> List[DetectionResult]:
        """Find detection results by dataset ID."""
        with self.session_factory() as session:
            models = session.query(DetectionResultModel).filter_by(dataset_id=dataset_id).all()
            return [self._model_to_entity(model) for model in models]
    
    def find_recent(self, limit: int = 10) -> List[DetectionResult]:
        """Find most recent detection results."""
        with self.session_factory() as session:
            models = session.query(DetectionResultModel).order_by(
                DetectionResultModel.created_at.desc()
            ).limit(limit).all()
            return [self._model_to_entity(model) for model in models]
    
    def find_all(self) -> List[DetectionResult]:
        """Find all detection results."""
        with self.session_factory() as session:
            models = session.query(DetectionResultModel).all()
            return [self._model_to_entity(model) for model in models]
    
    def delete(self, result_id: UUID) -> bool:
        """Delete detection result by ID."""
        with self.session_factory() as session:
            model = session.query(DetectionResultModel).filter_by(id=result_id).first()
            if model:
                session.delete(model)
                session.commit()
                return True
            return False
    
    def exists(self, result_id: UUID) -> bool:
        """Check if detection result exists."""
        with self.session_factory() as session:
            return session.query(DetectionResultModel).filter_by(id=result_id).first() is not None
    
    def count(self) -> int:
        """Count total number of detection results."""
        with self.session_factory() as session:
            return session.query(DetectionResultModel).count()
    
    def get_summary_stats(self, result_id: UUID) -> Dict[str, Any]:
        """Get summary statistics for a result."""
        with self.session_factory() as session:
            model = session.query(DetectionResultModel).filter_by(id=result_id).first()
            if not model:
                return {}
            
            result = self._model_to_entity(model)
            
            # Calculate statistics from the result
            stats = {
                "id": str(result.id),
                "detector_id": str(result.detector_id),
                "dataset_id": str(result.dataset_id),
                "timestamp": result.timestamp.isoformat(),
            }
            
            # Add computed statistics if available
            if hasattr(result, 'n_samples'):
                stats.update({
                    "n_samples": result.n_samples,
                    "n_anomalies": getattr(result, 'n_anomalies', 0),
                    "anomaly_rate": getattr(result, 'anomaly_rate', 0.0),
                    "threshold": getattr(result, 'threshold', None),
                    "execution_time_ms": getattr(result, 'execution_time_ms', None),
                    "score_statistics": getattr(result, 'score_statistics', {}),
                    "has_confidence_intervals": getattr(result, 'has_confidence_intervals', False)
                })
            
            return stats
    
    def _model_to_entity(self, model: DetectionResultModel) -> DetectionResult:
        """Convert database model to domain entity."""
        # Deserialize scores
        scores = []
        if model.scores:
            from pynomaly.domain.value_objects import AnomalyScore
            scores = [
                AnomalyScore(value=score_data["value"], confidence=score_data.get("confidence"))
                for score_data in model.scores
            ]
        
        return DetectionResult(
            detector_id=model.detector_id,
            dataset_id=model.dataset_id,
            scores=scores,
            metadata=model.metadata or {},
            id=model.id,
            timestamp=model.created_at
        )