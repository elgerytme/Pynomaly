"""Database repository implementations using SQLAlchemy."""

from __future__ import annotations

import json
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
    metadata = Column(JSONType)
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
    metadata = Column(JSONType)
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
    metadata = Column(JSONType)
    created_at = Column(DateTime, nullable=False)


class DatabaseDetectorRepository(DetectorRepositoryProtocol):
    """Database-backed detector repository."""
    
    def __init__(self, session: Session):
        self.session = session
    
    async def save(self, detector: Detector) -> None:
        """Save detector to database."""
        model = DetectorModel(
            id=detector.id,
            algorithm=detector.algorithm,
            parameters=detector.parameters,
            is_fitted=detector.is_fitted,
            model_data=detector.model_data,
            metadata=detector.metadata,
            created_at=detector.created_at,
            updated_at=detector.updated_at
        )
        
        existing = self.session.query(DetectorModel).filter_by(id=detector.id).first()
        if existing:
            # Update existing
            existing.algorithm = detector.algorithm
            existing.parameters = detector.parameters
            existing.is_fitted = detector.is_fitted
            existing.model_data = detector.model_data
            existing.metadata = detector.metadata
            existing.updated_at = detector.updated_at
        else:
            # Insert new
            self.session.add(model)
        
        self.session.commit()
    
    async def find_by_id(self, detector_id: UUID) -> Optional[Detector]:
        """Find detector by ID."""
        model = self.session.query(DetectorModel).filter_by(id=detector_id).first()
        if not model:
            return None
        
        return Detector(
            algorithm=model.algorithm,
            parameters=model.parameters,
            is_fitted=model.is_fitted,
            model_data=model.model_data,
            metadata=model.metadata or {},
            id=model.id,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    async def find_by_algorithm(self, algorithm: str) -> List[Detector]:
        """Find detectors by algorithm."""
        models = self.session.query(DetectorModel).filter_by(algorithm=algorithm).all()
        
        return [
            Detector(
                algorithm=model.algorithm,
                parameters=model.parameters,
                is_fitted=model.is_fitted,
                model_data=model.model_data,
                metadata=model.metadata or {},
                id=model.id,
                created_at=model.created_at,
                updated_at=model.updated_at
            )
            for model in models
        ]
    
    async def find_all(self) -> List[Detector]:
        """Find all detectors."""
        models = self.session.query(DetectorModel).all()
        
        return [
            Detector(
                algorithm=model.algorithm,
                parameters=model.parameters,
                is_fitted=model.is_fitted,
                model_data=model.model_data,
                metadata=model.metadata or {},
                id=model.id,
                created_at=model.created_at,
                updated_at=model.updated_at
            )
            for model in models
        ]
    
    async def delete(self, detector_id: UUID) -> None:
        """Delete detector by ID."""
        model = self.session.query(DetectorModel).filter_by(id=detector_id).first()
        if model:
            self.session.delete(model)
            self.session.commit()


class DatabaseDatasetRepository(DatasetRepositoryProtocol):
    """Database-backed dataset repository."""
    
    def __init__(self, session: Session):
        self.session = session
    
    async def save(self, dataset: Dataset) -> None:
        """Save dataset to database."""
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
        
        existing = self.session.query(DatasetModel).filter_by(id=dataset.id).first()
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
            self.session.add(model)
        
        self.session.commit()
    
    async def find_by_id(self, dataset_id: UUID) -> Optional[Dataset]:
        """Find dataset by ID."""
        model = self.session.query(DatasetModel).filter_by(id=dataset_id).first()
        if not model:
            return None
        
        # Note: We don't load the actual data here, just metadata
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
    
    async def find_by_name(self, name: str) -> Optional[Dataset]:
        """Find dataset by name."""
        model = self.session.query(DatasetModel).filter_by(name=name).first()
        if not model:
            return None
        
        return Dataset(
            name=model.name,
            data=None,
            description=model.description,
            file_path=model.file_path,
            target_column=model.target_column,
            features=model.features or [],
            metadata=model.metadata or {},
            id=model.id,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    async def find_all(self) -> List[Dataset]:
        """Find all datasets."""
        models = self.session.query(DatasetModel).all()
        
        return [
            Dataset(
                name=model.name,
                data=None,
                description=model.description,
                file_path=model.file_path,
                target_column=model.target_column,
                features=model.features or [],
                metadata=model.metadata or {},
                id=model.id,
                created_at=model.created_at,
                updated_at=model.updated_at
            )
            for model in models
        ]
    
    async def delete(self, dataset_id: UUID) -> None:
        """Delete dataset by ID."""
        model = self.session.query(DatasetModel).filter_by(id=dataset_id).first()
        if model:
            self.session.delete(model)
            self.session.commit()


class DatabaseDetectionResultRepository(DetectionResultRepositoryProtocol):
    """Database-backed detection result repository."""
    
    def __init__(self, session: Session):
        self.session = session
    
    async def save(self, result: DetectionResult) -> None:
        """Save detection result to database."""
        # Serialize scores for storage
        scores_data = [
            {"value": score.value, "confidence": score.confidence}
            for score in result.scores
        ]
        
        model = DetectionResultModel(
            id=result.id,
            detector_id=result.detector_id,
            dataset_id=result.dataset_id,
            scores=scores_data,
            labels=result.labels,
            metadata=result.metadata,
            created_at=result.created_at
        )
        
        existing = self.session.query(DetectionResultModel).filter_by(id=result.id).first()
        if existing:
            # Update existing
            existing.scores = scores_data
            existing.labels = result.labels
            existing.metadata = result.metadata
        else:
            # Insert new
            self.session.add(model)
        
        self.session.commit()
    
    async def find_by_id(self, result_id: UUID) -> Optional[DetectionResult]:
        """Find detection result by ID."""
        model = self.session.query(DetectionResultModel).filter_by(id=result_id).first()
        if not model:
            return None
        
        # Deserialize scores
        from pynomaly.domain.value_objects import AnomalyScore
        scores = [
            AnomalyScore(value=score_data["value"], confidence=score_data["confidence"])
            for score_data in model.scores
        ]
        
        return DetectionResult(
            detector_id=model.detector_id,
            dataset_id=model.dataset_id,
            scores=scores,
            labels=model.labels,
            metadata=model.metadata or {},
            id=model.id,
            created_at=model.created_at
        )
    
    async def find_by_detector(self, detector_id: UUID) -> List[DetectionResult]:
        """Find detection results by detector ID."""
        models = self.session.query(DetectionResultModel).filter_by(detector_id=detector_id).all()
        
        from pynomaly.domain.value_objects import AnomalyScore
        
        return [
            DetectionResult(
                detector_id=model.detector_id,
                dataset_id=model.dataset_id,
                scores=[
                    AnomalyScore(value=score_data["value"], confidence=score_data["confidence"])
                    for score_data in model.scores
                ],
                labels=model.labels,
                metadata=model.metadata or {},
                id=model.id,
                created_at=model.created_at
            )
            for model in models
        ]
    
    async def find_by_dataset(self, dataset_id: UUID) -> List[DetectionResult]:
        """Find detection results by dataset ID."""
        models = self.session.query(DetectionResultModel).filter_by(dataset_id=dataset_id).all()
        
        from pynomaly.domain.value_objects import AnomalyScore
        
        return [
            DetectionResult(
                detector_id=model.detector_id,
                dataset_id=model.dataset_id,
                scores=[
                    AnomalyScore(value=score_data["value"], confidence=score_data["confidence"])
                    for score_data in model.scores
                ],
                labels=model.labels,
                metadata=model.metadata or {},
                id=model.id,
                created_at=model.created_at
            )
            for model in models
        ]
    
    async def find_all(self) -> List[DetectionResult]:
        """Find all detection results."""
        models = self.session.query(DetectionResultModel).all()
        
        from pynomaly.domain.value_objects import AnomalyScore
        
        return [
            DetectionResult(
                detector_id=model.detector_id,
                dataset_id=model.dataset_id,
                scores=[
                    AnomalyScore(value=score_data["value"], confidence=score_data["confidence"])
                    for score_data in model.scores
                ],
                labels=model.labels,
                metadata=model.metadata or {},
                id=model.id,
                created_at=model.created_at
            )
            for model in models
        ]
    
    async def delete(self, result_id: UUID) -> None:
        """Delete detection result by ID."""
        model = self.session.query(DetectionResultModel).filter_by(id=result_id).first()
        if model:
            self.session.delete(model)
            self.session.commit()