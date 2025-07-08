"""Repository interface and implementations for performance baselines."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

from pynomaly.domain.entities.model_performance import ModelPerformanceBaseline


class PerformanceBaselineRepository(ABC):
    """Abstract repository for performance baselines."""

    @abstractmethod
    def get_baseline(self, model_id: str, version: str) -> Optional[ModelPerformanceBaseline]:
        """Get baseline for a specific model and version.
        
        Args:
            model_id: The model identifier
            version: The model version
            
        Returns:
            The baseline if found, None otherwise
        """
        pass

    @abstractmethod
    def save_baseline(self, baseline: ModelPerformanceBaseline) -> None:
        """Save a performance baseline.
        
        Args:
            baseline: The baseline to save
        """
        pass

    @abstractmethod
    def update_baseline(self, baseline: ModelPerformanceBaseline) -> None:
        """Update an existing performance baseline.
        
        Args:
            baseline: The baseline to update
        """
        pass

    @abstractmethod
    def delete_baseline(self, model_id: str, version: str) -> None:
        """Delete a baseline for a specific model and version.
        
        Args:
            model_id: The model identifier
            version: The model version
        """
        pass

    @abstractmethod
    def get_latest_baseline(self, model_id: str) -> Optional[ModelPerformanceBaseline]:
        """Get the latest baseline for a model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            The latest baseline if found, None otherwise
        """
        pass

    @abstractmethod
    def list_baselines(self, model_id: str) -> List[ModelPerformanceBaseline]:
        """List all baselines for a specific model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            List of baselines for the model
        """
        pass

    @abstractmethod
    def baseline_exists(self, model_id: str, version: str) -> bool:
        """Check if a baseline exists for a specific model and version.
        
        Args:
            model_id: The model identifier
            version: The model version
            
        Returns:
            True if baseline exists, False otherwise
        """
        pass


class InMemoryPerformanceBaselineRepository(PerformanceBaselineRepository):
    """In-memory implementation of PerformanceBaselineRepository for MVP."""

    def __init__(self):
        self._baselines: Dict[str, ModelPerformanceBaseline] = {}

    def _get_key(self, model_id: str, version: str) -> str:
        """Get storage key for model and version combination."""
        return f"{model_id}:{version}"

    def get_baseline(self, model_id: str, version: str) -> Optional[ModelPerformanceBaseline]:
        """Get baseline for a specific model and version."""
        key = self._get_key(model_id, version)
        return self._baselines.get(key)

    def save_baseline(self, baseline: ModelPerformanceBaseline) -> None:
        """Save a performance baseline."""
        key = self._get_key(baseline.model_id, baseline.version)
        self._baselines[key] = baseline

    def update_baseline(self, baseline: ModelPerformanceBaseline) -> None:
        """Update an existing performance baseline."""
        key = self._get_key(baseline.model_id, baseline.version)
        if key not in self._baselines:
            raise ValueError(f"Baseline for model {baseline.model_id} version {baseline.version} not found")
        self._baselines[key] = baseline

    def delete_baseline(self, model_id: str, version: str) -> None:
        """Delete a baseline for a specific model and version."""
        key = self._get_key(model_id, version)
        if key in self._baselines:
            del self._baselines[key]

    def get_latest_baseline(self, model_id: str) -> Optional[ModelPerformanceBaseline]:
        """Get the latest baseline for a model."""
        model_baselines = [
            baseline for baseline in self._baselines.values()
            if baseline.model_id == model_id
        ]
        
        if not model_baselines:
            return None
        
        # Sort by version string (this is a simple approach; in production, 
        # you might want semantic versioning comparison)
        model_baselines.sort(key=lambda b: b.version, reverse=True)
        return model_baselines[0]

    def list_baselines(self, model_id: str) -> List[ModelPerformanceBaseline]:
        """List all baselines for a specific model."""
        return [
            baseline for baseline in self._baselines.values()
            if baseline.model_id == model_id
        ]

    def baseline_exists(self, model_id: str, version: str) -> bool:
        """Check if a baseline exists for a specific model and version."""
        key = self._get_key(model_id, version)
        return key in self._baselines


class SQLAlchemyPerformanceBaselineRepository(PerformanceBaselineRepository):
    """SQLAlchemy implementation of PerformanceBaselineRepository (placeholder)."""

    def __init__(self, session_factory):
        """Initialize with SQLAlchemy session factory.
        
        Args:
            session_factory: SQLAlchemy session factory
        """
        self.session_factory = session_factory

    def get_baseline(self, model_id: str, version: str) -> Optional[ModelPerformanceBaseline]:
        """Get baseline for a specific model and version from database."""
        # TODO: Implement database retrieval
        # Example structure:
        # with self.session_factory() as session:
        #     db_baseline = session.query(ModelPerformanceBaselineModel).filter_by(
        #         model_id=model_id,
        #         version=version
        #     ).first()
        #     return self._to_domain_entity(db_baseline) if db_baseline else None
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def save_baseline(self, baseline: ModelPerformanceBaseline) -> None:
        """Save a performance baseline to database."""
        # TODO: Implement database persistence
        # Example structure:
        # with self.session_factory() as session:
        #     db_baseline = ModelPerformanceBaselineModel(
        #         model_id=baseline.model_id,
        #         version=baseline.version,
        #         mean=baseline.mean,
        #         std=baseline.std,
        #         pct_thresholds=baseline.pct_thresholds,
        #         created_at=datetime.utcnow(),
        #         updated_at=datetime.utcnow()
        #     )
        #     session.add(db_baseline)
        #     session.commit()
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def update_baseline(self, baseline: ModelPerformanceBaseline) -> None:
        """Update an existing performance baseline in database."""
        # TODO: Implement database update
        # Example structure:
        # with self.session_factory() as session:
        #     db_baseline = session.query(ModelPerformanceBaselineModel).filter_by(
        #         model_id=baseline.model_id,
        #         version=baseline.version
        #     ).first()
        #     if not db_baseline:
        #         raise ValueError(f"Baseline for model {baseline.model_id} version {baseline.version} not found")
        #     db_baseline.mean = baseline.mean
        #     db_baseline.std = baseline.std
        #     db_baseline.pct_thresholds = baseline.pct_thresholds
        #     db_baseline.updated_at = datetime.utcnow()
        #     session.commit()
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def delete_baseline(self, model_id: str, version: str) -> None:
        """Delete a baseline for a specific model and version from database."""
        # TODO: Implement database deletion
        # Example structure:
        # with self.session_factory() as session:
        #     db_baseline = session.query(ModelPerformanceBaselineModel).filter_by(
        #         model_id=model_id,
        #         version=version
        #     ).first()
        #     if db_baseline:
        #         session.delete(db_baseline)
        #         session.commit()
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def get_latest_baseline(self, model_id: str) -> Optional[ModelPerformanceBaseline]:
        """Get the latest baseline for a model from database."""
        # TODO: Implement database retrieval with ordering
        # Example structure:
        # with self.session_factory() as session:
        #     db_baseline = session.query(ModelPerformanceBaselineModel).filter_by(
        #         model_id=model_id
        #     ).order_by(ModelPerformanceBaselineModel.version.desc()).first()
        #     return self._to_domain_entity(db_baseline) if db_baseline else None
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def list_baselines(self, model_id: str) -> List[ModelPerformanceBaseline]:
        """List all baselines for a specific model from database."""
        # TODO: Implement database retrieval
        # Example structure:
        # with self.session_factory() as session:
        #     db_baselines = session.query(ModelPerformanceBaselineModel).filter_by(
        #         model_id=model_id
        #     ).order_by(ModelPerformanceBaselineModel.version).all()
        #     return [self._to_domain_entity(db_baseline) for db_baseline in db_baselines]
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def baseline_exists(self, model_id: str, version: str) -> bool:
        """Check if a baseline exists for a specific model and version in database."""
        # TODO: Implement database existence check
        # Example structure:
        # with self.session_factory() as session:
        #     return session.query(ModelPerformanceBaselineModel).filter_by(
        #         model_id=model_id,
        #         version=version
        #     ).first() is not None
        raise NotImplementedError("SQLAlchemy implementation not yet available")

    def _to_domain_entity(self, db_model) -> ModelPerformanceBaseline:
        """Convert database model to domain entity."""
        # TODO: Implement conversion from DB model to domain entity
        # Example structure:
        # return ModelPerformanceBaseline(
        #     model_id=db_model.model_id,
        #     version=db_model.version,
        #     mean=db_model.mean,
        #     std=db_model.std,
        #     pct_thresholds=db_model.pct_thresholds or {}
        # )
        raise NotImplementedError("SQLAlchemy implementation not yet available")
