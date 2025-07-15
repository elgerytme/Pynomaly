"""Mappers for converting between domain entities and ORM models for experiments."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from pynomaly_mlops.domain.entities.experiment import (
    Experiment, ExperimentRun, ExperimentStatus, ExperimentRunStatus
)
from .experiment_models import ExperimentORM, ExperimentRunORM, ExperimentMetricORM, ExperimentComparisonORM


class ExperimentMapper:
    """Mapper for Experiment entity and ExperimentORM model."""
    
    @staticmethod
    def to_domain(orm: ExperimentORM) -> Experiment:
        """Convert ExperimentORM to Experiment domain entity.
        
        Args:
            orm: ExperimentORM instance
            
        Returns:
            Experiment domain entity
        """
        return Experiment(
            id=orm.id,
            name=orm.name,
            description=orm.description,
            tags=orm.tags or {},
            created_at=orm.created_at,
            updated_at=orm.updated_at,
            created_by=orm.created_by,
            status=ExperimentStatus(orm.status),
            runs=[]  # Runs loaded separately to avoid N+1 queries
        )
    
    @staticmethod
    def to_orm(experiment: Experiment) -> ExperimentORM:
        """Convert Experiment domain entity to ExperimentORM.
        
        Args:
            experiment: Experiment domain entity
            
        Returns:
            ExperimentORM instance
        """
        return ExperimentORM(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            tags=experiment.tags,
            created_at=experiment.created_at,
            updated_at=experiment.updated_at,
            created_by=experiment.created_by,
            status=experiment.status.value
        )
    
    @staticmethod
    def update_orm(orm: ExperimentORM, experiment: Experiment) -> None:
        """Update ExperimentORM with values from Experiment domain entity.
        
        Args:
            orm: ExperimentORM instance to update
            experiment: Experiment domain entity with new values
        """
        orm.name = experiment.name
        orm.description = experiment.description
        orm.tags = experiment.tags
        orm.updated_at = experiment.updated_at
        orm.status = experiment.status.value


class ExperimentRunMapper:
    """Mapper for ExperimentRun entity and ExperimentRunORM model."""
    
    @staticmethod
    def to_domain(orm: ExperimentRunORM) -> ExperimentRun:
        """Convert ExperimentRunORM to ExperimentRun domain entity.
        
        Args:
            orm: ExperimentRunORM instance
            
        Returns:
            ExperimentRun domain entity
        """
        return ExperimentRun(
            id=orm.id,
            experiment_id=orm.experiment_id,
            name=orm.name,
            parameters=orm.parameters or {},
            metrics=orm.metrics or {},
            artifacts=orm.artifacts or {},
            tags=orm.tags or {},
            status=ExperimentRunStatus(orm.status),
            start_time=orm.start_time,
            end_time=orm.end_time,
            created_by=orm.created_by,
            parent_run_id=orm.parent_run_id,
            source_version=orm.source_version,
            entry_point=orm.entry_point,
            notes=orm.notes
        )
    
    @staticmethod
    def to_orm(run: ExperimentRun) -> ExperimentRunORM:
        """Convert ExperimentRun domain entity to ExperimentRunORM.
        
        Args:
            run: ExperimentRun domain entity
            
        Returns:
            ExperimentRunORM instance
        """
        return ExperimentRunORM(
            id=run.id,
            experiment_id=run.experiment_id,
            name=run.name,
            parameters=run.parameters,
            metrics=run.metrics,
            artifacts=run.artifacts,
            tags=run.tags,
            status=run.status.value,
            start_time=run.start_time,
            end_time=run.end_time,
            created_by=run.created_by,
            parent_run_id=run.parent_run_id,
            source_version=run.source_version,
            entry_point=run.entry_point,
            notes=run.notes
        )
    
    @staticmethod
    def update_orm(orm: ExperimentRunORM, run: ExperimentRun) -> None:
        """Update ExperimentRunORM with values from ExperimentRun domain entity.
        
        Args:
            orm: ExperimentRunORM instance to update
            run: ExperimentRun domain entity with new values
        """
        orm.name = run.name
        orm.parameters = run.parameters
        orm.metrics = run.metrics
        orm.artifacts = run.artifacts
        orm.tags = run.tags
        orm.status = run.status.value
        orm.end_time = run.end_time
        orm.notes = run.notes


class ExperimentMetricMapper:
    """Mapper for time-series experiment metrics."""
    
    @staticmethod
    def to_domain(orm: ExperimentMetricORM) -> Dict[str, Any]:
        """Convert ExperimentMetricORM to domain representation.
        
        Args:
            orm: ExperimentMetricORM instance
            
        Returns:
            Dictionary with metric data
        """
        return {
            "key": orm.key,
            "value": orm.value,
            "step": orm.step,
            "timestamp": orm.timestamp
        }
    
    @staticmethod
    def to_orm(run_id: UUID, key: str, value: float, step: Optional[float] = None,
               timestamp: Optional[datetime] = None) -> ExperimentMetricORM:
        """Convert metric data to ExperimentMetricORM.
        
        Args:
            run_id: ID of the experiment run
            key: Metric name
            value: Metric value
            step: Optional step number
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            ExperimentMetricORM instance
        """
        return ExperimentMetricORM(
            run_id=run_id,
            key=key,
            value=value,
            step=step,
            timestamp=timestamp or datetime.utcnow()
        )


class ExperimentComparisonMapper:
    """Mapper for experiment comparison entities."""
    
    @staticmethod
    def to_domain(orm: ExperimentComparisonORM) -> Dict[str, Any]:
        """Convert ExperimentComparisonORM to domain representation.
        
        Args:
            orm: ExperimentComparisonORM instance
            
        Returns:
            Dictionary with comparison data
        """
        return {
            "id": orm.id,
            "name": orm.name,
            "description": orm.description,
            "run_ids": orm.run_ids,
            "comparison_config": orm.comparison_config,
            "created_at": orm.created_at,
            "created_by": orm.created_by,
            "is_public": orm.is_public
        }
    
    @staticmethod
    def to_orm(name: str, run_ids: List[UUID], comparison_config: Dict[str, Any],
               created_by: str, description: Optional[str] = None,
               is_public: bool = False) -> ExperimentComparisonORM:
        """Convert comparison data to ExperimentComparisonORM.
        
        Args:
            name: Comparison name
            run_ids: List of run IDs to compare
            comparison_config: Configuration for comparison
            created_by: User who created the comparison
            description: Optional description
            is_public: Whether comparison is publicly visible
            
        Returns:
            ExperimentComparisonORM instance
        """
        return ExperimentComparisonORM(
            name=name,
            description=description,
            run_ids=[str(run_id) for run_id in run_ids],
            comparison_config=comparison_config,
            created_by=created_by,
            is_public=is_public
        )