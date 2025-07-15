"""Repository interface for DataScienceModel entities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import UUID

from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from packages.data_science.domain.entities.data_science_model import (
    DataScienceModel,
    ModelType,
    ModelStatus,
)


class DataScienceModelRepository(RepositoryInterface[DataScienceModel], ABC):
    """Repository interface for data science model persistence operations."""

    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[DataScienceModel]:
        """Find model by name.
        
        Args:
            name: Model name to search for
            
        Returns:
            DataScienceModel if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_algorithm(self, algorithm: str) -> list[DataScienceModel]:
        """Find models by algorithm.
        
        Args:
            algorithm: Algorithm name to search for
            
        Returns:
            List of models using the specified algorithm
        """
        pass

    @abstractmethod
    async def find_by_type(self, model_type: ModelType) -> list[DataScienceModel]:
        """Find models by type.
        
        Args:
            model_type: Model type to search for
            
        Returns:
            List of models of the specified type
        """
        pass

    @abstractmethod
    async def find_by_status(self, status: ModelStatus) -> list[DataScienceModel]:
        """Find models by status.
        
        Args:
            status: Model status to search for
            
        Returns:
            List of models with the specified status
        """
        pass

    @abstractmethod
    async def find_by_experiment_id(self, experiment_id: str) -> list[DataScienceModel]:
        """Find models by experiment ID.
        
        Args:
            experiment_id: Experiment ID to search for
            
        Returns:
            List of models from the specified experiment
        """
        pass

    @abstractmethod
    async def find_by_parent_model_id(self, parent_model_id: str) -> list[DataScienceModel]:
        """Find child models by parent model ID.
        
        Args:
            parent_model_id: Parent model ID to search for
            
        Returns:
            List of child models
        """
        pass

    @abstractmethod
    async def find_by_tags(self, tags: list[str]) -> list[DataScienceModel]:
        """Find models by tags.
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of models containing any of the specified tags
        """
        pass

    @abstractmethod
    async def find_deployed_models(self) -> list[DataScienceModel]:
        """Find all deployed models.
        
        Returns:
            List of deployed models
        """
        pass

    @abstractmethod
    async def find_models_by_performance_threshold(
        self, metric_name: str, threshold: float, operator: str = ">"
    ) -> list[DataScienceModel]:
        """Find models by performance metric threshold.
        
        Args:
            metric_name: Name of the performance metric
            threshold: Threshold value
            operator: Comparison operator (>, <, >=, <=, ==)
            
        Returns:
            List of models meeting the performance criteria
        """
        pass

    @abstractmethod
    async def find_latest_version_by_name(self, name: str) -> Optional[DataScienceModel]:
        """Find the latest version of a model by name.
        
        Args:
            name: Model name to search for
            
        Returns:
            Latest version of the model if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_models_requiring_retraining(
        self, performance_threshold: float
    ) -> list[DataScienceModel]:
        """Find models that may require retraining based on performance degradation.
        
        Args:
            performance_threshold: Minimum acceptable performance
            
        Returns:
            List of models that may need retraining
        """
        pass

    @abstractmethod
    async def get_model_lineage(self, model_id: UUID) -> list[DataScienceModel]:
        """Get complete model lineage (parents and children).
        
        Args:
            model_id: Model ID to get lineage for
            
        Returns:
            List of models in the lineage chain
        """
        pass

    @abstractmethod
    async def archive_old_versions(
        self, name: str, keep_latest_n: int = 5
    ) -> int:
        """Archive old versions of a model, keeping only the latest N versions.
        
        Args:
            name: Model name
            keep_latest_n: Number of latest versions to keep
            
        Returns:
            Number of models archived
        """
        pass

    @abstractmethod
    async def get_performance_history(
        self, model_id: UUID, metric_name: str
    ) -> list[dict[str, Any]]:
        """Get performance history for a model and metric.
        
        Args:
            model_id: Model ID
            metric_name: Performance metric name
            
        Returns:
            List of performance measurements over time
        """
        pass

    @abstractmethod
    async def bulk_update_status(
        self, model_ids: list[UUID], new_status: ModelStatus
    ) -> int:
        """Bulk update status for multiple models.
        
        Args:
            model_ids: List of model IDs to update
            new_status: New status to set
            
        Returns:
            Number of models updated
        """
        pass

    @abstractmethod
    async def find_similar_models(
        self, model: DataScienceModel, similarity_threshold: float = 0.8
    ) -> list[tuple[DataScienceModel, float]]:
        """Find models similar to the given model.
        
        Args:
            model: Reference model
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of tuples (similar_model, similarity_score)
        """
        pass

    @abstractmethod
    async def get_model_dependencies(self, model_id: UUID) -> dict[str, Any]:
        """Get model dependencies and requirements.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dictionary of model dependencies
        """
        pass

    @abstractmethod
    async def validate_model_deployment_readiness(
        self, model_id: UUID
    ) -> dict[str, Any]:
        """Validate if a model is ready for deployment.
        
        Args:
            model_id: Model ID to validate
            
        Returns:
            Validation results with readiness status and issues
        """
        pass