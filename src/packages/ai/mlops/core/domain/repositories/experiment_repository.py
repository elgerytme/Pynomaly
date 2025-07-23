"""Experiment repository interface for MLOps domain."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.experiment import Experiment


class ExperimentRepository(ABC):
    """Abstract repository for experiments."""
    
    @abstractmethod
    async def save(self, experiment: Experiment) -> Experiment:
        """Save an experiment."""
        pass
    
    @abstractmethod
    async def find_by_id(self, experiment_id: UUID) -> Optional[Experiment]:
        """Find experiment by ID."""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Experiment]:
        """Find experiment by name."""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[Experiment]:
        """Find all experiments."""
        pass
    
    @abstractmethod
    async def find_by_model_id(self, model_id: UUID) -> List[Experiment]:
        """Find experiments by model ID."""
        pass
    
    @abstractmethod
    async def delete(self, experiment_id: UUID) -> None:
        """Delete an experiment."""
        pass