"""In-memory repository implementation for experiments."""

from typing import Dict, List, Optional
from uuid import UUID

from ...domain.entities.experiment import Experiment
from ...domain.repositories.experiment_repository import ExperimentRepository


class InMemoryExperimentRepository(ExperimentRepository):
    """In-memory implementation of experiment repository."""
    
    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
    
    async def save(self, experiment: Experiment) -> Experiment:
        """Save an experiment."""
        experiment_id_str = str(experiment.id)
        self._experiments[experiment_id_str] = experiment
        return experiment
    
    async def find_by_id(self, experiment_id: UUID) -> Optional[Experiment]:
        """Find experiment by ID."""
        return self._experiments.get(str(experiment_id))
    
    async def find_by_name(self, name: str) -> Optional[Experiment]:
        """Find experiment by name."""
        for experiment in self._experiments.values():
            if experiment.name == name:
                return experiment
        return None
    
    async def find_all(self) -> List[Experiment]:
        """Find all experiments."""
        return list(self._experiments.values())
    
    async def find_by_model_id(self, model_id: UUID) -> List[Experiment]:
        """Find experiments by model ID."""
        return [
            experiment for experiment in self._experiments.values()
            if experiment.model_id == model_id
        ]
    
    async def delete(self, experiment_id: UUID) -> None:
        """Delete an experiment."""
        experiment_id_str = str(experiment_id)
        if experiment_id_str in self._experiments:
            del self._experiments[experiment_id_str]