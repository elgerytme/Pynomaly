"""Repository interface for Experiment entities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from packages.core.domain.abstractions.repository_interface import RepositoryInterface
# TODO: Implement within data platform science domain - from packages.data_science.domain.entities.experiment import Experiment


class ExperimentRepository(RepositoryInterface[Experiment], ABC):
    """Repository interface for experiment persistence operations."""

    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Experiment]:
        """Find experiment by name.
        
        Args:
            name: Experiment name to search for
            
        Returns:
            Experiment if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_status(self, status: str) -> list[Experiment]:
        """Find experiments by status.
        
        Args:
            status: Experiment status to search for
            
        Returns:
            List of experiments with the specified status
        """
        pass

    @abstractmethod
    async def find_by_algorithm(self, algorithm_name: str) -> list[Experiment]:
        """Find experiments by algorithm.
        
        Args:
            algorithm_name: Algorithm name to search for
            
        Returns:
            List of experiments using the specified algorithm
        """
        pass

    @abstractmethod
    async def find_by_dataset(self, dataset_id: str) -> list[Experiment]:
        """Find experiments by dataset.
        
        Args:
            dataset_id: Dataset ID to search for
            
        Returns:
            List of experiments using the specified dataset
        """
        pass

    @abstractmethod
    async def find_by_created_by(self, created_by: str) -> list[Experiment]:
        """Find experiments by creator.
        
        Args:
            created_by: User who created experiments
            
        Returns:
            List of experiments created by the specified user
        """
        pass

    @abstractmethod
    async def find_by_tags(self, tags: list[str]) -> list[Experiment]:
        """Find experiments by tags.
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of experiments containing any of the specified tags
        """
        pass

    @abstractmethod
    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[Experiment]:
        """Find experiments created within a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of experiments created within the date range
        """
        pass

    @abstractmethod
    async def find_running_experiments(self) -> list[Experiment]:
        """Find all currently running experiments.
        
        Returns:
            List of running experiments
        """
        pass

    @abstractmethod
    async def find_completed_experiments(self) -> list[Experiment]:
        """Find all completed experiments.
        
        Returns:
            List of completed experiments
        """
        pass

    @abstractmethod
    async def find_failed_experiments(self) -> list[Experiment]:
        """Find all failed experiments.
        
        Returns:
            List of failed experiments
        """
        pass

    @abstractmethod
    async def find_by_parent_experiment(self, parent_experiment_id: UUID) -> list[Experiment]:
        """Find child experiments by parent experiment ID.
        
        Args:
            parent_experiment_id: Parent experiment ID
            
        Returns:
            List of child experiments
        """
        pass

    @abstractmethod
    async def find_related_experiments(self, experiment_id: UUID) -> list[Experiment]:
        """Find experiments related to the given experiment.
        
        Args:
            experiment_id: Reference experiment ID
            
        Returns:
            List of related experiments
        """
        pass

    @abstractmethod
    async def find_by_objective(self, objective_keywords: list[str]) -> list[Experiment]:
        """Find experiments by objective keywords.
        
        Args:
            objective_keywords: Keywords to search in objectives
            
        Returns:
            List of experiments matching the objective criteria
        """
        pass

    @abstractmethod
    async def find_by_performance_range(
        self, metric_name: str, min_value: float, max_value: float
    ) -> list[Experiment]:
        """Find experiments by performance metric range.
        
        Args:
            metric_name: Name of the performance metric
            min_value: Minimum metric value
            max_value: Maximum metric value
            
        Returns:
            List of experiments within the performance range
        """
        pass

    @abstractmethod
    async def find_best_experiments(
        self, metric_name: str, limit: int = 10
    ) -> list[Experiment]:
        """Find the best performing experiments for a given metric.
        
        Args:
            metric_name: Performance metric to rank by
            limit: Maximum number of experiments to return
            
        Returns:
            List of top performing experiments
        """
        pass

    @abstractmethod
    async def get_experiment_lineage(self, experiment_id: UUID) -> list[Experiment]:
        """Get complete experiment lineage (parents and children).
        
        Args:
            experiment_id: Experiment ID to get lineage for
            
        Returns:
            List of experiments in the lineage chain
        """
        pass

    @abstractmethod
    async def archive_old_experiments(
        self, older_than_days: int, exclude_successful: bool = True
    ) -> int:
        """Archive old experiments.
        
        Args:
            older_than_days: Archive experiments older than this many days
            exclude_successful: Whether to exclude successful experiments
            
        Returns:
            Number of experiments archived
        """
        pass

    @abstractmethod
    async def get_performance_trends(
        self, metric_name: str, days: int = 30
    ) -> list[dict[str, Any]]:
        """Get performance trends over time.
        
        Args:
            metric_name: Performance metric name
            days: Number of days to analyze
            
        Returns:
            List of performance trend data points
        """
        pass

    @abstractmethod
    async def find_duplicate_experiments(
        self, similarity_threshold: float = 0.9
    ) -> list[list[Experiment]]:
        """Find groups of duplicate or very similar experiments.
        
        Args:
            similarity_threshold: Minimum similarity score to consider duplicates
            
        Returns:
            List of experiment groups that are duplicates
        """
        pass

    @abstractmethod
    async def get_resource_usage_stats(self) -> dict[str, Any]:
        """Get resource usage statistics across all experiments.
        
        Returns:
            Dictionary of resource usage statistics
        """
        pass

    @abstractmethod
    async def find_experiments_by_resource_usage(
        self, resource_type: str, threshold: float
    ) -> list[Experiment]:
        """Find experiments by resource usage threshold.
        
        Args:
            resource_type: Type of resource (cpu, memory, gpu, etc.)
            threshold: Resource usage threshold
            
        Returns:
            List of experiments exceeding the resource threshold
        """
        pass

    @abstractmethod
    async def bulk_update_status(
        self, experiment_ids: list[UUID], new_status: str
    ) -> int:
        """Bulk update status for multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to update
            new_status: New status to set
            
        Returns:
            Number of experiments updated
        """
        pass

    @abstractmethod
    async def get_collaboration_stats(self) -> dict[str, Any]:
        """Get collaboration statistics across experiments.
        
        Returns:
            Dictionary of collaboration metrics
        """
        pass