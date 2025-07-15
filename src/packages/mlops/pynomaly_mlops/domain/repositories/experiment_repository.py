"""Repository interface for experiment entities."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from pynomaly_mlops.domain.entities.experiment import Experiment, ExperimentRun, ExperimentStatus, ExperimentRunStatus


class ExperimentRepository(ABC):
    """Abstract repository for experiment entities."""
    
    # Experiment operations
    @abstractmethod
    async def save(self, experiment: Experiment) -> Experiment:
        """Save experiment to storage.
        
        Args:
            experiment: Experiment to save
            
        Returns:
            Saved experiment
        """
        
    @abstractmethod
    async def get_by_id(self, experiment_id: UUID) -> Optional[Experiment]:
        """Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment if found, None otherwise
        """
        
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Experiment]:
        """Get experiment by name.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment if found, None otherwise
        """
        
    @abstractmethod
    async def list_experiments(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[ExperimentStatus] = None,
        created_by: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        order_by: str = "created_at",
        ascending: bool = False
    ) -> List[Experiment]:
        """List experiments with filtering and pagination.
        
        Args:
            limit: Maximum number of experiments to return
            offset: Number of experiments to skip
            status: Filter by experiment status
            created_by: Filter by creator
            tags: Filter by tags (key-value pairs)
            order_by: Field to order by
            ascending: Sort direction
            
        Returns:
            List of experiments
        """
        
    @abstractmethod
    async def delete(self, experiment_id: UUID) -> bool:
        """Delete experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if deleted, False if not found
        """
        
    @abstractmethod
    async def search_experiments(
        self, 
        query: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Experiment]:
        """Search experiments by name, description, or tags.
        
        Args:
            query: Search query
            limit: Maximum results
            offset: Offset for pagination
            
        Returns:
            List of matching experiments
        """
    
    # Experiment run operations
    @abstractmethod
    async def save_run(self, run: ExperimentRun) -> ExperimentRun:
        """Save experiment run to storage.
        
        Args:
            run: ExperimentRun to save
            
        Returns:
            Saved experiment run
        """
        
    @abstractmethod
    async def get_run_by_id(self, run_id: UUID) -> Optional[ExperimentRun]:
        """Get experiment run by ID.
        
        Args:
            run_id: Run ID
            
        Returns:
            ExperimentRun if found, None otherwise
        """
        
    @abstractmethod
    async def list_runs(
        self,
        experiment_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0,
        status: Optional[ExperimentRunStatus] = None,
        created_by: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        order_by: str = "start_time",
        ascending: bool = False
    ) -> List[ExperimentRun]:
        """List experiment runs with filtering and pagination.
        
        Args:
            experiment_id: Filter by experiment ID
            limit: Maximum number of runs to return
            offset: Number of runs to skip
            status: Filter by run status
            created_by: Filter by creator
            tags: Filter by tags
            order_by: Field to order by
            ascending: Sort direction
            
        Returns:
            List of experiment runs
        """
        
    @abstractmethod
    async def delete_run(self, run_id: UUID) -> bool:
        """Delete experiment run by ID.
        
        Args:
            run_id: Run ID
            
        Returns:
            True if deleted, False if not found
        """
    
    # Metrics operations
    @abstractmethod
    async def log_metric(
        self, 
        run_id: UUID, 
        key: str, 
        value: float, 
        step: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log a metric for an experiment run.
        
        Args:
            run_id: Run ID
            key: Metric name
            value: Metric value
            step: Optional step number
            timestamp: Optional timestamp
        """
        
    @abstractmethod
    async def get_metrics(
        self, 
        run_id: UUID, 
        keys: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get time-series metrics for a run.
        
        Args:
            run_id: Run ID
            keys: Optional list of metric keys to filter
            
        Returns:
            Dictionary mapping metric keys to lists of metric data
        """
    
    # Analysis operations
    @abstractmethod
    async def compare_runs(
        self, 
        run_ids: List[UUID], 
        metric_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple experiment runs.
        
        Args:
            run_ids: List of run IDs to compare
            metric_keys: Optional list of metrics to include in comparison
            
        Returns:
            Comparison data structure
        """