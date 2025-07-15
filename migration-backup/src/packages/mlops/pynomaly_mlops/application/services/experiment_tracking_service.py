"""Application service for experiment tracking and management."""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4

from pynomaly_mlops.domain.entities.experiment import (
    Experiment, ExperimentRun, ExperimentStatus, ExperimentRunStatus
)
from pynomaly_mlops.domain.repositories.experiment_repository import ExperimentRepository
from pynomaly_mlops.infrastructure.storage.artifact_storage import ArtifactStorageService


class ExperimentTrackingService:
    """Service for managing ML experiments and runs."""
    
    def __init__(
        self, 
        experiment_repository: ExperimentRepository,
        artifact_storage: ArtifactStorageService
    ):
        """Initialize service with dependencies.
        
        Args:
            experiment_repository: Repository for experiment persistence
            artifact_storage: Storage service for artifacts
        """
        self.experiment_repository = experiment_repository
        self.artifact_storage = artifact_storage
    
    async def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> Experiment:
        """Create a new experiment.
        
        Args:
            name: Experiment name
            description: Optional description
            tags: Optional tags
            created_by: User creating the experiment
            
        Returns:
            Created experiment
            
        Raises:
            ValueError: If experiment with name already exists
        """
        # Check if experiment already exists
        existing = await self.experiment_repository.get_by_name(name)
        if existing:
            raise ValueError(f"Experiment with name '{name}' already exists")
        
        experiment = Experiment(
            id=uuid4(),
            name=name,
            description=description,
            tags=tags or {},
            created_by=created_by,
            status=ExperimentStatus.ACTIVE
        )
        
        return await self.experiment_repository.save(experiment)
    
    async def get_experiment(self, experiment_id: UUID) -> Optional[Experiment]:
        """Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment if found, None otherwise
        """
        return await self.experiment_repository.get_by_id(experiment_id)
    
    async def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Get experiment by name.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment if found, None otherwise
        """
        return await self.experiment_repository.get_by_name(name)
    
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
        """List experiments with filtering.
        
        Args:
            limit: Maximum results
            offset: Pagination offset
            status: Filter by status
            created_by: Filter by creator
            tags: Filter by tags
            order_by: Sort field
            ascending: Sort direction
            
        Returns:
            List of experiments
        """
        return await self.experiment_repository.list_experiments(
            limit=limit,
            offset=offset,
            status=status,
            created_by=created_by,
            tags=tags,
            order_by=order_by,
            ascending=ascending
        )
    
    async def update_experiment(
        self,
        experiment_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        status: Optional[ExperimentStatus] = None
    ) -> Optional[Experiment]:
        """Update experiment details.
        
        Args:
            experiment_id: Experiment ID
            name: New name
            description: New description
            tags: New tags
            status: New status
            
        Returns:
            Updated experiment if found, None otherwise
        """
        experiment = await self.experiment_repository.get_by_id(experiment_id)
        if not experiment:
            return None
        
        # Update fields if provided
        if name is not None:
            experiment.name = name
        if description is not None:
            experiment.description = description
        if tags is not None:
            experiment.tags = tags
        if status is not None:
            experiment.status = status
        
        experiment.updated_at = datetime.utcnow()
        
        return await self.experiment_repository.save(experiment)
    
    async def delete_experiment(self, experiment_id: UUID) -> bool:
        """Delete experiment and all associated runs.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if deleted, False if not found
        """
        return await self.experiment_repository.delete(experiment_id)
    
    async def start_run(
        self,
        experiment_id: UUID,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system",
        parent_run_id: Optional[UUID] = None,
        source_version: Optional[str] = None,
        entry_point: Optional[str] = None
    ) -> ExperimentRun:
        """Start a new experiment run.
        
        Args:
            experiment_id: Experiment ID
            name: Optional run name
            parameters: Run parameters
            tags: Run tags
            created_by: User starting the run
            parent_run_id: Optional parent run ID
            source_version: Source code version
            entry_point: Entry point script
            
        Returns:
            Created experiment run
            
        Raises:
            ValueError: If experiment not found
        """
        # Verify experiment exists
        experiment = await self.experiment_repository.get_by_id(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment with ID {experiment_id} not found")
        
        run = ExperimentRun(
            id=uuid4(),
            experiment_id=experiment_id,
            name=name,
            parameters=parameters or {},
            tags=tags or {},
            status=ExperimentRunStatus.RUNNING,
            created_by=created_by,
            parent_run_id=parent_run_id,
            source_version=source_version,
            entry_point=entry_point
        )
        
        return await self.experiment_repository.save_run(run)
    
    async def end_run(
        self,
        run_id: UUID,
        status: ExperimentRunStatus = ExperimentRunStatus.COMPLETED,
        notes: Optional[str] = None
    ) -> Optional[ExperimentRun]:
        """End an experiment run.
        
        Args:
            run_id: Run ID
            status: Final status
            notes: Optional notes
            
        Returns:
            Updated run if found, None otherwise
        """
        run = await self.experiment_repository.get_run_by_id(run_id)
        if not run:
            return None
        
        run.status = status
        run.end_time = datetime.utcnow()
        if notes:
            run.notes = notes
        
        return await self.experiment_repository.save_run(run)
    
    async def log_metric(
        self,
        run_id: UUID,
        key: str,
        value: float,
        step: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log a metric for a run.
        
        Args:
            run_id: Run ID
            key: Metric name
            value: Metric value
            step: Optional step number
            timestamp: Optional timestamp
            
        Raises:
            ValueError: If run not found
        """
        # Verify run exists
        run = await self.experiment_repository.get_run_by_id(run_id)
        if not run:
            raise ValueError(f"Run with ID {run_id} not found")
        
        await self.experiment_repository.log_metric(
            run_id=run_id,
            key=key,
            value=value,
            step=step,
            timestamp=timestamp
        )
    
    async def log_metrics(
        self,
        run_id: UUID,
        metrics: Dict[str, float],
        step: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log multiple metrics for a run.
        
        Args:
            run_id: Run ID
            metrics: Dictionary of metric name to value
            step: Optional step number
            timestamp: Optional timestamp
        """
        for key, value in metrics.items():
            await self.log_metric(
                run_id=run_id,
                key=key,
                value=value,
                step=step,
                timestamp=timestamp
            )
    
    async def log_artifact(
        self,
        run_id: UUID,
        artifact_name: str,
        artifact_data: Any,
        artifact_type: str = "model"
    ) -> str:
        """Log an artifact for a run.
        
        Args:
            run_id: Run ID
            artifact_name: Artifact name
            artifact_data: Artifact data
            artifact_type: Type of artifact
            
        Returns:
            Artifact URI
            
        Raises:
            ValueError: If run not found
        """
        # Verify run exists
        run = await self.experiment_repository.get_run_by_id(run_id)
        if not run:
            raise ValueError(f"Run with ID {run_id} not found")
        
        # Store artifact
        artifact_uri = await self.artifact_storage.store_model(
            model=artifact_data,
            model_id=f"{run_id}_{artifact_name}",
            metadata={
                "run_id": str(run_id),
                "artifact_name": artifact_name,
                "artifact_type": artifact_type,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Update run with artifact info
        if not run.artifacts:
            run.artifacts = {}
        run.artifacts[artifact_name] = {
            "uri": artifact_uri,
            "type": artifact_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.experiment_repository.save_run(run)
        
        return artifact_uri
    
    async def get_run_metrics(
        self,
        run_id: UUID,
        keys: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get time-series metrics for a run.
        
        Args:
            run_id: Run ID
            keys: Optional metric keys to filter
            
        Returns:
            Dictionary of metric histories
        """
        return await self.experiment_repository.get_metrics(run_id, keys)
    
    async def compare_runs(
        self,
        run_ids: List[UUID],
        metric_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple runs.
        
        Args:
            run_ids: List of run IDs
            metric_keys: Optional metrics to compare
            
        Returns:
            Comparison data
        """
        return await self.experiment_repository.compare_runs(run_ids, metric_keys)
    
    async def search_experiments(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Experiment]:
        """Search experiments.
        
        Args:
            query: Search query
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of matching experiments
        """
        return await self.experiment_repository.search_experiments(
            query=query,
            limit=limit,
            offset=offset
        )
    
    async def get_experiment_summary(self, experiment_id: UUID) -> Optional[Dict[str, Any]]:
        """Get experiment summary with statistics.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment summary if found, None otherwise
        """
        experiment = await self.experiment_repository.get_by_id(experiment_id)
        if not experiment:
            return None
        
        # Get all runs for the experiment
        runs = await self.experiment_repository.list_runs(
            experiment_id=experiment_id,
            limit=1000  # Large limit to get all runs
        )
        
        # Calculate statistics
        total_runs = len(runs)
        completed_runs = len([r for r in runs if r.status == ExperimentRunStatus.COMPLETED])
        failed_runs = len([r for r in runs if r.status == ExperimentRunStatus.FAILED])
        running_runs = len([r for r in runs if r.status == ExperimentRunStatus.RUNNING])
        
        # Get latest run
        latest_run = None
        if runs:
            latest_run = max(runs, key=lambda r: r.start_time)
        
        # Aggregate metrics from completed runs
        all_metrics = {}
        for run in runs:
            if run.status == ExperimentRunStatus.COMPLETED and run.metrics:
                for key, value in run.metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # Calculate metric statistics
        metric_stats = {}
        for key, values in all_metrics.items():
            if values:
                metric_stats[key] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "latest": values[-1] if values else None
                }
        
        return {
            "experiment": {
                "id": str(experiment.id),
                "name": experiment.name,
                "description": experiment.description,
                "status": experiment.status.value,
                "created_at": experiment.created_at.isoformat(),
                "updated_at": experiment.updated_at.isoformat(),
                "created_by": experiment.created_by,
                "tags": experiment.tags
            },
            "statistics": {
                "total_runs": total_runs,
                "completed_runs": completed_runs,
                "failed_runs": failed_runs,
                "running_runs": running_runs,
                "success_rate": completed_runs / total_runs if total_runs > 0 else 0
            },
            "latest_run": {
                "id": str(latest_run.id),
                "name": latest_run.name,
                "status": latest_run.status.value,
                "start_time": latest_run.start_time.isoformat(),
                "end_time": latest_run.end_time.isoformat() if latest_run.end_time else None,
                "metrics": latest_run.metrics
            } if latest_run else None,
            "metric_summary": metric_stats
        }