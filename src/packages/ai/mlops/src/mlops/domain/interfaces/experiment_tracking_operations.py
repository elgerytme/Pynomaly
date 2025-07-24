"""Domain interfaces for experiment tracking operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import pandas as pd


class ExperimentStatus(Enum):
    """Experiment status states."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(Enum):
    """Run status states."""
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for creating experiments."""
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentInfo:
    """Experiment information."""
    experiment_id: str
    name: str
    description: Optional[str]
    tags: List[str]
    status: ExperimentStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    run_count: int
    metadata: Dict[str, Any]


@dataclass
class RunConfig:
    """Configuration for creating experiment runs."""
    experiment_id: str
    detector_name: str
    dataset_name: str
    parameters: Dict[str, Any]
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RunMetrics:
    """Run performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None  
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    loss: Optional[float] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None


@dataclass
class RunInfo:
    """Run information."""
    run_id: str
    experiment_id: str
    detector_name: str
    dataset_name: str
    parameters: Dict[str, Any]
    metrics: RunMetrics
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    artifacts: Dict[str, str]
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class ArtifactInfo:
    """Artifact information."""
    artifact_id: str
    name: str
    path: str
    size_bytes: int
    content_type: str
    checksum: str
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class ComparisonRequest:
    """Request for comparing experiment runs."""
    experiment_id: str
    run_ids: Optional[List[str]] = None
    metric_names: Optional[List[str]] = None
    include_parameters: bool = True
    sort_by: str = "f1_score"
    sort_ascending: bool = False


@dataclass
class ComparisonResult:
    """Result of run comparison."""
    experiment_id: str
    comparison_data: pd.DataFrame
    best_run_id: str
    best_metric_value: float
    metric_used: str
    comparison_summary: Dict[str, Any]


class ExperimentTrackingPort(ABC):
    """Port for experiment tracking operations."""

    @abstractmethod
    async def create_experiment(self, config: ExperimentConfig, created_by: str) -> str:
        """Create a new experiment.
        
        Args:
            config: Experiment configuration
            created_by: User who created the experiment
            
        Returns:
            Experiment ID
        """
        pass

    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentInfo]:
        """Get experiment information.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment information or None if not found
        """
        pass

    @abstractmethod
    async def list_experiments(
        self,
        created_by: Optional[str] = None,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ExperimentInfo]:
        """List experiments with optional filters.
        
        Args:
            created_by: Filter by creator
            status: Filter by status
            tags: Filter by tags
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of experiment information
        """
        pass

    @abstractmethod
    async def update_experiment_status(
        self, 
        experiment_id: str, 
        status: ExperimentStatus
    ) -> bool:
        """Update experiment status.
        
        Args:
            experiment_id: ID of the experiment
            status: New status
            
        Returns:
            True if update successful
        """
        pass

    @abstractmethod
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its runs.
        
        Args:
            experiment_id: ID of the experiment to delete
            
        Returns:
            True if deletion successful
        """
        pass


class ExperimentRunPort(ABC):
    """Port for experiment run operations."""

    @abstractmethod
    async def create_run(self, config: RunConfig) -> str:
        """Create a new experiment run.
        
        Args:
            config: Run configuration
            
        Returns:
            Run ID
        """
        pass

    @abstractmethod
    async def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Get run information.
        
        Args:
            run_id: ID of the run
            
        Returns:
            Run information or None if not found
        """
        pass

    @abstractmethod
    async def list_runs(
        self,
        experiment_id: str,
        status: Optional[RunStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[RunInfo]:
        """List runs for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            status: Filter by status
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of run information
        """
        pass

    @abstractmethod
    async def update_run_metrics(self, run_id: str, metrics: RunMetrics) -> bool:
        """Update run metrics.
        
        Args:
            run_id: ID of the run
            metrics: Performance metrics
            
        Returns:
            True if update successful
        """
        pass

    @abstractmethod
    async def update_run_status(self, run_id: str, status: RunStatus) -> bool:
        """Update run status.
        
        Args:
            run_id: ID of the run
            status: New status
            
        Returns:
            True if update successful
        """
        pass

    @abstractmethod
    async def finish_run(self, run_id: str, final_metrics: RunMetrics) -> bool:
        """Finish a run with final metrics.
        
        Args:
            run_id: ID of the run
            final_metrics: Final performance metrics
            
        Returns:
            True if finish successful
        """
        pass


class ArtifactManagementPort(ABC):
    """Port for artifact management operations."""

    @abstractmethod
    async def log_artifact(
        self,
        run_id: str,
        artifact_name: str,
        artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an artifact for a run.
        
        Args:
            run_id: ID of the run
            artifact_name: Name of the artifact
            artifact_path: Path to the artifact file
            metadata: Optional artifact metadata
            
        Returns:
            Artifact ID
        """
        pass

    @abstractmethod
    async def get_artifact(self, artifact_id: str) -> Optional[ArtifactInfo]:
        """Get artifact information.
        
        Args:
            artifact_id: ID of the artifact
            
        Returns:
            Artifact information or None if not found
        """
        pass

    @abstractmethod
    async def download_artifact(self, artifact_id: str, download_path: str) -> bool:
        """Download an artifact to local path.
        
        Args:
            artifact_id: ID of the artifact
            download_path: Local path to download to
            
        Returns:
            True if download successful
        """
        pass

    @abstractmethod
    async def list_run_artifacts(self, run_id: str) -> List[ArtifactInfo]:
        """List artifacts for a run.
        
        Args:
            run_id: ID of the run
            
        Returns:
            List of artifact information
        """
        pass

    @abstractmethod
    async def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact.
        
        Args:
            artifact_id: ID of the artifact to delete
            
        Returns:
            True if deletion successful
        """
        pass


class ExperimentAnalysisPort(ABC):
    """Port for experiment analysis operations."""

    @abstractmethod
    async def compare_runs(self, request: ComparisonRequest) -> ComparisonResult:
        """Compare runs within an experiment.
        
        Args:
            request: Comparison request configuration
            
        Returns:
            Comparison results
        """
        pass

    @abstractmethod
    async def get_best_run(
        self,
        experiment_id: str,
        metric: str = "f1_score",
        higher_is_better: bool = True
    ) -> Optional[RunInfo]:
        """Get the best run from an experiment.
        
        Args:
            experiment_id: ID of the experiment
            metric: Metric to optimize for
            higher_is_better: Whether higher values are better
            
        Returns:
            Best run information or None if no runs found
        """
        pass

    @abstractmethod
    async def create_leaderboard(
        self,
        experiment_ids: Optional[List[str]] = None,
        metric: str = "f1_score",
        limit: int = 50
    ) -> pd.DataFrame:
        """Create a leaderboard across experiments.
        
        Args:
            experiment_ids: Experiments to include (None = all)
            metric: Metric to rank by
            limit: Maximum number of entries
            
        Returns:
            Leaderboard DataFrame
        """
        pass

    @abstractmethod
    async def generate_experiment_report(
        self,
        experiment_id: str,
        include_artifacts: bool = False
    ) -> str:
        """Generate a comprehensive experiment report.
        
        Args:
            experiment_id: ID of the experiment
            include_artifacts: Whether to include artifact information
            
        Returns:
            Report content (markdown format)
        """
        pass

    @abstractmethod
    async def export_experiment_data(
        self,
        experiment_id: str,
        export_format: str = "csv",
        include_artifacts: bool = False
    ) -> str:
        """Export experiment data in specified format.
        
        Args:
            experiment_id: ID of the experiment
            export_format: Export format (csv, json, parquet)
            include_artifacts: Whether to include artifacts
            
        Returns:
            Path to exported file
        """
        pass


class MetricsTrackingPort(ABC):
    """Port for metrics tracking operations."""

    @abstractmethod
    async def log_metric(
        self,
        run_id: str,
        metric_name: str,
        metric_value: float,
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log a metric value for a run.
        
        Args:
            run_id: ID of the run
            metric_name: Name of the metric
            metric_value: Value of the metric
            step: Optional step number
            timestamp: Optional timestamp
        """
        pass

    @abstractmethod
    async def log_metrics_batch(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log multiple metrics for a run.
        
        Args:
            run_id: ID of the run
            metrics: Dictionary of metric names and values
            step: Optional step number
            timestamp: Optional timestamp
        """
        pass

    @abstractmethod
    async def get_metric_history(
        self,
        run_id: str,
        metric_name: str
    ) -> List[Dict[str, Any]]:
        """Get metric history for a run.
        
        Args:
            run_id: ID of the run
            metric_name: Name of the metric
            
        Returns:
            List of metric values with timestamps
        """
        pass

    @abstractmethod
    async def get_all_metrics(self, run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics for a run.
        
        Args:
            run_id: ID of the run
            
        Returns:
            Dictionary of metric histories
        """
        pass


class ExperimentSearchPort(ABC):
    """Port for experiment search operations."""

    @abstractmethod
    async def search_experiments(
        self,
        query: str,
        search_fields: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ExperimentInfo]:
        """Search experiments by query.
        
        Args:
            query: Search query string
            search_fields: Fields to search in (name, description, tags)
            filters: Additional filters
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of matching experiments
        """
        pass

    @abstractmethod
    async def search_runs(
        self,
        query: str,
        experiment_id: Optional[str] = None,
        search_fields: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[RunInfo]:
        """Search runs by query.
        
        Args:
            query: Search query string
            experiment_id: Optional experiment to search within
            search_fields: Fields to search in
            filters: Additional filters
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of matching runs
        """
        pass

    @abstractmethod
    async def find_similar_runs(
        self,
        run_id: str,
        similarity_threshold: float = 0.8,
        limit: int = 10
    ) -> List[RunInfo]:
        """Find runs similar to the given run.
        
        Args:
            run_id: Reference run ID
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results
            
        Returns:
            List of similar runs
        """
        pass