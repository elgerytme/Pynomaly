"""Experiment Tracking Integration Service.

This service provides unified experiment tracking across anomaly detection
and MLOps systems, enabling consistent experiment management and comparison.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Union
from dataclasses import dataclass, asdict
from enum import Enum

# Anomaly detection imports
try:
    from ai.mlops.domain.services.mlops_service import MLOpsService
    from ai.mlops.domain.entities.experiment import ExperimentRun
except ImportError:
    from anomaly_detection.domain.services.mlops_service import MLOpsService, ExperimentRun

try:
    from data.processing.domain.entities.detection_result import DetectionResult
except ImportError:
    from anomaly_detection.domain.entities.detection_result import DetectionResult

# Type stub for MLOps integration (to avoid hard dependency)
try:
    from ai.mlops.domain.services.experiment_tracking_service import ExperimentTrackingService
    from ai.mlops.domain.entities.experiment import Experiment as MLOpsExperiment
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False
    ExperimentTrackingService = Any
    MLOpsExperiment = Any


class ExperimentStatus(Enum):
    """Status of an experiment."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UnifiedExperiment:
    """Unified experiment that spans both systems."""
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    created_by: str
    created_at: datetime
    updated_at: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: List[str]
    run_count: int
    anomaly_experiment_id: Optional[str] = None
    mlops_experiment_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "parameters": self.parameters,
            "metrics": self.metrics,
            "tags": self.tags,
            "run_count": self.run_count,
            "anomaly_experiment_id": self.anomaly_experiment_id,
            "mlops_experiment_id": self.mlops_experiment_id,
            "metadata": self.metadata or {}
        }


@dataclass
class UnifiedExperimentRun:
    """Unified experiment run that spans both systems."""
    run_id: str
    experiment_id: str
    name: str
    status: ExperimentStatus
    started_at: datetime
    ended_at: Optional[datetime]
    duration_seconds: Optional[float]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    model_path: Optional[str]
    tags: List[str]
    created_by: str
    anomaly_run_id: Optional[str] = None
    mlops_run_id: Optional[str] = None
    detection_results: Optional[List[DetectionResult]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "model_path": self.model_path,
            "tags": self.tags,
            "created_by": self.created_by,
            "anomaly_run_id": self.anomaly_run_id,
            "mlops_run_id": self.mlops_run_id,
            "detection_results": [r.to_dict() for r in self.detection_results] if self.detection_results else []
        }


@dataclass
class ExperimentComparisonResult:
    """Result of comparing multiple experiments or runs."""
    comparison_id: str
    experiment_ids: List[str]
    run_ids: List[str]
    best_run: Dict[str, Any]
    performance_summary: Dict[str, Dict[str, float]]
    parameter_differences: Dict[str, List[Any]]
    recommendations: List[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "comparison_id": self.comparison_id,
            "experiment_ids": self.experiment_ids,
            "run_ids": self.run_ids,
            "best_run": self.best_run,
            "performance_summary": self.performance_summary,
            "parameter_differences": self.parameter_differences,
            "recommendations": self.recommendations,
            "created_at": self.created_at.isoformat()
        }


class ExperimentTrackingIntegration:
    """Unified experiment tracking across anomaly detection and MLOps systems."""
    
    def __init__(
        self,
        anomaly_mlops_service: MLOpsService,
        mlops_experiment_service: Optional[ExperimentTrackingService] = None
    ):
        """Initialize experiment tracking integration.
        
        Args:
            anomaly_mlops_service: Anomaly detection MLOps service
            mlops_experiment_service: Optional MLOps experiment tracking service
        """
        self.anomaly_mlops_service = anomaly_mlops_service
        self.mlops_experiment_service = mlops_experiment_service
        self.logger = logging.getLogger(__name__)
        
        # Registry for mapping between systems
        self._experiment_mappings: Dict[str, Dict[str, str]] = {}
        self._run_mappings: Dict[str, Dict[str, str]] = {}
        
        # Check MLOps availability
        self.mlops_integration_enabled = MLOPS_AVAILABLE and mlops_experiment_service is not None
        
        self.logger.info(
            f"Experiment Tracking Integration initialized. MLOps integration: {self.mlops_integration_enabled}"
        )
    
    async def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> UnifiedExperiment:
        """Create a unified experiment in both systems.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: Optional tags
            parameters: Optional experiment parameters
            created_by: User creating the experiment
            
        Returns:
            Unified experiment object
        """
        unified_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        try:
            # Create in anomaly detection system
            anomaly_exp_id = self.anomaly_mlops_service.create_experiment(
                experiment_name=f"{name}_unified_{unified_id[:8]}",
                description=description,
                tags=dict((tag, "true") for tag in (tags or [])),
                created_by=created_by
            )
            
            # Create in MLOps system (if available)
            mlops_exp_id = None
            if self.mlops_integration_enabled:
                mlops_exp_id = await self._create_mlops_experiment(
                    unified_id, name, description, tags, created_by
                )
            
            # Create unified experiment
            unified_experiment = UnifiedExperiment(
                experiment_id=unified_id,
                name=name,
                description=description,
                status=ExperimentStatus.RUNNING,
                created_by=created_by,
                created_at=current_time,
                updated_at=current_time,
                parameters=parameters or {},
                metrics={},
                tags=tags or [],
                run_count=0,
                anomaly_experiment_id=anomaly_exp_id,
                mlops_experiment_id=mlops_exp_id
            )
            
            # Store mapping
            self._experiment_mappings[unified_id] = {
                "anomaly_experiment_id": anomaly_exp_id,
                "mlops_experiment_id": mlops_exp_id or "",
                "created_at": current_time.isoformat()
            }
            
            self.logger.info(f"Created unified experiment '{name}' with ID: {unified_id}")
            return unified_experiment
            
        except Exception as e:
            self.logger.error(f"Failed to create unified experiment '{name}': {e}")
            raise
    
    async def _create_mlops_experiment(
        self,
        unified_id: str,
        name: str,
        description: str,
        tags: Optional[List[str]],
        created_by: str
    ) -> Optional[str]:
        """Create experiment in MLOps system."""
        if not self.mlops_integration_enabled:
            return None
        
        try:
            mlops_experiment = await self.mlops_experiment_service.create_experiment(
                name=f"{name}_unified_{unified_id[:8]}",
                description=description,
                created_by=created_by,
                tags=tags or [],
                metadata={"unified_experiment_id": unified_id}
            )
            return str(mlops_experiment.id)
        except Exception as e:
            self.logger.warning(f"Failed to create MLOps experiment: {e}")
            return None
    
    async def start_run(
        self,
        experiment_id: str,
        run_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        created_by: str = "system"
    ) -> UnifiedExperimentRun:
        """Start a unified experiment run.
        
        Args:
            experiment_id: Unified experiment ID
            run_name: Name for the run
            parameters: Run parameters
            tags: Optional tags
            created_by: User starting the run
            
        Returns:
            Unified experiment run object
        """
        if experiment_id not in self._experiment_mappings:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        unified_run_id = str(uuid.uuid4())
        current_time = datetime.now()
        mapping = self._experiment_mappings[experiment_id]
        
        try:
            # Start run in anomaly detection system
            anomaly_run_id = self.anomaly_mlops_service.start_run(
                experiment_id=mapping["anomaly_experiment_id"],
                parameters=parameters,
                tags=dict((tag, "true") for tag in (tags or []))
            )
            
            # Start run in MLOps system (if available)
            mlops_run_id = None
            if self.mlops_integration_enabled and mapping["mlops_experiment_id"]:
                mlops_run_id = await self._start_mlops_run(
                    mapping["mlops_experiment_id"], run_name, parameters, tags, created_by
                )
            
            # Create unified run
            unified_run = UnifiedExperimentRun(
                run_id=unified_run_id,
                experiment_id=experiment_id,
                name=run_name,
                status=ExperimentStatus.RUNNING,
                started_at=current_time,
                ended_at=None,
                duration_seconds=None,
                parameters=parameters or {},
                metrics={},
                artifacts={},
                model_path=None,
                tags=tags or [],
                created_by=created_by,
                anomaly_run_id=anomaly_run_id,
                mlops_run_id=mlops_run_id
            )
            
            # Store run mapping
            self._run_mappings[unified_run_id] = {
                "experiment_id": experiment_id,
                "anomaly_run_id": anomaly_run_id,
                "mlops_run_id": mlops_run_id or "",
                "started_at": current_time.isoformat()
            }
            
            self.logger.info(f"Started unified run '{run_name}' with ID: {unified_run_id}")
            return unified_run
            
        except Exception as e:
            self.logger.error(f"Failed to start unified run '{run_name}': {e}")
            raise
    
    async def _start_mlops_run(
        self,
        mlops_experiment_id: str,
        run_name: str,
        parameters: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
        created_by: str
    ) -> Optional[str]:
        """Start run in MLOps system."""
        if not self.mlops_integration_enabled:
            return None
        
        try:
            mlops_run = await self.mlops_experiment_service.start_run(
                experiment_id=uuid.UUID(mlops_experiment_id),
                run_name=run_name,
                parameters=parameters or {},
                created_by=created_by,
                tags=tags or []
            )
            return str(mlops_run.id)
        except Exception as e:
            self.logger.warning(f"Failed to start MLOps run: {e}")
            return None
    
    def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log metrics for a unified run.
        
        Args:
            run_id: Unified run ID
            metrics: Metrics to log
            step: Optional step number
        """
        if run_id not in self._run_mappings:
            raise ValueError(f"Run {run_id} not found")
        
        mapping = self._run_mappings[run_id]
        
        try:
            # Log to anomaly detection system
            self.anomaly_mlops_service.log_metrics(
                mapping["anomaly_run_id"], metrics, step
            )
            
            # Log to MLOps system (if available)
            if self.mlops_integration_enabled and mapping["mlops_run_id"]:
                # MLOps system logging would go here
                pass
            
            self.logger.debug(f"Logged metrics for unified run {run_id}: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics for run {run_id}: {e}")
            raise
    
    def log_parameters(
        self,
        run_id: str,
        parameters: Dict[str, Any]
    ) -> None:
        """Log parameters for a unified run.
        
        Args:
            run_id: Unified run ID
            parameters: Parameters to log
        """
        if run_id not in self._run_mappings:
            raise ValueError(f"Run {run_id} not found")
        
        mapping = self._run_mappings[run_id]
        
        try:
            # Update parameters in anomaly detection system
            anomaly_run = self.anomaly_mlops_service._runs.get(mapping["anomaly_run_id"])
            if anomaly_run:
                anomaly_run.parameters.update(parameters)
            
            # Log to MLOps system (if available)
            if self.mlops_integration_enabled and mapping["mlops_run_id"]:
                # MLOps system logging would go here
                pass
            
            self.logger.debug(f"Logged parameters for unified run {run_id}: {parameters}")
            
        except Exception as e:
            self.logger.error(f"Failed to log parameters for run {run_id}: {e}")
            raise
    
    def log_artifact(
        self,
        run_id: str,
        artifact_name: str,
        artifact_path: str
    ) -> None:
        """Log an artifact for a unified run.
        
        Args:
            run_id: Unified run ID
            artifact_name: Name of the artifact
            artifact_path: Path to the artifact
        """
        if run_id not in self._run_mappings:
            raise ValueError(f"Run {run_id} not found")
        
        mapping = self._run_mappings[run_id]
        
        try:
            # Log to anomaly detection system
            self.anomaly_mlops_service.log_artifact(
                mapping["anomaly_run_id"], artifact_name, artifact_path
            )
            
            # Log to MLOps system (if available)
            if self.mlops_integration_enabled and mapping["mlops_run_id"]:
                # MLOps system artifact logging would go here
                pass
            
            self.logger.debug(f"Logged artifact '{artifact_name}' for unified run {run_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log artifact for run {run_id}: {e}")
            raise
    
    def log_model(
        self,
        run_id: str,
        model: Any,
        model_name: str = "model"
    ) -> str:
        """Log a model for a unified run.
        
        Args:
            run_id: Unified run ID
            model: Model object
            model_name: Name for the model
            
        Returns:
            Path where model was saved
        """
        if run_id not in self._run_mappings:
            raise ValueError(f"Run {run_id} not found")
        
        mapping = self._run_mappings[run_id]
        
        try:
            # Log to anomaly detection system
            model_path = self.anomaly_mlops_service.log_model(
                mapping["anomaly_run_id"], model, model_name
            )
            
            # Log to MLOps system (if available)
            if self.mlops_integration_enabled and mapping["mlops_run_id"]:
                # MLOps system model logging would go here
                pass
            
            self.logger.info(f"Logged model '{model_name}' for unified run {run_id}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"Failed to log model for run {run_id}: {e}")
            raise
    
    async def end_run(
        self,
        run_id: str,
        status: str = "completed"
    ) -> UnifiedExperimentRun:
        """End a unified experiment run.
        
        Args:
            run_id: Unified run ID
            status: Final status
            
        Returns:
            Updated unified run object
        """
        if run_id not in self._run_mappings:
            raise ValueError(f"Run {run_id} not found")
        
        mapping = self._run_mappings[run_id]
        current_time = datetime.now()
        
        try:
            # End run in anomaly detection system
            self.anomaly_mlops_service.end_run(mapping["anomaly_run_id"], status)
            
            # End run in MLOps system (if available)
            if self.mlops_integration_enabled and mapping["mlops_run_id"]:
                # MLOps system run ending would go here
                pass
            
            # Get final run data
            updated_run = await self.get_run(run_id)
            updated_run.ended_at = current_time
            updated_run.status = ExperimentStatus(status)
            
            # Calculate duration
            if updated_run.started_at:
                duration = current_time - updated_run.started_at
                updated_run.duration_seconds = duration.total_seconds()
            
            self.logger.info(f"Ended unified run {run_id} with status: {status}")
            return updated_run
            
        except Exception as e:
            self.logger.error(f"Failed to end unified run {run_id}: {e}")
            raise
    
    async def get_experiment(self, experiment_id: str) -> Optional[UnifiedExperiment]:
        """Get unified experiment by ID.
        
        Args:
            experiment_id: Unified experiment ID
            
        Returns:
            Unified experiment or None
        """
        if experiment_id not in self._experiment_mappings:
            return None
        
        mapping = self._experiment_mappings[experiment_id]
        
        try:
            # Get runs for this experiment
            runs = await self.list_runs(experiment_id)
            run_count = len(runs)
            
            # Get latest metrics from runs
            latest_metrics = {}
            if runs:
                latest_run = max(runs, key=lambda r: r.started_at)
                latest_metrics = latest_run.metrics
            
            # Get experiment data from anomaly detection system
            anomaly_experiment = self.anomaly_mlops_service._experiments.get(
                mapping["anomaly_experiment_id"]
            )
            
            if not anomaly_experiment:
                return None
            
            return UnifiedExperiment(
                experiment_id=experiment_id,
                name=anomaly_experiment.experiment_name,
                description=anomaly_experiment.description,
                status=ExperimentStatus.RUNNING if run_count == 0 else ExperimentStatus.COMPLETED,
                created_by=anomaly_experiment.created_by,
                created_at=anomaly_experiment.created_at,
                updated_at=datetime.now(),
                parameters=anomaly_experiment.parameters,
                metrics=latest_metrics,
                tags=list(anomaly_experiment.tags.keys()),
                run_count=run_count,
                anomaly_experiment_id=mapping["anomaly_experiment_id"],
                mlops_experiment_id=mapping["mlops_experiment_id"] or None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment {experiment_id}: {e}")
            return None
    
    async def get_run(self, run_id: str) -> Optional[UnifiedExperimentRun]:
        """Get unified run by ID.
        
        Args:
            run_id: Unified run ID
            
        Returns:
            Unified run or None
        """
        if run_id not in self._run_mappings:
            return None
        
        mapping = self._run_mappings[run_id]
        
        try:
            # Get run from anomaly detection system
            anomaly_run = self.anomaly_mlops_service._runs.get(mapping["anomaly_run_id"])
            
            if not anomaly_run:
                return None
            
            # Calculate duration
            duration_seconds = None
            if anomaly_run.end_time:
                duration = anomaly_run.end_time - anomaly_run.start_time
                duration_seconds = duration.total_seconds()
            
            return UnifiedExperimentRun(
                run_id=run_id,
                experiment_id=mapping["experiment_id"],
                name=f"run_{run_id[:8]}",
                status=ExperimentStatus(anomaly_run.status),
                started_at=anomaly_run.start_time,
                ended_at=anomaly_run.end_time,
                duration_seconds=duration_seconds,
                parameters=anomaly_run.parameters,
                metrics=anomaly_run.metrics,
                artifacts=anomaly_run.artifacts,
                model_path=anomaly_run.model_path,
                tags=[],
                created_by="system",
                anomaly_run_id=mapping["anomaly_run_id"],
                mlops_run_id=mapping["mlops_run_id"] or None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get run {run_id}: {e}")
            return None
    
    async def list_experiments(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[UnifiedExperiment]:
        """List all unified experiments.
        
        Args:
            filters: Optional filters
            
        Returns:
            List of unified experiments
        """
        experiments = []
        
        for experiment_id in self._experiment_mappings.keys():
            try:
                experiment = await self.get_experiment(experiment_id)
                if experiment and self._matches_experiment_filters(experiment, filters):
                    experiments.append(experiment)
            except Exception as e:
                self.logger.warning(f"Failed to get experiment {experiment_id}: {e}")
                continue
        
        return experiments
    
    async def list_runs(
        self,
        experiment_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[UnifiedExperimentRun]:
        """List unified runs.
        
        Args:
            experiment_id: Optional experiment ID to filter by
            filters: Optional additional filters
            
        Returns:
            List of unified runs
        """
        runs = []
        
        for run_id, mapping in self._run_mappings.items():
            if experiment_id and mapping["experiment_id"] != experiment_id:
                continue
            
            try:
                run = await self.get_run(run_id)
                if run and self._matches_run_filters(run, filters):
                    runs.append(run)
            except Exception as e:
                self.logger.warning(f"Failed to get run {run_id}: {e}")
                continue
        
        return runs
    
    def _matches_experiment_filters(
        self, 
        experiment: UnifiedExperiment, 
        filters: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if experiment matches filters."""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key == "status" and experiment.status.value != value:
                return False
            elif key == "created_by" and experiment.created_by != value:
                return False
            elif key == "tags" and not any(tag in experiment.tags for tag in value):
                return False
        
        return True
    
    def _matches_run_filters(
        self, 
        run: UnifiedExperimentRun, 
        filters: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if run matches filters."""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key == "status" and run.status.value != value:
                return False
            elif key == "created_by" and run.created_by != value:
                return False
            elif key == "min_duration" and (not run.duration_seconds or run.duration_seconds < value):
                return False
        
        return True
    
    async def compare_runs(
        self, 
        run_ids: List[str],
        metrics_to_compare: Optional[List[str]] = None
    ) -> ExperimentComparisonResult:
        """Compare multiple experiment runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics_to_compare: Optional specific metrics to compare
            
        Returns:
            Comparison result
        """
        comparison_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        # Get all runs
        runs = []
        for run_id in run_ids:
            run = await self.get_run(run_id)
            if run:
                runs.append(run)
        
        if not runs:
            raise ValueError("No valid runs found for comparison")
        
        # Determine metrics to compare
        if not metrics_to_compare:
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.metrics.keys())
            metrics_to_compare = list(all_metrics)
        
        # Calculate performance summary
        performance_summary = {}
        for run in runs:
            performance_summary[run.run_id] = {
                metric: run.metrics.get(metric, 0.0)
                for metric in metrics_to_compare
            }
        
        # Find best run (highest average of normalized metrics)
        best_run_id = None
        best_score = -float('inf')
        
        for run in runs:
            score = sum(run.metrics.get(metric, 0.0) for metric in metrics_to_compare)
            if score > best_score:
                best_score = score
                best_run_id = run.run_id
        
        best_run = next(run.to_dict() for run in runs if run.run_id == best_run_id)
        
        # Analyze parameter differences
        parameter_differences = {}
        all_params = set()
        for run in runs:
            all_params.update(run.parameters.keys())
        
        for param in all_params:
            values = [run.parameters.get(param, None) for run in runs]
            unique_values = list(set(values))
            if len(unique_values) > 1:
                parameter_differences[param] = unique_values
        
        # Generate recommendations
        recommendations = self._generate_comparison_recommendations(
            runs, performance_summary, parameter_differences
        )
        
        return ExperimentComparisonResult(
            comparison_id=comparison_id,
            experiment_ids=list(set(run.experiment_id for run in runs)),
            run_ids=run_ids,
            best_run=best_run,
            performance_summary=performance_summary,
            parameter_differences=parameter_differences,
            recommendations=recommendations,
            created_at=current_time
        )
    
    def _generate_comparison_recommendations(
        self,
        runs: List[UnifiedExperimentRun],
        performance_summary: Dict[str, Dict[str, float]],
        parameter_differences: Dict[str, List[Any]]
    ) -> List[str]:
        """Generate recommendations based on run comparison."""
        recommendations = []
        
        if len(runs) < 2:
            recommendations.append("Need at least 2 runs for meaningful comparison")
            return recommendations
        
        # Find most impactful parameters
        if parameter_differences:
            recommendations.append(
                f"Key parameters that vary: {', '.join(list(parameter_differences.keys())[:3])}"
            )
        
        # Performance insights
        if performance_summary:
            metrics = list(next(iter(performance_summary.values())).keys())
            if metrics:
                best_metric = metrics[0]  # Simplified - use first metric
                values = [summary.get(best_metric, 0) for summary in performance_summary.values()]
                if max(values) - min(values) > 0.1:
                    recommendations.append(f"Significant variation in {best_metric} performance")
        
        # Duration insights
        durations = [run.duration_seconds for run in runs if run.duration_seconds]
        if len(durations) > 1:
            avg_duration = sum(durations) / len(durations)
            fastest_run = min(runs, key=lambda r: r.duration_seconds or float('inf'))
            if fastest_run.duration_seconds and fastest_run.duration_seconds < avg_duration * 0.8:
                recommendations.append(f"Run {fastest_run.run_id[:8]} was significantly faster")
        
        return recommendations
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about the integration.
        
        Returns:
            Integration statistics
        """
        return {
            "total_experiments": len(self._experiment_mappings),
            "total_runs": len(self._run_mappings),
            "anomaly_experiments": sum(
                1 for m in self._experiment_mappings.values() 
                if m["anomaly_experiment_id"]
            ),
            "mlops_experiments": sum(
                1 for m in self._experiment_mappings.values() 
                if m["mlops_experiment_id"]
            ),
            "mlops_integration_enabled": self.mlops_integration_enabled,
            "active_runs": len([
                run_id for run_id, mapping in self._run_mappings.items()
                if self.anomaly_mlops_service._runs.get(mapping["anomaly_run_id"], {}).get("status") == "running"
            ])
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the integration.
        
        Returns:
            Health status information
        """
        status = {
            "experiment_tracking_integration": "healthy",
            "anomaly_detection_service": "unknown",
            "mlops_service": "unknown",
            "experiment_mappings": len(self._experiment_mappings),
            "run_mappings": len(self._run_mappings),
            "last_check": datetime.now().isoformat()
        }
        
        try:
            # Check anomaly detection service
            if hasattr(self.anomaly_mlops_service, '_experiments'):
                status["anomaly_detection_service"] = "healthy"
        except Exception:
            status["anomaly_detection_service"] = "unhealthy"
        
        try:
            # Check MLOps service
            if self.mlops_integration_enabled and self.mlops_experiment_service:
                status["mlops_service"] = "healthy"
            else:
                status["mlops_service"] = "disabled"
        except Exception:
            status["mlops_service"] = "unhealthy"
        
        return status


# Global integration instance
_experiment_tracking_integration: Optional[ExperimentTrackingIntegration] = None


def initialize_experiment_tracking_integration(
    anomaly_mlops_service: MLOpsService,
    mlops_experiment_service: Optional[ExperimentTrackingService] = None
) -> ExperimentTrackingIntegration:
    """Initialize global experiment tracking integration.
    
    Args:
        anomaly_mlops_service: Anomaly detection MLOps service
        mlops_experiment_service: Optional MLOps experiment tracking service
        
    Returns:
        Initialized experiment tracking integration
    """
    global _experiment_tracking_integration
    _experiment_tracking_integration = ExperimentTrackingIntegration(
        anomaly_mlops_service=anomaly_mlops_service,
        mlops_experiment_service=mlops_experiment_service
    )
    return _experiment_tracking_integration


def get_experiment_tracking_integration() -> Optional[ExperimentTrackingIntegration]:
    """Get global experiment tracking integration instance.
    
    Returns:
        Experiment tracking integration instance or None
    """
    return _experiment_tracking_integration