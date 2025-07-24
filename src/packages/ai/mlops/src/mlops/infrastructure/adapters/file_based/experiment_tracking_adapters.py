"""File-based implementations for experiment tracking operations."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from mlops.domain.interfaces.experiment_tracking_operations import (
    ExperimentTrackingPort,
    ExperimentRunPort,
    ArtifactManagementPort,
    ExperimentAnalysisPort,
    MetricsTrackingPort,
    ExperimentSearchPort,
    ExperimentConfig,
    ExperimentInfo,
    ExperimentStatus,
    RunConfig,
    RunInfo,
    RunStatus,
    RunMetrics,
    ComparisonRequest,
    ComparisonResult
)

logger = logging.getLogger(__name__)


class FileBasedExperimentTracking(ExperimentTrackingPort):
    """File-based implementation for experiment tracking."""
    
    def __init__(self, storage_path: str = "./mlops_data/experiments"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._experiments_file = self._storage_path / "experiments.json"
        self._experiments = self._load_experiments()
        logger.info(f"FileBasedExperimentTracking initialized with storage at {storage_path}")
    
    def _load_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Load experiments from file."""
        if self._experiments_file.exists():
            try:
                with open(self._experiments_file, 'r') as f:
                    data = json.load(f)
                    # Convert string timestamps back to datetime objects
                    for exp_data in data.values():
                        exp_data['created_at'] = datetime.fromisoformat(exp_data['created_at'])
                        exp_data['updated_at'] = datetime.fromisoformat(exp_data['updated_at'])
                    return data
            except Exception as e:
                logger.warning(f"Failed to load experiments file: {e}")
        return {}
    
    def _save_experiments(self) -> None:
        """Save experiments to file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = {}
            for exp_id, exp_data in self._experiments.items():
                serializable_data[exp_id] = exp_data.copy()
                serializable_data[exp_id]['created_at'] = exp_data['created_at'].isoformat()
                serializable_data[exp_id]['updated_at'] = exp_data['updated_at'].isoformat()
            
            with open(self._experiments_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save experiments file: {e}")
    
    async def create_experiment(self, config: ExperimentConfig, created_by: str) -> str:
        """Create a new experiment."""
        experiment_id = f"exp_{str(uuid4())[:8]}"
        
        experiment_data = {
            "experiment_id": experiment_id,
            "name": config.name,
            "description": config.description,
            "tags": config.tags,
            "metadata": config.metadata,
            "created_by": created_by,
            "status": ExperimentStatus.CREATED.value,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        self._experiments[experiment_id] = experiment_data
        self._save_experiments()
        
        logger.info(f"Created experiment {experiment_id}")
        return experiment_id
    
    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentInfo]:
        """Get experiment information."""
        exp_data = self._experiments.get(experiment_id)
        if not exp_data:
            return None
        
        return ExperimentInfo(
            experiment_id=exp_data["experiment_id"],
            name=exp_data["name"],
            description=exp_data["description"],
            tags=exp_data["tags"],
            metadata=exp_data["metadata"],
            created_by=exp_data["created_by"],
            status=ExperimentStatus(exp_data["status"]),
            created_at=exp_data["created_at"],
            updated_at=exp_data["updated_at"]
        )
    
    async def list_experiments(
        self, 
        created_by: Optional[str] = None,
        status: Optional[ExperimentStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ExperimentInfo]:
        """List experiments with filters."""
        experiments = []
        
        for exp_data in self._experiments.values():
            # Apply filters
            if created_by and exp_data["created_by"] != created_by:
                continue
            if status and exp_data["status"] != status.value:
                continue
            
            experiments.append(ExperimentInfo(
                experiment_id=exp_data["experiment_id"],
                name=exp_data["name"],
                description=exp_data["description"],
                tags=exp_data["tags"],
                metadata=exp_data["metadata"],
                created_by=exp_data["created_by"],
                status=ExperimentStatus(exp_data["status"]),
                created_at=exp_data["created_at"],
                updated_at=exp_data["updated_at"]
            ))
        
        # Sort by created_at descending
        experiments.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        return experiments[offset:offset + limit]
    
    async def update_experiment_status(
        self,
        experiment_id: str,
        status: ExperimentStatus
    ) -> bool:
        """Update experiment status."""
        if experiment_id not in self._experiments:
            return False
        
        self._experiments[experiment_id]["status"] = status.value
        self._experiments[experiment_id]["updated_at"] = datetime.utcnow()
        self._save_experiments()
        
        logger.info(f"Updated experiment {experiment_id} status to {status.value}")
        return True
    
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        if experiment_id not in self._experiments:
            return False
        
        del self._experiments[experiment_id]
        self._save_experiments()
        
        # Also delete runs directory
        runs_dir = self._storage_path / "runs" / experiment_id
        if runs_dir.exists():
            import shutil
            shutil.rmtree(runs_dir)
        
        logger.info(f"Deleted experiment {experiment_id}")
        return True


class FileBasedExperimentRun(ExperimentRunPort):
    """File-based implementation for experiment runs."""
    
    def __init__(self, storage_path: str = "./mlops_data/runs"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._runs_file = self._storage_path / "runs.json"
        self._runs = self._load_runs()
        logger.info(f"FileBasedExperimentRun initialized with storage at {storage_path}")
    
    def _load_runs(self) -> Dict[str, Dict[str, Any]]:
        """Load runs from file."""
        if self._runs_file.exists():
            try:
                with open(self._runs_file, 'r') as f:
                    data = json.load(f)
                    # Convert string timestamps back to datetime objects
                    for run_data in data.values():
                        run_data['created_at'] = datetime.fromisoformat(run_data['created_at'])
                        run_data['updated_at'] = datetime.fromisoformat(run_data['updated_at'])
                        if run_data.get('started_at'):
                            run_data['started_at'] = datetime.fromisoformat(run_data['started_at'])
                        if run_data.get('finished_at'):
                            run_data['finished_at'] = datetime.fromisoformat(run_data['finished_at'])
                    return data
            except Exception as e:
                logger.warning(f"Failed to load runs file: {e}")
        return {}
    
    def _save_runs(self) -> None:
        """Save runs to file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = {}
            for run_id, run_data in self._runs.items():
                serializable_data[run_id] = run_data.copy()
                serializable_data[run_id]['created_at'] = run_data['created_at'].isoformat()
                serializable_data[run_id]['updated_at'] = run_data['updated_at'].isoformat()
                if run_data.get('started_at'):
                    serializable_data[run_id]['started_at'] = run_data['started_at'].isoformat()
                if run_data.get('finished_at'):
                    serializable_data[run_id]['finished_at'] = run_data['finished_at'].isoformat()
            
            with open(self._runs_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save runs file: {e}")
    
    async def create_run(self, config: RunConfig) -> str:
        """Create a new experiment run."""
        run_id = f"run_{str(uuid4())[:8]}"
        
        run_data = {
            "run_id": run_id,
            "experiment_id": config.experiment_id,
            "detector_name": config.detector_name,
            "dataset_name": config.dataset_name,
            "parameters": config.parameters,
            "tags": config.tags or [],
            "status": RunStatus.STARTED.value,
            "metrics": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "started_at": datetime.utcnow(),
            "finished_at": None
        }
        
        self._runs[run_id] = run_data
        self._save_runs()
        
        logger.info(f"Created run {run_id} for experiment {config.experiment_id}")
        return run_id
    
    async def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Get run information."""
        run_data = self._runs.get(run_id)
        if not run_data:
            return None
        
        # Convert metrics dict to RunMetrics object
        metrics_data = run_data.get("metrics", {})
        metrics = RunMetrics(
            accuracy=metrics_data.get("accuracy"),
            precision=metrics_data.get("precision"),
            recall=metrics_data.get("recall"),
            f1_score=metrics_data.get("f1_score"),
            auc_roc=metrics_data.get("auc_roc"),
            loss=metrics_data.get("loss"),
            training_time=metrics_data.get("training_time"),
            inference_time=metrics_data.get("inference_time"),
            custom_metrics={k: v for k, v in metrics_data.items() 
                          if k not in ["accuracy", "precision", "recall", "f1_score", 
                                     "auc_roc", "loss", "training_time", "inference_time"]}
        )
        
        return RunInfo(
            run_id=run_data["run_id"],
            experiment_id=run_data["experiment_id"],
            detector_name=run_data["detector_name"],
            dataset_name=run_data["dataset_name"],
            parameters=run_data["parameters"],
            tags=run_data["tags"],
            status=RunStatus(run_data["status"]),
            metrics=metrics,
            created_at=run_data["created_at"],
            updated_at=run_data["updated_at"],
            started_at=run_data.get("started_at"),
            finished_at=run_data.get("finished_at")
        )
    
    async def list_runs(
        self,
        experiment_id: str,
        status: Optional[RunStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[RunInfo]:
        """List runs for an experiment."""
        runs = []
        
        for run_data in self._runs.values():
            # Filter by experiment_id
            if run_data["experiment_id"] != experiment_id:
                continue
            
            # Apply status filter
            if status and run_data["status"] != status.value:
                continue
            
            # Convert to RunInfo
            metrics_data = run_data.get("metrics", {})
            metrics = RunMetrics(
                accuracy=metrics_data.get("accuracy"),
                precision=metrics_data.get("precision"),
                recall=metrics_data.get("recall"),
                f1_score=metrics_data.get("f1_score"),
                auc_roc=metrics_data.get("auc_roc"),
                loss=metrics_data.get("loss"),
                training_time=metrics_data.get("training_time"),
                inference_time=metrics_data.get("inference_time"),
                custom_metrics={k: v for k, v in metrics_data.items() 
                              if k not in ["accuracy", "precision", "recall", "f1_score", 
                                         "auc_roc", "loss", "training_time", "inference_time"]}
            )
            
            runs.append(RunInfo(
                run_id=run_data["run_id"],
                experiment_id=run_data["experiment_id"],
                detector_name=run_data["detector_name"],
                dataset_name=run_data["dataset_name"],
                parameters=run_data["parameters"],
                tags=run_data["tags"],
                status=RunStatus(run_data["status"]),
                metrics=metrics,
                created_at=run_data["created_at"],
                updated_at=run_data["updated_at"],
                started_at=run_data.get("started_at"),
                finished_at=run_data.get("finished_at")
            ))
        
        # Sort by created_at descending
        runs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        return runs[offset:offset + limit]
    
    async def update_run_status(self, run_id: str, status: RunStatus) -> bool:
        """Update run status."""
        if run_id not in self._runs:
            return False
        
        self._runs[run_id]["status"] = status.value
        self._runs[run_id]["updated_at"] = datetime.utcnow()
        
        if status == RunStatus.RUNNING and not self._runs[run_id].get("started_at"):
            self._runs[run_id]["started_at"] = datetime.utcnow()
        elif status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
            self._runs[run_id]["finished_at"] = datetime.utcnow()
        
        self._save_runs()
        logger.info(f"Updated run {run_id} status to {status.value}")
        return True
    
    async def update_run_metrics(self, run_id: str, metrics: RunMetrics) -> bool:
        """Update run metrics."""
        if run_id not in self._runs:
            return False
        
        # Convert RunMetrics to dict
        metrics_dict = {}
        if metrics.accuracy is not None:
            metrics_dict["accuracy"] = metrics.accuracy
        if metrics.precision is not None:
            metrics_dict["precision"] = metrics.precision
        if metrics.recall is not None:
            metrics_dict["recall"] = metrics.recall
        if metrics.f1_score is not None:
            metrics_dict["f1_score"] = metrics.f1_score
        if metrics.auc_roc is not None:
            metrics_dict["auc_roc"] = metrics.auc_roc
        if metrics.loss is not None:
            metrics_dict["loss"] = metrics.loss
        if metrics.training_time is not None:
            metrics_dict["training_time"] = metrics.training_time
        if metrics.inference_time is not None:
            metrics_dict["inference_time"] = metrics.inference_time
        
        # Add custom metrics
        metrics_dict.update(metrics.custom_metrics)
        
        self._runs[run_id]["metrics"] = metrics_dict
        self._runs[run_id]["updated_at"] = datetime.utcnow()
        self._save_runs()
        
        logger.info(f"Updated metrics for run {run_id}")
        return True
    
    async def finish_run(self, run_id: str, final_metrics: RunMetrics) -> bool:
        """Finish a run with final metrics."""
        if run_id not in self._runs:
            return False
        
        # Update metrics
        await self.update_run_metrics(run_id, final_metrics)
        
        # Update status
        await self.update_run_status(run_id, RunStatus.COMPLETED)
        
        logger.info(f"Finished run {run_id}")
        return True
    
    async def delete_run(self, run_id: str) -> bool:
        """Delete a run."""
        if run_id not in self._runs:
            return False
        
        del self._runs[run_id]
        self._save_runs()
        
        logger.info(f"Deleted run {run_id}")
        return True


class FileBasedArtifactManagement(ArtifactManagementPort):
    """File-based implementation for artifact management."""
    
    def __init__(self, storage_path: str = "./mlops_data/artifacts"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._artifacts_file = self._storage_path / "artifacts.json"
        self._artifacts = self._load_artifacts()
        logger.info(f"FileBasedArtifactManagement initialized with storage at {storage_path}")
    
    def _load_artifacts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load artifacts metadata from file."""
        if self._artifacts_file.exists():
            try:
                with open(self._artifacts_file, 'r') as f:
                    data = json.load(f)
                    # Convert string timestamps back to datetime objects
                    for run_artifacts in data.values():
                        for artifact in run_artifacts:
                            artifact['created_at'] = datetime.fromisoformat(artifact['created_at'])
                    return data
            except Exception as e:
                logger.warning(f"Failed to load artifacts file: {e}")
        return {}
    
    def _save_artifacts(self) -> None:
        """Save artifacts metadata to file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_data = {}
            for run_id, artifacts in self._artifacts.items():
                serializable_data[run_id] = []
                for artifact in artifacts:
                    serializable_artifact = artifact.copy()
                    serializable_artifact['created_at'] = artifact['created_at'].isoformat()
                    serializable_data[run_id].append(serializable_artifact)
            
            with open(self._artifacts_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save artifacts file: {e}")
    
    async def store_artifact(
        self,
        run_id: str,
        artifact_name: str,
        artifact_data: bytes,
        artifact_type: str = "binary",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store an artifact."""
        artifact_id = f"artifact_{str(uuid4())[:8]}"
        
        # Create run-specific directory
        run_dir = self._storage_path / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save artifact data to file
        artifact_path = run_dir / f"{artifact_id}_{artifact_name}"
        with open(artifact_path, 'wb') as f:
            f.write(artifact_data)
        
        # Store artifact metadata
        if run_id not in self._artifacts:
            self._artifacts[run_id] = []
        
        artifact_metadata = {
            "artifact_id": artifact_id,
            "name": artifact_name,
            "type": artifact_type,
            "size_bytes": len(artifact_data),
            "path": str(artifact_path),
            "metadata": metadata or {},
            "created_at": datetime.utcnow()
        }
        
        self._artifacts[run_id].append(artifact_metadata)
        self._save_artifacts()
        
        logger.info(f"Stored artifact {artifact_name} for run {run_id}")
        return artifact_id
    
    async def retrieve_artifact(self, run_id: str, artifact_id: str) -> Optional[bytes]:
        """Retrieve an artifact."""
        run_artifacts = self._artifacts.get(run_id, [])
        artifact_metadata = next((a for a in run_artifacts if a["artifact_id"] == artifact_id), None)
        
        if not artifact_metadata:
            return None
        
        try:
            artifact_path = Path(artifact_metadata["path"])
            if artifact_path.exists():
                with open(artifact_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to retrieve artifact {artifact_id}: {e}")
        
        return None
    
    async def list_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        """List artifacts for a run."""
        return self._artifacts.get(run_id, [])
    
    async def delete_artifact(self, run_id: str, artifact_id: str) -> bool:
        """Delete an artifact."""
        run_artifacts = self._artifacts.get(run_id, [])
        artifact_metadata = next((a for a in run_artifacts if a["artifact_id"] == artifact_id), None)
        
        if not artifact_metadata:
            return False
        
        try:
            # Delete file
            artifact_path = Path(artifact_metadata["path"])
            if artifact_path.exists():
                artifact_path.unlink()
            
            # Remove from metadata
            self._artifacts[run_id] = [a for a in run_artifacts if a["artifact_id"] != artifact_id]
            self._save_artifacts()
            
            logger.info(f"Deleted artifact {artifact_id} for run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            return False


class FileBasedExperimentAnalysis(ExperimentAnalysisPort):
    """File-based implementation for experiment analysis."""
    
    def __init__(self, run_port: FileBasedExperimentRun):
        self._run_port = run_port
        logger.info("FileBasedExperimentAnalysis initialized")
    
    async def compare_runs(self, request: ComparisonRequest) -> ComparisonResult:
        """Compare experiment runs."""
        import pandas as pd
        
        # Get runs to compare
        if request.run_ids:
            runs = []
            for run_id in request.run_ids:
                run = await self._run_port.get_run(run_id)
                if run:
                    runs.append(run)
        else:
            runs = await self._run_port.list_runs(request.experiment_id)
        
        if not runs:
            return ComparisonResult(
                experiment_id=request.experiment_id,
                runs_compared=[],
                comparison_data=pd.DataFrame(),
                metric_used=request.sort_by,
                summary_statistics={}
            )
        
        # Create comparison DataFrame
        comparison_data = []
        for run in runs:
            row = {
                "run_id": run.run_id,
                "detector_name": run.detector_name,
                "dataset_name": run.dataset_name,
                "status": run.status.value,
                "accuracy": run.metrics.accuracy,
                "precision": run.metrics.precision,
                "recall": run.metrics.recall,
                "f1_score": run.metrics.f1_score,
                "auc_roc": run.metrics.auc_roc,
                "loss": run.metrics.loss,
                "training_time": run.metrics.training_time,
                "inference_time": run.metrics.inference_time
            }
            
            # Add parameters if requested
            if request.include_parameters:
                for param_name, param_value in run.parameters.items():
                    row[f"param_{param_name}"] = param_value
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric
        if request.sort_by in df.columns and not df[request.sort_by].isna().all():
            df = df.sort_values(request.sort_by, ascending=False, na_last=True)
        
        # Calculate summary statistics
        numeric_columns = df.select_dtypes(include=[float, int]).columns
        summary_stats = {}
        for col in numeric_columns:
            if not df[col].isna().all():
                summary_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "count": int(df[col].count())
                }
        
        return ComparisonResult(
            experiment_id=request.experiment_id,
            runs_compared=[run.run_id for run in runs],
            comparison_data=df,
            metric_used=request.sort_by,
            summary_statistics=summary_stats
        )
    
    async def get_best_run(
        self,
        experiment_id: str,
        metric: str = "f1_score",
        higher_is_better: bool = True
    ) -> Optional[RunInfo]:
        """Get the best performing run."""
        runs = await self._run_port.list_runs(experiment_id, status=RunStatus.COMPLETED)
        
        if not runs:
            return None
        
        # Filter runs that have the specified metric
        runs_with_metric = []
        for run in runs:
            metric_value = getattr(run.metrics, metric, None)
            if metric_value is not None:
                runs_with_metric.append((run, metric_value))
        
        if not runs_with_metric:
            return None
        
        # Find best run
        best_run, _ = max(runs_with_metric, key=lambda x: x[1] if higher_is_better else -x[1])
        return best_run
    
    async def generate_experiment_report(
        self,
        experiment_id: str,
        include_artifacts: bool = False
    ) -> str:
        """Generate experiment report."""
        # Get experiment runs
        runs = await self._run_port.list_runs(experiment_id)
        
        report_lines = [
            f"# Experiment Report: {experiment_id}",
            f"Generated at: {datetime.utcnow().isoformat()}",
            "",
            f"## Overview",
            f"- Total runs: {len(runs)}",
            f"- Completed runs: {len([r for r in runs if r.status == RunStatus.COMPLETED])}",
            f"- Failed runs: {len([r for r in runs if r.status == RunStatus.FAILED])}",
            ""
        ]
        
        if runs:
            # Performance summary
            completed_runs = [r for r in runs if r.status == RunStatus.COMPLETED]
            if completed_runs:
                f1_scores = [r.metrics.f1_score for r in completed_runs if r.metrics.f1_score is not None]
                if f1_scores:
                    report_lines.extend([
                        "## Performance Summary",
                        f"- Best F1 Score: {max(f1_scores):.3f}",
                        f"- Average F1 Score: {sum(f1_scores)/len(f1_scores):.3f}",
                        f"- Worst F1 Score: {min(f1_scores):.3f}",
                        ""
                    ])
            
            # Runs detail
            report_lines.append("## Runs Detail")
            for run in runs[:10]:  # Limit to first 10 runs
                report_lines.extend([
                    f"### Run {run.run_id}",
                    f"- Detector: {run.detector_name}",
                    f"- Dataset: {run.dataset_name}",
                    f"- Status: {run.status.value}",
                    f"- F1 Score: {run.metrics.f1_score or 'N/A'}",
                    f"- Accuracy: {run.metrics.accuracy or 'N/A'}",
                    ""
                ])
        
        return "\n".join(report_lines)


class FileBasedMetricsTracking(MetricsTrackingPort):
    """File-based implementation for metrics tracking."""
    
    def __init__(self, storage_path: str = "./mlops_data/metrics"):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileBasedMetricsTracking initialized with storage at {storage_path}")
    
    async def log_metrics_batch(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log batch of metrics."""
        metrics_file = self._storage_path / f"{run_id}_metrics.json"
        
        # Load existing metrics
        existing_metrics = []
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing metrics: {e}")
        
        # Add new metrics entry
        metrics_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": step,
            "metrics": metrics
        }
        existing_metrics.append(metrics_entry)
        
        # Save updated metrics
        try:
            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
            logger.info(f"Logged metrics for run {run_id}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    async def get_metrics_history(
        self,
        run_id: str,
        metric_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics history for a run."""
        metrics_file = self._storage_path / f"{run_id}_metrics.json"
        
        if not metrics_file.exists():
            return []
        
        try:
            with open(metrics_file, 'r') as f:
                metrics_history = json.load(f)
            
            # Filter by metric names if specified
            if metric_names:
                filtered_history = []
                for entry in metrics_history:
                    filtered_metrics = {
                        name: value for name, value in entry["metrics"].items()
                        if name in metric_names
                    }
                    if filtered_metrics:
                        filtered_entry = entry.copy()
                        filtered_entry["metrics"] = filtered_metrics
                        filtered_history.append(filtered_entry)
                return filtered_history
            
            return metrics_history
        except Exception as e:
            logger.error(f"Failed to load metrics history: {e}")
            return []


class FileBasedExperimentSearch(ExperimentSearchPort):
    """File-based implementation for experiment search."""
    
    def __init__(self, experiment_port: FileBasedExperimentTracking):
        self._experiment_port = experiment_port
        logger.info("FileBasedExperimentSearch initialized")
    
    async def search_experiments(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[ExperimentInfo]:
        """Search experiments by query."""
        # Get all experiments
        all_experiments = await self._experiment_port.list_experiments(limit=1000)
        
        # Apply text search
        query_lower = query.lower()
        matching_experiments = []
        
        for exp in all_experiments:
            match_score = 0
            
            # Search in name
            if query_lower in exp.name.lower():
                match_score += 10
            
            # Search in description
            if exp.description and query_lower in exp.description.lower():
                match_score += 5
            
            # Search in tags
            for tag in exp.tags:
                if query_lower in tag.lower():
                    match_score += 3
            
            # Search in metadata
            for key, value in exp.metadata.items():
                if query_lower in str(key).lower() or query_lower in str(value).lower():
                    match_score += 2
            
            if match_score > 0:
                matching_experiments.append((exp, match_score))
        
        # Sort by match score
        matching_experiments.sort(key=lambda x: x[1], reverse=True)
        
        # Apply filters
        if filters:
            filtered_experiments = []
            for exp, score in matching_experiments:
                include = True
                
                if "created_by" in filters and exp.created_by != filters["created_by"]:
                    include = False
                
                if "tags" in filters:
                    required_tags = filters["tags"]
                    if not any(tag in exp.tags for tag in required_tags):
                        include = False
                
                if include:
                    filtered_experiments.append((exp, score))
            
            matching_experiments = filtered_experiments
        
        # Apply limit and return experiments
        return [exp for exp, _ in matching_experiments[:limit]]