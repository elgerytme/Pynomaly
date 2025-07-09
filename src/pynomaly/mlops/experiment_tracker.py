#!/usr/bin/env python3
"""
MLOps Experiment Tracker for Pynomaly.
This module provides comprehensive experiment tracking, metrics logging, and model comparison.
"""

import json
import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """Experiment run data structure."""

    run_id: str
    experiment_id: str
    name: str
    status: str
    start_time: datetime
    end_time: datetime | None
    duration_seconds: float | None
    parameters: dict[str, Any]
    metrics: dict[str, float]
    artifacts: dict[str, str]
    tags: list[str]
    notes: str
    author: str
    git_commit: str | None
    environment: dict[str, str]


@dataclass
class MetricLog:
    """Metric log entry."""

    timestamp: datetime
    step: int
    metric_name: str
    value: float
    run_id: str


class ExperimentTracker:
    """MLOps Experiment Tracker for tracking ML experiments."""

    def __init__(self, tracking_path: str = "mlops/experiments"):
        """Initialize experiment tracker."""
        self.tracking_path = Path(tracking_path)
        self.runs_path = self.tracking_path / "runs"
        self.metrics_path = self.tracking_path / "metrics"
        self.artifacts_path = self.tracking_path / "artifacts"
        self.logs_path = self.tracking_path / "logs"

        # Create directory structure
        for path in [
            self.runs_path,
            self.metrics_path,
            self.artifacts_path,
            self.logs_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Current run tracking
        self.current_run: ExperimentRun | None = None
        self.current_run_metrics: list[MetricLog] = []

        # Experiment index
        self.experiment_index_path = self.tracking_path / "experiment_index.json"
        self.experiment_index = self._load_experiment_index()

        logger.info(f"Experiment tracker initialized at {self.tracking_path}")

    def _load_experiment_index(self) -> dict[str, Any]:
        """Load experiment index from file."""
        if self.experiment_index_path.exists():
            try:
                with open(self.experiment_index_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load experiment index: {e}")

        return {
            "experiments": {},
            "runs": {},
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
        }

    def _save_experiment_index(self):
        """Save experiment index to file."""
        try:
            self.experiment_index["updated_at"] = datetime.now().isoformat()
            with open(self.experiment_index_path, "w") as f:
                json.dump(self.experiment_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save experiment index: {e}")

    def create_experiment(
        self, name: str, description: str = "", tags: list[str] = None
    ) -> str:
        """Create a new experiment."""
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

        experiment_data = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "runs": [],
            "status": "active",
        }

        # Save experiment
        experiment_path = self.tracking_path / f"experiment_{experiment_id}.json"
        with open(experiment_path, "w") as f:
            json.dump(experiment_data, f, indent=2, default=str)

        # Update index
        self.experiment_index["experiments"][experiment_id] = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "runs_count": 0,
            "status": "active",
        }
        self._save_experiment_index()

        logger.info(f"✅ Created experiment: {name} ({experiment_id})")
        return experiment_id

    @contextmanager
    def start_run(
        self,
        experiment_id: str,
        run_name: str,
        parameters: dict[str, Any] = None,
        tags: list[str] = None,
        notes: str = "",
    ):
        """Context manager for experiment runs."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"

        # Create run
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            name=run_name,
            status="running",
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            parameters=parameters or {},
            metrics={},
            artifacts={},
            tags=tags or [],
            notes=notes,
            author=os.getenv("USER", "unknown"),
            git_commit=self._get_git_commit(),
            environment=self._get_environment_info(),
        )

        self.current_run = run
        self.current_run_metrics = []

        logger.info(f"🚀 Starting run: {run_name} ({run_id})")

        try:
            yield run_id
        except Exception as e:
            logger.error(f"Run failed: {e}")
            run.status = "failed"
            raise
        finally:
            # End run
            run.end_time = datetime.now()
            run.duration_seconds = (run.end_time - run.start_time).total_seconds()
            run.status = "completed" if run.status == "running" else run.status

            # Save run
            self._save_run(run)

            # Reset current run
            self.current_run = None
            self.current_run_metrics = []

            logger.info(f"✅ Completed run: {run_name} ({run_id})")

    def log_parameter(self, name: str, value: Any):
        """Log a parameter for the current run."""
        if not self.current_run:
            raise ValueError("No active run. Use start_run() context manager.")

        self.current_run.parameters[name] = value
        logger.debug(f"Logged parameter: {name} = {value}")

    def log_metric(self, name: str, value: float, step: int = 0):
        """Log a metric for the current run."""
        if not self.current_run:
            raise ValueError("No active run. Use start_run() context manager.")

        # Update current run metrics
        self.current_run.metrics[name] = value

        # Log metric with timestamp
        metric_log = MetricLog(
            timestamp=datetime.now(),
            step=step,
            metric_name=name,
            value=value,
            run_id=self.current_run.run_id,
        )
        self.current_run_metrics.append(metric_log)

        logger.debug(f"Logged metric: {name} = {value} (step {step})")

    def log_metrics(self, metrics: dict[str, float], step: int = 0):
        """Log multiple metrics for the current run."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_artifact(self, name: str, artifact_path: str):
        """Log an artifact for the current run."""
        if not self.current_run:
            raise ValueError("No active run. Use start_run() context manager.")

        # Copy artifact to artifacts directory
        artifact_dir = self.artifacts_path / self.current_run.run_id
        artifact_dir.mkdir(exist_ok=True)

        # Copy file
        import shutil

        src_path = Path(artifact_path)
        dst_path = artifact_dir / src_path.name
        shutil.copy2(src_path, dst_path)

        # Log artifact
        self.current_run.artifacts[name] = str(dst_path)
        logger.debug(f"Logged artifact: {name} -> {dst_path}")

    def log_model_performance(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None
    ):
        """Log comprehensive model performance metrics."""
        if not self.current_run:
            raise ValueError("No active run. Use start_run() context manager.")

        try:
            # Calculate metrics
            metrics = {}

            # Classification metrics
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["precision"] = precision_score(y_true, y_pred, average="binary")
                metrics["recall"] = recall_score(y_true, y_pred, average="binary")
                metrics["f1_score"] = f1_score(y_true, y_pred, average="binary")

                if y_prob is not None:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            else:  # Multi-class classification
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["precision_macro"] = precision_score(
                    y_true, y_pred, average="macro"
                )
                metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")
                metrics["f1_score_macro"] = f1_score(y_true, y_pred, average="macro")
                metrics["precision_micro"] = precision_score(
                    y_true, y_pred, average="micro"
                )
                metrics["recall_micro"] = recall_score(y_true, y_pred, average="micro")
                metrics["f1_score_micro"] = f1_score(y_true, y_pred, average="micro")

            # Log all metrics
            self.log_metrics(metrics)

            # Generate and save confusion matrix
            self._save_confusion_matrix(y_true, y_pred)

            # Generate and save classification report
            self._save_classification_report(y_true, y_pred)

            logger.info("✅ Logged model performance metrics")

        except Exception as e:
            logger.error(f"Failed to log model performance: {e}")

    def _save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Save confusion matrix visualization."""
        try:
            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            # Save plot
            plot_path = (
                self.artifacts_path / self.current_run.run_id / "confusion_matrix.png"
            )
            plot_path.parent.mkdir(exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Log as artifact
            self.current_run.artifacts["confusion_matrix"] = str(plot_path)

        except Exception as e:
            logger.error(f"Failed to save confusion matrix: {e}")

    def _save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Save classification report."""
        try:
            report = classification_report(y_true, y_pred, output_dict=True)

            # Save report as JSON
            report_path = (
                self.artifacts_path
                / self.current_run.run_id
                / "classification_report.json"
            )
            report_path.parent.mkdir(exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            # Log as artifact
            self.current_run.artifacts["classification_report"] = str(report_path)

        except Exception as e:
            logger.error(f"Failed to save classification report: {e}")

    def _save_run(self, run: ExperimentRun):
        """Save run data to file."""
        try:
            # Save run data
            run_path = self.runs_path / f"{run.run_id}.json"
            with open(run_path, "w") as f:
                json.dump(asdict(run), f, indent=2, default=str)

            # Save metrics
            if self.current_run_metrics:
                metrics_path = self.metrics_path / f"{run.run_id}_metrics.json"
                metrics_data = [asdict(m) for m in self.current_run_metrics]
                with open(metrics_path, "w") as f:
                    json.dump(metrics_data, f, indent=2, default=str)

            # Update experiment index
            self.experiment_index["runs"][run.run_id] = {
                "experiment_id": run.experiment_id,
                "name": run.name,
                "status": run.status,
                "start_time": run.start_time.isoformat(),
                "duration_seconds": run.duration_seconds,
                "author": run.author,
            }

            # Update experiment runs count
            if run.experiment_id in self.experiment_index["experiments"]:
                self.experiment_index["experiments"][run.experiment_id][
                    "runs_count"
                ] += 1

            self._save_experiment_index()

        except Exception as e:
            logger.error(f"Failed to save run: {e}")

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Get run by ID."""
        try:
            run_path = self.runs_path / f"{run_id}.json"
            if not run_path.exists():
                return None

            with open(run_path) as f:
                run_data = json.load(f)

            # Convert datetime strings back to datetime objects
            run_data["start_time"] = datetime.fromisoformat(run_data["start_time"])
            if run_data["end_time"]:
                run_data["end_time"] = datetime.fromisoformat(run_data["end_time"])

            return ExperimentRun(**run_data)

        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None

    def list_runs(
        self, experiment_id: str = None, status: str = None
    ) -> list[ExperimentRun]:
        """List runs with optional filtering."""
        runs = []

        try:
            for run_id in self.experiment_index["runs"]:
                run = self.get_run(run_id)
                if run is None:
                    continue

                # Apply filters
                if experiment_id and run.experiment_id != experiment_id:
                    continue
                if status and run.status != status:
                    continue

                runs.append(run)

            # Sort by start time (newest first)
            runs.sort(key=lambda r: r.start_time, reverse=True)

        except Exception as e:
            logger.error(f"Failed to list runs: {e}")

        return runs

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        """Compare multiple runs."""
        logger.info(f"Comparing runs: {run_ids}")

        try:
            comparison_data = {
                "runs": [],
                "parameter_comparison": {},
                "metric_comparison": {},
                "best_run": None,
                "timestamp": datetime.now().isoformat(),
            }

            runs = []
            for run_id in run_ids:
                run = self.get_run(run_id)
                if run:
                    runs.append(run)
                    comparison_data["runs"].append(
                        {
                            "run_id": run_id,
                            "name": run.name,
                            "experiment_id": run.experiment_id,
                            "status": run.status,
                            "duration_seconds": run.duration_seconds,
                            "parameters": run.parameters,
                            "metrics": run.metrics,
                            "start_time": run.start_time.isoformat(),
                            "author": run.author,
                        }
                    )

            if not runs:
                return comparison_data

            # Compare parameters
            all_params = set()
            for run in runs:
                all_params.update(run.parameters.keys())

            for param_name in all_params:
                param_values = []
                for run in runs:
                    if param_name in run.parameters:
                        param_values.append(
                            {"run_id": run.run_id, "value": run.parameters[param_name]}
                        )
                comparison_data["parameter_comparison"][param_name] = param_values

            # Compare metrics
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.metrics.keys())

            for metric_name in all_metrics:
                metric_values = []
                for run in runs:
                    if metric_name in run.metrics:
                        metric_values.append(
                            {"run_id": run.run_id, "value": run.metrics[metric_name]}
                        )

                if metric_values:
                    # Find best run for this metric
                    best_value = max(metric_values, key=lambda x: x["value"])
                    comparison_data["metric_comparison"][metric_name] = {
                        "values": metric_values,
                        "best_run": best_value["run_id"],
                        "best_value": best_value["value"],
                    }

            # Determine overall best run (based on first metric)
            if comparison_data["metric_comparison"]:
                first_metric = list(comparison_data["metric_comparison"].keys())[0]
                comparison_data["best_run"] = comparison_data["metric_comparison"][
                    first_metric
                ]["best_run"]

            logger.info(f"✅ Run comparison completed for {len(runs)} runs")
            return comparison_data

        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            raise

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any]:
        """Get experiment summary with statistics."""
        try:
            runs = self.list_runs(experiment_id=experiment_id)

            if not runs:
                return {
                    "experiment_id": experiment_id,
                    "total_runs": 0,
                    "status": "no_runs",
                    "message": "No runs found for this experiment",
                }

            # Calculate statistics
            completed_runs = [r for r in runs if r.status == "completed"]
            failed_runs = [r for r in runs if r.status == "failed"]

            # Metrics statistics
            metric_stats = {}
            if completed_runs:
                all_metrics = set()
                for run in completed_runs:
                    all_metrics.update(run.metrics.keys())

                for metric_name in all_metrics:
                    values = []
                    for run in completed_runs:
                        if metric_name in run.metrics:
                            values.append(run.metrics[metric_name])

                    if values:
                        metric_stats[metric_name] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values),
                            "count": len(values),
                        }

            # Find best run
            best_run = None
            if completed_runs and metric_stats:
                # Use first metric for ranking
                first_metric = list(metric_stats.keys())[0]
                best_run = max(
                    completed_runs, key=lambda r: r.metrics.get(first_metric, 0)
                )

            summary = {
                "experiment_id": experiment_id,
                "total_runs": len(runs),
                "completed_runs": len(completed_runs),
                "failed_runs": len(failed_runs),
                "success_rate": len(completed_runs) / len(runs) * 100,
                "avg_duration_seconds": np.mean(
                    [r.duration_seconds for r in completed_runs if r.duration_seconds]
                ),
                "metric_statistics": metric_stats,
                "best_run": {
                    "run_id": best_run.run_id,
                    "name": best_run.name,
                    "metrics": best_run.metrics,
                    "start_time": best_run.start_time.isoformat(),
                }
                if best_run
                else None,
                "latest_run": {
                    "run_id": runs[0].run_id,
                    "name": runs[0].name,
                    "status": runs[0].status,
                    "start_time": runs[0].start_time.isoformat(),
                },
                "timestamp": datetime.now().isoformat(),
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            raise

    def delete_run(self, run_id: str) -> bool:
        """Delete a run."""
        try:
            # Delete run file
            run_path = self.runs_path / f"{run_id}.json"
            if run_path.exists():
                run_path.unlink()

            # Delete metrics file
            metrics_path = self.metrics_path / f"{run_id}_metrics.json"
            if metrics_path.exists():
                metrics_path.unlink()

            # Delete artifacts directory
            artifacts_dir = self.artifacts_path / run_id
            if artifacts_dir.exists():
                import shutil

                shutil.rmtree(artifacts_dir)

            # Update index
            if run_id in self.experiment_index["runs"]:
                del self.experiment_index["runs"][run_id]
                self._save_experiment_index()

            logger.info(f"✅ Deleted run: {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete run {run_id}: {e}")
            return False

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_environment_info(self) -> dict[str, str]:
        """Get environment information."""
        try:
            import platform
            import sys

            return {
                "python_version": sys.version,
                "platform": platform.platform(),
                "hostname": platform.node(),
                "user": os.getenv("USER", "unknown"),
                "working_directory": os.getcwd(),
            }
        except Exception as e:
            logger.warning(f"Failed to get environment info: {e}")
            return {}

    def export_experiment_data(self, experiment_id: str, format: str = "json") -> str:
        """Export experiment data in various formats."""
        try:
            runs = self.list_runs(experiment_id=experiment_id)
            summary = self.get_experiment_summary(experiment_id)

            export_data = {
                "experiment_id": experiment_id,
                "summary": summary,
                "runs": [asdict(run) for run in runs],
                "exported_at": datetime.now().isoformat(),
                "format": format,
            }

            # Export to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_{experiment_id}_{timestamp}.{format}"
            export_path = self.tracking_path / "exports" / filename
            export_path.parent.mkdir(exist_ok=True)

            if format == "json":
                with open(export_path, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format == "csv":
                # Create CSV with run data
                df = pd.DataFrame([asdict(run) for run in runs])
                df.to_csv(export_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"✅ Exported experiment data: {export_path}")
            return str(export_path)

        except Exception as e:
            logger.error(f"Failed to export experiment data: {e}")
            raise


# Global experiment tracker instance
experiment_tracker = ExperimentTracker()

# Make tracker available for import
__all__ = ["ExperimentTracker", "ExperimentRun", "MetricLog", "experiment_tracker"]
