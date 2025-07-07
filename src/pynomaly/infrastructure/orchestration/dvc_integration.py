"""DVC (Data Version Control) integration for data pipeline management."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DVCConfig(BaseModel):
    """DVC configuration."""
    
    repo_path: str = "."
    remote_name: str = "origin"
    remote_url: Optional[str] = None
    cache_dir: Optional[str] = None
    auto_stage: bool = True
    auto_push: bool = False


class DVCPipelineStage(BaseModel):
    """DVC pipeline stage configuration."""
    
    name: str
    cmd: str
    deps: List[str] = []
    outs: List[str] = []
    params: List[str] = []
    metrics: List[str] = []
    plots: List[str] = []
    cache: bool = True
    always_changed: bool = False


class DVCPipeline(BaseModel):
    """DVC pipeline configuration."""
    
    stages: Dict[str, DVCPipelineStage]
    vars: Dict[str, Any] = {}


class DVCIntegration:
    """DVC integration for data versioning and pipeline management."""
    
    def __init__(self, config: DVCConfig):
        self.config = config
        self.repo_path = Path(config.repo_path)
        self.dvc_dir = self.repo_path / ".dvc"
        self.pipeline_file = self.repo_path / "dvc.yaml"
        self.params_file = self.repo_path / "params.yaml"
        self.metrics_file = self.repo_path / "metrics.yaml"
    
    async def initialize_dvc_repo(self) -> bool:
        """Initialize DVC repository."""
        try:
            if not self.dvc_dir.exists():
                result = await self._run_dvc_command(["init"])
                if result.returncode != 0:
                    logger.error(f"Failed to initialize DVC repo: {result.stderr}")
                    return False
                
                logger.info("DVC repository initialized")
            
            # Configure remote if specified
            if self.config.remote_url:
                await self.add_remote(self.config.remote_name, self.config.remote_url)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DVC repo: {e}")
            return False
    
    async def add_remote(self, name: str, url: str, default: bool = True) -> bool:
        """Add DVC remote storage."""
        try:
            # Add remote
            result = await self._run_dvc_command(["remote", "add", name, url])
            if result.returncode != 0:
                logger.error(f"Failed to add DVC remote: {result.stderr}")
                return False
            
            # Set as default if specified
            if default:
                result = await self._run_dvc_command(["remote", "default", name])
                if result.returncode != 0:
                    logger.warning(f"Failed to set default remote: {result.stderr}")
            
            logger.info(f"Added DVC remote: {name} -> {url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add DVC remote: {e}")
            return False
    
    async def add_data(self, data_path: str, commit_message: Optional[str] = None) -> bool:
        """Add data file to DVC tracking."""
        try:
            # Add to DVC
            result = await self._run_dvc_command(["add", data_path])
            if result.returncode != 0:
                logger.error(f"Failed to add data to DVC: {result.stderr}")
                return False
            
            # Stage .dvc file for git
            if self.config.auto_stage:
                await self._run_git_command(["add", f"{data_path}.dvc", ".gitignore"])
                
                if commit_message:
                    await self._run_git_command(["commit", "-m", commit_message])
            
            logger.info(f"Added data to DVC: {data_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add data to DVC: {e}")
            return False
    
    async def push_data(self, remote_name: Optional[str] = None) -> bool:
        """Push data to DVC remote."""
        try:
            cmd = ["push"]
            if remote_name:
                cmd.extend(["-r", remote_name])
            
            result = await self._run_dvc_command(cmd)
            if result.returncode != 0:
                logger.error(f"Failed to push data to DVC remote: {result.stderr}")
                return False
            
            logger.info("Pushed data to DVC remote")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push data to DVC: {e}")
            return False
    
    async def pull_data(self, remote_name: Optional[str] = None) -> bool:
        """Pull data from DVC remote."""
        try:
            cmd = ["pull"]
            if remote_name:
                cmd.extend(["-r", remote_name])
            
            result = await self._run_dvc_command(cmd)
            if result.returncode != 0:
                logger.error(f"Failed to pull data from DVC remote: {result.stderr}")
                return False
            
            logger.info("Pulled data from DVC remote")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull data from DVC: {e}")
            return False
    
    async def create_pipeline(self, pipeline: DVCPipeline) -> bool:
        """Create DVC pipeline from configuration."""
        try:
            # Convert pipeline to DVC format
            dvc_pipeline = {
                "stages": {}
            }
            
            # Add variables if any
            if pipeline.vars:
                dvc_pipeline["vars"] = pipeline.vars
            
            # Convert stages
            for stage_name, stage_config in pipeline.stages.items():
                dvc_stage = {
                    "cmd": stage_config.cmd
                }
                
                if stage_config.deps:
                    dvc_stage["deps"] = stage_config.deps
                
                if stage_config.outs:
                    dvc_stage["outs"] = stage_config.outs
                
                if stage_config.params:
                    dvc_stage["params"] = stage_config.params
                
                if stage_config.metrics:
                    dvc_stage["metrics"] = stage_config.metrics
                
                if stage_config.plots:
                    dvc_stage["plots"] = stage_config.plots
                
                if not stage_config.cache:
                    if "outs" in dvc_stage:
                        dvc_stage["outs"] = [{"path": out, "cache": False} for out in dvc_stage["outs"]]
                
                if stage_config.always_changed:
                    dvc_stage["always_changed"] = True
                
                dvc_pipeline["stages"][stage_name] = dvc_stage
            
            # Write pipeline file
            with open(self.pipeline_file, 'w') as f:
                yaml.dump(dvc_pipeline, f, default_flow_style=False)
            
            logger.info(f"Created DVC pipeline with {len(pipeline.stages)} stages")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create DVC pipeline: {e}")
            return False
    
    async def run_pipeline(
        self,
        stage: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False
    ) -> bool:
        """Run DVC pipeline."""
        try:
            cmd = ["repro"]
            
            if stage:
                cmd.append(stage)
            
            if force:
                cmd.append("--force")
            
            if dry_run:
                cmd.append("--dry")
            
            result = await self._run_dvc_command(cmd)
            if result.returncode != 0:
                logger.error(f"DVC pipeline execution failed: {result.stderr}")
                return False
            
            logger.info(f"DVC pipeline executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to run DVC pipeline: {e}")
            return False
    
    async def update_params(self, params: Dict[str, Any]) -> bool:
        """Update pipeline parameters."""
        try:
            # Load existing params if file exists
            existing_params = {}
            if self.params_file.exists():
                with open(self.params_file, 'r') as f:
                    existing_params = yaml.safe_load(f) or {}
            
            # Update with new params
            existing_params.update(params)
            
            # Write updated params
            with open(self.params_file, 'w') as f:
                yaml.dump(existing_params, f, default_flow_style=False)
            
            logger.info(f"Updated {len(params)} parameters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            return False
    
    async def get_params(self) -> Dict[str, Any]:
        """Get current pipeline parameters."""
        try:
            if not self.params_file.exists():
                return {}
            
            with open(self.params_file, 'r') as f:
                params = yaml.safe_load(f) or {}
            
            return params
            
        except Exception as e:
            logger.error(f"Failed to get parameters: {e}")
            return {}
    
    async def log_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Log metrics to DVC."""
        try:
            # Load existing metrics if file exists
            existing_metrics = {}
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    existing_metrics = yaml.safe_load(f) or {}
            
            # Update with new metrics
            existing_metrics.update(metrics)
            
            # Write updated metrics
            with open(self.metrics_file, 'w') as f:
                yaml.dump(existing_metrics, f, default_flow_style=False)
            
            logger.info(f"Logged {len(metrics)} metrics")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        try:
            if not self.metrics_file.exists():
                return {}
            
            with open(self.metrics_file, 'r') as f:
                metrics = yaml.safe_load(f) or {}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    async def show_metrics(self, all_branches: bool = False) -> Dict[str, Any]:
        """Show metrics comparison across branches/commits."""
        try:
            cmd = ["metrics", "show", "--json"]
            if all_branches:
                cmd.append("--all-branches")
            
            result = await self._run_dvc_command(cmd)
            if result.returncode != 0:
                logger.error(f"Failed to show metrics: {result.stderr}")
                return {}
            
            return json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Failed to show metrics: {e}")
            return {}
    
    async def diff_metrics(self, revision: Optional[str] = None) -> Dict[str, Any]:
        """Show metrics diff between revisions."""
        try:
            cmd = ["metrics", "diff", "--json"]
            if revision:
                cmd.append(revision)
            
            result = await self._run_dvc_command(cmd)
            if result.returncode != 0:
                logger.error(f"Failed to diff metrics: {result.stderr}")
                return {}
            
            return json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Failed to diff metrics: {e}")
            return {}
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get DVC pipeline status."""
        try:
            result = await self._run_dvc_command(["status", "--json"])
            if result.returncode != 0:
                logger.error(f"Failed to get pipeline status: {result.stderr}")
                return {}
            
            return json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {}
    
    async def create_experiment(
        self,
        name: str,
        params: Dict[str, Any],
        queue: bool = False
    ) -> Optional[str]:
        """Create DVC experiment."""
        try:
            # Update parameters for experiment
            await self.update_params(params)
            
            cmd = ["exp", "run"]
            if name:
                cmd.extend(["--name", name])
            if queue:
                cmd.append("--queue")
            
            result = await self._run_dvc_command(cmd)
            if result.returncode != 0:
                logger.error(f"Failed to create experiment: {result.stderr}")
                return None
            
            # Extract experiment ID from output
            # This is a simplified extraction - real implementation would parse properly
            experiment_id = name or "latest"
            
            logger.info(f"Created DVC experiment: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return None
    
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List DVC experiments."""
        try:
            result = await self._run_dvc_command(["exp", "show", "--json"])
            if result.returncode != 0:
                logger.error(f"Failed to list experiments: {result.stderr}")
                return []
            
            experiments_data = json.loads(result.stdout)
            
            # Parse experiments data - simplified version
            experiments = []
            for branch, data in experiments_data.items():
                if isinstance(data, list):
                    for exp in data:
                        experiments.append({
                            "branch": branch,
                            "experiment": exp.get("rev", "unknown"),
                            "timestamp": exp.get("timestamp"),
                            "params": exp.get("params", {}),
                            "metrics": exp.get("metrics", {})
                        })
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    async def _run_dvc_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run DVC command."""
        full_cmd = ["dvc"] + cmd
        
        result = subprocess.run(
            full_cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        logger.debug(f"DVC command: {' '.join(full_cmd)}")
        logger.debug(f"DVC stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"DVC stderr: {result.stderr}")
        
        return result
    
    async def _run_git_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run Git command."""
        full_cmd = ["git"] + cmd
        
        result = subprocess.run(
            full_cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        logger.debug(f"Git command: {' '.join(full_cmd)}")
        
        return result


# DVC pipeline templates for anomaly detection
class DVCAnomalyDetectionPipeline:
    """Pre-configured DVC pipelines for anomaly detection workflows."""
    
    @staticmethod
    def create_basic_pipeline(
        data_source: str,
        model_type: str = "isolation_forest"
    ) -> DVCPipeline:
        """Create basic anomaly detection pipeline."""
        
        stages = {
            "data_ingestion": DVCPipelineStage(
                name="data_ingestion",
                cmd="python -m pynomaly.infrastructure.orchestration.task_operators data_ingestion_task",
                deps=[],
                outs=["data/raw/input.csv"],
                params=["data_ingestion.connection_id", "data_ingestion.table_name"]
            ),
            
            "data_preprocessing": DVCPipelineStage(
                name="data_preprocessing",
                cmd="python -m pynomaly.infrastructure.orchestration.task_operators data_preprocessing_task",
                deps=["data/raw/input.csv"],
                outs=["data/processed/preprocessed.csv"],
                params=["preprocessing.steps"]
            ),
            
            "feature_engineering": DVCPipelineStage(
                name="feature_engineering",
                cmd="python -m pynomaly.infrastructure.orchestration.task_operators feature_engineering_task",
                deps=["data/processed/preprocessed.csv"],
                outs=["data/features/features.csv"],
                params=["feature_engineering.configs"]
            ),
            
            "model_training": DVCPipelineStage(
                name="model_training",
                cmd="python -m pynomaly.infrastructure.orchestration.task_operators model_training_task",
                deps=["data/features/features.csv"],
                outs=["models/detector.pkl"],
                params=["model.detector_name", "model.parameters"],
                metrics=["metrics/training.yaml"]
            ),
            
            "anomaly_detection": DVCPipelineStage(
                name="anomaly_detection",
                cmd="python -m pynomaly.infrastructure.orchestration.task_operators anomaly_detection_task",
                deps=["data/features/features.csv", "models/detector.pkl"],
                outs=["results/anomalies.csv"],
                metrics=["metrics/detection.yaml"],
                plots=["plots/anomaly_scores.json"]
            ),
            
            "validation": DVCPipelineStage(
                name="validation",
                cmd="python -m pynomaly.infrastructure.orchestration.task_operators validation_task",
                deps=["results/anomalies.csv"],
                outs=["reports/validation.json"],
                metrics=["metrics/validation.yaml"]
            )
        }
        
        return DVCPipeline(
            stages=stages,
            vars={
                "data_source": data_source,
                "model_type": model_type
            }
        )
    
    @staticmethod
    def create_ensemble_pipeline() -> DVCPipeline:
        """Create ensemble anomaly detection pipeline."""
        
        stages = {
            "data_ingestion": DVCPipelineStage(
                name="data_ingestion",
                cmd="python -m pynomaly.infrastructure.orchestration.task_operators data_ingestion_task",
                deps=[],
                outs=["data/raw/input.csv"],
                params=["data_ingestion"]
            ),
            
            "data_split": DVCPipelineStage(
                name="data_split",
                cmd="python scripts/split_data.py",
                deps=["data/raw/input.csv"],
                outs=["data/split/train.csv", "data/split/test.csv"],
                params=["split.test_size", "split.random_state"]
            ),
            
            "train_isolation_forest": DVCPipelineStage(
                name="train_isolation_forest",
                cmd="python scripts/train_model.py isolation_forest",
                deps=["data/split/train.csv"],
                outs=["models/isolation_forest.pkl"],
                params=["models.isolation_forest"],
                metrics=["metrics/isolation_forest.yaml"]
            ),
            
            "train_local_outlier_factor": DVCPipelineStage(
                name="train_local_outlier_factor",
                cmd="python scripts/train_model.py local_outlier_factor",
                deps=["data/split/train.csv"],
                outs=["models/local_outlier_factor.pkl"],
                params=["models.local_outlier_factor"],
                metrics=["metrics/local_outlier_factor.yaml"]
            ),
            
            "train_one_class_svm": DVCPipelineStage(
                name="train_one_class_svm",
                cmd="python scripts/train_model.py one_class_svm",
                deps=["data/split/train.csv"],
                outs=["models/one_class_svm.pkl"],
                params=["models.one_class_svm"],
                metrics=["metrics/one_class_svm.yaml"]
            ),
            
            "ensemble_prediction": DVCPipelineStage(
                name="ensemble_prediction",
                cmd="python scripts/ensemble_predict.py",
                deps=[
                    "data/split/test.csv",
                    "models/isolation_forest.pkl",
                    "models/local_outlier_factor.pkl",
                    "models/one_class_svm.pkl"
                ],
                outs=["results/ensemble_predictions.csv"],
                params=["ensemble.strategy", "ensemble.weights"],
                metrics=["metrics/ensemble.yaml"],
                plots=["plots/ensemble_comparison.json"]
            ),
            
            "evaluation": DVCPipelineStage(
                name="evaluation",
                cmd="python scripts/evaluate.py",
                deps=["results/ensemble_predictions.csv"],
                outs=["reports/evaluation.html"],
                metrics=["metrics/final_evaluation.yaml"],
                plots=["plots/final_results.json"]
            )
        }
        
        return DVCPipeline(stages=stages)


# Integration with workflow orchestrator
class DVCWorkflowIntegration:
    """Integration between DVC and workflow orchestration."""
    
    def __init__(self, dvc_config: DVCConfig):
        self.dvc = DVCIntegration(dvc_config)
    
    async def create_workflow_pipeline(
        self,
        workflow_config: Dict[str, Any]
    ) -> bool:
        """Create DVC pipeline from workflow configuration."""
        try:
            # Convert workflow tasks to DVC stages
            dvc_stages = {}
            
            for task in workflow_config.get("tasks", []):
                stage = DVCPipelineStage(
                    name=task["task_id"],
                    cmd=f"python -m pynomaly.infrastructure.orchestration.task_operators {task['function_name']}",
                    deps=task.get("input_files", []),
                    outs=task.get("output_files", []),
                    params=task.get("param_files", [])
                )
                dvc_stages[task["task_id"]] = stage
            
            pipeline = DVCPipeline(stages=dvc_stages)
            return await self.dvc.create_pipeline(pipeline)
            
        except Exception as e:
            logger.error(f"Failed to create DVC pipeline from workflow: {e}")
            return False
    
    async def sync_workflow_params(
        self,
        workflow_params: Dict[str, Any]
    ) -> bool:
        """Sync workflow parameters with DVC."""
        return await self.dvc.update_params(workflow_params)
    
    async def track_workflow_artifacts(
        self,
        artifacts: List[str]
    ) -> bool:
        """Track workflow artifacts with DVC."""
        try:
            for artifact in artifacts:
                success = await self.dvc.add_data(
                    artifact,
                    f"Track workflow artifact: {artifact}"
                )
                if not success:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to track workflow artifacts: {e}")
            return False