"""Application service for experiment tracking."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pandas as pd


class ExperimentTrackingService:
    """Service for tracking ML experiments and results."""
    
    def __init__(self, tracking_path: Path):
        """Initialize experiment tracking service.
        
        Args:
            tracking_path: Base path for experiment storage
        """
        self.tracking_path = tracking_path
        self.tracking_path.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.tracking_path / "experiments.json"
        self._load_experiments()
    
    def _load_experiments(self) -> None:
        """Load existing experiments from file."""
        if self.experiments_file.exists():
            with open(self.experiments_file, "r") as f:
                self.experiments = json.load(f)
        else:
            self.experiments = {}
    
    def _save_experiments(self) -> None:
        """Save experiments to file."""
        with open(self.experiments_file, "w") as f:
            json.dump(self.experiments, f, indent=2)
    
    async def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a new experiment.
        
        Args:
            name: Experiment name
            description: Optional description
            tags: Optional tags for categorization
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid4())
        
        experiment = {
            "id": experiment_id,
            "name": name,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat(),
            "runs": []
        }
        
        self.experiments[experiment_id] = experiment
        self._save_experiments()
        
        # Create experiment directory
        exp_dir = self.tracking_path / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        return experiment_id
    
    async def log_run(
        self,
        experiment_id: str,
        detector_name: str,
        dataset_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None
    ) -> str:
        """Log a run within an experiment.
        
        Args:
            experiment_id: ID of the experiment
            detector_name: Name of the detector used
            dataset_name: Name of the dataset used
            parameters: Hyperparameters used
            metrics: Performance metrics
            artifacts: Optional paths to artifacts
            
        Returns:
            Run ID
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        run_id = str(uuid4())
        
        run = {
            "id": run_id,
            "detector_name": detector_name,
            "dataset_name": dataset_name,
            "parameters": parameters,
            "metrics": metrics,
            "artifacts": artifacts or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.experiments[experiment_id]["runs"].append(run)
        self._save_experiments()
        
        # Save run details
        run_dir = self.tracking_path / experiment_id / run_id
        run_dir.mkdir(exist_ok=True)
        
        with open(run_dir / "run.json", "w") as f:
            json.dump(run, f, indent=2)
        
        return run_id
    
    async def compare_runs(
        self,
        experiment_id: str,
        run_ids: Optional[List[str]] = None,
        metric: str = "f1"
    ) -> pd.DataFrame:
        """Compare runs within an experiment.
        
        Args:
            experiment_id: ID of the experiment
            run_ids: Specific runs to compare (None = all)
            metric: Metric to sort by
            
        Returns:
            DataFrame with run comparisons
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        runs = self.experiments[experiment_id]["runs"]
        
        if run_ids:
            runs = [r for r in runs if r["id"] in run_ids]
        
        # Create comparison dataframe
        comparison_data = []
        for run in runs:
            row = {
                "run_id": run["id"],
                "detector": run["detector_name"],
                "dataset": run["dataset_name"],
                "timestamp": run["timestamp"],
                **run["metrics"]
            }
            # Add key parameters
            for param, value in run["parameters"].items():
                row[f"param_{param}"] = value
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by metric if available
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        return df
    
    async def get_best_run(
        self,
        experiment_id: str,
        metric: str = "f1",
        higher_is_better: bool = True
    ) -> Dict[str, Any]:
        """Get the best run from an experiment.
        
        Args:
            experiment_id: ID of the experiment
            metric: Metric to optimize
            higher_is_better: Whether higher values are better
            
        Returns:
            Best run details
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        runs = self.experiments[experiment_id]["runs"]
        
        if not runs:
            raise ValueError("No runs found in experiment")
        
        # Find best run
        best_run = None
        best_value = float('-inf') if higher_is_better else float('inf')
        
        for run in runs:
            if metric in run["metrics"]:
                value = run["metrics"][metric]
                if higher_is_better and value > best_value:
                    best_value = value
                    best_run = run
                elif not higher_is_better and value < best_value:
                    best_value = value
                    best_run = run
        
        if best_run is None:
            raise ValueError(f"No runs found with metric {metric}")
        
        return best_run
    
    async def log_artifact(
        self,
        experiment_id: str,
        run_id: str,
        artifact_name: str,
        artifact_path: str
    ) -> None:
        """Log an artifact for a run.
        
        Args:
            experiment_id: ID of the experiment
            run_id: ID of the run
            artifact_name: Name of the artifact
            artifact_path: Path to the artifact
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Find run
        run = None
        for r in self.experiments[experiment_id]["runs"]:
            if r["id"] == run_id:
                run = r
                break
        
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        
        # Update artifacts
        run["artifacts"][artifact_name] = artifact_path
        self._save_experiments()
    
    async def create_leaderboard(
        self,
        experiment_ids: Optional[List[str]] = None,
        metric: str = "f1"
    ) -> pd.DataFrame:
        """Create a leaderboard across experiments.
        
        Args:
            experiment_ids: Experiments to include (None = all)
            metric: Metric to rank by
            
        Returns:
            Leaderboard DataFrame
        """
        if experiment_ids is None:
            experiment_ids = list(self.experiments.keys())
        
        leaderboard_data = []
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                continue
            
            experiment = self.experiments[exp_id]
            
            for run in experiment["runs"]:
                if metric in run["metrics"]:
                    leaderboard_data.append({
                        "experiment": experiment["name"],
                        "run_id": run["id"],
                        "detector": run["detector_name"],
                        "dataset": run["dataset_name"],
                        metric: run["metrics"][metric],
                        "timestamp": run["timestamp"]
                    })
        
        df = pd.DataFrame(leaderboard_data)
        
        if not df.empty:
            df = df.sort_values(metric, ascending=False)
            df["rank"] = range(1, len(df) + 1)
        
        return df
    
    async def export_experiment(
        self,
        experiment_id: str,
        export_path: Path
    ) -> None:
        """Export experiment data and artifacts.
        
        Args:
            experiment_id: ID of experiment to export
            export_path: Path to export to
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export experiment metadata
        experiment = self.experiments[experiment_id]
        with open(export_path / "experiment.json", "w") as f:
            json.dump(experiment, f, indent=2)
        
        # Export comparison
        comparison_df = await self.compare_runs(experiment_id)
        comparison_df.to_csv(export_path / "comparison.csv", index=False)
        
        # Create summary report
        report = self._generate_experiment_report(experiment)
        with open(export_path / "report.md", "w") as f:
            f.write(report)
    
    def _generate_experiment_report(self, experiment: Dict[str, Any]) -> str:
        """Generate markdown report for experiment."""
        report = f"""# Experiment: {experiment['name']}

**ID**: {experiment['id']}  
**Created**: {experiment['created_at']}  
**Description**: {experiment.get('description', 'N/A')}  
**Tags**: {', '.join(experiment.get('tags', []))}

## Summary

Total runs: {len(experiment['runs'])}

## Best Performing Runs

| Metric | Best Value | Detector | Dataset | Run ID |
|--------|------------|----------|---------|--------|
"""
        
        # Find best runs for common metrics
        metrics_to_check = ["f1", "auc_roc", "precision", "recall"]
        
        for metric in metrics_to_check:
            best_value = 0
            best_run = None
            
            for run in experiment["runs"]:
                if metric in run["metrics"] and run["metrics"][metric] > best_value:
                    best_value = run["metrics"][metric]
                    best_run = run
            
            if best_run:
                report += f"| {metric} | {best_value:.4f} | "
                report += f"{best_run['detector_name']} | "
                report += f"{best_run['dataset_name']} | "
                report += f"{best_run['id'][:8]} |\n"
        
        report += "\n## All Runs\n\n"
        
        for i, run in enumerate(experiment["runs"], 1):
            report += f"### Run {i}: {run['detector_name']}\n"
            report += f"- **ID**: {run['id']}\n"
            report += f"- **Dataset**: {run['dataset_name']}\n"
            report += f"- **Timestamp**: {run['timestamp']}\n"
            report += f"- **Metrics**: {json.dumps(run['metrics'], indent=2)}\n"
            report += f"- **Parameters**: {json.dumps(run['parameters'], indent=2)}\n\n"
        
        return report