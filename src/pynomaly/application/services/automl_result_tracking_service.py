"""
AutoML Result Tracking Service

Comprehensive tracking and management of AutoML experiment results,
model performance history, and comparative analysis.

This completes Issue #96: Complete AutoML Service Implementation
- AutoML result tracking
- Model comparison and ranking
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from pynomaly.domain.abstractions import BaseEntity
from pynomaly.infrastructure.logging.structured_logger import StructuredLogger


class ExperimentStatus(str):
    """Experiment status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    
    # Anomaly detection specific metrics
    contamination_rate: Optional[float] = None
    anomaly_score_mean: Optional[float] = None
    anomaly_score_std: Optional[float] = None
    
    # Cross-validation metrics
    cv_scores: List[float] = Field(default_factory=list)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Additional metrics
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    model_size_bytes: Optional[int] = None


class HyperparameterResult(BaseModel):
    """Hyperparameter optimization result."""
    parameter_name: str
    parameter_value: Any
    parameter_type: str
    search_space: Optional[Dict[str, Any]] = None
    importance_score: Optional[float] = None


class AutoMLExperiment(BaseEntity):
    """AutoML experiment tracking entity."""
    
    experiment_name: str
    dataset_name: str
    dataset_id: UUID
    
    # Experiment configuration
    algorithm_name: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_engineering_steps: List[str] = Field(default_factory=list)
    
    # Execution details
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_duration: Optional[float] = None
    
    # Results
    metrics: ModelMetrics = Field(default_factory=ModelMetrics)
    hyperparameter_results: List[HyperparameterResult] = Field(default_factory=list)
    
    # Model artifacts
    model_path: Optional[str] = None
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    
    # Metadata
    experiment_notes: str = ""
    tags: List[str] = Field(default_factory=list)
    pipeline_id: Optional[UUID] = None
    
    # Comparison metrics
    rank: Optional[int] = None
    pareto_optimal: bool = False


class ExperimentComparison(BaseModel):
    """Comparison between multiple experiments."""
    
    comparison_id: UUID = Field(default_factory=uuid4)
    experiment_ids: List[UUID]
    comparison_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Comparison criteria
    primary_metric: str = "f1_score"
    secondary_metrics: List[str] = Field(default_factory=list)
    
    # Results
    ranked_experiments: List[UUID] = Field(default_factory=list)
    statistical_significance: Dict[str, Any] = Field(default_factory=dict)
    pareto_frontier: List[UUID] = Field(default_factory=list)
    
    # Insights
    best_performing: Optional[UUID] = None
    most_efficient: Optional[UUID] = None
    most_stable: Optional[UUID] = None
    recommendations: List[str] = Field(default_factory=list)


class AutoMLResultTrackingService:
    """
    Service for tracking and analyzing AutoML experiment results.
    
    Provides comprehensive experiment tracking, model comparison,
    and performance analysis capabilities.
    """
    
    def __init__(
        self,
        tracking_db_path: str = "automl_tracking.db",
        logger: Optional[StructuredLogger] = None
    ):
        """Initialize the result tracking service."""
        self.db_path = Path(tracking_db_path)
        self.logger = logger or StructuredLogger(__name__)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for experiment tracking."""
        with sqlite3.connect(self.db_path) as conn:
            # Experiments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    algorithm_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    execution_duration REAL,
                    hyperparameters TEXT,
                    metrics TEXT,
                    model_path TEXT,
                    feature_importance TEXT,
                    experiment_notes TEXT,
                    tags TEXT,
                    pipeline_id TEXT,
                    rank INTEGER,
                    pareto_optimal BOOLEAN,
                    created_at TEXT,
                    updated_at TEXT,
                    version INTEGER
                )
            """)
            
            # Hyperparameter results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hyperparameter_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    parameter_name TEXT NOT NULL,
                    parameter_value TEXT NOT NULL,
                    parameter_type TEXT NOT NULL,
                    search_space TEXT,
                    importance_score REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Comparisons table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_comparisons (
                    id TEXT PRIMARY KEY,
                    experiment_ids TEXT NOT NULL,
                    comparison_date TEXT NOT NULL,
                    primary_metric TEXT NOT NULL,
                    secondary_metrics TEXT,
                    ranked_experiments TEXT,
                    statistical_significance TEXT,
                    pareto_frontier TEXT,
                    best_performing TEXT,
                    most_efficient TEXT,
                    most_stable TEXT,
                    recommendations TEXT
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_algorithm ON experiments(algorithm_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hyperparams_experiment ON hyperparameter_results(experiment_id)")
    
    async def create_experiment(
        self,
        experiment_name: str,
        dataset_name: str,
        dataset_id: UUID,
        algorithm_name: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        pipeline_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None
    ) -> AutoMLExperiment:
        """Create a new experiment for tracking."""
        experiment = AutoMLExperiment(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            algorithm_name=algorithm_name,
            hyperparameters=hyperparameters or {},
            pipeline_id=pipeline_id,
            tags=tags or []
        )
        
        await self._save_experiment(experiment)
        
        self.logger.info(
            "Created new AutoML experiment",
            experiment_id=str(experiment.id),
            experiment_name=experiment_name,
            algorithm=algorithm_name
        )
        
        return experiment
    
    async def start_experiment(self, experiment_id: UUID) -> AutoMLExperiment:
        """Mark experiment as started."""
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.utcnow()
        experiment.mark_as_updated()
        
        await self._save_experiment(experiment)
        
        self.logger.info(
            "Started AutoML experiment",
            experiment_id=str(experiment_id)
        )
        
        return experiment
    
    async def complete_experiment(
        self,
        experiment_id: UUID,
        metrics: ModelMetrics,
        model_path: Optional[str] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        hyperparameter_results: Optional[List[HyperparameterResult]] = None
    ) -> AutoMLExperiment:
        """Mark experiment as completed with results."""
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = datetime.utcnow()
        experiment.metrics = metrics
        experiment.model_path = model_path
        experiment.feature_importance = feature_importance or {}
        experiment.hyperparameter_results = hyperparameter_results or []
        
        if experiment.start_time:
            experiment.execution_duration = (
                experiment.end_time - experiment.start_time
            ).total_seconds()
        
        experiment.mark_as_updated()
        await self._save_experiment(experiment)
        
        self.logger.info(
            "Completed AutoML experiment",
            experiment_id=str(experiment_id),
            f1_score=metrics.f1_score,
            execution_duration=experiment.execution_duration
        )
        
        return experiment
    
    async def fail_experiment(
        self,
        experiment_id: UUID,
        error_message: str
    ) -> AutoMLExperiment:
        """Mark experiment as failed."""
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment.status = ExperimentStatus.FAILED
        experiment.end_time = datetime.utcnow()
        experiment.experiment_notes = f"FAILED: {error_message}"
        experiment.mark_as_updated()
        
        await self._save_experiment(experiment)
        
        self.logger.error(
            "AutoML experiment failed",
            experiment_id=str(experiment_id),
            error=error_message
        )
        
        return experiment
    
    async def get_experiment(self, experiment_id: UUID) -> Optional[AutoMLExperiment]:
        """Get experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (str(experiment_id),)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return self._row_to_experiment(row)
    
    async def list_experiments(
        self,
        dataset_name: Optional[str] = None,
        algorithm_name: Optional[str] = None,
        status: Optional[ExperimentStatus] = None,
        limit: int = 100
    ) -> List[AutoMLExperiment]:
        """List experiments with optional filtering."""
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        
        if dataset_name:
            query += " AND dataset_name = ?"
            params.append(dataset_name)
        
        if algorithm_name:
            query += " AND algorithm_name = ?"
            params.append(algorithm_name)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            experiments = []
            for row in cursor.fetchall():
                experiments.append(self._row_to_experiment(row))
            
            return experiments
    
    async def compare_experiments(
        self,
        experiment_ids: List[UUID],
        primary_metric: str = "f1_score",
        secondary_metrics: Optional[List[str]] = None
    ) -> ExperimentComparison:
        """Compare multiple experiments and rank them."""
        experiments = []
        for exp_id in experiment_ids:
            exp = await self.get_experiment(exp_id)
            if exp and exp.status == ExperimentStatus.COMPLETED:
                experiments.append(exp)
        
        if len(experiments) < 2:
            raise ValueError("Need at least 2 completed experiments for comparison")
        
        comparison = ExperimentComparison(
            experiment_ids=experiment_ids,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics or []
        )
        
        # Rank experiments by primary metric
        experiments_with_scores = []
        for exp in experiments:
            score = getattr(exp.metrics, primary_metric, None)
            if score is not None:
                experiments_with_scores.append((exp, score))
        
        # Sort by score (descending - higher is better)
        experiments_with_scores.sort(key=lambda x: x[1], reverse=True)
        comparison.ranked_experiments = [exp.id for exp, _ in experiments_with_scores]
        
        # Find best performing
        if experiments_with_scores:
            comparison.best_performing = experiments_with_scores[0][0].id
        
        # Find most efficient (best performance / execution time ratio)
        efficiency_scores = []
        for exp, score in experiments_with_scores:
            if exp.execution_duration and exp.execution_duration > 0:
                efficiency = score / exp.execution_duration
                efficiency_scores.append((exp, efficiency))
        
        if efficiency_scores:
            efficiency_scores.sort(key=lambda x: x[1], reverse=True)
            comparison.most_efficient = efficiency_scores[0][0].id
        
        # Find most stable (lowest CV std)
        stability_scores = []
        for exp, _ in experiments_with_scores:
            if exp.metrics.cv_scores and len(exp.metrics.cv_scores) > 1:
                cv_std = np.std(exp.metrics.cv_scores)
                stability_scores.append((exp, cv_std))
        
        if stability_scores:
            stability_scores.sort(key=lambda x: x[1])  # Lower std is better
            comparison.most_stable = stability_scores[0][0].id
        
        # Generate recommendations
        comparison.recommendations = self._generate_comparison_recommendations(
            experiments_with_scores, comparison
        )
        
        # Calculate Pareto frontier for multi-objective optimization
        if secondary_metrics:
            comparison.pareto_frontier = self._calculate_pareto_frontier(
                experiments_with_scores, primary_metric, secondary_metrics
            )
        
        # Save comparison
        await self._save_comparison(comparison)
        
        self.logger.info(
            "Completed experiment comparison",
            comparison_id=str(comparison.comparison_id),
            experiments_count=len(experiments),
            best_performing=str(comparison.best_performing)
        )
        
        return comparison
    
    async def get_leaderboard(
        self,
        dataset_name: Optional[str] = None,
        metric: str = "f1_score",
        limit: int = 10
    ) -> List[AutoMLExperiment]:
        """Get top performing experiments as a leaderboard."""
        experiments = await self.list_experiments(
            dataset_name=dataset_name,
            status=ExperimentStatus.COMPLETED,
            limit=100
        )
        
        # Filter experiments that have the requested metric
        experiments_with_metric = []
        for exp in experiments:
            score = getattr(exp.metrics, metric, None)
            if score is not None:
                experiments_with_metric.append((exp, score))
        
        # Sort by metric score (descending)
        experiments_with_metric.sort(key=lambda x: x[1], reverse=True)
        
        # Update ranks
        leaderboard = []
        for rank, (exp, score) in enumerate(experiments_with_metric[:limit], 1):
            exp.rank = rank
            leaderboard.append(exp)
        
        return leaderboard
    
    async def get_experiment_insights(
        self,
        dataset_name: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get insights and analytics from experiment history."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = """
            SELECT * FROM experiments 
            WHERE created_at > ? AND status = 'completed'
        """
        params = [cutoff_date.isoformat()]
        
        if dataset_name:
            query += " AND dataset_name = ?"
            params.append(dataset_name)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            experiments = []
            for row in cursor.fetchall():
                experiments.append(self._row_to_experiment(row))
        
        if not experiments:
            return {"message": "No completed experiments found in the specified period"}
        
        # Calculate insights
        insights = {
            "summary": {
                "total_experiments": len(experiments),
                "unique_algorithms": len(set(exp.algorithm_name for exp in experiments)),
                "average_execution_time": np.mean([
                    exp.execution_duration for exp in experiments 
                    if exp.execution_duration
                ]),
                "best_f1_score": max([
                    exp.metrics.f1_score for exp in experiments 
                    if exp.metrics.f1_score
                ], default=0)
            },
            "algorithm_performance": {},
            "trends": {},
            "recommendations": []
        }
        
        # Algorithm performance analysis
        algorithm_stats = {}
        for exp in experiments:
            algo = exp.algorithm_name
            if algo not in algorithm_stats:
                algorithm_stats[algo] = {"experiments": [], "scores": []}
            
            algorithm_stats[algo]["experiments"].append(exp)
            if exp.metrics.f1_score:
                algorithm_stats[algo]["scores"].append(exp.metrics.f1_score)
        
        for algo, stats in algorithm_stats.items():
            if stats["scores"]:
                insights["algorithm_performance"][algo] = {
                    "count": len(stats["experiments"]),
                    "mean_f1": np.mean(stats["scores"]),
                    "std_f1": np.std(stats["scores"]),
                    "best_f1": max(stats["scores"])
                }
        
        # Generate recommendations based on insights
        best_algorithm = max(
            insights["algorithm_performance"].items(),
            key=lambda x: x[1]["mean_f1"]
        )[0] if insights["algorithm_performance"] else None
        
        if best_algorithm:
            insights["recommendations"].append(
                f"Consider using {best_algorithm} - shows best average performance"
            )
        
        return insights
    
    def _generate_comparison_recommendations(
        self,
        experiments_with_scores: List[tuple],
        comparison: ExperimentComparison
    ) -> List[str]:
        """Generate recommendations based on experiment comparison."""
        recommendations = []
        
        if len(experiments_with_scores) < 2:
            return recommendations
        
        best_exp, best_score = experiments_with_scores[0]
        second_exp, second_score = experiments_with_scores[1]
        
        # Performance gap analysis
        performance_gap = (best_score - second_score) / best_score * 100
        
        if performance_gap < 5:
            recommendations.append(
                "Top models show similar performance - consider ensemble methods"
            )
        elif performance_gap > 20:
            recommendations.append(
                f"Clear winner: {best_exp.algorithm_name} significantly outperforms others"
            )
        
        # Efficiency analysis
        if best_exp.execution_duration and second_exp.execution_duration:
            time_ratio = best_exp.execution_duration / second_exp.execution_duration
            if time_ratio > 2:
                recommendations.append(
                    f"Consider {second_exp.algorithm_name} for faster predictions"
                )
        
        # Stability analysis
        if best_exp.metrics.cv_std and best_exp.metrics.cv_std > 0.1:
            recommendations.append(
                "Best model shows high variance - consider model regularization"
            )
        
        return recommendations
    
    def _calculate_pareto_frontier(
        self,
        experiments_with_scores: List[tuple],
        primary_metric: str,
        secondary_metrics: List[str]
    ) -> List[UUID]:
        """Calculate Pareto frontier for multi-objective optimization."""
        # Simplified Pareto frontier calculation
        # In practice, this would be more sophisticated
        pareto_optimal = []
        
        for i, (exp1, score1) in enumerate(experiments_with_scores):
            is_dominated = False
            
            for j, (exp2, score2) in enumerate(experiments_with_scores):
                if i == j:
                    continue
                
                # Check if exp2 dominates exp1
                dominates = True
                for metric in [primary_metric] + secondary_metrics:
                    val1 = getattr(exp1.metrics, metric, 0) or 0
                    val2 = getattr(exp2.metrics, metric, 0) or 0
                    
                    if val1 > val2:  # exp1 is better in this metric
                        dominates = False
                        break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(exp1.id)
        
        return pareto_optimal
    
    async def _save_experiment(self, experiment: AutoMLExperiment) -> None:
        """Save experiment to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments (
                    id, experiment_name, dataset_name, dataset_id, algorithm_name,
                    status, start_time, end_time, execution_duration, hyperparameters,
                    metrics, model_path, feature_importance, experiment_notes,
                    tags, pipeline_id, rank, pareto_optimal, created_at, updated_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(experiment.id),
                experiment.experiment_name,
                experiment.dataset_name,
                str(experiment.dataset_id),
                experiment.algorithm_name,
                experiment.status,
                experiment.start_time.isoformat() if experiment.start_time else None,
                experiment.end_time.isoformat() if experiment.end_time else None,
                experiment.execution_duration,
                json.dumps(experiment.hyperparameters),
                experiment.metrics.model_dump_json(),
                experiment.model_path,
                json.dumps(experiment.feature_importance),
                experiment.experiment_notes,
                json.dumps(experiment.tags),
                str(experiment.pipeline_id) if experiment.pipeline_id else None,
                experiment.rank,
                experiment.pareto_optimal,
                experiment.created_at.isoformat(),
                experiment.updated_at.isoformat(),
                experiment.version
            ))
            
            # Save hyperparameter results
            conn.execute(
                "DELETE FROM hyperparameter_results WHERE experiment_id = ?",
                (str(experiment.id),)
            )
            
            for hp_result in experiment.hyperparameter_results:
                conn.execute("""
                    INSERT INTO hyperparameter_results (
                        experiment_id, parameter_name, parameter_value, parameter_type,
                        search_space, importance_score
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(experiment.id),
                    hp_result.parameter_name,
                    json.dumps(hp_result.parameter_value),
                    hp_result.parameter_type,
                    json.dumps(hp_result.search_space) if hp_result.search_space else None,
                    hp_result.importance_score
                ))
    
    async def _save_comparison(self, comparison: ExperimentComparison) -> None:
        """Save experiment comparison to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiment_comparisons (
                    id, experiment_ids, comparison_date, primary_metric, secondary_metrics,
                    ranked_experiments, statistical_significance, pareto_frontier,
                    best_performing, most_efficient, most_stable, recommendations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(comparison.comparison_id),
                json.dumps([str(id) for id in comparison.experiment_ids]),
                comparison.comparison_date.isoformat(),
                comparison.primary_metric,
                json.dumps(comparison.secondary_metrics),
                json.dumps([str(id) for id in comparison.ranked_experiments]),
                json.dumps(comparison.statistical_significance),
                json.dumps([str(id) for id in comparison.pareto_frontier]),
                str(comparison.best_performing) if comparison.best_performing else None,
                str(comparison.most_efficient) if comparison.most_efficient else None,
                str(comparison.most_stable) if comparison.most_stable else None,
                json.dumps(comparison.recommendations)
            ))
    
    def _row_to_experiment(self, row) -> AutoMLExperiment:
        """Convert database row to AutoMLExperiment."""
        # Database columns mapping
        (id, experiment_name, dataset_name, dataset_id, algorithm_name,
         status, start_time, end_time, execution_duration, hyperparameters,
         metrics, model_path, feature_importance, experiment_notes,
         tags, pipeline_id, rank, pareto_optimal, created_at, updated_at, version) = row
        
        return AutoMLExperiment(
            id=UUID(id),
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            dataset_id=UUID(dataset_id),
            algorithm_name=algorithm_name,
            status=status,
            start_time=datetime.fromisoformat(start_time) if start_time else None,
            end_time=datetime.fromisoformat(end_time) if end_time else None,
            execution_duration=execution_duration,
            hyperparameters=json.loads(hyperparameters) if hyperparameters else {},
            metrics=ModelMetrics.model_validate_json(metrics) if metrics else ModelMetrics(),
            model_path=model_path,
            feature_importance=json.loads(feature_importance) if feature_importance else {},
            experiment_notes=experiment_notes or "",
            tags=json.loads(tags) if tags else [],
            pipeline_id=UUID(pipeline_id) if pipeline_id else None,
            rank=rank,
            pareto_optimal=bool(pareto_optimal),
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            version=version
        )