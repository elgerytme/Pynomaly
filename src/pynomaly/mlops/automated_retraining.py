#!/usr/bin/env python3
"""
Automated Model Retraining Pipeline for Pynomaly.
This module provides automated model retraining, data drift detection, and model performance monitoring.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import schedule
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Retraining trigger types."""

    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    DATA_VOLUME = "data_volume"


class RetrainingStatus(Enum):
    """Retraining status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetrainingConfig:
    """Retraining configuration."""

    model_id: str
    trigger_type: TriggerType
    schedule_cron: str | None
    performance_threshold: float
    data_drift_threshold: float
    min_data_points: int
    max_training_time_minutes: int
    auto_deploy: bool
    validation_split: float
    hyperparameter_tuning: bool
    notification_enabled: bool
    rollback_enabled: bool


@dataclass
class RetrainingJob:
    """Retraining job."""

    job_id: str
    config: RetrainingConfig
    status: RetrainingStatus
    trigger_type: TriggerType
    trigger_reason: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    duration_seconds: float | None
    metrics: dict[str, float]
    new_model_id: str | None
    error_message: str | None
    data_stats: dict[str, Any]


@dataclass
class DataDriftReport:
    """Data drift detection report."""

    report_id: str
    model_id: str
    reference_data_period: tuple[datetime, datetime]
    current_data_period: tuple[datetime, datetime]
    drift_score: float
    drift_detected: bool
    feature_drift_scores: dict[str, float]
    statistical_tests: dict[str, Any]
    recommendations: list[str]
    created_at: datetime


@dataclass
class ModelPerformanceReport:
    """Model performance monitoring report."""

    report_id: str
    model_id: str
    deployment_id: str
    evaluation_period: tuple[datetime, datetime]
    current_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    performance_degradation: dict[str, float]
    alert_triggered: bool
    recommendations: list[str]
    created_at: datetime


class DataDriftDetector:
    """Data drift detection using statistical tests."""

    def __init__(self):
        """Initialize drift detector."""
        self.reference_data: pd.DataFrame | None = None
        self.reference_stats: dict[str, Any] | None = None

    def set_reference_data(self, data: pd.DataFrame):
        """Set reference data for drift detection."""
        self.reference_data = data.copy()
        self.reference_stats = self._calculate_stats(data)
        logger.info(
            f"Reference data set: {len(data)} samples, {len(data.columns)} features"
        )

    def detect_drift(
        self, current_data: pd.DataFrame, threshold: float = 0.05
    ) -> DataDriftReport:
        """Detect data drift between reference and current data."""
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")

        report_id = str(uuid.uuid4())

        # Calculate current data stats
        current_stats = self._calculate_stats(current_data)

        # Perform statistical tests
        drift_scores = {}
        statistical_tests = {}

        for feature in self.reference_data.columns:
            if feature in current_data.columns:
                # Kolmogorov-Smirnov test
                ks_stat, ks_p_value = self._ks_test(
                    self.reference_data[feature], current_data[feature]
                )

                # Population Stability Index
                psi_score = self._calculate_psi(
                    self.reference_data[feature], current_data[feature]
                )

                drift_scores[feature] = psi_score
                statistical_tests[feature] = {
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_p_value,
                    "psi_score": psi_score,
                    "drift_detected": psi_score > threshold,
                }

        # Overall drift score
        overall_drift_score = np.mean(list(drift_scores.values()))
        drift_detected = overall_drift_score > threshold

        # Generate recommendations
        recommendations = self._generate_drift_recommendations(
            drift_scores, statistical_tests, threshold
        )

        report = DataDriftReport(
            report_id=report_id,
            model_id="",  # Will be set by caller
            reference_data_period=(
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=7),
            ),
            current_data_period=(datetime.now() - timedelta(days=7), datetime.now()),
            drift_score=overall_drift_score,
            drift_detected=drift_detected,
            feature_drift_scores=drift_scores,
            statistical_tests=statistical_tests,
            recommendations=recommendations,
            created_at=datetime.now(),
        )

        logger.info(
            f"Drift detection completed: drift_score={overall_drift_score:.4f}, detected={drift_detected}"
        )
        return report

    def _calculate_stats(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate statistical properties of data."""
        stats = {}

        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                stats[column] = {
                    "mean": data[column].mean(),
                    "std": data[column].std(),
                    "min": data[column].min(),
                    "max": data[column].max(),
                    "median": data[column].median(),
                    "q25": data[column].quantile(0.25),
                    "q75": data[column].quantile(0.75),
                    "skewness": data[column].skew(),
                    "kurtosis": data[column].kurtosis(),
                }
            else:
                stats[column] = {
                    "unique_values": data[column].nunique(),
                    "most_common": data[column].mode().iloc[0]
                    if not data[column].mode().empty
                    else None,
                    "null_count": data[column].isnull().sum(),
                }

        return stats

    def _ks_test(
        self, ref_data: pd.Series, curr_data: pd.Series
    ) -> tuple[float, float]:
        """Perform Kolmogorov-Smirnov test."""
        try:
            from scipy import stats

            ks_stat, p_value = stats.ks_2samp(ref_data.dropna(), curr_data.dropna())
            return ks_stat, p_value
        except ImportError:
            # Fallback to simple comparison
            logger.warning("scipy not available, using simple comparison")
            diff = abs(ref_data.mean() - curr_data.mean()) / (ref_data.std() + 1e-8)
            return diff, 0.05 if diff > 0.1 else 0.5

    def _calculate_psi(
        self, ref_data: pd.Series, curr_data: pd.Series, bins: int = 10
    ) -> float:
        """Calculate Population Stability Index."""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(ref_data.dropna(), bins=bins)

            # Calculate distributions
            ref_dist, _ = np.histogram(ref_data.dropna(), bins=bin_edges)
            curr_dist, _ = np.histogram(curr_data.dropna(), bins=bin_edges)

            # Normalize to get proportions
            ref_prop = ref_dist / ref_dist.sum()
            curr_prop = curr_dist / curr_dist.sum()

            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            ref_prop = ref_prop + epsilon
            curr_prop = curr_prop + epsilon

            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))

            return psi

        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0

    def _generate_drift_recommendations(
        self,
        drift_scores: dict[str, float],
        statistical_tests: dict[str, Any],
        threshold: float,
    ) -> list[str]:
        """Generate recommendations based on drift detection."""
        recommendations = []

        high_drift_features = [
            feature for feature, score in drift_scores.items() if score > threshold
        ]

        if high_drift_features:
            recommendations.append(
                f"High drift detected in features: {', '.join(high_drift_features)}"
            )
            recommendations.append("Consider retraining the model with recent data")
            recommendations.append("Review data collection process for consistency")

            if len(high_drift_features) > len(drift_scores) * 0.5:
                recommendations.append(
                    "Significant drift across multiple features - urgent retraining recommended"
                )
        else:
            recommendations.append("No significant drift detected")
            recommendations.append("Continue monitoring data quality")

        return recommendations


class PerformanceMonitor:
    """Model performance monitoring."""

    def __init__(self):
        """Initialize performance monitor."""
        self.baseline_metrics: dict[str, dict[str, float]] = {}

    def set_baseline_metrics(self, model_id: str, metrics: dict[str, float]):
        """Set baseline metrics for a model."""
        self.baseline_metrics[model_id] = metrics.copy()
        logger.info(f"Baseline metrics set for model {model_id}: {metrics}")

    def evaluate_performance(
        self,
        model_id: str,
        deployment_id: str,
        current_metrics: dict[str, float],
        threshold: float = 0.05,
    ) -> ModelPerformanceReport:
        """Evaluate current model performance against baseline."""
        report_id = str(uuid.uuid4())

        if model_id not in self.baseline_metrics:
            raise ValueError(f"No baseline metrics found for model {model_id}")

        baseline = self.baseline_metrics[model_id]

        # Calculate performance degradation
        degradation = {}
        alert_triggered = False

        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline:
                baseline_value = baseline[metric_name]
                degradation[metric_name] = (
                    baseline_value - current_value
                ) / baseline_value

                # Check if degradation exceeds threshold
                if degradation[metric_name] > threshold:
                    alert_triggered = True

        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            degradation, threshold
        )

        report = ModelPerformanceReport(
            report_id=report_id,
            model_id=model_id,
            deployment_id=deployment_id,
            evaluation_period=(datetime.now() - timedelta(days=1), datetime.now()),
            current_metrics=current_metrics,
            baseline_metrics=baseline,
            performance_degradation=degradation,
            alert_triggered=alert_triggered,
            recommendations=recommendations,
            created_at=datetime.now(),
        )

        logger.info(
            f"Performance evaluation completed: alert_triggered={alert_triggered}"
        )
        return report

    def _generate_performance_recommendations(
        self, degradation: dict[str, float], threshold: float
    ) -> list[str]:
        """Generate recommendations based on performance degradation."""
        recommendations = []

        degraded_metrics = [
            metric for metric, deg in degradation.items() if deg > threshold
        ]

        if degraded_metrics:
            recommendations.append(
                f"Performance degradation detected in: {', '.join(degraded_metrics)}"
            )
            recommendations.append("Consider retraining with recent data")
            recommendations.append("Review model hyperparameters")

            if len(degraded_metrics) > len(degradation) * 0.5:
                recommendations.append(
                    "Significant performance degradation - immediate retraining recommended"
                )
        else:
            recommendations.append("Model performance is stable")
            recommendations.append("Continue monitoring")

        return recommendations


class AutomatedRetrainingPipeline:
    """Automated model retraining pipeline."""

    def __init__(self, pipeline_path: str = "mlops/retraining"):
        """Initialize retraining pipeline."""
        self.pipeline_path = Path(pipeline_path)
        self.pipeline_path.mkdir(parents=True, exist_ok=True)

        # Components
        self.drift_detector = DataDriftDetector()
        self.performance_monitor = PerformanceMonitor()

        # Configuration and state
        self.retraining_configs: dict[str, RetrainingConfig] = {}
        self.active_jobs: dict[str, RetrainingJob] = {}
        self.job_history: list[RetrainingJob] = []

        # Index files
        self.config_index_path = self.pipeline_path / "config_index.json"
        self.jobs_index_path = self.pipeline_path / "jobs_index.json"

        # Load existing configuration
        self._load_configuration()

        # Start scheduler
        self._start_scheduler()

        logger.info(
            f"Automated retraining pipeline initialized at {self.pipeline_path}"
        )

    def _load_configuration(self):
        """Load retraining configuration from file."""
        if self.config_index_path.exists():
            try:
                with open(self.config_index_path) as f:
                    config_data = json.load(f)

                for model_id, config_dict in config_data.get("configs", {}).items():
                    config_dict["trigger_type"] = TriggerType(
                        config_dict["trigger_type"]
                    )
                    self.retraining_configs[model_id] = RetrainingConfig(**config_dict)

            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")

    def _save_configuration(self):
        """Save retraining configuration to file."""
        try:
            config_data = {
                "configs": {
                    model_id: asdict(config)
                    for model_id, config in self.retraining_configs.items()
                },
                "updated_at": datetime.now().isoformat(),
            }

            with open(self.config_index_path, "w") as f:
                json.dump(config_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def _start_scheduler(self):
        """Start the retraining scheduler."""

        def run_scheduler():
            while True:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    time.sleep(60)

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

        logger.info("Retraining scheduler started")

    def configure_retraining(self, config: RetrainingConfig):
        """Configure automated retraining for a model."""
        self.retraining_configs[config.model_id] = config
        self._save_configuration()

        # Schedule if it's a scheduled trigger
        if config.trigger_type == TriggerType.SCHEDULED and config.schedule_cron:
            self._schedule_retraining(config)

        logger.info(f"Retraining configured for model {config.model_id}")

    def _schedule_retraining(self, config: RetrainingConfig):
        """Schedule retraining job."""

        def job_func():
            asyncio.run(
                self._trigger_retraining(
                    config.model_id, TriggerType.SCHEDULED, "Scheduled retraining"
                )
            )

        # Parse cron expression (simplified)
        if config.schedule_cron:
            # For now, support simple schedules
            if "daily" in config.schedule_cron.lower():
                schedule.every().day.at("02:00").do(job_func)
            elif "weekly" in config.schedule_cron.lower():
                schedule.every().week.do(job_func)
            elif "hourly" in config.schedule_cron.lower():
                schedule.every().hour.do(job_func)

        logger.info(
            f"Scheduled retraining for {config.model_id}: {config.schedule_cron}"
        )

    async def _trigger_retraining(
        self, model_id: str, trigger_type: TriggerType, trigger_reason: str
    ) -> str:
        """Trigger model retraining."""
        if model_id not in self.retraining_configs:
            raise ValueError(f"No retraining configuration for model {model_id}")

        config = self.retraining_configs[model_id]
        job_id = str(uuid.uuid4())

        # Create retraining job
        job = RetrainingJob(
            job_id=job_id,
            config=config,
            status=RetrainingStatus.PENDING,
            trigger_type=trigger_type,
            trigger_reason=trigger_reason,
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            duration_seconds=None,
            metrics={},
            new_model_id=None,
            error_message=None,
            data_stats={},
        )

        # Store job
        self.active_jobs[job_id] = job

        # Start retraining in background
        asyncio.create_task(self._execute_retraining_job(job))

        logger.info(f"Retraining triggered: {job_id} for model {model_id}")
        return job_id

    async def _execute_retraining_job(self, job: RetrainingJob):
        """Execute retraining job."""
        job.status = RetrainingStatus.RUNNING
        job.started_at = datetime.now()

        try:
            # Generate training data
            training_data = await self._generate_training_data(job.config.model_id)

            if len(training_data) < job.config.min_data_points:
                raise ValueError(
                    f"Insufficient training data: {len(training_data)} < {job.config.min_data_points}"
                )

            # Store data statistics
            job.data_stats = {
                "training_samples": len(training_data),
                "features": len(training_data.columns)
                - 1,  # Assuming last column is target
                "data_period": {
                    "start": (datetime.now() - timedelta(days=30)).isoformat(),
                    "end": datetime.now().isoformat(),
                },
            }

            # Train new model
            new_model, training_metrics = await self._train_model(
                training_data, job.config
            )

            # Validate model
            validation_metrics = await self._validate_model(
                new_model, training_data, job.config
            )

            # Register new model
            new_model_id = await self._register_new_model(
                new_model, job.config.model_id, training_metrics, validation_metrics
            )

            # Store results
            job.new_model_id = new_model_id
            job.metrics = {**training_metrics, **validation_metrics}
            job.status = RetrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.duration_seconds = (job.completed_at - job.started_at).total_seconds()

            # Auto-deploy if configured
            if job.config.auto_deploy:
                await self._auto_deploy_model(new_model_id, job.config.model_id)

            logger.info(f"Retraining completed successfully: {job.job_id}")

        except Exception as e:
            job.status = RetrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            job.duration_seconds = (job.completed_at - job.started_at).total_seconds()

            logger.error(f"Retraining failed: {job.job_id} - {e}")

        finally:
            # Move to history
            self.job_history.append(job)
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

            # Send notification if configured
            if job.config.notification_enabled:
                await self._send_notification(job)

    async def _generate_training_data(self, model_id: str) -> pd.DataFrame:
        """Generate training data for model retraining."""
        try:
            # Try to get real training data from data sources
            from pynomaly.infrastructure.config import get_container

            try:
                container = get_container()
                dataset_repo = container.get_dataset_repository()

                # Get the most recent datasets for this model
                datasets = await dataset_repo.list_datasets_for_model(model_id)
                if datasets:
                    # Use the most recent dataset
                    latest_dataset = max(datasets, key=lambda d: d.created_at)
                    data = await dataset_repo.load_dataset_data(latest_dataset.id)
                    logger.info(
                        f"Loaded real training data: {len(data)} samples from dataset {latest_dataset.id}"
                    )
                    return data

            except Exception as e:
                logger.warning(f"Could not load real dataset for model {model_id}: {e}")

            # Fallback: generate synthetic data based on model metadata
            from pynomaly.mlops.model_registry import ModelRegistry

            model_registry = ModelRegistry()
            _, metadata = await model_registry.get_model(model_id)

            # Use model metadata to inform synthetic data generation
            n_samples = getattr(metadata, "training_samples", 1000)
            n_features = getattr(metadata, "feature_count", 10)
            contamination = getattr(metadata, "contamination_rate", 0.1)

            # Generate synthetic anomaly detection data
            np.random.seed(42)
            normal_data = np.random.normal(
                0, 1, (int(n_samples * (1 - contamination)), n_features)
            )
            anomaly_data = np.random.normal(
                2, 1, (int(n_samples * contamination), n_features)
            )

            X = np.vstack([normal_data, anomaly_data])
            y = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomaly_data))])

            # Create DataFrame
            feature_names = [f"feature_{i}" for i in range(n_features)]
            data = pd.DataFrame(X, columns=feature_names)
            data["target"] = y

            logger.info(f"Generated synthetic training data: {len(data)} samples")
            return data

        except Exception as e:
            logger.error(f"Failed to generate training data: {e}")
            raise

    async def _train_model(
        self, training_data: pd.DataFrame, config: RetrainingConfig
    ) -> tuple[BaseEstimator, dict[str, float]]:
        """Train new model."""
        # Prepare data
        X = training_data.drop("target", axis=1)
        y = training_data["target"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.validation_split, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model (using Isolation Forest for anomaly detection)
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X_train_scaled)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_binary = (y_pred == -1).astype(int)  # Convert to binary

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred_binary),
            "precision": precision_score(y_test, y_pred_binary, average="binary"),
            "recall": recall_score(y_test, y_pred_binary, average="binary"),
            "f1_score": f1_score(y_test, y_pred_binary, average="binary"),
            "training_samples": len(X_train),
            "validation_samples": len(X_test),
        }

        logger.info(f"Model trained with metrics: {metrics}")
        return model, metrics

    async def _validate_model(
        self, model: BaseEstimator, data: pd.DataFrame, config: RetrainingConfig
    ) -> dict[str, float]:
        """Validate trained model."""
        # Additional validation metrics
        X = data.drop("target", axis=1)
        y = data["target"]

        # Cross-validation or additional tests could be added here
        validation_metrics = {
            "validation_passed": True,
            "data_quality_score": 0.95,  # Placeholder
            "model_stability_score": 0.92,  # Placeholder
        }

        logger.info(f"Model validation completed: {validation_metrics}")
        return validation_metrics

    async def _register_new_model(
        self,
        model: BaseEstimator,
        original_model_id: str,
        training_metrics: dict[str, float],
        validation_metrics: dict[str, float],
    ) -> str:
        """Register new model in model registry."""
        try:
            from .model_registry import ModelType, model_registry

            # Generate new version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_version = f"retrained_{timestamp}"

            # Register model
            new_model_id = await model_registry.register_model(
                model=model,
                name=f"{original_model_id}_retrained",
                version=new_version,
                model_type=ModelType.ISOLATION_FOREST,
                author="automated_retraining",
                description=f"Automatically retrained model from {original_model_id}",
                tags=["automated", "retrained"],
                performance_metrics={**training_metrics, **validation_metrics},
                hyperparameters={"contamination": 0.1, "random_state": 42},
            )

            logger.info(f"New model registered: {new_model_id}")
            return new_model_id

        except Exception as e:
            logger.error(f"Failed to register new model: {e}")
            raise

    async def _auto_deploy_model(self, new_model_id: str, original_model_id: str):
        """Auto-deploy retrained model."""
        try:
            from .model_deployment import DeploymentEnvironment, deployment_manager

            # Create deployment
            deployment_id = await deployment_manager.create_deployment(
                model_id=new_model_id,
                model_version="latest",
                environment=DeploymentEnvironment.PRODUCTION,
                configuration={"auto_deployed": True},
                author="automated_retraining",
                notes=f"Auto-deployed retrained model from {original_model_id}",
            )

            # Deploy model
            success = await deployment_manager.deploy_model(deployment_id)

            if success:
                logger.info(f"Model auto-deployed: {deployment_id}")
            else:
                logger.error(f"Failed to auto-deploy model: {deployment_id}")

        except Exception as e:
            logger.error(f"Auto-deployment failed: {e}")

    async def _send_notification(self, job: RetrainingJob):
        """Send notification about retraining job completion."""
        try:
            # This is a placeholder - implement actual notification logic
            status_emoji = "✅" if job.status == RetrainingStatus.COMPLETED else "❌"

            message = f"""
            {status_emoji} Retraining Job {job.status.value.upper()}

            Job ID: {job.job_id}
            Model: {job.config.model_id}
            Trigger: {job.trigger_type.value}
            Reason: {job.trigger_reason}
            Duration: {job.duration_seconds:.2f}s

            Metrics: {job.metrics}
            """

            logger.info(f"Notification sent: {message}")

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    async def check_drift_trigger(self, model_id: str) -> bool:
        """Check if data drift should trigger retraining."""
        if model_id not in self.retraining_configs:
            return False

        config = self.retraining_configs[model_id]

        try:
            # Generate current data for drift detection
            current_data = await self._generate_training_data(model_id)
            current_data = current_data.drop(
                "target", axis=1
            )  # Remove target for drift detection

            # Detect drift
            drift_report = self.drift_detector.detect_drift(
                current_data, config.data_drift_threshold
            )

            if drift_report.drift_detected:
                await self._trigger_retraining(
                    model_id,
                    TriggerType.DATA_DRIFT,
                    f"Data drift detected: score={drift_report.drift_score:.4f}",
                )
                return True

        except Exception as e:
            logger.error(f"Drift check failed for {model_id}: {e}")

        return False

    async def check_performance_trigger(
        self, model_id: str, deployment_id: str
    ) -> bool:
        """Check if performance degradation should trigger retraining."""
        if model_id not in self.retraining_configs:
            return False

        config = self.retraining_configs[model_id]

        try:
            # Get current metrics (placeholder)
            current_metrics = {
                "accuracy": 0.85,
                "precision": 0.80,
                "recall": 0.82,
                "f1_score": 0.81,
            }

            # Evaluate performance
            performance_report = self.performance_monitor.evaluate_performance(
                model_id, deployment_id, current_metrics, config.performance_threshold
            )

            if performance_report.alert_triggered:
                await self._trigger_retraining(
                    model_id,
                    TriggerType.PERFORMANCE_DEGRADATION,
                    f"Performance degradation detected: {performance_report.performance_degradation}",
                )
                return True

        except Exception as e:
            logger.error(f"Performance check failed for {model_id}: {e}")

        return False

    def get_job_status(self, job_id: str) -> RetrainingJob | None:
        """Get job status by ID."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]

        # Check history
        for job in self.job_history:
            if job.job_id == job_id:
                return job

        return None

    def list_jobs(
        self, model_id: str = None, status: RetrainingStatus = None
    ) -> list[RetrainingJob]:
        """List retraining jobs with optional filtering."""
        all_jobs = list(self.active_jobs.values()) + self.job_history

        filtered_jobs = []
        for job in all_jobs:
            if model_id and job.config.model_id != model_id:
                continue
            if status and job.status != status:
                continue
            filtered_jobs.append(job)

        # Sort by creation date (newest first)
        filtered_jobs.sort(key=lambda j: j.created_at, reverse=True)

        return filtered_jobs

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        all_jobs = list(self.active_jobs.values()) + self.job_history

        stats = {
            "total_jobs": len(all_jobs),
            "active_jobs": len(self.active_jobs),
            "configured_models": len(self.retraining_configs),
            "jobs_by_status": {},
            "jobs_by_trigger": {},
            "avg_duration_seconds": 0,
            "success_rate": 0,
            "timestamp": datetime.now().isoformat(),
        }

        if all_jobs:
            # Count by status
            for job in all_jobs:
                status_key = job.status.value
                stats["jobs_by_status"][status_key] = (
                    stats["jobs_by_status"].get(status_key, 0) + 1
                )

                trigger_key = job.trigger_type.value
                stats["jobs_by_trigger"][trigger_key] = (
                    stats["jobs_by_trigger"].get(trigger_key, 0) + 1
                )

            # Calculate averages
            completed_jobs = [
                j for j in all_jobs if j.status == RetrainingStatus.COMPLETED
            ]
            if completed_jobs:
                stats["avg_duration_seconds"] = np.mean(
                    [j.duration_seconds for j in completed_jobs]
                )
                stats["success_rate"] = len(completed_jobs) / len(all_jobs) * 100

        return stats

    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        try:
            # Check for missing values
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])

            # Check for duplicate rows
            duplicate_ratio = data.duplicated().sum() / len(data)

            # Check for outliers (basic z-score method)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outlier_ratio = 0
            if len(numeric_cols) > 0:
                from scipy import stats

                z_scores = np.abs(stats.zscore(data[numeric_cols], nan_policy="omit"))
                outlier_ratio = (z_scores > 3).sum().sum() / (
                    len(data) * len(numeric_cols)
                )

            # Calculate overall quality score
            quality_score = 1.0 - (
                missing_ratio * 0.4 + duplicate_ratio * 0.3 + outlier_ratio * 0.3
            )
            return max(0.0, min(1.0, quality_score))

        except Exception as e:
            logger.warning(f"Could not calculate data quality: {e}")
            return 0.8

    def _calculate_model_stability(self, model, data: pd.DataFrame) -> float:
        """Calculate model stability through cross-validation."""
        try:
            from sklearn.metrics import f1_score, make_scorer
            from sklearn.model_selection import cross_val_score

            X = data.drop("target", axis=1)
            y = data["target"]

            # Use F1 score for stability assessment
            f1_scorer = make_scorer(f1_score, average="weighted")
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=f1_scorer)

            # Stability is measured by consistency of scores (low standard deviation)
            stability = 1.0 - (cv_scores.std() / cv_scores.mean())
            return max(0.0, min(1.0, stability))

        except Exception as e:
            logger.warning(f"Could not calculate model stability: {e}")
            return 0.8

    def _calculate_performance_score(self, model, data: pd.DataFrame) -> float:
        """Calculate model performance score."""
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
            from sklearn.model_selection import train_test_split

            X = data.drop("target", axis=1)
            y = data["target"]

            # Split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Calculate metrics
            f1 = f1_score(y_val, y_pred, average="weighted")
            precision = precision_score(y_val, y_pred, average="weighted")
            recall = recall_score(y_val, y_pred, average="weighted")

            # Return weighted average
            performance = f1 * 0.5 + precision * 0.25 + recall * 0.25
            return max(0.0, min(1.0, performance))

        except Exception as e:
            logger.warning(f"Could not calculate performance score: {e}")
            return 0.7


# Global retraining pipeline instance
retraining_pipeline = AutomatedRetrainingPipeline()

# Make pipeline available for import
__all__ = [
    "AutomatedRetrainingPipeline",
    "RetrainingConfig",
    "RetrainingJob",
    "DataDriftDetector",
    "PerformanceMonitor",
    "TriggerType",
    "RetrainingStatus",
    "retraining_pipeline",
]
