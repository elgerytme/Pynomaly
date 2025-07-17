"""
Intelligent Model Retraining Service

Advanced automated retraining system that intelligently decides when and how to retrain models
based on drift detection, performance metrics, and business rules.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from pynomaly_detection.application.services.advanced_ml_pipeline_service import (
    AdvancedMLPipelineService,
    OptimizationStrategy,
)
from pynomaly_detection.application.services.model_drift_detection_service import (
    DriftReport,
    DriftSeverity,
    DriftType,
    ModelDriftDetectionService,
)
from pynomaly_detection.infrastructure.logging.structured_logger import StructuredLogger
from pynomaly_detection.infrastructure.monitoring.metrics_service import MetricsService


class RetrainingTrigger(Enum):
    """Triggers for model retraining."""

    PERFORMANCE_DRIFT = "performance_drift"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    FEEDBACK_THRESHOLD = "feedback_threshold"
    EXTERNAL_EVENT = "external_event"


class RetrainingStrategy(Enum):
    """Retraining strategies."""

    FULL_RETRAIN = "full_retrain"
    INCREMENTAL_UPDATE = "incremental_update"
    ENSEMBLE_UPDATE = "ensemble_update"
    TRANSFER_LEARNING = "transfer_learning"
    ADAPTIVE_LEARNING = "adaptive_learning"


class RetrainingUrgency(Enum):
    """Urgency levels for retraining."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RetrainingConfig:
    """Configuration for intelligent retraining."""

    model_id: str
    retraining_strategy: RetrainingStrategy = RetrainingStrategy.FULL_RETRAIN
    max_retraining_frequency: timedelta = field(
        default_factory=lambda: timedelta(hours=24)
    )
    min_retraining_interval: timedelta = field(
        default_factory=lambda: timedelta(hours=6)
    )
    performance_threshold: float = 0.05  # 5% degradation triggers retraining
    data_drift_threshold: float = 0.1
    concept_drift_threshold: float = 0.15
    feedback_threshold: int = 100  # Number of negative feedback items
    auto_approve_low_risk: bool = True
    auto_approve_medium_risk: bool = False
    require_human_approval_high_risk: bool = True
    backup_model_count: int = 3
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    minimum_improvement: float = 0.02  # 2% minimum improvement to deploy
    rollback_conditions: dict[str, float] = field(
        default_factory=lambda: {
            "performance_drop": 0.1,
            "error_rate_increase": 0.05,
            "latency_increase": 2.0,
        }
    )


@dataclass
class RetrainingRequest:
    """Request for model retraining."""

    request_id: str
    model_id: str
    trigger: RetrainingTrigger
    urgency: RetrainingUrgency
    created_at: datetime = field(default_factory=datetime.now)
    trigger_data: dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    requested_by: str = "system"
    estimated_duration: timedelta | None = None
    estimated_cost: float | None = None


@dataclass
class RetrainingJob:
    """Active retraining job."""

    job_id: str
    request: RetrainingRequest
    config: RetrainingConfig
    status: str = "pending"  # pending, running, completed, failed, cancelled
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0
    current_stage: str = "initialization"
    experiment_id: str | None = None
    new_model_performance: dict[str, float] = field(default_factory=dict)
    old_model_performance: dict[str, float] = field(default_factory=dict)
    improvement_metrics: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None
    resource_usage: dict[str, float] = field(default_factory=dict)


class IntelligentRetrainingDecisionEngine:
    """Decision engine for intelligent retraining decisions."""

    def __init__(self):
        self.logger = StructuredLogger("retraining_decision_engine")

        # Decision weights
        self.decision_weights = {
            "performance_impact": 0.4,
            "drift_severity": 0.3,
            "business_impact": 0.2,
            "resource_availability": 0.1,
        }

        # Risk assessment factors
        self.risk_factors = {
            "model_criticality": 1.0,
            "data_volume": 0.3,
            "feature_complexity": 0.2,
            "deployment_complexity": 0.3,
            "rollback_difficulty": 0.2,
        }

    async def should_retrain(
        self,
        model_id: str,
        drift_report: DriftReport,
        performance_metrics: dict[str, float],
        config: RetrainingConfig,
        business_context: dict[str, Any] = None,
    ) -> tuple[bool, RetrainingUrgency, str]:
        """Determine if model should be retrained."""

        self.logger.info(f"Evaluating retraining decision for model {model_id}")

        # Calculate decision score
        decision_score = await self._calculate_decision_score(
            drift_report, performance_metrics, config, business_context
        )

        # Determine urgency
        urgency = self._calculate_urgency(
            drift_report, performance_metrics, decision_score
        )

        # Make decision
        should_retrain = decision_score > 0.5

        # Generate justification
        justification = self._generate_justification(
            drift_report, performance_metrics, decision_score, urgency
        )

        self.logger.info(
            f"Retraining decision for {model_id}: {should_retrain}, "
            f"Urgency: {urgency.value}, Score: {decision_score:.3f}"
        )

        return should_retrain, urgency, justification

    async def _calculate_decision_score(
        self,
        drift_report: DriftReport,
        performance_metrics: dict[str, float],
        config: RetrainingConfig,
        business_context: dict[str, Any],
    ) -> float:
        """Calculate decision score for retraining."""

        scores = {}

        # Performance impact score
        scores["performance_impact"] = self._calculate_performance_impact_score(
            performance_metrics, config
        )

        # Drift severity score
        scores["drift_severity"] = self._calculate_drift_severity_score(
            drift_report, config
        )

        # Business impact score
        scores["business_impact"] = self._calculate_business_impact_score(
            business_context or {}
        )

        # Resource availability score
        scores["resource_availability"] = self._calculate_resource_availability_score()

        # Calculate weighted score
        weighted_score = sum(
            scores[factor] * weight
            for factor, weight in self.decision_weights.items()
            if factor in scores
        )

        return min(1.0, max(0.0, weighted_score))

    def _calculate_performance_impact_score(
        self, performance_metrics: dict[str, float], config: RetrainingConfig
    ) -> float:
        """Calculate performance impact score."""

        if not performance_metrics:
            return 0.0

        # Check for performance degradation
        degradation_score = 0.0

        # Example: check accuracy degradation
        if "accuracy_drop" in performance_metrics:
            accuracy_drop = performance_metrics["accuracy_drop"]
            if accuracy_drop > config.performance_threshold:
                degradation_score = min(
                    1.0, accuracy_drop / (config.performance_threshold * 2)
                )

        # Check other metrics
        if "error_rate_increase" in performance_metrics:
            error_increase = performance_metrics["error_rate_increase"]
            degradation_score = max(degradation_score, min(1.0, error_increase * 2))

        return degradation_score

    def _calculate_drift_severity_score(
        self, drift_report: DriftReport, config: RetrainingConfig
    ) -> float:
        """Calculate drift severity score."""

        if not drift_report.overall_drift_detected:
            return 0.0

        severity_scores = {
            DriftSeverity.LOW: 0.2,
            DriftSeverity.MEDIUM: 0.5,
            DriftSeverity.HIGH: 0.8,
            DriftSeverity.CRITICAL: 1.0,
        }

        base_score = severity_scores.get(drift_report.overall_drift_severity, 0.0)

        # Adjust based on drift types
        drift_type_multipliers = {
            DriftType.CONCEPT_DRIFT: 1.5,
            DriftType.COVARIATE_SHIFT: 1.2,
            DriftType.DATA_DRIFT: 1.0,
        }

        max_multiplier = 1.0
        for drift_type in drift_report.drift_types_detected:
            multiplier = drift_type_multipliers.get(drift_type, 1.0)
            max_multiplier = max(max_multiplier, multiplier)

        return min(1.0, base_score * max_multiplier)

    def _calculate_business_impact_score(
        self, business_context: dict[str, Any]
    ) -> float:
        """Calculate business impact score."""

        # Default low impact if no context provided
        if not business_context:
            return 0.3

        impact_score = 0.0

        # Check business criticality
        if business_context.get("is_critical_model", False):
            impact_score += 0.5

        # Check user complaints
        user_complaints = business_context.get("user_complaints", 0)
        if user_complaints > 0:
            impact_score += min(0.3, user_complaints / 100)

        # Check financial impact
        financial_impact = business_context.get("estimated_financial_impact", 0)
        if financial_impact > 1000:  # $1000 threshold
            impact_score += min(0.4, financial_impact / 10000)

        return min(1.0, impact_score)

    def _calculate_resource_availability_score(self) -> float:
        """Calculate resource availability score."""

        # Simplified resource availability check
        # In practice, this would check actual compute resources, budget, etc.

        # Assume resources are generally available
        return 0.8

    def _calculate_urgency(
        self,
        drift_report: DriftReport,
        performance_metrics: dict[str, float],
        decision_score: float,
    ) -> RetrainingUrgency:
        """Calculate retraining urgency."""

        # Critical conditions
        if (
            drift_report.overall_drift_severity == DriftSeverity.CRITICAL
            or performance_metrics.get("accuracy_drop", 0) > 0.2
            or DriftType.CONCEPT_DRIFT in drift_report.drift_types_detected
        ):
            return RetrainingUrgency.CRITICAL

        # High urgency conditions
        if (
            drift_report.overall_drift_severity == DriftSeverity.HIGH
            or performance_metrics.get("accuracy_drop", 0) > 0.1
            or decision_score > 0.8
        ):
            return RetrainingUrgency.HIGH

        # Medium urgency conditions
        if (
            drift_report.overall_drift_severity == DriftSeverity.MEDIUM
            or performance_metrics.get("accuracy_drop", 0) > 0.05
            or decision_score > 0.6
        ):
            return RetrainingUrgency.MEDIUM

        return RetrainingUrgency.LOW

    def _generate_justification(
        self,
        drift_report: DriftReport,
        performance_metrics: dict[str, float],
        decision_score: float,
        urgency: RetrainingUrgency,
    ) -> str:
        """Generate justification for retraining decision."""

        reasons = []

        # Drift-based reasons
        if drift_report.overall_drift_detected:
            reasons.append(
                f"{drift_report.overall_drift_severity.value} drift detected "
                f"in {len(drift_report.drifted_features)} features"
            )

        # Performance-based reasons
        accuracy_drop = performance_metrics.get("accuracy_drop", 0)
        if accuracy_drop > 0.05:
            reasons.append(f"Model accuracy dropped by {accuracy_drop:.1%}")

        error_increase = performance_metrics.get("error_rate_increase", 0)
        if error_increase > 0.02:
            reasons.append(f"Error rate increased by {error_increase:.1%}")

        # Concept drift
        if DriftType.CONCEPT_DRIFT in drift_report.drift_types_detected:
            reasons.append("Concept drift detected - target relationships have changed")

        if not reasons:
            reasons.append("Proactive retraining based on system analysis")

        justification = (
            f"Retraining recommended ({urgency.value} priority): " + "; ".join(reasons)
        )
        justification += f". Decision confidence: {decision_score:.1%}"

        return justification


class IntelligentRetrainingService:
    """Service for intelligent model retraining."""

    def __init__(
        self,
        ml_pipeline_service: AdvancedMLPipelineService,
        drift_detection_service: ModelDriftDetectionService,
    ):
        self.ml_pipeline_service = ml_pipeline_service
        self.drift_detection_service = drift_detection_service
        self.logger = StructuredLogger("intelligent_retraining")
        self.metrics_service = MetricsService()

        # Core components
        self.decision_engine = IntelligentRetrainingDecisionEngine()

        # State management
        self.retraining_configs: dict[str, RetrainingConfig] = {}
        self.active_jobs: dict[str, RetrainingJob] = {}
        self.job_history: list[RetrainingJob] = []
        self.pending_requests: list[RetrainingRequest] = []

        # Monitoring
        self.is_running = False
        self.monitoring_interval = 300  # 5 minutes

        # Callbacks
        self.job_callbacks: list[Callable[[RetrainingJob], None]] = []

    async def register_model_for_retraining(
        self, model_id: str, config: RetrainingConfig
    ):
        """Register a model for intelligent retraining monitoring."""

        self.retraining_configs[model_id] = config

        self.logger.info(f"Registered model {model_id} for intelligent retraining")

    async def start_monitoring(self):
        """Start the intelligent retraining monitoring system."""

        self.is_running = True
        self.logger.info("Starting intelligent retraining monitoring")

        while self.is_running:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)

    async def _monitoring_cycle(self):
        """Execute one monitoring cycle."""

        self.logger.debug("Starting monitoring cycle")

        # Check each registered model
        for model_id, config in self.retraining_configs.items():
            try:
                await self._check_model_for_retraining(model_id, config)
            except Exception as e:
                self.logger.error(f"Error checking model {model_id}: {e}")

        # Process pending requests
        await self._process_pending_requests()

        # Update active jobs
        await self._update_active_jobs()

        self.logger.debug("Monitoring cycle completed")

    async def _check_model_for_retraining(
        self, model_id: str, config: RetrainingConfig
    ):
        """Check if a model needs retraining."""

        # Skip if already retraining
        if any(job.request.model_id == model_id for job in self.active_jobs.values()):
            return

        # Check minimum interval
        last_retraining = self._get_last_retraining_time(model_id)
        if (
            last_retraining
            and datetime.now() - last_retraining < config.min_retraining_interval
        ):
            return

        # Get latest drift report
        try:
            # This would get the latest drift report for the model
            # For now, we'll simulate
            drift_report = await self._get_latest_drift_report(model_id)
            if not drift_report:
                return

            # Get performance metrics
            performance_metrics = await self._get_performance_metrics(model_id)

            # Make retraining decision
            (
                should_retrain,
                urgency,
                justification,
            ) = await self.decision_engine.should_retrain(
                model_id, drift_report, performance_metrics, config
            )

            if should_retrain:
                await self._create_retraining_request(
                    model_id,
                    RetrainingTrigger.PERFORMANCE_DRIFT,
                    urgency,
                    justification,
                )

        except Exception as e:
            self.logger.error(f"Error checking model {model_id} for retraining: {e}")

    async def _create_retraining_request(
        self,
        model_id: str,
        trigger: RetrainingTrigger,
        urgency: RetrainingUrgency,
        justification: str,
        requested_by: str = "system",
    ) -> str:
        """Create a retraining request."""

        request_id = f"retrain_{model_id}_{int(time.time())}"

        request = RetrainingRequest(
            request_id=request_id,
            model_id=model_id,
            trigger=trigger,
            urgency=urgency,
            justification=justification,
            requested_by=requested_by,
        )

        self.pending_requests.append(request)

        self.logger.info(
            f"Created retraining request {request_id} for model {model_id} "
            f"(urgency: {urgency.value})"
        )

        return request_id

    async def _process_pending_requests(self):
        """Process pending retraining requests."""

        if not self.pending_requests:
            return

        # Sort by urgency and creation time
        urgency_order = {
            RetrainingUrgency.CRITICAL: 0,
            RetrainingUrgency.HIGH: 1,
            RetrainingUrgency.MEDIUM: 2,
            RetrainingUrgency.LOW: 3,
        }

        self.pending_requests.sort(
            key=lambda r: (urgency_order[r.urgency], r.created_at)
        )

        # Process highest priority requests
        for request in self.pending_requests[:]:
            config = self.retraining_configs.get(request.model_id)
            if not config:
                self.pending_requests.remove(request)
                continue

            # Check approval requirements
            if await self._should_auto_approve(request, config):
                await self._start_retraining_job(request, config)
                self.pending_requests.remove(request)
            elif request.urgency == RetrainingUrgency.CRITICAL:
                # Auto-approve critical requests
                await self._start_retraining_job(request, config)
                self.pending_requests.remove(request)

    async def _should_auto_approve(
        self, request: RetrainingRequest, config: RetrainingConfig
    ) -> bool:
        """Determine if request should be auto-approved."""

        if request.urgency == RetrainingUrgency.LOW and config.auto_approve_low_risk:
            return True

        if (
            request.urgency == RetrainingUrgency.MEDIUM
            and config.auto_approve_medium_risk
        ):
            return True

        if (
            request.urgency == RetrainingUrgency.HIGH
            and not config.require_human_approval_high_risk
        ):
            return True

        return False

    async def _start_retraining_job(
        self, request: RetrainingRequest, config: RetrainingConfig
    ) -> str:
        """Start a retraining job."""

        job_id = f"job_{request.request_id}_{int(time.time())}"

        job = RetrainingJob(
            job_id=job_id,
            request=request,
            config=config,
            started_at=datetime.now(),
        )

        self.active_jobs[job_id] = job

        # Start the retraining process in background
        asyncio.create_task(self._execute_retraining_job(job))

        self.logger.info(
            f"Started retraining job {job_id} for model {request.model_id}"
        )

        return job_id

    async def _execute_retraining_job(self, job: RetrainingJob):
        """Execute a retraining job."""

        try:
            job.status = "running"
            job.current_stage = "data_preparation"
            job.progress = 0.1

            # Notify callbacks
            await self._notify_job_callbacks(job)

            # Get training data
            await self._update_job_stage(job, "data_collection", 0.2)
            X_train, y_train, X_val, y_val = await self._get_training_data(
                job.request.model_id
            )

            # Create experiment
            await self._update_job_stage(job, "experiment_creation", 0.3)
            experiment_id = await self._create_retraining_experiment(job)
            job.experiment_id = experiment_id

            # Run training
            await self._update_job_stage(job, "model_training", 0.4)
            training_results = (
                await self.ml_pipeline_service.run_hyperparameter_optimization(
                    experiment_id, X_train, y_train, X_val, y_val
                )
            )

            # Evaluate new model
            await self._update_job_stage(job, "model_evaluation", 0.7)
            evaluation_results = await self._evaluate_new_model(job, X_val, y_val)

            # Compare with old model
            await self._update_job_stage(job, "model_comparison", 0.8)
            comparison_results = await self._compare_models(job, X_val, y_val)

            # Decide on deployment
            await self._update_job_stage(job, "deployment_decision", 0.9)
            should_deploy = await self._should_deploy_new_model(job, comparison_results)

            if should_deploy:
                await self._update_job_stage(job, "model_deployment", 0.95)
                deployment_id = await self.ml_pipeline_service.deploy_model(
                    experiment_id, f"retrained_{job.request.model_id}"
                )
                job.trigger_data["deployment_id"] = deployment_id

            # Complete job
            job.status = "completed"
            job.completed_at = datetime.now()
            job.progress = 1.0
            job.current_stage = "completed"

            # Store results
            job.new_model_performance = evaluation_results
            job.improvement_metrics = comparison_results

            self.logger.info(f"Retraining job {job.job_id} completed successfully")

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()

            self.logger.error(f"Retraining job {job.job_id} failed: {e}")

        finally:
            # Move to history
            self.job_history.append(job)
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

            # Notify callbacks
            await self._notify_job_callbacks(job)

            # Record metrics
            self.metrics_service.record_retraining_job(
                model_id=job.request.model_id,
                job_status=job.status,
                duration=(job.completed_at - job.started_at).total_seconds()
                if job.completed_at
                else 0,
                trigger=job.request.trigger.value,
                urgency=job.request.urgency.value,
            )

    async def _update_job_stage(self, job: RetrainingJob, stage: str, progress: float):
        """Update job stage and progress."""

        job.current_stage = stage
        job.progress = progress

        await self._notify_job_callbacks(job)

        self.logger.debug(f"Job {job.job_id} stage: {stage} ({progress:.1%})")

    async def _notify_job_callbacks(self, job: RetrainingJob):
        """Notify job callbacks."""

        for callback in self.job_callbacks:
            try:
                callback(job)
            except Exception as e:
                self.logger.error(f"Error in job callback: {e}")

    async def _get_training_data(
        self, model_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get training data for model retraining."""

        # This would implement actual data retrieval logic
        # For now, return dummy data

        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        # Split into train/validation
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        return X_train, y_train, X_val, y_val

    async def _create_retraining_experiment(self, job: RetrainingJob) -> str:
        """Create ML experiment for retraining."""

        experiment_name = f"retrain_{job.request.model_id}_{job.job_id}"

        # Define hyperparameter space based on strategy
        hyperparameter_space = self._get_hyperparameter_space(
            job.config.retraining_strategy
        )

        experiment_id = await self.ml_pipeline_service.create_model_experiment(
            name=experiment_name,
            description=f"Retraining for {job.request.model_id} triggered by {job.request.trigger.value}",
            algorithm="isolation_forest",
            hyperparameter_space=hyperparameter_space,
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
        )

        return experiment_id

    def _get_hyperparameter_space(self, strategy: RetrainingStrategy) -> dict[str, Any]:
        """Get hyperparameter space based on retraining strategy."""

        if strategy == RetrainingStrategy.FULL_RETRAIN:
            return {
                "contamination": [0.05, 0.1, 0.15, 0.2],
                "n_estimators": [50, 100, 200],
                "max_samples": ["auto", 0.5, 0.7, 1.0],
            }
        else:
            # Simplified parameter space for other strategies
            return {
                "contamination": [0.1, 0.15],
                "n_estimators": [100],
            }

    async def _evaluate_new_model(
        self, job: RetrainingJob, X_val: np.ndarray, y_val: np.ndarray
    ) -> dict[str, float]:
        """Evaluate the newly trained model."""

        if not job.experiment_id:
            return {}

        # Get experiment status
        experiment_status = await self.ml_pipeline_service.get_experiment_status(
            job.experiment_id
        )

        if experiment_status.get("results"):
            return experiment_status["results"].get("validation_metrics", {})

        return {}

    async def _compare_models(
        self, job: RetrainingJob, X_val: np.ndarray, y_val: np.ndarray
    ) -> dict[str, float]:
        """Compare new model with existing model."""

        # This would implement actual model comparison
        # For now, return simulated improvement metrics

        return {
            "accuracy_improvement": 0.03,
            "precision_improvement": 0.02,
            "recall_improvement": 0.04,
            "f1_improvement": 0.025,
        }

    async def _should_deploy_new_model(
        self, job: RetrainingJob, comparison_results: dict[str, float]
    ) -> bool:
        """Determine if new model should be deployed."""

        min_improvement = job.config.minimum_improvement

        # Check if there's sufficient improvement
        accuracy_improvement = comparison_results.get("accuracy_improvement", 0)

        return accuracy_improvement >= min_improvement

    async def _get_latest_drift_report(self, model_id: str) -> DriftReport | None:
        """Get the latest drift report for a model."""

        # This would implement actual drift report retrieval
        # For now, return None to indicate no drift report available
        return None

    async def _get_performance_metrics(self, model_id: str) -> dict[str, float]:
        """Get current performance metrics for a model."""

        # This would implement actual performance metric retrieval
        # For now, return simulated metrics

        return {
            "accuracy_drop": 0.03,
            "error_rate_increase": 0.02,
            "latency_increase": 1.5,
        }

    def _get_last_retraining_time(self, model_id: str) -> datetime | None:
        """Get the last retraining time for a model."""

        # Find most recent completed job for this model
        model_jobs = [
            job
            for job in self.job_history
            if job.request.model_id == model_id and job.status == "completed"
        ]

        if model_jobs:
            latest_job = max(model_jobs, key=lambda j: j.completed_at or datetime.min)
            return latest_job.completed_at

        return None

    async def _update_active_jobs(self):
        """Update status of active jobs."""

        # This would check job status and update progress
        # Most of the job execution is handled in _execute_retraining_job
        pass

    def add_job_callback(self, callback: Callable[[RetrainingJob], None]):
        """Add callback for job status updates."""
        self.job_callbacks.append(callback)

    def get_active_jobs(self) -> list[RetrainingJob]:
        """Get list of active retraining jobs."""
        return list(self.active_jobs.values())

    def get_job_history(
        self, model_id: str = None, limit: int = 50
    ) -> list[RetrainingJob]:
        """Get retraining job history."""

        jobs = self.job_history

        if model_id:
            jobs = [job for job in jobs if job.request.model_id == model_id]

        # Sort by completion time (most recent first)
        jobs.sort(key=lambda j: j.completed_at or datetime.min, reverse=True)

        return jobs[:limit]

    async def request_manual_retraining(
        self,
        model_id: str,
        urgency: RetrainingUrgency = RetrainingUrgency.MEDIUM,
        justification: str = "Manual retraining request",
        requested_by: str = "user",
    ) -> str:
        """Request manual retraining for a model."""

        if model_id not in self.retraining_configs:
            raise ValueError(f"Model {model_id} not registered for retraining")

        return await self._create_retraining_request(
            model_id, RetrainingTrigger.MANUAL, urgency, justification, requested_by
        )

    async def cancel_retraining_job(self, job_id: str) -> bool:
        """Cancel an active retraining job."""

        if job_id not in self.active_jobs:
            return False

        job = self.active_jobs[job_id]
        job.status = "cancelled"
        job.completed_at = datetime.now()

        # Move to history
        self.job_history.append(job)
        del self.active_jobs[job_id]

        await self._notify_job_callbacks(job)

        self.logger.info(f"Cancelled retraining job {job_id}")

        return True

    async def stop_monitoring(self):
        """Stop the intelligent retraining monitoring."""

        self.is_running = False
        self.logger.info("Stopped intelligent retraining monitoring")

    def get_service_stats(self) -> dict[str, Any]:
        """Get service statistics."""

        return {
            "registered_models": len(self.retraining_configs),
            "active_jobs": len(self.active_jobs),
            "pending_requests": len(self.pending_requests),
            "completed_jobs": len(
                [j for j in self.job_history if j.status == "completed"]
            ),
            "failed_jobs": len([j for j in self.job_history if j.status == "failed"]),
            "is_monitoring": self.is_running,
            "monitoring_interval": self.monitoring_interval,
        }
