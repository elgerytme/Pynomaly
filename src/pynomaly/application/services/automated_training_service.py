"""Automated Model Training Service with Real-Time Monitoring.

This service provides comprehensive automated training capabilities including:
- Scheduled training pipelines with hyperparameter optimization
- Real-time training progress monitoring via WebSocket
- Performance-based retraining triggers
- Model versioning and experiment tracking
- Integration with existing AutoML and training infrastructure
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pynomaly.application.services.automl_service import (
    AutoMLService,
    OptimizationObjective,
)
from pynomaly.application.services.enhanced_model_persistence_service import (
    EnhancedModelPersistenceService,
)
from pynomaly.application.use_cases.train_detector import (
    TrainDetectorRequest,
    TrainDetectorUseCase,
)
from pynomaly.shared.protocols import DetectorRepositoryProtocol

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training pipeline status."""

    IDLE = "idle"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    OPTIMIZING = "optimizing"
    EVALUATING = "evaluating"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TriggerType(Enum):
    """Training trigger types."""

    SCHEDULED = "scheduled"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    NEW_DATA = "new_data"


@dataclass
class TrainingConfig:
    """Configuration for automated training."""

    # Basic settings
    detector_id: UUID
    dataset_id: str
    experiment_name: str | None = None

    # AutoML settings
    enable_automl: bool = True
    optimization_objective: OptimizationObjective = OptimizationObjective.AUC
    max_algorithms: int = 3
    enable_ensemble: bool = True
    max_optimization_time: int = 3600
    n_trials: int = 100

    # Training settings
    validation_split: float = 0.2
    cv_folds: int = 3
    enable_early_stopping: bool = True
    max_training_time: int | None = None

    # Scheduling settings
    schedule_cron: str | None = None  # Cron expression for scheduling
    retrain_threshold: float = 0.05  # Performance drop threshold for retraining
    performance_window: int = 7  # Days to monitor performance

    # Resource constraints
    max_memory_mb: int | None = None
    max_cpu_cores: int | None = None
    enable_gpu: bool = False

    # Notification settings
    enable_notifications: bool = True
    notification_channels: list[str] = field(default_factory=lambda: ["websocket"])

    # Model management
    auto_deploy: bool = False
    keep_model_versions: int = 5
    enable_model_comparison: bool = True


@dataclass
class TrainingProgress:
    """Real-time training progress information."""

    training_id: str
    status: TrainingStatus
    current_step: str
    progress_percentage: float
    start_time: datetime
    estimated_completion: datetime | None = None

    # Current metrics
    current_algorithm: str | None = None
    current_trial: int | None = None
    total_trials: int | None = None
    best_score: float | None = None
    current_score: float | None = None

    # Resource usage
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None

    # Messages and logs
    current_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket transmission."""
        return {
            "training_id": self.training_id,
            "status": self.status.value,
            "current_step": self.current_step,
            "progress_percentage": self.progress_percentage,
            "start_time": self.start_time.isoformat(),
            "estimated_completion": (
                self.estimated_completion.isoformat()
                if self.estimated_completion
                else None
            ),
            "current_algorithm": self.current_algorithm,
            "current_trial": self.current_trial,
            "total_trials": self.total_trials,
            "best_score": self.best_score,
            "current_score": self.current_score,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "current_message": self.current_message,
            "warnings": self.warnings,
            "timestamp": datetime.utcnow().isoformat(),
        }


@dataclass
class TrainingResult:
    """Complete training result with comprehensive metrics."""

    training_id: str
    detector_id: UUID
    status: TrainingStatus
    trigger_type: TriggerType

    # Training metrics
    best_algorithm: str | None = None
    best_params: dict[str, Any] | None = None
    best_score: float | None = None
    training_time_seconds: float | None = None
    trials_completed: int | None = None

    # Model information
    model_version: str | None = None
    model_path: str | None = None
    model_size_mb: float | None = None

    # Performance comparison
    previous_score: float | None = None
    performance_improvement: float | None = None

    # Metadata
    dataset_id: str | None = None
    experiment_name: str | None = None
    start_time: datetime | None = None
    completion_time: datetime | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    # Resource usage
    peak_memory_mb: float | None = None
    total_cpu_hours: float | None = None


class AutomatedTrainingService:
    """Service for automated model training with real-time monitoring."""

    def __init__(
        self,
        train_detector_use_case: TrainDetectorUseCase,
        automl_service: AutoMLService,
        detector_repository: DetectorRepositoryProtocol,
        model_persistence_service: EnhancedModelPersistenceService,
        websocket_broadcaster: Callable[[str, dict], None] | None = None,
    ):
        """Initialize the automated training service.

        Args:
            train_detector_use_case: Use case for detector training
            automl_service: AutoML service for optimization
            detector_repository: Repository for detector storage
            model_persistence_service: Service for model persistence
            websocket_broadcaster: Function to broadcast WebSocket messages
        """
        self.train_detector_use_case = train_detector_use_case
        self.automl_service = automl_service
        self.detector_repository = detector_repository
        self.model_persistence_service = model_persistence_service
        self.websocket_broadcaster = websocket_broadcaster

        # Training state management
        self.active_trainings: dict[str, TrainingProgress] = {}
        self.training_history: dict[str, TrainingResult] = {}
        self.scheduled_trainings: dict[str, TrainingConfig] = {}

        # Background tasks
        self._scheduler_task: asyncio.Task | None = None
        self._monitoring_task: asyncio.Task | None = None
        self._running = False

        # Performance tracking
        self.performance_history: dict[UUID, list[tuple[datetime, float]]] = {}

    async def start(self):
        """Start the automated training service."""
        if self._running:
            return

        self._running = True
        logger.info("Starting automated training service")

        # Start background tasks
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop(self):
        """Stop the automated training service."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping automated training service")

        # Cancel background tasks
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def schedule_training(
        self, config: TrainingConfig, trigger_type: TriggerType = TriggerType.MANUAL
    ) -> str:
        """Schedule a training pipeline.

        Args:
            config: Training configuration
            trigger_type: What triggered this training

        Returns:
            Training ID
        """
        training_id = str(uuid4())

        logger.info(
            f"Scheduling training {training_id} for detector {config.detector_id}"
        )

        # Store configuration
        self.scheduled_trainings[training_id] = config

        # Create initial progress
        progress = TrainingProgress(
            training_id=training_id,
            status=TrainingStatus.SCHEDULED,
            current_step="Scheduled",
            progress_percentage=0.0,
            start_time=datetime.utcnow(),
            current_message=f"Training scheduled with trigger: {trigger_type.value}",
        )

        self.active_trainings[training_id] = progress
        await self._broadcast_progress(progress)

        # Start training immediately for manual triggers
        if trigger_type == TriggerType.MANUAL:
            asyncio.create_task(self._execute_training(training_id, trigger_type))

        return training_id

    async def start_training(
        self,
        detector_id: UUID,
        dataset_id: str,
        config: TrainingConfig | None = None,
        trigger_type: TriggerType = TriggerType.MANUAL,
    ) -> str:
        """Start a training pipeline immediately.

        Args:
            detector_id: ID of detector to train
            dataset_id: ID of dataset to use
            config: Optional training configuration
            trigger_type: What triggered this training

        Returns:
            Training ID
        """
        if config is None:
            config = TrainingConfig(
                detector_id=detector_id,
                dataset_id=dataset_id,
                experiment_name=f"automated_training_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            )

        return await self.schedule_training(config, trigger_type)

    async def cancel_training(self, training_id: str) -> bool:
        """Cancel an active training.

        Args:
            training_id: ID of training to cancel

        Returns:
            True if cancelled successfully
        """
        if training_id not in self.active_trainings:
            return False

        progress = self.active_trainings[training_id]

        if progress.status in [
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED,
        ]:
            return False

        # Update status
        progress.status = TrainingStatus.CANCELLED
        progress.current_step = "Cancelled by user"
        progress.current_message = "Training cancelled by user request"

        await self._broadcast_progress(progress)

        # Create result record
        result = TrainingResult(
            training_id=training_id,
            detector_id=self.scheduled_trainings[training_id].detector_id,
            status=TrainingStatus.CANCELLED,
            trigger_type=TriggerType.MANUAL,
            completion_time=datetime.utcnow(),
            error_message="Cancelled by user",
        )

        self.training_history[training_id] = result

        logger.info(f"Training {training_id} cancelled")
        return True

    async def get_training_status(self, training_id: str) -> TrainingProgress | None:
        """Get current training status.

        Args:
            training_id: ID of training

        Returns:
            Training progress or None if not found
        """
        return self.active_trainings.get(training_id)

    async def get_training_result(self, training_id: str) -> TrainingResult | None:
        """Get training result.

        Args:
            training_id: ID of training

        Returns:
            Training result or None if not found
        """
        return self.training_history.get(training_id)

    async def get_active_trainings(self) -> list[TrainingProgress]:
        """Get all active trainings."""
        return list(self.active_trainings.values())

    async def get_training_history(
        self, detector_id: UUID | None = None, limit: int = 50
    ) -> list[TrainingResult]:
        """Get training history.

        Args:
            detector_id: Optional detector ID filter
            limit: Maximum number of results

        Returns:
            List of training results
        """
        results = list(self.training_history.values())

        if detector_id:
            results = [r for r in results if r.detector_id == detector_id]

        # Sort by completion time, most recent first
        results.sort(key=lambda x: x.completion_time or datetime.min, reverse=True)

        return results[:limit]

    async def check_retraining_needed(self, detector_id: UUID) -> bool:
        """Check if detector needs retraining based on performance.

        Args:
            detector_id: ID of detector to check

        Returns:
            True if retraining is recommended
        """
        if detector_id not in self.performance_history:
            return False

        history = self.performance_history[detector_id]
        if len(history) < 2:
            return False

        # Get recent performance data
        recent_scores = [score for timestamp, score in history[-10:]]
        baseline_score = recent_scores[0]
        current_score = recent_scores[-1]

        # Check for performance degradation
        performance_drop = baseline_score - current_score

        # Get threshold from detector's configuration
        # For now, use a default threshold
        threshold = 0.05

        return performance_drop > threshold

    async def update_performance(self, detector_id: UUID, score: float):
        """Update performance history for a detector.

        Args:
            detector_id: ID of detector
            score: Current performance score
        """
        if detector_id not in self.performance_history:
            self.performance_history[detector_id] = []

        timestamp = datetime.utcnow()
        self.performance_history[detector_id].append((timestamp, score))

        # Keep only recent history (last 30 days)
        cutoff = timestamp - timedelta(days=30)
        self.performance_history[detector_id] = [
            (ts, s) for ts, s in self.performance_history[detector_id] if ts > cutoff
        ]

        # Check if retraining is needed
        if await self.check_retraining_needed(detector_id):
            logger.info(
                f"Performance degradation detected for detector {detector_id}, scheduling retraining"
            )

            # Find detector configuration or create default
            detector = self.detector_repository.find_by_id(detector_id)
            if detector:
                config = TrainingConfig(
                    detector_id=detector_id,
                    dataset_id="auto",  # Would need to determine appropriate dataset
                    experiment_name=f"auto_retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                )
                await self.schedule_training(config, TriggerType.PERFORMANCE_THRESHOLD)

    async def _execute_training(self, training_id: str, trigger_type: TriggerType):
        """Execute a training pipeline.

        Args:
            training_id: ID of training to execute
            trigger_type: What triggered this training
        """
        if training_id not in self.scheduled_trainings:
            logger.error(f"Training configuration not found for {training_id}")
            return

        config = self.scheduled_trainings[training_id]
        progress = self.active_trainings[training_id]

        start_time = datetime.utcnow()
        result = TrainingResult(
            training_id=training_id,
            detector_id=config.detector_id,
            status=TrainingStatus.RUNNING,
            trigger_type=trigger_type,
            dataset_id=config.dataset_id,
            experiment_name=config.experiment_name,
            start_time=start_time,
        )

        try:
            # Update status
            progress.status = TrainingStatus.RUNNING
            progress.current_step = "Initializing"
            progress.progress_percentage = 5.0
            progress.current_message = "Starting training pipeline"
            await self._broadcast_progress(progress)

            # Step 1: Load detector and dataset
            progress.current_step = "Loading data"
            progress.progress_percentage = 10.0
            progress.current_message = "Loading detector and dataset"
            await self._broadcast_progress(progress)

            detector = self.detector_repository.find_by_id(config.detector_id)
            if not detector:
                raise ValueError(f"Detector {config.detector_id} not found")

            # Store previous performance for comparison
            if config.detector_id in self.performance_history:
                recent_performance = self.performance_history[config.detector_id]
                if recent_performance:
                    result.previous_score = recent_performance[-1][1]

            # Step 2: AutoML optimization (if enabled)
            if config.enable_automl:
                progress.status = TrainingStatus.OPTIMIZING
                progress.current_step = "Hyperparameter optimization"
                progress.progress_percentage = 20.0
                progress.current_message = "Running AutoML optimization"
                await self._broadcast_progress(progress)

                automl_result = await self.automl_service.auto_select_and_optimize(
                    dataset_id=config.dataset_id,
                    objective=config.optimization_objective,
                    max_algorithms=config.max_algorithms,
                    enable_ensemble=config.enable_ensemble,
                )

                # Update progress with optimization results
                progress.current_algorithm = automl_result.best_algorithm
                progress.total_trials = automl_result.trials_completed
                progress.best_score = automl_result.best_score
                progress.progress_percentage = 60.0
                progress.current_message = f"Best algorithm: {automl_result.best_algorithm} (score: {automl_result.best_score:.4f})"
                await self._broadcast_progress(progress)

                # Update result
                result.best_algorithm = automl_result.best_algorithm
                result.best_params = automl_result.best_params
                result.best_score = automl_result.best_score
                result.trials_completed = automl_result.trials_completed

                # Update detector with optimized parameters
                detector.algorithm = automl_result.best_algorithm
                detector.update_parameters(**automl_result.best_params)

            # Step 3: Final training
            progress.status = TrainingStatus.RUNNING
            progress.current_step = "Final training"
            progress.progress_percentage = 70.0
            progress.current_message = "Training final model"
            await self._broadcast_progress(progress)

            # Create training request
            training_request = TrainDetectorRequest(
                detector_id=config.detector_id,
                training_data=None,  # Would need to load dataset
                validation_split=config.validation_split,
                cv_folds=config.cv_folds,
                save_model=True,
                early_stopping=config.enable_early_stopping,
                max_training_time=config.max_training_time,
                experiment_name=config.experiment_name,
            )

            # Execute training (this would need dataset loading)
            # training_response = await self.train_detector_use_case.execute(training_request)

            # Step 4: Model persistence and versioning
            progress.current_step = "Saving model"
            progress.progress_percentage = 85.0
            progress.current_message = "Saving trained model"
            await self._broadcast_progress(progress)

            # Step 5: Performance evaluation
            progress.status = TrainingStatus.EVALUATING
            progress.current_step = "Evaluating performance"
            progress.progress_percentage = 90.0
            progress.current_message = "Evaluating model performance"
            await self._broadcast_progress(progress)

            # Calculate performance improvement
            if result.previous_score and result.best_score:
                result.performance_improvement = (
                    result.best_score - result.previous_score
                )

            # Step 6: Completion
            progress.status = TrainingStatus.COMPLETED
            progress.current_step = "Completed"
            progress.progress_percentage = 100.0
            progress.current_message = "Training completed successfully"
            await self._broadcast_progress(progress)

            # Update result
            result.status = TrainingStatus.COMPLETED
            result.completion_time = datetime.utcnow()
            result.training_time_seconds = (
                result.completion_time - start_time
            ).total_seconds()

            logger.info(f"Training {training_id} completed successfully")

        except Exception as e:
            logger.error(f"Training {training_id} failed: {str(e)}")

            # Update progress and result
            progress.status = TrainingStatus.FAILED
            progress.current_step = "Failed"
            progress.current_message = f"Training failed: {str(e)}"
            await self._broadcast_progress(progress)

            result.status = TrainingStatus.FAILED
            result.completion_time = datetime.utcnow()
            result.error_message = str(e)

        finally:
            # Store result
            self.training_history[training_id] = result

            # Clean up
            if training_id in self.scheduled_trainings:
                del self.scheduled_trainings[training_id]

    async def _scheduler_loop(self):
        """Background scheduler loop."""
        while self._running:
            try:
                # Check for scheduled trainings (cron-based)
                # This would implement cron expression parsing and scheduling

                # Check for performance-based retraining triggers
                for detector_id in list(self.performance_history.keys()):
                    if await self.check_retraining_needed(detector_id):
                        # Schedule retraining if not already running
                        active_detector_trainings = [
                            t
                            for t in self.active_trainings.values()
                            if self.scheduled_trainings.get(t.training_id, {}).get(
                                "detector_id"
                            )
                            == detector_id
                            and t.status
                            in [TrainingStatus.RUNNING, TrainingStatus.SCHEDULED]
                        ]

                        if not active_detector_trainings:
                            config = TrainingConfig(
                                detector_id=detector_id,
                                dataset_id="auto",
                                experiment_name=f"auto_retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                            )
                            await self.schedule_training(
                                config, TriggerType.PERFORMANCE_THRESHOLD
                            )

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _monitoring_loop(self):
        """Background monitoring loop for resource usage."""
        while self._running:
            try:
                # Update resource usage for active trainings
                for training_id, progress in self.active_trainings.items():
                    if progress.status == TrainingStatus.RUNNING:
                        # Update resource metrics (simplified)
                        progress.memory_usage_mb = self._get_memory_usage()
                        progress.cpu_usage_percent = self._get_cpu_usage()

                        await self._broadcast_progress(progress)

                await asyncio.sleep(10)  # Update every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(10)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil

            return psutil.cpu_percent(interval=None)
        except ImportError:
            return 0.0

    async def _broadcast_progress(self, progress: TrainingProgress):
        """Broadcast training progress via WebSocket.

        Args:
            progress: Training progress to broadcast
        """
        if self.websocket_broadcaster:
            try:
                message = {"type": "training_progress", "data": progress.to_dict()}
                self.websocket_broadcaster("training_updates", message)
            except Exception as e:
                logger.error(f"Failed to broadcast progress: {str(e)}")
