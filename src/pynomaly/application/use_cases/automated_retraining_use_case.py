"""Use case for automated model retraining workflows.

This module implements the core use case for A-001: Automated Model Retraining Workflows.
It orchestrates automated model retraining based on performance degradation triggers,
data drift detection, and scheduled retraining policies.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol
from uuid import UUID, uuid4

from pynomaly.application.services.automated_training_service import (
    AutomatedTrainingService,
    TrainingConfig,
    TrainingResult,
    TriggerType,
)
from pynomaly.application.use_cases.drift_monitoring_use_case import DriftMonitoringUseCase
from pynomaly.application.use_cases.train_detector import TrainDetectorUseCase
from pynomaly.domain.entities import Detector, Dataset
from pynomaly.domain.exceptions import DomainError, ValidationError
from pynomaly.shared.protocols import DetectorRepositoryProtocol

logger = logging.getLogger(__name__)


class RetrainingPolicy(Enum):
    """Policies for automated retraining."""
    
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    COMBINED = "combined"


class RetrainingDecision(Enum):
    """Decisions for retraining necessity."""
    
    NO_RETRAINING_NEEDED = "no_retraining_needed"
    PERFORMANCE_RETRAINING = "performance_retraining"
    DRIFT_RETRAINING = "drift_retraining"
    SCHEDULED_RETRAINING = "scheduled_retraining"
    EMERGENCY_RETRAINING = "emergency_retraining"


@dataclass
class PerformanceDegradationTrigger:
    """Configuration for performance degradation triggers."""
    
    metric_name: str
    threshold: float
    evaluation_window_days: int = 7
    min_samples_required: int = 100
    consecutive_failures_threshold: int = 3
    severity_multiplier: float = 1.0


@dataclass
class RetrainingConfiguration:
    """Configuration for automated retraining."""
    
    detector_id: UUID
    policy: RetrainingPolicy
    enabled: bool = True
    
    # Performance degradation settings
    performance_triggers: List[PerformanceDegradationTrigger] = field(default_factory=list)
    
    # Data drift settings
    drift_threshold: float = 0.1
    drift_check_interval_hours: int = 24
    
    # Scheduled retraining settings
    schedule_cron: Optional[str] = None
    max_model_age_days: Optional[int] = None
    
    # Training configuration
    training_config: Optional[TrainingConfig] = None
    
    # Notification settings
    notify_on_trigger: bool = True
    notify_on_completion: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    
    # Constraints
    max_concurrent_retrainings: int = 3
    min_retraining_interval_hours: int = 6
    max_retraining_attempts: int = 3
    
    # Validation settings
    require_performance_improvement: bool = True
    improvement_threshold: float = 0.02
    validation_dataset_id: Optional[str] = None


@dataclass
class RetrainingRequest:
    """Request for automated model retraining."""
    
    detector_id: UUID
    trigger_type: TriggerType
    trigger_reason: str
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=datetime.utcnow)
    
    # Performance trigger specific
    performance_metrics: Optional[Dict[str, float]] = None
    performance_degradation: Optional[float] = None
    
    # Drift trigger specific
    drift_score: Optional[float] = None
    drift_features: Optional[List[str]] = None
    
    # Custom configuration override
    training_config_override: Optional[TrainingConfig] = None


@dataclass
class RetrainingResponse:
    """Response from automated model retraining."""
    
    retraining_id: str
    detector_id: UUID
    decision: RetrainingDecision
    training_id: Optional[str] = None
    
    # Status information
    status: str = "initiated"
    message: str = ""
    
    # Performance comparison
    baseline_metrics: Optional[Dict[str, float]] = None
    new_metrics: Optional[Dict[str, float]] = None
    improvement: Optional[float] = None
    
    # Timing information
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Result details
    training_result: Optional[TrainingResult] = None
    validation_passed: bool = False
    deployed: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class ModelPerformanceMonitor(Protocol):
    """Protocol for monitoring model performance degradation."""
    
    async def evaluate_performance_degradation(
        self, 
        detector_id: UUID,
        baseline_period_days: int = 30,
        evaluation_period_days: int = 7
    ) -> Dict[str, Any]:
        """Evaluate if model performance has degraded."""
        ...
    
    async def get_performance_metrics(
        self, 
        detector_id: UUID,
        time_window_days: int = 7
    ) -> Dict[str, float]:
        """Get current performance metrics for a detector."""
        ...


class AutomatedRetrainingUseCase:
    """Use case for automated model retraining workflows."""
    
    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        training_service: AutomatedTrainingService,
        drift_monitoring_use_case: DriftMonitoringUseCase,
        train_detector_use_case: TrainDetectorUseCase,
        performance_monitor: Optional[ModelPerformanceMonitor] = None,
        notification_service: Optional[Any] = None,
    ):
        """Initialize the automated retraining use case.
        
        Args:
            detector_repository: Repository for detector storage
            training_service: Service for automated training
            drift_monitoring_use_case: Use case for drift monitoring
            train_detector_use_case: Use case for training detectors
            performance_monitor: Service for monitoring performance degradation
            notification_service: Service for sending notifications
        """
        self.detector_repository = detector_repository
        self.training_service = training_service
        self.drift_monitoring_use_case = drift_monitoring_use_case
        self.train_detector_use_case = train_detector_use_case
        self.performance_monitor = performance_monitor
        self.notification_service = notification_service
        
        # Active retraining configurations
        self.retraining_configs: Dict[UUID, RetrainingConfiguration] = {}
        
        # Active retraining requests
        self.active_retrainings: Dict[str, RetrainingResponse] = {}
        
        # Background monitoring tasks
        self.monitoring_tasks: Dict[UUID, asyncio.Task] = {}
        
        # Performance history for tracking
        self.performance_history: Dict[UUID, List[Dict[str, Any]]] = {}
        
        logger.info("Automated retraining use case initialized")
    
    async def configure_retraining(
        self, 
        config: RetrainingConfiguration
    ) -> RetrainingConfiguration:
        """Configure automated retraining for a detector.
        
        Args:
            config: Retraining configuration
            
        Returns:
            Updated configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        if not config.detector_id:
            raise ValueError("Detector ID is required")
            
        detector = self.detector_repository.find_by_id(config.detector_id)
        if not detector:
            raise ValueError(f"Detector {config.detector_id} not found")
        
        # Validate performance triggers
        if config.policy in [RetrainingPolicy.PERFORMANCE_DEGRADATION, RetrainingPolicy.COMBINED]:
            if not config.performance_triggers:
                raise ValueError("Performance triggers required for performance-based retraining")
        
        # Validate scheduled settings
        if config.policy in [RetrainingPolicy.SCHEDULED, RetrainingPolicy.COMBINED]:
            if not config.schedule_cron and not config.max_model_age_days:
                raise ValueError("Schedule configuration required for scheduled retraining")
        
        # Store configuration
        self.retraining_configs[config.detector_id] = config
        
        # Start monitoring if enabled
        if config.enabled:
            await self._start_monitoring(config.detector_id)
        
        logger.info(f"Configured automated retraining for detector {config.detector_id}")
        return config
    
    async def evaluate_retraining_necessity(
        self, 
        detector_id: UUID,
        force_evaluation: bool = False
    ) -> RetrainingDecision:
        """Evaluate if retraining is necessary for a detector.
        
        Args:
            detector_id: ID of the detector to evaluate
            force_evaluation: Force evaluation even if recently checked
            
        Returns:
            Decision about retraining necessity
        """
        config = self.retraining_configs.get(detector_id)
        if not config or not config.enabled:
            return RetrainingDecision.NO_RETRAINING_NEEDED
        
        # Check if minimum retraining interval has passed
        if not force_evaluation and not await self._can_retrain(detector_id):
            return RetrainingDecision.NO_RETRAINING_NEEDED
        
        # Check for performance degradation
        if config.policy in [RetrainingPolicy.PERFORMANCE_DEGRADATION, RetrainingPolicy.COMBINED]:
            if await self._check_performance_degradation(detector_id, config):
                return RetrainingDecision.PERFORMANCE_RETRAINING
        
        # Check for data drift
        if config.policy in [RetrainingPolicy.DATA_DRIFT, RetrainingPolicy.COMBINED]:
            if await self._check_data_drift(detector_id, config):
                return RetrainingDecision.DRIFT_RETRAINING
        
        # Check for scheduled retraining
        if config.policy in [RetrainingPolicy.SCHEDULED, RetrainingPolicy.COMBINED]:
            if await self._check_scheduled_retraining(detector_id, config):
                return RetrainingDecision.SCHEDULED_RETRAINING
        
        return RetrainingDecision.NO_RETRAINING_NEEDED
    
    async def request_retraining(
        self, 
        request: RetrainingRequest
    ) -> RetrainingResponse:
        """Request automated model retraining.
        
        Args:
            request: Retraining request
            
        Returns:
            Retraining response
        """
        retraining_id = str(uuid4())
        
        # Validate detector exists
        detector = self.detector_repository.find_by_id(request.detector_id)
        if not detector:
            raise ValueError(f"Detector {request.detector_id} not found")
        
        # Get configuration
        config = self.retraining_configs.get(request.detector_id)
        if not config:
            raise ValueError(f"No retraining configuration for detector {request.detector_id}")
        
        # Check constraints
        if not await self._can_retrain(request.detector_id):
            response = RetrainingResponse(
                retraining_id=retraining_id,
                detector_id=request.detector_id,
                decision=RetrainingDecision.NO_RETRAINING_NEEDED,
                status="rejected",
                message="Minimum retraining interval not met"
            )
            return response
        
        # Create response
        response = RetrainingResponse(
            retraining_id=retraining_id,
            detector_id=request.detector_id,
            decision=RetrainingDecision.PERFORMANCE_RETRAINING,  # Will be updated based on trigger
            metadata=request.metadata
        )
        
        # Store active retraining
        self.active_retrainings[retraining_id] = response
        
        try:
            # Execute retraining workflow
            await self._execute_retraining_workflow(request, response, config)
            
        except Exception as e:
            logger.error(f"Retraining failed for {request.detector_id}: {e}")
            response.status = "failed"
            response.message = str(e)
            response.completed_at = datetime.utcnow()
            
            # Send failure notification
            if config.notify_on_completion:
                await self._send_notification(
                    "retraining_failed",
                    f"Retraining failed for detector {request.detector_id}: {e}",
                    config.notification_channels
                )
        
        return response
    
    async def get_retraining_status(self, retraining_id: str) -> Optional[RetrainingResponse]:
        """Get status of a retraining request.
        
        Args:
            retraining_id: ID of the retraining request
            
        Returns:
            Retraining response if found
        """
        return self.active_retrainings.get(retraining_id)
    
    async def cancel_retraining(self, retraining_id: str) -> bool:
        """Cancel an active retraining request.
        
        Args:
            retraining_id: ID of the retraining request
            
        Returns:
            True if cancelled successfully
        """
        response = self.active_retrainings.get(retraining_id)
        if not response:
            return False
        
        # Cancel training if in progress
        if response.training_id:
            await self.training_service.cancel_training(response.training_id)
        
        # Update status
        response.status = "cancelled"
        response.completed_at = datetime.utcnow()
        
        logger.info(f"Cancelled retraining {retraining_id}")
        return True
    
    async def get_retraining_history(
        self, 
        detector_id: UUID,
        limit: int = 50
    ) -> List[RetrainingResponse]:
        """Get retraining history for a detector.
        
        Args:
            detector_id: ID of the detector
            limit: Maximum number of results
            
        Returns:
            List of retraining responses
        """
        # Filter by detector ID and sort by completion time
        history = [
            response for response in self.active_retrainings.values()
            if response.detector_id == detector_id and response.completed_at
        ]
        
        history.sort(key=lambda x: x.completed_at or datetime.min, reverse=True)
        return history[:limit]
    
    async def _start_monitoring(self, detector_id: UUID) -> None:
        """Start monitoring for a detector."""
        if detector_id in self.monitoring_tasks:
            # Cancel existing task
            self.monitoring_tasks[detector_id].cancel()
        
        # Start new monitoring task
        self.monitoring_tasks[detector_id] = asyncio.create_task(
            self._monitoring_loop(detector_id)
        )
        
        logger.info(f"Started monitoring for detector {detector_id}")
    
    async def _monitoring_loop(self, detector_id: UUID) -> None:
        """Background monitoring loop for a detector."""
        config = self.retraining_configs.get(detector_id)
        if not config:
            return
        
        try:
            while config.enabled:
                # Evaluate retraining necessity
                decision = await self.evaluate_retraining_necessity(detector_id)
                
                if decision != RetrainingDecision.NO_RETRAINING_NEEDED:
                    # Create retraining request
                    request = RetrainingRequest(
                        detector_id=detector_id,
                        trigger_type=TriggerType.PERFORMANCE_THRESHOLD,
                        trigger_reason=f"Automated trigger: {decision.value}",
                        priority=1 if decision == RetrainingDecision.EMERGENCY_RETRAINING else 2
                    )
                    
                    # Request retraining
                    await self.request_retraining(request)
                
                # Wait before next check
                await asyncio.sleep(config.drift_check_interval_hours * 3600)
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for detector {detector_id}")
        except Exception as e:
            logger.error(f"Monitoring error for detector {detector_id}: {e}")
    
    async def _can_retrain(self, detector_id: UUID) -> bool:
        """Check if detector can be retrained based on constraints."""
        config = self.retraining_configs.get(detector_id)
        if not config:
            return False
        
        # Check minimum interval
        last_retraining = None
        for response in self.active_retrainings.values():
            if (response.detector_id == detector_id and 
                response.completed_at and 
                response.status == "completed"):
                if not last_retraining or response.completed_at > last_retraining:
                    last_retraining = response.completed_at
        
        if last_retraining:
            min_interval = timedelta(hours=config.min_retraining_interval_hours)
            if datetime.utcnow() - last_retraining < min_interval:
                return False
        
        # Check concurrent retrainings
        active_count = sum(
            1 for response in self.active_retrainings.values()
            if (response.detector_id == detector_id and 
                response.status in ["initiated", "running"])
        )
        
        return active_count < config.max_concurrent_retrainings
    
    async def _check_performance_degradation(
        self, 
        detector_id: UUID,
        config: RetrainingConfiguration
    ) -> bool:
        """Check if performance has degraded."""
        if not self.performance_monitor:
            logger.warning("Performance monitor not available")
            return False
        
        try:
            # Get current performance metrics
            current_metrics = await self.performance_monitor.get_performance_metrics(
                detector_id, 
                config.performance_triggers[0].evaluation_window_days
            )
            
            # Check each performance trigger
            for trigger in config.performance_triggers:
                if trigger.metric_name in current_metrics:
                    current_value = current_metrics[trigger.metric_name]
                    
                    # Check if below threshold
                    if current_value < trigger.threshold:
                        logger.info(
                            f"Performance degradation detected for {detector_id}: "
                            f"{trigger.metric_name} = {current_value} < {trigger.threshold}"
                        )
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
            return False
    
    async def _check_data_drift(
        self, 
        detector_id: UUID,
        config: RetrainingConfiguration
    ) -> bool:
        """Check if data drift has occurred."""
        try:
            # Use drift monitoring use case
            drift_result = await self.drift_monitoring_use_case.perform_drift_check(
                str(detector_id)
            )
            
            if drift_result.drift_detected:
                # Check if drift score exceeds threshold
                if hasattr(drift_result, 'drift_score') and drift_result.drift_score:
                    if drift_result.drift_score > config.drift_threshold:
                        logger.info(
                            f"Data drift detected for {detector_id}: "
                            f"score = {drift_result.drift_score} > {config.drift_threshold}"
                        )
                        return True
                else:
                    # Default to detected if no score available
                    logger.info(f"Data drift detected for {detector_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return False
    
    async def _check_scheduled_retraining(
        self, 
        detector_id: UUID,
        config: RetrainingConfiguration
    ) -> bool:
        """Check if scheduled retraining is needed."""
        try:
            # Check model age
            if config.max_model_age_days:
                detector = self.detector_repository.find_by_id(detector_id)
                if detector and detector.created_at:
                    age_days = (datetime.utcnow() - detector.created_at).days
                    if age_days >= config.max_model_age_days:
                        logger.info(
                            f"Scheduled retraining needed for {detector_id}: "
                            f"age = {age_days} days >= {config.max_model_age_days}"
                        )
                        return True
            
            # Check cron schedule (simplified - would need proper cron parsing)
            if config.schedule_cron:
                # This is a simplified check - in production, use a proper cron library
                # For now, just check if it's been more than a day since last training
                last_training = await self._get_last_training_time(detector_id)
                if last_training:
                    hours_since = (datetime.utcnow() - last_training).total_seconds() / 3600
                    if hours_since >= 24:  # Daily schedule assumption
                        logger.info(f"Scheduled retraining needed for {detector_id}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking scheduled retraining: {e}")
            return False
    
    async def _get_last_training_time(self, detector_id: UUID) -> Optional[datetime]:
        """Get the last training time for a detector."""
        # Check training history
        history = await self.training_service.get_training_history(detector_id, limit=1)
        if history:
            return history[0].completion_time
        return None
    
    async def _execute_retraining_workflow(
        self,
        request: RetrainingRequest,
        response: RetrainingResponse,
        config: RetrainingConfiguration
    ) -> None:
        """Execute the retraining workflow."""
        response.status = "running"
        
        # Get baseline metrics
        if self.performance_monitor:
            try:
                response.baseline_metrics = await self.performance_monitor.get_performance_metrics(
                    request.detector_id
                )
            except Exception as e:
                logger.warning(f"Could not get baseline metrics: {e}")
        
        # Prepare training configuration
        training_config = request.training_config_override or config.training_config
        if not training_config:
            training_config = TrainingConfig(
                detector_id=request.detector_id,
                dataset_id="auto",  # Would need proper dataset selection
                experiment_name=f"automated_retraining_{response.retraining_id}",
                enable_automl=True,
                enable_model_comparison=True
            )
        
        # Start training
        training_id = await self.training_service.start_training(
            detector_id=request.detector_id,
            dataset_id=training_config.dataset_id,
            config=training_config,
            trigger_type=request.trigger_type
        )
        
        response.training_id = training_id
        
        # Wait for training completion
        training_result = await self._wait_for_training_completion(training_id)
        response.training_result = training_result
        
        # Validate results
        if training_result and training_result.status.value == "completed":
            validation_passed = await self._validate_training_results(
                request.detector_id,
                training_result,
                response,
                config
            )
            response.validation_passed = validation_passed
            
            if validation_passed:
                response.status = "completed"
                response.message = "Retraining completed successfully"
                response.deployed = True
            else:
                response.status = "completed_validation_failed"
                response.message = "Retraining completed but validation failed"
        else:
            response.status = "failed"
            response.message = f"Training failed: {training_result.error_message if training_result else 'Unknown error'}"
        
        response.completed_at = datetime.utcnow()
        
        # Send notifications
        if config.notify_on_completion:
            await self._send_notification(
                "retraining_completed",
                f"Retraining completed for detector {request.detector_id}: {response.status}",
                config.notification_channels
            )
    
    async def _wait_for_training_completion(self, training_id: str) -> Optional[TrainingResult]:
        """Wait for training to complete."""
        max_wait_time = 3600  # 1 hour timeout
        poll_interval = 10  # 10 seconds
        waited = 0
        
        while waited < max_wait_time:
            result = await self.training_service.get_training_result(training_id)
            if result and result.status.value in ["completed", "failed", "cancelled"]:
                return result
            
            await asyncio.sleep(poll_interval)
            waited += poll_interval
        
        logger.error(f"Training {training_id} timed out after {max_wait_time} seconds")
        return None
    
    async def _validate_training_results(
        self,
        detector_id: UUID,
        training_result: TrainingResult,
        response: RetrainingResponse,
        config: RetrainingConfiguration
    ) -> bool:
        """Validate training results."""
        try:
            # Check if training was successful
            if training_result.status.value != "completed":
                return False
            
            # Check if performance improvement is required
            if config.require_performance_improvement:
                if not response.baseline_metrics:
                    logger.warning("No baseline metrics available for validation")
                    return True  # Allow if no baseline
                
                # Get new metrics
                if self.performance_monitor:
                    new_metrics = await self.performance_monitor.get_performance_metrics(
                        detector_id
                    )
                    response.new_metrics = new_metrics
                    
                    # Calculate improvement
                    if response.baseline_metrics and new_metrics:
                        improvement = self._calculate_improvement(
                            response.baseline_metrics,
                            new_metrics
                        )
                        response.improvement = improvement
                        
                        if improvement < config.improvement_threshold:
                            logger.warning(
                                f"Performance improvement {improvement} < {config.improvement_threshold}"
                            )
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating training results: {e}")
            return False
    
    def _calculate_improvement(
        self, 
        baseline: Dict[str, float], 
        new_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall performance improvement."""
        # Simple average improvement calculation
        # In practice, this would be more sophisticated
        improvements = []
        
        for metric_name, baseline_value in baseline.items():
            if metric_name in new_metrics:
                new_value = new_metrics[metric_name]
                if baseline_value > 0:
                    improvement = (new_value - baseline_value) / baseline_value
                    improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    async def _send_notification(
        self,
        event_type: str,
        message: str,
        channels: List[str]
    ) -> None:
        """Send notification about retraining events."""
        if not self.notification_service:
            logger.info(f"Notification: {event_type} - {message}")
            return
        
        try:
            for channel in channels:
                await self.notification_service.send_notification(
                    channel=channel,
                    event_type=event_type,
                    message=message
                )
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def stop_monitoring(self) -> None:
        """Stop all monitoring tasks."""
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            *self.monitoring_tasks.values(),
            return_exceptions=True
        )
        
        self.monitoring_tasks.clear()
        logger.info("Stopped all monitoring tasks")
