"""Performance degradation monitoring service for continuous threshold monitoring."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Protocol, Any
from uuid import uuid4

# TODO: Implement within data platform science domain - from packages.data_science.domain.entities.model_performance_degradation import (
    ModelPerformanceDegradation,
    DegradationStatus,
    RecoveryAction,
)
# TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
)
# TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    PerformanceDegradationMetrics,
    DegradationMetricType,
    DegradationSeverity,
)
# TODO: Implement within data platform science domain - from packages.data_science.domain.services.performance_baseline_service import (
    PerformanceBaselineService,
)
# TODO: Implement within data platform science domain - from packages.data_science.domain.services.performance_history_service import (
    PerformanceHistoryService,
)
from packages.core.domain.entities.alert import (
    Alert,
    AlertType,
    AlertSeverity,
    AlertCondition,
)


class ModelMetricsRepository(Protocol):
    """Protocol for model metrics repository."""
    
    async def get_latest_metrics(self, model_id: str) -> Optional[ModelPerformanceMetrics]:
        """Get latest performance metrics for a model."""
        ...
    
    async def get_historical_metrics(
        self, 
        model_id: str, 
        days_back: int = 30
    ) -> List[ModelPerformanceMetrics]:
        """Get historical performance metrics for a model."""
        ...


class DegradationRepository(Protocol):
    """Protocol for degradation monitoring repository."""
    
    async def save(self, degradation: ModelPerformanceDegradation) -> None:
        """Save degradation monitoring entity."""
        ...
    
    async def get_by_model_id(self, model_id: str) -> Optional[ModelPerformanceDegradation]:
        """Get degradation monitoring by model ID."""
        ...
    
    async def get_all_active(self) -> List[ModelPerformanceDegradation]:
        """Get all active degradation monitoring entities."""
        ...
    
    async def update(self, degradation: ModelPerformanceDegradation) -> None:
        """Update degradation monitoring entity."""
        ...


class AlertService(Protocol):
    """Protocol for alert service."""
    
    async def create_alert(self, alert: Alert) -> None:
        """Create a new alert."""
        ...
    
    async def trigger_alert(self, alert_id: str, context: Dict[str, Any]) -> None:
        """Trigger an existing alert."""
        ...


class NotificationService(Protocol):
    """Protocol for notification service."""
    
    async def send_degradation_notification(
        self,
        model_id: str,
        degradation_summary: Dict[str, Any],
        recipients: List[str],
    ) -> None:
        """Send degradation notification."""
        ...


class PerformanceDegradationMonitoringService:
    """Service for continuous monitoring of model performance degradation.
    
    This service continuously monitors model performance against established
    baselines and thresholds, triggering alerts and recovery actions when
    degradation is detected.
    """
    
    def __init__(
        self,
        degradation_repository: DegradationRepository,
        metrics_repository: ModelMetricsRepository,
        baseline_service: PerformanceBaselineService,
        alert_service: AlertService,
        notification_service: NotificationService,
        history_service: Optional[PerformanceHistoryService] = None,
        monitoring_interval_minutes: int = 15,
    ):
        """Initialize the monitoring service.
        
        Args:
            degradation_repository: Repository for degradation entities
            metrics_repository: Repository for performance metrics
            baseline_service: Service for baseline calculations
            alert_service: Service for alert management
            notification_service: Service for notifications
            history_service: Service for performance history tracking
            monitoring_interval_minutes: How often to check for degradation
        """
        self.degradation_repository = degradation_repository
        self.metrics_repository = metrics_repository
        self.baseline_service = baseline_service
        self.alert_service = alert_service
        self.notification_service = notification_service
        self.history_service = history_service
        self.monitoring_interval_minutes = monitoring_interval_minutes
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._is_running = False
    
    async def start_monitoring(self) -> None:
        """Start the monitoring service."""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start monitoring all active degradation entities
        active_degradations = await self.degradation_repository.get_all_active()
        
        for degradation in active_degradations:
            if degradation.monitoring_enabled:
                await self._start_model_monitoring(degradation)
        
        # Start the main monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring service."""
        self._is_running = False
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        self._monitoring_tasks.clear()
    
    async def setup_degradation_monitoring(
        self,
        model_id: str,
        model_name: str,
        model_version: str,
        task_type: str,
        degradation_thresholds: Dict[str, float],
        auto_recovery_enabled: bool = False,
        notification_recipients: List[str] = None,
    ) -> ModelPerformanceDegradation:
        """Setup degradation monitoring for a model.
        
        Args:
            model_id: ID of the model to monitor
            model_name: Name of the model
            model_version: Version of the model
            task_type: Type of ML task
            degradation_thresholds: Thresholds for each metric type
            auto_recovery_enabled: Whether to enable automatic recovery
            notification_recipients: List of notification recipients
        
        Returns:
            Created degradation monitoring entity
        """
        # Check if monitoring already exists
        existing = await self.degradation_repository.get_by_model_id(model_id)
        if existing:
            raise ValueError(f"Monitoring already exists for model {model_id}")
        
        # Get historical metrics to establish baseline
        historical_metrics = await self.metrics_repository.get_historical_metrics(
            model_id, days_back=30
        )
        
        if not historical_metrics:
            raise ValueError(f"No historical metrics found for model {model_id}")
        
        # Establish baseline
        # TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.model_performance_metrics import ModelTask
        task_enum = ModelTask(task_type)
        
        baseline_metrics = self.baseline_service.establish_baseline(
            historical_metrics, 
            baseline_method="recent_average"
        )
        
        # Create degradation metrics
        degradation_metrics = []
        for metric_type_str, threshold_pct in degradation_thresholds.items():
            metric_type = DegradationMetricType(metric_type_str)
            baseline_value = self.baseline_service._extract_metric_value(
                baseline_metrics, metric_type
            )
            
            if baseline_value is not None:
                # Calculate threshold value based on percentage
                if metric_type in [
                    DegradationMetricType.MSE_INCREASE,
                    DegradationMetricType.RMSE_INCREASE,
                    DegradationMetricType.MAE_INCREASE,
                    DegradationMetricType.PREDICTION_TIME_INCREASE,
                ]:
                    threshold_value = baseline_value * (1 + threshold_pct / 100)
                else:
                    threshold_value = baseline_value * (1 - threshold_pct / 100)
                
                degradation_metric = PerformanceDegradationMetrics(
                    metric_type=metric_type,
                    threshold_value=threshold_value,
                    baseline_value=baseline_value,
                    alert_enabled=True,
                )
                degradation_metrics.append(degradation_metric)
        
        # Create degradation monitoring entity
        degradation = ModelPerformanceDegradation(
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            task_type=task_enum,
            baseline_metrics=baseline_metrics,
            degradation_metrics=degradation_metrics,
            auto_recovery_enabled=auto_recovery_enabled,
            notification_settings={
                "recipients": notification_recipients or [],
                "enabled": True,
            },
        )
        
        await self.degradation_repository.save(degradation)
        
        # Start monitoring this model if the service is running
        if self._is_running:
            await self._start_model_monitoring(degradation)
        
        return degradation
    
    async def evaluate_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Manually evaluate model performance for degradation.
        
        Args:
            model_id: ID of the model to evaluate
        
        Returns:
            Evaluation results
        """
        degradation = await self.degradation_repository.get_by_model_id(model_id)
        if not degradation:
            raise ValueError(f"No degradation monitoring found for model {model_id}")
        
        # Get latest metrics
        latest_metrics = await self.metrics_repository.get_latest_metrics(model_id)
        if not latest_metrics:
            return {"error": "No latest metrics found", "model_id": model_id}
        
        # Evaluate performance
        evaluation_result = degradation.evaluate_performance(latest_metrics)
        
        # Update the entity
        await self.degradation_repository.update(degradation)
        
        # Record history if service is available
        if self.history_service:
            await self.history_service.record_degradation_event(
                degradation, evaluation_result, latest_metrics
            )
        
        # Handle alerts if needed
        if evaluation_result.get("should_alert", False):
            await self._handle_degradation_alert(degradation, evaluation_result)
        
        # Handle auto-recovery if enabled
        if (degradation.auto_recovery_enabled and 
            degradation.status in [DegradationStatus.DEGRADED, DegradationStatus.CRITICAL]):
            await self._handle_auto_recovery(degradation, evaluation_result)
        
        return evaluation_result
    
    async def update_baseline(self, model_id: str, baseline_method: str = "recent_average") -> None:
        """Update the baseline for a model.
        
        Args:
            model_id: ID of the model
            baseline_method: Method for baseline calculation
        """
        degradation = await self.degradation_repository.get_by_model_id(model_id)
        if not degradation:
            raise ValueError(f"No degradation monitoring found for model {model_id}")
        
        historical_metrics = await self.metrics_repository.get_historical_metrics(
            model_id, days_back=30
        )
        
        if not historical_metrics:
            raise ValueError(f"No historical metrics found for model {model_id}")
        
        new_baseline = self.baseline_service.establish_baseline(
            historical_metrics, baseline_method=baseline_method
        )
        
        degradation.update_baseline_metrics(new_baseline)
        await self.degradation_repository.update(degradation)
    
    async def enable_monitoring(self, model_id: str) -> None:
        """Enable monitoring for a model."""
        degradation = await self.degradation_repository.get_by_model_id(model_id)
        if not degradation:
            raise ValueError(f"No degradation monitoring found for model {model_id}")
        
        degradation.enable_monitoring()
        await self.degradation_repository.update(degradation)
        
        if self._is_running:
            await self._start_model_monitoring(degradation)
    
    async def disable_monitoring(self, model_id: str) -> None:
        """Disable monitoring for a model."""
        degradation = await self.degradation_repository.get_by_model_id(model_id)
        if not degradation:
            raise ValueError(f"No degradation monitoring found for model {model_id}")
        
        degradation.disable_monitoring()
        await self.degradation_repository.update(degradation)
        
        # Stop monitoring task
        if model_id in self._monitoring_tasks:
            self._monitoring_tasks[model_id].cancel()
            del self._monitoring_tasks[model_id]
    
    async def trigger_recovery_action(
        self, 
        model_id: str, 
        action: RecoveryAction, 
        initiated_by: str,
        context: Dict[str, Any] = None
    ) -> None:
        """Manually trigger a recovery action.
        
        Args:
            model_id: ID of the model
            action: Recovery action to trigger
            initiated_by: Who initiated the action
            context: Additional context for the action
        """
        degradation = await self.degradation_repository.get_by_model_id(model_id)
        if not degradation:
            raise ValueError(f"No degradation monitoring found for model {model_id}")
        
        degradation.trigger_recovery_action(action, initiated_by, context)
        await self.degradation_repository.update(degradation)
        
        # Execute the recovery action
        success = False
        try:
            await self._execute_recovery_action(degradation, action, context or {})
            success = True
        except Exception as e:
            print(f"Recovery action failed: {e}")
            success = False
        
        # Record recovery action in history if service is available
        if self.history_service:
            await self.history_service.record_recovery_action(
                model_id, action, initiated_by, success, context
            )
    
    async def get_monitoring_status(self, model_id: str) -> Dict[str, Any]:
        """Get monitoring status for a model.
        
        Args:
            model_id: ID of the model
        
        Returns:
            Monitoring status summary
        """
        degradation = await self.degradation_repository.get_by_model_id(model_id)
        if not degradation:
            return {"error": "No monitoring found", "model_id": model_id}
        
        return degradation.get_current_degradation_summary()
    
    async def list_all_monitoring(self) -> List[Dict[str, Any]]:
        """List all active monitoring."""
        active_degradations = await self.degradation_repository.get_all_active()
        return [d.get_current_degradation_summary() for d in active_degradations]
    
    async def get_performance_history(self, model_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Get performance history for a model.
        
        Args:
            model_id: ID of the model
            days_back: Number of days to look back
            
        Returns:
            Performance history data
        """
        if not self.history_service:
            return {"error": "History service not available"}
        
        timeline = await self.history_service.get_degradation_timeline(model_id, days_back)
        patterns = await self.history_service.analyze_degradation_patterns(model_id, days_back)
        stability = await self.history_service.get_performance_stability_score(model_id, days_back)
        
        return {
            "model_id": model_id,
            "timeline": timeline,
            "patterns": patterns,
            "stability": stability,
            "analysis_period_days": days_back,
        }
    
    async def get_stability_report(self, model_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Get stability report for a model.
        
        Args:
            model_id: ID of the model
            days_back: Number of days to analyze
            
        Returns:
            Stability report
        """
        if not self.history_service:
            return {"error": "History service not available"}
        
        return await self.history_service.get_performance_stability_score(model_id, days_back)
    
    async def compare_models(self, model_ids: List[str], days_back: int = 30) -> Dict[str, Any]:
        """Compare performance across multiple models.
        
        Args:
            model_ids: List of model IDs to compare
            days_back: Number of days to analyze
            
        Returns:
            Comparison results
        """
        if not self.history_service:
            return {"error": "History service not available"}
        
        return await self.history_service.compare_model_performance_history(model_ids, days_back)
    
    async def cleanup_history(self) -> int:
        """Clean up old history records.
        
        Returns:
            Number of records cleaned up
        """
        if not self.history_service:
            return 0
        
        return await self.history_service.cleanup_old_history()
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that checks all models periodically."""
        while self._is_running:
            try:
                # Get all active monitoring entities
                active_degradations = await self.degradation_repository.get_all_active()
                
                for degradation in active_degradations:
                    if (degradation.monitoring_enabled and 
                        degradation.is_due_for_evaluation()):
                        
                        try:
                            await self.evaluate_model_performance(degradation.model_id)
                        except Exception as e:
                            # Log error but continue monitoring other models
                            print(f"Error evaluating model {degradation.model_id}: {e}")
                
                # Wait before next iteration
                await asyncio.sleep(self.monitoring_interval_minutes * 60)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _start_model_monitoring(self, degradation: ModelPerformanceDegradation) -> None:
        """Start monitoring for a specific model."""
        model_id = degradation.model_id
        
        # Cancel existing task if any
        if model_id in self._monitoring_tasks:
            self._monitoring_tasks[model_id].cancel()
        
        # Create new monitoring task
        task = asyncio.create_task(self._model_monitoring_task(degradation))
        self._monitoring_tasks[model_id] = task
    
    async def _model_monitoring_task(self, degradation: ModelPerformanceDegradation) -> None:
        """Individual monitoring task for a model."""
        model_id = degradation.model_id
        
        while self._is_running and degradation.monitoring_enabled:
            try:
                if degradation.is_due_for_evaluation():
                    await self.evaluate_model_performance(model_id)
                
                # Wait based on the model's evaluation interval
                await asyncio.sleep(degradation.evaluation_interval_minutes * 60)
                
                # Refresh degradation entity
                degradation = await self.degradation_repository.get_by_model_id(model_id)
                if not degradation or not degradation.monitoring_enabled:
                    break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in model monitoring task for {model_id}: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _handle_degradation_alert(
        self, 
        degradation: ModelPerformanceDegradation,
        evaluation_result: Dict[str, Any]
    ) -> None:
        """Handle degradation alert creation and sending."""
        if not degradation.should_send_alert():
            return
        
        # Create alert
        severity_mapping = {
            DegradationSeverity.MINOR: AlertSeverity.LOW,
            DegradationSeverity.MODERATE: AlertSeverity.MEDIUM,
            DegradationSeverity.MAJOR: AlertSeverity.HIGH,
            DegradationSeverity.CRITICAL: AlertSeverity.CRITICAL,
        }
        
        max_severity = DegradationSeverity.MINOR
        degradations = evaluation_result.get("degradations", [])
        for deg in degradations:
            deg_severity = DegradationSeverity(deg["severity"])
            if deg_severity.value > max_severity.value:
                max_severity = deg_severity
        
        alert_condition = AlertCondition(
            metric_name="model_performance",
            operator="lt",
            threshold=0.0,  # Will be set dynamically
            description=f"Model {degradation.model_name} performance degradation detected"
        )
        
        alert = Alert(
            name=f"Performance Degradation - {degradation.model_name}",
            description=f"Performance degradation detected for model {degradation.model_name} (v{degradation.model_version})",
            alert_type=AlertType.MODEL_PERFORMANCE,
            severity=severity_mapping[max_severity],
            condition=alert_condition,
            created_by="degradation_monitoring_service",
            source="performance_monitoring",
            tags=[
                "model_performance",
                "degradation",
                degradation.model_id,
                degradation.task_type.value,
            ],
            metadata={
                "model_id": degradation.model_id,
                "model_name": degradation.model_name,
                "model_version": degradation.model_version,
                "task_type": degradation.task_type.value,
                "degradations": degradations,
                "evaluation_result": evaluation_result,
            }
        )
        
        await self.alert_service.create_alert(alert)
        
        # Send notifications
        if degradation.notification_settings.get("enabled", True):
            recipients = degradation.notification_settings.get("recipients", [])
            if recipients:
                await self.notification_service.send_degradation_notification(
                    degradation.model_id,
                    degradation.get_current_degradation_summary(),
                    recipients
                )
        
        # Mark alert as sent
        degradation.mark_alert_sent()
        await self.degradation_repository.update(degradation)
    
    async def _handle_auto_recovery(
        self,
        degradation: ModelPerformanceDegradation,
        evaluation_result: Dict[str, Any]
    ) -> None:
        """Handle automatic recovery actions."""
        recommended_actions = evaluation_result.get("recovery_actions_recommended", [])
        
        for action_str in recommended_actions:
            try:
                action = RecoveryAction(action_str)
                
                # Only trigger certain actions automatically
                if action in [
                    RecoveryAction.ADJUST_THRESHOLD,
                    RecoveryAction.DATA_QUALITY_CHECK,
                    RecoveryAction.INFRASTRUCTURE_SCALING,
                ]:
                    degradation.trigger_recovery_action(
                        action, 
                        "auto_recovery_system",
                        {"evaluation_result": evaluation_result}
                    )
                    
                    await self._execute_recovery_action(degradation, action, evaluation_result)
                    
            except ValueError:
                continue  # Skip invalid action strings
        
        await self.degradation_repository.update(degradation)
    
    async def _execute_recovery_action(
        self,
        degradation: ModelPerformanceDegradation,
        action: RecoveryAction,
        context: Dict[str, Any]
    ) -> None:
        """Execute a recovery action."""
        try:
            if action == RecoveryAction.ADJUST_THRESHOLD:
                await self._adjust_thresholds(degradation, context)
            elif action == RecoveryAction.DATA_QUALITY_CHECK:
                await self._trigger_data_quality_check(degradation, context)
            elif action == RecoveryAction.INFRASTRUCTURE_SCALING:
                await self._trigger_infrastructure_scaling(degradation, context)
            # Add more recovery actions as needed
            
            degradation.complete_recovery_action(action, True, "Recovery action completed successfully")
            
        except Exception as e:
            degradation.complete_recovery_action(action, False, f"Recovery action failed: {str(e)}")
            raise
    
    async def _adjust_thresholds(self, degradation: ModelPerformanceDegradation, context: Dict[str, Any]) -> None:
        """Adjust degradation thresholds automatically."""
        # Get recent performance data
        historical_metrics = await self.metrics_repository.get_historical_metrics(
            degradation.model_id, days_back=7
        )
        
        if len(historical_metrics) < 10:
            return  # Not enough data for adjustment
        
        # Suggest new thresholds based on recent variability
        suggested_thresholds = self.baseline_service.suggest_degradation_thresholds(
            historical_metrics, degradation.task_type
        )
        
        # Update thresholds (make them slightly more lenient)
        for metric in degradation.degradation_metrics:
            if metric.metric_type in suggested_thresholds:
                new_threshold_pct = suggested_thresholds[metric.metric_type] * 1.2  # 20% more lenient
                
                if metric.metric_type in [
                    DegradationMetricType.MSE_INCREASE,
                    DegradationMetricType.RMSE_INCREASE,
                    DegradationMetricType.MAE_INCREASE,
                    DegradationMetricType.PREDICTION_TIME_INCREASE,
                ]:
                    new_threshold = metric.baseline_value * (1 + new_threshold_pct / 100)
                else:
                    new_threshold = metric.baseline_value * (1 - new_threshold_pct / 100)
                
                metric.threshold_value = new_threshold
    
    async def _trigger_data_quality_check(self, degradation: ModelPerformanceDegradation, context: Dict[str, Any]) -> None:
        """Trigger a data quality check."""
        # This would integrate with a data quality service
        # For now, just log the action
        print(f"Triggering data quality check for model {degradation.model_id}")
    
    async def _trigger_infrastructure_scaling(self, degradation: ModelPerformanceDegradation, context: Dict[str, Any]) -> None:
        """Trigger infrastructure scaling."""
        # This would integrate with an infrastructure management service
        # For now, just log the action
        print(f"Triggering infrastructure scaling for model {degradation.model_id}")