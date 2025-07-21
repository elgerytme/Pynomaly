"""Model Performance Degradation domain entity for monitoring and alerting."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_entity import BaseEntity
# TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    PerformanceDegradationMetrics,
    DegradationSeverity,
    DegradationMetricType,
)
# TODO: Implement within data platform science domain - from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask,
)


class DegradationStatus(str, Enum):
    """Status of performance degradation monitoring."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    MONITORING_DISABLED = "monitoring_disabled"


class RecoveryAction(str, Enum):
    """Actions that can be taken for performance recovery."""
    
    RETRAIN_MODEL = "retrain_model"
    ADJUST_THRESHOLD = "adjust_threshold"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_QUALITY_CHECK = "data_quality_check"
    INFRASTRUCTURE_SCALING = "infrastructure_scaling"
    MODEL_ROLLBACK = "model_rollback"
    MANUAL_INTERVENTION = "manual_intervention"


class ModelPerformanceDegradation(BaseEntity):
    """Domain entity for detecting and managing model performance degradation.
    
    This entity monitors model performance against established baselines and
    thresholds, triggering alerts and recovery actions when degradation is detected.
    
    Attributes:
        model_id: ID of the model being monitored
        model_name: Name of the model for display purposes
        model_version: Version of the model being monitored
        task_type: Type of ML task (classification, regression, etc.)
        status: Current degradation status
        baseline_metrics: Baseline performance metrics for comparison
        current_metrics: Current performance metrics
        degradation_metrics: List of degradation metrics being monitored
        detection_history: History of degradation detections
        recovery_actions: List of recovery actions taken
        monitoring_enabled: Whether monitoring is currently enabled
        auto_recovery_enabled: Whether automatic recovery is enabled
        last_evaluation_at: When degradation was last evaluated
        degradation_detected_at: When degradation was first detected
        recovery_started_at: When recovery process was started
        recovery_completed_at: When recovery was completed
        alert_sent_at: When alert was last sent
        consecutive_healthy_evaluations: Number of consecutive healthy evaluations
        evaluation_interval_minutes: Interval between evaluations
        notification_settings: Settings for notifications
    """
    
    model_id: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)
    task_type: ModelTask
    status: DegradationStatus = Field(default=DegradationStatus.HEALTHY)
    
    # Performance metrics
    baseline_metrics: Optional[ModelPerformanceMetrics] = None
    current_metrics: Optional[ModelPerformanceMetrics] = None
    degradation_metrics: List[PerformanceDegradationMetrics] = Field(default_factory=list)
    
    # History and tracking
    detection_history: List[Dict[str, Any]] = Field(default_factory=list)
    recovery_actions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Configuration
    monitoring_enabled: bool = Field(default=True)
    auto_recovery_enabled: bool = Field(default=False)
    evaluation_interval_minutes: int = Field(default=60, gt=0)
    
    # Timestamps
    last_evaluation_at: Optional[datetime] = None
    degradation_detected_at: Optional[datetime] = None
    recovery_started_at: Optional[datetime] = None
    recovery_completed_at: Optional[datetime] = None
    alert_sent_at: Optional[datetime] = None
    
    # Counters
    consecutive_healthy_evaluations: int = Field(default=0, ge=0)
    
    # Settings
    notification_settings: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('degradation_metrics')
    def validate_degradation_metrics(cls, v: List[PerformanceDegradationMetrics]) -> List[PerformanceDegradationMetrics]:
        """Validate degradation metrics list."""
        if len(v) != len(set(metric.metric_type for metric in v)):
            raise ValueError("Duplicate degradation metric types not allowed")
        return v
    
    @validator('task_type')
    def validate_task_type_compatibility(cls, v: ModelTask, values: Dict[str, Any]) -> ModelTask:
        """Validate task type compatibility with metrics."""
        baseline_metrics = values.get('baseline_metrics')
        if baseline_metrics and baseline_metrics.task_type != v:
            raise ValueError("Task type must match baseline metrics task type")
        return v
    
    def add_degradation_metric(self, metric: PerformanceDegradationMetrics) -> None:
        """Add a new degradation metric to monitor."""
        # Remove existing metric of same type
        self.degradation_metrics = [
            m for m in self.degradation_metrics 
            if m.metric_type != metric.metric_type
        ]
        
        self.degradation_metrics.append(metric)
        self.mark_as_updated()
    
    def remove_degradation_metric(self, metric_type: DegradationMetricType) -> None:
        """Remove a degradation metric from monitoring."""
        self.degradation_metrics = [
            m for m in self.degradation_metrics 
            if m.metric_type != metric_type
        ]
        self.mark_as_updated()
    
    def update_baseline_metrics(self, new_baseline: ModelPerformanceMetrics) -> None:
        """Update the baseline metrics for comparison."""
        if new_baseline.task_type != self.task_type:
            raise ValueError("Baseline metrics task type must match model task type")
        
        self.baseline_metrics = new_baseline
        
        # Update threshold values in degradation metrics based on new baseline
        for metric in self.degradation_metrics:
            if metric.metric_type == DegradationMetricType.ACCURACY_DROP:
                baseline_accuracy = new_baseline.accuracy
                if baseline_accuracy:
                    metric.baseline_value = baseline_accuracy
            elif metric.metric_type == DegradationMetricType.MSE_INCREASE:
                baseline_mse = new_baseline.mse
                if baseline_mse:
                    metric.baseline_value = baseline_mse
            # Add more metric type mappings as needed
        
        self.mark_as_updated()
    
    def evaluate_performance(self, current_metrics: ModelPerformanceMetrics) -> Dict[str, Any]:
        """Evaluate current performance against baseline and thresholds."""
        if not self.monitoring_enabled:
            return {"status": "monitoring_disabled", "degradations": []}
        
        if not self.baseline_metrics:
            return {"status": "no_baseline", "degradations": []}
        
        self.current_metrics = current_metrics
        self.last_evaluation_at = datetime.utcnow()
        
        degradations = []
        overall_degraded = False
        max_severity = DegradationSeverity.MINOR
        
        # Evaluate each degradation metric
        for metric in self.degradation_metrics:
            current_value = self._extract_metric_value(current_metrics, metric.metric_type)
            
            if current_value is not None:
                updated_metric = metric.update_current_value(current_value)
                
                if updated_metric.is_degraded():
                    degradations.append({
                        "metric_type": metric.metric_type.value,
                        "severity": updated_metric.severity.value,
                        "degradation_percentage": updated_metric.calculate_degradation_percentage(),
                        "current_value": current_value,
                        "threshold_value": metric.threshold_value,
                        "baseline_value": metric.baseline_value,
                        "consecutive_breaches": updated_metric.consecutive_breaches,
                        "should_alert": updated_metric.should_alert(),
                        "alert_message": updated_metric.get_alert_message(),
                        "recovery_recommendation": updated_metric.get_recovery_recommendation(),
                    })
                    
                    overall_degraded = True
                    if updated_metric.severity.value > max_severity.value:
                        max_severity = updated_metric.severity
                
                # Update the metric in the list
                self.degradation_metrics = [
                    updated_metric if m.metric_type == metric.metric_type else m
                    for m in self.degradation_metrics
                ]
        
        # Update status based on evaluation
        previous_status = self.status
        
        if overall_degraded:
            if max_severity == DegradationSeverity.CRITICAL:
                self.status = DegradationStatus.CRITICAL
            else:
                self.status = DegradationStatus.DEGRADED
            
            # Record degradation detection
            if previous_status == DegradationStatus.HEALTHY:
                self.degradation_detected_at = datetime.utcnow()
                self.consecutive_healthy_evaluations = 0
            
            # Add to detection history
            self.detection_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "status": self.status.value,
                "degradations": degradations,
                "evaluation_id": str(self.id),
            })
            
        else:
            # Check if we're recovering
            if previous_status in [DegradationStatus.DEGRADED, DegradationStatus.CRITICAL]:
                self.consecutive_healthy_evaluations += 1
                
                # Consider recovered after several consecutive healthy evaluations
                if self.consecutive_healthy_evaluations >= 3:
                    self.status = DegradationStatus.HEALTHY
                    self.recovery_completed_at = datetime.utcnow()
                    self.degradation_detected_at = None
                else:
                    self.status = DegradationStatus.RECOVERING
            else:
                self.consecutive_healthy_evaluations += 1
                self.status = DegradationStatus.HEALTHY
        
        self.mark_as_updated()
        
        return {
            "status": self.status.value,
            "previous_status": previous_status.value,
            "degradations": degradations,
            "overall_severity": max_severity.value if overall_degraded else None,
            "consecutive_healthy_evaluations": self.consecutive_healthy_evaluations,
            "should_alert": any(d["should_alert"] for d in degradations),
            "recovery_actions_recommended": self._get_recovery_actions_recommended(degradations),
        }
    
    def _extract_metric_value(self, metrics: ModelPerformanceMetrics, metric_type: DegradationMetricType) -> Optional[float]:
        """Extract the specific metric value from performance metrics."""
        metric_mapping = {
            DegradationMetricType.ACCURACY_DROP: metrics.accuracy,
            DegradationMetricType.PRECISION_DROP: metrics.precision,
            DegradationMetricType.RECALL_DROP: metrics.recall,
            DegradationMetricType.F1_SCORE_DROP: metrics.f1_score,
            DegradationMetricType.AUC_DROP: metrics.roc_auc,
            DegradationMetricType.MSE_INCREASE: metrics.mse,
            DegradationMetricType.RMSE_INCREASE: metrics.rmse,
            DegradationMetricType.MAE_INCREASE: metrics.mae,
            DegradationMetricType.R2_SCORE_DROP: metrics.r2_score,
            DegradationMetricType.PREDICTION_TIME_INCREASE: metrics.prediction_time_seconds,
            DegradationMetricType.CONFIDENCE_DROP: metrics.prediction_confidence,
            DegradationMetricType.STABILITY_DECREASE: metrics.prediction_stability,
        }
        
        return metric_mapping.get(metric_type)
    
    def _get_recovery_actions_recommended(self, degradations: List[Dict[str, Any]]) -> List[str]:
        """Get list of recommended recovery actions based on degradations."""
        actions = set()
        
        for degradation in degradations:
            metric_type = degradation["metric_type"]
            severity = degradation["severity"]
            
            if severity == "critical":
                actions.add(RecoveryAction.RETRAIN_MODEL.value)
                actions.add(RecoveryAction.DATA_QUALITY_CHECK.value)
            elif severity == "major":
                actions.add(RecoveryAction.FEATURE_ENGINEERING.value)
                actions.add(RecoveryAction.ADJUST_THRESHOLD.value)
            
            # Specific recommendations based on metric type
            if "time" in metric_type:
                actions.add(RecoveryAction.INFRASTRUCTURE_SCALING.value)
            elif "accuracy" in metric_type or "precision" in metric_type:
                actions.add(RecoveryAction.DATA_QUALITY_CHECK.value)
        
        return list(actions)
    
    def trigger_recovery_action(self, action: RecoveryAction, initiated_by: str, context: Dict[str, Any] = None) -> None:
        """Trigger a recovery action."""
        if not self.auto_recovery_enabled and action != RecoveryAction.MANUAL_INTERVENTION:
            raise ValueError("Auto recovery is disabled")
        
        recovery_record = {
            "action": action.value,
            "initiated_by": initiated_by,
            "initiated_at": datetime.utcnow().isoformat(),
            "context": context or {},
            "status": "initiated",
        }
        
        self.recovery_actions.append(recovery_record)
        self.recovery_started_at = datetime.utcnow()
        self.mark_as_updated()
    
    def complete_recovery_action(self, action: RecoveryAction, success: bool, notes: str = "") -> None:
        """Mark a recovery action as completed."""
        for recovery in reversed(self.recovery_actions):
            if recovery["action"] == action.value and recovery["status"] == "initiated":
                recovery["status"] = "completed" if success else "failed"
                recovery["completed_at"] = datetime.utcnow().isoformat()
                recovery["success"] = success
                recovery["notes"] = notes
                break
        
        self.mark_as_updated()
    
    def enable_monitoring(self) -> None:
        """Enable degradation monitoring."""
        self.monitoring_enabled = True
        self.status = DegradationStatus.HEALTHY
        self.mark_as_updated()
    
    def disable_monitoring(self) -> None:
        """Disable degradation monitoring."""
        self.monitoring_enabled = False
        self.status = DegradationStatus.MONITORING_DISABLED
        self.mark_as_updated()
    
    def is_due_for_evaluation(self) -> bool:
        """Check if model is due for performance evaluation."""
        if not self.monitoring_enabled:
            return False
        
        if not self.last_evaluation_at:
            return True
        
        minutes_since_last = (datetime.utcnow() - self.last_evaluation_at).total_seconds() / 60
        return minutes_since_last >= self.evaluation_interval_minutes
    
    def should_send_alert(self) -> bool:
        """Check if an alert should be sent."""
        if not self.monitoring_enabled:
            return False
        
        if self.status not in [DegradationStatus.DEGRADED, DegradationStatus.CRITICAL]:
            return False
        
        # Check if alert was sent recently (prevent spam)
        if self.alert_sent_at:
            minutes_since_alert = (datetime.utcnow() - self.alert_sent_at).total_seconds() / 60
            if minutes_since_alert < 60:  # Don't send alerts more than once per hour
                return False
        
        return any(metric.should_alert() for metric in self.degradation_metrics)
    
    def mark_alert_sent(self) -> None:
        """Mark that an alert has been sent."""
        self.alert_sent_at = datetime.utcnow()
        self.mark_as_updated()
    
    def get_current_degradation_summary(self) -> Dict[str, Any]:
        """Get summary of current degradation state."""
        active_degradations = [
            metric for metric in self.degradation_metrics
            if metric.is_degraded()
        ]
        
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status.value,
            "monitoring_enabled": self.monitoring_enabled,
            "degradation_count": len(active_degradations),
            "degraded_metrics": [metric.metric_type.value for metric in active_degradations],
            "max_severity": max(
                (metric.severity.value for metric in active_degradations),
                default=DegradationSeverity.MINOR.value
            ),
            "last_evaluation_at": self.last_evaluation_at.isoformat() if self.last_evaluation_at else None,
            "degradation_detected_at": self.degradation_detected_at.isoformat() if self.degradation_detected_at else None,
            "should_alert": self.should_send_alert(),
            "consecutive_healthy_evaluations": self.consecutive_healthy_evaluations,
            "recovery_actions_taken": len(self.recovery_actions),
        }
    
    def get_performance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance degradation history."""
        return self.detection_history[-limit:]
    
    def validate_invariants(self) -> None:
        """Validate domain invariants."""
        super().validate_invariants()
        
        # Business rule: Cannot have current metrics without baseline
        if self.current_metrics and not self.baseline_metrics:
            raise ValueError("Cannot have current metrics without baseline metrics")
        
        # Business rule: Degradation metrics must be compatible with task type
        if self.baseline_metrics:
            for metric in self.degradation_metrics:
                if not self._is_metric_compatible_with_task(metric.metric_type, self.task_type):
                    raise ValueError(f"Metric {metric.metric_type.value} not compatible with task {self.task_type.value}")
    
    def _is_metric_compatible_with_task(self, metric_type: DegradationMetricType, task_type: ModelTask) -> bool:
        """Check if metric type is compatible with model task type."""
        classification_metrics = {
            DegradationMetricType.ACCURACY_DROP,
            DegradationMetricType.PRECISION_DROP,
            DegradationMetricType.RECALL_DROP,
            DegradationMetricType.F1_SCORE_DROP,
            DegradationMetricType.AUC_DROP,
        }
        
        regression_metrics = {
            DegradationMetricType.MSE_INCREASE,
            DegradationMetricType.RMSE_INCREASE,
            DegradationMetricType.MAE_INCREASE,
            DegradationMetricType.R2_SCORE_DROP,
        }
        
        general_metrics = {
            DegradationMetricType.STABILITY_DECREASE,
            DegradationMetricType.CONFIDENCE_DROP,
            DegradationMetricType.PREDICTION_TIME_INCREASE,
            DegradationMetricType.THROUGHPUT_DECREASE,
        }
        
        if task_type in [ModelTask.BINARY_CLASSIFICATION, ModelTask.MULTICLASS_CLASSIFICATION]:
            return metric_type in classification_metrics or metric_type in general_metrics
        elif task_type in [ModelTask.REGRESSION, ModelTask.TIME_SERIES]:
            return metric_type in regression_metrics or metric_type in general_metrics
        else:
            return metric_type in general_metrics