"""Service for detecting model performance degradation and triggering automated responses."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from pynomaly.domain.value_objects.model_metadata import ModelMetadata
from pynomaly.domain.entities.anomaly_detection_result import AnomalyDetectionResult
from pynomaly.infrastructure.monitoring.metrics_service import MetricsService
from pynomaly.infrastructure.monitoring.alerting_service import AlertingService
from pynomaly.application.services.performance_monitoring_service import PerformanceMonitoringService
from pynomaly.application.services.automl_service import AutoMLService
from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    PerformanceDegradationMetrics,
    DegradationSeverity,
    DegradationMetricType
)
from packages.data_science.domain.value_objects.model_performance_metrics import ModelPerformanceMetrics


class DegradationAction(str, Enum):
    """Actions to take when performance degradation is detected."""
    
    ALERT_ONLY = "alert_only"
    RETRAIN_MODEL = "retrain_model"
    SCALE_RESOURCES = "scale_resources"
    ROLLBACK_MODEL = "rollback_model"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class DegradationResponse:
    """Response to performance degradation detection."""
    
    action: DegradationAction
    severity: DegradationSeverity
    metric_type: DegradationMetricType
    degradation_percentage: float
    recommendation: str
    estimated_recovery_time: Optional[timedelta] = None
    auto_recovery_triggered: bool = False


class ModelPerformanceDegradationService:
    """Service for comprehensive model performance degradation detection and response.
    
    This service integrates existing monitoring components to provide unified
    performance degradation detection with automated response capabilities.
    """

    def __init__(
        self,
        metrics_service: MetricsService,
        alerting_service: AlertingService,
        performance_service: PerformanceMonitoringService,
        automl_service: AutoMLService
    ):
        self.metrics_service = metrics_service
        self.alerting_service = alerting_service
        self.performance_service = performance_service
        self.automl_service = automl_service
        self.logger = logging.getLogger(__name__)
        
        # Degradation thresholds by metric type
        self.degradation_thresholds = {
            DegradationMetricType.ACCURACY_DROP: {
                DegradationSeverity.MINOR: 0.05,    # 5% drop
                DegradationSeverity.MODERATE: 0.10,  # 10% drop
                DegradationSeverity.MAJOR: 0.15,     # 15% drop
                DegradationSeverity.CRITICAL: 0.25   # 25% drop
            },
            DegradationMetricType.F1_SCORE_DROP: {
                DegradationSeverity.MINOR: 0.05,
                DegradationSeverity.MODERATE: 0.10,
                DegradationSeverity.MAJOR: 0.15,
                DegradationSeverity.CRITICAL: 0.20
            },
            DegradationMetricType.PREDICTION_TIME_INCREASE: {
                DegradationSeverity.MINOR: 0.20,    # 20% increase
                DegradationSeverity.MODERATE: 0.50,  # 50% increase
                DegradationSeverity.MAJOR: 1.0,      # 100% increase
                DegradationSeverity.CRITICAL: 2.0    # 200% increase
            },
            DegradationMetricType.THROUGHPUT_DECREASE: {
                DegradationSeverity.MINOR: 0.15,    # 15% decrease
                DegradationSeverity.MODERATE: 0.30,  # 30% decrease
                DegradationSeverity.MAJOR: 0.50,     # 50% decrease
                DegradationSeverity.CRITICAL: 0.70   # 70% decrease
            }
        }
        
        # Action mappings by severity
        self.severity_actions = {
            DegradationSeverity.MINOR: DegradationAction.ALERT_ONLY,
            DegradationSeverity.MODERATE: DegradationAction.SCALE_RESOURCES,
            DegradationSeverity.MAJOR: DegradationAction.RETRAIN_MODEL,
            DegradationSeverity.CRITICAL: DegradationAction.MANUAL_INTERVENTION
        }

    async def monitor_model_performance(
        self,
        model_id: str,
        monitoring_window_hours: int = 24
    ) -> List[DegradationResponse]:
        """Monitor model performance and detect degradation over a time window.
        
        Args:
            model_id: Unique identifier for the model to monitor
            monitoring_window_hours: Time window for performance analysis
            
        Returns:
            List of degradation responses if any degradation is detected
        """
        try:
            self.logger.info(f"Starting performance monitoring for model {model_id}")
            
            # Get model metadata and current performance
            model_metrics = await self._get_current_model_metrics(model_id)
            baseline_metrics = await self._get_baseline_metrics(model_id)
            
            if not baseline_metrics:
                self.logger.warning(f"No baseline metrics found for model {model_id}")
                return []
            
            # Detect performance degradation
            degradations = await self._detect_performance_degradation(
                model_id, model_metrics, baseline_metrics
            )
            
            responses = []
            for degradation in degradations:
                response = await self._handle_performance_degradation(
                    model_id, degradation
                )
                responses.append(response)
            
            return responses
            
        except Exception as e:
            self.logger.error(f"Error monitoring model performance: {e}")
            return []

    async def _get_current_model_metrics(self, model_id: str) -> ModelPerformanceMetrics:
        """Get current performance metrics for a model."""
        try:
            # Get recent performance data from metrics service
            recent_metrics = await self.metrics_service.get_model_metrics(
                model_id=model_id,
                time_range=timedelta(hours=1)
            )
            
            # Convert to ModelPerformanceMetrics if needed
            if isinstance(recent_metrics, dict):
                return ModelPerformanceMetrics(**recent_metrics)
            
            return recent_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            raise

    async def _get_baseline_metrics(self, model_id: str) -> Optional[ModelPerformanceMetrics]:
        """Get baseline performance metrics for comparison."""
        try:
            # Get baseline from performance monitoring service
            baseline_data = await self.performance_service.get_performance_baseline(model_id)
            
            if baseline_data:
                return ModelPerformanceMetrics(**baseline_data)
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting baseline metrics: {e}")
            return None

    async def _detect_performance_degradation(
        self,
        model_id: str,
        current_metrics: ModelPerformanceMetrics,
        baseline_metrics: ModelPerformanceMetrics
    ) -> List[PerformanceDegradationMetrics]:
        """Detect performance degradation by comparing current and baseline metrics."""
        degradations = []
        
        try:
            # Check accuracy degradation
            if hasattr(current_metrics, 'accuracy') and hasattr(baseline_metrics, 'accuracy'):
                accuracy_degradation = self._check_metric_degradation(
                    DegradationMetricType.ACCURACY_DROP,
                    current_metrics.accuracy,
                    baseline_metrics.accuracy,
                    higher_is_better=True
                )
                if accuracy_degradation:
                    degradations.append(accuracy_degradation)
            
            # Check F1 score degradation
            if hasattr(current_metrics, 'f1_score') and hasattr(baseline_metrics, 'f1_score'):
                f1_degradation = self._check_metric_degradation(
                    DegradationMetricType.F1_SCORE_DROP,
                    current_metrics.f1_score,
                    baseline_metrics.f1_score,
                    higher_is_better=True
                )
                if f1_degradation:
                    degradations.append(f1_degradation)
            
            # Check prediction time degradation
            if hasattr(current_metrics, 'prediction_time_ms') and hasattr(baseline_metrics, 'prediction_time_ms'):
                latency_degradation = self._check_metric_degradation(
                    DegradationMetricType.PREDICTION_TIME_INCREASE,
                    current_metrics.prediction_time_ms,
                    baseline_metrics.prediction_time_ms,
                    higher_is_better=False
                )
                if latency_degradation:
                    degradations.append(latency_degradation)
            
            return degradations
            
        except Exception as e:
            self.logger.error(f"Error detecting performance degradation: {e}")
            return []

    def _check_metric_degradation(
        self,
        metric_type: DegradationMetricType,
        current_value: float,
        baseline_value: float,
        higher_is_better: bool = True
    ) -> Optional[PerformanceDegradationMetrics]:
        """Check if a specific metric shows degradation."""
        try:
            if higher_is_better:
                degradation_percentage = (baseline_value - current_value) / baseline_value
            else:
                degradation_percentage = (current_value - baseline_value) / baseline_value
            
            # Only consider positive degradation
            if degradation_percentage <= 0:
                return None
            
            # Determine severity
            severity = self._determine_degradation_severity(metric_type, degradation_percentage)
            
            if severity:
                return PerformanceDegradationMetrics(
                    metric_type=metric_type,
                    degradation_percentage=degradation_percentage,
                    severity=severity,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    threshold_breached=True,
                    consecutive_breaches=1,  # Will be updated by monitoring system
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking metric degradation: {e}")
            return None

    def _determine_degradation_severity(
        self,
        metric_type: DegradationMetricType,
        degradation_percentage: float
    ) -> Optional[DegradationSeverity]:
        """Determine severity level based on degradation percentage."""
        thresholds = self.degradation_thresholds.get(metric_type, {})
        
        if degradation_percentage >= thresholds.get(DegradationSeverity.CRITICAL, float('inf')):
            return DegradationSeverity.CRITICAL
        elif degradation_percentage >= thresholds.get(DegradationSeverity.MAJOR, float('inf')):
            return DegradationSeverity.MAJOR
        elif degradation_percentage >= thresholds.get(DegradationSeverity.MODERATE, float('inf')):
            return DegradationSeverity.MODERATE
        elif degradation_percentage >= thresholds.get(DegradationSeverity.MINOR, float('inf')):
            return DegradationSeverity.MINOR
        
        return None

    async def _handle_performance_degradation(
        self,
        model_id: str,
        degradation: PerformanceDegradationMetrics
    ) -> DegradationResponse:
        """Handle detected performance degradation with appropriate actions."""
        try:
            # Determine action based on severity
            action = self.severity_actions.get(
                degradation.severity,
                DegradationAction.ALERT_ONLY
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(degradation)
            
            # Create response
            response = DegradationResponse(
                action=action,
                severity=degradation.severity,
                metric_type=degradation.metric_type,
                degradation_percentage=degradation.degradation_percentage,
                recommendation=recommendation
            )
            
            # Execute action
            if action == DegradationAction.ALERT_ONLY:
                await self._send_degradation_alert(model_id, degradation)
                
            elif action == DegradationAction.SCALE_RESOURCES:
                await self._scale_model_resources(model_id, degradation)
                response.auto_recovery_triggered = True
                response.estimated_recovery_time = timedelta(minutes=10)
                
            elif action == DegradationAction.RETRAIN_MODEL:
                await self._trigger_model_retraining(model_id, degradation)
                response.auto_recovery_triggered = True
                response.estimated_recovery_time = timedelta(hours=2)
                
            elif action == DegradationAction.MANUAL_INTERVENTION:
                await self._escalate_to_manual_intervention(model_id, degradation)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling performance degradation: {e}")
            return DegradationResponse(
                action=DegradationAction.ALERT_ONLY,
                severity=degradation.severity,
                metric_type=degradation.metric_type,
                degradation_percentage=degradation.degradation_percentage,
                recommendation="Error handling degradation - manual review required"
            )

    def _generate_recommendation(self, degradation: PerformanceDegradationMetrics) -> str:
        """Generate actionable recommendations based on degradation type and severity."""
        base_recommendations = {
            DegradationMetricType.ACCURACY_DROP: [
                "Consider retraining with recent data",
                "Review data quality and feature drift",
                "Evaluate model complexity and overfitting"
            ],
            DegradationMetricType.F1_SCORE_DROP: [
                "Analyze class distribution changes",
                "Review threshold optimization",
                "Consider ensemble methods"
            ],
            DegradationMetricType.PREDICTION_TIME_INCREASE: [
                "Scale compute resources",
                "Optimize model architecture",
                "Consider model pruning or quantization"
            ],
            DegradationMetricType.THROUGHPUT_DECREASE: [
                "Increase replica count",
                "Optimize batch processing",
                "Review system bottlenecks"
            ]
        }
        
        recommendations = base_recommendations.get(degradation.metric_type, ["Review model performance"])
        
        severity_prefix = {
            DegradationSeverity.MINOR: "Monitor closely and",
            DegradationSeverity.MODERATE: "Take action soon to",
            DegradationSeverity.MAJOR: "Immediately",
            DegradationSeverity.CRITICAL: "URGENT:"
        }
        
        prefix = severity_prefix.get(degradation.severity, "")
        return f"{prefix} {recommendations[0]}. Degradation: {degradation.degradation_percentage:.1%}"

    async def _send_degradation_alert(self, model_id: str, degradation: PerformanceDegradationMetrics):
        """Send alert for performance degradation."""
        alert_title = f"Model Performance Degradation Detected - {model_id}"
        alert_message = (
            f"Model {model_id} shows {degradation.severity.value} degradation in "
            f"{degradation.metric_type.value}: {degradation.degradation_percentage:.1%}"
        )
        
        await self.alerting_service.send_alert(
            title=alert_title,
            message=alert_message,
            severity=degradation.severity.value,
            tags={"model_id": model_id, "metric_type": degradation.metric_type.value}
        )

    async def _scale_model_resources(self, model_id: str, degradation: PerformanceDegradationMetrics):
        """Scale model resources to address performance issues."""
        self.logger.info(f"Scaling resources for model {model_id} due to {degradation.metric_type.value}")
        
        # This would integrate with infrastructure scaling services
        # For now, log the action
        scaling_factor = {
            DegradationSeverity.MODERATE: 1.5,
            DegradationSeverity.MAJOR: 2.0
        }.get(degradation.severity, 1.2)
        
        self.logger.info(f"Triggering {scaling_factor}x resource scaling for model {model_id}")

    async def _trigger_model_retraining(self, model_id: str, degradation: PerformanceDegradationMetrics):
        """Trigger automated model retraining."""
        self.logger.info(f"Triggering retraining for model {model_id} due to {degradation.metric_type.value}")
        
        try:
            # Get model configuration for retraining
            retraining_config = await self._get_retraining_config(model_id)
            
            # Trigger retraining through AutoML service
            await self.automl_service.retrain_model(
                model_id=model_id,
                config=retraining_config,
                reason=f"Performance degradation: {degradation.metric_type.value}"
            )
            
            self.logger.info(f"Retraining initiated for model {model_id}")
            
        except Exception as e:
            self.logger.error(f"Error triggering retraining: {e}")
            # Fallback to alert
            await self._send_degradation_alert(model_id, degradation)

    async def _escalate_to_manual_intervention(self, model_id: str, degradation: PerformanceDegradationMetrics):
        """Escalate critical degradation to manual intervention."""
        self.logger.critical(f"Critical degradation detected for model {model_id} - manual intervention required")
        
        # Send high-priority alert
        await self.alerting_service.send_alert(
            title=f"CRITICAL: Manual Intervention Required - {model_id}",
            message=(
                f"Critical performance degradation detected in {degradation.metric_type.value}: "
                f"{degradation.degradation_percentage:.1%}. Immediate manual review required."
            ),
            severity="critical",
            priority="high",
            tags={"model_id": model_id, "requires_intervention": "true"}
        )

    async def _get_retraining_config(self, model_id: str) -> Dict[str, Any]:
        """Get retraining configuration for a model."""
        # This would retrieve stored model configuration
        # For now, return default configuration
        return {
            "use_latest_data": True,
            "validation_split": 0.2,
            "early_stopping": True,
            "max_epochs": 100,
            "performance_threshold": 0.8
        }

    async def start_continuous_monitoring(self, model_ids: List[str], check_interval_minutes: int = 30):
        """Start continuous performance monitoring for specified models."""
        self.logger.info(f"Starting continuous monitoring for {len(model_ids)} models")
        
        while True:
            try:
                tasks = [
                    self.monitor_model_performance(model_id)
                    for model_id in model_ids
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for model_id, result in zip(model_ids, results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Error monitoring model {model_id}: {result}")
                    elif result:
                        self.logger.info(f"Detected {len(result)} degradations for model {model_id}")
                
                # Wait for next check
                await asyncio.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying