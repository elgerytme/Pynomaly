"""Comprehensive model performance monitoring system for anomaly detection models."""

from __future__ import annotations

import time
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import json
import threading
import warnings
from contextlib import contextmanager

from ..logging import get_logger
from .metrics_collector import get_metrics_collector, ModelMetrics
from ...domain.entities.detection_result import DetectionResult

logger = get_logger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance metrics."""
    
    model_id: str
    algorithm: str
    timestamp: datetime
    
    # Accuracy metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    auc_roc: Optional[float] = None
    
    # Detection quality metrics
    anomaly_rate: float = 0.0
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    true_positive_rate: Optional[float] = None
    
    # Performance metrics
    prediction_time_ms: float = 0.0
    training_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Data quality metrics
    data_drift_score: Optional[float] = None
    concept_drift_score: Optional[float] = None
    prediction_confidence: Optional[float] = None
    
    # Business metrics
    samples_processed: int = 0
    predictions_made: int = 0
    anomalies_detected: int = 0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    model_version: Optional[str] = None
    dataset_size: Optional[int] = None
    feature_count: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    
    metric_name: str
    operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    threshold_value: float
    severity: str  # 'critical', 'warning', 'info'
    enabled: bool = True
    cooldown_minutes: int = 5


@dataclass
class PerformanceAlert:
    """Performance alert."""
    
    alert_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    message: str
    timestamp: datetime
    model_id: str
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


class ModelPerformanceMonitor:
    """Comprehensive model performance monitoring system."""
    
    def __init__(
        self,
        retention_hours: int = 168,  # 1 week
        alert_enabled: bool = True,
        metrics_buffer_size: int = 10000
    ):
        self.retention_hours = retention_hours
        self.alert_enabled = alert_enabled
        self.metrics_buffer_size = metrics_buffer_size
        
        # Storage
        self._metrics_buffer: deque = deque(maxlen=metrics_buffer_size)
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._alert_history: deque = deque(maxlen=1000)
        self._active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Alert configuration
        self._alert_thresholds: Dict[str, AlertThreshold] = {}
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Performance baselines
        self._performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Default alert thresholds
        self._setup_default_thresholds()
        
        logger.info("Model performance monitor initialized",
                   retention_hours=retention_hours,
                   alert_enabled=alert_enabled)
    
    def _setup_default_thresholds(self) -> None:
        """Setup default alert thresholds."""
        default_thresholds = [
            AlertThreshold("precision", "lt", 0.7, "warning"),
            AlertThreshold("recall", "lt", 0.7, "warning"),
            AlertThreshold("f1_score", "lt", 0.7, "warning"),
            AlertThreshold("false_positive_rate", "gt", 0.1, "warning"),
            AlertThreshold("prediction_time_ms", "gt", 5000, "warning"),
            AlertThreshold("memory_usage_mb", "gt", 1000, "critical"),
            AlertThreshold("data_drift_score", "gt", 0.3, "warning"),
            AlertThreshold("concept_drift_score", "gt", 0.3, "critical"),
        ]
        
        for threshold in default_thresholds:
            self._alert_thresholds[threshold.metric_name] = threshold
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._is_running:
            return
        
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info("Started model performance monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        self._is_running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped model performance monitoring")
    
    def record_prediction_metrics(
        self,
        model_id: str,
        algorithm: str,
        prediction_result: DetectionResult,
        prediction_time_ms: float,
        ground_truth: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """Record metrics for a prediction operation."""
        
        with self._lock:
            # Calculate basic metrics
            metrics = ModelPerformanceMetrics(
                model_id=model_id,
                algorithm=algorithm,
                timestamp=datetime.utcnow(),
                prediction_time_ms=prediction_time_ms,
                samples_processed=prediction_result.total_samples,
                predictions_made=prediction_result.total_samples,
                anomalies_detected=prediction_result.anomaly_count,
                anomaly_rate=prediction_result.anomaly_rate,
                **kwargs
            )
            
            # Calculate accuracy metrics if ground truth available
            if ground_truth is not None:
                self._calculate_accuracy_metrics(metrics, prediction_result, ground_truth)
            
            # Calculate prediction confidence if available
            if prediction_result.confidence_scores is not None:
                metrics.prediction_confidence = float(np.mean(prediction_result.confidence_scores))
            
            # Store metrics
            self._metrics_buffer.append(metrics)
            self._performance_history[model_id].append(metrics)
            
            # Check for alerts
            if self.alert_enabled:
                self._check_alerts(metrics)
            
            # Update performance baselines
            self._update_baselines(model_id, metrics)
            
            logger.debug("Recorded prediction metrics",
                        model_id=model_id,
                        algorithm=algorithm,
                        prediction_time_ms=prediction_time_ms,
                        anomalies_detected=metrics.anomalies_detected)
    
    def record_training_metrics(
        self,
        model_id: str,
        algorithm: str,
        training_time_ms: float,
        dataset_size: int,
        feature_count: int,
        **kwargs
    ) -> None:
        """Record metrics for a training operation."""
        
        with self._lock:
            metrics = ModelPerformanceMetrics(
                model_id=model_id,
                algorithm=algorithm,
                timestamp=datetime.utcnow(),
                training_time_ms=training_time_ms,
                dataset_size=dataset_size,
                feature_count=feature_count,
                **kwargs
            )
            
            self._metrics_buffer.append(metrics)
            self._performance_history[model_id].append(metrics)
            
            logger.info("Recorded training metrics",
                       model_id=model_id,
                       algorithm=algorithm,
                       training_time_ms=training_time_ms,
                       dataset_size=dataset_size)
    
    def _calculate_accuracy_metrics(
        self,
        metrics: ModelPerformanceMetrics,
        prediction_result: DetectionResult,
        ground_truth: np.ndarray
    ) -> None:
        """Calculate accuracy metrics using ground truth."""
        try:
            from sklearn.metrics import (
                precision_score, recall_score, f1_score, accuracy_score,
                roc_auc_score, confusion_matrix
            )
            
            y_pred = prediction_result.predictions
            y_true = ground_truth
            
            # Convert to binary format (1 for anomaly, 0 for normal)
            y_pred_binary = (y_pred == -1).astype(int)
            y_true_binary = (y_true == -1).astype(int)
            
            # Calculate metrics
            metrics.precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
            metrics.recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
            metrics.f1_score = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
            metrics.accuracy = float(accuracy_score(y_true_binary, y_pred_binary))
            
            # Calculate AUC-ROC if confidence scores available
            if prediction_result.confidence_scores is not None:
                try:
                    metrics.auc_roc = float(roc_auc_score(y_true_binary, prediction_result.confidence_scores))
                except ValueError:
                    pass  # Skip if not enough positive/negative samples
            
            # Calculate confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
            
            if tp + fn > 0:
                metrics.true_positive_rate = float(tp / (tp + fn))
            if fp + tn > 0:
                metrics.false_positive_rate = float(fp / (fp + tn))
            if fn + tp > 0:
                metrics.false_negative_rate = float(fn / (fn + tp))
                
        except ImportError:
            logger.warning("sklearn not available for accuracy metrics calculation")
        except Exception as e:
            logger.error("Error calculating accuracy metrics", error=str(e))
    
    def record_drift_metrics(
        self,
        model_id: str,
        data_drift_score: Optional[float] = None,
        concept_drift_score: Optional[float] = None,
        **kwargs
    ) -> None:
        """Record drift detection metrics."""
        
        with self._lock:
            metrics = ModelPerformanceMetrics(
                model_id=model_id,
                algorithm="drift_detection",
                timestamp=datetime.utcnow(),
                data_drift_score=data_drift_score,
                concept_drift_score=concept_drift_score,
                **kwargs
            )
            
            self._metrics_buffer.append(metrics)
            self._performance_history[model_id].append(metrics)
            
            # Check for drift alerts
            if self.alert_enabled:
                self._check_alerts(metrics)
            
            logger.info("Recorded drift metrics",
                       model_id=model_id,
                       data_drift_score=data_drift_score,
                       concept_drift_score=concept_drift_score)
    
    def _check_alerts(self, metrics: ModelPerformanceMetrics) -> None:
        """Check if metrics trigger any alerts."""
        
        for threshold_name, threshold in self._alert_thresholds.items():
            if not threshold.enabled:
                continue
            
            metric_value = getattr(metrics, threshold_name, None)
            if metric_value is None:
                metric_value = metrics.custom_metrics.get(threshold_name)
            
            if metric_value is None:
                continue
            
            # Check threshold condition
            triggered = False
            if threshold.operator == 'gt' and metric_value > threshold.threshold_value:
                triggered = True
            elif threshold.operator == 'lt' and metric_value < threshold.threshold_value:
                triggered = True
            elif threshold.operator == 'gte' and metric_value >= threshold.threshold_value:
                triggered = True
            elif threshold.operator == 'lte' and metric_value <= threshold.threshold_value:
                triggered = True
            elif threshold.operator == 'eq' and metric_value == threshold.threshold_value:
                triggered = True
            
            if triggered:
                self._trigger_alert(threshold_name, metric_value, threshold, metrics.model_id)
    
    def _trigger_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold: AlertThreshold,
        model_id: str
    ) -> None:
        """Trigger a performance alert."""
        
        alert_key = f"{model_id}_{metric_name}"
        
        # Check cooldown period
        if alert_key in self._active_alerts:
            last_alert = self._active_alerts[alert_key]
            cooldown_end = last_alert.timestamp + timedelta(minutes=threshold.cooldown_minutes)
            if datetime.utcnow() < cooldown_end:
                return
        
        # Create alert
        alert = PerformanceAlert(
            alert_id=f"{alert_key}_{int(time.time())}",
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold.threshold_value,
            severity=threshold.severity,
            message=f"Model {model_id}: {metric_name} is {current_value:.4f} (threshold: {threshold.operator} {threshold.threshold_value})",
            timestamp=datetime.utcnow(),
            model_id=model_id
        )
        
        # Store alert
        self._active_alerts[alert_key] = alert
        self._alert_history.append(alert)
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Error in alert callback", error=str(e))
        
        logger.warning("Performance alert triggered",
                      alert_id=alert.alert_id,
                      model_id=model_id,
                      metric_name=metric_name,
                      current_value=current_value,
                      threshold_value=threshold.threshold_value,
                      severity=threshold.severity)
    
    def _update_baselines(self, model_id: str, metrics: ModelPerformanceMetrics) -> None:
        """Update performance baselines for the model."""
        
        if model_id not in self._performance_baselines:
            self._performance_baselines[model_id] = {}
        
        baselines = self._performance_baselines[model_id]
        
        # Update rolling averages for key metrics
        key_metrics = [
            'precision', 'recall', 'f1_score', 'prediction_time_ms',
            'anomaly_rate', 'prediction_confidence'
        ]
        
        for metric_name in key_metrics:
            value = getattr(metrics, metric_name, None)
            if value is not None:
                current_avg = baselines.get(f"{metric_name}_avg", value)
                # Exponential moving average with alpha=0.1
                baselines[f"{metric_name}_avg"] = 0.9 * current_avg + 0.1 * value
    
    def get_model_performance_summary(self, model_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for a model."""
        
        with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_metrics = [
                m for m in self._performance_history[model_id]
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"model_id": model_id, "message": "No recent metrics available"}
            
            # Calculate aggregated metrics
            summary = {
                "model_id": model_id,
                "period_hours": hours,
                "total_predictions": sum(m.predictions_made for m in recent_metrics),
                "total_anomalies_detected": sum(m.anomalies_detected for m in recent_metrics),
                "avg_prediction_time_ms": np.mean([m.prediction_time_ms for m in recent_metrics if m.prediction_time_ms > 0]),
                "avg_anomaly_rate": np.mean([m.anomaly_rate for m in recent_metrics]),
                "metrics_count": len(recent_metrics),
                "first_metric_time": recent_metrics[0].timestamp.isoformat(),
                "last_metric_time": recent_metrics[-1].timestamp.isoformat()
            }
            
            # Add accuracy metrics if available
            accuracy_metrics = [m for m in recent_metrics if m.precision is not None]
            if accuracy_metrics:
                summary.update({
                    "avg_precision": np.mean([m.precision for m in accuracy_metrics]),
                    "avg_recall": np.mean([m.recall for m in accuracy_metrics]),
                    "avg_f1_score": np.mean([m.f1_score for m in accuracy_metrics]),
                    "avg_accuracy": np.mean([m.accuracy for m in accuracy_metrics if m.accuracy is not None])
                })
            
            # Add drift metrics if available
            drift_metrics = [m for m in recent_metrics if m.data_drift_score is not None]
            if drift_metrics:
                summary.update({
                    "avg_data_drift_score": np.mean([m.data_drift_score for m in drift_metrics]),
                    "max_data_drift_score": np.max([m.data_drift_score for m in drift_metrics])
                })
            
            # Add performance trends
            if len(recent_metrics) > 1:
                summary["performance_trends"] = self._calculate_trends(recent_metrics)
            
            return summary
    
    def _calculate_trends(self, metrics: List[ModelPerformanceMetrics]) -> Dict[str, str]:
        """Calculate performance trends."""
        
        trends = {}
        
        # Calculate trends for key metrics
        trend_metrics = ['precision', 'recall', 'f1_score', 'prediction_time_ms', 'anomaly_rate']
        
        for metric_name in trend_metrics:
            values = [getattr(m, metric_name) for m in metrics if getattr(m, metric_name) is not None]
            
            if len(values) >= 2:
                # Simple trend calculation
                first_half = np.mean(values[:len(values)//2])
                second_half = np.mean(values[len(values)//2:])
                
                if second_half > first_half * 1.05:
                    trends[metric_name] = "improving"
                elif second_half < first_half * 0.95:
                    trends[metric_name] = "degrading"
                else:
                    trends[metric_name] = "stable"
        
        return trends
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts."""
        
        with self._lock:
            return [alert for alert in self._active_alerts.values() if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        
        with self._lock:
            for alert in self._active_alerts.values():
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolution_timestamp = datetime.utcnow()
                    logger.info("Alert resolved", alert_id=alert_id)
                    return True
            
            return False
    
    def add_alert_threshold(self, threshold: AlertThreshold) -> None:
        """Add or update an alert threshold."""
        
        with self._lock:
            self._alert_thresholds[threshold.metric_name] = threshold
            logger.info("Alert threshold updated",
                       metric_name=threshold.metric_name,
                       threshold_value=threshold.threshold_value,
                       operator=threshold.operator)
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add a callback function for alerts."""
        
        self._alert_callbacks.append(callback)
        logger.info("Alert callback added")
    
    def export_metrics(self, output_path: Path, model_id: Optional[str] = None, hours: int = 24) -> None:
        """Export metrics to a file."""
        
        with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Filter metrics
            if model_id:
                metrics_to_export = [
                    m for m in self._performance_history[model_id]
                    if m.timestamp >= cutoff_time
                ]
            else:
                metrics_to_export = [
                    m for m in self._metrics_buffer
                    if m.timestamp >= cutoff_time
                ]
            
            # Convert to serializable format
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "model_id": model_id,
                "period_hours": hours,
                "metrics_count": len(metrics_to_export),
                "metrics": []
            }
            
            for metric in metrics_to_export:
                metric_dict = {
                    "model_id": metric.model_id,
                    "algorithm": metric.algorithm,
                    "timestamp": metric.timestamp.isoformat(),
                    "precision": metric.precision,
                    "recall": metric.recall,
                    "f1_score": metric.f1_score,
                    "accuracy": metric.accuracy,
                    "prediction_time_ms": metric.prediction_time_ms,
                    "anomaly_rate": metric.anomaly_rate,
                    "samples_processed": metric.samples_processed,
                    "custom_metrics": metric.custom_metrics
                }
                export_data["metrics"].append(metric_dict)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info("Metrics exported",
                       output_path=str(output_path),
                       metrics_count=len(metrics_to_export))
    
    def get_model_comparison(self, model_ids: List[str], hours: int = 24) -> Dict[str, Any]:
        """Compare performance across multiple models."""
        
        comparison = {
            "comparison_timestamp": datetime.utcnow().isoformat(),
            "period_hours": hours,
            "models": {}
        }
        
        for model_id in model_ids:
            summary = self.get_model_performance_summary(model_id, hours)
            comparison["models"][model_id] = summary
        
        # Add comparative analysis
        if len(model_ids) > 1:
            comparison["analysis"] = self._analyze_model_comparison(comparison["models"])
        
        return comparison
    
    def _analyze_model_comparison(self, models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comparison between models."""
        
        analysis = {}
        
        # Find best performing model for each metric
        metrics_to_compare = ['avg_precision', 'avg_recall', 'avg_f1_score', 'avg_prediction_time_ms']
        
        for metric in metrics_to_compare:
            values = {model_id: data.get(metric) for model_id, data in models.items()}
            values = {k: v for k, v in values.items() if v is not None}
            
            if values:
                if metric == 'avg_prediction_time_ms':  # Lower is better
                    best_model = min(values, key=values.get)
                else:  # Higher is better
                    best_model = max(values, key=values.get)
                
                analysis[f"best_{metric}"] = {
                    "model_id": best_model,
                    "value": values[best_model]
                }
        
        return analysis
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cleaning up old metrics."""
        
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
                
                with self._lock:
                    # Clean up metrics buffer
                    initial_size = len(self._metrics_buffer)
                    self._metrics_buffer = deque(
                        [m for m in self._metrics_buffer if m.timestamp >= cutoff_time],
                        maxlen=self.metrics_buffer_size
                    )
                    
                    # Clean up performance history
                    for model_id in self._performance_history:
                        history = self._performance_history[model_id]
                        cleaned_history = deque(
                            [m for m in history if m.timestamp >= cutoff_time],
                            maxlen=1000
                        )
                        self._performance_history[model_id] = cleaned_history
                    
                    # Clean up old alerts
                    self._alert_history = deque(
                        [a for a in self._alert_history if a.timestamp >= cutoff_time],
                        maxlen=1000
                    )
                    
                    cleaned_count = initial_size - len(self._metrics_buffer)
                    
                    if cleaned_count > 0:
                        logger.info("Cleaned up old metrics",
                                   cleaned_count=cleaned_count,
                                   retention_hours=self.retention_hours)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup worker", error=str(e))
    
    @contextmanager
    def monitor_prediction(self, model_id: str, algorithm: str):
        """Context manager for monitoring a prediction operation."""
        
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            # Record basic timing metrics
            metrics = ModelPerformanceMetrics(
                model_id=model_id,
                algorithm=algorithm,
                timestamp=datetime.utcnow(),
                prediction_time_ms=duration_ms
            )
            
            with self._lock:
                self._metrics_buffer.append(metrics)
                self._performance_history[model_id].append(metrics)


# Global instance
_model_performance_monitor: Optional[ModelPerformanceMonitor] = None


def get_model_performance_monitor() -> ModelPerformanceMonitor:
    """Get the global model performance monitor instance."""
    
    global _model_performance_monitor
    
    if _model_performance_monitor is None:
        _model_performance_monitor = ModelPerformanceMonitor()
    
    return _model_performance_monitor


def initialize_monitoring(
    retention_hours: int = 168,
    alert_enabled: bool = True,
    metrics_buffer_size: int = 10000
) -> ModelPerformanceMonitor:
    """Initialize the global model performance monitor."""
    
    global _model_performance_monitor
    
    _model_performance_monitor = ModelPerformanceMonitor(
        retention_hours=retention_hours,
        alert_enabled=alert_enabled,
        metrics_buffer_size=metrics_buffer_size
    )
    
    return _model_performance_monitor