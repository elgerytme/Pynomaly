"""Model performance monitoring API endpoints."""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from ...infrastructure.logging import get_logger
from ...infrastructure.repositories.model_repository import ModelRepository

logger = get_logger(__name__)
router = APIRouter(prefix="/performance", tags=["performance-monitoring"])


class PerformanceMetrics(BaseModel):
    """Performance metrics data model."""
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy")
    precision: float = Field(..., ge=0.0, le=1.0, description="Model precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Model recall")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="Model F1-score")
    auc_roc: Optional[float] = Field(None, ge=0.0, le=1.0, description="Area under ROC curve")
    response_time_ms: Optional[float] = Field(None, ge=0.0, description="Average response time in milliseconds")
    throughput: Optional[float] = Field(None, ge=0.0, description="Throughput in samples per second")
    memory_usage_mb: Optional[float] = Field(None, ge=0.0, description="Memory usage in MB")


class PerformanceAlert(BaseModel):
    """Performance alert data model."""
    alert_id: str
    model_id: str
    alert_type: str = Field(..., description="Type of alert (degradation, drift, error)")
    severity: str = Field(..., description="Alert severity (low, medium, high, critical)")
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False


class MonitoringConfig(BaseModel):
    """Monitoring configuration model."""
    model_id: str
    enabled: bool = True
    check_frequency_minutes: int = Field(default=30, ge=1, le=1440)
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "accuracy_degradation": 0.05,
            "precision_degradation": 0.05,
            "recall_degradation": 0.05,
            "f1_degradation": 0.05,
            "response_time_threshold": 1000.0,
            "memory_threshold": 1024.0
        }
    )
    notification_channels: List[str] = Field(default_factory=list)


class PerformanceReport(BaseModel):
    """Performance monitoring report model."""
    model_id: str
    report_period: Dict[str, datetime]
    current_metrics: PerformanceMetrics
    baseline_metrics: Optional[PerformanceMetrics]
    performance_trend: List[Dict[str, Any]]
    alerts_summary: Dict[str, int]
    drift_detected: bool
    recommendations: List[str]


class ThresholdOptimizationRequest(BaseModel):
    """Threshold optimization request model."""
    model_id: str
    evaluation_data: List[Dict[str, Any]]
    optimization_objective: str = Field(default="f1_score", description="Objective to optimize (f1_score, precision, recall, balanced)")
    threshold_range: Dict[str, float] = Field(default_factory=lambda: {"min": 0.01, "max": 0.99, "step": 0.01})


class ThresholdOptimizationResult(BaseModel):
    """Threshold optimization result model."""
    model_id: str
    optimal_threshold: float
    optimal_metrics: PerformanceMetrics
    threshold_analysis: List[Dict[str, Any]]
    optimization_objective: str
    improvement_percentage: float


# In-memory storage for demonstration (in production, use proper database)
_monitoring_configs: Dict[str, MonitoringConfig] = {}
_performance_history: Dict[str, List[Dict[str, Any]]] = {}
_active_alerts: Dict[str, List[PerformanceAlert]] = {}


@router.post("/monitor/configure/{model_id}")
async def configure_monitoring(
    model_id: str,
    config: MonitoringConfig,
    model_repository: ModelRepository = Depends()
) -> Dict[str, str]:
    """Configure performance monitoring for a model."""
    try:
        # Verify model exists
        model = model_repository.load(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Store configuration
        config.model_id = model_id
        _monitoring_configs[model_id] = config
        
        logger.info("Monitoring configured for model", model_id=model_id, config=config.dict())
        
        return {
            "message": f"Monitoring configured for model {model_id}",
            "status": "active" if config.enabled else "disabled"
        }
        
    except Exception as e:
        logger.error("Failed to configure monitoring", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to configure monitoring: {str(e)}")


@router.get("/monitor/config/{model_id}")
async def get_monitoring_config(model_id: str) -> MonitoringConfig:
    """Get monitoring configuration for a model."""
    if model_id not in _monitoring_configs:
        raise HTTPException(status_code=404, detail=f"No monitoring configuration found for model {model_id}")
    
    return _monitoring_configs[model_id]


@router.post("/monitor/metrics/{model_id}")
async def submit_performance_metrics(
    model_id: str,
    metrics: PerformanceMetrics,
    timestamp: Optional[datetime] = None
) -> Dict[str, str]:
    """Submit performance metrics for a model."""
    try:
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store metrics in history
        if model_id not in _performance_history:
            _performance_history[model_id] = []
        
        metric_entry = {
            "timestamp": timestamp.isoformat(),
            "metrics": metrics.dict()
        }
        
        _performance_history[model_id].append(metric_entry)
        
        # Keep only last 1000 entries
        if len(_performance_history[model_id]) > 1000:
            _performance_history[model_id] = _performance_history[model_id][-1000:]
        
        # Check for alerts if monitoring is configured
        if model_id in _monitoring_configs:
            await _check_performance_alerts(model_id, metrics, timestamp)
        
        logger.info("Performance metrics submitted", model_id=model_id, metrics=metrics.dict())
        
        return {"message": "Metrics submitted successfully", "timestamp": timestamp.isoformat()}
        
    except Exception as e:
        logger.error("Failed to submit metrics", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to submit metrics: {str(e)}")


@router.get("/monitor/metrics/{model_id}")
async def get_performance_history(
    model_id: str,
    hours: int = 24,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Get performance metrics history for a model."""
    if model_id not in _performance_history:
        return []
    
    # Filter by time range
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    filtered_history = [
        entry for entry in _performance_history[model_id]
        if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
    ]
    
    # Apply limit
    return filtered_history[-limit:]


@router.get("/monitor/report/{model_id}")
async def get_performance_report(
    model_id: str,
    hours: int = 24
) -> PerformanceReport:
    """Generate performance monitoring report for a model."""
    try:
        # Get recent metrics
        recent_metrics = await get_performance_history(model_id, hours)
        
        if not recent_metrics:
            raise HTTPException(status_code=404, detail=f"No performance data found for model {model_id}")
        
        # Calculate current metrics (latest entry)
        latest_entry = recent_metrics[-1]
        current_metrics = PerformanceMetrics(**latest_entry["metrics"])
        
        # Calculate baseline metrics (average of first 10 entries or all if less than 10)
        baseline_entries = recent_metrics[:min(10, len(recent_metrics))]
        if baseline_entries:
            baseline_dict = {}
            for key in baseline_entries[0]["metrics"].keys():
                values = [entry["metrics"][key] for entry in baseline_entries if entry["metrics"][key] is not None]
                if values:
                    baseline_dict[key] = sum(values) / len(values)
            baseline_metrics = PerformanceMetrics(**baseline_dict) if baseline_dict else None
        else:
            baseline_metrics = None
        
        # Get alerts summary
        model_alerts = _active_alerts.get(model_id, [])
        alerts_summary = {}
        for alert in model_alerts:
            if not alert.resolved:
                alerts_summary[alert.severity] = alerts_summary.get(alert.severity, 0) + 1
        
        # Detect drift (simplified)
        drift_detected = False
        if baseline_metrics:
            drift_detected = abs(current_metrics.accuracy - baseline_metrics.accuracy) > 0.05
        
        # Generate recommendations
        recommendations = _generate_performance_recommendations(current_metrics, baseline_metrics, model_alerts)
        
        return PerformanceReport(
            model_id=model_id,
            report_period={
                "start": recent_metrics[0]["timestamp"],
                "end": recent_metrics[-1]["timestamp"]
            },
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            performance_trend=recent_metrics,
            alerts_summary=alerts_summary,
            drift_detected=drift_detected,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate performance report", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/monitor/alerts/{model_id}")
async def get_model_alerts(
    model_id: str,
    resolved: Optional[bool] = None,
    severity: Optional[str] = None
) -> List[PerformanceAlert]:
    """Get alerts for a specific model."""
    model_alerts = _active_alerts.get(model_id, [])
    
    # Apply filters
    if resolved is not None:
        model_alerts = [alert for alert in model_alerts if alert.resolved == resolved]
    
    if severity is not None:
        model_alerts = [alert for alert in model_alerts if alert.severity == severity]
    
    return model_alerts


@router.post("/monitor/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> Dict[str, str]:
    """Resolve a performance alert."""
    for model_id, alerts in _active_alerts.items():
        for alert in alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info("Alert resolved", alert_id=alert_id, model_id=model_id)
                return {"message": f"Alert {alert_id} resolved successfully"}
    
    raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")


@router.post("/optimize/thresholds")
async def optimize_detection_thresholds(request: ThresholdOptimizationRequest) -> ThresholdOptimizationResult:
    """Optimize detection thresholds for a model."""
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        import numpy as np
        
        model_id = request.model_id
        evaluation_data = request.evaluation_data
        objective = request.optimization_objective
        
        if not evaluation_data:
            raise HTTPException(status_code=400, detail="Evaluation data is required")
        
        # Extract features and labels from evaluation data
        try:
            # Assume evaluation data has 'features' and 'label' keys
            X = np.array([item['features'] for item in evaluation_data])
            y_true = np.array([item['label'] for item in evaluation_data])
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Invalid evaluation data format: missing {str(e)}")
        
        # Load model and get prediction scores
        model_repository = ModelRepository()
        model = model_repository.load(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Get anomaly scores (this would depend on your model implementation)
        # For demonstration, we'll simulate scores
        scores = np.random.random(len(X))  # Replace with actual model scoring
        
        # Optimize threshold
        threshold_range = np.arange(
            request.threshold_range["min"],
            request.threshold_range["max"],
            request.threshold_range["step"]
        )
        
        best_threshold = None
        best_score = -1
        best_metrics = None
        threshold_analysis = []
        
        for threshold in threshold_range:
            y_pred = (scores >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate objective score
            if objective == "f1_score":
                current_score = f1
            elif objective == "precision":
                current_score = precision
            elif objective == "recall":
                current_score = recall
            elif objective == "balanced":
                current_score = (precision + recall) / 2
            else:
                current_score = f1  # Default to F1
            
            threshold_analysis.append({
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "objective_score": float(current_score)
            })
            
            if current_score > best_score:
                best_score = current_score
                best_threshold = threshold
                best_metrics = PerformanceMetrics(
                    accuracy=(np.sum(y_true == y_pred) / len(y_true)),
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    auc_roc=roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else None
                )
        
        # Calculate improvement
        baseline_metrics = threshold_analysis[len(threshold_analysis)//2]  # Use middle threshold as baseline
        improvement = ((best_score - baseline_metrics["objective_score"]) / baseline_metrics["objective_score"]) * 100 if baseline_metrics["objective_score"] > 0 else 0
        
        logger.info("Threshold optimization completed", 
                   model_id=model_id, 
                   optimal_threshold=best_threshold,
                   improvement=improvement)
        
        return ThresholdOptimizationResult(
            model_id=model_id,
            optimal_threshold=float(best_threshold),
            optimal_metrics=best_metrics,
            threshold_analysis=threshold_analysis,
            optimization_objective=objective,
            improvement_percentage=float(improvement)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Threshold optimization failed", model_id=request.model_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


async def _check_performance_alerts(model_id: str, metrics: PerformanceMetrics, timestamp: datetime):
    """Check for performance alerts based on configured thresholds."""
    if model_id not in _monitoring_configs:
        return
    
    config = _monitoring_configs[model_id]
    if not config.enabled:
        return
    
    thresholds = config.alert_thresholds
    new_alerts = []
    
    # Get baseline metrics (average of last 10 entries)
    if model_id in _performance_history and len(_performance_history[model_id]) >= 10:
        recent_entries = _performance_history[model_id][-10:]
        baseline_accuracy = sum(entry["metrics"]["accuracy"] for entry in recent_entries) / len(recent_entries)
        
        # Check for accuracy degradation
        accuracy_degradation = baseline_accuracy - metrics.accuracy
        if accuracy_degradation > thresholds.get("accuracy_degradation", 0.05):
            alert = PerformanceAlert(
                alert_id=f"{model_id}_accuracy_{int(timestamp.timestamp())}",
                model_id=model_id,
                alert_type="degradation",
                severity="high" if accuracy_degradation > 0.1 else "medium",
                message=f"Accuracy degradation detected: {accuracy_degradation:.3f}",
                metric_name="accuracy",
                current_value=metrics.accuracy,
                threshold_value=baseline_accuracy - thresholds["accuracy_degradation"],
                timestamp=timestamp
            )
            new_alerts.append(alert)
    
    # Check response time threshold
    if metrics.response_time_ms and metrics.response_time_ms > thresholds.get("response_time_threshold", 1000):
        alert = PerformanceAlert(
            alert_id=f"{model_id}_response_time_{int(timestamp.timestamp())}",
            model_id=model_id,
            alert_type="performance",
            severity="medium",
            message=f"High response time detected: {metrics.response_time_ms:.1f}ms",
            metric_name="response_time_ms",
            current_value=metrics.response_time_ms,
            threshold_value=thresholds["response_time_threshold"],
            timestamp=timestamp
        )
        new_alerts.append(alert)
    
    # Store new alerts
    if new_alerts:
        if model_id not in _active_alerts:
            _active_alerts[model_id] = []
        _active_alerts[model_id].extend(new_alerts)
        
        # Keep only last 100 alerts per model
        _active_alerts[model_id] = _active_alerts[model_id][-100:]


def _generate_performance_recommendations(
    current_metrics: PerformanceMetrics,
    baseline_metrics: Optional[PerformanceMetrics],
    alerts: List[PerformanceAlert]
) -> List[str]:
    """Generate performance recommendations based on metrics and alerts."""
    recommendations = []
    
    # Check for performance degradation
    if baseline_metrics:
        if current_metrics.accuracy < baseline_metrics.accuracy - 0.05:
            recommendations.append("Model accuracy has degraded significantly. Consider retraining with recent data.")
        
        if current_metrics.f1_score < baseline_metrics.f1_score - 0.05:
            recommendations.append("F1-score has decreased. Review data quality and model parameters.")
    
    # Check current performance levels
    if current_metrics.accuracy < 0.8:
        recommendations.append("Current accuracy is below 80%. Investigate data quality and model suitability.")
    
    if current_metrics.precision < 0.7:
        recommendations.append("Low precision indicates high false positive rate. Consider adjusting decision threshold.")
    
    if current_metrics.recall < 0.7:
        recommendations.append("Low recall indicates missing anomalies. Consider lowering detection threshold.")
    
    # Check for active alerts
    high_severity_alerts = [alert for alert in alerts if alert.severity == "high" and not alert.resolved]
    if high_severity_alerts:
        recommendations.append("High severity alerts detected. Immediate attention required.")
    
    # Performance optimization suggestions
    if current_metrics.response_time_ms and current_metrics.response_time_ms > 500:
        recommendations.append("Response time is high. Consider model optimization or resource scaling.")
    
    # Default recommendation if everything looks good
    if not recommendations:
        recommendations.append("Model performance appears stable. Continue regular monitoring.")
    
    return recommendations