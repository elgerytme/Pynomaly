"""AI-powered auto-scaling engine for intelligent resource management."""

from __future__ import annotations

import asyncio
import json
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import threading

import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ....infrastructure.logging import get_logger

logger = get_logger(__name__)

# Lazy import metrics collector to avoid None issues
def get_safe_metrics_collector():
    """Get metrics collector with safe fallback."""
    try:
        from ....infrastructure.monitoring import get_metrics_collector
        return get_metrics_collector()
    except Exception:
        class MockMetricsCollector:
            def record_metric(self, *args, **kwargs):
                pass
        return MockMetricsCollector()


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"    # Horizontal scaling up (add instances)
    SCALE_IN = "scale_in"      # Horizontal scaling down (remove instances)
    NO_ACTION = "no_action"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    INSTANCES = "instances"
    THREADS = "threads"
    CONNECTIONS = "connections"


class ScalingStrategy(Enum):
    """Scaling strategies."""
    REACTIVE = "reactive"          # React to current metrics
    PREDICTIVE = "predictive"      # Predict future needs
    HYBRID = "hybrid"              # Combination of both
    REINFORCEMENT = "reinforcement" # Learn from past actions


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    network_io_mbps: float
    active_connections: int
    queue_length: int
    response_time_ms: float
    throughput_requests_per_second: float
    error_rate_percent: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingDecision:
    """Scaling decision with rationale."""
    timestamp: datetime
    action: ScalingAction
    resource_type: ResourceType
    current_value: float
    target_value: float
    confidence: float  # 0-1
    reasoning: str
    expected_impact: str
    estimated_cost_change: float  # Positive for increase, negative for decrease
    urgency: float  # 0-1, higher means more urgent
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingEvent:
    """Record of a scaling action taken."""
    timestamp: datetime
    decision: ScalingDecision
    execution_status: str  # "success", "failed", "partial"
    actual_impact: Optional[Dict[str, float]] = None
    cost_impact: Optional[float] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None


class PredictiveModel:
    """Predictive model for forecasting resource needs."""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.last_trained = None
        self.prediction_horizon_minutes = 30
        self.training_data_buffer = deque(maxlen=1000)
        self.model_accuracy = 0.0
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "linear":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def add_training_data(self, metrics: ResourceMetrics, target_values: Dict[str, float]):
        """Add data point for model training."""
        features = self._extract_features(metrics)
        self.training_data_buffer.append({
            "features": features,
            "targets": target_values,
            "timestamp": metrics.timestamp
        })
    
    def train_model(self, target_metric: str = "cpu_usage_percent") -> float:
        """Train the predictive model."""
        if len(self.training_data_buffer) < 50:  # Need minimum data
            return 0.0
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for data_point in self.training_data_buffer:
                if target_metric in data_point["targets"]:
                    X.append(data_point["features"])
                    y.append(data_point["targets"][target_metric])
            
            if len(X) < 10:
                return 0.0
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Calculate accuracy using cross-validation-like approach
            mid_point = len(X) // 2
            X_train, X_test = X_scaled[:mid_point], X_scaled[mid_point:]
            y_train, y_test = y[:mid_point], y[mid_point:]
            
            if len(X_test) > 0:
                temp_model = type(self.model)(**self.model.get_params() if hasattr(self.model, 'get_params') else {})
                temp_model.fit(X_train, y_train)
                predictions = temp_model.predict(X_test)
                
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                
                # Calculate accuracy as 1 - normalized MAE
                max_val = max(np.max(y_test), np.max(predictions))
                min_val = min(np.min(y_test), np.min(predictions))
                normalized_mae = mae / (max_val - min_val) if max_val != min_val else 0
                
                self.model_accuracy = max(0, 1 - normalized_mae)
            else:
                self.model_accuracy = 0.0
            
            self.last_trained = datetime.utcnow()
            
            logger.info(f"Model trained for {target_metric}",
                       accuracy=self.model_accuracy,
                       training_samples=len(X))
            
            return self.model_accuracy
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return 0.0
    
    def predict(self, metrics: ResourceMetrics, steps_ahead: int = 1) -> Dict[str, float]:
        """Predict future resource utilization."""
        if self.model is None or self.last_trained is None:
            return {}
        
        try:
            features = self._extract_features(metrics)
            features_scaled = self.scaler.transform([features])
            
            # For multi-step prediction, we'd typically use a more sophisticated approach
            # For now, use single-step prediction
            prediction = self.model.predict(features_scaled)[0]
            
            return {
                "predicted_value": float(prediction),
                "confidence": self.model_accuracy,
                "prediction_time": datetime.utcnow().isoformat(),
                "steps_ahead": steps_ahead
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {}
    
    def _extract_features(self, metrics: ResourceMetrics) -> List[float]:
        """Extract features from metrics for model input."""
        # Time-based features
        timestamp = metrics.timestamp
        hour_of_day = timestamp.hour / 24.0
        day_of_week = timestamp.weekday() / 7.0
        
        # Resource utilization features
        features = [
            metrics.cpu_usage_percent / 100.0,
            metrics.memory_usage_percent / 100.0,
            metrics.disk_usage_percent / 100.0,
            metrics.network_io_mbps / 1000.0,  # Normalize to GB
            metrics.active_connections / 1000.0,  # Normalize
            metrics.queue_length / 100.0,  # Normalize
            metrics.response_time_ms / 1000.0,  # Convert to seconds
            metrics.throughput_requests_per_second / 100.0,  # Normalize
            metrics.error_rate_percent / 100.0,
            hour_of_day,
            day_of_week
        ]
        
        # Add custom metrics
        for value in metrics.custom_metrics.values():
            features.append(value)
        
        return features


class AutoScalingEngine:
    """AI-powered auto-scaling engine with predictive capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize auto-scaling engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Scaling configuration
        self.scaling_strategy = ScalingStrategy(
            self.config.get("scaling_strategy", "hybrid")
        )
        self.min_scaling_interval_seconds = self.config.get("min_scaling_interval", 300)  # 5 minutes
        self.cooldown_period_seconds = self.config.get("cooldown_period", 600)  # 10 minutes
        
        # Thresholds
        self.cpu_scale_up_threshold = self.config.get("cpu_scale_up_threshold", 80.0)
        self.cpu_scale_down_threshold = self.config.get("cpu_scale_down_threshold", 30.0)
        self.memory_scale_up_threshold = self.config.get("memory_scale_up_threshold", 85.0)
        self.memory_scale_down_threshold = self.config.get("memory_scale_down_threshold", 40.0)
        
        # Resource limits
        self.min_instances = self.config.get("min_instances", 1)
        self.max_instances = self.config.get("max_instances", 10)
        self.min_cpu_cores = self.config.get("min_cpu_cores", 1)
        self.max_cpu_cores = self.config.get("max_cpu_cores", 16)
        self.min_memory_gb = self.config.get("min_memory_gb", 1)
        self.max_memory_gb = self.config.get("max_memory_gb", 64)
        
        # State tracking
        self.current_instances = self.config.get("initial_instances", 2)
        self.current_cpu_cores = self.config.get("initial_cpu_cores", 2)
        self.current_memory_gb = self.config.get("initial_memory_gb", 4)
        
        # Historical data
        self.metrics_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        self.last_scaling_action = None
        
        # Predictive models
        self.predictive_models = {
            "cpu": PredictiveModel("random_forest"),
            "memory": PredictiveModel("random_forest"),
            "throughput": PredictiveModel("linear")
        }
        
        # Anomaly detection for unusual patterns
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_trained = False
        
        # Threading
        self._running = False
        self._monitoring_thread = None
        self._lock = threading.RLock()
        
        logger.info("AutoScalingEngine initialized",
                   strategy=self.scaling_strategy.value,
                   min_instances=self.min_instances,
                   max_instances=self.max_instances)
    
    def start_monitoring(self, metrics_callback: Callable[[], ResourceMetrics]):
        """Start continuous monitoring and auto-scaling.
        
        Args:
            metrics_callback: Function to get current metrics
        """
        if self._running:
            return
        
        self._running = True
        self.metrics_callback = metrics_callback
        
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        
        logger.info("Auto-scaling monitoring stopped")
    
    async def evaluate_scaling_decision(
        self,
        current_metrics: ResourceMetrics
    ) -> Optional[ScalingDecision]:
        """Evaluate whether scaling action is needed.
        
        Args:
            current_metrics: Current resource metrics
            
        Returns:
            Scaling decision if action is needed, None otherwise
        """
        with self._lock:
            # Add metrics to history
            self.metrics_history.append(current_metrics)
            
            # Check cooldown period
            if self._is_in_cooldown():
                return None
            
            # Update predictive models
            self._update_predictive_models(current_metrics)
            
            # Check for anomalies
            is_anomaly = self._detect_anomaly(current_metrics)
            
            # Generate scaling decision based on strategy
            if self.scaling_strategy == ScalingStrategy.REACTIVE:
                decision = await self._reactive_scaling_decision(current_metrics)
            elif self.scaling_strategy == ScalingStrategy.PREDICTIVE:
                decision = await self._predictive_scaling_decision(current_metrics)
            elif self.scaling_strategy == ScalingStrategy.HYBRID:
                decision = await self._hybrid_scaling_decision(current_metrics)
            else:  # REINFORCEMENT
                decision = await self._reinforcement_scaling_decision(current_metrics)
            
            # Adjust decision based on anomaly
            if decision and is_anomaly:
                decision.confidence *= 0.7  # Reduce confidence during anomalies
                decision.reasoning += " (anomaly detected)"
            
            # Validate decision against constraints
            if decision:
                decision = self._validate_scaling_decision(decision)
            
            return decision
    
    async def execute_scaling_decision(
        self,
        decision: ScalingDecision,
        executor: Callable[[ScalingDecision], Tuple[bool, str, float]]
    ) -> ScalingEvent:
        """Execute a scaling decision.
        
        Args:
            decision: Scaling decision to execute
            executor: Function to execute the scaling action
            
        Returns:
            Scaling event record
        """
        start_time = time.time()
        
        try:
            # Execute the scaling action
            success, message, cost_impact = executor(decision)
            
            execution_time = time.time() - start_time
            
            # Create scaling event
            event = ScalingEvent(
                timestamp=datetime.utcnow(),
                decision=decision,
                execution_status="success" if success else "failed",
                cost_impact=cost_impact,
                execution_time_seconds=execution_time,
                error_message=None if success else message
            )
            
            # Update internal state if successful
            if success:
                self._update_resource_state(decision)
                self.last_scaling_action = datetime.utcnow()
            
            # Record event
            self.scaling_history.append(event)
            
            # Record metrics
            metrics = get_safe_metrics_collector()
            metrics.record_metric(
                "autoscaling.decisions.executed",
                1,
                {
                    "action": decision.action.value,
                    "resource_type": decision.resource_type.value,
                    "success": success
                }
            )
            
            logger.info("Scaling decision executed",
                       action=decision.action.value,
                       resource=decision.resource_type.value,
                       success=success,
                       execution_time=execution_time)
            
            return event
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            event = ScalingEvent(
                timestamp=datetime.utcnow(),
                decision=decision,
                execution_status="failed",
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
            
            self.scaling_history.append(event)
            
            logger.error("Scaling decision execution failed",
                        action=decision.action.value,
                        error=str(e))
            
            return event
    
    def get_scaling_recommendations(
        self,
        time_horizon_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get scaling recommendations for future time periods.
        
        Args:
            time_horizon_hours: How far ahead to predict
            
        Returns:
            List of recommendations
        """
        if not self.metrics_history:
            return []
        
        recommendations = []
        current_metrics = self.metrics_history[-1]
        
        # Generate predictions for different time periods
        time_periods = [1, 6, 12, 24]  # Hours ahead
        
        for hours_ahead in time_periods:
            if hours_ahead > time_horizon_hours:
                break
            
            # Get predictions from models
            predictions = {}
            for metric_name, model in self.predictive_models.items():
                pred = model.predict(current_metrics, steps_ahead=hours_ahead)
                if pred:
                    predictions[metric_name] = pred
            
            # Generate recommendations based on predictions
            if predictions:
                rec = {
                    "time_ahead_hours": hours_ahead,
                    "predicted_metrics": predictions,
                    "recommendations": self._generate_proactive_recommendations(predictions),
                    "confidence": np.mean([p.get("confidence", 0) for p in predictions.values()]),
                    "timestamp": datetime.utcnow().isoformat()
                }
                recommendations.append(rec)
        
        return recommendations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.scaling_history:
            return {"error": "No scaling history available"}
        
        # Calculate performance metrics
        successful_actions = [e for e in self.scaling_history if e.execution_status == "success"]
        failed_actions = [e for e in self.scaling_history if e.execution_status == "failed"]
        
        # Cost analysis
        total_cost_impact = sum(
            e.cost_impact for e in successful_actions 
            if e.cost_impact is not None
        )
        
        # Response time analysis
        avg_execution_time = np.mean([
            e.execution_time_seconds for e in successful_actions 
            if e.execution_time_seconds is not None
        ]) if successful_actions else 0
        
        # Action type distribution
        action_counts = defaultdict(int)
        for event in self.scaling_history:
            action_counts[event.decision.action.value] += 1
        
        # Model accuracy
        model_accuracies = {
            name: model.model_accuracy 
            for name, model in self.predictive_models.items()
        }
        
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "time_period_hours": 24,
                "current_configuration": {
                    "instances": self.current_instances,
                    "cpu_cores": self.current_cpu_cores,
                    "memory_gb": self.current_memory_gb
                }
            },
            "performance_summary": {
                "total_scaling_actions": len(self.scaling_history),
                "successful_actions": len(successful_actions),
                "failed_actions": len(failed_actions),
                "success_rate": len(successful_actions) / len(self.scaling_history) if self.scaling_history else 0,
                "avg_execution_time_seconds": avg_execution_time,
                "total_cost_impact": total_cost_impact
            },
            "action_analysis": {
                "action_distribution": dict(action_counts),
                "recent_actions": [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "action": e.decision.action.value,
                        "resource": e.decision.resource_type.value,
                        "confidence": e.decision.confidence,
                        "status": e.execution_status
                    }
                    for e in list(self.scaling_history)[-10:]  # Last 10 actions
                ]
            },
            "model_performance": {
                "model_accuracies": model_accuracies,
                "training_data_points": len(self.metrics_history),
                "anomaly_detection_enabled": self.anomaly_trained
            },
            "recommendations": self._generate_optimization_recommendations()
        }
        
        return report
    
    def _monitoring_loop(self):
        """Main monitoring loop for continuous auto-scaling."""
        while self._running:
            try:
                if hasattr(self, 'metrics_callback'):
                    # Get current metrics
                    current_metrics = self.metrics_callback()
                    
                    # Evaluate scaling decision
                    decision = asyncio.run(self.evaluate_scaling_decision(current_metrics))
                    
                    if decision and decision.confidence > 0.7:  # High confidence threshold
                        logger.info("Auto-scaling decision made",
                                   action=decision.action.value,
                                   resource=decision.resource_type.value,
                                   confidence=decision.confidence)
                        
                        # Here you would typically execute the scaling decision
                        # For now, just log it
                
                # Sleep for monitoring interval
                time.sleep(self.min_scaling_interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Sleep longer on error
    
    def _is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period after last scaling action."""
        if not self.last_scaling_action:
            return False
        
        time_since_last = (datetime.utcnow() - self.last_scaling_action).total_seconds()
        return time_since_last < self.cooldown_period_seconds
    
    def _update_predictive_models(self, metrics: ResourceMetrics):
        """Update predictive models with new data."""
        # Prepare target values for training
        target_values = {
            "cpu_usage_percent": metrics.cpu_usage_percent,
            "memory_usage_percent": metrics.memory_usage_percent,
            "throughput_requests_per_second": metrics.throughput_requests_per_second
        }
        
        # Add data to models
        for model_name, model in self.predictive_models.items():
            model.add_training_data(metrics, target_values)
            
            # Retrain periodically
            if (model.last_trained is None or 
                datetime.utcnow() - model.last_trained > timedelta(hours=1)):
                
                if model_name in target_values:
                    model.train_model(model_name.replace("_", "_usage_") + "_percent" if model_name in ["cpu", "memory"] else model_name)
    
    def _detect_anomaly(self, metrics: ResourceMetrics) -> bool:
        """Detect if current metrics represent an anomaly."""
        if len(self.metrics_history) < 50:
            return False
        
        try:
            # Prepare data for anomaly detection
            if not self.anomaly_trained:
                # Train anomaly detector
                training_data = []
                for m in list(self.metrics_history)[-50:]:  # Last 50 points
                    training_data.append([
                        m.cpu_usage_percent,
                        m.memory_usage_percent,
                        m.response_time_ms,
                        m.throughput_requests_per_second,
                        m.error_rate_percent
                    ])
                
                self.anomaly_detector.fit(training_data)
                self.anomaly_trained = True
            
            # Check current metrics
            current_data = [[
                metrics.cpu_usage_percent,
                metrics.memory_usage_percent,
                metrics.response_time_ms,
                metrics.throughput_requests_per_second,
                metrics.error_rate_percent
            ]]
            
            prediction = self.anomaly_detector.predict(current_data)
            return prediction[0] == -1  # -1 indicates anomaly
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False
    
    async def _reactive_scaling_decision(
        self,
        metrics: ResourceMetrics
    ) -> Optional[ScalingDecision]:
        """Make reactive scaling decision based on current metrics."""
        decisions = []
        
        # CPU-based scaling
        if metrics.cpu_usage_percent > self.cpu_scale_up_threshold:
            if self.current_cpu_cores < self.max_cpu_cores:
                decisions.append(ScalingDecision(
                    timestamp=datetime.utcnow(),
                    action=ScalingAction.SCALE_UP,
                    resource_type=ResourceType.CPU,
                    current_value=metrics.cpu_usage_percent,
                    target_value=self.cpu_scale_up_threshold * 0.7,  # Target 70% of threshold
                    confidence=0.8,
                    reasoning=f"CPU usage {metrics.cpu_usage_percent:.1f}% exceeds threshold {self.cpu_scale_up_threshold}%",
                    expected_impact="Reduce CPU utilization and improve response times",
                    estimated_cost_change=50.0,  # Estimated monthly cost increase
                    urgency=min(1.0, (metrics.cpu_usage_percent - self.cpu_scale_up_threshold) / 20.0)
                ))
        
        elif metrics.cpu_usage_percent < self.cpu_scale_down_threshold:
            if self.current_cpu_cores > self.min_cpu_cores:
                decisions.append(ScalingDecision(
                    timestamp=datetime.utcnow(),
                    action=ScalingAction.SCALE_DOWN,
                    resource_type=ResourceType.CPU,
                    current_value=metrics.cpu_usage_percent,
                    target_value=self.cpu_scale_down_threshold * 1.5,  # Target above threshold
                    confidence=0.7,
                    reasoning=f"CPU usage {metrics.cpu_usage_percent:.1f}% below threshold {self.cpu_scale_down_threshold}%",
                    expected_impact="Reduce costs while maintaining performance",
                    estimated_cost_change=-30.0,  # Estimated monthly cost decrease
                    urgency=0.3  # Lower urgency for scale down
                ))
        
        # Memory-based scaling
        if metrics.memory_usage_percent > self.memory_scale_up_threshold:
            if self.current_memory_gb < self.max_memory_gb:
                decisions.append(ScalingDecision(
                    timestamp=datetime.utcnow(),
                    action=ScalingAction.SCALE_UP,
                    resource_type=ResourceType.MEMORY,
                    current_value=metrics.memory_usage_percent,
                    target_value=self.memory_scale_up_threshold * 0.7,
                    confidence=0.9,  # Memory scaling is usually more predictable
                    reasoning=f"Memory usage {metrics.memory_usage_percent:.1f}% exceeds threshold {self.memory_scale_up_threshold}%",
                    expected_impact="Prevent out-of-memory errors and improve performance",
                    estimated_cost_change=40.0,
                    urgency=min(1.0, (metrics.memory_usage_percent - self.memory_scale_up_threshold) / 15.0)
                ))
        
        # Return highest priority decision
        if decisions:
            return max(decisions, key=lambda d: d.urgency * d.confidence)
        
        return None
    
    async def _predictive_scaling_decision(
        self,
        metrics: ResourceMetrics
    ) -> Optional[ScalingDecision]:
        """Make predictive scaling decision based on forecasted metrics."""
        decisions = []
        
        # Get predictions from models
        cpu_prediction = self.predictive_models["cpu"].predict(metrics, steps_ahead=6)  # 6 intervals ahead
        memory_prediction = self.predictive_models["memory"].predict(metrics, steps_ahead=6)
        
        # CPU prediction
        if cpu_prediction and cpu_prediction.get("confidence", 0) > 0.6:
            predicted_cpu = cpu_prediction["predicted_value"]
            
            if predicted_cpu > self.cpu_scale_up_threshold:
                decisions.append(ScalingDecision(
                    timestamp=datetime.utcnow(),
                    action=ScalingAction.SCALE_UP,
                    resource_type=ResourceType.CPU,
                    current_value=metrics.cpu_usage_percent,
                    target_value=predicted_cpu * 0.7,
                    confidence=cpu_prediction["confidence"] * 0.8,  # Reduce for prediction uncertainty
                    reasoning=f"Predicted CPU usage {predicted_cpu:.1f}% will exceed threshold",
                    expected_impact="Proactive scaling to prevent performance degradation",
                    estimated_cost_change=50.0,
                    urgency=0.6  # Medium urgency for predictive scaling
                ))
        
        # Memory prediction
        if memory_prediction and memory_prediction.get("confidence", 0) > 0.6:
            predicted_memory = memory_prediction["predicted_value"]
            
            if predicted_memory > self.memory_scale_up_threshold:
                decisions.append(ScalingDecision(
                    timestamp=datetime.utcnow(),
                    action=ScalingAction.SCALE_UP,
                    resource_type=ResourceType.MEMORY,
                    current_value=metrics.memory_usage_percent,
                    target_value=predicted_memory * 0.7,
                    confidence=memory_prediction["confidence"] * 0.8,
                    reasoning=f"Predicted memory usage {predicted_memory:.1f}% will exceed threshold",
                    expected_impact="Proactive scaling to prevent memory pressure",
                    estimated_cost_change=40.0,
                    urgency=0.7
                ))
        
        # Return highest confidence decision
        if decisions:
            return max(decisions, key=lambda d: d.confidence)
        
        return None
    
    async def _hybrid_scaling_decision(
        self,
        metrics: ResourceMetrics
    ) -> Optional[ScalingDecision]:
        """Make hybrid scaling decision combining reactive and predictive approaches."""
        # Get both reactive and predictive decisions
        reactive_decision = await self._reactive_scaling_decision(metrics)
        predictive_decision = await self._predictive_scaling_decision(metrics)
        
        # Combine decisions intelligently
        if reactive_decision and predictive_decision:
            # If both agree on action type, increase confidence
            if reactive_decision.action == predictive_decision.action:
                combined_confidence = min(1.0, reactive_decision.confidence + predictive_decision.confidence * 0.3)
                reactive_decision.confidence = combined_confidence
                reactive_decision.reasoning += f" (supported by prediction: {predictive_decision.reasoning})"
                return reactive_decision
            else:
                # If they disagree, prefer reactive with medium confidence
                reactive_decision.confidence *= 0.8
                reactive_decision.reasoning += " (predictive model suggests different action)"
                return reactive_decision
        
        elif reactive_decision:
            return reactive_decision
        
        elif predictive_decision:
            # Only predictive, reduce confidence
            predictive_decision.confidence *= 0.7
            return predictive_decision
        
        return None
    
    async def _reinforcement_scaling_decision(
        self,
        metrics: ResourceMetrics
    ) -> Optional[ScalingDecision]:
        """Make scaling decision based on reinforcement learning from past actions."""
        # This is a simplified version - in production, you'd use a proper RL algorithm
        
        # Analyze recent scaling history for patterns
        if len(self.scaling_history) < 5:
            # Fall back to reactive if insufficient history
            return await self._reactive_scaling_decision(metrics)
        
        recent_events = list(self.scaling_history)[-10:]
        
        # Calculate success rate for different action types
        action_success_rates = defaultdict(list)
        for event in recent_events:
            success = 1.0 if event.execution_status == "success" else 0.0
            action_success_rates[event.decision.action.value].append(success)
        
        # Get base decision from hybrid approach
        base_decision = await self._hybrid_scaling_decision(metrics)
        
        if base_decision:
            # Adjust confidence based on historical success rate
            action_type = base_decision.action.value
            if action_type in action_success_rates:
                success_rate = np.mean(action_success_rates[action_type])
                base_decision.confidence *= success_rate
                base_decision.reasoning += f" (historical success rate: {success_rate:.1%})"
        
        return base_decision
    
    def _validate_scaling_decision(self, decision: ScalingDecision) -> Optional[ScalingDecision]:
        """Validate scaling decision against constraints."""
        # Check resource limits
        if decision.resource_type == ResourceType.CPU:
            if decision.action == ScalingAction.SCALE_UP:
                if self.current_cpu_cores >= self.max_cpu_cores:
                    return None
            elif decision.action == ScalingAction.SCALE_DOWN:
                if self.current_cpu_cores <= self.min_cpu_cores:
                    return None
        
        elif decision.resource_type == ResourceType.MEMORY:
            if decision.action == ScalingAction.SCALE_UP:
                if self.current_memory_gb >= self.max_memory_gb:
                    return None
            elif decision.action == ScalingAction.SCALE_DOWN:
                if self.current_memory_gb <= self.min_memory_gb:
                    return None
        
        elif decision.resource_type == ResourceType.INSTANCES:
            if decision.action in [ScalingAction.SCALE_OUT]:
                if self.current_instances >= self.max_instances:
                    return None
            elif decision.action in [ScalingAction.SCALE_IN]:
                if self.current_instances <= self.min_instances:
                    return None
        
        # Check confidence threshold
        if decision.confidence < 0.5:
            return None
        
        return decision
    
    def _update_resource_state(self, decision: ScalingDecision):
        """Update internal resource state after successful scaling."""
        if decision.resource_type == ResourceType.CPU:
            if decision.action == ScalingAction.SCALE_UP:
                self.current_cpu_cores = min(self.max_cpu_cores, self.current_cpu_cores + 1)
            elif decision.action == ScalingAction.SCALE_DOWN:
                self.current_cpu_cores = max(self.min_cpu_cores, self.current_cpu_cores - 1)
        
        elif decision.resource_type == ResourceType.MEMORY:
            if decision.action == ScalingAction.SCALE_UP:
                self.current_memory_gb = min(self.max_memory_gb, self.current_memory_gb * 1.5)
            elif decision.action == ScalingAction.SCALE_DOWN:
                self.current_memory_gb = max(self.min_memory_gb, self.current_memory_gb * 0.75)
        
        elif decision.resource_type == ResourceType.INSTANCES:
            if decision.action == ScalingAction.SCALE_OUT:
                self.current_instances = min(self.max_instances, self.current_instances + 1)
            elif decision.action == ScalingAction.SCALE_IN:
                self.current_instances = max(self.min_instances, self.current_instances - 1)
    
    def _generate_proactive_recommendations(self, predictions: Dict[str, Dict]) -> List[str]:
        """Generate proactive recommendations based on predictions."""
        recommendations = []
        
        for metric, prediction in predictions.items():
            predicted_value = prediction.get("predicted_value", 0)
            confidence = prediction.get("confidence", 0)
            
            if confidence > 0.7:  # High confidence predictions
                if metric == "cpu" and predicted_value > 80:
                    recommendations.append(f"Consider scaling up CPU resources (predicted usage: {predicted_value:.1f}%)")
                elif metric == "memory" and predicted_value > 85:
                    recommendations.append(f"Consider scaling up memory resources (predicted usage: {predicted_value:.1f}%)")
                elif metric == "throughput" and predicted_value < 50:
                    recommendations.append(f"Consider scaling down resources due to low predicted throughput")
        
        return recommendations
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance history."""
        recommendations = []
        
        if len(self.scaling_history) < 5:
            recommendations.append("Insufficient scaling history for detailed recommendations")
            return recommendations
        
        # Analyze success rates
        recent_events = list(self.scaling_history)[-20:]
        success_rate = len([e for e in recent_events if e.execution_status == "success"]) / len(recent_events)
        
        if success_rate < 0.8:
            recommendations.append("Consider reviewing scaling thresholds - success rate is below 80%")
        
        # Analyze scaling frequency
        scaling_frequency = len(recent_events) / max(1, (datetime.utcnow() - recent_events[0].timestamp).total_seconds() / 3600)
        
        if scaling_frequency > 2:  # More than 2 scaling actions per hour
            recommendations.append("High scaling frequency detected - consider adjusting thresholds or cooldown periods")
        
        # Model accuracy recommendations
        low_accuracy_models = [
            name for name, model in self.predictive_models.items()
            if model.model_accuracy < 0.6
        ]
        
        if low_accuracy_models:
            recommendations.append(f"Consider improving predictive models for: {', '.join(low_accuracy_models)}")
        
        return recommendations


# Global auto-scaling engine instance
_auto_scaling_engine: Optional[AutoScalingEngine] = None


def get_auto_scaling_engine(config: Optional[Dict[str, Any]] = None) -> AutoScalingEngine:
    """Get the global auto-scaling engine instance."""
    global _auto_scaling_engine
    
    if _auto_scaling_engine is None or config is not None:
        _auto_scaling_engine = AutoScalingEngine(config)
    
    return _auto_scaling_engine


def initialize_auto_scaling_engine(config: Optional[Dict[str, Any]] = None) -> AutoScalingEngine:
    """Initialize the global auto-scaling engine."""
    global _auto_scaling_engine
    _auto_scaling_engine = AutoScalingEngine(config)
    return _auto_scaling_engine