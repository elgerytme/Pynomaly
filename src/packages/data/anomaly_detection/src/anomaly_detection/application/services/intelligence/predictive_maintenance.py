"""Predictive maintenance framework for system health monitoring and failure prediction."""

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
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

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


class MaintenanceStatus(Enum):
    """Maintenance status types."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    MAINTENANCE_REQUIRED = "maintenance_required"
    FAILED = "failed"


class ComponentType(Enum):
    """Types of system components."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    API_SERVICE = "api_service"
    MODEL_INFERENCE = "model_inference"
    CACHE = "cache"
    QUEUE = "queue"
    LOAD_BALANCER = "load_balancer"


class FailureMode(Enum):
    """Types of potential failures."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONNECTION_FAILURE = "connection_failure"
    DATA_CORRUPTION = "data_corruption"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    MEMORY_LEAK = "memory_leak"
    DISK_FULL = "disk_full"
    HIGH_ERROR_RATE = "high_error_rate"
    CASCADING_FAILURE = "cascading_failure"


@dataclass
class ComponentHealth:
    """Health metrics for a system component."""
    component_id: str
    component_type: ComponentType
    timestamp: datetime
    
    # Resource utilization
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_mbps: float
    
    # Performance metrics
    response_time_ms: float
    throughput_ops_per_second: float
    error_rate_percent: float
    availability_percent: float
    
    # Health indicators
    temperature_celsius: Optional[float] = None
    uptime_hours: Optional[float] = None
    connection_count: Optional[int] = None
    queue_depth: Optional[int] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MaintenanceAlert:
    """Predictive maintenance alert."""
    alert_id: str
    component_id: str
    component_type: ComponentType
    predicted_failure_mode: FailureMode
    status: MaintenanceStatus
    confidence: float  # 0-1
    time_to_failure_hours: Optional[float]
    
    # Alert details
    title: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    recommended_actions: List[str]
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    evidence: Dict[str, Any] = field(default_factory=dict)
    historical_patterns: List[str] = field(default_factory=list)


@dataclass
class MaintenanceRecommendation:
    """Maintenance recommendation with scheduling."""
    recommendation_id: str
    component_id: str
    maintenance_type: str  # "preventive", "corrective", "predictive"
    urgency: str  # "low", "medium", "high", "emergency"
    
    # Recommendation details
    title: str
    description: str
    estimated_duration_hours: float
    estimated_cost: Optional[float] = None
    required_resources: List[str] = field(default_factory=list)
    
    # Scheduling
    recommended_start_time: Optional[datetime] = None
    maintenance_window_hours: Optional[int] = None
    can_be_automated: bool = False
    
    # Impact assessment
    expected_downtime_minutes: Optional[float] = None
    impact_on_other_components: List[str] = field(default_factory=list)
    risk_if_delayed: str = "medium"
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HealthPredictor:
    """Machine learning model for predicting component health."""
    
    def __init__(self, component_type: ComponentType, model_type: str = "random_forest"):
        self.component_type = component_type
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.last_trained = None
        
        # Training data buffer
        self.training_data = deque(maxlen=5000)
        self.model_accuracy = 0.0
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def add_training_data(
        self, 
        health_data: ComponentHealth, 
        failure_occurred: bool,
        failure_mode: Optional[FailureMode] = None
    ):
        """Add training data point."""
        features = self._extract_features(health_data)
        
        self.training_data.append({
            "features": features,
            "failure_occurred": failure_occurred,
            "failure_mode": failure_mode.value if failure_mode else None,
            "timestamp": health_data.timestamp
        })
    
    def train_model(self) -> float:
        """Train the predictive model."""
        if len(self.training_data) < 100:
            return 0.0
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for data_point in self.training_data:
                X.append(data_point["features"])
                y.append(1 if data_point["failure_occurred"] else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Handle class imbalance by ensuring we have both classes
            if len(set(y)) < 2:
                logger.warning(f"Insufficient class diversity for {self.component_type.value}")
                return 0.0
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            predictions = self.model.predict(X_test_scaled)
            accuracy = np.mean(predictions == y_test)
            
            self.model_accuracy = accuracy
            self.last_trained = datetime.utcnow()
            
            logger.info(f"Model trained for {self.component_type.value}",
                       accuracy=accuracy,
                       training_samples=len(X))
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Model training failed for {self.component_type.value}: {e}")
            return 0.0
    
    def predict_failure(
        self, 
        health_data: ComponentHealth,
        time_horizon_hours: int = 24
    ) -> Dict[str, Any]:
        """Predict probability of failure within time horizon."""
        if self.model is None or self.last_trained is None:
            return {}
        
        try:
            features = self._extract_features(health_data)
            features_scaled = self.scaler.transform([features])
            
            # Get probability prediction
            failure_probability = self.model.predict_proba(features_scaled)[0]
            
            # Get feature importance for explanation
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                for i, importance in enumerate(self.model.feature_importances_):
                    if i < len(self.feature_names):
                        feature_importance[self.feature_names[i]] = float(importance)
            
            return {
                "failure_probability": float(failure_probability[1]) if len(failure_probability) > 1 else 0.0,
                "confidence": self.model_accuracy,
                "time_horizon_hours": time_horizon_hours,
                "feature_importance": feature_importance,
                "prediction_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.component_type.value}: {e}")
            return {}
    
    def _extract_features(self, health_data: ComponentHealth) -> List[float]:
        """Extract features from health data."""
        # Time-based features
        timestamp = health_data.timestamp
        hour_of_day = timestamp.hour / 24.0
        day_of_week = timestamp.weekday() / 7.0
        
        # Core features
        features = [
            health_data.cpu_usage_percent / 100.0,
            health_data.memory_usage_percent / 100.0,
            health_data.disk_usage_percent / 100.0,
            health_data.network_io_mbps / 1000.0,
            health_data.response_time_ms / 1000.0,
            health_data.throughput_ops_per_second / 100.0,
            health_data.error_rate_percent / 100.0,
            health_data.availability_percent / 100.0,
            hour_of_day,
            day_of_week
        ]
        
        # Optional features
        if health_data.temperature_celsius is not None:
            features.append(health_data.temperature_celsius / 100.0)
        else:
            features.append(0.0)
        
        if health_data.uptime_hours is not None:
            features.append(min(health_data.uptime_hours / 168.0, 1.0))  # Normalize to weeks
        else:
            features.append(0.0)
        
        if health_data.connection_count is not None:
            features.append(health_data.connection_count / 1000.0)
        else:
            features.append(0.0)
        
        if health_data.queue_depth is not None:
            features.append(health_data.queue_depth / 100.0)
        else:
            features.append(0.0)
        
        # Add custom metrics
        for value in health_data.custom_metrics.values():
            features.append(value)
        
        # Set feature names on first extraction
        if not self.feature_names:
            self.feature_names = [
                "cpu_usage", "memory_usage", "disk_usage", "network_io",
                "response_time", "throughput", "error_rate", "availability",
                "hour_of_day", "day_of_week", "temperature", "uptime",
                "connections", "queue_depth"
            ] + list(health_data.custom_metrics.keys())
        
        return features


class MaintenanceScheduler:
    """Intelligent maintenance scheduling system."""
    
    def __init__(self):
        self.scheduled_maintenance = {}
        self.maintenance_windows = {
            "weekday_night": {"start_hour": 2, "end_hour": 6, "days": [0, 1, 2, 3, 4]},
            "weekend": {"start_hour": 1, "end_hour": 23, "days": [5, 6]},
            "emergency_anytime": {"start_hour": 0, "end_hour": 24, "days": list(range(7))}
        }
        
    def schedule_maintenance(
        self,
        recommendation: MaintenanceRecommendation,
        preferred_window: str = "weekday_night"
    ) -> datetime:
        """Schedule maintenance within preferred window."""
        
        current_time = datetime.utcnow()
        window = self.maintenance_windows.get(preferred_window, self.maintenance_windows["weekday_night"])
        
        # Find next available slot
        next_slot = self._find_next_available_slot(
            current_time,
            window,
            recommendation.estimated_duration_hours
        )
        
        recommendation.recommended_start_time = next_slot
        self.scheduled_maintenance[recommendation.recommendation_id] = recommendation
        
        logger.info(f"Maintenance scheduled",
                   component=recommendation.component_id,
                   scheduled_time=next_slot.isoformat(),
                   window=preferred_window)
        
        return next_slot
    
    def _find_next_available_slot(
        self,
        from_time: datetime,
        window: Dict[str, Any],
        duration_hours: float
    ) -> datetime:
        """Find next available maintenance slot."""
        
        # Start looking from tomorrow if it's too late today
        search_start = from_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        for days_ahead in range(14):  # Look up to 2 weeks ahead
            candidate_date = search_start + timedelta(days=days_ahead)
            
            if candidate_date.weekday() in window["days"]:
                # Find suitable time within the window
                start_hour = window["start_hour"]
                end_hour = window["end_hour"]
                
                # Check if we have enough time in the window
                available_hours = end_hour - start_hour
                if available_hours >= duration_hours:
                    maintenance_time = candidate_date.replace(hour=start_hour)
                    
                    # Check for conflicts with existing maintenance
                    if not self._has_scheduling_conflict(maintenance_time, duration_hours):
                        return maintenance_time
        
        # If no suitable slot found, use emergency window
        return from_time + timedelta(hours=1)
    
    def _has_scheduling_conflict(self, start_time: datetime, duration_hours: float) -> bool:
        """Check if proposed time conflicts with existing maintenance."""
        end_time = start_time + timedelta(hours=duration_hours)
        
        for scheduled in self.scheduled_maintenance.values():
            if scheduled.recommended_start_time is None:
                continue
                
            scheduled_start = scheduled.recommended_start_time
            scheduled_end = scheduled_start + timedelta(hours=scheduled.estimated_duration_hours)
            
            # Check for overlap
            if (start_time < scheduled_end and end_time > scheduled_start):
                return True
        
        return False


class PredictiveMaintenanceEngine:
    """Main predictive maintenance engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize predictive maintenance engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configuration
        self.monitoring_interval_seconds = self.config.get("monitoring_interval", 300)  # 5 minutes
        self.prediction_horizon_hours = self.config.get("prediction_horizon", 24)
        self.alert_threshold = self.config.get("alert_threshold", 0.7)  # Failure probability threshold
        
        # Components being monitored
        self.monitored_components = {}
        
        # Predictive models for each component type
        self.predictors = {}
        for component_type in ComponentType:
            self.predictors[component_type] = HealthPredictor(component_type)
        
        # Health history
        self.health_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Alerts and recommendations
        self.active_alerts = {}
        self.maintenance_recommendations = {}
        
        # Scheduler
        self.scheduler = MaintenanceScheduler()
        
        # Anomaly detection for unusual patterns
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_trained = False
        
        # Threading
        self._running = False
        self._monitoring_thread = None
        self._lock = threading.RLock()
        
        logger.info("PredictiveMaintenanceEngine initialized",
                   prediction_horizon_hours=self.prediction_horizon_hours,
                   alert_threshold=self.alert_threshold)
    
    def register_component(
        self,
        component_id: str,
        component_type: ComponentType,
        health_callback: Callable[[], ComponentHealth]
    ):
        """Register a component for monitoring.
        
        Args:
            component_id: Unique component identifier
            component_type: Type of component
            health_callback: Function to get current health metrics
        """
        self.monitored_components[component_id] = {
            "type": component_type,
            "health_callback": health_callback,
            "registered_at": datetime.utcnow()
        }
        
        logger.info(f"Component registered for monitoring",
                   component_id=component_id,
                   component_type=component_type.value)
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("Predictive maintenance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        
        logger.info("Predictive maintenance monitoring stopped")
    
    async def analyze_component_health(
        self,
        component_id: str
    ) -> Dict[str, Any]:
        """Analyze health of a specific component.
        
        Args:
            component_id: Component to analyze
            
        Returns:
            Health analysis results
        """
        if component_id not in self.monitored_components:
            return {"error": "Component not registered"}
        
        with self._lock:
            try:
                # Get current health data
                component_info = self.monitored_components[component_id]
                current_health = component_info["health_callback"]()
                
                # Add to history
                self.health_history[component_id].append(current_health)
                
                # Get predictor for component type
                predictor = self.predictors[component_info["type"]]
                
                # Predict failure probability
                prediction = predictor.predict_failure(
                    current_health,
                    self.prediction_horizon_hours
                )
                
                # Detect anomalies
                is_anomaly = self._detect_health_anomaly(component_id, current_health)
                
                # Generate health status
                health_status = self._determine_health_status(
                    current_health,
                    prediction.get("failure_probability", 0),
                    is_anomaly
                )
                
                # Check if alert is needed
                alert = None
                if prediction.get("failure_probability", 0) > self.alert_threshold:
                    alert = await self._generate_maintenance_alert(
                        component_id,
                        current_health,
                        prediction,
                        health_status
                    )
                
                analysis = {
                    "component_id": component_id,
                    "component_type": component_info["type"].value,
                    "current_health": {
                        "timestamp": current_health.timestamp.isoformat(),
                        "cpu_usage_percent": current_health.cpu_usage_percent,
                        "memory_usage_percent": current_health.memory_usage_percent,
                        "response_time_ms": current_health.response_time_ms,
                        "error_rate_percent": current_health.error_rate_percent,
                        "availability_percent": current_health.availability_percent
                    },
                    "health_status": health_status.value,
                    "prediction": prediction,
                    "is_anomaly": is_anomaly,
                    "alert": alert.__dict__ if alert else None,
                    "trends": self._analyze_health_trends(component_id),
                    "recommendations": self._generate_health_recommendations(
                        component_id, current_health, prediction
                    )
                }
                
                return analysis
                
            except Exception as e:
                logger.error(f"Health analysis failed for {component_id}: {e}")
                return {"error": str(e)}
    
    def get_maintenance_schedule(
        self,
        days_ahead: int = 7
    ) -> List[Dict[str, Any]]:
        """Get scheduled maintenance for the next N days.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of scheduled maintenance items
        """
        cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
        
        scheduled_items = []
        for rec_id, recommendation in self.scheduler.scheduled_maintenance.items():
            if (recommendation.recommended_start_time and 
                recommendation.recommended_start_time <= cutoff_date):
                
                scheduled_items.append({
                    "recommendation_id": rec_id,
                    "component_id": recommendation.component_id,
                    "maintenance_type": recommendation.maintenance_type,
                    "title": recommendation.title,
                    "urgency": recommendation.urgency,
                    "scheduled_time": recommendation.recommended_start_time.isoformat(),
                    "estimated_duration_hours": recommendation.estimated_duration_hours,
                    "estimated_downtime_minutes": recommendation.expected_downtime_minutes,
                    "can_be_automated": recommendation.can_be_automated
                })
        
        # Sort by scheduled time
        scheduled_items.sort(key=lambda x: x["scheduled_time"])
        
        return scheduled_items
    
    def generate_maintenance_report(self) -> Dict[str, Any]:
        """Generate comprehensive maintenance report."""
        
        # Analyze component health across all monitored components
        component_summary = {}
        for component_id, component_info in self.monitored_components.items():
            try:
                health_data = component_info["health_callback"]()
                component_type = component_info["type"]
                predictor = self.predictors[component_type]
                
                prediction = predictor.predict_failure(health_data)
                
                component_summary[component_id] = {
                    "type": component_type.value,
                    "health_status": self._determine_health_status(
                        health_data,
                        prediction.get("failure_probability", 0),
                        False
                    ).value,
                    "failure_probability": prediction.get("failure_probability", 0),
                    "last_updated": health_data.timestamp.isoformat()
                }
            except Exception as e:
                component_summary[component_id] = {
                    "error": str(e),
                    "last_updated": datetime.utcnow().isoformat()
                }
        
        # Model performance summary
        model_performance = {}
        for component_type, predictor in self.predictors.items():
            model_performance[component_type.value] = {
                "accuracy": predictor.model_accuracy,
                "last_trained": predictor.last_trained.isoformat() if predictor.last_trained else None,
                "training_data_points": len(predictor.training_data)
            }
        
        # Active alerts summary
        alerts_summary = {
            "total_active_alerts": len(self.active_alerts),
            "critical_alerts": len([
                alert for alert in self.active_alerts.values()
                if alert.status == MaintenanceStatus.CRITICAL
            ]),
            "warning_alerts": len([
                alert for alert in self.active_alerts.values()
                if alert.status == MaintenanceStatus.WARNING
            ])
        }
        
        # Maintenance schedule summary
        upcoming_maintenance = self.get_maintenance_schedule(7)
        
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "monitoring_components": len(self.monitored_components),
                "prediction_horizon_hours": self.prediction_horizon_hours
            },
            "system_health_overview": {
                "components": component_summary,
                "overall_health_score": self._calculate_overall_health_score(component_summary)
            },
            "predictive_models": {
                "model_performance": model_performance,
                "total_predictions_made": sum(len(predictor.training_data) for predictor in self.predictors.values())
            },
            "alerts_and_notifications": alerts_summary,
            "maintenance_schedule": {
                "upcoming_maintenance_count": len(upcoming_maintenance),
                "upcoming_maintenance": upcoming_maintenance,
                "total_scheduled_hours": sum(
                    item["estimated_duration_hours"] for item in upcoming_maintenance
                )
            },
            "recommendations": self._generate_system_recommendations()
        }
        
        return report
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Analyze all monitored components
                for component_id in list(self.monitored_components.keys()):
                    try:
                        analysis = asyncio.run(self.analyze_component_health(component_id))
                        
                        # Record metrics
                        if "current_health" in analysis and "prediction" in analysis:
                            metrics = get_safe_metrics_collector()
                            metrics.record_metric(
                                "maintenance.component.health_score",
                                self._health_to_score(analysis["health_status"]),
                                {"component_id": component_id}
                            )
                            
                            if "failure_probability" in analysis["prediction"]:
                                metrics.record_metric(
                                    "maintenance.component.failure_probability",
                                    analysis["prediction"]["failure_probability"],
                                    {"component_id": component_id}
                                )
                    
                    except Exception as e:
                        logger.error(f"Component analysis failed: {component_id}: {e}")
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Sleep longer on error
    
    def _detect_health_anomaly(self, component_id: str, health_data: ComponentHealth) -> bool:
        """Detect if current health data represents an anomaly."""
        history = list(self.health_history[component_id])
        
        if len(history) < 50:
            return False
        
        try:
            # Prepare training data for anomaly detection
            if not self.anomaly_trained:
                training_data = []
                for h in history[-50:]:
                    training_data.append([
                        h.cpu_usage_percent,
                        h.memory_usage_percent,
                        h.response_time_ms,
                        h.error_rate_percent,
                        h.availability_percent
                    ])
                
                self.anomaly_detector.fit(training_data)
                self.anomaly_trained = True
            
            # Check current data
            current_data = [[
                health_data.cpu_usage_percent,
                health_data.memory_usage_percent,
                health_data.response_time_ms,
                health_data.error_rate_percent,
                health_data.availability_percent
            ]]
            
            prediction = self.anomaly_detector.predict(current_data)
            return prediction[0] == -1
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False
    
    def _determine_health_status(
        self,
        health_data: ComponentHealth,
        failure_probability: float,
        is_anomaly: bool
    ) -> MaintenanceStatus:
        """Determine overall health status."""
        
        # Critical conditions
        if (health_data.error_rate_percent > 10 or 
            health_data.availability_percent < 95 or
            failure_probability > 0.8):
            return MaintenanceStatus.CRITICAL
        
        # Warning conditions
        if (health_data.cpu_usage_percent > 80 or
            health_data.memory_usage_percent > 85 or
            health_data.response_time_ms > 5000 or
            failure_probability > self.alert_threshold or
            is_anomaly):
            return MaintenanceStatus.WARNING
        
        # Maintenance required
        if (health_data.cpu_usage_percent > 70 or
            health_data.memory_usage_percent > 75 or
            failure_probability > 0.3):
            return MaintenanceStatus.MAINTENANCE_REQUIRED
        
        return MaintenanceStatus.HEALTHY
    
    async def _generate_maintenance_alert(
        self,
        component_id: str,
        health_data: ComponentHealth,
        prediction: Dict[str, Any],
        status: MaintenanceStatus
    ) -> MaintenanceAlert:
        """Generate maintenance alert."""
        
        failure_probability = prediction.get("failure_probability", 0)
        
        # Determine predicted failure mode
        predicted_failure_mode = self._predict_failure_mode(health_data)
        
        # Calculate time to failure (simplified estimation)
        time_to_failure = None
        if failure_probability > 0.5:
            # Rough estimation based on probability and current metrics
            urgency_factor = max(health_data.cpu_usage_percent, health_data.memory_usage_percent) / 100.0
            time_to_failure = max(1, 48 * (1 - failure_probability) * (1 - urgency_factor))
        
        alert = MaintenanceAlert(
            alert_id=f"alert_{component_id}_{int(time.time())}",
            component_id=component_id,
            component_type=health_data.component_type,
            predicted_failure_mode=predicted_failure_mode,
            status=status,
            confidence=prediction.get("confidence", 0.5),
            time_to_failure_hours=time_to_failure,
            title=f"{status.value.title()} - {component_id}",
            description=f"Component {component_id} showing {status.value} status with {failure_probability:.1%} failure probability",
            severity=self._map_status_to_severity(status),
            recommended_actions=self._generate_recommended_actions(health_data, predicted_failure_mode),
            evidence={
                "cpu_usage": health_data.cpu_usage_percent,
                "memory_usage": health_data.memory_usage_percent,
                "response_time": health_data.response_time_ms,
                "error_rate": health_data.error_rate_percent
            }
        )
        
        # Store active alert
        self.active_alerts[alert.alert_id] = alert
        
        logger.warning(f"Maintenance alert generated",
                      component_id=component_id,
                      status=status.value,
                      failure_probability=failure_probability)
        
        return alert
    
    def _predict_failure_mode(self, health_data: ComponentHealth) -> FailureMode:
        """Predict the most likely failure mode based on health data."""
        
        if health_data.memory_usage_percent > 90:
            return FailureMode.RESOURCE_EXHAUSTION
        elif health_data.response_time_ms > 10000:
            return FailureMode.PERFORMANCE_DEGRADATION
        elif health_data.error_rate_percent > 5:
            return FailureMode.HIGH_ERROR_RATE
        elif health_data.disk_usage_percent > 95:
            return FailureMode.DISK_FULL
        elif health_data.availability_percent < 99:
            return FailureMode.SERVICE_UNAVAILABLE
        else:
            return FailureMode.PERFORMANCE_DEGRADATION
    
    def _generate_recommended_actions(
        self,
        health_data: ComponentHealth,
        failure_mode: FailureMode
    ) -> List[str]:
        """Generate recommended actions based on failure mode."""
        
        actions = []
        
        if failure_mode == FailureMode.RESOURCE_EXHAUSTION:
            actions.extend([
                "Scale up memory allocation",
                "Review memory usage patterns",
                "Implement memory cleanup routines",
                "Consider horizontal scaling"
            ])
        elif failure_mode == FailureMode.PERFORMANCE_DEGRADATION:
            actions.extend([
                "Optimize database queries",
                "Review application performance",
                "Scale up CPU resources",
                "Implement caching strategies"
            ])
        elif failure_mode == FailureMode.HIGH_ERROR_RATE:
            actions.extend([
                "Review error logs",
                "Check external dependencies",
                "Implement circuit breakers",
                "Review recent deployments"
            ])
        elif failure_mode == FailureMode.DISK_FULL:
            actions.extend([
                "Clean up temporary files",
                "Archive old logs",
                "Increase disk space",
                "Implement log rotation"
            ])
        else:
            actions.extend([
                "Monitor component closely",
                "Review system logs",
                "Consider preventive maintenance"
            ])
        
        return actions
    
    def _analyze_health_trends(self, component_id: str) -> Dict[str, str]:
        """Analyze health trends for a component."""
        history = list(self.health_history[component_id])
        
        if len(history) < 10:
            return {"status": "insufficient_data"}
        
        # Analyze trends in key metrics
        trends = {}
        
        # CPU trend
        cpu_values = [h.cpu_usage_percent for h in history[-10:]]
        trends["cpu"] = self._calculate_trend(cpu_values)
        
        # Memory trend
        memory_values = [h.memory_usage_percent for h in history[-10:]]
        trends["memory"] = self._calculate_trend(memory_values)
        
        # Response time trend
        response_time_values = [h.response_time_ms for h in history[-10:]]
        trends["response_time"] = self._calculate_trend(response_time_values)
        
        # Error rate trend
        error_rate_values = [h.error_rate_percent for h in history[-10:]]
        trends["error_rate"] = self._calculate_trend(error_rate_values)
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 5:
            return "stable"
        
        # Simple trend analysis
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        change_percent = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
        
        if change_percent > 15:
            return "increasing"
        elif change_percent < -15:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_health_recommendations(
        self,
        component_id: str,
        health_data: ComponentHealth,
        prediction: Dict[str, Any]
    ) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        failure_probability = prediction.get("failure_probability", 0)
        
        if failure_probability > 0.7:
            recommendations.append(f"Schedule immediate maintenance for {component_id}")
        elif failure_probability > 0.4:
            recommendations.append(f"Plan preventive maintenance for {component_id}")
        
        if health_data.cpu_usage_percent > 80:
            recommendations.append("Consider CPU scaling or optimization")
        
        if health_data.memory_usage_percent > 85:
            recommendations.append("Review memory usage and consider scaling")
        
        if health_data.response_time_ms > 2000:
            recommendations.append("Investigate performance bottlenecks")
        
        if health_data.error_rate_percent > 1:
            recommendations.append("Review error patterns and fix underlying issues")
        
        return recommendations
    
    def _calculate_overall_health_score(self, component_summary: Dict[str, Any]) -> float:
        """Calculate overall system health score."""
        if not component_summary:
            return 0.0
        
        health_scores = []
        for component_data in component_summary.values():
            if "health_status" in component_data:
                score = self._health_to_score(component_data["health_status"])
                health_scores.append(score)
        
        return np.mean(health_scores) if health_scores else 0.0
    
    def _health_to_score(self, health_status: str) -> float:
        """Convert health status to numeric score."""
        status_scores = {
            "healthy": 100.0,
            "maintenance_required": 75.0,
            "warning": 50.0,
            "critical": 25.0,
            "failed": 0.0
        }
        return status_scores.get(health_status, 50.0)
    
    def _map_status_to_severity(self, status: MaintenanceStatus) -> str:
        """Map maintenance status to severity level."""
        severity_map = {
            MaintenanceStatus.HEALTHY: "low",
            MaintenanceStatus.MAINTENANCE_REQUIRED: "medium",
            MaintenanceStatus.WARNING: "high",
            MaintenanceStatus.CRITICAL: "critical",
            MaintenanceStatus.FAILED: "critical"
        }
        return severity_map.get(status, "medium")
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide recommendations."""
        recommendations = []
        
        # Analyze model performance
        low_accuracy_models = [
            component_type.value for component_type, predictor in self.predictors.items()
            if predictor.model_accuracy < 0.7
        ]
        
        if low_accuracy_models:
            recommendations.append(f"Improve predictive models for: {', '.join(low_accuracy_models)}")
        
        # Check active alerts
        critical_count = len([
            alert for alert in self.active_alerts.values()
            if alert.status == MaintenanceStatus.CRITICAL
        ])
        
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical maintenance alerts immediately")
        
        # Check maintenance schedule
        upcoming_maintenance = self.get_maintenance_schedule(7)
        if len(upcoming_maintenance) > 5:
            recommendations.append("High maintenance workload - consider prioritization and resource planning")
        
        return recommendations


# Global predictive maintenance engine instance
_predictive_maintenance_engine: Optional[PredictiveMaintenanceEngine] = None


def get_predictive_maintenance_engine(config: Optional[Dict[str, Any]] = None) -> PredictiveMaintenanceEngine:
    """Get the global predictive maintenance engine instance."""
    global _predictive_maintenance_engine
    
    if _predictive_maintenance_engine is None or config is not None:
        _predictive_maintenance_engine = PredictiveMaintenanceEngine(config)
    
    return _predictive_maintenance_engine


def initialize_predictive_maintenance_engine(config: Optional[Dict[str, Any]] = None) -> PredictiveMaintenanceEngine:
    """Initialize the global predictive maintenance engine."""
    global _predictive_maintenance_engine
    _predictive_maintenance_engine = PredictiveMaintenanceEngine(config)
    return _predictive_maintenance_engine