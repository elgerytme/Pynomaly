"""
Autonomous quality monitoring service for self-healing data pipelines.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

from data_quality.domain.entities.quality_anomaly import QualityAnomaly
from data_quality.domain.entities.quality_lineage import QualityLineage
from core.shared.error_handling import handle_exceptions
from core.domain.abstractions.base_service import BaseService


logger = logging.getLogger(__name__)


class QualityTrend(Enum):
    """Quality trend indicators."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


@dataclass
class QualityMetric:
    """Quality metric with historical tracking."""
    name: str
    value: float
    timestamp: datetime
    threshold: float
    is_critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityThreshold:
    """Adaptive quality threshold configuration."""
    metric_name: str
    current_value: float
    historical_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    adaptive_factor: float = 1.0
    learning_rate: float = 0.1
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_threshold(self, new_value: float) -> None:
        """Update threshold using adaptive learning."""
        self.historical_values.append(new_value)
        
        # Calculate adaptive threshold based on historical data
        if len(self.historical_values) >= 10:
            mean_value = sum(self.historical_values) / len(self.historical_values)
            std_dev = (sum((x - mean_value) ** 2 for x in self.historical_values) / len(self.historical_values)) ** 0.5
            
            # Adaptive threshold with learning
            new_threshold = mean_value - (self.adaptive_factor * std_dev)
            
            # Apply learning rate to smooth changes
            self.current_value = (1 - self.learning_rate) * self.current_value + self.learning_rate * new_threshold
            
            # Ensure within bounds
            self.current_value = max(self.min_threshold, min(self.max_threshold, self.current_value))
            
        self.last_updated = datetime.utcnow()


@dataclass
class QualityPrediction:
    """Quality degradation prediction."""
    metric_name: str
    current_value: float
    predicted_value: float
    prediction_confidence: float
    time_horizon: timedelta
    risk_level: str
    recommended_actions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QualityState:
    """Quality state with historical context."""
    dataset_id: str
    overall_score: float
    metric_scores: Dict[str, float]
    trend: QualityTrend
    anomalies: List[QualityAnomaly]
    predictions: List[QualityPrediction]
    last_updated: datetime = field(default_factory=datetime.utcnow)
    historical_states: deque = field(default_factory=lambda: deque(maxlen=100))


class AutonomousQualityMonitoringService(BaseService):
    """Service for autonomous quality monitoring with self-healing capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the autonomous quality monitoring service."""
        super().__init__(config)
        self.config = config
        self.quality_states: Dict[str, QualityState] = {}
        self.quality_thresholds: Dict[str, QualityThreshold] = {}
        self.quality_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_detectors: Dict[str, Any] = {}
        self.prediction_models: Dict[str, Any] = {}
        
        # Monitoring configuration
        self.monitoring_interval = config.get("monitoring_interval", 30)  # seconds
        self.prediction_horizon = config.get("prediction_horizon", 300)  # seconds
        self.anomaly_threshold = config.get("anomaly_threshold", 0.95)
        self.learning_enabled = config.get("learning_enabled", True)
        
        # Initialize components
        self._initialize_anomaly_detectors()
        self._initialize_prediction_models()
        
        # Start monitoring tasks
        asyncio.create_task(self._continuous_monitoring_task())
        asyncio.create_task(self._threshold_adaptation_task())
        asyncio.create_task(self._prediction_task())
    
    def _initialize_anomaly_detectors(self) -> None:
        """Initialize anomaly detection models."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            # Initialize different anomaly detectors for different quality metrics
            self.anomaly_detectors = {
                "completeness": {
                    "model": IsolationForest(contamination=0.1, random_state=42),
                    "scaler": StandardScaler(),
                    "trained": False
                },
                "validity": {
                    "model": IsolationForest(contamination=0.1, random_state=42),
                    "scaler": StandardScaler(),
                    "trained": False
                },
                "consistency": {
                    "model": IsolationForest(contamination=0.1, random_state=42),
                    "scaler": StandardScaler(),
                    "trained": False
                },
                "uniqueness": {
                    "model": IsolationForest(contamination=0.1, random_state=42),
                    "scaler": StandardScaler(),
                    "trained": False
                }
            }
            
            logger.info("Initialized anomaly detection models")
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple anomaly detection")
            self.anomaly_detectors = {}
    
    def _initialize_prediction_models(self) -> None:
        """Initialize prediction models for quality forecasting."""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            
            # Initialize prediction models for different quality metrics
            self.prediction_models = {
                "completeness": {
                    "model": LinearRegression(),
                    "backup_model": RandomForestRegressor(n_estimators=10, random_state=42),
                    "trained": False,
                    "feature_window": 20
                },
                "validity": {
                    "model": LinearRegression(),
                    "backup_model": RandomForestRegressor(n_estimators=10, random_state=42),
                    "trained": False,
                    "feature_window": 20
                },
                "consistency": {
                    "model": LinearRegression(),
                    "backup_model": RandomForestRegressor(n_estimators=10, random_state=42),
                    "trained": False,
                    "feature_window": 20
                },
                "uniqueness": {
                    "model": LinearRegression(),
                    "backup_model": RandomForestRegressor(n_estimators=10, random_state=42),
                    "trained": False,
                    "feature_window": 20
                }
            }
            
            logger.info("Initialized prediction models")
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple prediction")
            self.prediction_models = {}
    
    async def _continuous_monitoring_task(self) -> None:
        """Continuous monitoring task for quality metrics."""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Monitor all registered datasets
                for dataset_id in self.quality_states.keys():
                    await self._monitor_dataset_quality(dataset_id)
                
                # Detect anomalies
                await self._detect_quality_anomalies()
                
                # Update quality trends
                await self._update_quality_trends()
                
            except Exception as e:
                logger.error(f"Continuous monitoring error: {str(e)}")
    
    async def _threshold_adaptation_task(self) -> None:
        """Task for adaptive threshold adjustment."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if self.learning_enabled:
                    await self._adapt_quality_thresholds()
                
            except Exception as e:
                logger.error(f"Threshold adaptation error: {str(e)}")
    
    async def _prediction_task(self) -> None:
        """Task for quality degradation prediction."""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                
                # Generate predictions for all datasets
                for dataset_id in self.quality_states.keys():
                    await self._generate_quality_predictions(dataset_id)
                
            except Exception as e:
                logger.error(f"Prediction task error: {str(e)}")
    
    @handle_exceptions
    async def register_dataset(self, dataset_id: str, initial_metrics: Dict[str, float]) -> None:
        """Register a dataset for autonomous monitoring."""
        # Create initial quality state
        quality_state = QualityState(
            dataset_id=dataset_id,
            overall_score=sum(initial_metrics.values()) / len(initial_metrics),
            metric_scores=initial_metrics.copy(),
            trend=QualityTrend.STABLE,
            anomalies=[],
            predictions=[]
        )
        
        self.quality_states[dataset_id] = quality_state
        
        # Initialize adaptive thresholds
        for metric_name, value in initial_metrics.items():
            threshold_key = f"{dataset_id}:{metric_name}"
            self.quality_thresholds[threshold_key] = QualityThreshold(
                metric_name=metric_name,
                current_value=value * 0.8,  # Start with 80% of initial value
                min_threshold=0.0,
                max_threshold=1.0
            )
        
        logger.info(f"Registered dataset for autonomous monitoring: {dataset_id}")
    
    @handle_exceptions
    async def update_quality_metrics(self, dataset_id: str, metrics: Dict[str, float]) -> None:
        """Update quality metrics for a dataset."""
        timestamp = datetime.utcnow()
        
        # Update metrics history
        for metric_name, value in metrics.items():
            metric_key = f"{dataset_id}:{metric_name}"
            self.quality_metrics[metric_key].append(QualityMetric(
                name=metric_name,
                value=value,
                timestamp=timestamp,
                threshold=self.quality_thresholds.get(metric_key, QualityThreshold(metric_name, 0.5)).current_value
            ))
        
        # Update quality state
        if dataset_id in self.quality_states:
            state = self.quality_states[dataset_id]
            state.metric_scores.update(metrics)
            state.overall_score = sum(metrics.values()) / len(metrics)
            state.last_updated = timestamp
            
            # Store historical state
            state.historical_states.append({
                "timestamp": timestamp,
                "overall_score": state.overall_score,
                "metric_scores": metrics.copy()
            })
    
    async def _monitor_dataset_quality(self, dataset_id: str) -> None:
        """Monitor quality for a specific dataset."""
        state = self.quality_states.get(dataset_id)
        if not state:
            return
        
        # Check for critical quality issues
        critical_issues = []
        
        for metric_name, score in state.metric_scores.items():
            threshold_key = f"{dataset_id}:{metric_name}"
            threshold = self.quality_thresholds.get(threshold_key)
            
            if threshold and score < threshold.current_value:
                critical_issues.append({
                    "metric": metric_name,
                    "current_value": score,
                    "threshold": threshold.current_value,
                    "severity": "critical" if score < threshold.current_value * 0.8 else "warning"
                })
        
        # Generate alerts for critical issues
        if critical_issues:
            await self._generate_quality_alerts(dataset_id, critical_issues)
    
    async def _detect_quality_anomalies(self) -> None:
        """Detect anomalies in quality metrics."""
        for dataset_id, state in self.quality_states.items():
            anomalies = []
            
            for metric_name, score in state.metric_scores.items():
                # Use ML-based anomaly detection if available
                if metric_name in self.anomaly_detectors:
                    anomaly_detected = await self._ml_anomaly_detection(
                        dataset_id, metric_name, score
                    )
                else:
                    # Use statistical anomaly detection
                    anomaly_detected = await self._statistical_anomaly_detection(
                        dataset_id, metric_name, score
                    )
                
                if anomaly_detected:
                    anomaly = QualityAnomaly(
                        dataset_id=dataset_id,
                        metric_name=metric_name,
                        anomaly_type="quality_degradation",
                        severity="medium",
                        description=f"Anomaly detected in {metric_name}: {score}",
                        detection_timestamp=datetime.utcnow(),
                        confidence_score=0.85
                    )
                    anomalies.append(anomaly)
            
            # Update state with detected anomalies
            state.anomalies = anomalies
    
    async def _ml_anomaly_detection(self, dataset_id: str, metric_name: str, value: float) -> bool:
        """ML-based anomaly detection."""
        detector_config = self.anomaly_detectors.get(metric_name)
        if not detector_config:
            return False
        
        metric_key = f"{dataset_id}:{metric_name}"
        metric_history = self.quality_metrics.get(metric_key, deque())
        
        if len(metric_history) < 50:  # Need minimum data for training
            return False
        
        try:
            # Prepare features (recent values)
            recent_values = [m.value for m in list(metric_history)[-50:]]
            
            # Train model if not already trained
            if not detector_config["trained"]:
                import numpy as np
                features = np.array(recent_values[:-1]).reshape(-1, 1)
                detector_config["scaler"].fit(features)
                detector_config["model"].fit(detector_config["scaler"].transform(features))
                detector_config["trained"] = True
            
            # Predict anomaly
            import numpy as np
            current_feature = np.array([[value]])
            scaled_feature = detector_config["scaler"].transform(current_feature)
            anomaly_score = detector_config["model"].decision_function(scaled_feature)[0]
            
            # Return True if anomaly detected
            return anomaly_score < -0.5
            
        except Exception as e:
            logger.error(f"ML anomaly detection error: {str(e)}")
            return False
    
    async def _statistical_anomaly_detection(self, dataset_id: str, metric_name: str, value: float) -> bool:
        """Statistical anomaly detection using z-score."""
        metric_key = f"{dataset_id}:{metric_name}"
        metric_history = self.quality_metrics.get(metric_key, deque())
        
        if len(metric_history) < 20:
            return False
        
        # Calculate z-score
        recent_values = [m.value for m in list(metric_history)[-20:]]
        mean_value = sum(recent_values) / len(recent_values)
        std_dev = (sum((x - mean_value) ** 2 for x in recent_values) / len(recent_values)) ** 0.5
        
        if std_dev == 0:
            return False
        
        z_score = abs(value - mean_value) / std_dev
        return z_score > 2.5  # Anomaly if z-score > 2.5
    
    async def _update_quality_trends(self) -> None:
        """Update quality trends for all datasets."""
        for dataset_id, state in self.quality_states.items():
            if len(state.historical_states) < 10:
                continue
            
            # Calculate trend based on recent history
            recent_scores = [s["overall_score"] for s in list(state.historical_states)[-10:]]
            
            # Linear regression to determine trend
            try:
                import numpy as np
                x = np.arange(len(recent_scores))
                y = np.array(recent_scores)
                
                # Simple linear regression
                slope = np.polyfit(x, y, 1)[0]
                
                # Determine trend
                if slope > 0.01:
                    state.trend = QualityTrend.IMPROVING
                elif slope < -0.01:
                    state.trend = QualityTrend.DEGRADING
                else:
                    # Check for volatility
                    std_dev = np.std(recent_scores)
                    if std_dev > 0.1:
                        state.trend = QualityTrend.VOLATILE
                    else:
                        state.trend = QualityTrend.STABLE
                        
            except Exception as e:
                logger.error(f"Trend calculation error: {str(e)}")
                state.trend = QualityTrend.STABLE
    
    async def _adapt_quality_thresholds(self) -> None:
        """Adapt quality thresholds based on historical data."""
        for threshold_key, threshold in self.quality_thresholds.items():
            metric_history = self.quality_metrics.get(threshold_key, deque())
            
            if len(metric_history) < 10:
                continue
            
            # Get recent values
            recent_values = [m.value for m in list(metric_history)[-50:]]
            
            # Update threshold based on recent performance
            if recent_values:
                avg_value = sum(recent_values) / len(recent_values)
                threshold.update_threshold(avg_value)
    
    async def _generate_quality_predictions(self, dataset_id: str) -> None:
        """Generate quality predictions for a dataset."""
        state = self.quality_states.get(dataset_id)
        if not state:
            return
        
        predictions = []
        
        for metric_name, current_score in state.metric_scores.items():
            prediction = await self._predict_metric_quality(dataset_id, metric_name, current_score)
            if prediction:
                predictions.append(prediction)
        
        state.predictions = predictions
    
    async def _predict_metric_quality(self, dataset_id: str, metric_name: str, 
                                    current_value: float) -> Optional[QualityPrediction]:
        """Predict quality for a specific metric."""
        metric_key = f"{dataset_id}:{metric_name}"
        metric_history = self.quality_metrics.get(metric_key, deque())
        
        if len(metric_history) < 20:
            return None
        
        try:
            # Use ML prediction if available
            if metric_name in self.prediction_models:
                return await self._ml_quality_prediction(dataset_id, metric_name, current_value)
            else:
                # Use simple trend-based prediction
                return await self._trend_based_prediction(dataset_id, metric_name, current_value)
                
        except Exception as e:
            logger.error(f"Prediction error for {metric_name}: {str(e)}")
            return None
    
    async def _ml_quality_prediction(self, dataset_id: str, metric_name: str, 
                                   current_value: float) -> Optional[QualityPrediction]:
        """ML-based quality prediction."""
        model_config = self.prediction_models.get(metric_name)
        if not model_config:
            return None
        
        metric_key = f"{dataset_id}:{metric_name}"
        metric_history = self.quality_metrics.get(metric_key, deque())
        
        try:
            # Prepare features
            recent_values = [m.value for m in list(metric_history)[-model_config["feature_window"]:]]
            
            if len(recent_values) < model_config["feature_window"]:
                return None
            
            # Train model if not already trained
            if not model_config["trained"]:
                import numpy as np
                
                # Create training data (features: past values, target: next value)
                X = []
                y = []
                
                for i in range(len(recent_values) - 5):
                    X.append(recent_values[i:i+5])
                    y.append(recent_values[i+5])
                
                if len(X) >= 10:
                    X = np.array(X)
                    y = np.array(y)
                    
                    try:
                        model_config["model"].fit(X, y)
                        model_config["trained"] = True
                    except:
                        # Use backup model if primary fails
                        model_config["backup_model"].fit(X, y)
                        model_config["model"] = model_config["backup_model"]
                        model_config["trained"] = True
            
            # Make prediction
            if model_config["trained"]:
                import numpy as np
                feature = np.array([recent_values[-5:]]).reshape(1, -1)
                predicted_value = model_config["model"].predict(feature)[0]
                
                # Calculate confidence based on recent accuracy
                confidence = max(0.5, 1.0 - abs(predicted_value - current_value))
                
                # Determine risk level
                risk_level = "low"
                if predicted_value < current_value * 0.9:
                    risk_level = "high"
                elif predicted_value < current_value * 0.95:
                    risk_level = "medium"
                
                # Generate recommendations
                recommendations = []
                if risk_level == "high":
                    recommendations.extend([
                        f"Immediate attention required for {metric_name}",
                        "Consider automated remediation",
                        "Review data pipeline configuration"
                    ])
                elif risk_level == "medium":
                    recommendations.extend([
                        f"Monitor {metric_name} closely",
                        "Consider preventive measures"
                    ])
                
                return QualityPrediction(
                    metric_name=metric_name,
                    current_value=current_value,
                    predicted_value=predicted_value,
                    prediction_confidence=confidence,
                    time_horizon=timedelta(seconds=self.prediction_horizon),
                    risk_level=risk_level,
                    recommended_actions=recommendations
                )
            
        except Exception as e:
            logger.error(f"ML prediction error: {str(e)}")
        
        return None
    
    async def _trend_based_prediction(self, dataset_id: str, metric_name: str, 
                                    current_value: float) -> Optional[QualityPrediction]:
        """Simple trend-based prediction."""
        metric_key = f"{dataset_id}:{metric_name}"
        metric_history = self.quality_metrics.get(metric_key, deque())
        
        recent_values = [m.value for m in list(metric_history)[-10:]]
        
        if len(recent_values) < 5:
            return None
        
        # Calculate simple trend
        early_avg = sum(recent_values[:len(recent_values)//2]) / (len(recent_values)//2)
        late_avg = sum(recent_values[len(recent_values)//2:]) / (len(recent_values) - len(recent_values)//2)
        
        trend = late_avg - early_avg
        predicted_value = current_value + trend
        
        # Ensure prediction is within valid range
        predicted_value = max(0.0, min(1.0, predicted_value))
        
        # Calculate confidence
        confidence = max(0.3, 1.0 - abs(trend))
        
        # Determine risk level
        risk_level = "low"
        if predicted_value < current_value * 0.9:
            risk_level = "high"
        elif predicted_value < current_value * 0.95:
            risk_level = "medium"
        
        return QualityPrediction(
            metric_name=metric_name,
            current_value=current_value,
            predicted_value=predicted_value,
            prediction_confidence=confidence,
            time_horizon=timedelta(seconds=self.prediction_horizon),
            risk_level=risk_level,
            recommended_actions=[f"Monitor {metric_name} trend"]
        )
    
    async def _generate_quality_alerts(self, dataset_id: str, issues: List[Dict[str, Any]]) -> None:
        """Generate quality alerts for critical issues."""
        for issue in issues:
            logger.warning(
                f"Quality alert for {dataset_id}: {issue['metric']} = {issue['current_value']:.3f} "
                f"(threshold: {issue['threshold']:.3f}, severity: {issue['severity']})"
            )
            
            # In a real implementation, this would send notifications
            # to monitoring systems, Slack, email, etc.
    
    @handle_exceptions
    async def get_quality_state(self, dataset_id: str) -> Optional[QualityState]:
        """Get current quality state for a dataset."""
        return self.quality_states.get(dataset_id)
    
    @handle_exceptions
    async def get_quality_dashboard(self) -> Dict[str, Any]:
        """Get quality monitoring dashboard data."""
        dashboard = {
            "timestamp": datetime.utcnow(),
            "total_datasets": len(self.quality_states),
            "datasets": {},
            "global_health": 0.0,
            "active_anomalies": 0,
            "predictions_count": 0
        }
        
        total_score = 0.0
        total_anomalies = 0
        total_predictions = 0
        
        for dataset_id, state in self.quality_states.items():
            dashboard["datasets"][dataset_id] = {
                "overall_score": state.overall_score,
                "trend": state.trend.value,
                "anomalies_count": len(state.anomalies),
                "predictions_count": len(state.predictions),
                "last_updated": state.last_updated,
                "metric_scores": state.metric_scores
            }
            
            total_score += state.overall_score
            total_anomalies += len(state.anomalies)
            total_predictions += len(state.predictions)
        
        if self.quality_states:
            dashboard["global_health"] = total_score / len(self.quality_states)
        
        dashboard["active_anomalies"] = total_anomalies
        dashboard["predictions_count"] = total_predictions
        
        return dashboard
    
    @handle_exceptions
    async def get_quality_forecasts(self, dataset_id: str) -> List[QualityPrediction]:
        """Get quality forecasts for a dataset."""
        state = self.quality_states.get(dataset_id)
        if not state:
            return []
        
        return state.predictions
    
    async def shutdown(self) -> None:
        """Shutdown the autonomous quality monitoring service."""
        logger.info("Shutting down autonomous quality monitoring service...")
        
        # Clear all data
        self.quality_states.clear()
        self.quality_thresholds.clear()
        self.quality_metrics.clear()
        self.anomaly_detectors.clear()
        self.prediction_models.clear()
        
        logger.info("Autonomous quality monitoring service shutdown complete")