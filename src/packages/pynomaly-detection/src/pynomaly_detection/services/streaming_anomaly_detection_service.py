"""
Real-time Streaming Anomaly Detection Service

High-performance streaming anomaly detection with real-time processing,
adaptive thresholds, and intelligent alerting.
"""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from pynomaly.infrastructure.logging.structured_logger import StructuredLogger
from pynomaly.infrastructure.monitoring.metrics_service import MetricsService


class AnomalyThreatLevel(Enum):
    """Anomaly threat levels."""

    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StreamingMode(Enum):
    """Streaming processing modes."""

    REAL_TIME = "real_time"
    MICRO_BATCH = "micro_batch"
    WINDOWED = "windowed"


@dataclass
class StreamingDataPoint:
    """A single data point in the stream."""

    timestamp: datetime
    features: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    source_id: str = ""
    correlation_id: str = ""


@dataclass
class AnomalyAlert:
    """Anomaly detection alert."""

    alert_id: str
    timestamp: datetime
    data_point: StreamingDataPoint
    anomaly_score: float
    threat_level: AnomalyThreatLevel
    confidence: float
    explanation: str
    recommendations: list[str]
    affected_features: list[str]
    historical_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingDetectorConfig:
    """Configuration for streaming anomaly detector."""

    detector_id: str
    detector_type: str = "isolation_forest"
    window_size: int = 1000
    batch_size: int = 50
    update_frequency: int = 100  # Update model every N samples
    anomaly_threshold: float = 0.1
    adaptation_rate: float = 0.01
    enable_adaptive_threshold: bool = True
    enable_trend_detection: bool = True
    enable_seasonal_adjustment: bool = True
    threat_level_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9,
        }
    )


class AdaptiveThresholdManager:
    """Manages adaptive thresholds for anomaly detection."""

    def __init__(self, initial_threshold: float = 0.1, adaptation_rate: float = 0.01):
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.recent_scores = deque(maxlen=1000)
        self.false_positive_rate = 0.0
        self.false_negative_rate = 0.0

    def update_threshold(self, scores: list[float], feedback: list[bool] = None):
        """Update threshold based on recent scores and feedback."""

        self.recent_scores.extend(scores)

        if len(self.recent_scores) < 50:
            return

        # Calculate percentile-based threshold
        scores_array = np.array(self.recent_scores)
        percentile_threshold = np.percentile(scores_array, 90)

        # Adapt based on feedback if available
        if feedback:
            self._adapt_with_feedback(scores, feedback)
        else:
            # Auto-adapt based on score distribution
            self._auto_adapt(percentile_threshold)

    def _adapt_with_feedback(self, scores: list[float], feedback: list[bool]):
        """Adapt threshold based on user feedback."""

        true_anomalies = [s for s, f in zip(scores, feedback, strict=False) if f]
        false_positives = [
            s
            for s, f in zip(scores, feedback, strict=False)
            if not f and s > self.threshold
        ]

        if true_anomalies:
            # Lower threshold if we're missing true anomalies
            min_true_anomaly = min(true_anomalies)
            if min_true_anomaly < self.threshold:
                self.threshold = max(0.01, self.threshold - self.adaptation_rate)

        if false_positives:
            # Raise threshold if we have too many false positives
            max_false_positive = max(false_positives)
            if max_false_positive > self.threshold:
                self.threshold = min(0.99, self.threshold + self.adaptation_rate)

    def _auto_adapt(self, percentile_threshold: float):
        """Auto-adapt threshold based on score distribution."""

        # Gradually move toward percentile-based threshold
        target_threshold = percentile_threshold * 0.8  # Be slightly more conservative

        if abs(target_threshold - self.threshold) > 0.05:
            direction = 1 if target_threshold > self.threshold else -1
            self.threshold += direction * self.adaptation_rate


class StreamingAnomalyDetector:
    """Real-time streaming anomaly detector."""

    def __init__(self, config: StreamingDetectorConfig):
        self.config = config
        self.logger = StructuredLogger(f"streaming_detector_{config.detector_id}")

        # Core components
        self.model = self._initialize_model()
        self.scaler = StandardScaler()
        self.threshold_manager = AdaptiveThresholdManager(
            config.anomaly_threshold, config.adaptation_rate
        )

        # Streaming data management
        self.data_buffer = deque(maxlen=config.window_size)
        self.recent_scores = deque(maxlen=config.window_size)
        self.batch_buffer = []

        # Model state
        self.is_trained = False
        self.last_update_time = datetime.now()
        self.samples_processed = 0
        self.samples_since_update = 0

        # Performance tracking
        self.performance_stats = {
            "total_processed": 0,
            "anomalies_detected": 0,
            "average_processing_time": 0.0,
            "model_updates": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        # Callbacks
        self.anomaly_callbacks: list[Callable[[AnomalyAlert], None]] = []

    def _initialize_model(self):
        """Initialize the anomaly detection model."""

        if self.config.detector_type == "isolation_forest":
            return IsolationForest(
                contamination=self.config.anomaly_threshold, random_state=42, n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported detector type: {self.config.detector_type}")

    async def process_data_point(
        self, data_point: StreamingDataPoint
    ) -> AnomalyAlert | None:
        """Process a single data point and detect anomalies."""

        start_time = time.time()

        try:
            # Add to buffer
            self.data_buffer.append(data_point)
            self.batch_buffer.append(data_point)
            self.samples_processed += 1
            self.samples_since_update += 1

            # Initial training if needed
            if not self.is_trained and len(self.data_buffer) >= 100:
                await self._initial_training()

            if not self.is_trained:
                return None

            # Process in micro-batches for efficiency
            if len(self.batch_buffer) >= self.config.batch_size:
                alerts = await self._process_batch()
                self.batch_buffer = []

                # Find alert for current data point
                current_alert = None
                for alert in alerts:
                    if alert.data_point.correlation_id == data_point.correlation_id:
                        current_alert = alert
                        break

                processing_time = time.time() - start_time
                self._update_performance_stats(processing_time)

                return current_alert

            # Single point processing for real-time mode
            if self.config.detector_type == "real_time":
                alert = await self._process_single_point(data_point)

                processing_time = time.time() - start_time
                self._update_performance_stats(processing_time)

                return alert

            return None

        except Exception as e:
            self.logger.error(f"Error processing data point: {e}")
            return None

    async def _process_batch(self) -> list[AnomalyAlert]:
        """Process a batch of data points."""

        if not self.batch_buffer:
            return []

        # Extract features
        features = np.array([dp.features for dp in self.batch_buffer])

        # Normalize features
        try:
            features_scaled = self.scaler.transform(features)
        except Exception:
            # If scaler not fitted or shape mismatch, refit
            features_scaled = self.scaler.fit_transform(features)

        # Get anomaly scores
        anomaly_scores = self.model.decision_function(features_scaled)

        # Convert to probabilities (0-1 range)
        probabilities = self._scores_to_probabilities(anomaly_scores)

        # Update threshold
        self.threshold_manager.update_threshold(probabilities.tolist())

        # Track scores
        self.recent_scores.extend(probabilities)

        # Generate alerts
        alerts = []
        for i, (data_point, score) in enumerate(
            zip(self.batch_buffer, probabilities, strict=False)
        ):
            if score > self.threshold_manager.threshold:
                alert = await self._create_anomaly_alert(data_point, score)
                alerts.append(alert)

                # Trigger callbacks
                for callback in self.anomaly_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        self.logger.error(f"Error in anomaly callback: {e}")

        # Check if model needs updating
        if self.samples_since_update >= self.config.update_frequency:
            await self._update_model()

        return alerts

    async def _process_single_point(
        self, data_point: StreamingDataPoint
    ) -> AnomalyAlert | None:
        """Process a single data point for real-time detection."""

        features = data_point.features.reshape(1, -1)

        try:
            features_scaled = self.scaler.transform(features)
        except Exception:
            return None

        # Get anomaly score
        anomaly_score = self.model.decision_function(features_scaled)[0]
        probability = self._scores_to_probabilities(np.array([anomaly_score]))[0]

        # Track score
        self.recent_scores.append(probability)

        # Check for anomaly
        if probability > self.threshold_manager.threshold:
            alert = await self._create_anomaly_alert(data_point, probability)

            # Trigger callbacks
            for callback in self.anomaly_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in anomaly callback: {e}")

            return alert

        return None

    async def _initial_training(self):
        """Perform initial model training."""

        if len(self.data_buffer) < 100:
            return

        self.logger.info("Starting initial model training")

        # Extract features from buffer
        features = np.array([dp.features for dp in self.data_buffer])

        # Fit scaler and model
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)

        self.is_trained = True
        self.last_update_time = datetime.now()
        self.performance_stats["model_updates"] += 1

        self.logger.info("Initial model training completed")

    async def _update_model(self):
        """Update model with recent data."""

        if len(self.data_buffer) < 100:
            return

        self.logger.info("Updating anomaly detection model")

        # Get recent data
        recent_data = list(self.data_buffer)[-self.config.update_frequency :]
        features = np.array([dp.features for dp in recent_data])

        # Update scaler incrementally (simplified approach)
        try:
            # Partial fit on recent data
            features_scaled = self.scaler.fit_transform(features)

            # Retrain model with recent data
            self.model.fit(features_scaled)

            self.last_update_time = datetime.now()
            self.samples_since_update = 0
            self.performance_stats["model_updates"] += 1

            self.logger.info("Model update completed")

        except Exception as e:
            self.logger.error(f"Error updating model: {e}")

    async def _create_anomaly_alert(
        self, data_point: StreamingDataPoint, anomaly_score: float
    ) -> AnomalyAlert:
        """Create an anomaly alert."""

        # Determine threat level
        threat_level = self._calculate_threat_level(anomaly_score)

        # Calculate confidence
        confidence = min(anomaly_score * 2, 1.0)  # Simple confidence calculation

        # Generate explanation
        explanation = self._generate_explanation(data_point, anomaly_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(threat_level, data_point)

        # Identify affected features
        affected_features = self._identify_affected_features(data_point)

        # Create historical context
        historical_context = self._create_historical_context()

        alert = AnomalyAlert(
            alert_id=f"alert_{self.config.detector_id}_{int(time.time())}",
            timestamp=data_point.timestamp,
            data_point=data_point,
            anomaly_score=anomaly_score,
            threat_level=threat_level,
            confidence=confidence,
            explanation=explanation,
            recommendations=recommendations,
            affected_features=affected_features,
            historical_context=historical_context,
        )

        self.performance_stats["anomalies_detected"] += 1

        return alert

    def _calculate_threat_level(self, anomaly_score: float) -> AnomalyThreatLevel:
        """Calculate threat level based on anomaly score."""

        thresholds = self.config.threat_level_thresholds

        if anomaly_score >= thresholds["critical"]:
            return AnomalyThreatLevel.CRITICAL
        elif anomaly_score >= thresholds["high"]:
            return AnomalyThreatLevel.HIGH
        elif anomaly_score >= thresholds["medium"]:
            return AnomalyThreatLevel.MEDIUM
        elif anomaly_score >= thresholds["low"]:
            return AnomalyThreatLevel.LOW
        else:
            return AnomalyThreatLevel.NORMAL

    def _generate_explanation(
        self, data_point: StreamingDataPoint, score: float
    ) -> str:
        """Generate human-readable explanation for the anomaly."""

        threshold = self.threshold_manager.threshold
        severity = "significant" if score > threshold * 2 else "moderate"

        return (
            f"Detected {severity} anomaly with score {score:.3f} "
            f"(threshold: {threshold:.3f}). "
            f"Data point deviates from normal patterns observed in recent history."
        )

    def _generate_recommendations(
        self, threat_level: AnomalyThreatLevel, data_point: StreamingDataPoint
    ) -> list[str]:
        """Generate recommendations based on threat level."""

        recommendations = []

        if threat_level == AnomalyThreatLevel.CRITICAL:
            recommendations.extend(
                [
                    "IMMEDIATE ACTION REQUIRED: Investigate this anomaly urgently",
                    "Consider shutting down affected systems if necessary",
                    "Escalate to security team for potential breach investigation",
                    "Review recent system changes and access logs",
                ]
            )
        elif threat_level == AnomalyThreatLevel.HIGH:
            recommendations.extend(
                [
                    "High priority investigation recommended",
                    "Review system metrics and performance indicators",
                    "Check for related anomalies in the same time period",
                    "Consider increasing monitoring frequency",
                ]
            )
        elif threat_level == AnomalyThreatLevel.MEDIUM:
            recommendations.extend(
                [
                    "Monitor closely for additional anomalies",
                    "Review data source for potential issues",
                    "Consider adjusting normal behavior baselines",
                ]
            )
        else:
            recommendations.extend(
                ["Log for trend analysis", "Monitor for pattern changes over time"]
            )

        return recommendations

    def _identify_affected_features(self, data_point: StreamingDataPoint) -> list[str]:
        """Identify which features contributed most to the anomaly."""

        # Simplified approach - in practice, you'd use feature importance
        # or SHAP values for better explanation

        if len(self.data_buffer) < 10:
            return []

        recent_features = np.array([dp.features for dp in list(self.data_buffer)[-10:]])
        current_features = data_point.features

        # Calculate feature-wise deviations
        feature_means = np.mean(recent_features, axis=0)
        feature_stds = np.std(recent_features, axis=0)

        # Identify features with high deviation
        deviations = np.abs((current_features - feature_means) / (feature_stds + 1e-6))

        # Get top contributing features
        top_features = np.argsort(deviations)[-3:]  # Top 3 features

        return [f"feature_{i}" for i in top_features]

    def _create_historical_context(self) -> dict[str, Any]:
        """Create historical context for the alert."""

        context = {
            "recent_anomaly_rate": self._calculate_recent_anomaly_rate(),
            "model_last_updated": self.last_update_time.isoformat(),
            "samples_processed": self.samples_processed,
            "current_threshold": self.threshold_manager.threshold,
        }

        if len(self.recent_scores) > 0:
            scores_array = np.array(self.recent_scores)
            context.update(
                {
                    "recent_score_mean": float(np.mean(scores_array)),
                    "recent_score_std": float(np.std(scores_array)),
                    "recent_score_max": float(np.max(scores_array)),
                }
            )

        return context

    def _calculate_recent_anomaly_rate(self) -> float:
        """Calculate the rate of anomalies in recent data."""

        if len(self.recent_scores) == 0:
            return 0.0

        threshold = self.threshold_manager.threshold
        anomaly_count = sum(1 for score in self.recent_scores if score > threshold)

        return anomaly_count / len(self.recent_scores)

    def _scores_to_probabilities(self, scores: np.ndarray) -> np.ndarray:
        """Convert anomaly scores to probabilities."""

        # Isolation Forest returns negative scores for anomalies
        # Convert to probabilities (0-1 range)

        # Normalize scores to 0-1 range
        min_score = np.min(scores)
        max_score = np.max(scores)

        if max_score == min_score:
            return np.zeros_like(scores)

        normalized = (scores - min_score) / (max_score - min_score)

        # Invert for Isolation Forest (lower scores = more anomalous)
        probabilities = 1 - normalized

        return probabilities

    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics."""

        total_processed = self.performance_stats["total_processed"]
        current_avg = self.performance_stats["average_processing_time"]

        # Update rolling average
        new_avg = (current_avg * total_processed + processing_time) / (
            total_processed + 1
        )

        self.performance_stats["total_processed"] += 1
        self.performance_stats["average_processing_time"] = new_avg

    def add_anomaly_callback(self, callback: Callable[[AnomalyAlert], None]):
        """Add callback function to be called when anomalies are detected."""
        self.anomaly_callbacks.append(callback)

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.performance_stats,
            "current_threshold": self.threshold_manager.threshold,
            "buffer_size": len(self.data_buffer),
            "is_trained": self.is_trained,
            "recent_anomaly_rate": self._calculate_recent_anomaly_rate(),
        }


class StreamingAnomalyDetectionService:
    """Service for managing multiple streaming anomaly detectors."""

    def __init__(self):
        self.logger = StructuredLogger("streaming_anomaly_service")
        self.metrics_service = MetricsService()

        # Detector management
        self.detectors: dict[str, StreamingAnomalyDetector] = {}
        self.detector_configs: dict[str, StreamingDetectorConfig] = {}

        # Global callbacks
        self.global_callbacks: list[Callable[[AnomalyAlert], None]] = []

        # Performance tracking
        self.service_stats = {
            "total_alerts": 0,
            "active_detectors": 0,
            "total_data_points": 0,
        }

    async def create_detector(self, config: StreamingDetectorConfig) -> str:
        """Create a new streaming anomaly detector."""

        if config.detector_id in self.detectors:
            raise ValueError(f"Detector {config.detector_id} already exists")

        detector = StreamingAnomalyDetector(config)

        # Add global callbacks to detector
        for callback in self.global_callbacks:
            detector.add_anomaly_callback(callback)

        # Add service-level callback for tracking
        detector.add_anomaly_callback(self._on_anomaly_detected)

        self.detectors[config.detector_id] = detector
        self.detector_configs[config.detector_id] = config

        self.service_stats["active_detectors"] = len(self.detectors)

        self.logger.info(f"Created streaming detector: {config.detector_id}")

        return config.detector_id

    async def process_data_stream(
        self, detector_id: str, data_points: list[StreamingDataPoint]
    ) -> list[AnomalyAlert]:
        """Process a stream of data points through a detector."""

        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")

        detector = self.detectors[detector_id]
        alerts = []

        for data_point in data_points:
            alert = await detector.process_data_point(data_point)
            if alert:
                alerts.append(alert)

            self.service_stats["total_data_points"] += 1

        return alerts

    async def process_single_point(
        self, detector_id: str, data_point: StreamingDataPoint
    ) -> AnomalyAlert | None:
        """Process a single data point through a detector."""

        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")

        detector = self.detectors[detector_id]
        alert = await detector.process_data_point(data_point)

        self.service_stats["total_data_points"] += 1

        return alert

    def _on_anomaly_detected(self, alert: AnomalyAlert):
        """Handle anomaly detection at service level."""

        self.service_stats["total_alerts"] += 1

        # Record metrics
        self.metrics_service.record_anomaly_detection(
            detector_id=alert.alert_id.split("_")[1],  # Extract detector ID
            threat_level=alert.threat_level.value,
            anomaly_score=alert.anomaly_score,
            confidence=alert.confidence,
        )

        self.logger.info(
            f"Anomaly detected: {alert.alert_id}, "
            f"Threat: {alert.threat_level.value}, "
            f"Score: {alert.anomaly_score:.3f}"
        )

    def add_global_callback(self, callback: Callable[[AnomalyAlert], None]):
        """Add global callback for all detectors."""

        self.global_callbacks.append(callback)

        # Add to existing detectors
        for detector in self.detectors.values():
            detector.add_anomaly_callback(callback)

    def get_detector_stats(self, detector_id: str) -> dict[str, Any]:
        """Get statistics for a specific detector."""

        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")

        return self.detectors[detector_id].get_performance_stats()

    def get_service_stats(self) -> dict[str, Any]:
        """Get service-level statistics."""

        return {
            **self.service_stats,
            "detector_stats": {
                detector_id: detector.get_performance_stats()
                for detector_id, detector in self.detectors.items()
            },
        }

    async def shutdown_detector(self, detector_id: str):
        """Shutdown a specific detector."""

        if detector_id in self.detectors:
            del self.detectors[detector_id]
            del self.detector_configs[detector_id]

            self.service_stats["active_detectors"] = len(self.detectors)

            self.logger.info(f"Shutdown detector: {detector_id}")

    async def shutdown_all(self):
        """Shutdown all detectors."""

        for detector_id in list(self.detectors.keys()):
            await self.shutdown_detector(detector_id)

        self.logger.info("All streaming detectors shutdown")


# Example usage and helper functions
async def create_sample_streaming_service() -> StreamingAnomalyDetectionService:
    """Create a sample streaming anomaly detection service."""

    service = StreamingAnomalyDetectionService()

    # Create a default detector
    config = StreamingDetectorConfig(
        detector_id="default_detector",
        detector_type="isolation_forest",
        window_size=1000,
        batch_size=50,
        update_frequency=100,
        anomaly_threshold=0.1,
    )

    await service.create_detector(config)

    return service


def create_sample_data_point(
    features: list[float], source_id: str = "default"
) -> StreamingDataPoint:
    """Create a sample data point for testing."""

    return StreamingDataPoint(
        timestamp=datetime.now(),
        features=np.array(features),
        source_id=source_id,
        correlation_id=f"{source_id}_{int(time.time())}",
    )
