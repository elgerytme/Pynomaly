"""Intelligent alert management service with ML-powered noise reduction."""

import asyncio
import logging
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from monorepo.domain.entities.alert import (
    Alert,
    AlertCategory,
    AlertCorrelation,
    AlertMetadata,
    AlertSeverity,
    AlertSource,
    AlertStatus,
    MLNoiseFeatures,
    NoiseClassification,
)

logger = logging.getLogger(__name__)


class AlertCorrelationEngine:
    """Intelligent alert correlation using ML techniques."""

    def __init__(self):
        self.alert_history: dict[UUID, Alert] = {}
        self.feature_cache: dict[UUID, np.ndarray] = {}
        self.correlation_patterns: dict[str, list[UUID]] = defaultdict(list)

    def add_alert(self, alert: Alert):
        """Add alert to correlation engine."""
        self.alert_history[alert.alert_id] = alert

        # Extract features for correlation
        features = self._extract_correlation_features(alert)
        self.feature_cache[alert.alert_id] = features

    def find_correlations(
        self, alert: Alert, time_window_minutes: int = 60
    ) -> list[AlertCorrelation]:
        """Find correlations for a given alert."""
        correlations = []

        # Time-based correlation
        temporal_correlations = self._find_temporal_correlations(
            alert, time_window_minutes
        )
        correlations.extend(temporal_correlations)

        # Pattern-based correlation
        pattern_correlations = self._find_pattern_correlations(alert)
        correlations.extend(pattern_correlations)

        # Causal correlation
        causal_correlations = self._find_causal_correlations(alert)
        correlations.extend(causal_correlations)

        return correlations

    def _find_temporal_correlations(
        self, alert: Alert, time_window_minutes: int
    ) -> list[AlertCorrelation]:
        """Find alerts that occurred in close temporal proximity."""
        correlations = []
        time_threshold = timedelta(minutes=time_window_minutes)

        for other_alert_id, other_alert in self.alert_history.items():
            if other_alert_id == alert.alert_id:
                continue

            time_diff = abs(alert.created_at - other_alert.created_at)

            if time_diff <= time_threshold:
                # Calculate correlation strength based on time proximity
                correlation_strength = 1.0 - (
                    time_diff.total_seconds() / (time_window_minutes * 60)
                )

                correlation = AlertCorrelation(
                    related_alerts={other_alert_id},
                    correlation_type="temporal",
                    correlation_strength=correlation_strength,
                    correlation_reason=f"Occurred within {time_diff.total_seconds():.0f}s of each other",
                    time_window_minutes=time_window_minutes,
                )
                correlations.append(correlation)

        return correlations

    def _find_pattern_correlations(self, alert: Alert) -> list[AlertCorrelation]:
        """Find alerts with similar patterns using ML clustering."""
        correlations = []

        if alert.alert_id not in self.feature_cache:
            return correlations

        alert_features = self.feature_cache[alert.alert_id]
        similar_alerts = []

        for other_alert_id, other_features in self.feature_cache.items():
            if other_alert_id == alert.alert_id:
                continue

            # Calculate cosine similarity
            similarity = cosine_similarity([alert_features], [other_features])[0][0]

            if similarity > 0.7:  # High similarity threshold
                similar_alerts.append((other_alert_id, similarity))

        if similar_alerts:
            # Group by similarity ranges
            similar_alerts.sort(key=lambda x: x[1], reverse=True)

            correlation = AlertCorrelation(
                related_alerts={aid for aid, _ in similar_alerts[:5]},  # Top 5 similar
                correlation_type="pattern",
                correlation_strength=statistics.mean(
                    [sim for _, sim in similar_alerts[:5]]
                ),
                correlation_reason=f"Similar patterns detected (avg similarity: {statistics.mean([sim for _, sim in similar_alerts[:5]]):.3f})",
                pattern_similarity=statistics.mean(
                    [sim for _, sim in similar_alerts[:5]]
                ),
                feature_overlap=self._calculate_feature_overlap(
                    alert_features, [feat for _, feat in similar_alerts[:5]]
                ),
            )
            correlations.append(correlation)

        return correlations

    def _find_causal_correlations(self, alert: Alert) -> list[AlertCorrelation]:
        """Find potential causal relationships between alerts."""
        correlations = []

        # Look for alerts that might be root causes
        potential_causes = []

        for other_alert_id, other_alert in self.alert_history.items():
            if other_alert_id == alert.alert_id:
                continue

            # Check if other alert occurred before this one
            if other_alert.created_at < alert.created_at:
                time_diff = alert.created_at - other_alert.created_at

                # Look for specific causal patterns
                if self._is_potential_cause(other_alert, alert):
                    causal_strength = self._calculate_causal_strength(
                        other_alert, alert, time_diff
                    )
                    potential_causes.append((other_alert_id, causal_strength))

        if potential_causes:
            # Sort by causal strength
            potential_causes.sort(key=lambda x: x[1], reverse=True)

            correlation = AlertCorrelation(
                related_alerts={aid for aid, _ in potential_causes[:3]},  # Top 3 causes
                correlation_type="causal",
                correlation_strength=statistics.mean(
                    [strength for _, strength in potential_causes[:3]]
                ),
                correlation_reason="Potential causal relationship detected",
                causal_chain=[aid for aid, _ in potential_causes],
                root_cause_alert=potential_causes[0][0] if potential_causes else None,
            )
            correlations.append(correlation)

        return correlations

    def _extract_correlation_features(self, alert: Alert) -> np.ndarray:
        """Extract features for correlation analysis."""
        features = []

        # Basic features
        features.append(float(alert.severity.value == "critical"))
        features.append(float(alert.severity.value == "high"))
        features.append(float(alert.category.value == "anomaly_processing"))
        features.append(float(alert.category.value == "system_performance"))
        features.append(float(alert.source.value == "detector"))

        # Metadata features
        features.append(alert.metadata.anomaly_score or 0.0)
        features.append(alert.metadata.confidence_level or 0.0)
        features.append(alert.metadata.system_load or 0.0)
        features.append(alert.metadata.memory_usage or 0.0)
        features.append(alert.metadata.cpu_usage or 0.0)

        # Temporal features
        hour_of_day = alert.created_at.hour
        features.append(np.sin(2 * np.pi * hour_of_day / 24))  # Cyclical encoding
        features.append(np.cos(2 * np.pi * hour_of_day / 24))

        day_of_week = alert.created_at.weekday()
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))

        # Resource impact
        features.append(len(alert.metadata.affected_resources))
        features.append(len(alert.metadata.related_measurements))
        features.append(float(alert.metadata.customer_affected or False))

        return np.array(features)

    def _calculate_feature_overlap(
        self, features1: np.ndarray, features_list: list[np.ndarray]
    ) -> float:
        """Calculate feature overlap between alerts."""
        if not features_list:
            return 0.0

        overlaps = []
        for features2 in features_list:
            # Calculate element-wise similarity
            overlap = np.mean(1.0 - np.abs(features1 - features2))
            overlaps.append(overlap)

        return statistics.mean(overlaps)

    def _is_potential_cause(self, potential_cause: Alert, effect_alert: Alert) -> bool:
        """Determine if one alert could be the cause of another."""
        # Infrastructure issues often cause application issues
        if (
            potential_cause.category == AlertCategory.INFRASTRUCTURE
            and effect_alert.category == AlertCategory.ANOMALY_DETECTION
        ):
            return True

        # System performance issues can cause anomalies
        if (
            potential_cause.category == AlertCategory.SYSTEM_PERFORMANCE
            and effect_alert.category == AlertCategory.ANOMALY_DETECTION
        ):
            return True

        # Security issues can cascade
        if (
            potential_cause.category == AlertCategory.SECURITY
            and effect_alert.category
            in [AlertCategory.SYSTEM_PERFORMANCE, AlertCategory.ANOMALY_DETECTION]
        ):
            return True

        # Same tenant issues
        if (
            potential_cause.metadata.tenant_id
            and effect_alert.metadata.tenant_id
            and potential_cause.metadata.tenant_id == effect_alert.metadata.tenant_id
        ):
            return True

        return False

    def _calculate_causal_strength(
        self, cause: Alert, effect: Alert, time_diff: timedelta
    ) -> float:
        """Calculate the strength of a potential causal relationship."""
        strength = 0.0

        # Time proximity (closer in time = stronger causality)
        max_causal_window = timedelta(hours=1)
        if time_diff <= max_causal_window:
            strength += 0.4 * (
                1.0 - time_diff.total_seconds() / max_causal_window.total_seconds()
            )

        # Severity relationship (higher severity cause = stronger)
        severity_weights = {
            AlertSeverity.CRITICAL: 1.0,
            AlertSeverity.HIGH: 0.8,
            AlertSeverity.MEDIUM: 0.6,
            AlertSeverity.LOW: 0.4,
            AlertSeverity.INFO: 0.2,
        }
        strength += 0.3 * severity_weights.get(cause.severity, 0.5)

        # Category relationship
        if self._is_potential_cause(cause, effect):
            strength += 0.3

        return min(1.0, strength)


class NoiseClassificationModel:
    """ML processor for classifying alerts as signal vs noise."""

    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance: dict[str, float] = {}

        # Training data storage
        self.training_features: list[list[float]] = []
        self.training_labels: list[int] = []  # 1 = signal, 0 = noise

    def extract_features(
        self, alert: Alert, alert_history: list[Alert]
    ) -> MLNoiseFeatures:
        """Extract features for noise classification."""
        features = MLNoiseFeatures()

        # Temporal features
        features.hour_of_day = alert.created_at.hour
        features.day_of_week = alert.created_at.weekday()
        features.is_business_hours = 9 <= alert.created_at.hour <= 17
        features.is_weekend = alert.created_at.weekday() >= 5

        # Alert frequency features
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        one_week_ago = now - timedelta(weeks=1)

        features.alerts_last_hour = len(
            [a for a in alert_history if a.created_at >= one_hour_ago]
        )
        features.alerts_last_day = len(
            [a for a in alert_history if a.created_at >= one_day_ago]
        )

        # Similar alerts in the last week
        similar_alerts = [
            a
            for a in alert_history
            if (
                a.created_at >= one_week_ago
                and a.category == alert.category
                and a.severity == alert.severity
            )
        ]
        features.similar_alerts_last_week = len(similar_alerts)

        # System context features
        features.system_load_percentile = self._calculate_percentile(
            alert.metadata.system_load or 0.0,
            [a.metadata.system_load for a in alert_history if a.metadata.system_load],
        )
        features.memory_pressure = alert.metadata.memory_usage or 0.0
        features.cpu_utilization = alert.metadata.cpu_usage or 0.0

        # Processing quality features
        features.processor_confidence = alert.metadata.confidence_level or 0.0
        features.anomaly_score_percentile = self._calculate_percentile(
            alert.metadata.anomaly_score or 0.0,
            [
                a.metadata.anomaly_score
                for a in alert_history
                if a.metadata.anomaly_score
            ],
        )

        # Historical context features
        resolved_alerts = [a for a in alert_history if a.status == AlertStatus.RESOLVED]
        if resolved_alerts:
            false_positives = [
                a
                for a in resolved_alerts
                if a.resolution_quality and a.resolution_quality < 0.5
            ]
            features.false_positive_rate_7d = len(false_positives) / len(
                resolved_alerts
            )

            resolution_times = [
                a.time_to_resolve.total_seconds() / 60
                for a in resolved_alerts
                if a.time_to_resolve
            ]
            features.resolution_time_avg = (
                statistics.mean(resolution_times) if resolution_times else 0.0
            )

        # Correlation features
        if alert.correlation:
            features.has_correlated_alerts = len(alert.correlation.related_alerts) > 0
            features.correlation_strength_max = alert.correlation.correlation_strength
            features.num_related_alerts = len(alert.correlation.related_alerts)

        return features

    def predict_noise_probability(
        self, features: MLNoiseFeatures
    ) -> tuple[NoiseClassification, float]:
        """Predict if an alert is noise and return confidence."""
        if not self.is_trained:
            # Use heuristic rules if processor isn't trained
            return self._heuristic_classification(features)

        # Use trained ML processor
        feature_vector = np.array(features.to_feature_vector()).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(feature_vector_scaled)[0]
        noise_probability = probabilities[0]  # Probability of being noise
        signal_probability = probabilities[1]  # Probability of being signal

        # Determine classification
        if signal_probability > 0.7:
            classification = NoiseClassification.SIGNAL
            confidence = signal_probability
        elif noise_probability > 0.7:
            classification = NoiseClassification.NOISE
            confidence = noise_probability
        else:
            classification = NoiseClassification.UNKNOWN
            confidence = max(signal_probability, noise_probability)

        return classification, confidence

    def add_training_sample(self, features: MLNoiseFeatures, is_signal: bool):
        """Add a training sample to improve the processor."""
        self.training_features.append(features.to_feature_vector())
        self.training_labels.append(1 if is_signal else 0)

        # Retrain if we have enough samples
        if len(self.training_features) >= 100 and len(self.training_features) % 50 == 0:
            self._retrain_processor()

    def _retrain_model(self):
        """Retrain the classification processor with accumulated data."""
        if len(self.training_features) < 50:
            return

        try:
            X = np.array(self.training_features)
            y = np.array(self.training_labels)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train classifier
            self.classifier.fit(X_scaled, y)
            self.is_trained = True

            # Store feature importance
            feature_names = [
                "hour_of_day",
                "day_of_week",
                "is_business_hours",
                "is_weekend",
                "alerts_last_hour",
                "alerts_last_day",
                "similar_alerts_last_week",
                "system_load_percentile",
                "memory_pressure",
                "cpu_utilization",
                "processor_confidence",
                "anomaly_score_percentile",
                "feature_stability",
                "false_positive_rate_7d",
                "resolution_time_avg",
                "escalation_frequency",
                "has_correlated_alerts",
                "correlation_strength_max",
                "num_related_alerts",
            ]

            self.feature_importance = dict(
                zip(feature_names, self.classifier.feature_importances_, strict=False)
            )

            logger.info(
                f"Retrained noise classification processor with {len(self.training_features)} samples"
            )

        except Exception as e:
            logger.error(f"Error retraining noise classification processor: {e}")

    def _heuristic_classification(
        self, features: MLNoiseFeatures
    ) -> tuple[NoiseClassification, float]:
        """Fallback heuristic classification when ML processor isn't available."""
        score = 0.0

        # High frequency of similar alerts suggests noise
        if features.similar_alerts_last_week > 10:
            score += 0.3

        # Very high alert frequency suggests storm/noise
        if features.alerts_last_hour > 20:
            score += 0.4

        # Low confidence suggests noise
        if features.processor_confidence < 0.3:
            score += 0.2

        # During non-business hours with no correlation
        if not features.is_business_hours and not features.has_correlated_alerts:
            score += 0.1

        # High false positive rate suggests noise
        if features.false_positive_rate_7d > 0.5:
            score += 0.2

        # Classify based on score
        if score > 0.6:
            return NoiseClassification.NOISE, score
        elif score < 0.3:
            return NoiseClassification.SIGNAL, 1.0 - score
        else:
            return NoiseClassification.UNKNOWN, 0.5

    def _calculate_percentile(
        self, value: float, historical_values: list[float]
    ) -> float:
        """Calculate percentile of a value within historical values."""
        if not historical_values:
            return 0.5

        historical_values = [v for v in historical_values if v is not None]
        if not historical_values:
            return 0.5

        sorted_values = sorted(historical_values)
        rank = sum(1 for v in sorted_values if v <= value)
        return rank / len(sorted_values)


class IntelligentAlertService:
    """Main service for intelligent alert management with ML-powered noise reduction."""

    def __init__(self):
        self.alerts: dict[UUID, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)  # Keep last 10k alerts
        self.correlation_engine = AlertCorrelationEngine()
        self.noise_classifier = NoiseClassificationModel()

        # Alert processing queues
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.suppression_rules: dict[str, dict] = {}

        # Performance measurements
        self.measurements = {
            "total_alerts": 0,
            "suppressed_alerts": 0,
            "correlated_alerts": 0,
            "noise_classifications": defaultdict(int),
            "processing_times": [],
        }

    async def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        category: AlertCategory,
        source: AlertSource,
        metadata: AlertMetadata | None = None,
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> Alert:
        """Create a new alert with intelligent processing."""

        # Create base alert
        alert = Alert(
            title=title,
            description=description,
            severity=severity,
            category=category,
            source=source,
            message=message or description,
            details=details or {},
            metadata=metadata or AlertMetadata(),
        )

        # Add to processing queue for intelligent analysis
        await self.processing_queue.put(alert)

        # Store alert
        self.alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.measurements["total_alerts"] += 1

        return alert

    async def process_alert_intelligence(self, alert: Alert) -> Alert:
        """Apply intelligent processing to an alert."""
        start_time = datetime.utcnow()

        try:
            # Step 1: Find correlations
            correlations = self.correlation_engine.find_correlations(alert)
            if correlations:
                # Use the strongest correlation
                alert.correlation = max(
                    correlations, key=lambda c: c.correlation_strength
                )
                self.measurements["correlated_alerts"] += 1

            # Step 2: ML-based noise classification
            features = self.noise_classifier.extract_features(
                alert, list(self.alert_history)
            )
            (
                classification,
                confidence,
            ) = self.noise_classifier.predict_noise_probability(features)

            alert.update_ml_classification(classification, confidence, features)
            self.measurements["noise_classifications"][classification.value] += 1

            # Step 3: Apply intelligent suppression
            should_suppress, reason = await self._should_suppress_alert(alert)
            if should_suppress:
                alert.suppress(reason, "intelligent_system")
                self.measurements["suppressed_alerts"] += 1

            # Step 4: Add to correlation engine for future correlations
            self.correlation_engine.add_alert(alert)

            # Record processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.measurements["processing_times"].append(processing_time)

            logger.info(
                f"Processed alert {alert.alert_id} - Classification: {classification.value}, "
                f"Confidence: {confidence:.3f}, Suppressed: {should_suppress}"
            )

            return alert

        except Exception as e:
            logger.error(
                f"Error processing alert intelligence for {alert.alert_id}: {e}"
            )
            return alert

    async def _should_suppress_alert(self, alert: Alert) -> tuple[bool, str]:
        """Determine if an alert should be suppressed."""

        # Suppress if classified as noise with high confidence
        if (
            alert.noise_classification == NoiseClassification.NOISE
            and alert.noise_confidence > 0.8
        ):
            return (
                True,
                f"ML classified as noise (confidence: {alert.noise_confidence:.3f})",
            )

        # Suppress duplicates within time window
        duplicate_window = timedelta(minutes=5)
        for other_alert in reversed(list(self.alert_history)):
            if other_alert.alert_id == alert.alert_id:
                continue

            if (datetime.utcnow() - other_alert.created_at) > duplicate_window:
                break

            if self._are_alerts_duplicate(alert, other_alert):
                return True, f"Duplicate of alert {other_alert.alert_id}"

        # Suppress during maintenance windows
        if await self._is_maintenance_window():
            return True, "Suppressed during maintenance window"

        # Suppress if too many similar alerts
        similar_count = self._count_similar_alerts(alert, timedelta(hours=1))
        if similar_count > 10:
            return True, f"Too many similar alerts ({similar_count} in last hour)"

        return False, ""

    def _are_alerts_duplicate(self, alert1: Alert, alert2: Alert) -> bool:
        """Check if two alerts are duplicates."""
        # Same category, severity, and source
        if (
            alert1.category == alert2.category
            and alert1.severity == alert2.severity
            and alert1.source == alert2.source
        ):
            # Same tenant if applicable
            if (
                alert1.metadata.tenant_id
                and alert2.metadata.tenant_id
                and alert1.metadata.tenant_id == alert2.metadata.tenant_id
            ):
                return True

            # Same affected resources
            if (
                set(alert1.metadata.affected_resources)
                == set(alert2.metadata.affected_resources)
                and alert1.metadata.affected_resources
            ):
                return True

        return False

    async def _is_maintenance_window(self) -> bool:
        """Check if we're currently in a maintenance window."""
        # This would typically check against a maintenance schedule
        # For now, return False (no maintenance window)
        return False

    def _count_similar_alerts(self, alert: Alert, time_window: timedelta) -> int:
        """Count similar alerts within a time window."""
        cutoff_time = datetime.utcnow() - time_window
        count = 0

        for other_alert in reversed(list(self.alert_history)):
            if other_alert.created_at < cutoff_time:
                break

            if (
                other_alert.alert_id != alert.alert_id
                and other_alert.category == alert.category
                and other_alert.severity == alert.severity
            ):
                count += 1

        return count

    async def get_alert(self, alert_id: UUID) -> Alert | None:
        """Get alert by ID."""
        return self.alerts.get(alert_id)

    async def list_alerts(
        self,
        status_filter: AlertStatus | None = None,
        severity_filter: AlertSeverity | None = None,
        category_filter: AlertCategory | None = None,
        tenant_id_filter: UUID | None = None,
        limit: int = 100,
        include_suppressed: bool = False,
    ) -> list[Alert]:
        """List alerts with filtering options."""

        alerts = list(self.alerts.values())

        # Apply filters
        if status_filter:
            alerts = [a for a in alerts if a.status == status_filter]

        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]

        if category_filter:
            alerts = [a for a in alerts if a.category == category_filter]

        if tenant_id_filter:
            alerts = [a for a in alerts if a.metadata.tenant_id == tenant_id_filter]

        if not include_suppressed:
            alerts = [a for a in alerts if not a.is_suppressed()]

        # Sort by priority score (descending)
        alerts.sort(key=lambda a: a.calculate_priority_score(), reverse=True)

        return alerts[:limit]

    async def acknowledge_alert(
        self, alert_id: UUID, acknowledged_by: str, note: str = ""
    ) -> bool:
        """Acknowledge an alert."""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False

        alert.acknowledge(acknowledged_by, note)

        # Provide feedback to ML processor
        self.noise_classifier.add_training_sample(alert.ml_features, is_signal=True)

        return True

    async def resolve_alert(
        self,
        alert_id: UUID,
        resolved_by: str,
        resolution_note: str = "",
        quality_score: float | None = None,
    ) -> bool:
        """Resolve an alert."""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False

        alert.resolve(resolved_by, resolution_note, quality_score)

        # Provide feedback to ML processor
        is_signal = quality_score is None or quality_score > 0.5
        self.noise_classifier.add_training_sample(
            alert.ml_features, is_signal=is_signal
        )

        return True

    async def suppress_alert(
        self,
        alert_id: UUID,
        suppressed_by: str,
        reason: str = "",
        duration_minutes: int | None = None,
    ) -> bool:
        """Manually suppress an alert."""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False

        alert.suppress(reason, suppressed_by, duration_minutes)

        # Provide feedback to ML processor (suppressed alerts are likely noise)
        self.noise_classifier.add_training_sample(alert.ml_features, is_signal=False)

        return True

    async def escalate_alert(
        self, alert_id: UUID, escalated_by: str, reason: str = ""
    ) -> bool:
        """Escalate an alert."""
        alert = self.alerts.get(alert_id)
        if not alert:
            return False

        alert.escalate(escalated_by, reason)

        # Escalated alerts are definitely signals
        self.noise_classifier.add_training_sample(alert.ml_features, is_signal=True)

        return True

    async def get_alert_analytics(self, days: int = 7) -> dict[str, Any]:
        """Get comprehensive alert analytics."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_alerts = [a for a in self.alert_history if a.created_at >= cutoff_date]

        analytics = {
            "total_alerts": len(recent_alerts),
            "alert_distribution": {
                "by_severity": defaultdict(int),
                "by_category": defaultdict(int),
                "by_status": defaultdict(int),
                "by_noise_classification": defaultdict(int),
            },
            "noise_reduction_stats": {
                "total_suppressed": len(
                    [a for a in recent_alerts if a.is_suppressed()]
                ),
                "ml_classified_noise": len(
                    [
                        a
                        for a in recent_alerts
                        if a.noise_classification == NoiseClassification.NOISE
                    ]
                ),
                "signal_to_noise_ratio": 0.0,
                "avg_noise_confidence": 0.0,
            },
            "correlation_stats": {
                "correlated_alerts": len([a for a in recent_alerts if a.correlation]),
                "avg_correlation_strength": 0.0,
                "correlation_types": defaultdict(int),
            },
            "performance_measurements": {
                "avg_processing_time": (
                    statistics.mean(self.measurements["processing_times"][-1000:])
                    if self.measurements["processing_times"]
                    else 0.0
                ),
                "total_processed": self.measurements["total_alerts"],
                "suppression_rate": self.measurements["suppressed_alerts"]
                / max(1, self.measurements["total_alerts"]),
            },
        }

        # Calculate distributions
        for alert in recent_alerts:
            analytics["alert_distribution"]["by_severity"][alert.severity.value] += 1
            analytics["alert_distribution"]["by_category"][alert.category.value] += 1
            analytics["alert_distribution"]["by_status"][alert.status.value] += 1
            analytics["alert_distribution"]["by_noise_classification"][
                alert.noise_classification.value
            ] += 1

        # Calculate noise reduction stats
        noise_alerts = [
            a
            for a in recent_alerts
            if a.noise_classification == NoiseClassification.NOISE
        ]
        signal_alerts = [
            a
            for a in recent_alerts
            if a.noise_classification == NoiseClassification.SIGNAL
        ]

        if noise_alerts:
            analytics["noise_reduction_stats"]["avg_noise_confidence"] = (
                statistics.mean([a.noise_confidence for a in noise_alerts])
            )

        if signal_alerts and noise_alerts:
            analytics["noise_reduction_stats"]["signal_to_noise_ratio"] = len(
                signal_alerts
            ) / len(noise_alerts)

        # Calculate correlation stats
        correlated_alerts = [a for a in recent_alerts if a.correlation]
        if correlated_alerts:
            analytics["correlation_stats"]["avg_correlation_strength"] = (
                statistics.mean(
                    [a.correlation.correlation_strength for a in correlated_alerts]
                )
            )

            for alert in correlated_alerts:
                analytics["correlation_stats"]["correlation_types"][
                    alert.correlation.correlation_type
                ] += 1

        return analytics

    async def start_processing_worker(self):
        """Start the background worker for processing alerts."""
        while True:
            try:
                alert = await self.processing_queue.get()
                await self.process_alert_intelligence(alert)
                self.processing_queue.task_done()
            except Exception as e:
                logger.error(f"Error in alert processing worker: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
