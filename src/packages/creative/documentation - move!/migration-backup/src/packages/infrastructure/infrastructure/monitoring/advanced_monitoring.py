#!/usr/bin/env python3
"""
Advanced Monitoring and Observability Framework

This module provides enhanced monitoring capabilities including:
- Predictive alerting with ML-based anomaly detection
- Automated capacity planning and scaling recommendations
- Comprehensive SLI/SLO monitoring framework
- Intelligent alert correlation and noise reduction
- Business intelligence integration
"""

import asyncio
import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Counter, Gauge
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels for intelligent correlation."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SLIType(Enum):
    """Service Level Indicator types."""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CAPACITY = "capacity"


@dataclass
class SLI:
    """Service Level Indicator definition."""

    name: str
    type: SLIType
    query: str
    target_value: float
    time_window: str
    description: str
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class SLO:
    """Service Level Objective definition."""

    name: str
    sli: SLI
    target_percentage: float  # e.g., 99.9
    time_period: str  # e.g., "30d"
    description: str
    burn_rate_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class AlertEvent:
    """Alert event representation for correlation."""

    id: str
    timestamp: datetime
    severity: AlertSeverity
    source: str
    title: str
    description: str
    labels: dict[str, str]
    metrics: dict[str, float]
    resolved: bool = False
    correlation_id: str | None = None


@dataclass
class CapacityPrediction:
    """Capacity planning prediction result."""

    metric_name: str
    current_value: float
    predicted_value: float
    prediction_date: datetime
    confidence: float
    recommended_action: str
    threshold_breach_date: datetime | None = None


class MetricPredictor(ABC):
    """Abstract base class for metric prediction algorithms."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Train the prediction model on historical data."""
        pass

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """Predict future values for given number of time steps."""
        pass

    @abstractmethod
    def get_confidence(self) -> float:
        """Get prediction confidence score."""
        pass


class LinearTrendPredictor(MetricPredictor):
    """Simple linear trend-based predictor for capacity planning."""

    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
        self.r_squared = 0.0
        self.fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        """Fit linear trend to historical data."""
        if len(data) < 2:
            logger.warning("Insufficient data for linear trend fitting")
            return

        # Convert timestamp to numerical values
        x = np.arange(len(data))
        y = data["value"].values

        # Calculate linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        # Calculate slope and intercept
        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        self.intercept = (sum_y - self.slope * sum_x) / n

        # Calculate R-squared
        y_pred = self.slope * x + self.intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        self.fitted = True
        logger.info(
            f"Linear trend fitted: slope={self.slope:.4f}, R²={self.r_squared:.4f}"
        )

    def predict(self, steps: int) -> np.ndarray:
        """Predict future values using linear trend."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        future_x = np.arange(steps)
        return self.slope * future_x + self.intercept

    def get_confidence(self) -> float:
        """Return R-squared as confidence measure."""
        return max(0.0, min(1.0, self.r_squared))


class SeasonalTrendPredictor(MetricPredictor):
    """Seasonal trend predictor using moving averages and seasonal decomposition."""

    def __init__(self, seasonal_period: int = 24):
        self.seasonal_period = seasonal_period
        self.trend = None
        self.seasonal = None
        self.residual_std = 0.0
        self.fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        """Fit seasonal trend model to historical data."""
        if len(data) < self.seasonal_period * 2:
            logger.warning(
                f"Insufficient data for seasonal analysis (need {self.seasonal_period * 2}, got {len(data)})"
            )
            return

        values = data["value"].values

        # Calculate trend using moving average
        trend = np.convolve(
            values, np.ones(self.seasonal_period) / self.seasonal_period, mode="same"
        )

        # Calculate seasonal component
        detrended = values - trend
        seasonal_matrix = detrended.reshape(-1, self.seasonal_period)
        seasonal = np.mean(seasonal_matrix, axis=0)

        # Extend seasonal pattern to match data length
        seasonal_extended = np.tile(seasonal, len(values) // self.seasonal_period + 1)[
            : len(values)
        ]

        # Calculate residuals
        residuals = values - trend - seasonal_extended
        self.residual_std = np.std(residuals)

        self.trend = trend
        self.seasonal = seasonal
        self.fitted = True

        logger.info(
            f"Seasonal model fitted: period={self.seasonal_period}, residual_std={self.residual_std:.4f}"
        )

    def predict(self, steps: int) -> np.ndarray:
        """Predict future values using seasonal trend."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        # Extend trend linearly
        last_trend = self.trend[-1]
        trend_slope = (self.trend[-1] - self.trend[-min(10, len(self.trend))]) / min(
            10, len(self.trend)
        )
        future_trend = last_trend + trend_slope * np.arange(1, steps + 1)

        # Extend seasonal pattern
        future_seasonal = np.tile(self.seasonal, steps // self.seasonal_period + 1)[
            :steps
        ]

        return future_trend + future_seasonal

    def get_confidence(self) -> float:
        """Return confidence based on residual analysis."""
        if not self.fitted or self.residual_std == 0:
            return 0.0

        # Confidence inversely related to residual standard deviation
        return max(
            0.0,
            min(1.0, 1.0 - min(self.residual_std / np.mean(np.abs(self.trend)), 1.0)),
        )


class AnomalyDetector:
    """ML-based anomaly detection for monitoring data."""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination, random_state=42
        )
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, data: pd.DataFrame) -> None:
        """Train anomaly detection model on historical data."""
        if len(data) < 10:
            logger.warning("Insufficient data for anomaly detection training")
            return

        # Prepare features (could be extended with more sophisticated feature engineering)
        features = data[["value"]].values

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train isolation forest
        self.isolation_forest.fit(features_scaled)
        self.fitted = True

        logger.info(f"Anomaly detector trained on {len(data)} samples")

    def detect_anomalies(self, data: pd.DataFrame) -> list[bool]:
        """Detect anomalies in new data."""
        if not self.fitted:
            raise ValueError("Model must be fitted before anomaly detection")

        features = data[["value"]].values
        features_scaled = self.scaler.transform(features)

        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.isolation_forest.predict(features_scaled)
        return [pred == -1 for pred in predictions]

    def get_anomaly_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores for data points."""
        if not self.fitted:
            raise ValueError("Model must be fitted before scoring")

        features = data[["value"]].values
        features_scaled = self.scaler.transform(features)

        # Get anomaly scores (lower scores indicate more anomalous)
        return self.isolation_forest.decision_function(features_scaled)


class AlertCorrelationEngine:
    """Intelligent alert correlation and noise reduction."""

    def __init__(self, correlation_window: timedelta = timedelta(minutes=5)):
        self.correlation_window = correlation_window
        self.alert_history: list[AlertEvent] = []
        self.correlation_rules: list[dict[str, Any]] = []
        self.suppression_rules: list[dict[str, Any]] = []

    def add_correlation_rule(self, rule: dict[str, Any]) -> None:
        """Add a correlation rule for grouping related alerts."""
        self.correlation_rules.append(rule)

    def add_suppression_rule(self, rule: dict[str, Any]) -> None:
        """Add a suppression rule for reducing alert noise."""
        self.suppression_rules.append(rule)

    def process_alert(self, alert: AlertEvent) -> tuple[bool, str | None]:
        """Process an alert and determine if it should be sent or suppressed."""
        # Check suppression rules
        for rule in self.suppression_rules:
            if self._matches_suppression_rule(alert, rule):
                logger.info(f"Alert {alert.id} suppressed by rule: {rule['name']}")
                return False, f"suppressed_by_{rule['name']}"

        # Check for correlation with recent alerts
        correlation_id = self._find_correlation(alert)
        alert.correlation_id = correlation_id

        # Add to history
        self.alert_history.append(alert)

        # Clean old alerts from history
        self._cleanup_history()

        return True, correlation_id

    def _matches_suppression_rule(
        self, alert: AlertEvent, rule: dict[str, Any]
    ) -> bool:
        """Check if alert matches a suppression rule."""
        conditions = rule.get("conditions", {})

        # Check severity
        if (
            "severity" in conditions
            and alert.severity.value not in conditions["severity"]
        ):
            return False

        # Check source
        if "source" in conditions and alert.source not in conditions["source"]:
            return False

        # Check labels
        if "labels" in conditions:
            for key, values in conditions["labels"].items():
                if key not in alert.labels or alert.labels[key] not in values:
                    return False

        # Check time-based suppression
        if "time_window" in rule:
            window = timedelta(seconds=rule["time_window"])
            cutoff_time = alert.timestamp - window

            similar_alerts = [
                a
                for a in self.alert_history
                if a.timestamp > cutoff_time
                and a.source == alert.source
                and a.title == alert.title
                and not a.resolved
            ]

            if len(similar_alerts) >= rule.get("max_alerts_in_window", 1):
                return True

        return False

    def _find_correlation(self, alert: AlertEvent) -> str | None:
        """Find correlation with recent alerts."""
        cutoff_time = alert.timestamp - self.correlation_window
        recent_alerts = [a for a in self.alert_history if a.timestamp > cutoff_time]

        # Apply correlation rules
        for rule in self.correlation_rules:
            correlated_alerts = []
            for recent_alert in recent_alerts:
                if self._alerts_match_rule(alert, recent_alert, rule):
                    correlated_alerts.append(recent_alert)

            if correlated_alerts:
                # Use existing correlation ID or create new one
                existing_correlation = next(
                    (a.correlation_id for a in correlated_alerts if a.correlation_id),
                    None,
                )
                return existing_correlation or f"corr_{alert.timestamp.isoformat()}"

        return None

    def _alerts_match_rule(
        self, alert1: AlertEvent, alert2: AlertEvent, rule: dict[str, Any]
    ) -> bool:
        """Check if two alerts match a correlation rule."""
        conditions = rule.get("conditions", {})

        # Check if alerts are from related sources
        if "source_pattern" in conditions:
            pattern = conditions["source_pattern"]
            if not (pattern in alert1.source and pattern in alert2.source):
                return False

        # Check if alerts have similar labels
        if "label_overlap" in conditions:
            required_overlap = conditions["label_overlap"]
            common_labels = set(alert1.labels.keys()) & set(alert2.labels.keys())
            if len(common_labels) < required_overlap:
                return False

        # Check severity correlation
        if "severity_correlation" in conditions:
            severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
            sev1 = severity_map.get(alert1.severity.value, 0)
            sev2 = severity_map.get(alert2.severity.value, 0)
            if abs(sev1 - sev2) > conditions["severity_correlation"]:
                return False

        return True

    def _cleanup_history(self) -> None:
        """Remove old alerts from history."""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours of history
        self.alert_history = [
            a for a in self.alert_history if a.timestamp > cutoff_time
        ]


class SLOMonitor:
    """Service Level Objective monitoring and alerting."""

    def __init__(self):
        self.slos: dict[str, SLO] = {}
        self.sli_metrics: dict[str, list[tuple[datetime, float]]] = {}

    def add_slo(self, slo: SLO) -> None:
        """Add an SLO to monitor."""
        self.slos[slo.name] = slo
        if slo.sli.name not in self.sli_metrics:
            self.sli_metrics[slo.sli.name] = []
        logger.info(f"Added SLO: {slo.name} (target: {slo.target_percentage}%)")

    def record_sli_measurement(
        self, sli_name: str, value: float, timestamp: datetime | None = None
    ) -> None:
        """Record an SLI measurement."""
        if timestamp is None:
            timestamp = datetime.now()

        if sli_name not in self.sli_metrics:
            self.sli_metrics[sli_name] = []

        self.sli_metrics[sli_name].append((timestamp, value))

        # Keep only recent measurements (configurable window)
        cutoff_time = timestamp - timedelta(days=30)
        self.sli_metrics[sli_name] = [
            (ts, val) for ts, val in self.sli_metrics[sli_name] if ts > cutoff_time
        ]

    def calculate_slo_compliance(
        self, slo_name: str, time_period: timedelta | None = None
    ) -> dict[str, Any]:
        """Calculate SLO compliance for a given time period."""
        if slo_name not in self.slos:
            raise ValueError(f"SLO {slo_name} not found")

        slo = self.slos[slo_name]
        sli_name = slo.sli.name

        if sli_name not in self.sli_metrics:
            return {"compliance": 0.0, "error": "No SLI measurements found"}

        # Determine time window
        if time_period is None:
            # Parse time period from SLO definition (e.g., "30d", "7d", "24h")
            period_str = slo.time_period
            if period_str.endswith("d"):
                time_period = timedelta(days=int(period_str[:-1]))
            elif period_str.endswith("h"):
                time_period = timedelta(hours=int(period_str[:-1]))
            elif period_str.endswith("m"):
                time_period = timedelta(minutes=int(period_str[:-1]))
            else:
                time_period = timedelta(days=30)  # Default to 30 days

        cutoff_time = datetime.now() - time_period
        relevant_measurements = [
            (ts, val) for ts, val in self.sli_metrics[sli_name] if ts > cutoff_time
        ]

        if not relevant_measurements:
            return {"compliance": 0.0, "error": "No measurements in time period"}

        # Calculate compliance based on SLI type
        values = [val for _, val in relevant_measurements]

        if slo.sli.type in [SLIType.AVAILABILITY, SLIType.ERROR_RATE]:
            # For availability and error rate, calculate percentage of good measurements
            good_measurements = sum(1 for val in values if val >= slo.sli.target_value)
            compliance = (good_measurements / len(values)) * 100
        elif slo.sli.type == SLIType.LATENCY:
            # For latency, calculate percentage of measurements below threshold
            good_measurements = sum(1 for val in values if val <= slo.sli.target_value)
            compliance = (good_measurements / len(values)) * 100
        else:
            # For other metrics, use target as threshold
            good_measurements = sum(1 for val in values if val >= slo.sli.target_value)
            compliance = (good_measurements / len(values)) * 100

        # Calculate error budget
        error_budget_consumed = max(
            0, 100 - slo.target_percentage - (compliance - slo.target_percentage)
        )
        error_budget_remaining = max(
            0, 100 - slo.target_percentage - error_budget_consumed
        )

        # Calculate burn rate
        recent_window = timedelta(hours=1)
        recent_cutoff = datetime.now() - recent_window
        recent_measurements = [
            (ts, val) for ts, val in relevant_measurements if ts > recent_cutoff
        ]

        if recent_measurements:
            recent_values = [val for _, val in recent_measurements]
            if slo.sli.type == SLIType.LATENCY:
                recent_good = sum(
                    1 for val in recent_values if val <= slo.sli.target_value
                )
            else:
                recent_good = sum(
                    1 for val in recent_values if val >= slo.sli.target_value
                )
            recent_compliance = (recent_good / len(recent_values)) * 100
            burn_rate = max(0, slo.target_percentage - recent_compliance)
        else:
            burn_rate = 0.0

        return {
            "compliance": compliance,
            "target": slo.target_percentage,
            "measurement_count": len(relevant_measurements),
            "error_budget_consumed": error_budget_consumed,
            "error_budget_remaining": error_budget_remaining,
            "burn_rate": burn_rate,
            "time_period": str(time_period),
            "last_measurement": max(ts for ts, _ in relevant_measurements)
            if relevant_measurements
            else None,
        }

    def check_burn_rate_alerts(self, slo_name: str) -> list[dict[str, Any]]:
        """Check for burn rate threshold violations."""
        if slo_name not in self.slos:
            return []

        slo = self.slos[slo_name]
        compliance = self.calculate_slo_compliance(slo_name)

        alerts = []
        current_burn_rate = compliance.get("burn_rate", 0.0)

        for threshold_name, threshold_value in slo.burn_rate_thresholds.items():
            if current_burn_rate > threshold_value:
                alerts.append(
                    {
                        "slo_name": slo_name,
                        "threshold_name": threshold_name,
                        "threshold_value": threshold_value,
                        "current_burn_rate": current_burn_rate,
                        "severity": self._determine_burn_rate_severity(threshold_name),
                        "message": f"SLO {slo_name} burn rate {current_burn_rate:.2f}% exceeds {threshold_name} threshold {threshold_value:.2f}%",
                    }
                )

        return alerts

    def _determine_burn_rate_severity(self, threshold_name: str) -> AlertSeverity:
        """Determine alert severity based on burn rate threshold name."""
        severity_map = {
            "critical": AlertSeverity.CRITICAL,
            "high": AlertSeverity.HIGH,
            "warning": AlertSeverity.MEDIUM,
            "low": AlertSeverity.LOW,
        }

        for key, severity in severity_map.items():
            if key in threshold_name.lower():
                return severity

        return AlertSeverity.MEDIUM


class CapacityPlanner:
    """Automated capacity planning and scaling recommendations."""

    def __init__(self):
        self.predictors: dict[str, MetricPredictor] = {}
        self.thresholds: dict[str, dict[str, float]] = {}
        self.historical_data: dict[str, pd.DataFrame] = {}

    def add_metric(
        self, metric_name: str, predictor: MetricPredictor, thresholds: dict[str, float]
    ) -> None:
        """Add a metric for capacity planning."""
        self.predictors[metric_name] = predictor
        self.thresholds[metric_name] = thresholds
        self.historical_data[metric_name] = pd.DataFrame(columns=["timestamp", "value"])
        logger.info(f"Added metric for capacity planning: {metric_name}")

    def add_measurement(
        self, metric_name: str, value: float, timestamp: datetime | None = None
    ) -> None:
        """Add a measurement for capacity planning."""
        if metric_name not in self.historical_data:
            return

        if timestamp is None:
            timestamp = datetime.now()

        new_row = pd.DataFrame([{"timestamp": timestamp, "value": value}])
        self.historical_data[metric_name] = pd.concat(
            [self.historical_data[metric_name], new_row], ignore_index=True
        )

        # Keep only recent data (configurable, default 30 days)
        cutoff_time = timestamp - timedelta(days=30)
        self.historical_data[metric_name] = self.historical_data[metric_name][
            pd.to_datetime(self.historical_data[metric_name]["timestamp"]) > cutoff_time
        ]

    def generate_capacity_predictions(
        self, forecast_days: int = 30
    ) -> list[CapacityPrediction]:
        """Generate capacity predictions for all monitored metrics."""
        predictions = []

        for metric_name in self.predictors:
            prediction = self._predict_metric_capacity(metric_name, forecast_days)
            if prediction:
                predictions.append(prediction)

        return predictions

    def _predict_metric_capacity(
        self, metric_name: str, forecast_days: int
    ) -> CapacityPrediction | None:
        """Generate capacity prediction for a specific metric."""
        if (
            metric_name not in self.historical_data
            or len(self.historical_data[metric_name]) < 10
        ):
            logger.warning(f"Insufficient data for capacity prediction: {metric_name}")
            return None

        try:
            # Prepare data
            data = self.historical_data[metric_name].copy()
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data.sort_values("timestamp")

            # Train predictor
            predictor = self.predictors[metric_name]
            predictor.fit(data)

            # Make prediction
            current_value = data["value"].iloc[-1]
            predicted_values = predictor.predict(
                forecast_days * 24
            )  # Assuming hourly data
            predicted_value = (
                predicted_values[-1] if len(predicted_values) > 0 else current_value
            )

            # Calculate prediction date
            prediction_date = datetime.now() + timedelta(days=forecast_days)

            # Get confidence
            confidence = predictor.get_confidence()

            # Determine recommended action
            thresholds = self.thresholds.get(metric_name, {})
            recommended_action = self._determine_capacity_action(
                metric_name, current_value, predicted_value, thresholds
            )

            # Calculate threshold breach date
            threshold_breach_date = self._calculate_breach_date(
                metric_name, data, predicted_values, thresholds
            )

            return CapacityPrediction(
                metric_name=metric_name,
                current_value=current_value,
                predicted_value=predicted_value,
                prediction_date=prediction_date,
                confidence=confidence,
                recommended_action=recommended_action,
                threshold_breach_date=threshold_breach_date,
            )

        except Exception as e:
            logger.error(f"Error generating capacity prediction for {metric_name}: {e}")
            return None

    def _determine_capacity_action(
        self,
        metric_name: str,
        current: float,
        predicted: float,
        thresholds: dict[str, float],
    ) -> str:
        """Determine recommended capacity action based on prediction."""
        if not thresholds:
            return "monitor"

        critical_threshold = thresholds.get("critical", float("inf"))
        warning_threshold = thresholds.get("warning", critical_threshold * 0.8)

        growth_rate = (predicted - current) / current if current > 0 else 0

        if predicted >= critical_threshold:
            return "immediate_scale_up_required"
        elif predicted >= warning_threshold:
            return "plan_scale_up"
        elif growth_rate > 0.5:  # 50% growth
            return "monitor_closely"
        elif growth_rate < -0.2:  # 20% decrease
            return "consider_scale_down"
        else:
            return "maintain_current_capacity"

    def _calculate_breach_date(
        self,
        metric_name: str,
        data: pd.DataFrame,
        predictions: np.ndarray,
        thresholds: dict[str, float],
    ) -> datetime | None:
        """Calculate when a threshold will be breached based on predictions."""
        warning_threshold = thresholds.get("warning")
        if warning_threshold is None:
            return None

        current_value = data["value"].iloc[-1]

        # Find when prediction will exceed threshold
        for i, predicted_value in enumerate(predictions):
            if predicted_value >= warning_threshold:
                # Calculate breach date (assuming hourly predictions)
                breach_date = datetime.now() + timedelta(hours=i)
                return breach_date

        return None


class AdvancedMonitoringOrchestrator:
    """Main orchestrator for advanced monitoring capabilities."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.anomaly_detector = AnomalyDetector()
        self.alert_correlator = AlertCorrelationEngine()
        self.slo_monitor = SLOMonitor()
        self.capacity_planner = CapacityPlanner()

        # Metrics
        self.registry = CollectorRegistry()
        self.anomaly_counter = Counter(
            "monitoring_anomalies_detected_total",
            "Total number of anomalies detected",
            ["metric_name", "severity"],
            registry=self.registry,
        )
        self.slo_compliance_gauge = Gauge(
            "slo_compliance_percentage",
            "SLO compliance percentage",
            ["slo_name"],
            registry=self.registry,
        )
        self.capacity_prediction_gauge = Gauge(
            "capacity_prediction_value",
            "Predicted capacity value",
            ["metric_name"],
            registry=self.registry,
        )

        self._setup_default_configuration()

    def _setup_default_configuration(self) -> None:
        """Setup default monitoring configuration."""
        # Default correlation rules
        self.alert_correlator.add_correlation_rule(
            {
                "name": "service_related",
                "conditions": {
                    "source_pattern": "service",
                    "label_overlap": 1,
                    "severity_correlation": 2,
                },
            }
        )

        # Default suppression rules
        self.alert_correlator.add_suppression_rule(
            {
                "name": "duplicate_suppression",
                "conditions": {
                    "time_window": 300,  # 5 minutes
                    "max_alerts_in_window": 1,
                },
            }
        )

        # Default SLOs
        self.slo_monitor.add_slo(
            SLO(
                name="api_availability",
                sli=SLI(
                    name="api_success_rate",
                    type=SLIType.AVAILABILITY,
                    query='rate(http_requests_total{status!~"5.."}[5m]) / rate(http_requests_total[5m])',
                    target_value=0.99,
                    time_window="5m",
                    description="API success rate (non-5xx responses)",
                ),
                target_percentage=99.9,
                time_period="30d",
                description="API availability SLO",
                burn_rate_thresholds={"critical": 10.0, "high": 5.0, "warning": 2.0},
            )
        )

        self.slo_monitor.add_slo(
            SLO(
                name="api_latency",
                sli=SLI(
                    name="api_p95_latency",
                    type=SLIType.LATENCY,
                    query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                    target_value=0.5,  # 500ms
                    time_window="5m",
                    description="API P95 latency",
                ),
                target_percentage=95.0,
                time_period="30d",
                description="API latency SLO",
                burn_rate_thresholds={"critical": 10.0, "high": 5.0, "warning": 2.0},
            )
        )

        # Default capacity planning metrics
        self.capacity_planner.add_metric(
            "cpu_usage", LinearTrendPredictor(), {"warning": 70.0, "critical": 85.0}
        )

        self.capacity_planner.add_metric(
            "memory_usage", LinearTrendPredictor(), {"warning": 75.0, "critical": 90.0}
        )

        self.capacity_planner.add_metric(
            "disk_usage", LinearTrendPredictor(), {"warning": 80.0, "critical": 95.0}
        )

    async def process_metric_data(
        self, metric_name: str, value: float, timestamp: datetime | None = None
    ) -> None:
        """Process incoming metric data through all monitoring components."""
        if timestamp is None:
            timestamp = datetime.now()

        # Add to capacity planner
        self.capacity_planner.add_measurement(metric_name, value, timestamp)

        # Prepare data for anomaly detection
        data = pd.DataFrame([{"timestamp": timestamp, "value": value}])

        try:
            # Check for anomalies if model is trained
            if self.anomaly_detector.fitted:
                anomalies = self.anomaly_detector.detect_anomalies(data)
                if anomalies[0]:  # If current value is anomalous
                    self.anomaly_counter.labels(
                        metric_name=metric_name, severity="medium"
                    ).inc()

                    # Create anomaly alert
                    alert = AlertEvent(
                        id=f"anomaly_{metric_name}_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        severity=AlertSeverity.MEDIUM,
                        source="anomaly_detector",
                        title=f"Anomaly detected in {metric_name}",
                        description=f"Anomalous value {value} detected for metric {metric_name}",
                        labels={"metric_name": metric_name, "type": "anomaly"},
                        metrics={"value": value},
                    )

                    # Process through correlation engine
                    should_send, correlation_id = self.alert_correlator.process_alert(
                        alert
                    )
                    if should_send:
                        logger.warning(
                            f"Anomaly alert: {alert.title} (correlation: {correlation_id})"
                        )

        except Exception as e:
            logger.error(f"Error processing metric data for anomaly detection: {e}")

    async def update_sli_measurement(
        self, sli_name: str, value: float, timestamp: datetime | None = None
    ) -> None:
        """Update SLI measurement and check SLO compliance."""
        self.slo_monitor.record_sli_measurement(sli_name, value, timestamp)

        # Check all SLOs that use this SLI
        for slo_name, slo in self.slo_monitor.slos.items():
            if slo.sli.name == sli_name:
                compliance = self.slo_monitor.calculate_slo_compliance(slo_name)
                self.slo_compliance_gauge.labels(slo_name=slo_name).set(
                    compliance.get("compliance", 0)
                )

                # Check for burn rate alerts
                burn_rate_alerts = self.slo_monitor.check_burn_rate_alerts(slo_name)
                for alert_data in burn_rate_alerts:
                    alert = AlertEvent(
                        id=f"slo_burn_{slo_name}_{timestamp or datetime.now()}",
                        timestamp=timestamp or datetime.now(),
                        severity=alert_data["severity"],
                        source="slo_monitor",
                        title=f"SLO burn rate alert: {slo_name}",
                        description=alert_data["message"],
                        labels={"slo_name": slo_name, "type": "slo_burn_rate"},
                        metrics={"burn_rate": alert_data["current_burn_rate"]},
                    )

                    should_send, correlation_id = self.alert_correlator.process_alert(
                        alert
                    )
                    if should_send:
                        logger.warning(
                            f"SLO burn rate alert: {alert.title} (correlation: {correlation_id})"
                        )

    async def generate_capacity_report(self, forecast_days: int = 30) -> dict[str, Any]:
        """Generate comprehensive capacity planning report."""
        predictions = self.capacity_planner.generate_capacity_predictions(forecast_days)

        # Update metrics
        for prediction in predictions:
            self.capacity_prediction_gauge.labels(
                metric_name=prediction.metric_name
            ).set(prediction.predicted_value)

        # Analyze recommendations
        immediate_actions = [
            p for p in predictions if "immediate" in p.recommended_action
        ]
        planned_actions = [p for p in predictions if "plan" in p.recommended_action]
        monitoring_required = [
            p for p in predictions if "monitor" in p.recommended_action
        ]

        report = {
            "timestamp": datetime.now().isoformat(),
            "forecast_days": forecast_days,
            "total_metrics": len(predictions),
            "summary": {
                "immediate_actions_required": len(immediate_actions),
                "planned_actions_required": len(planned_actions),
                "monitoring_required": len(monitoring_required),
            },
            "predictions": [
                {
                    "metric_name": p.metric_name,
                    "current_value": p.current_value,
                    "predicted_value": p.predicted_value,
                    "prediction_date": p.prediction_date.isoformat(),
                    "confidence": p.confidence,
                    "recommended_action": p.recommended_action,
                    "threshold_breach_date": p.threshold_breach_date.isoformat()
                    if p.threshold_breach_date
                    else None,
                }
                for p in predictions
            ],
            "immediate_actions": [
                {
                    "metric_name": p.metric_name,
                    "action": p.recommended_action,
                    "urgency": "high"
                    if p.threshold_breach_date
                    and p.threshold_breach_date < datetime.now() + timedelta(days=7)
                    else "medium",
                }
                for p in immediate_actions
            ],
        }

        logger.info(
            f"Generated capacity report: {len(predictions)} metrics analyzed, {len(immediate_actions)} immediate actions required"
        )
        return report

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get overall monitoring system status."""
        # Calculate SLO compliance summary
        slo_compliance_summary = {}
        for slo_name in self.slo_monitor.slos:
            compliance = self.slo_monitor.calculate_slo_compliance(slo_name)
            slo_compliance_summary[slo_name] = {
                "compliance": compliance.get("compliance", 0),
                "target": compliance.get("target", 0),
                "status": "good"
                if compliance.get("compliance", 0) >= compliance.get("target", 100)
                else "bad",
            }

        # Calculate active alerts
        active_alerts = len(
            [a for a in self.alert_correlator.alert_history if not a.resolved]
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "anomaly_detection": {
                    "status": "active" if self.anomaly_detector.fitted else "training",
                    "model_trained": self.anomaly_detector.fitted,
                },
                "alert_correlation": {
                    "status": "active",
                    "active_alerts": active_alerts,
                    "correlation_rules": len(self.alert_correlator.correlation_rules),
                    "suppression_rules": len(self.alert_correlator.suppression_rules),
                },
                "slo_monitoring": {
                    "status": "active",
                    "slos_defined": len(self.slo_monitor.slos),
                    "compliance_summary": slo_compliance_summary,
                },
                "capacity_planning": {
                    "status": "active",
                    "metrics_monitored": len(self.capacity_planner.predictors),
                    "models_trained": len(
                        [
                            p
                            for p in self.capacity_planner.predictors.values()
                            if hasattr(p, "fitted") and p.fitted
                        ]
                    ),
                },
            },
            "overall_status": "healthy"
            if active_alerts < 10
            and all(
                info["compliance"] >= info["target"]
                for info in slo_compliance_summary.values()
            )
            else "degraded",
        }


async def main():
    """Demo function for advanced monitoring."""
    logging.basicConfig(level=logging.INFO)

    # Initialize advanced monitoring
    monitor = AdvancedMonitoringOrchestrator()

    # Simulate some metric data
    import random

    for i in range(100):
        # Simulate CPU usage with some anomalies
        base_cpu = 45 + 10 * math.sin(i / 10)  # Seasonal pattern
        noise = random.gauss(0, 5)
        anomaly = 50 if i % 20 == 0 else 0  # Inject anomalies
        cpu_value = max(0, min(100, base_cpu + noise + anomaly))

        await monitor.process_metric_data("cpu_usage", cpu_value)

        # Simulate API latency
        base_latency = 0.2 + 0.1 * random.random()
        if i % 15 == 0:  # Occasional spikes
            base_latency += 0.5

        await monitor.update_sli_measurement("api_p95_latency", base_latency)

        # Simulate API success rate
        success_rate = 0.995 if random.random() > 0.1 else 0.985
        await monitor.update_sli_measurement("api_success_rate", success_rate)

        await asyncio.sleep(0.1)  # Small delay to simulate real-time data

    # Generate capacity report
    capacity_report = await monitor.generate_capacity_report(30)
    print(f"Capacity Report: {json.dumps(capacity_report, indent=2)}")

    # Get monitoring status
    status = monitor.get_monitoring_status()
    print(f"Monitoring Status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
