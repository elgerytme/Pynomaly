"""Predictive maintenance analytics for system health forecasting and capacity planning."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE_REQUIRED = "maintenance_required"


class ForecastHorizon(str, Enum):
    """Forecast time horizons."""

    SHORT_TERM = "short_term"  # 1-24 hours
    MEDIUM_TERM = "medium_term"  # 1-7 days
    LONG_TERM = "long_term"  # 1-30 days
    EXTENDED = "extended"  # 30+ days


class MaintenanceAction(str, Enum):
    """Types of maintenance actions."""

    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"
    SCHEDULED = "scheduled"
    OPTIMIZATION = "optimization"


class ComponentType(str, Enum):
    """Types of system components."""

    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    DETECTOR = "detector"
    MODEL = "model"
    API_GATEWAY = "api_gateway"
    LOAD_BALANCER = "load_balancer"


@dataclass
class HealthMetric:
    """Health metric measurement."""

    metric_name: str
    value: float
    timestamp: datetime
    component_type: ComponentType
    component_id: str
    unit: str = ""
    threshold_warning: float | None = None
    threshold_critical: float | None = None
    trend: str | None = None  # "increasing", "decreasing", "stable"


@dataclass
class ComponentHealth:
    """Health status of a system component."""

    component_id: str
    component_type: ComponentType
    current_status: HealthStatus
    health_score: float  # 0.0 to 1.0
    last_updated: datetime
    metrics: dict[str, HealthMetric] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    maintenance_history: list[dict[str, Any]] = field(default_factory=list)
    predicted_failures: list[dict[str, Any]] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)


@dataclass
class CapacityForecast:
    """Capacity planning forecast."""

    component_type: ComponentType
    current_utilization: float
    forecast_horizon: ForecastHorizon
    predicted_utilization: list[tuple[datetime, float]]
    capacity_exhaustion_date: datetime | None = None
    recommended_scaling: dict[str, Any] | None = None
    confidence_level: float = 0.0


@dataclass
class MaintenanceRecommendation:
    """Maintenance recommendation."""

    component_id: str
    action_type: MaintenanceAction
    priority: int  # 1 (highest) to 5 (lowest)
    estimated_downtime: timedelta
    cost_estimate: float | None = None
    failure_probability: float = 0.0
    impact_assessment: str = ""
    recommended_window: tuple[datetime, datetime] | None = None
    dependencies: list[str] = field(default_factory=list)


class HealthPredictor:
    """Predictor for component health degradation."""

    def __init__(self, lookback_hours: int = 168):  # 1 week
        self.lookback_hours = lookback_hours
        self.trend_models = {}
        self.seasonal_patterns = {}

    async def predict_health_degradation(
        self, component: ComponentHealth, forecast_hours: int = 24
    ) -> dict[str, Any]:
        """Predict health degradation for a component."""
        try:
            predictions = {}

            for metric_name, metric in component.metrics.items():
                # Simple trend-based prediction
                trend_prediction = await self._predict_metric_trend(
                    metric, forecast_hours
                )
                predictions[metric_name] = trend_prediction

            # Calculate overall health prediction
            health_prediction = await self._predict_overall_health(
                component, predictions, forecast_hours
            )

            return {
                "component_id": component.component_id,
                "forecast_hours": forecast_hours,
                "metric_predictions": predictions,
                "health_prediction": health_prediction,
                "failure_probability": self._calculate_failure_probability(predictions),
                "recommended_actions": self._generate_health_recommendations(
                    predictions
                ),
            }

        except Exception as e:
            logger.error(f"Error predicting health for {component.component_id}: {e}")
            return {"error": str(e)}

    async def _predict_metric_trend(
        self, metric: HealthMetric, forecast_hours: int
    ) -> dict[str, Any]:
        """Predict trend for a specific metric."""
        # Simplified linear trend prediction
        # In practice, you'd use more sophisticated time series models

        current_value = metric.value
        trend_direction = metric.trend or "stable"

        # Estimate trend rate (simplified)
        if trend_direction == "increasing":
            hourly_change_rate = current_value * 0.01  # 1% per hour
        elif trend_direction == "decreasing":
            hourly_change_rate = -current_value * 0.01
        else:
            hourly_change_rate = 0.0

        # Generate forecast points
        forecast_points = []
        for hour in range(1, forecast_hours + 1):
            predicted_value = current_value + (hourly_change_rate * hour)

            # Add some noise for realism
            noise = np.random.normal(0, current_value * 0.02)
            predicted_value += noise

            # Ensure non-negative values
            predicted_value = max(0, predicted_value)

            forecast_time = metric.timestamp + timedelta(hours=hour)
            forecast_points.append((forecast_time, predicted_value))

        # Check for threshold violations
        threshold_violations = []
        for forecast_time, value in forecast_points:
            if metric.threshold_warning and value > metric.threshold_warning:
                threshold_violations.append(
                    {
                        "type": "warning",
                        "timestamp": forecast_time,
                        "value": value,
                        "threshold": metric.threshold_warning,
                    }
                )
            if metric.threshold_critical and value > metric.threshold_critical:
                threshold_violations.append(
                    {
                        "type": "critical",
                        "timestamp": forecast_time,
                        "value": value,
                        "threshold": metric.threshold_critical,
                    }
                )

        return {
            "metric_name": metric.metric_name,
            "current_value": current_value,
            "trend": trend_direction,
            "forecast_points": forecast_points,
            "threshold_violations": threshold_violations,
            "confidence": 0.7,  # Simplified confidence
        }

    async def _predict_overall_health(
        self,
        component: ComponentHealth,
        metric_predictions: dict[str, Any],
        forecast_hours: int,
    ) -> dict[str, Any]:
        """Predict overall component health."""
        # Calculate health score trajectory
        health_trajectory = []

        for hour in range(1, forecast_hours + 1):
            # Calculate weighted health score based on metric predictions
            weighted_health = 0.0
            total_weight = 0.0

            for metric_name, prediction in metric_predictions.items():
                forecast_points = prediction.get("forecast_points", [])
                if hour <= len(forecast_points):
                    _, predicted_value = forecast_points[hour - 1]

                    # Get current metric
                    current_metric = component.metrics.get(metric_name)
                    if current_metric:
                        # Calculate health contribution (simplified)
                        if current_metric.threshold_critical:
                            health_contrib = 1.0 - min(
                                1.0, predicted_value / current_metric.threshold_critical
                            )
                        else:
                            health_contrib = 0.8  # Default if no threshold

                        weight = 1.0  # Equal weight for all metrics (can be customized)
                        weighted_health += health_contrib * weight
                        total_weight += weight

            if total_weight > 0:
                overall_health = weighted_health / total_weight
            else:
                overall_health = component.health_score

            # Determine status
            if overall_health > 0.8:
                status = HealthStatus.HEALTHY
            elif overall_health > 0.6:
                status = HealthStatus.WARNING
            elif overall_health > 0.4:
                status = HealthStatus.DEGRADED
            elif overall_health > 0.2:
                status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.FAILED

            forecast_time = component.last_updated + timedelta(hours=hour)
            health_trajectory.append(
                {
                    "timestamp": forecast_time,
                    "health_score": overall_health,
                    "status": status.value,
                }
            )

        return {
            "current_health_score": component.health_score,
            "current_status": component.current_status.value,
            "health_trajectory": health_trajectory,
            "predicted_degradation_rate": self._calculate_degradation_rate(
                health_trajectory
            ),
            "time_to_critical": self._estimate_time_to_critical(health_trajectory),
        }

    def _calculate_failure_probability(self, predictions: dict[str, Any]) -> float:
        """Calculate probability of component failure."""
        # Count threshold violations
        total_violations = 0
        critical_violations = 0

        for prediction in predictions.values():
            violations = prediction.get("threshold_violations", [])
            total_violations += len(violations)
            critical_violations += len(
                [v for v in violations if v["type"] == "critical"]
            )

        # Simple probability calculation
        if critical_violations > 0:
            failure_prob = min(0.9, 0.3 + (critical_violations * 0.2))
        elif total_violations > 0:
            failure_prob = min(0.5, 0.1 + (total_violations * 0.1))
        else:
            failure_prob = 0.05  # Base failure probability

        return failure_prob

    def _generate_health_recommendations(
        self, predictions: dict[str, Any]
    ) -> list[str]:
        """Generate health-based recommendations."""
        recommendations = []

        for prediction in predictions.values():
            violations = prediction.get("threshold_violations", [])

            if violations:
                metric_name = prediction.get("metric_name", "unknown")
                critical_violations = [v for v in violations if v["type"] == "critical"]
                warning_violations = [v for v in violations if v["type"] == "warning"]

                if critical_violations:
                    recommendations.append(
                        f"URGENT: {metric_name} predicted to exceed critical threshold. "
                        f"Schedule immediate maintenance."
                    )
                elif warning_violations:
                    recommendations.append(
                        f"ATTENTION: {metric_name} predicted to exceed warning threshold. "
                        f"Plan preventive maintenance."
                    )

        if not recommendations:
            recommendations.append(
                "Component health appears stable. Continue monitoring."
            )

        return recommendations

    def _calculate_degradation_rate(
        self, health_trajectory: list[dict[str, Any]]
    ) -> float:
        """Calculate health degradation rate per hour."""
        if len(health_trajectory) < 2:
            return 0.0

        # Calculate slope of health score over time
        health_scores = [point["health_score"] for point in health_trajectory]

        # Simple linear regression
        n = len(health_scores)
        x = list(range(n))
        y = health_scores

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope  # Health score change per hour

    def _estimate_time_to_critical(
        self, health_trajectory: list[dict[str, Any]]
    ) -> int | None:
        """Estimate hours until component reaches critical status."""
        for i, point in enumerate(health_trajectory):
            if point["status"] in ["critical", "failed"]:
                return i + 1  # Hours from now

        return None  # No critical status predicted in forecast horizon


class CapacityPlanner:
    """Capacity planning and resource forecasting."""

    def __init__(self):
        self.utilization_history = defaultdict(deque)
        self.growth_patterns = {}

    async def forecast_capacity_needs(
        self,
        component_type: ComponentType,
        current_utilization: float,
        historical_data: list[tuple[datetime, float]],
        forecast_horizon: ForecastHorizon = ForecastHorizon.MEDIUM_TERM,
    ) -> CapacityForecast:
        """Forecast capacity needs for a component type."""
        try:
            # Determine forecast period
            if forecast_horizon == ForecastHorizon.SHORT_TERM:
                forecast_hours = 24
            elif forecast_horizon == ForecastHorizon.MEDIUM_TERM:
                forecast_hours = 168  # 1 week
            elif forecast_horizon == ForecastHorizon.LONG_TERM:
                forecast_hours = 720  # 30 days
            else:  # EXTENDED
                forecast_hours = 2160  # 90 days

            # Analyze historical trends
            trend_analysis = await self._analyze_utilization_trends(historical_data)

            # Generate utilization forecast
            predicted_utilization = await self._predict_utilization(
                current_utilization, trend_analysis, forecast_hours
            )

            # Estimate capacity exhaustion
            exhaustion_date = self._estimate_capacity_exhaustion(predicted_utilization)

            # Generate scaling recommendations
            scaling_recommendations = await self._generate_scaling_recommendations(
                component_type, predicted_utilization, exhaustion_date
            )

            # Calculate confidence
            confidence = self._calculate_forecast_confidence(
                historical_data, trend_analysis
            )

            return CapacityForecast(
                component_type=component_type,
                current_utilization=current_utilization,
                forecast_horizon=forecast_horizon,
                predicted_utilization=predicted_utilization,
                capacity_exhaustion_date=exhaustion_date,
                recommended_scaling=scaling_recommendations,
                confidence_level=confidence,
            )

        except Exception as e:
            logger.error(f"Error forecasting capacity for {component_type}: {e}")
            return CapacityForecast(
                component_type=component_type,
                current_utilization=current_utilization,
                forecast_horizon=forecast_horizon,
                predicted_utilization=[],
                confidence_level=0.0,
            )

    async def _analyze_utilization_trends(
        self, historical_data: list[tuple[datetime, float]]
    ) -> dict[str, Any]:
        """Analyze utilization trends from historical data."""
        if not historical_data or len(historical_data) < 2:
            return {"trend": "stable", "growth_rate": 0.0, "seasonality": None}

        # Sort by timestamp
        sorted_data = sorted(historical_data, key=lambda x: x[0])
        timestamps = [item[0] for item in sorted_data]
        values = [item[1] for item in sorted_data]

        # Calculate growth rate
        time_span_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
        if time_span_hours > 0:
            total_growth = values[-1] - values[0]
            growth_rate_per_hour = total_growth / time_span_hours
        else:
            growth_rate_per_hour = 0.0

        # Determine trend direction
        if growth_rate_per_hour > 0.001:  # 0.1% per hour
            trend = "increasing"
        elif growth_rate_per_hour < -0.001:
            trend = "decreasing"
        else:
            trend = "stable"

        # Simple seasonality detection (daily patterns)
        hourly_averages = defaultdict(list)
        for timestamp, value in sorted_data:
            hour = timestamp.hour
            hourly_averages[hour].append(value)

        hourly_means = {
            hour: np.mean(values) for hour, values in hourly_averages.items()
        }

        # Check if there's significant hourly variation
        if hourly_means:
            hourly_std = np.std(list(hourly_means.values()))
            overall_mean = np.mean(list(hourly_means.values()))
            seasonality_strength = hourly_std / overall_mean if overall_mean > 0 else 0

            if seasonality_strength > 0.1:  # 10% variation
                seasonality = {
                    "type": "daily",
                    "strength": seasonality_strength,
                    "hourly_patterns": hourly_means,
                }
            else:
                seasonality = None
        else:
            seasonality = None

        return {
            "trend": trend,
            "growth_rate_per_hour": growth_rate_per_hour,
            "seasonality": seasonality,
            "volatility": np.std(values) if len(values) > 1 else 0.0,
        }

    async def _predict_utilization(
        self,
        current_utilization: float,
        trend_analysis: dict[str, Any],
        forecast_hours: int,
    ) -> list[tuple[datetime, float]]:
        """Predict utilization over forecast horizon."""
        predictions = []
        base_time = datetime.now()

        growth_rate = trend_analysis.get("growth_rate_per_hour", 0.0)
        seasonality = trend_analysis.get("seasonality")
        volatility = trend_analysis.get("volatility", 0.0)

        for hour in range(1, forecast_hours + 1):
            # Base prediction with trend
            predicted_value = current_utilization + (growth_rate * hour)

            # Apply seasonality if detected
            if seasonality and seasonality["type"] == "daily":
                forecast_time = base_time + timedelta(hours=hour)
                hour_of_day = forecast_time.hour

                hourly_patterns = seasonality.get("hourly_patterns", {})
                if hour_of_day in hourly_patterns:
                    # Apply seasonal adjustment
                    overall_mean = np.mean(list(hourly_patterns.values()))
                    seasonal_factor = hourly_patterns[hour_of_day] / overall_mean
                    predicted_value *= seasonal_factor

            # Add some random variation based on volatility
            if volatility > 0:
                noise = np.random.normal(0, volatility * 0.1)
                predicted_value += noise

            # Ensure utilization stays within reasonable bounds
            predicted_value = max(0.0, min(1.0, predicted_value))

            forecast_time = base_time + timedelta(hours=hour)
            predictions.append((forecast_time, predicted_value))

        return predictions

    def _estimate_capacity_exhaustion(
        self, predicted_utilization: list[tuple[datetime, float]]
    ) -> datetime | None:
        """Estimate when capacity will be exhausted (95% utilization)."""
        exhaustion_threshold = 0.95

        for timestamp, utilization in predicted_utilization:
            if utilization >= exhaustion_threshold:
                return timestamp

        return None  # No exhaustion predicted in forecast horizon

    async def _generate_scaling_recommendations(
        self,
        component_type: ComponentType,
        predicted_utilization: list[tuple[datetime, float]],
        exhaustion_date: datetime | None,
    ) -> dict[str, Any]:
        """Generate scaling recommendations."""
        recommendations = {
            "scaling_needed": False,
            "recommended_action": "monitor",
            "scaling_factor": 1.0,
            "target_date": None,
            "justification": "",
        }

        if not predicted_utilization:
            return recommendations

        # Find peak utilization in forecast
        peak_utilization = max(util for _, util in predicted_utilization)

        # Check if scaling is needed
        if peak_utilization > 0.8:  # 80% threshold for scaling
            recommendations["scaling_needed"] = True

            if peak_utilization > 0.95:
                recommendations["recommended_action"] = "immediate_scaling"
                recommendations["scaling_factor"] = 1.5  # 50% increase
                recommendations["target_date"] = datetime.now() + timedelta(hours=24)
                recommendations["justification"] = (
                    f"Critical: Peak utilization of {peak_utilization:.1%} predicted"
                )

            elif peak_utilization > 0.85:
                recommendations["recommended_action"] = "planned_scaling"
                recommendations["scaling_factor"] = 1.3  # 30% increase
                recommendations["target_date"] = datetime.now() + timedelta(days=7)
                recommendations["justification"] = (
                    f"High utilization of {peak_utilization:.1%} predicted"
                )

            else:
                recommendations["recommended_action"] = "prepare_scaling"
                recommendations["scaling_factor"] = 1.2  # 20% increase
                recommendations["target_date"] = datetime.now() + timedelta(days=14)
                recommendations["justification"] = (
                    f"Moderate utilization of {peak_utilization:.1%} predicted"
                )

        # Add exhaustion date if applicable
        if exhaustion_date:
            recommendations["capacity_exhaustion_date"] = exhaustion_date
            recommendations["days_until_exhaustion"] = (
                exhaustion_date - datetime.now()
            ).days

        return recommendations

    def _calculate_forecast_confidence(
        self,
        historical_data: list[tuple[datetime, float]],
        trend_analysis: dict[str, Any],
    ) -> float:
        """Calculate confidence in the forecast."""
        if not historical_data:
            return 0.0

        # Base confidence on data availability
        data_points = len(historical_data)
        data_confidence = min(1.0, data_points / 100)  # 100 points = full confidence

        # Adjust based on trend stability
        volatility = trend_analysis.get("volatility", 0.0)
        stability_confidence = max(
            0.0, 1.0 - (volatility * 2)
        )  # High volatility reduces confidence

        # Adjust based on time span
        if data_points >= 2:
            time_span_hours = (
                historical_data[-1][0] - historical_data[0][0]
            ).total_seconds() / 3600
            time_confidence = min(
                1.0, time_span_hours / 168
            )  # 1 week = full confidence
        else:
            time_confidence = 0.0

        # Combined confidence
        overall_confidence = (
            0.4 * data_confidence + 0.4 * stability_confidence + 0.2 * time_confidence
        )

        return overall_confidence


class PredictiveMaintenanceAnalytics:
    """Main service for predictive maintenance analytics."""

    def __init__(
        self,
        health_check_interval_minutes: int = 30,
        forecast_update_interval_hours: int = 6,
        max_components: int = 1000,
    ):
        """Initialize predictive maintenance analytics service.

        Args:
            health_check_interval_minutes: Interval for health checks
            forecast_update_interval_hours: Interval for forecast updates
            max_components: Maximum number of components to track
        """
        self.health_check_interval = timedelta(minutes=health_check_interval_minutes)
        self.forecast_update_interval = timedelta(hours=forecast_update_interval_hours)
        self.max_components = max_components

        # Component tracking
        self.components: dict[str, ComponentHealth] = {}
        self.capacity_forecasts: dict[ComponentType, CapacityForecast] = {}
        self.maintenance_recommendations: list[MaintenanceRecommendation] = []

        # Services
        self.health_predictor = HealthPredictor()
        self.capacity_planner = CapacityPlanner()

        # Analytics
        self.system_health_history: deque = deque(maxlen=1000)
        self.maintenance_history: deque = deque(maxlen=500)
        self.alert_history: deque = deque(maxlen=1000)

        # Background tasks
        self.background_tasks = set()
        self._running = False

        logger.info("Initialized predictive maintenance analytics service")

    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._running:
            return

        self._running = True

        # Start health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.background_tasks.add(health_task)
        health_task.add_done_callback(self.background_tasks.discard)

        # Start capacity forecasting task
        capacity_task = asyncio.create_task(self._capacity_forecasting_loop())
        self.background_tasks.add(capacity_task)
        capacity_task.add_done_callback(self.background_tasks.discard)

        logger.info("Started predictive maintenance monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        self._running = False

        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        logger.info("Stopped predictive maintenance monitoring")

    async def register_component(
        self,
        component_id: str,
        component_type: ComponentType,
        initial_metrics: dict[str, HealthMetric] | None = None,
    ) -> bool:
        """Register a component for monitoring.

        Args:
            component_id: Unique component identifier
            component_type: Type of component
            initial_metrics: Initial health metrics

        Returns:
            Success status
        """
        try:
            if len(self.components) >= self.max_components:
                logger.warning(
                    f"Maximum components limit reached: {self.max_components}"
                )
                return False

            component = ComponentHealth(
                component_id=component_id,
                component_type=component_type,
                current_status=HealthStatus.HEALTHY,
                health_score=1.0,
                last_updated=datetime.now(),
                metrics=initial_metrics or {},
            )

            self.components[component_id] = component

            logger.info(
                f"Registered component: {component_id} ({component_type.value})"
            )
            return True

        except Exception as e:
            logger.error(f"Error registering component {component_id}: {e}")
            return False

    async def update_component_metrics(
        self, component_id: str, metrics: dict[str, HealthMetric]
    ) -> bool:
        """Update metrics for a component.

        Args:
            component_id: Component identifier
            metrics: New health metrics

        Returns:
            Success status
        """
        try:
            if component_id not in self.components:
                logger.warning(f"Component not found: {component_id}")
                return False

            component = self.components[component_id]

            # Update metrics
            component.metrics.update(metrics)
            component.last_updated = datetime.now()

            # Recalculate health score
            component.health_score = await self._calculate_health_score(component)

            # Update status based on health score
            component.current_status = self._determine_health_status(
                component.health_score
            )

            # Check for alerts
            await self._check_component_alerts(component)

            return True

        except Exception as e:
            logger.error(f"Error updating metrics for {component_id}: {e}")
            return False

    async def _calculate_health_score(self, component: ComponentHealth) -> float:
        """Calculate overall health score for a component."""
        if not component.metrics:
            return 1.0  # Default healthy score

        metric_scores = []

        for metric in component.metrics.values():
            # Calculate metric health score
            if metric.threshold_critical and metric.value > metric.threshold_critical:
                score = 0.0  # Critical threshold exceeded
            elif metric.threshold_warning and metric.value > metric.threshold_warning:
                # Linearly interpolate between warning and critical
                if metric.threshold_critical:
                    ratio = (metric.value - metric.threshold_warning) / (
                        metric.threshold_critical - metric.threshold_warning
                    )
                    score = 0.5 * (1.0 - ratio)  # 0.0 to 0.5
                else:
                    score = 0.5  # Warning threshold exceeded, no critical threshold
            else:
                # Below warning threshold
                if metric.threshold_warning:
                    ratio = metric.value / metric.threshold_warning
                    score = 1.0 - (0.5 * ratio)  # 0.5 to 1.0
                else:
                    score = 0.8  # Default good score if no thresholds

            metric_scores.append(max(0.0, min(1.0, score)))

        # Return average of all metric scores
        return np.mean(metric_scores) if metric_scores else 1.0

    def _determine_health_status(self, health_score: float) -> HealthStatus:
        """Determine health status from health score."""
        if health_score >= 0.8:
            return HealthStatus.HEALTHY
        elif health_score >= 0.6:
            return HealthStatus.WARNING
        elif health_score >= 0.4:
            return HealthStatus.DEGRADED
        elif health_score >= 0.2:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILED

    async def _check_component_alerts(self, component: ComponentHealth) -> None:
        """Check for alerts based on component metrics."""
        new_alerts = []

        for metric_name, metric in component.metrics.items():
            if metric.threshold_critical and metric.value > metric.threshold_critical:
                alert = f"CRITICAL: {metric_name} exceeds critical threshold ({metric.value:.2f} > {metric.threshold_critical:.2f})"
                new_alerts.append(alert)
            elif metric.threshold_warning and metric.value > metric.threshold_warning:
                alert = f"WARNING: {metric_name} exceeds warning threshold ({metric.value:.2f} > {metric.threshold_warning:.2f})"
                new_alerts.append(alert)

        if new_alerts:
            # Add new alerts
            component.alerts.extend(new_alerts)

            # Keep only recent alerts
            component.alerts = component.alerts[-10:]

            # Log alerts
            for alert in new_alerts:
                self.alert_history.append(
                    {
                        "timestamp": datetime.now(),
                        "component_id": component.component_id,
                        "alert": alert,
                    }
                )
                logger.warning(f"Component {component.component_id}: {alert}")

    async def _health_monitoring_loop(self) -> None:
        """Background loop for health monitoring."""
        while self._running:
            try:
                await self._perform_health_analysis()
                await asyncio.sleep(self.health_check_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _perform_health_analysis(self) -> None:
        """Perform health analysis for all components."""
        for component in self.components.values():
            try:
                # Generate health predictions
                health_prediction = (
                    await self.health_predictor.predict_health_degradation(
                        component, forecast_hours=24
                    )
                )

                # Update component with predictions
                component.predicted_failures = [health_prediction]

                # Generate maintenance recommendations
                if health_prediction.get("failure_probability", 0) > 0.3:
                    await self._generate_maintenance_recommendation(
                        component, health_prediction
                    )

            except Exception as e:
                logger.error(
                    f"Error analyzing health for {component.component_id}: {e}"
                )

        # Update system health history
        system_health = await self._calculate_system_health()
        self.system_health_history.append(
            {
                "timestamp": datetime.now(),
                "overall_health": system_health,
                "component_count": len(self.components),
                "healthy_components": len(
                    [
                        c
                        for c in self.components.values()
                        if c.current_status == HealthStatus.HEALTHY
                    ]
                ),
                "critical_components": len(
                    [
                        c
                        for c in self.components.values()
                        if c.current_status == HealthStatus.CRITICAL
                    ]
                ),
            }
        )

    async def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        if not self.components:
            return 1.0

        component_scores = [
            component.health_score for component in self.components.values()
        ]
        return np.mean(component_scores)

    async def _generate_maintenance_recommendation(
        self, component: ComponentHealth, health_prediction: dict[str, Any]
    ) -> None:
        """Generate maintenance recommendation for a component."""
        failure_probability = health_prediction.get("failure_probability", 0.0)

        # Determine action type and priority
        if failure_probability > 0.7:
            action_type = MaintenanceAction.EMERGENCY
            priority = 1
            estimated_downtime = timedelta(hours=4)
        elif failure_probability > 0.5:
            action_type = MaintenanceAction.CORRECTIVE
            priority = 2
            estimated_downtime = timedelta(hours=2)
        else:
            action_type = MaintenanceAction.PREVENTIVE
            priority = 3
            estimated_downtime = timedelta(hours=1)

        # Create recommendation
        recommendation = MaintenanceRecommendation(
            component_id=component.component_id,
            action_type=action_type,
            priority=priority,
            estimated_downtime=estimated_downtime,
            failure_probability=failure_probability,
            impact_assessment=f"Component health score: {component.health_score:.2f}",
            recommended_window=(
                datetime.now() + timedelta(hours=1),
                datetime.now() + timedelta(hours=48),
            ),
        )

        # Add to recommendations if not already present
        existing_recommendations = [
            r
            for r in self.maintenance_recommendations
            if r.component_id == component.component_id
        ]

        if not existing_recommendations:
            self.maintenance_recommendations.append(recommendation)

            # Keep only recent recommendations
            self.maintenance_recommendations = self.maintenance_recommendations[-100:]

    async def _capacity_forecasting_loop(self) -> None:
        """Background loop for capacity forecasting."""
        while self._running:
            try:
                await self._perform_capacity_analysis()
                await asyncio.sleep(self.forecast_update_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in capacity forecasting loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying

    async def _perform_capacity_analysis(self) -> None:
        """Perform capacity analysis for all component types."""
        # Group components by type
        component_groups = defaultdict(list)
        for component in self.components.values():
            component_groups[component.component_type].append(component)

        # Forecast capacity for each component type
        for component_type, components in component_groups.items():
            try:
                # Calculate average utilization (simplified)
                current_utilizations = []
                for component in components:
                    # Use health score as proxy for utilization
                    utilization = 1.0 - component.health_score
                    current_utilizations.append(utilization)

                if current_utilizations:
                    avg_utilization = np.mean(current_utilizations)

                    # Generate historical data (simplified)
                    historical_data = [
                        (
                            datetime.now() - timedelta(hours=i),
                            avg_utilization + np.random.normal(0, 0.1),
                        )
                        for i in range(168, 0, -1)  # 1 week of hourly data
                    ]

                    # Generate capacity forecast
                    forecast = await self.capacity_planner.forecast_capacity_needs(
                        component_type=component_type,
                        current_utilization=avg_utilization,
                        historical_data=historical_data,
                        forecast_horizon=ForecastHorizon.MEDIUM_TERM,
                    )

                    self.capacity_forecasts[component_type] = forecast

            except Exception as e:
                logger.error(f"Error forecasting capacity for {component_type}: {e}")

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        # Overall system health
        system_health = await self._calculate_system_health()

        # Component status summary
        status_counts = defaultdict(int)
        for component in self.components.values():
            status_counts[component.current_status.value] += 1

        # Recent alerts
        recent_alerts = list(self.alert_history)[-10:] if self.alert_history else []

        # High priority maintenance recommendations
        high_priority_maintenance = [
            r for r in self.maintenance_recommendations if r.priority <= 2
        ]

        # Capacity warnings
        capacity_warnings = []
        for component_type, forecast in self.capacity_forecasts.items():
            if forecast.capacity_exhaustion_date:
                days_until_exhaustion = (
                    forecast.capacity_exhaustion_date - datetime.now()
                ).days
                if days_until_exhaustion <= 7:
                    capacity_warnings.append(
                        {
                            "component_type": component_type.value,
                            "days_until_exhaustion": days_until_exhaustion,
                            "current_utilization": forecast.current_utilization,
                        }
                    )

        return {
            "system_health": {
                "overall_score": system_health,
                "status": self._determine_health_status(system_health).value,
                "component_count": len(self.components),
                "status_distribution": dict(status_counts),
            },
            "alerts": {
                "recent_count": len(recent_alerts),
                "recent_alerts": recent_alerts,
                "critical_alerts": len(
                    [a for a in recent_alerts if "CRITICAL" in a.get("alert", "")]
                ),
            },
            "maintenance": {
                "pending_recommendations": len(self.maintenance_recommendations),
                "high_priority_count": len(high_priority_maintenance),
                "emergency_actions_needed": len(
                    [
                        r
                        for r in high_priority_maintenance
                        if r.action_type == MaintenanceAction.EMERGENCY
                    ]
                ),
            },
            "capacity": {
                "forecasts_available": len(self.capacity_forecasts),
                "capacity_warnings": capacity_warnings,
                "scaling_needed": len(
                    [
                        f
                        for f in self.capacity_forecasts.values()
                        if f.recommended_scaling
                        and f.recommended_scaling.get("scaling_needed", False)
                    ]
                ),
            },
            "monitoring_status": {
                "monitoring_active": self._running,
                "last_health_check": (
                    max([c.last_updated for c in self.components.values()])
                    if self.components
                    else None
                ),
                "components_monitored": len(self.components),
            },
        }
