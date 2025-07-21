"""
Predictive Quality Service

Provides application-level services for predictive data quality monitoring,
including trend analysis, forecasting, and proactive alerting.
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

from ...domain.entities.quality_prediction import (
    QualityPrediction,
    QualityTrend,
    QualityForecast,
    QualityAlert,
    QualityMetricPoint,
    PredictionType,
    PredictionConfidence,
    TrendDirection,
    SeasonalPattern
)


class PredictiveQualityService:
    """Service for predictive data quality monitoring."""
    
    def __init__(self):
        self._predictions: Dict[UUID, QualityPrediction] = {}
        self._trends: Dict[UUID, List[QualityTrend]] = {}
        self._forecasts: Dict[UUID, List[QualityForecast]] = {}
        self._alerts: Dict[UUID, List[QualityAlert]] = {}
        
        # Historical metrics for analysis
        self._metric_history: Dict[UUID, List[QualityMetricPoint]] = {}
        
        # Model configurations
        self._prediction_models = {
            "linear_regression": self._linear_regression_predict,
            "exponential_smoothing": self._exponential_smoothing_predict,
            "seasonal_decomposition": self._seasonal_decomposition_predict,
            "anomaly_detection": self._anomaly_detection_predict
        }
        
        # Default thresholds
        self._default_thresholds = {
            "quality_degradation": 0.1,  # 10% drop in quality
            "freshness_decay": timedelta(hours=24),
            "volume_anomaly": 0.2,  # 20% volume change
            "completeness_drop": 0.05,  # 5% completeness drop
        }
    
    def add_metric_point(self, asset_id: UUID, metric_type: str, value: float, timestamp: datetime = None) -> None:
        """Add a quality metric point for analysis."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        point = QualityMetricPoint(
            timestamp=timestamp,
            value=value,
            metric_type=metric_type
        )
        
        if asset_id not in self._metric_history:
            self._metric_history[asset_id] = []
        
        self._metric_history[asset_id].append(point)
        
        # Keep only last 10000 points per asset
        if len(self._metric_history[asset_id]) > 10000:
            self._metric_history[asset_id] = self._metric_history[asset_id][-10000:]
        
        # Trigger analysis for new predictions
        self._analyze_metrics_for_predictions(asset_id, metric_type)
    
    def analyze_trends(self, asset_id: UUID, metric_type: str, days: int = 30) -> QualityTrend:
        """Analyze trends in quality metrics."""
        if asset_id not in self._metric_history:
            raise ValueError(f"No metric history found for asset {asset_id}")
        
        # Filter metrics by type and time range
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        relevant_points = [
            point for point in self._metric_history[asset_id]
            if point.metric_type == metric_type and point.timestamp >= cutoff_time
        ]
        
        if len(relevant_points) < 5:
            raise ValueError(f"Insufficient data points for trend analysis (need at least 5, got {len(relevant_points)})")
        
        # Sort by timestamp
        relevant_points.sort(key=lambda p: p.timestamp)
        
        # Calculate trend statistics
        values = [point.value for point in relevant_points]
        timestamps = [point.timestamp.timestamp() for point in relevant_points]
        
        # Linear regression for trend
        slope, intercept, r_squared = self._calculate_linear_regression(timestamps, values)
        
        # Determine trend direction
        if abs(slope) < 0.001:  # Very small slope
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.DEGRADING
        
        # Check for volatility
        if self._calculate_volatility(values) > 0.2:  # High volatility threshold
            direction = TrendDirection.VOLATILE
        
        # Detect seasonal patterns
        seasonal_pattern, seasonal_strength = self._detect_seasonality(relevant_points)
        
        # Detect anomalies
        anomaly_timestamps = self._detect_anomalies(relevant_points)
        
        trend = QualityTrend(
            asset_id=asset_id,
            metric_type=metric_type,
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            start_time=relevant_points[0].timestamp,
            end_time=relevant_points[-1].timestamp,
            data_points=len(relevant_points),
            mean_value=statistics.mean(values),
            std_deviation=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            seasonal_pattern=seasonal_pattern,
            seasonal_strength=seasonal_strength,
            anomaly_count=len(anomaly_timestamps),
            anomaly_timestamps=anomaly_timestamps
        )
        
        # Store trend
        if asset_id not in self._trends:
            self._trends[asset_id] = []
        self._trends[asset_id].append(trend)
        
        # Keep only last 100 trends per asset
        if len(self._trends[asset_id]) > 100:
            self._trends[asset_id] = self._trends[asset_id][-100:]
        
        return trend
    
    def create_prediction(
        self,
        asset_id: UUID,
        prediction_type: PredictionType,
        target_time: datetime,
        model_name: str = "linear_regression",
        features: List[str] = None
    ) -> QualityPrediction:
        """Create a prediction for future quality metrics."""
        
        # Get prediction model
        if model_name not in self._prediction_models:
            raise ValueError(f"Unknown prediction model: {model_name}")
        
        prediction_func = self._prediction_models[model_name]
        
        # Generate prediction
        predicted_value, confidence_score, confidence_interval = prediction_func(
            asset_id, prediction_type, target_time, features or []
        )
        
        # Determine confidence level
        confidence = self._map_confidence_score(confidence_score)
        
        # Calculate impact score
        impact_score = self._calculate_impact_score(asset_id, prediction_type, predicted_value)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(prediction_type, predicted_value, impact_score)
        
        # Calculate time horizon
        time_horizon = target_time - datetime.utcnow()
        
        prediction = QualityPrediction(
            asset_id=asset_id,
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            target_time=target_time,
            time_horizon=time_horizon,
            confidence=confidence,
            confidence_score=confidence_score,
            prediction_interval=confidence_interval,
            model_name=model_name,
            features_used=features or [],
            impact_score=impact_score,
            recommended_actions=recommendations
        )
        
        # Store prediction
        self._predictions[prediction.id] = prediction
        
        # Create alert if prediction indicates issues
        if self._should_create_alert(prediction):
            self._create_prediction_alert(prediction)
        
        return prediction
    
    def create_forecast(
        self,
        asset_id: UUID,
        metric_type: str,
        horizon_hours: int = 24,
        resolution_hours: int = 1,
        model_name: str = "exponential_smoothing"
    ) -> QualityForecast:
        """Create a forecast for quality metrics over a time period."""
        
        forecast_horizon = timedelta(hours=horizon_hours)
        resolution = timedelta(hours=resolution_hours)
        
        # Generate forecast points
        current_time = datetime.utcnow()
        forecast_times = []
        time = current_time
        
        while time <= current_time + forecast_horizon:
            forecast_times.append(time)
            time += resolution
        
        # Generate forecasted values
        forecasted_values = []
        confidence_intervals = {"lower": [], "upper": []}
        
        for forecast_time in forecast_times:
            # Use prediction model to forecast each point
            predicted_value, confidence_score, interval = self._prediction_models[model_name](
                asset_id, PredictionType.QUALITY_DEGRADATION, forecast_time, []
            )
            
            point = QualityMetricPoint(
                timestamp=forecast_time,
                value=predicted_value,
                metric_type=metric_type
            )
            forecasted_values.append(point)
            
            if interval:
                lower_point = QualityMetricPoint(
                    timestamp=forecast_time,
                    value=interval["lower"],
                    metric_type=metric_type
                )
                upper_point = QualityMetricPoint(
                    timestamp=forecast_time,
                    value=interval["upper"],
                    metric_type=metric_type
                )
                confidence_intervals["lower"].append(lower_point)
                confidence_intervals["upper"].append(upper_point)
        
        # Analyze overall trend
        values = [point.value for point in forecasted_values]
        overall_trend = self._determine_forecast_trend(values)
        trend_strength = self._calculate_trend_strength(values)
        
        # Detect seasonal patterns
        seasonal_pattern = self._detect_forecast_seasonality(forecasted_values)
        
        # Predict anomalies
        anomaly_times, anomaly_probs = self._predict_anomalies(asset_id, forecast_times)
        
        forecast = QualityForecast(
            asset_id=asset_id,
            metric_type=metric_type,
            forecast_horizon=forecast_horizon,
            resolution=resolution,
            forecasted_values=forecasted_values,
            confidence_intervals=confidence_intervals,
            model_name=model_name,
            overall_trend=overall_trend,
            trend_strength=trend_strength,
            seasonal_pattern=seasonal_pattern,
            predicted_anomalies=anomaly_times,
            anomaly_probabilities=anomaly_probs
        )
        
        # Store forecast
        if asset_id not in self._forecasts:
            self._forecasts[asset_id] = []
        self._forecasts[asset_id].append(forecast)
        
        # Keep only last 50 forecasts per asset
        if len(self._forecasts[asset_id]) > 50:
            self._forecasts[asset_id] = self._forecasts[asset_id][-50:]
        
        return forecast
    
    def validate_prediction(self, prediction_id: UUID, actual_value: float) -> None:
        """Validate a prediction against actual observed value."""
        prediction = self._predictions.get(prediction_id)
        if not prediction:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        prediction.validate_prediction(actual_value)
    
    def get_prediction_accuracy_report(self, asset_id: UUID = None, days: int = 30) -> Dict[str, Any]:
        """Generate a report on prediction accuracy."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Filter predictions
        relevant_predictions = []
        for prediction in self._predictions.values():
            if prediction.is_validated and prediction.validation_time >= cutoff_time:
                if asset_id is None or prediction.asset_id == asset_id:
                    relevant_predictions.append(prediction)
        
        if not relevant_predictions:
            return {"message": "No validated predictions found for the specified period"}
        
        # Calculate accuracy metrics
        accuracies = [p.get_accuracy() for p in relevant_predictions if p.get_accuracy() is not None]
        errors = [p.prediction_error for p in relevant_predictions if p.prediction_error is not None]
        
        # Group by prediction type
        by_type = {}
        for prediction in relevant_predictions:
            pred_type = prediction.prediction_type.value
            if pred_type not in by_type:
                by_type[pred_type] = []
            by_type[pred_type].append(prediction)
        
        type_accuracies = {}
        for pred_type, predictions in by_type.items():
            type_accuracies[pred_type] = {
                "count": len(predictions),
                "avg_accuracy": statistics.mean([p.get_accuracy() for p in predictions if p.get_accuracy() is not None]),
                "avg_error": statistics.mean([p.prediction_error for p in predictions if p.prediction_error is not None])
            }
        
        return {
            "total_predictions": len(relevant_predictions),
            "avg_accuracy": statistics.mean(accuracies) if accuracies else 0,
            "avg_error": statistics.mean(errors) if errors else 0,
            "accuracy_by_type": type_accuracies,
            "high_accuracy_predictions": len([a for a in accuracies if a >= 0.8]),
            "low_accuracy_predictions": len([a for a in accuracies if a < 0.5])
        }
    
    def get_active_alerts(self, asset_id: UUID = None) -> List[QualityAlert]:
        """Get active quality prediction alerts."""
        active_alerts = []
        
        for asset_alerts in self._alerts.values():
            for alert in asset_alerts:
                if alert.status == "open" or alert.status == "acknowledged":
                    if asset_id is None or alert.asset_id == asset_id:
                        active_alerts.append(alert)
        
        return sorted(active_alerts, key=lambda a: a.created_at, reverse=True)
    
    def _analyze_metrics_for_predictions(self, asset_id: UUID, metric_type: str) -> None:
        """Analyze metrics to create automatic predictions."""
        try:
            # Analyze recent trends
            trend = self.analyze_trends(asset_id, metric_type, days=7)
            
            # Create predictions based on trends
            if trend.direction == TrendDirection.DEGRADING and trend.is_significant_trend():
                # Predict quality degradation
                target_time = datetime.utcnow() + timedelta(hours=24)
                self.create_prediction(
                    asset_id=asset_id,
                    prediction_type=PredictionType.QUALITY_DEGRADATION,
                    target_time=target_time
                )
            
            # Check for anomalies
            if trend.anomaly_count > 3:  # Multiple recent anomalies
                target_time = datetime.utcnow() + timedelta(hours=12)
                self.create_prediction(
                    asset_id=asset_id,
                    prediction_type=PredictionType.VOLUME_ANOMALY,
                    target_time=target_time
                )
                
        except ValueError:
            # Not enough data for analysis
            pass
    
    def _linear_regression_predict(
        self, asset_id: UUID, prediction_type: PredictionType, target_time: datetime, features: List[str]
    ) -> Tuple[float, float, Optional[Dict[str, float]]]:
        """Linear regression prediction model."""
        
        # Get historical data
        if asset_id not in self._metric_history:
            return 0.5, 0.3, None  # Default prediction with low confidence
        
        # Use last 30 days of data
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        relevant_points = [
            point for point in self._metric_history[asset_id]
            if point.timestamp >= cutoff_time
        ]
        
        if len(relevant_points) < 5:
            return 0.5, 0.3, None
        
        # Sort by timestamp
        relevant_points.sort(key=lambda p: p.timestamp)
        
        # Prepare data for regression
        values = [point.value for point in relevant_points]
        timestamps = [point.timestamp.timestamp() for point in relevant_points]
        target_timestamp = target_time.timestamp()
        
        # Calculate linear regression
        slope, intercept, r_squared = self._calculate_linear_regression(timestamps, values)
        
        # Predict value
        predicted_value = slope * target_timestamp + intercept
        
        # Confidence based on R-squared and data quantity
        confidence_score = min(0.9, r_squared * (len(relevant_points) / 100))
        
        # Calculate prediction interval
        residuals = []
        for i, timestamp in enumerate(timestamps):
            predicted = slope * timestamp + intercept
            residuals.append(abs(values[i] - predicted))
        
        if residuals:
            error_margin = statistics.mean(residuals) * 2  # 2-sigma interval
            confidence_interval = {
                "lower": predicted_value - error_margin,
                "upper": predicted_value + error_margin
            }
        else:
            confidence_interval = None
        
        return predicted_value, confidence_score, confidence_interval
    
    def _exponential_smoothing_predict(
        self, asset_id: UUID, prediction_type: PredictionType, target_time: datetime, features: List[str]
    ) -> Tuple[float, float, Optional[Dict[str, float]]]:
        """Exponential smoothing prediction model."""
        
        # Get historical data
        if asset_id not in self._metric_history:
            return 0.5, 0.3, None
        
        relevant_points = self._metric_history[asset_id][-100:]  # Last 100 points
        
        if len(relevant_points) < 3:
            return 0.5, 0.3, None
        
        # Sort by timestamp
        relevant_points.sort(key=lambda p: p.timestamp)
        values = [point.value for point in relevant_points]
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing factor
        smoothed_values = [values[0]]
        
        for i in range(1, len(values)):
            smoothed = alpha * values[i] + (1 - alpha) * smoothed_values[-1]
            smoothed_values.append(smoothed)
        
        # Predict next value
        predicted_value = smoothed_values[-1]
        
        # Calculate prediction error
        errors = []
        for i in range(1, len(smoothed_values)):
            if i < len(values):
                errors.append(abs(values[i] - smoothed_values[i-1]))
        
        # Confidence based on prediction accuracy
        if errors:
            avg_error = statistics.mean(errors)
            confidence_score = max(0.1, 1.0 - (avg_error / max(values)))
            
            confidence_interval = {
                "lower": predicted_value - avg_error,
                "upper": predicted_value + avg_error
            }
        else:
            confidence_score = 0.5
            confidence_interval = None
        
        return predicted_value, confidence_score, confidence_interval
    
    def _seasonal_decomposition_predict(
        self, asset_id: UUID, prediction_type: PredictionType, target_time: datetime, features: List[str]
    ) -> Tuple[float, float, Optional[Dict[str, float]]]:
        """Seasonal decomposition prediction model."""
        
        # Get historical data
        if asset_id not in self._metric_history:
            return 0.5, 0.3, None
        
        # Use last 90 days for seasonal analysis
        cutoff_time = datetime.utcnow() - timedelta(days=90)
        relevant_points = [
            point for point in self._metric_history[asset_id]
            if point.timestamp >= cutoff_time
        ]
        
        if len(relevant_points) < 20:  # Need more data for seasonal analysis
            return self._linear_regression_predict(asset_id, prediction_type, target_time, features)
        
        # Sort by timestamp
        relevant_points.sort(key=lambda p: p.timestamp)
        values = [point.value for point in relevant_points]
        
        # Simple seasonal decomposition (daily pattern)
        daily_averages = {}
        for point in relevant_points:
            hour = point.timestamp.hour
            if hour not in daily_averages:
                daily_averages[hour] = []
            daily_averages[hour].append(point.value)
        
        # Calculate seasonal component for target time
        target_hour = target_time.hour
        if target_hour in daily_averages:
            seasonal_component = statistics.mean(daily_averages[target_hour])
        else:
            seasonal_component = statistics.mean(values)
        
        # Calculate trend component
        recent_avg = statistics.mean(values[-7:]) if len(values) >= 7 else statistics.mean(values)
        overall_avg = statistics.mean(values)
        trend_component = recent_avg - overall_avg
        
        # Combine components
        predicted_value = seasonal_component + trend_component
        
        # Confidence based on seasonal pattern strength
        seasonal_variance = statistics.variance([statistics.mean(hourly_vals) for hourly_vals in daily_averages.values()])
        overall_variance = statistics.variance(values)
        
        if overall_variance > 0:
            seasonal_strength = min(1.0, seasonal_variance / overall_variance)
            confidence_score = 0.5 + (seasonal_strength * 0.4)  # Base 0.5, up to 0.9
        else:
            confidence_score = 0.5
        
        # Simple confidence interval
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        confidence_interval = {
            "lower": predicted_value - std_dev,
            "upper": predicted_value + std_dev
        } if std_dev > 0 else None
        
        return predicted_value, confidence_score, confidence_interval
    
    def _anomaly_detection_predict(
        self, asset_id: UUID, prediction_type: PredictionType, target_time: datetime, features: List[str]
    ) -> Tuple[float, float, Optional[Dict[str, float]]]:
        """Anomaly detection based prediction model."""
        
        # Get historical data
        if asset_id not in self._metric_history:
            return 0.1, 0.6, None  # Low anomaly probability
        
        relevant_points = self._metric_history[asset_id][-100:]  # Last 100 points
        
        if len(relevant_points) < 10:
            return 0.1, 0.6, None
        
        values = [point.value for point in relevant_points]
        
        # Calculate statistical properties
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        # Count recent anomalies (values outside 2 standard deviations)
        recent_anomalies = 0
        for value in values[-10:]:  # Last 10 points
            if abs(value - mean_val) > 2 * std_val:
                recent_anomalies += 1
        
        # Predict anomaly probability
        anomaly_probability = min(0.9, recent_anomalies / 10.0)
        
        # Confidence is high for anomaly detection
        confidence_score = 0.8
        
        return anomaly_probability, confidence_score, None
    
    def _calculate_linear_regression(self, x_values: List[float], y_values: List[float]) -> Tuple[float, float, float]:
        """Calculate linear regression slope, intercept, and R-squared."""
        n = len(x_values)
        if n < 2:
            return 0.0, 0.0, 0.0
        
        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        # Calculate slope and intercept
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0, y_mean, 0.0
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        ss_res = sum((y_values[i] - (slope * x_values[i] + intercept)) ** 2 for i in range(n))
        ss_tot = sum((y_values[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return slope, intercept, max(0.0, r_squared)
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility of values."""
        if len(values) < 2:
            return 0.0
        
        # Calculate percentage changes
        changes = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                change = abs(values[i] - values[i-1]) / abs(values[i-1])
                changes.append(change)
        
        return statistics.stdev(changes) if len(changes) > 1 else 0.0
    
    def _detect_seasonality(self, points: List[QualityMetricPoint]) -> Tuple[SeasonalPattern, float]:
        """Detect seasonal patterns in data."""
        if len(points) < 24:  # Need at least 24 points for daily pattern
            return SeasonalPattern.NONE, 0.0
        
        # Group by hour of day
        hourly_groups = {}
        for point in points:
            hour = point.timestamp.hour
            if hour not in hourly_groups:
                hourly_groups[hour] = []
            hourly_groups[hour].append(point.value)
        
        # Calculate hourly averages
        hourly_averages = {hour: statistics.mean(values) for hour, values in hourly_groups.items()}
        
        if len(hourly_averages) < 12:  # Need reasonable coverage
            return SeasonalPattern.NONE, 0.0
        
        # Calculate variance within hours vs between hours
        within_hour_variance = 0.0
        between_hour_variance = 0.0
        
        all_values = [point.value for point in points]
        overall_mean = statistics.mean(all_values)
        
        # Between-hour variance
        for hour_avg in hourly_averages.values():
            between_hour_variance += (hour_avg - overall_mean) ** 2
        between_hour_variance /= len(hourly_averages)
        
        # Within-hour variance
        count = 0
        for hour, values in hourly_groups.items():
            if len(values) > 1:
                hour_mean = hourly_averages[hour]
                for value in values:
                    within_hour_variance += (value - hour_mean) ** 2
                    count += 1
        
        if count > 0:
            within_hour_variance /= count
        
        # Seasonal strength
        total_variance = between_hour_variance + within_hour_variance
        seasonal_strength = between_hour_variance / total_variance if total_variance > 0 else 0.0
        
        # Determine pattern type based on strength
        if seasonal_strength > 0.3:
            return SeasonalPattern.DAILY, seasonal_strength
        elif seasonal_strength > 0.1:
            # Could be weekly pattern - need more sophisticated analysis
            return SeasonalPattern.WEEKLY, seasonal_strength
        else:
            return SeasonalPattern.NONE, seasonal_strength
    
    def _detect_anomalies(self, points: List[QualityMetricPoint]) -> List[datetime]:
        """Detect anomalies in time series data."""
        if len(points) < 10:
            return []
        
        values = [point.value for point in points]
        timestamps = [point.timestamp for point in points]
        
        # Calculate rolling statistics
        window_size = min(10, len(values) // 3)
        anomalies = []
        
        for i in range(window_size, len(values)):
            # Get window of previous values
            window = values[i-window_size:i]
            window_mean = statistics.mean(window)
            window_std = statistics.stdev(window) if len(window) > 1 else 0
            
            # Check if current value is anomalous
            if window_std > 0:
                z_score = abs(values[i] - window_mean) / window_std
                if z_score > 2.5:  # 2.5 standard deviations
                    anomalies.append(timestamps[i])
        
        return anomalies
    
    def _map_confidence_score(self, score: float) -> PredictionConfidence:
        """Map numerical confidence score to confidence level."""
        if score >= 0.8:
            return PredictionConfidence.VERY_HIGH
        elif score >= 0.6:
            return PredictionConfidence.HIGH
        elif score >= 0.4:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    def _calculate_impact_score(self, asset_id: UUID, prediction_type: PredictionType, predicted_value: float) -> float:
        """Calculate impact score for a prediction."""
        # Base impact on prediction type
        base_impact = {
            PredictionType.QUALITY_DEGRADATION: 0.8,
            PredictionType.ACCURACY_DECLINE: 0.7,
            PredictionType.CONSISTENCY_BREACH: 0.6,
            PredictionType.COMPLETENESS_DROP: 0.5,
            PredictionType.FRESHNESS_DECAY: 0.4,
            PredictionType.VOLUME_ANOMALY: 0.3,
            PredictionType.SCHEMA_DRIFT: 0.9
        }.get(prediction_type, 0.5)
        
        # Adjust based on predicted value severity
        if prediction_type in [PredictionType.QUALITY_DEGRADATION, PredictionType.ACCURACY_DECLINE]:
            # Lower values are worse for quality metrics
            severity_factor = max(0.1, 1.0 - predicted_value)
        else:
            # Higher values might be worse for other metrics
            severity_factor = min(1.0, max(0.1, predicted_value))
        
        return min(1.0, base_impact * severity_factor)
    
    def _generate_recommendations(self, prediction_type: PredictionType, predicted_value: float, impact_score: float) -> List[str]:
        """Generate recommendations based on prediction."""
        recommendations = []
        
        if prediction_type == PredictionType.QUALITY_DEGRADATION:
            recommendations.extend([
                "Review data ingestion processes for potential issues",
                "Validate data source quality and upstream dependencies",
                "Consider implementing additional data validation rules"
            ])
            
            if impact_score > 0.7:
                recommendations.append("Alert data stewards and downstream consumers")
        
        elif prediction_type == PredictionType.FRESHNESS_DECAY:
            recommendations.extend([
                "Check data pipeline execution schedules",
                "Verify data source availability and connectivity",
                "Review data processing performance metrics"
            ])
        
        elif prediction_type == PredictionType.VOLUME_ANOMALY:
            recommendations.extend([
                "Investigate potential data source changes",
                "Check for seasonal patterns or expected business changes",
                "Monitor upstream systems for capacity issues"
            ])
        
        elif prediction_type == PredictionType.SCHEMA_DRIFT:
            recommendations.extend([
                "Review recent schema changes in upstream systems",
                "Update data transformation logic if needed",
                "Coordinate with data providers on schema evolution"
            ])
        
        return recommendations
    
    def _should_create_alert(self, prediction: QualityPrediction) -> bool:
        """Determine if an alert should be created for a prediction."""
        # Create alerts for high-impact predictions with good confidence
        return (prediction.impact_score > 0.6 and 
                prediction.confidence_score > 0.5 and
                prediction.time_horizon <= timedelta(days=7))
    
    def _create_prediction_alert(self, prediction: QualityPrediction) -> QualityAlert:
        """Create an alert based on a prediction."""
        severity = prediction.get_severity_level()
        
        alert = QualityAlert(
            prediction_id=prediction.id,
            asset_id=prediction.asset_id,
            alert_type=prediction.prediction_type.value,
            severity=severity,
            title=f"Predicted {prediction.prediction_type.value.replace('_', ' ').title()}",
            description=f"Quality prediction indicates potential {prediction.prediction_type.value} with {prediction.confidence.value} confidence",
            expected_time=prediction.target_time,
            recommended_actions=prediction.recommended_actions
        )
        
        # Store alert
        if prediction.asset_id not in self._alerts:
            self._alerts[prediction.asset_id] = []
        self._alerts[prediction.asset_id].append(alert)
        
        return alert
    
    def _determine_forecast_trend(self, values: List[float]) -> TrendDirection:
        """Determine overall trend direction for forecast."""
        if len(values) < 2:
            return TrendDirection.STABLE
        
        # Compare first and last values
        change = (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else 0
        
        if abs(change) < 0.05:  # Less than 5% change
            return TrendDirection.STABLE
        elif change > 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DEGRADING
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate strength of trend in forecasted values."""
        if len(values) < 3:
            return 0.0
        
        # Calculate correlation with time
        time_points = list(range(len(values)))
        _, _, r_squared = self._calculate_linear_regression(time_points, values)
        
        return r_squared
    
    def _detect_forecast_seasonality(self, forecasted_values: List[QualityMetricPoint]) -> SeasonalPattern:
        """Detect seasonal patterns in forecast."""
        # Simplified - just check for daily patterns in forecast
        if len(forecasted_values) >= 24:
            return SeasonalPattern.DAILY
        else:
            return SeasonalPattern.NONE
    
    def _predict_anomalies(self, asset_id: UUID, forecast_times: List[datetime]) -> Tuple[List[datetime], List[float]]:
        """Predict when anomalies might occur in the forecast period."""
        # Simplified anomaly prediction based on historical patterns
        anomaly_times = []
        anomaly_probs = []
        
        # If there's a pattern of anomalies at certain times, predict them
        if asset_id in self._metric_history:
            historical_points = self._metric_history[asset_id]
            anomaly_hours = []
            
            # Find hours when anomalies typically occur
            for point in historical_points:
                values = [p.value for p in historical_points if abs((p.timestamp - point.timestamp).total_seconds()) < 3600]
                if len(values) > 3:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values)
                    
                    if std_val > 0 and abs(point.value - mean_val) > 2 * std_val:
                        anomaly_hours.append(point.timestamp.hour)
            
            # Predict anomalies for forecast times that match historical anomaly hours
            for forecast_time in forecast_times:
                if forecast_time.hour in anomaly_hours:
                    anomaly_prob = len([h for h in anomaly_hours if h == forecast_time.hour]) / len(anomaly_hours)
                    if anomaly_prob > 0.2:  # 20% threshold
                        anomaly_times.append(forecast_time)
                        anomaly_probs.append(anomaly_prob)
        
        return anomaly_times, anomaly_probs