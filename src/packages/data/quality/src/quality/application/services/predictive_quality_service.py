"""Predictive Quality Analytics Service.

Service for predicting future quality issues, trends, and providing
proactive quality management using machine learning forecasting models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
from enum import Enum
import pickle
import warnings
from pathlib import Path

# ML imports for prediction
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

# Time series imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available - some time series features will be disabled")

from ...domain.entities.quality_profile import DataQualityProfile, DatasetId
from ...domain.entities.quality_scores import QualityScores
# TODO: QualityIssue entity needs to be created or use QualityIssue instead
from ...domain.entities.quality_issue import QualityIssue, ImpactLevel
from .quality_assessment_service import QualityAssessmentService

logger = logging.getLogger(__name__)


class PredictionHorizon(Enum):
    """Prediction time horizons."""
    SHORT_TERM = "short_term"     # 1-7 days
    MEDIUM_TERM = "medium_term"   # 1-4 weeks
    LONG_TERM = "long_term"       # 1-6 months


class PredictionModel(Enum):
    """Available prediction models."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ENSEMBLE = "ensemble"


class TrendDirection(Enum):
    """Quality trend directions."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


@dataclass(frozen=True)
class PredictiveQualityConfig:
    """Configuration for predictive quality analytics."""
    # Model selection
    default_models: List[PredictionModel] = field(default_factory=lambda: [
        PredictionModel.RANDOM_FOREST,
        PredictionModel.GRADIENT_BOOSTING,
        PredictionModel.LINEAR_REGRESSION
    ])
    
    # Time series parameters
    min_historical_points: int = 30
    max_historical_points: int = 1000
    seasonality_periods: List[int] = field(default_factory=lambda: [7, 30, 90])
    
    # Prediction parameters
    prediction_confidence: float = 0.95
    min_prediction_accuracy: float = 0.7
    trend_detection_window: int = 14
    
    # Feature engineering
    enable_feature_engineering: bool = True
    lag_features: List[int] = field(default_factory=lambda: [1, 3, 7, 14])
    rolling_window_sizes: List[int] = field(default_factory=lambda: [3, 7, 14, 30])
    
    # Anomaly prediction
    enable_anomaly_prediction: bool = True
    anomaly_prediction_threshold: float = 0.7
    
    # Performance optimization
    enable_model_caching: bool = True
    model_retrain_interval_hours: int = 168  # 1 week
    cross_validation_splits: int = 5
    
    # Alert thresholds
    quality_decline_threshold: float = 0.1
    trend_significance_threshold: float = 0.05
    prediction_uncertainty_threshold: float = 0.3


@dataclass
class QualityPrediction:
    """Quality prediction result."""
    dataset_id: str
    prediction_date: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: PredictionHorizon
    model_used: PredictionModel
    feature_importance: Dict[str, float]
    prediction_confidence: float
    trend_direction: TrendDirection
    
    # Metadata
    model_accuracy: float
    historical_data_points: int
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    
    def is_decline_predicted(self, current_value: float, threshold: float = 0.1) -> bool:
        """Check if prediction indicates quality decline."""
        decline = (current_value - self.predicted_value) / current_value
        return decline > threshold
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get prediction summary."""
        return {
            'dataset_id': self.dataset_id,
            'predicted_value': round(self.predicted_value, 4),
            'confidence_interval': [round(self.confidence_interval[0], 4), 
                                   round(self.confidence_interval[1], 4)],
            'prediction_confidence': round(self.prediction_confidence, 3),
            'trend_direction': self.trend_direction.value,
            'model_used': self.model_used.value,
            'model_accuracy': round(self.model_accuracy, 3),
            'prediction_horizon': self.prediction_horizon.value,
            'prediction_date': self.prediction_date.isoformat(),
            'top_features': dict(sorted(self.feature_importance.items(), 
                                      key=lambda x: abs(x[1]), reverse=True)[:5])
        }


@dataclass
class QualityForecast:
    """Quality forecast for multiple time periods."""
    dataset_id: str
    forecast_dates: List[datetime]
    forecasted_values: List[float]
    confidence_bands: List[Tuple[float, float]]
    overall_trend: TrendDirection
    forecast_accuracy: float
    model_used: PredictionModel
    
    # Analysis
    risk_periods: List[datetime]
    improvement_periods: List[datetime]
    forecast_summary: Dict[str, Any]
    
    def get_risk_assessment(self) -> Dict[str, Any]:
        """Get risk assessment from forecast."""
        return {
            'overall_trend': self.overall_trend.value,
            'risk_periods_count': len(self.risk_periods),
            'improvement_periods_count': len(self.improvement_periods),
            'forecast_accuracy': self.forecast_accuracy,
            'forecast_range': len(self.forecast_dates),
            'min_predicted_quality': min(self.forecasted_values),
            'max_predicted_quality': max(self.forecasted_values),
            'quality_volatility': np.std(self.forecasted_values) if self.forecasted_values else 0
        }


class PredictiveQualityService:
    """Service for predictive quality analytics."""
    
    def __init__(self, config: PredictiveQualityConfig = None):
        """Initialize predictive quality service.
        
        Args:
            config: Service configuration
        """
        self.config = config or PredictiveQualityConfig()
        self._models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        self._historical_data: Dict[str, pd.DataFrame] = {}
        
        logger.info("Predictive Quality Service initialized")
    
    def add_historical_data(self,
                          dataset_id: str,
                          quality_data: List[Dict[str, Any]]) -> None:
        """Add historical quality data for prediction modeling.
        
        Args:
            dataset_id: Dataset identifier
            quality_data: List of quality measurements with timestamps
        """
        df = pd.DataFrame(quality_data)
        
        # Ensure required columns
        if 'timestamp' not in df.columns:
            raise ValueError("Historical data must include 'timestamp' column")
        if 'quality_score' not in df.columns:
            raise ValueError("Historical data must include 'quality_score' column")
        
        # Convert timestamp and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Store historical data
        self._historical_data[dataset_id] = df
        
        logger.info(f"Added historical data for {dataset_id}: {len(df)} records")
    
    def predict_quality(self,
                       dataset_id: str,
                       prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM,
                       target_date: Optional[datetime] = None,
                       model_type: Optional[PredictionModel] = None) -> Optional[QualityPrediction]:
        """Predict quality for a specific dataset and time.
        
        Args:
            dataset_id: Dataset identifier
            prediction_horizon: Time horizon for prediction
            target_date: Specific date to predict (defaults to appropriate future date)
            model_type: Specific model to use (defaults to best performing)
            
        Returns:
            Quality prediction result
        """
        if dataset_id not in self._historical_data:
            logger.warning(f"No historical data available for {dataset_id}")
            return None
        
        df = self._historical_data[dataset_id]
        
        if len(df) < self.config.min_historical_points:
            logger.warning(f"Insufficient historical data for {dataset_id}: {len(df)} points")
            return None
        
        # Determine target date if not provided
        if target_date is None:
            last_date = df['timestamp'].max()
            if prediction_horizon == PredictionHorizon.SHORT_TERM:
                target_date = last_date + timedelta(days=3)
            elif prediction_horizon == PredictionHorizon.MEDIUM_TERM:
                target_date = last_date + timedelta(weeks=2)
            else:  # LONG_TERM
                target_date = last_date + timedelta(days=90)
        
        # Prepare features
        features_df = self._engineer_features(df)
        
        if features_df.empty:
            logger.warning(f"No features could be engineered for {dataset_id}")
            return None
        
        # Select and train model
        model, model_accuracy = self._get_or_train_model(dataset_id, features_df, model_type)
        
        if model is None:
            logger.warning(f"Could not train model for {dataset_id}")
            return None
        
        # Make prediction
        prediction_features = self._prepare_prediction_features(features_df, target_date)
        
        try:
            predicted_value = model.predict([prediction_features])[0]
            
            # Calculate confidence interval (simplified approach)
            confidence_interval = self._calculate_confidence_interval(
                model, features_df, predicted_value, model_accuracy
            )
            
            # Determine trend direction
            trend_direction = self._analyze_trend(df['quality_score'].tail(self.config.trend_detection_window))
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model, features_df.columns)
            
            prediction = QualityPrediction(
                dataset_id=dataset_id,
                prediction_date=target_date,
                predicted_value=max(0.0, min(1.0, predicted_value)),  # Clamp to [0,1]
                confidence_interval=confidence_interval,
                prediction_horizon=prediction_horizon,
                model_used=model_type or PredictionModel.RANDOM_FOREST,
                feature_importance=feature_importance,
                prediction_confidence=model_accuracy,
                trend_direction=trend_direction,
                model_accuracy=model_accuracy,
                historical_data_points=len(df)
            )
            
            logger.info(f"Generated prediction for {dataset_id}: {predicted_value:.3f}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction for {dataset_id}: {e}")
            return None
    
    def generate_quality_forecast(self,
                                dataset_id: str,
                                forecast_days: int = 30,
                                model_type: Optional[PredictionModel] = None) -> Optional[QualityForecast]:
        """Generate quality forecast for multiple future periods.
        
        Args:
            dataset_id: Dataset identifier
            forecast_days: Number of days to forecast
            model_type: Specific model to use
            
        Returns:
            Quality forecast result
        """
        if dataset_id not in self._historical_data:
            return None
        
        df = self._historical_data[dataset_id]
        
        if len(df) < self.config.min_historical_points:
            return None
        
        # Generate forecast dates
        last_date = df['timestamp'].max()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        # Prepare features for time series forecasting
        features_df = self._engineer_features(df)
        model, model_accuracy = self._get_or_train_model(dataset_id, features_df, model_type)
        
        if model is None:
            return None
        
        # Generate forecasts
        forecasted_values = []
        confidence_bands = []
        
        # For simplicity, using iterative prediction
        current_data = df.copy()
        
        for forecast_date in forecast_dates:
            # Prepare features for this date
            prediction_features = self._prepare_prediction_features(
                self._engineer_features(current_data), forecast_date
            )
            
            # Make prediction
            predicted_value = model.predict([prediction_features])[0]
            predicted_value = max(0.0, min(1.0, predicted_value))
            
            forecasted_values.append(predicted_value)
            
            # Calculate confidence interval
            ci = self._calculate_confidence_interval(model, features_df, predicted_value, model_accuracy)
            confidence_bands.append(ci)
            
            # Add prediction to current data for next iteration
            new_row = pd.DataFrame({
                'timestamp': [forecast_date],
                'quality_score': [predicted_value]
            })
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        # Analyze forecast
        overall_trend = self._analyze_trend(pd.Series(forecasted_values))
        risk_periods = [date for date, value in zip(forecast_dates, forecasted_values) 
                       if value < 0.7]  # Quality threshold
        improvement_periods = [date for date, value in zip(forecast_dates, forecasted_values) 
                             if value > df['quality_score'].tail(7).mean()]
        
        forecast = QualityForecast(
            dataset_id=dataset_id,
            forecast_dates=forecast_dates,
            forecasted_values=forecasted_values,
            confidence_bands=confidence_bands,
            overall_trend=overall_trend,
            forecast_accuracy=model_accuracy,
            model_used=model_type or PredictionModel.RANDOM_FOREST,
            risk_periods=risk_periods,
            improvement_periods=improvement_periods,
            forecast_summary={
                'avg_quality': np.mean(forecasted_values),
                'min_quality': min(forecasted_values),
                'max_quality': max(forecasted_values),
                'volatility': np.std(forecasted_values),
                'trend_slope': self._calculate_trend_slope(forecasted_values)
            }
        )
        
        logger.info(f"Generated forecast for {dataset_id}: {forecast_days} days")
        return forecast
    
    def predict_anomaly_likelihood(self,
                                 dataset_id: str,
                                 prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM) -> Optional[Dict[str, Any]]:
        """Predict likelihood of quality anomalies.
        
        Args:
            dataset_id: Dataset identifier
            prediction_horizon: Time horizon for prediction
            
        Returns:
            Anomaly likelihood prediction
        """
        if not self.config.enable_anomaly_prediction:
            return None
        
        prediction = self.predict_quality(dataset_id, prediction_horizon)
        if not prediction:
            return None
        
        df = self._historical_data[dataset_id]
        
        # Calculate historical anomaly patterns
        recent_quality = df['quality_score'].tail(30)
        quality_std = recent_quality.std()
        quality_mean = recent_quality.mean()
        
        # Calculate anomaly likelihood based on prediction
        predicted_deviation = abs(prediction.predicted_value - quality_mean) / quality_std if quality_std > 0 else 0
        
        # Consider trend direction
        trend_risk_multiplier = {
            TrendDirection.DECLINING: 1.5,
            TrendDirection.VOLATILE: 1.3,
            TrendDirection.STABLE: 1.0,
            TrendDirection.IMPROVING: 0.7
        }
        
        anomaly_likelihood = min(predicted_deviation * trend_risk_multiplier[prediction.trend_direction], 1.0)
        
        # Classify risk level
        if anomaly_likelihood >= 0.8:
            risk_level = "high"
        elif anomaly_likelihood >= 0.6:
            risk_level = "medium"
        elif anomaly_likelihood >= 0.3:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        return {
            'dataset_id': dataset_id,
            'anomaly_likelihood': round(anomaly_likelihood, 3),
            'risk_level': risk_level,
            'prediction_horizon': prediction_horizon.value,
            'contributing_factors': {
                'predicted_deviation': round(predicted_deviation, 3),
                'trend_direction': prediction.trend_direction.value,
                'prediction_confidence': prediction.prediction_confidence,
                'historical_volatility': round(quality_std, 3)
            },
            'recommendations': self._generate_anomaly_prevention_recommendations(anomaly_likelihood, prediction)
        }
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for prediction models."""
        if not self.config.enable_feature_engineering:
            return df[['quality_score']].copy()
        
        features_df = df[['timestamp', 'quality_score']].copy()
        features_df = features_df.set_index('timestamp')
        
        # Lag features
        for lag in self.config.lag_features:
            features_df[f'quality_lag_{lag}'] = features_df['quality_score'].shift(lag)
        
        # Rolling statistics
        for window in self.config.rolling_window_sizes:
            features_df[f'quality_rolling_mean_{window}'] = features_df['quality_score'].rolling(window).mean()
            features_df[f'quality_rolling_std_{window}'] = features_df['quality_score'].rolling(window).std()
            features_df[f'quality_rolling_min_{window}'] = features_df['quality_score'].rolling(window).min()
            features_df[f'quality_rolling_max_{window}'] = features_df['quality_score'].rolling(window).max()
        
        # Time-based features
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['day_of_month'] = features_df.index.day
        features_df['month'] = features_df.index.month
        features_df['hour'] = features_df.index.hour
        
        # Trend features
        features_df['quality_trend_7d'] = features_df['quality_score'].rolling(7).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        )
        
        # Volatility features
        features_df['quality_volatility_7d'] = features_df['quality_score'].rolling(7).std()
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def _get_or_train_model(self,
                          dataset_id: str,
                          features_df: pd.DataFrame,
                          model_type: Optional[PredictionModel] = None) -> Tuple[Any, float]:
        """Get existing model or train new one."""
        model_key = f"{dataset_id}_{model_type or 'default'}"
        
        # Check if model exists and is recent
        if self.config.enable_model_caching and model_key in self._models:
            metadata = self._model_metadata.get(model_key, {})
            last_training = metadata.get('last_training')
            
            if last_training:
                age_hours = (datetime.now() - last_training).total_seconds() / 3600
                if age_hours < self.config.model_retrain_interval_hours:
                    return self._models[model_key], metadata.get('accuracy', 0.0)
        
        # Train new model
        model, accuracy = self._train_prediction_model(features_df, model_type)
        
        if model is not None:
            self._models[model_key] = model
            self._model_metadata[model_key] = {
                'last_training': datetime.now(),
                'accuracy': accuracy,
                'feature_count': len(features_df.columns) - 1,  # Exclude target
                'training_samples': len(features_df)
            }
        
        return model, accuracy
    
    def _train_prediction_model(self,
                              features_df: pd.DataFrame,
                              model_type: Optional[PredictionModel] = None) -> Tuple[Any, float]:
        """Train prediction model."""
        if len(features_df) < 10:  # Minimum samples for training
            return None, 0.0
        
        # Prepare data
        X = features_df.drop('quality_score', axis=1)
        y = features_df['quality_score']
        
        # Handle infinite or very large values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Split data
        test_size = min(0.3, max(0.1, 10 / len(X)))  # Adaptive test size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Select model
        if model_type == PredictionModel.LINEAR_REGRESSION:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
        elif model_type == PredictionModel.GRADIENT_BOOSTING:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:  # Default to Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = r2_score(y_test, y_pred)
            
            return model, max(0.0, accuracy)  # Ensure non-negative accuracy
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None, 0.0
    
    def _prepare_prediction_features(self,
                                   features_df: pd.DataFrame,
                                   target_date: datetime) -> List[float]:
        """Prepare features for a specific prediction date."""
        if features_df.empty:
            return []
        
        # Use the last available feature row as template
        last_features = features_df.iloc[-1].drop('quality_score')
        
        # Update time-based features for target date
        if 'day_of_week' in last_features.index:
            last_features['day_of_week'] = target_date.weekday()
        if 'day_of_month' in last_features.index:
            last_features['day_of_month'] = target_date.day
        if 'month' in last_features.index:
            last_features['month'] = target_date.month
        if 'hour' in last_features.index:
            last_features['hour'] = target_date.hour
        
        return last_features.fillna(0).tolist()
    
    def _calculate_confidence_interval(self,
                                     model: Any,
                                     features_df: pd.DataFrame,
                                     predicted_value: float,
                                     model_accuracy: float) -> Tuple[float, float]:
        """Calculate confidence interval for prediction."""
        # Simplified confidence interval based on model accuracy and historical variance
        y = features_df['quality_score']
        historical_std = y.std()
        
        # Adjust confidence based on model accuracy
        uncertainty = historical_std * (1 - model_accuracy)
        
        # Calculate confidence interval
        z_score = 1.96  # 95% confidence
        margin = z_score * uncertainty
        
        lower_bound = max(0.0, predicted_value - margin)
        upper_bound = min(1.0, predicted_value + margin)
        
        return (lower_bound, upper_bound)
    
    def _analyze_trend(self, values: pd.Series) -> TrendDirection:
        """Analyze trend direction from values."""
        if len(values) < 3:
            return TrendDirection.STABLE
        
        # Calculate trend slope
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Calculate volatility
        volatility = values.std() / values.mean() if values.mean() > 0 else 0
        
        # Determine trend direction
        if volatility > 0.2:  # High volatility threshold
            return TrendDirection.VOLATILE
        elif slope > self.config.trend_significance_threshold:
            return TrendDirection.IMPROVING
        elif slope < -self.config.trend_significance_threshold:
            return TrendDirection.DECLINING
        else:
            return TrendDirection.STABLE
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'coef_'):
                # Linear models in pipeline
                importances = np.abs(model.named_steps['model'].coef_)
            else:
                # Default: equal importance
                importances = np.ones(len(feature_names)) / len(feature_names)
            
            return dict(zip(feature_names, importances))
            
        except Exception:
            return {name: 0.0 for name in feature_names}
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope from values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _generate_anomaly_prevention_recommendations(self,
                                                   anomaly_likelihood: float,
                                                   prediction: QualityPrediction) -> List[str]:
        """Generate recommendations for anomaly prevention."""
        recommendations = []
        
        if anomaly_likelihood >= 0.8:
            recommendations.append("Immediate attention required - implement real-time monitoring")
            recommendations.append("Consider increasing data validation frequency")
        elif anomaly_likelihood >= 0.6:
            recommendations.append("Increase monitoring frequency for early detection")
            recommendations.append("Review recent data processing changes")
        elif anomaly_likelihood >= 0.3:
            recommendations.append("Monitor trend closely for early warning signs")
        
        if prediction.trend_direction == TrendDirection.DECLINING:
            recommendations.append("Investigate root causes of quality decline")
        elif prediction.trend_direction == TrendDirection.VOLATILE:
            recommendations.append("Stabilize data processing pipeline to reduce volatility")
        
        # Feature-based recommendations
        top_features = sorted(prediction.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        if top_features:
            recommendations.append(f"Focus on monitoring key quality drivers: {', '.join([f[0] for f in top_features])}")
        
        return recommendations
    
    def get_prediction_summary(self, dataset_ids: List[str] = None) -> Dict[str, Any]:
        """Get summary of prediction capabilities and recent predictions.
        
        Args:
            dataset_ids: Specific datasets to summarize (defaults to all)
            
        Returns:
            Prediction summary
        """
        if dataset_ids is None:
            dataset_ids = list(self._historical_data.keys())
        
        summary = {
            'total_datasets': len(dataset_ids),
            'datasets_with_sufficient_data': 0,
            'trained_models': len(self._models),
            'prediction_capabilities': {},
            'recent_trends': {}
        }
        
        for dataset_id in dataset_ids:
            if dataset_id in self._historical_data:
                df = self._historical_data[dataset_id]
                
                if len(df) >= self.config.min_historical_points:
                    summary['datasets_with_sufficient_data'] += 1
                    
                    # Analyze recent trend
                    recent_quality = df['quality_score'].tail(self.config.trend_detection_window)
                    trend = self._analyze_trend(recent_quality)
                    
                    summary['recent_trends'][dataset_id] = {
                        'trend_direction': trend.value,
                        'current_quality': recent_quality.iloc[-1] if len(recent_quality) > 0 else None,
                        'quality_change_7d': (recent_quality.iloc[-1] - recent_quality.iloc[0]) if len(recent_quality) > 1 else 0,
                        'data_points': len(df)
                    }
                    
                    summary['prediction_capabilities'][dataset_id] = {
                        'can_predict': True,
                        'historical_points': len(df),
                        'last_update': df['timestamp'].max().isoformat(),
                        'available_horizons': [h.value for h in PredictionHorizon]
                    }
                else:
                    summary['prediction_capabilities'][dataset_id] = {
                        'can_predict': False,
                        'reason': f"Insufficient data ({len(df)} < {self.config.min_historical_points})",
                        'historical_points': len(df)
                    }
        
        return summary